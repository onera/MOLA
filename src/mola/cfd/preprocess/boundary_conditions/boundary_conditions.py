#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.
import os
import copy
from treelab import cgns
from mola import misc
from mola.logging import mola_logger, MolaException, MolaUserError, mute_stdout
from mola.cfd.preprocess.motion import motion

BoundaryConditionsNames = dict(
    Farfield                     = dict(elsa='nref', sonics='BCFarfield'),
    InflowStagnation             = dict(elsa='inj1', sonics='BCInflowSubsonicPressure'),
    InflowMassFlow               = dict(elsa='injmfr1', sonics='BCInflowSubsonicMassFlow'),
    OutflowPressure              = dict(elsa='outpres', sonics='BCOutflowSubsonic'),
    OutflowMassFlow              = dict(elsa='outmfr2'),
    OutflowRadialEquilibrium     = dict(elsa='outradeq'),
    MixingPlane                  = dict(elsa='stage_mxpl'),
    UnsteadyRotorStatorInterface = dict(elsa='stage_red'),
    WallViscous                  = dict(elsa='walladia', sonics='BCWallViscous'),
    WallViscousIsothermal        = dict(elsa='wallisoth', sonics='BCWallViscousIsothermal'),
    WallInviscid                 = dict(elsa='wallslip', sonics='BCWallInviscid'),
    SymmetryPlane                = dict(elsa='sym', sonics='BCSymmetryPlane'),
)

# Shortcuts for already defined boundary conditions
BoundaryConditionsNames.update(
    dict(
        Wall = BoundaryConditionsNames['WallViscous'],
    )
)

permeable_boundaries = ['Farfield', 'InflowStagnation', 'InflowMassFlow', 'OutflowPressure', 'OutflowMassFlow', 'OutflowRadialEquilibrium']
turbomachinery_interfaces = ['MixingPlane', 'UnsteadyRotorStatorInterface']


def apply(workflow, selected_boundaries_conditions=None):
    '''
    Set all boundary conditions for **workflow**.
    It transforms the tree attribute of the **workflow**.

    Parameters
    ----------
    workflow : Workflow object

    selected_boundaries_conditions : :py:class:`list` of :py:class:`dict`, optional
        Boudaries to apply. 
        If not given, the attribute `BoundaryConditions` of the **workflow** is used.
        Otherwise, it is possible to give a filtered list.
    '''
    if selected_boundaries_conditions is None:
        selected_boundaries_conditions = workflow.BoundaryConditions

    if len(selected_boundaries_conditions) != 0:
        mola_logger.info(f'Set boundary conditions:', rank=0)

    available_bc_names = [name for name, solvers in BoundaryConditionsNames.items() if workflow.Solver.lower() in solvers]
    alternative_available_bc_names = [solvers[workflow.Solver.lower()] for solvers in BoundaryConditionsNames.values() if workflow.Solver.lower() in solvers]

    for bc in selected_boundaries_conditions:

        _check_family_exists(workflow.tree, bc['Family'])
        
        bcName = bc['Type']
        if bcName == 'InterfaceBetweenWorkflows':
            continue
        if 'LinkedFamily' in bc:
            mola_logger.info(f'  > {bcName} between families {bc["Family"]} and {bc["LinkedFamily"]}', rank=0)
        else:
            mola_logger.info(f'  > {bcName} on family {bc["Family"]}', rank=0)
        
        if bcName in available_bc_names:
            # Define in the main MOLA preprocess, lower in this file
            MOLAGenericFunction = globals()[bcName]
            solverSpecificFunctionName = BoundaryConditionsNames[bcName][workflow.Solver]
            args, kwargs = MOLAGenericFunction(workflow, bc)
        elif bcName in alternative_available_bc_names:
            # Defined only in the specific solver module
            solverSpecificFunctionName = bcName
            args, kwargs = bc['args'], bc['kwargs']
        else:
            raise MolaUserError(
                f'Boundary condition {bcName} is not available. ' 
                f'Please choose one among conditions currently available for solver {workflow.Solver}: '
                f'{", ".join(available_bc_names)}'
                 )

        current_path = os.path.dirname(os.path.realpath(__file__))
        solverModule = misc.load_source('solverModule', os.path.join(current_path, f'solver_{workflow.Solver}.py'))
        try:
            solverSpecificFunction = getattr(solverModule, solverSpecificFunctionName)
        except AttributeError:
            raise MolaException(f'The function {solverSpecificFunctionName} does not exist for the solver {workflow.Solver}.')
        else:
            solverSpecificFunction(workflow, *args, **kwargs)

def _check_family_exists(tree, family_name):
    if not tree.get(Name=family_name, Type='Family', Depth=2):
        raise MolaException(f'Cannot apply a boundary condition on family {family_name}: This family does not exist in the mesh.')

def apply_function_to_BCDataSet(workflow, Family, functions_to_apply):
    '''
    Apply a function to all face centers in the BC attached to **Family**

    Parameters
    ----------
    workflow : Workflow object

    Family: str
        Name of the Family attached to the given boundary condition

    function_to_apply: fun
        Function to apply to all face centers of BC. The arguments of the function must be variables names
        present in the tree. 

    Return
    ------
    ???

    Example
    -------
    To define the wall velocity at the hub, a function could be defined: 

    .. code-block::python

        def hub_function(CoordinateX):
            omega = np.zeros(CoordinateX.shape, dtype=float)
            omega[(x1<=CoordinateX) & (CoordinateX<=x2)] = 500.
            return dict(Motion = omega * np.array(RotationAxis))

        apply_function_to_BCDataSet(workflow, 'Hub', hub_function)
    '''
    import Converter.PyTree as C
    import Converter.Internal as I

    bc_dict = dict()

    for base in workflow.tree.bases():
        bc_list = C.extractBCOfName(base, f'FamilySpecified:{Family}')
        bc_list = C.node2Center(bc_list)

        for bc in bc_list:

            VarDictToImpose = dict()
            for variable_name, function_to_apply in functions_to_apply.items():
                # args_names is the tuple of the names of arguments of function_to_apply
                args_names = function_to_apply.__code__.co_varnames[:function_to_apply.__code__.co_argcount]
                kwargs = dict()
                for arg_name in args_names:
                    # nodes = bc.group(Name=arg_name, Type='DataArray')
                    nodes = I.getNodesFromNameAndType(bc, arg_name, 'DataArray_t')
                    if len(nodes) == 0:
                        raise Exception(f'{arg_name} is not found in {bc.name()}')
                    elif len(nodes) == 1:
                        node = nodes[0]
                    else:
                        pass

                    kwargs[arg_name] = I.getValue(node)

                VarDictToImpose[variable_name] = function_to_apply(**kwargs)

            # Get BC path in the main tree
            zname, wname = bc[0].split(os.sep)
            bc_path = f'CGNSTree/{base[0]}/{zname}/ZoneBC/{wname}'

            bc_dict[bc_path] = VarDictToImpose

    return bc_dict      

def Wall(workflow, bc):
    Motion = bc.get('Motion', dict())
    motion.update_motion_with_defaults(Motion)
    return [bc['Family']], dict(Motion=Motion) 

WallViscous = Wall
WallInviscid = Wall

def Farfield(workflow, bc):
    return [bc['Family']], dict() 

def InflowStagnation(workflow, bc): 
    '''
    Set a Boundary Condition ``inj1``
    '''
    PressureStagnation    = bc.get('PressureStagnation', workflow.Flow['PressureStagnation'])
    TemperatureStagnation = bc.get('TemperatureStagnation', workflow.Flow['TemperatureStagnation'])
    EnthalpyStagnation    = bc.get('EnthalpyStagnation', workflow.Fluid['cp'] * TemperatureStagnation)
    VelocityUnitVectorX   = bc.get('VelocityUnitVectorX', workflow.Flow['Direction'][0])
    VelocityUnitVectorY   = bc.get('VelocityUnitVectorY', workflow.Flow['Direction'][1])
    VelocityUnitVectorZ   = bc.get('VelocityUnitVectorZ', workflow.Flow['Direction'][2])
    variableForInterpolation = bc.get('variableForInterpolation', 'ChannelHeight')   

    ImposedVariables = dict(
        PressureStagnation  = PressureStagnation,
        EnthalpyStagnation  = EnthalpyStagnation,
        VelocityUnitVectorX = VelocityUnitVectorX,
        VelocityUnitVectorY = VelocityUnitVectorY,
        VelocityUnitVectorZ = VelocityUnitVectorZ,
        **getPrimitiveTurbulentFieldForInjection(workflow, bc)
        )

    return [bc['Family']], dict(ImposedVariables=ImposedVariables, variableForInterpolation=variableForInterpolation) 

def InflowMassFlow(workflow, bc):
    Surface = bc.get('Surface', None)
    if not Surface:
        from mola.cfd.preprocess.mesh.tools import get_surface_of_family
        Surface = get_surface_of_family(workflow.tree, bc['Family'])
        try:
            Surface *= workflow.ApplicationContext['NormalizationCoefficient'][bc['Family']]['FluxCoef']
        except:
            pass

    MassFlow              = bc.get('MassFlow', workflow.Flow['MassFlow'])
    SurfacicMassFlow      = bc.get('SurfacicMassFlow', MassFlow / Surface)

    TemperatureStagnation = bc.get('TemperatureStagnation', workflow.Flow['TemperatureStagnation'])
    EnthalpyStagnation    = bc.get('EnthalpyStagnation', workflow.Fluid['cp'] * TemperatureStagnation)
    VelocityUnitVectorX   = bc.get('VelocityUnitVectorX', workflow.Flow['Direction'][0])
    VelocityUnitVectorY   = bc.get('VelocityUnitVectorY', workflow.Flow['Direction'][1])
    VelocityUnitVectorZ   = bc.get('VelocityUnitVectorZ', workflow.Flow['Direction'][2])
    variableForInterpolation = bc.get('variableForInterpolation', 'ChannelHeight')    
    # if not 'MassFlow' in bc:
    #     # used for getPrimitiveTurbulentFieldForInjection
    #     bc['MassFlow'] = SurfacicMassFlow * Surface

    ImposedVariables = dict(
        SurfacicMassFlow    = SurfacicMassFlow,
        EnthalpyStagnation  = EnthalpyStagnation,
        VelocityUnitVectorX = VelocityUnitVectorX,
        VelocityUnitVectorY = VelocityUnitVectorY,
        VelocityUnitVectorZ = VelocityUnitVectorZ,
        **getPrimitiveTurbulentFieldForInjection(workflow, bc)
        )
    return [bc['Family']], dict(ImposedVariables=ImposedVariables, variableForInterpolation=variableForInterpolation) 

def OutflowPressure(workflow, bc):
    Pressure = bc.get('Pressure', workflow.Flow['Pressure'])
    return [bc['Family']], dict(Pressure=Pressure) 

def OutflowMassFlow(workflow, bc):
    MassFlow = bc.get('MassFlow')
    if not MassFlow:
        MassFlow = workflow.Flow.get('MassFlow')
    if not MassFlow:
        from mola.cfd.preprocess.mesh.tools import get_surface_of_family
        surface = get_surface_of_family(workflow.tree, bc['Family'])
        MassFlow = workflow.Flow['Density']*workflow.Flow['Velocity']*surface

    try:
        fluxcoeff = workflow.ApplicationContext['NormalizationCoefficient'][bc['Family']]['FluxCoef']
    except: 
        fluxcoeff = 1.

    MassFlowOnBC = MassFlow / fluxcoeff
    return [bc['Family']], dict(MassFlow=MassFlowOnBC) 

def getPrimitiveTurbulentFieldForInjection(workflow, bc):
        '''
        Get the primitive (without the Density factor) turbulent variables (names and values) 
        to inject in an inflow boundary condition.

        For RSM models, see issue https://elsa.onera.fr/issues/5136 for the naming convention.

        Parameters
        ----------
        workflow, bc

        Returns
        -------
        dict
            Imposed turbulent variables
        '''
        # FIXME Fix this function, the behavior was corrected in MOLA v1
        TurbulenceLevel = bc.get('TurbulenceLevel', None)
        Viscosity_EddyMolecularRatio = bc.get('Viscosity_EddyMolecularRatio', None)
        if TurbulenceLevel and Viscosity_EddyMolecularRatio:
            
            FlowGen = workflow._FlowGenerator() 
            FlowGen.Turbulence.update(
                dict(Level=TurbulenceLevel, Viscosity_EddyMolecularRatio=Viscosity_EddyMolecularRatio)
            )
            FlowGen.set_turbulence_properties()
            Turbulence = FlowGen.Turbulence

        else:
            Turbulence = workflow.Turbulence

        turbDict = dict()
        for name, value in Turbulence['Conservatives'].items():
            # If the 'conservative' value is given in kwargs
            value = bc.get(name, value)

            if name.endswith('Density'):
                name = name.replace('Density', '')
                value /= workflow.Flow['Density']
            elif name == 'ReynoldsStressDissipationScale':
                name = 'TurbulentDissipationRate'
                value /= workflow.Flow['Density']
            elif name.startswith('ReynoldsStress'):
                name = name.replace('ReynoldsStress', 'VelocityCorrelation')
                value /= workflow.Flow['Density']
            turbDict[name] = value

            # If the 'primitive' value is given in kwargs
            turbDict[name] = bc.get(name, value)
            
        return turbDict

def OutflowRadialEquilibrium(workflow, bc):
    # kwargs = dict(
    #     valve_type = bc.get('valve_type', 0),
    #     valve_ref_pres = bc.get('valve_ref_pres'),
    #     valve_ref_mflow = bc.get('valve_ref_pres'), 
    #     valve_relax = bc.get('valve_relax', 0.1), 
    #     indpiv = bc.get('indpiv', 1),
    # )
    kwargs = copy.deepcopy(bc)
    kwargs.pop('Family')
    kwargs.pop('Type')
    return [bc['Family']], kwargs


def MixingPlane(workflow, bc):
    return [bc['Family'], bc['LinkedFamily']], dict() 

