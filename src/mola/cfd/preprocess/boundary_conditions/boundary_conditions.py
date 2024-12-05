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
import numpy as np
from treelab import cgns
from mola import misc
from mola.logging import mola_logger, MolaException, MolaUserError

# TODO for elsa, add injrot, wallisoth and Giles conditions
BoundaryConditionsNames = dict(
    Farfield                     = dict(elsa='nref',
                                        sonics='BCFarfield',
                                        fast='BCFarfield'),
    InflowStagnation             = dict(elsa='inj1', sonics='BCInflowSubsonicPressure'),
    InflowMassFlow               = dict(elsa='injmfr1', sonics='BCInflowSubsonicMassFlow'),
    OutflowPressure              = dict(elsa='outpres', sonics='BCOutflowSubsonic'),
    OutflowSupersonic            = dict(elsa='outsup'),
    OutflowMassFlow              = dict(elsa='outmfr2'),
    OutflowRadialEquilibrium     = dict(elsa='outradeq', sonics='BCOutflowRadialEquilibrium'),
    
    WallViscous                  = dict(elsa='walladia',
                                        sonics='BCWallViscous',
                                        fast='BCWall'),
    WallViscousIsothermal        = dict(sonics='BCWallViscousIsothermal'),
    WallInviscid                 = dict(elsa='wallslip',
                                        sonics='BCWallInviscid',
                                        fast='BCWall'),
    SymmetryPlane                = dict(elsa='sym',
                                        sonics='BCSymmetryPlane',
                                        fast='BCSymmetryPlane'),

    MixingPlane                  = dict(elsa='stage_mxpl'),  # use hybrid version by default ? 
    UnsteadyRotorStatorInterface = dict(elsa='stage_red'),  # use hybrid version by default ? 
    ChorochronicInterface        = dict(elsa='chorochronic'),
)

# Shortcuts for already defined boundary conditions
BoundaryConditionsNames.update(
    dict(
        Wall = BoundaryConditionsNames['WallViscous'],
    )
)

permeable_boundaries = ['Farfield', 'InflowStagnation', 'InflowMassFlow', 'OutflowPressure', 'OutflowMassFlow', 'OutflowRadialEquilibrium']
turbomachinery_interfaces = ['MixingPlane', 'UnsteadyRotorStatorInterface', 'ChorochronicInterface']

# def check_name_is_one_of_authorized_names(name, authorized_names):
#     import difflib
#     closest_names = difflib.get_close_matches(name, possibilities=authorized_names)
#     closest_msg = ""
#     if len(closest_names) > 0:
#         closest_msg = f"Did you mean {' or '.join(closest_names)}?"
#     raise NameError(f"Invalid name '{name}'. "+closest_msg)

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

    if workflow.Turbulence['Model'] == 'Euler':
        _adapt_bc_to_euler(workflow)

    for bc in selected_boundaries_conditions:

        _check_family_exists(workflow.tree, bc['Family'])
        
        bc_type = bc.pop('Type')
        if bc_type == 'InterfaceBetweenWorkflows':
            continue
        if 'LinkedFamily' in bc:
            mola_logger.info(f'  > {bc_type} between families {bc["Family"]} and {bc["LinkedFamily"]}', rank=0)
        else:
            mola_logger.info(f'  > {bc_type} on family {bc["Family"]}', rank=0)
        
        if bc_type in available_bc_names:
            solverSpecificFunctionName = BoundaryConditionsNames[bc_type][workflow.Solver]
        elif bc_type in alternative_available_bc_names:
            # Defined only in the specific solver module
            solverSpecificFunctionName = bc_type
        else:
            raise MolaUserError(
                f'Boundary condition {bc_type} is not available. ' 
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
            solverSpecificFunction(workflow, **bc)

def _check_family_exists(tree, family_name):
    if not tree.get(Name=family_name, Type='Family', Depth=2):
        raise MolaException(f'Cannot apply a boundary condition on family {family_name}: This family does not exist in the mesh.')

def _adapt_bc_to_euler(workflow):
    for bc in workflow.BoundaryConditions:
        if bc['Type'] in ['Wall', 'WallViscous']:
            mola_logger.warning(
                f"Inconsistency between BC {bc['Family']} of type {bc['Type']} and the Euler model.\n"
                "-> Type is automatically changed into WallInviscid."
                )
            bc['Type'] = 'WallInviscid'

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
                        raise Exception(f'Several nodes with name {arg_name} are found in {bc.name()}')

                    kwargs[arg_name] = I.getValue(node)

                VarDictToImpose[variable_name] = function_to_apply(**kwargs)

            # Get BC path in the main tree
            zname, wname = bc[0].split(os.sep)
            bc_path = f'CGNSTree/{base[0]}/{zname}/ZoneBC/{wname}'

            bc_dict[bc_path] = VarDictToImpose

    return bc_dict      

def get_fields_from_file(t, FamilyName, filename, var2interp, fileformat=None):

    # TODO This function is not working yet. The function migrateFields must be replaced.

    import Converter.PyTree as C
    import Converter.Internal as I
 
    input_data_from_file = dict()
    donor_tree = C.convertFile2PyTree(filename, format=fileformat)
    inlet_BC_nodes = C.extractBCOfName(t, f'FamilySpecified:{FamilyName}', reorder=False)

    I._adaptZoneNamesForSlash(inlet_BC_nodes)
    I._rmNodesByType(inlet_BC_nodes,'FlowSolution_t')
    J.migrateFields(donor_tree, inlet_BC_nodes)  # THIS LINE MUST BE REPLACED

    for w in inlet_BC_nodes:
        bcLongName = I.getName(w)  # from C.extractBCOfName: <zone>\<bc>
        zname, wname = bcLongName.split('\\')
        znode = I.getNodeFromNameAndType(t, zname, 'Zone_t')
        bcnode = I.getNodeFromNameAndType(znode, wname, 'BC_t')
        ImposedVariables = dict()
        for var in var2interp:
            FS = I.getNodeFromName(w, I.__FlowSolutionCenters__)
            varNode = I.getNodeFromName(FS, var) 
            if varNode:
                ImposedVariables[var] = np.asfortranarray(I.getValue(varNode))
            else:
                raise TypeError('variable {} not found in {}'.format(var, filename))
        
        input_data_from_file[bcnode] = ImposedVariables
    
    return input_data_from_file

def recompute_turbulence_variables(workflow, **kwargs):

    if 'TurbulenceLevel' in kwargs or 'Viscosity_EddyMolecularRatio' in kwargs:   
        mola_logger.info('  recomputing turbulent variables for this BC...')       

        workflow_copy = copy.copy(workflow)
        for name, value in kwargs.items():
            if name in workflow_copy.Fluid:
                workflow_copy.Fluid[name] = value
            elif name in workflow_copy.Flow:
                workflow_copy.Flow[name] = value
            elif name in workflow_copy.Turbulence:
                workflow_copy.Turbulence[name] = value
            elif name in workflow_copy.ApplicationContext:
                workflow_copy.ApplicationContext[name] = value
            else:
                raise MolaException(f'Variable {name} cannot be updated neither in Fluid, Flow, Turbulence or ApplicationContext attributes.')

        FlowGen = workflow.Flow['Generator']
        FlowGen.Turbulence.update(workflow_copy.Turbulence)
        FlowGen.Turbulence.update(workflow_copy.Turbulence)
        FlowGen.Turbulence.update(workflow_copy.Turbulence)
        FlowGen.Turbulence.update(workflow_copy.Turbulence)
        FlowGen.generate()
        Turbulence = FlowGen.Turbulence

        del workflow_copy

    else:
        Turbulence = workflow.Turbulence
    
    return Turbulence

def get_turbulent_primitives(workflow, **kwargs):
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
    if 'TurbulenceLevel' in kwargs or 'Viscosity_EddyMolecularRatio' in kwargs:   
        recompute_turbulence_variables(workflow, **kwargs)
    else:
        Turbulence = workflow.Turbulence
        
    turbDict = get_turbulent_primitives_from_conservatives(Turbulence, workflow.Flow['Density'], **kwargs)
        
    return turbDict

def get_turbulent_primitives_from_conservatives(Turbulence, Density, **kwargs):
    turbDict = dict()
    for name, value in Turbulence['Conservatives'].items():
        # If the 'conservative' value is given in kwargs
        value = kwargs.get(name, value)

        if name.endswith('Density'):
            name = name.replace('Density', '')
            value /= Density
        elif name == 'ReynoldsStressDissipationScale':
            name = 'TurbulentDissipationRate'
            value /= Density
        elif name.startswith('ReynoldsStress'):
            name = name.replace('ReynoldsStress', 'VelocityCorrelation')
            value /= Density
        turbDict[name] = value

        # If the 'primitive' value is given in kwargs
        turbDict[name] = kwargs.get(name, value)
    return turbDict
