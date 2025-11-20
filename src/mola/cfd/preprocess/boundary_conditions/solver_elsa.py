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

from pathlib import Path
import numpy as np

import Converter.PyTree as C
import Converter.Internal as I

from treelab import cgns
from mola import naming_conventions as names
from mola.logging import mola_logger, MolaException, MolaUserError, mute_stdout
from mola.cfd.preprocess.solver_specific_tools import solver_elsa
from mola.cfd.preprocess.motion import motion
from mola.cfd.preprocess.motion.solver_elsa import assert_rotation_axis_is_correct, translate_motion_to_elsa
from mola.cfd.preprocess.boundary_conditions import boundary_conditions
from mola.cfd.preprocess.mesh.families import get_zone_family_from_bc_or_gc_family
import mola.server as SV

DEFAULT_NUMBER_OF_HARMONICS = 10

def define_bc_family(tree, Family, Value):
    familyNode = tree.get(Name=Family, Type='Family', Depth=2)
    familyNode.findAndRemoveNode(Name='.Solver#BC', Depth=1)
    familyNode.findAndRemoveNodes(Type='FamilyBC', Depth=1)
    cgns.Node( Name='FamilyBC', Value=Value, Type='FamilyBC', Parent=familyNode )
    return familyNode

def impose_bc_fields(bc_node, ImposedVariables, GridLocation='FaceCenter', BCDataSetName='BCDataSet#Init', BCDataName='NeumannData'):
    if len(ImposedVariables) > 0:
        # do not add a node BCDataSet_t if there is no variable to impose
        BCDataSet = cgns.Node(Name=BCDataSetName, Value='Null', Type='BCDataSet', Parent=bc_node)
        cgns.Node(Name='GridLocation', Type='GridLocation', Value=GridLocation, Parent=BCDataSet)
        BCDataSet.setParameters(BCDataName, ContainerType='BCData', **ImposedVariables)

def wall(workflow, Family, Motion=None, bctype_cgns='BCWallViscous', bctype_elsa='walladia'):
    '''
    Set a wall boundary condition.

    Parameters
    ----------

        workflow : Workflow object

        Family : str
            Name of the family on which the boundary condition will be imposed

        Motion : dict, optional
            Example:

            .. code-block:: python
                Motion = dict(
                    RotationSpeed = [1000., 0., 0.],
                    RotationAxisOrigin = [0., 0., 0.],
                    TranslationSpeed = [0., 0., 0.]
                    )

        bctype_cgns : str, optional
            Type of the bc in CGNS standard, value of the node 'FamilyBC'.  

        bctype_elsa : str, optional
            Type of the bc in elsA convention, value of the node 'type'.    
    '''
    wall = define_bc_family(workflow.tree, Family, bctype_cgns)

    if Motion is None: 
        return
    else: 
        assert isinstance(Motion, dict)
        motion.update_motion_with_defaults(Motion)

    if callable(Motion) or any([callable(v) for v in Motion.values()]):
        # Put global parameters in the family
        Motion_default = dict(RotationSpeed=workflow.ApplicationContext['ShaftAxis'])
        motion.update_motion_with_defaults(Motion_default)
        assert_rotation_axis_is_correct(Motion_default)
        Motion_elsa = translate_motion_to_elsa(Motion_default)
        Motion_elsa.pop('omega')
        wall.setParameters('.Solver#BC',
                            type=bctype_elsa,
                            data_frame='user',
                            **Motion_elsa
                            )
        # Put omega values in each bc
        non_uniform_fields = boundary_conditions.apply_function_to_BCDataSet(
            workflow, 
            Family, 
            functions_to_apply=dict(RotationSpeed=Motion['RotationSpeed'])
            )
        for bc_path, ImposedVariables in non_uniform_fields.items():
            assert list(ImposedVariables) == ['RotationSpeed'], f'{list(ImposedVariables)=}'
            bc_node = workflow.tree.getAtPath(bc_path)
            impose_bc_fields(bc_node, dict(omega = ImposedVariables['RotationSpeed']))

    else:
        assert_rotation_axis_is_correct(Motion)
        wall.setParameters('.Solver#BC',
                            type=bctype_elsa,
                            data_frame='user',
                            **translate_motion_to_elsa(Motion)
                            )

def wallslip(workflow, Family, Motion=None):
    '''
    Set an inviscid wall boundary condition.

    .. note:: see `elsA Tutorial about wall conditions <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/BCsTutorials/tutorial-BC.html#wall-conditions/>`_

    '''
    wall(workflow, Family, Motion=Motion, bctype_cgns='BCWallInviscid', bctype_elsa='wallslip')

def walladia(workflow, Family, Motion=None):
    '''
    Set a viscous wall boundary condition.

    .. note:: see `elsA Tutorial about wall conditions <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/BCsTutorials/tutorial-BC.html#wall-conditions/>`_
    
    '''
    wall(workflow, Family, Motion=Motion, bctype_cgns='BCWallViscous', bctype_elsa='walladia')

def sym(workflow, Family):
    '''
    Set a symmetry boundary condition.

    .. note:: see `elsA Tutorial about symmetry condition <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/BCsTutorials/tutorial-BC.html#symmetry/>`_

    '''
    define_bc_family(workflow.tree, Family, 'BCSymmetryPlane')


# Physical boundary conditions
    
def nref(workflow, Family, **kwargs):
    '''
    Set a nref boundary condition.

    Parameters
    ----------

        workflow.tree : PyTree
            Tree to modify

        Family : str
            Name of the family on which the boundary condition will be imposed

    '''
    if not kwargs:
        define_bc_family(workflow.tree, Family, 'BCFarfield')
    else:
        variables_from_file = ['Density', 'MomentumX', 'MomentumY', 'MomentumZ', 'EnergyStagnationDensity']
        variables_from_file += list(workflow.Turbulence['Conservatives'])

        set_physical_boundary(workflow, Family, 
                            FamilyBC='BCFarfield', interface_function=nref_interface,
                            variables_from_file=variables_from_file,
                            **kwargs
                            )

def inj1(workflow, Family, **kwargs):
    set_physical_boundary(workflow, Family, 
                          FamilyBC='BCInflowSubsonic', interface_function=inj1_interface,
                          **kwargs
                          )

def injmfr1(workflow, Family, **kwargs):
    set_physical_boundary(workflow, Family, 
                          FamilyBC='BCInflowSubsonic', interface_function=injmfr1_interface,
                          **kwargs
                          )
    
def giles_inlet(workflow, Family, **kwargs):
    set_physical_boundary(workflow, Family, 
                          FamilyBC='BCInflowSubsonic', interface_function=giles_inlet_interface,
                          force_parameters_at_bc_level=True,
                          **kwargs
                          )

def outpres(workflow, Family, **kwargs):   
    set_physical_boundary(workflow, Family, 
                          FamilyBC='BCOutflowSubsonic', interface_function=outpres_interface,
                          **kwargs
                          )

def outsup(workflow, Family):
    define_bc_family(workflow.tree, Family, 'BCOutflowSupersonic')

def outmfr1(workflow, Family, **kwargs):
    set_physical_boundary(workflow, Family, 
                          FamilyBC='BCOutflowSubsonic', interface_function=outmfr1_interface,
                          **kwargs
                          )

def outmfr2(workflow, Family, **kwargs):
    set_physical_boundary(workflow, Family, 
                          FamilyBC='BCOutflowSubsonic', interface_function=outmfr2_interface,
                          **kwargs
                          )
  
def giles_outlet(workflow, Family, **kwargs):
    set_physical_boundary(workflow, Family, 
                          FamilyBC='BCOutflowSubsonic', interface_function=giles_outlet_interface,
                          force_parameters_at_bc_level=True,
                          **kwargs
                          )

def nref_interface(workflow, **kwargs):
    ImposedVariables = workflow.Flow['Conservatives'] + workflow.Turbulence['Conservatives']
    for key, value in kwargs.items():
        if key in ImposedVariables:
            ImposedVariables[key] = value
    NumericalParameters = dict(
        type = 'nref',
    )
    return ImposedVariables, NumericalParameters

def inj1_interface(workflow, **kwargs):
    '''
    This interface function must return a dict with the variables really expected by elsA
    '''
    PressureStagnation    = kwargs.get('PressureStagnation', workflow.Flow['PressureStagnation'])
    TemperatureStagnation = kwargs.get('TemperatureStagnation', workflow.Flow['TemperatureStagnation'])
    EnthalpyStagnation    = kwargs.get('EnthalpyStagnation', workflow.Fluid['cp'] * TemperatureStagnation)
    VelocityUnitVectorX   = kwargs.get('VelocityUnitVectorX', workflow.Flow['Direction'][0])
    VelocityUnitVectorY   = kwargs.get('VelocityUnitVectorY', workflow.Flow['Direction'][1])
    VelocityUnitVectorZ   = kwargs.get('VelocityUnitVectorZ', workflow.Flow['Direction'][2])

    ImposedVariables = dict(
        PressureStagnation  = PressureStagnation,
        EnthalpyStagnation  = EnthalpyStagnation,
        VelocityUnitVectorX = VelocityUnitVectorX,
        VelocityUnitVectorY = VelocityUnitVectorY,
        VelocityUnitVectorZ = VelocityUnitVectorZ,
        **boundary_conditions.get_turbulent_primitives(workflow, **kwargs)
        )
    NumericalParameters = dict(
        type = 'inj1',
    )
    return ImposedVariables, NumericalParameters
       
def injmfr1_interface(workflow, **kwargs):
    Surface = kwargs.get('Surface')
    if not Surface:
        from mola.cfd.preprocess.mesh.tools import get_surface_of_family
        Surface = get_surface_of_family(workflow.tree, kwargs['Family'])
        try:
            Surface *= workflow.ApplicationContext['NormalizationCoefficient'][kwargs['Family']]['FluxCoef']
        except:
            pass

    MassFlow = kwargs.get('MassFlow')
    if MassFlow is None:
        try:
            MassFlow = workflow.Flow['MassFlow']
        except:
            MolaException('Error for InflowMassFlow boundary condition: '
                          'MassFlow is neither given by user as a boundary parameter, '
                          'nor foundable in workflow Flow attribute.')
            
    SurfacicMassFlow      = kwargs.get('SurfacicMassFlow', MassFlow / Surface)

    TemperatureStagnation = kwargs.get('TemperatureStagnation', workflow.Flow['TemperatureStagnation'])
    EnthalpyStagnation    = kwargs.get('EnthalpyStagnation', workflow.Fluid['cp'] * TemperatureStagnation)
    VelocityUnitVectorX   = kwargs.get('VelocityUnitVectorX', workflow.Flow['Direction'][0])
    VelocityUnitVectorY   = kwargs.get('VelocityUnitVectorY', workflow.Flow['Direction'][1])
    VelocityUnitVectorZ   = kwargs.get('VelocityUnitVectorZ', workflow.Flow['Direction'][2])

    ImposedVariables = dict(
        SurfacicMassFlow    = SurfacicMassFlow,
        EnthalpyStagnation  = EnthalpyStagnation,
        VelocityUnitVectorX = VelocityUnitVectorX,
        VelocityUnitVectorY = VelocityUnitVectorY,
        VelocityUnitVectorZ = VelocityUnitVectorZ,
        **boundary_conditions.get_turbulent_primitives(workflow, **kwargs)
        )
    NumericalParameters = dict(
        type = 'injmfr1',
    )
    return ImposedVariables, NumericalParameters

def outpres_interface(workflow, **kwargs):
    ImposedVariables = dict(
        Pressure = kwargs.get('Pressure', workflow.Flow['Pressure'])
        )
    NumericalParameters = dict(
        type = 'outpres',
    )
    return ImposedVariables, NumericalParameters

def outmfr1_interface(workflow, **kwargs):

    SurfacicMassFlow = kwargs.get('SurfacicMassFlow')
    if not SurfacicMassFlow:

        from mola.cfd.preprocess.mesh.tools import get_surface_of_family
        surface = get_surface_of_family(workflow.tree, kwargs['Family'])

        MassFlow = kwargs.get('MassFlow')
        if not MassFlow:
            MassFlow = workflow.Flow.get('MassFlow')

        try:
            fluxcoeff = workflow.ApplicationContext['NormalizationCoefficient'][kwargs['Family']]['FluxCoef']
        except: 
            fluxcoeff = 1.

        SurfacicMassFlow = MassFlow / surface / fluxcoeff

    ImposedVariables = dict(
        surf_massflow = SurfacicMassFlow,
        )
    NumericalParameters = dict(
        type = 'outmfr1',
    )
    return ImposedVariables, NumericalParameters

def outmfr2_interface(workflow, groupmassflow=1, **kwargs):
    MassFlow = kwargs.get('MassFlow')
    if not MassFlow:
        MassFlow = workflow.Flow.get('MassFlow')

    if not MassFlow:
        from mola.cfd.preprocess.mesh.tools import get_surface_of_family
        surface = get_surface_of_family(workflow.tree, kwargs['Family'])
        MassFlow = workflow.Flow['Density']*workflow.Flow['Velocity']*surface

    try:
        fluxcoeff = workflow.ApplicationContext['NormalizationCoefficient'][kwargs['Family']]['FluxCoef']
    except: 
        fluxcoeff = 1.

    MassFlowOnBC = MassFlow / fluxcoeff

    ImposedVariables = dict(
        globalmassflow = MassFlowOnBC,
        )
    NumericalParameters = dict(
        type = 'outmfr2',
        groupmassflow = groupmassflow,
    )
    return ImposedVariables, NumericalParameters

def outradeq_interface(workflow, Family, **kwargs):
    
    try:
        # Case where Type='outradeq' (not to pass through MOLA OutflowRadialEquilibrium_interface)
        # and all elsA parameters are direcly given
        parameters = dict(
            valve_type = kwargs['valve_type'], 
            valve_ref_pres = kwargs['valve_ref_pres'],
            valve_ref_mflow = kwargs['valve_ref_mflow'], 
            valve_relax = kwargs['valve_relax'],
            indpiv = kwargs['indpiv'],
            dirorder = kwargs['dirorder'],
            )
        return parameters
    except KeyError:
        pass

    # Default values, will be updated below depending on the valve law
    valve_ref_pres = workflow.Flow['Pressure']
    valve_ref_mflow = workflow.Flow['MassFlow']
    valve_relax = 0.1
    indpiv = kwargs.get('indpiv', 1)
    dirorder = kwargs.get('dirorder', -1)

    ValveLaw = kwargs.get('ValveLaw')
    if not ValveLaw:
        if not 'MassFlow' in kwargs:
            valve_type = 0
            if 'PressureAtHub' in kwargs:
                indpiv = 1
                valve_ref_pres = kwargs['PressureAtHub']
            elif 'PressureAtShroud' in kwargs:
                indpiv = -1
                valve_ref_pres = kwargs['PressureAtShroud']
            elif 'PressureAtSpecifiedHeight' in kwargs:
                raise MolaUserError('only PressureAtHub or PressureAtShroud can be provided for a simulation with elsA using MOLA preprocessing.')
            else:
                raise MolaUserError('PressureAtHub is missing')

        else:
            valve_type = 2
            fluxcoeff = boundary_conditions.get_fluxcoeff_on_bc(workflow, Family)
            valve_ref_mflow = kwargs['MassFlow'] / fluxcoeff
            pref = kwargs.get('PressureRef')
            if pref is not None:
                valve_ref_pres = pref 

    elif ValveLaw['Type'] == 'Linear':
        valve_type = 1
        fluxcoeff = boundary_conditions.get_fluxcoeff_on_bc(workflow, Family)
        valve_ref_pres = ValveLaw['PressureRef']
        valve_ref_mflow = ValveLaw['MassFlowRef'] / fluxcoeff
        valve_relax = ValveLaw['RelaxationCoefficient']

    elif ValveLaw['Type'] == 'Quadratic':
        valve_type = 4
        fluxcoeff = boundary_conditions.get_fluxcoeff_on_bc(workflow, Family)
        valve_ref_pres = ValveLaw['PressureRef']
        valve_ref_mflow = ValveLaw['MassFlowRef'] / fluxcoeff
        valve_relax = ValveLaw['ValveCoefficient'] * workflow.Flow['PressureStagnation']

    else:
        raise MolaUserError(f"Valve law {ValveLaw['Type']} is not available with elsA. Available laws are 'Linear' and 'Quadratic'.")

    parameters = dict(
        valve_ref_pres = valve_ref_pres,
        indpiv = indpiv,
        dirorder = dirorder,
        )
    
    if valve_type != 0:
        parameters.update(dict(
            valve_type = valve_type, 
            valve_ref_mflow = valve_ref_mflow, 
            valve_relax = valve_relax,
            ))
    return parameters

def outradeqhyb_interface(workflow, Family, **kwargs):
    parameters = outradeq_interface(workflow, Family, **kwargs)
    # HACK about nbband: do not use the default value in etc (-1). 
    # According the doc, "If -1, the value is automatically determined", but instead nbband is set to 3.
    parameters['nbband'] = kwargs.get('nbband', 100)  
    parameters['c'] = kwargs.get('c', 0.3) # default value in etc is 0.1
    return parameters

def set_physical_boundary(workflow, Family, 
                          FamilyBC, interface_function,
                          File=None, variableForInterpolation='ChannelHeight',
                          BCDataSetName='BCDataSet#Init', BCDataName='DirichletData', 
                          force_parameters_at_bc_level=False,  # Even if only scalar parameters are imposed, force BCDataSet to be written in each BC and not in Family
                          **kwargs 
                          ):
    
    kwargs['Family'] = Family
    ImposedVariables, NumericalParameters = interface_function(workflow, **kwargs)
    # Generally, except some cases due to how BC is implemented in elsa:
    #    ImposedVariables -> in BCDataSet#Init
    #    NumericalParameters -> in .Solver#BC

    all_imposed_values_are_scalars = all([np.ndim(v) == 0 and not callable(v) for v in ImposedVariables.values()])

    FamilyNode = define_bc_family(workflow.tree, Family, FamilyBC)

    if File is not None:

        input_data_from_file = boundary_conditions.get_fields_from_file(
            workflow.tree, Family, File, var2interp=list(ImposedVariables)
            )
        for bc_path, ImposedVariables in input_data_from_file.items():  
            bc = workflow.tree.getAtPath(bc_path)
            _apply_BCDataSet_on_bc(bc, ImposedVariables, NumericalParameters, BCDataSetName, BCDataName, variableForInterpolation)

    elif not all_imposed_values_are_scalars or force_parameters_at_bc_level:
        for bc in boundary_conditions.get_bc_nodes_from_family(workflow.tree, Family):
            _apply_BCDataSet_on_bc(bc, ImposedVariables, NumericalParameters, BCDataSetName, BCDataName, variableForInterpolation)
    else:
        checkVariables(ImposedVariables)
        ImposedVariables = solver_elsa.translate_to_elsa(ImposedVariables)
        ImposedVariables.update(NumericalParameters)
        FamilyNode.setParameters('.Solver#BC', **ImposedVariables)    

def _apply_BCDataSet_on_bc(bc: cgns.Node, ImposedVariables, NumericalParameters, BCDataSetName='BCDataSet#Init', BCDataName='DirichletData', variableForInterpolation='ChannelHeight'):

    bc.setParameters('.Solver#BC', **NumericalParameters)
    
    var2interp_value = None
    if any([callable(value) for value in ImposedVariables.values()]):
        # At least one value will be intepolated regarding variableForInterpolation
        var2interp_value = _get_variable_on_bc(bc, variableForInterpolation)
        bc_shape = var2interp_value.shape
    else:
        x_bc = _get_variable_on_bc(bc, 'CoordinateX')
        bc_shape = x_bc.shape

    for var, value in ImposedVariables.items():
        if callable(value):
            ImposedVariables[var] = value(var2interp_value) 
        elif np.ndim(value)==0:
            # scalar value --> uniform data
            ImposedVariables[var] = value * np.ones(bc_shape, order='F')
        if not ImposedVariables[var].shape == bc_shape:
            raise MolaException((
                f'Wrong shape for variable {var}: {ImposedVariables[var].shape} '
                f'(shape {bc_shape} for {bc.path()})'
            ))
        
    checkVariables(ImposedVariables)
    impose_bc_fields(bc, ImposedVariables, BCDataSetName=BCDataSetName, BCDataName=BCDataName)

def _get_variable_on_bc(bc: cgns.Node, variableForInterpolation: str):
    zone = bc.getParent(Type='Zone_t')  # FIXME in treelab: allow Type='Zone', that is not possible just for this method
    if variableForInterpolation in ['Radius', 'radius']:
        # FIXME Generalize that to other axis that X
        y, z = zone.yz()
        radius = np.srqt(y**2+z**2)
    elif variableForInterpolation.startswith('Coordinate') or variableForInterpolation == 'ChannelHeight':
        try:
            radius = zone.get(Name=variableForInterpolation).value()
        except AttributeError:
            raise AttributeError(f'Variable {variableForInterpolation} not found in zone {zone.path()}')
    else:
        raise ValueError('varForInterpolation must be ChannelHeight, Radius, CoordinateX, CoordinateY or CoordinateZ')

    PointRangeNode = bc.get(Type='IndexRange')
    if PointRangeNode:
        # Structured mesh
        PointRange = PointRangeNode.value()
        bc_shape = PointRange[:, 1] - PointRange[:, 0]
        if bc_shape[0] == 0:
            bc_shape = (bc_shape[1], bc_shape[2])
            radius = radius[PointRange[0, 0]-1,
                            PointRange[1, 0]-1:PointRange[1, 1]-1, 
                            PointRange[2, 0]-1:PointRange[2, 1]-1]
        elif bc_shape[1] == 0:
            bc_shape = (bc_shape[0], bc_shape[2])
            radius = radius[PointRange[0, 0]-1:PointRange[0, 1]-1,
                            PointRange[1, 0]-1, 
                            PointRange[2, 0]-1:PointRange[2, 1]-1]
        elif bc_shape[2] == 0:
            bc_shape = (bc_shape[0], bc_shape[1])
            radius = radius[PointRange[0, 0]-1:PointRange[0, 1]-1,
                            PointRange[1, 0]-1:PointRange[1, 1]-1,
                            PointRange[2, 0]-1]
        else:
            raise ValueError(f'Wrong BC shape {bc_shape} in {bc.path()}')
    
    else: 
        # Unstructured mesh
        PointList = bc.get(Type='IndexArray').value()
        bc_shape = PointList.size
        radius = radius[PointList-1]

    return radius

def checkVariables(ImposedVariables):
    '''
    Check that variables in the input dictionary are well defined. Raise a
    ``ValueError`` if not.

    Parameters
    ----------

        ImposedVariables : dict
            Each key is a variable name. Based on this name, the value (float or
            numpy.array) is checked.
            For instance:

                * Variables such as pressure, temperature or turbulent quantities
                  must be strictly positive.

                * Components of a unit vector must be between -1 and 1.

    '''
    posiviteVars = ['PressureStagnation', 'EnthalpyStagnation',
        'stagnation_pressure', 'stagnation_enthalpy', 'stagnation_temperature',
        'Pressure', 'pressure', 'Temperature', 'wall_temp',
        'TurbulentEnergyKinetic', 'TurbulentDissipationRate', 'TurbulentDissipation', 'TurbulentLengthScale',
        'TurbulentSANuTilde', 'globalmassflow', 'MassFlow', 'surf_massflow']
    unitVectorComponent = ['VelocityUnitVectorX', 'VelocityUnitVectorY', 'VelocityUnitVectorZ',
        'txv', 'tyv', 'tzv']

    def positive(value):
        if value is None: 
            return False
        if isinstance(value, np.ndarray): return np.all(value>0)
        else: return value>0

    def unitComponent(value):
        if value is None: 
            return False
        if isinstance(value, np.ndarray): return np.all(np.absolute(value)<=1)
        else: return abs(value)<=1

    for var, value in ImposedVariables.items():
        if var in posiviteVars and not positive(value):
            raise ValueError('{} must be positive, but here it is equal to {}'.format(var, value))
        elif var in unitVectorComponent and not unitComponent(value):
            raise ValueError('{} must be between -1 and +1, but here it is equal to {}'.format(var, value))

@mute_stdout
def outradeq(workflow, Family, **kwargs):
    '''
    Set an outflow boundary condition of type ``outradeq``.

    .. important : This function has a dependency to the ETC module.

    Parameters
    ----------

        workflow : Workflow

        FamilyName : str
            Name of the family on which the boundary condition will be imposed

        valve_type : int
            Valve law type. See `elsA documentation about valve laws <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/STB-97020/Textes/Boundary/Valve.html>`_.
            If 0, not valve law is used. In that case, **valve_ref_pres** corresponds
            to the prescribed static pressure at the pivot index, and **valve_ref_mflow**
            and **valve_relax** are not used.

        valve_ref_pres : :py:class:`float` or :py:obj:`None`
            Reference static pressure at the pivot index.
            If :py:obj:`None`, the value ``ReferenceValues['Pressure']`` is taken.

        valve_ref_mflow : :py:class:`float` or :py:obj:`None`
            Reference mass flow rate.
            If :py:obj:`None`, the value ``ReferenceValues['MassFlow']`` is taken
            and normalized using information in **TurboConfiguration** to get
            the corresponding mass flow rate on the section of **FamilyName**
            actually simulated.

        valve_relax : float
            'Relaxation' parameter of the valve law. The default value is 0.1.
            Be careful:

            * for laws 1, 2 and 5, it is a real Relaxation coefficient without
              dimension.

            * for law 3, it is a value homogeneous with a pressure divided
              by a mass flow.

            * for law 4, it is a value homogeneous with a pressure.
        
        indpiv : int
            Index of the cell where the pivot value is imposed.

    '''
    if not workflow.tree.isStructured():
        raise MolaUserError(f'The boundary condition "outradeq" on Family {Family} is available only for structured mesh.')

    import etc.transform as trf
    t = workflow.tree

    params = outradeq_interface(workflow, Family, **kwargs)

    # Delete previous BC if it exists
    for bc in C.getFamilyBCs(t, Family):
        I._rmNodesByName(bc, '.Solver#BC')
    define_bc_family(t, Family, 'BCOutflowSubsonic')

    from etc.globborder.globborder_dict import globborder_dict
    gbd = globborder_dict(t, Family, config="axial")

    for bcn in C.getFamilyBCs(t, Family):
        bcpath = I.getPath(t, bcn)
        bc = trf.BCOutRadEq(t, bcn)
        bc.indpiv = params['indpiv']
        bc.dirorder = params['dirorder']
        # Valve laws:
        # <bc>.valve_law(valve_type, pref, Qref, valve_relax=relax, valve_file=None, valve_file_freq=1) # v4.2.01 pour valve_file*
        # valvelaws = [(1, 'SlopePsQ'),     # p(it+1) = p(it) + relax*( pref * (Q(it)/Qref) -p(it)) # relax = sans dim. # isoPs/Q
        #              (2, 'QTarget'),      # p(it+1) = p(it) + relax*pref * (Q(it)/Qref-1)         # relax = sans dim. # debit cible
        #              (3, 'QLinear'),      # p(it+1) = pref + relax*(Q(it)-Qref)                  # relax = Pascal    # lin en debit
        #              (4, 'QHyperbolic'),  # p(it+1) = pref + relax*(Q(it)/Qref)**2               # relax = Pascal    # comp. exp.
        #              (5, 'SlopePiQ')]     # p(it+1) = p(it) + relax*( pref * (Q(it)/Qref) -pi(it)) # relax = sans dim. # isoPi/Q
        # for law 5, pref = reference total pressure
        if params['valve_type'] == 0:
            bc.prespiv = params['valve_ref_pres']
        else:
            valve_law_dict = {1: 'SlopePsQ', 2: 'QTarget', 3: 'QLinear', 4: 'QHyperbolic'}
            bc.valve_law(valve_law_dict[params['valve_type']], params['valve_ref_pres'],
                         params['valve_ref_mflow'], valve_relax=params['valve_relax'], valve_file=f'prespiv_{Family}.log')
        globborder = bc.glob_border(current=Family)
        globborder.i_poswin = gbd[bcpath]['i_poswin']
        globborder.j_poswin = gbd[bcpath]['j_poswin']
        globborder.glob_dir_i = gbd[bcpath]['glob_dir_i']
        globborder.glob_dir_j = gbd[bcpath]['glob_dir_j']
        globborder.azi_orientation = gbd[bcpath]['azi_orientation']
        globborder.h_orientation = gbd[bcpath]['h_orientation']
        bc.create()

    workflow.tree = cgns.castNode(t)

@mute_stdout
def outradeqhyb(workflow, Family, **kwargs):
    '''
    Set an outflow boundary condition of type ``outradeqhyb``.

    .. important : This function has a dependency to the ETC module.

    Parameters
    ----------

        t : PyTree
            Tree to modify

        FamilyName : str
            Name of the family on which the boundary condition will be imposed

        valve_type : int
            Valve law type. See `elsA documentation about valve laws <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/STB-97020/Textes/Boundary/Valve.html>`_.
            Cannot be 0.

        valve_ref_pres : float
            Reference static pressure at the pivot index.

        valve_ref_mflow : float
            Reference mass flow rate.

        valve_relax : float
            'Relaxation' parameter of the valve law. The default value is 0.1.
            Be careful:

            * for laws 1, 2 and 5, it is a real Relaxation coefficient without
              dimension.

            * for law 3, it is a value homogeneous with a pressure divided
              by a mass flow.

            * for law 4, it is a value homogeneous with a pressure.
        
        indpiv : int
            Index of the cell where the pivot value is imposed.

        nbband : int
            Number of points in the radial distribution to compute.

        c : float
            Parameter for the distribution of radial points.
        
        ReferenceValues : :py:class:`dict` or :py:obj:`None`
            as produced by :py:func:`computeReferenceValues`

        TurboConfiguration : :py:class:`dict` or :py:obj:`None`
            as produced by :py:func:`getTurboConfiguration`


    '''
    import etc.transform as trf
    t = workflow.tree

    params = outradeqhyb_interface(workflow, Family, **kwargs)

    # Delete previous BC if it exists
    for bc in C.getFamilyBCs(t, Family):
        I._rmNodesByName(bc, '.Solver#BC')
    define_bc_family(t, Family, 'BCOutflowSubsonic')

    bc = trf.BCOutRadEqHyb(t, t.get(Name=Family, Type='Family'))
    bc.glob_border()
    bc.indpiv = params['indpiv']
    if 'valve_type' not in params or params['valve_type'] == 0:
        bc.prespiv = params['valve_ref_pres']
    else:
        valve_law_dict = {1: 'SlopePsQ', 2: 'QTarget', 3: 'QLinear', 4: 'QHyperbolic'}
        bc.valve_law(valve_law_dict[params['valve_type']], params['valve_ref_pres'],
                    params['valve_ref_mflow'], valve_relax=params['valve_relax'], 
                    valve_file=f'prespiv_{Family}.log')
    bc.dirorder = params['dirorder']

    write_radius_in_cgns = False
    if write_radius_in_cgns:
        # NOT WORKING FOR NOW
        radius = bc.repartition()
        radius.compute(t, nbband=params['nbband'], c=params['c'])
    
    else:
        radius_filename = f'radius_{Family}.plt'
        radius = bc.repartition(filename=radius_filename, fileformat="bin_tp")
        radius.compute(t, nbband=params['nbband'], c=params['c'])
        radius.write()
        # Move radius files to the RunDirectory
        # HACK This will be outdated as soon as the radius distribution is written directly in the CGNS file
        # see https://elsa-e.onera.fr/issues/10541
        if Path(workflow.RunManagement['RunDirectory']).resolve() != Path.cwd():
            SV.move_remote(
                source_path=radius_filename, 
                destination_path=Path(workflow.RunManagement['RunDirectory']) / Path(radius_filename), 
                destination_machine=workflow.RunManagement['Machine'],
                force_copy=True
                )
            
    bc.create()
    workflow.tree = cgns.castNode(t)

@mute_stdout
def stage_mxpl(workflow, Family, LinkedFamily):
    '''
    Set a mixing plane condition between families **Family** and **LinkedFamily**.

    .. important : This function has a dependency to the ETC module.

    '''
    if not workflow.tree.isStructured():
        raise MolaUserError(f'The boundary condition "stage_mxpl" on families {Family} and {LinkedFamily} is available only for structured mesh.')

    import etc.transform as trf

    # HACK: must change the type of all FamilyName to array
    # For a unknown reason, nodes FamilyName have value of type str instead of ndarray,
    # and that makes a bug in trf.defineBCStageFromBC (in CGU.getValueAsString(FamilyName))
    for FamilyName_node in workflow.tree.group(Type='FamilyName'):
        FamilyName_node.setValue(FamilyName_node.value())

    _fix_point_range_in_gc(workflow.tree)

    workflow.tree = trf.defineBCStageFromBC(workflow.tree, (Family, LinkedFamily))
    workflow.tree, stage = trf.newStageMxPlFromFamily(workflow.tree, Family, LinkedFamily)

    stage.jtype = 'nomatch_rad_line'
    stage.create()

    workflow.tree = cgns.castNode(workflow.tree)
    _restore_point_range_in_gc(workflow.tree)
    set_turbomachinery_interface_FamilyBC(workflow.tree, Family, LinkedFamily)
    # GC names must be unique to use globborders in elsa, otherwise the error "Error : duplicated object name!" will be raised
    I._correctPyTree(workflow.tree, level=4)

@mute_stdout
def stage_red(workflow, Family, LinkedFamily, SectorPassagePeriod=None):
    '''
    Set a RNA condition between families **Family** and **LinkedFamily**.

    .. important : This function has a dependency to the ETC module.

    '''
    if not workflow.tree.isStructured():
        raise MolaUserError(f'The boundary condition "stage_red" on families {Family} and {LinkedFamily} is available only for structured mesh.')

    import etc.transform as trf

    if SectorPassagePeriod is None:
        SectorPassagePeriod = compute_RNA_ref_time(workflow, Family, LinkedFamily)

    # HACK: must change the type of all FamilyName to array
    # For a unknown reason, nodes FamilyName have value of type str instead of ndarray,
    # and that makes a bug in trf.defineBCStageFromBC (in CGU.getValueAsString(FamilyName))
    for FamilyName_node in workflow.tree.group(Type='FamilyName'):
        FamilyName_node.setValue(FamilyName_node.value())

    _fix_point_range_in_gc(workflow.tree)

    workflow.tree = trf.defineBCStageFromBC(workflow.tree, (Family, LinkedFamily))
    workflow.tree, stage = trf.newStageRedFromFamily(workflow.tree, Family, LinkedFamily, stage_ref_time=SectorPassagePeriod)

    stage.create()

    workflow.tree = cgns.castNode(workflow.tree)
    _restore_point_range_in_gc(workflow.tree)
    set_turbomachinery_interface_FamilyBC(workflow.tree, Family, LinkedFamily)
    # GC names must be unique to use globborders in elsa, otherwise the error "Error : duplicated object name!" will be raised
    I._correctPyTree(workflow.tree, level=4)

@mute_stdout
def stage_mxpl_hyb(workflow, Family, LinkedFamily, nbband=100, c=0.3, mxpl_dirtype='axial', write_radius_in_cgns=False):
    '''
    Set a hybrid mixing plane condition between families **Family** and **LinkedFamily**.

    mxpl_dirtype should be in ['axial', 'centrifugal', 'centripetal']

    .. important : This function has a dependency to the ETC module.

    '''
    import etc.transform as trf

    # HACK: must change the type of all FamilyName to array
    # For a unknown reason, nodes FamilyName have value of type str instead of ndarray,
    # and that makes a bug in trf.defineBCStageFromBC (in CGU.getValueAsString(FamilyName))
    for FamilyName_node in workflow.tree.group(Type='FamilyName'):
        FamilyName_node.setValue(FamilyName_node.value())

    workflow.tree = trf.defineBCStageFromBC(workflow.tree, (Family, LinkedFamily))
    workflow.tree, stage = trf.newStageMxPlHybFromFamily(workflow.tree, Family, LinkedFamily)

    stage.jtype = 'nomatch_rad_line'
    stage.hray_tolerance = 1e-16

    if write_radius_in_cgns:

        for stg in stage.down:
            radius = stg.repartition(mxpl_dirtype=mxpl_dirtype, 
                                    parent=workflow.tree.get(Name=Family, Type='Family', Depth=2),
                                    tree=workflow.tree)
            radius.compute(workflow.tree, nbband=nbband, c=c)
            # the method compute adds file=None, format='CGNS', mxpl_dirtype=mxpl_dirtype in the .Solver#Property node of the first found GC
            # It also overrides MixingPlaneData/radius in the Family

        for stg in stage.up:
            radius = stg.repartition(mxpl_dirtype=mxpl_dirtype, 
                                    parent=workflow.tree.get(Name=LinkedFamily, Type='Family', Depth=2), 
                                    tree=workflow.tree)
            radius.compute(workflow.tree, nbband=nbband, c=c)
            # the method compute adds file=None, format='CGNS', mxpl_dirtype=mxpl_dirtype in the .Solver#Property node of the first found GC
            # It also overrides MixingPlaneData/radius in the Family

        stage.create()
        workflow.tree = cgns.castNode(workflow.tree)

        # HACK file, format and mxpl_dirtype parameters are written only in one GC, but they may be several GC
        for gc in workflow.tree.group(Type='GridConnectivity'):
            if not gc.get(Type='FamilyName', Value=Family) and not gc.get(Type='FamilyName', Value=LinkedFamily):
                continue
            
            sp_node = gc.get(Name='.Solver#Property')
            cgns.Node(Name='file', Type='DataArray', Parent=sp_node)
            cgns.Node(Name='format', Type='DataArray', Value='CGNS', Parent=sp_node)
            cgns.Node(Name='mxpl_dirtype', Type='DataArray', Value=mxpl_dirtype, Parent=sp_node)

    else:

        filename_left = f'radius_{LinkedFamily}.plt'
        for stg in stage.down:
            radius = stg.repartition(mxpl_dirtype='axial', filename=filename_left, fileformat="bin_tp")
        radius.compute(workflow.tree, nbband=nbband, c=c)
        radius.write()

        filename_right = f'radius_{Family}.plt'
        for stg in stage.up:
            radius = stg.repartition(mxpl_dirtype='axial', filename=filename_right, fileformat="bin_tp")
        radius.compute(workflow.tree, nbband=nbband, c=c)
        radius.write()

        stage.create()

        # Move radius files to the RunDirectory
        # HACK This will be outdated as soon as the radius distribution is written directly in the CGNS file
        # see https://elsa-e.onera.fr/issues/10541
        if Path(workflow.RunManagement['RunDirectory']).resolve() != Path.cwd():
            for filename in [filename_left, filename_right]:
                SV.move_remote(
                    source_path=filename, 
                    destination_path=Path(workflow.RunManagement['RunDirectory']) / Path(filename), 
                    destination_machine=workflow.RunManagement['Machine'],
                    force_copy=True
                    )
                
        workflow.tree = cgns.castNode(workflow.tree)

        
    set_turbomachinery_interface_FamilyBC(workflow.tree, Family, LinkedFamily)
    # GC names must be unique to use globborders in elsa, otherwise the error "Error : duplicated object name!" will be raised
    I._correctPyTree(workflow.tree, level=4)

@mute_stdout
def stage_red_hyb(workflow, Family, LinkedFamily, SectorPassagePeriod=None):
    '''
    Set a hybrid RNA condition between families **Family** and **LinkedFamily**.

    .. important : This function has a dependency to the ETC module.

    '''
    import etc.transform as trf

    if SectorPassagePeriod is None:
        SectorPassagePeriod = compute_RNA_ref_time(workflow, Family, LinkedFamily)

    # HACK: must change the type of all FamilyName to array
    # For a unknown reason, nodes FamilyName have value of type str instead of ndarray,
    # and that makes a bug in trf.defineBCStageFromBC (in CGU.getValueAsString(FamilyName))
    for FamilyName_node in workflow.tree.group(Type='FamilyName'):
        FamilyName_node.setValue(FamilyName_node.value())

    workflow.tree = trf.defineBCStageFromBC(workflow.tree, (Family, LinkedFamily))
    workflow.tree, stage = trf.newStageRedHybFromFamily(workflow.tree, Family, LinkedFamily, stage_ref_time=SectorPassagePeriod)

    stage.create()

    for gc in I.getNodesFromType(workflow.tree, 'GridConnectivity_t'):
        I._rmNodesByType(gc, 'FamilyBC_t')

    workflow.tree = cgns.castNode(workflow.tree)

def compute_RNA_ref_time(workflow, Family, LinkedFamily):
    '''
    see https://elsa-doc.onera.fr/restricted/MU_MT_tuto/latest/Tutos/Speciality/StageRed.html#numerical-parameters
    '''    
    row1 = get_zone_family_from_bc_or_gc_family(workflow.tree, Family)
    row2 = get_zone_family_from_bc_or_gc_family(workflow.tree, LinkedFamily)

    LapPeriod = 2*np.pi / abs(workflow.ApplicationContext['ShaftRotationSpeed'])

    N1 = workflow.ApplicationContext['Rows'][row1]['NumberOfBlades']
    N2 = workflow.ApplicationContext['Rows'][row2]['NumberOfBlades']
    K1 = workflow.ApplicationContext['Rows'][row1]['NumberOfBladesSimulated']
    K2 = workflow.ApplicationContext['Rows'][row2]['NumberOfBladesSimulated']

    Dm = 2 / (K1/N1 + K2/N2)
    SectorPassagePeriod = LapPeriod / Dm

    msg = f'The reference time period for RNA interface is equal to {1/Dm} rotation period.'
    if np.isclose(Dm, 1) or np.isclose(Dm, K1/N1):
        mola_logger.info(msg, rank=0)
    else:
        mola_logger.warning(msg, rank=0)

    return SectorPassagePeriod

def chorochronic(workflow, Family, LinkedFamily, NumbersOfHarmonics=DEFAULT_NUMBER_OF_HARMONICS, hybrid=False):
    '''
    Compute the parameters to run a chorochronic computation.
    
    Parameters
    ----------

        workflow : Workflow
            Workflow instance

        Family : str
            Name of the family on the first side of the chorochronic interface.

        LinkedFamily : str
            Name of the family on the second side of the chorochronic interface.

        NumbersOfHarmonics : int or tuple
            Numbers of harmonics of the first row and for the second row.
        
        hybrid : bool
            If True, use the `stage_choro_hyb` condition, else use `stage_choro`.
    '''   
    if isinstance(NumbersOfHarmonics, int):
        NumbersOfHarmonics = (NumbersOfHarmonics, NumbersOfHarmonics)
    if hybrid:
        stage_choro_hyb(workflow, Family, LinkedFamily)
    else:
        stage_choro(workflow, Family, LinkedFamily)
    workflow.tree = cgns.castNode(workflow.tree)
    convert_periodic_to_chorochrono(workflow.tree)
    workflow.tree = cgns.castNode(workflow.tree)
    row1 = get_zone_family_from_bc_or_gc_family(workflow.tree, Family)
    row2 = get_zone_family_from_bc_or_gc_family(workflow.tree, LinkedFamily)
    choroParamsRow1, choroParamsRow2 = compute_choro_parameters(workflow.ApplicationContext, row1, row2, Nharm_Row1=NumbersOfHarmonics[0], Nharm_Row2=NumbersOfHarmonics[1])
    add_choro_data(workflow.tree, row1, **choroParamsRow1) 
    add_choro_data(workflow.tree, row2, **choroParamsRow2) 

@mute_stdout
def stage_choro(workflow, Family, LinkedFamily):
    '''
    Set a chorochronic interface condition between families **Family** and **LinkedFamily**.

    .. important : This function has a dependency to the ETC module.
    '''
    if not workflow.tree.isStructured():
        raise MolaUserError(f'The boundary condition "stage_choro" on families {Family} and {LinkedFamily} is available only for structured mesh.')

    import etc.transform as trf

    # HACK: must change the type of all FamilyName to array
    # For a unknown reason, nodes FamilyName have value of type str instead of ndarray,
    # and that makes a bug in trf.defineBCStageFromBC (in CGU.getValueAsString(FamilyName))
    for FamilyName_node in workflow.tree.group(Type='FamilyName'):
        FamilyName_node.setValue(FamilyName_node.value())

    _fix_point_range_in_gc(workflow.tree)

    workflow.tree = trf.defineBCStageFromBC(workflow.tree, (Family, LinkedFamily))
    workflow.tree, stage = trf.newStageChoroFromFamily(workflow.tree, Family, LinkedFamily)

    stage.jtype = 'nomatch_rad_line'
    stage.stage_choro_type = 'characteristic'
    stage.harm_freq_comp = 1
    stage.choro_file_up = 'None'
    stage.file_up = None
    stage.choro_file_down = 'None'
    stage.file_down = None
    stage.nomatch_special = 'None'
    stage.format = 'CGNS'

    stage.create()

    workflow.tree = cgns.castNode(workflow.tree)
    _restore_point_range_in_gc(workflow.tree)
    set_turbomachinery_interface_FamilyBC(workflow.tree, Family, LinkedFamily)
    # GC names must be unique to use globborders in elsa, otherwise the error "Error : duplicated object name!" will be raised
    I._correctPyTree(workflow.tree, level=4)

@mute_stdout
def stage_choro_hyb(workflow, Family, LinkedFamily):
    '''
    Set a hybrid chorochronic interface condition between families **Family** and **LinkedFamily**.

    .. important : This function has a dependency to the ETC module.
    '''
    if workflow.tree.isStructured():
        # error for this BC with structured mesh for elsa<v5.4.02
        # see https://elsa-e.onera.fr/issues/11891#note-33
        raise MolaUserError((
            f'The boundary condition "stage_choro_hyb" on families {Family} and {LinkedFamily} '
            'is available only for structured mesh for elsa<v5.4.02. See https://elsa-e.onera.fr/issues/11891#note-33'
        ))
    
    mola_logger.warning('Condition stage_choro_hyb has not been validated yet through MOLA, hence there may be some unexpected behaviors!')

    import etc.transform as trf

    # HACK: must change the type of all FamilyName to array
    # For a unknown reason, nodes FamilyName have value of type str instead of ndarray,
    # and that makes a bug in trf.defineBCStageFromBC (in CGU.getValueAsString(FamilyName))
    for FamilyName_node in workflow.tree.group(Type='FamilyName'):
        FamilyName_node.setValue(FamilyName_node.value())

    workflow.tree = trf.defineBCStageFromBC(workflow.tree, (Family, LinkedFamily))
    workflow.tree, stage = trf.newStageChoroHybFromFamily(workflow.tree, Family, LinkedFamily)

    # FIXME this issue https://elsa-e.onera.fr/issues/11902
    # indicates that values for parameters below may be wrong...
    stage.jtype = 'nomatch_rad_line'
    stage.stage_choro_type = 'characteristic'
    stage.harm_freq_comp = 1
    stage.choro_file_up = 'None'
    stage.file_up = None
    stage.choro_file_down = 'None'
    stage.file_down = None
    stage.nomatch_special = 'None'
    stage.format = 'CGNS'

    stage.create()

    workflow.tree = cgns.castNode(workflow.tree)
    set_turbomachinery_interface_FamilyBC(workflow.tree, Family, LinkedFamily)
    # GC names must be unique to use globborders in elsa, otherwise the error "Error : duplicated object name!" will be raised
    I._correctPyTree(workflow.tree, level=4)

def convert_periodic_to_chorochrono(t):
    '''
    Convert the periodic boundary condition from a PyTree t to a chorochrono boundary condition.
    '''
    import etc.transform as trf
    gcnodes = []
    for gc_node in t.group(Type='GridConnectivity1to1'):
        if gc_node.get(Type='Periodic'):
            gcnodes.append(gc_node)

    for gcnode in gcnodes:
        # Force RotationAngle to be [X, 0, 0], else error
        RotationAngle = gcnode.get(Name='RotationAngle').value()
        for i, angle in enumerate(RotationAngle):
            if abs(angle) < 1e-10:
                RotationAngle[i] = 0.
        gc = trf.BCChoroChrono(t, gcnode, choro_file='None')
        gc.create()

def compute_choro_parameters(ApplicationContext, row1, row2, Nharm_Row1, Nharm_Row2, relax=1.0):
    '''
    Compute the parameters to run a chorochronic computation.
    '''       
    Nblade_Row1 = ApplicationContext['Rows'][row1]['NumberOfBlades']
    Nblade_Row2 = ApplicationContext['Rows'][row2]['NumberOfBlades']
    omega_Row1 = ApplicationContext['ShaftRotationSpeed'] if ApplicationContext['Rows'][row1]['IsRotating'] else 0.
    omega_Row2 = ApplicationContext['ShaftRotationSpeed'] if ApplicationContext['Rows'][row2]['IsRotating'] else 0.

    if Nharm_Row1 < DEFAULT_NUMBER_OF_HARMONICS:
        mola_logger.user_warning(f'The number of harmonics for row {row1} ({Nharm_Row1}) is lower than the recommended value ({DEFAULT_NUMBER_OF_HARMONICS}')
    if Nharm_Row2 < DEFAULT_NUMBER_OF_HARMONICS:
        mola_logger.user_warning(f'The number of harmonics for row {row2} ({Nharm_Row2}) is lower than the recommended value ({DEFAULT_NUMBER_OF_HARMONICS}')
                            
    mola_logger.info(f'      {Nharm_Row1} harmonics for {row1} family', rank=0)
    mola_logger.info(f'      {Nharm_Row2} harmonics for {row2} family', rank=0)

    choroParamsRow1 = dict(
        f_freq = Nblade_Row2*np.abs(omega_Row1-omega_Row2)/(2*np.pi), 
        f_omega = float(omega_Row1 - omega_Row2), 
        f_harm = float(Nharm_Row1), 
        f_relax = float(relax), 
        axis_ang_1 = Nblade_Row1, 
        axis_ang_2 = ApplicationContext['Rows'][row1]['NumberOfBladesSimulated']
        )
    choroParamsRow2 = dict(
        f_freq = Nblade_Row1*np.abs(omega_Row1-omega_Row2)/(2*np.pi), 
        f_omega = float(omega_Row2 - omega_Row1), 
        f_harm = float(Nharm_Row2), 
        f_relax = float(relax), 
        axis_ang_1 = Nblade_Row2, 
        axis_ang_2 = ApplicationContext['Rows'][row2]['NumberOfBladesSimulated']
        )
    
    return choroParamsRow1, choroParamsRow2

def add_choro_data(t, rowName, f_freq, f_omega, f_harm, f_relax, axis_ang_1, axis_ang_2):
    '''
    Add the chorochronic parameters computed using compute_choro_parameters() to the PyTree t.
    
    Parameters
    ----------

        t : PyTree
            Tree to modify

        rowName : str
            Name of the considered row (must be the name of a Family_t node). 

        freq : float
            Frequency of blade passage to next wheel, as provided by compute_choro_parameters().

        Nharm : float
            Number of harmonics of the considered row, as provided by compute_choro_parameters().

        omega : float
            rotation speed in rad/s relative to the other row, as provided by compute_choro_parameters().

        relax : float
            Relaxation coefficient for multichoro condition, as provided by compute_choro_parameters(). Equals 1.0 for a single stage rotor/stator stage.

        axis_ang_1 : float
           Number of blades in the considered row, as provided by compute_choro_parameters().

        axis_ang_2 : float
            Number of simulated passages for the considered row, as provided by compute_choro_parameters().

    '''
    fam_node = t.get(Name=rowName, Type='Family', Depth=2)
    motion_node = fam_node.get(Name='.Solver#Motion')
    if motion_node is None:
        raise MolaException(f'Motion has not been defined for family {rowName} (cannot find the node .Solver#Motion)')
    cgns.Node(Name='axis_ang_1', Value=axis_ang_1, Type='DataArray', Parent=motion_node)
    cgns.Node(Name='axis_ang_2', Value=axis_ang_2, Type='DataArray', Parent=motion_node)

    for zone in t.zones():
        if not zone.get(Type='*FamilyName', Value=rowName):
            # Not in the right family
            continue
        solver_param = zone.setParameters('.Solver#Param', 
                        f_freq=f_freq,
                        f_omega=f_omega, 
                        f_harm=f_harm,
                        f_relax=f_relax,
                        )
        for node in motion_node.group(Name='axis_*'):
            solver_param.addChild(node)
    
def set_turbomachinery_interface_FamilyBC(t, left, right):
    for gc in t.group(Type='GridConnectivity'):
        gc.findAndRemoveNodes(Type='FamilyBC')
    
    leftFamily = t.get(Name=left, Type='Family', Depth=2)
    rightFamily = t.get(Name=right, Type='Family', Depth=2)
    cgns.Node(Name='FamilyBC', Type='FamilyBC', Value='BCOutflow', Parent=leftFamily)
    cgns.Node(Name='FamilyBC', Type='FamilyBC', Value='BCInflow', Parent=rightFamily)

def _compute_giles_monitoring_flag(tree):
    '''
    Search existing nodes 'monitoring_flag' in tree to get the next one. 
    For instance, if there are already BCs with monitoring_flag=1 et 2, this function return 3
    '''
    flag = 1
    nodes = tree.group(Name='type', Value='nscbc_in') \
            + tree.group(Name='type', Value='nscbc_out') \
            + tree.group(Name='type', Value='nscbc_mxpl')

    for node in nodes:
        monitoring_flag = [sibling for sibling in node.siblings() if sibling.name()=='monitoring_flag'][0].value()
        if monitoring_flag >= flag:
            flag = flag + 1 

    return flag

def giles_inlet_interface(workflow, Family, 
                          NumberOfModes, # number of Fourier modes
                          **kwargs
                          ):  

    # creation of dictionnary of keys for Giles inlet BC  
    NumericalParameters = dict()

    # keys relative to NSCBC
    NumericalParameters['type'] = 'nscbc_in'                                                                   # mandatory key to have NSCBC-Giles treatment
    NumericalParameters['nscbc_giles'] = 'statio'                                                              # mandatory key to have NSCBC-Giles treatment
    NumericalParameters['nscbc_interpbc'] = 'linear'                                                           # mandatory value
    NumericalParameters['nscbc_fluxt'] = kwargs.get('nscbc_fluxt', 'fluxBothTransv')                           # recommended value - possible keys : 'classic'; 'fluxInviscidTransv'; 'fluxBothTransv' 
    NumericalParameters['nscbc_surf'] = kwargs.get('nscbc_surf',  'revolution')                                # recommended value - possible keys : 'flat', 'revolution' 
    NumericalParameters['nscbc_outwave'] = kwargs.get('nscbc_outwave',  'grad_etat')                           # recommended value - possible keys : 'grad_etat'; 'extrap_flux'
    NumericalParameters['nscbc_velocity_scale'] = kwargs.get('nscbc_velocity_scale', workflow.Flow['SoundSpeed'])  # default value - sound velocity
    NumericalParameters['nscbc_viscwall_len'] = kwargs.get('nscbc_viscwall_len', 5.e-4)                        # default value, could be updated by the user if convergence issue

    if kwargs.get('nscbc_viscwall_len_hub') is not None:
        NumericalParameters['nscbc_viscwall_len_hub'] = kwargs.get('nscbc_viscwall_len_hub')                    # value of nscbc_viscwall_len for the hub only
    if kwargs.get('nscbc_viscwall_len_carter') is not None:
        NumericalParameters['nscbc_viscwall_len_carter'] = kwargs.get('nscbc_viscwall_len_carter')              # value of nscbc_viscwall_len for the hub only           


    # keys relative to the Giles treatment 
    NumericalParameters['giles_opt'] = kwargs.get('giles_relax_opt', 'relax')                             # mandatory key for NSCBC-Giles treatment
    NumericalParameters['giles_restric_relax'] = 'inactive'                                               # mandatory key for NSCBC-Giles treatment
    NumericalParameters['giles_exact_lodi'] = kwargs.get('giles_exact_lodi',  'active')                   # recommended value - possible keys: 'inactive', 'partial', 'active'
    NumericalParameters['giles_nbMode'] = NumberOfModes  # to be given by the user - recommended value : ncells_theta/2 + 1 (odd_value)

    # keys relative to the monitoring and radii calculus - monitoring data stored in LOGS
    NumericalParameters['bnd_monitoring'] = 'active'                                                      # recommended value
    NumericalParameters['monitoring_comp_rad'] = 'auto'                                                   # recommended value - possible keys: 'from_file', 'monofenetre'
    NumericalParameters['monitoring_tol_rad'] = kwargs.get('monitoring_tol_rad',  1e-6)                   # recommended value
    NumericalParameters['monitoring_var'] = 'psta pgen Tgen ux uy uz diffPgen diffTgen diffVel'
    NumericalParameters['monitoring_file'] = f'{names.DIRECTORY_LOG}/{Family}_',
    NumericalParameters['monitoring_period'] = kwargs.get('monitoring_period',  20)                       # recommended value
    NumericalParameters['monitoring_flag'] = _compute_giles_monitoring_flag(workflow.tree)                # automatically computed

    # keys relative to the inlet BC
    NumericalParameters['nscbc_in_type'] = kwargs.get('nscbc_in_type','htpt')                             # 'htpt', 'htpt_reldir', 'htpt_tangcomp' 
    # - numerics -
    NumericalParameters['nscbc_relaxi1'] = kwargs.get('nscbc_relaxi1',  500.)                             # recommended value
    NumericalParameters['nscbc_relaxi2'] = kwargs.get('nscbc_relaxi2',  500.)                             # recommended value
    giles_relax_in = kwargs.get('giles_relax_in',  [200.,  500.,  1000.,  1000.])                        # recommended value
    NumericalParameters['giles_relax_in1'] = giles_relax_in[0]
    NumericalParameters['giles_relax_in2'] = giles_relax_in[1]
    NumericalParameters['giles_relax_in3'] = giles_relax_in[2]
    NumericalParameters['giles_relax_in4'] = giles_relax_in[3]   

    # Imposed variables at boundary conditions
    if 'File' not in kwargs:
        ImposedVariables, _ = inj1_interface(workflow, **kwargs)
        ImposedVariables['vtx'] = ImposedVariables.pop('VelocityUnitVectorX')
        ImposedVariables.pop('VelocityUnitVectorY')
        ImposedVariables.pop('VelocityUnitVectorZ')
        ImposedVariables['vtr'] = kwargs.get('VelocityUnitVectorR', 0.)
        ImposedVariables['vtt'] = kwargs.get('VelocityUnitVectorTheta', 0.)

        ImposedVariables = solver_elsa.translate_to_elsa(ImposedVariables)
        NumericalParameters.update(ImposedVariables)
    
    ImposedVariables = dict()

    return ImposedVariables, NumericalParameters
        
def giles_outlet_interface(workflow, Family, 
                          NumberOfModes,
                          **kwargs
                          ):  

    # creation of dictionnary of keys for Giles outlet BC  
    NumericalParameters = dict()

    # keys relative to NSCBC
    NumericalParameters['type'] = 'nscbc_out'                                                                   # mandatory key to have NSCBC-Giles treatment
    NumericalParameters['nscbc_giles'] = 'statio'                                                               # mandatory key to have NSCBC-Giles treatment
    NumericalParameters['nscbc_interpbc'] = 'linear'                                                            # mandatory value
    NumericalParameters['nscbc_fluxt'] = kwargs.get('nscbc_fluxt', 'fluxBothTransv')                            # recommended value - possible keys : 'classic'; 'fluxInviscidTransv'; 'fluxBothTransv'
    NumericalParameters['nscbc_surf'] = kwargs.get('nscbc_surf',  'revolution')                                 # recommended value - possible keys : 'flat', 'revolution'
    NumericalParameters['nscbc_outwave'] = kwargs.get('nscbc_outwave',  'grad_etat')                            # recommended value - possible keys : 'grad_etat'; 'extrap_flux'
    NumericalParameters['nscbc_velocity_scale'] = kwargs.get('nscbc_velocity_scale', workflow.Flow['SoundSpeed']) # default value - reference sound velocity 
    NumericalParameters['nscbc_viscwall_len'] = kwargs.get('nscbc_viscwall_len', 5.e-4)                         # default value, could be updated by the user if convergence issue

    if kwargs.get('nscbc_viscwall_len_hub') is not None:
        NumericalParameters['nscbc_viscwall_len_hub'] = kwargs.get('nscbc_viscwall_len_hub')                    # value of nscbc_viscwall_len for the hub only
    if kwargs.get('nscbc_viscwall_len_carter') is not None:
        NumericalParameters['nscbc_viscwall_len_carter'] = kwargs.get('nscbc_viscwall_len_carter')              # value of nscbc_viscwall_len for the hub only           


    # keys relative to the Giles treatment 
    NumericalParameters['giles_opt'] = 'relax'                                                        # mandatory key for NSCBC-Giles treatment
    NumericalParameters['giles_restric_relax'] = 'inactive'                                           # mandatory key for NSCBC-Giles treatment
    NumericalParameters['giles_exact_lodi'] = kwargs.get('giles_exact_lodi',  'active')               # recommended value - possible keys: 'inactive', 'partial', 'active'
    NumericalParameters['giles_nbMode'] = NumberOfModes                                         # given by the user - recommended value : ncells_theta/2 + 1 (odd_value)

    # keys relative to the monitoring and radii calculus - monitoring data stored in LOGS
    NumericalParameters['bnd_monitoring'] = 'active'                                                  # recommended value
    NumericalParameters['monitoring_comp_rad'] = 'auto'                                               # recommended value - possible keys: 'from_file', 'monofenetre'
    NumericalParameters['monitoring_tol_rad'] = kwargs.get('monitoring_tol_rad',  1e-6)               # recommended value
    NumericalParameters['monitoring_var'] = 'psta'
    NumericalParameters['monitoring_file'] = f'{names.DIRECTORY_LOG}/{Family}_',
    NumericalParameters['monitoring_period'] = kwargs.get('monitoring_period',  20)                   # recommended value   
    NumericalParameters['monitoring_flag'] = _compute_giles_monitoring_flag(workflow.tree)                            # automatically computed

    # keys relative to the outlet NSCBC/Giles
    NumericalParameters['nscbc_relaxo'] = kwargs.get('nscbc_relaxo',  200.)                           # recommended value
    NumericalParameters['giles_relaxo'] = kwargs.get('giles_relaxo',  200.)                           # recommended value   

    if 'File' not in kwargs:
        # Parameters related to radial equilibrium
        boundary_conditions.OutflowRadialEquilibrium_interface(workflow, kwargs)  # modify kwargs
        outradeq_params = outradeq_interface(workflow, Family, **kwargs)

        NumericalParameters.update(dict(
            monitoring_pressure = outradeq_params['valve_ref_pres'],
            monitoring_indpiv = outradeq_params['indpiv'],
        ))
        if 'valve_type' in outradeq_params:
            NumericalParameters.update(dict(
                valve_ref_type = outradeq_params['valve_type'],
                monitoring_valve_ref_mflow = outradeq_params['valve_ref_mflow'],
                valve_relax = outradeq_params['valve_relax'],
            ))

    ImposedVariables = dict()
    return ImposedVariables, NumericalParameters

def giles_stage_mxpl(workflow, Family, LinkedFamily,
                    NumberOfModes,
                    **kwargs
                    ):  
    
    flag = _compute_giles_monitoring_flag(workflow.tree)

    leftFamily = workflow.tree.get(Name=Family, Type='Family', Depth=2)
    rightFamily = workflow.tree.get(Name=LinkedFamily, Type='Family', Depth=2)
    cgns.Node(Name='FamilyBC', Type='FamilyBC', Value='BCOutflow', Parent=leftFamily)
    cgns.Node(Name='FamilyBC', Type='FamilyBC', Value='BCInflow', Parent=rightFamily)

    # creation of dictionnary of keys for Giles mxpl left 
    DictKeysGilesMxpl = {}

    # keys relative to NSCBC
    DictKeysGilesMxpl['type'] = 'nscbc_mxpl'                                                        # mandatory key to have NSCBC-Giles treatment
    DictKeysGilesMxpl['nscbc_giles'] = 'statio'                                                     # mandatory key to have NSCBC-Giles treatment
    #DictKeysGilesMxpl['nscbc_interpbc'] = 'linear'                                                 # necessary for Mxpl?
    DictKeysGilesMxpl['nscbc_fluxt'] = kwargs.get('nscbc_fluxt', 'fluxInviscidTransv')              # recommended value - possible keys : 'classic'; 'fluxInviscidTransv'; 'fluxBothTransv'
    DictKeysGilesMxpl['nscbc_surf'] = kwargs.get('nscbc_surf',  'revolution')                       # recommended value - possible keys : 'flat', 'revolution'
    DictKeysGilesMxpl['nscbc_outwave'] = kwargs.get('nscbc_outwave',  'grad_etat')                  # recommended value - possible keys : 'grad_etat'; 'extrap_flux'
    DictKeysGilesMxpl['nscbc_velocity_scale'] = kwargs.get('nscbc_velocity_scale', workflow.Flow['SoundSpeed'])    # default value - reference sound velocity 
    DictKeysGilesMxpl['nscbc_viscwall_len'] = kwargs.get('nscbc_viscwall_len', 5.e-4)               # default value, could be updated by the user if convergence issue

    if kwargs.get('nscbc_viscwall_len_hub') is not None:
        DictKeysGilesMxpl['nscbc_viscwall_len_hub'] = kwargs.get('nscbc_viscwall_len_hub')                    # value of nscbc_viscwall_len for the hub only
    if kwargs.get('nscbc_viscwall_len_carter') is not None:
        DictKeysGilesMxpl['nscbc_viscwall_len_carter'] = kwargs.get('nscbc_viscwall_len_carter')              # value of nscbc_viscwall_len for the hub only           

    # keys relative to the Giles treatment 
    DictKeysGilesMxpl['giles_opt'] = 'relax'                                                        # mandatory key for NSCBC-Giles treatment
    DictKeysGilesMxpl['giles_restric_relax'] = 'inactive'                                           # mandatory key for NSCBC-Giles treatment
    DictKeysGilesMxpl['giles_exact_lodi'] = kwargs.get('giles_exact_lodi',  'partial')               # recommended value - possible keys: 'inactive', 'partial', 'active'
    DictKeysGilesMxpl['giles_nbMode'] = NumberOfModes                         # given by the user - recommended value : ncells_theta/2 + 1 (odd_value)

    # keys relative to the mxpl NSCBC/Giles
    method = kwargs.get('method', 'Robust')
    if method == 'Robust':
        DictKeysGilesMxpl['nscbc_mxpl_type'] = kwargs.get('nscbc_mxpl_type',  'pshtpt')                 
        DictKeysGilesMxpl['nscbc_mxpl_avermean'] = kwargs.get('nscbc_mxpl_avermean',  'pshtpt')         
    elif method == 'Conservative':
        DictKeysGilesMxpl['nscbc_mxpl_type'] = kwargs.get('nscbc_mxpl_type',  'flux')                 
        DictKeysGilesMxpl['nscbc_mxpl_avermean'] = kwargs.get('nscbc_mxpl_avermean',  'flux')         
    DictKeysGilesMxpl['nscbc_mxpl_flag'] = flag   # index gathering left and right BCs for one given Mxpl interface. automatically computed, different for each pair of Mxpl planes
    DictKeysGilesMxpl['nscbc_relaxi1'] = kwargs.get('nscbc_relaxi1',  20.)                          # recommended value
    DictKeysGilesMxpl['nscbc_relaxi2'] = kwargs.get('nscbc_relaxi2',  20.)                          # recommended value
    DictKeysGilesMxpl['nscbc_relaxo'] = kwargs.get('nscbc_relaxo',  20.)                            # recommended value
    DictKeysGilesMxpl['giles_relax_in1'] = kwargs.get('giles_relax_in1',  50.)                      # recommended value
    DictKeysGilesMxpl['giles_relax_in2'] = kwargs.get('giles_relax_in2',  50.)                      # recommended value
    DictKeysGilesMxpl['giles_relax_in3'] = kwargs.get('giles_relax_in3',  50.)                      # recommended value
    DictKeysGilesMxpl['giles_relax_in4'] = kwargs.get('giles_relax_in4',  50.)                      # recommended value
    DictKeysGilesMxpl['giles_relax_out'] = kwargs.get('giles_relax_out',  50.)                      # recommended value 

    # keys relative to the monitoring and radii calculus - monitoring data stored in LOGS
    DictKeysGilesMxpl['bnd_monitoring'] = 'active'                                                  # recommended value
    DictKeysGilesMxpl['monitoring_comp_rad'] = 'auto'                                               # recommended value - possible keys: 'from_file', 'monofenetre'
    DictKeysGilesMxpl['monitoring_tol_rad'] = kwargs.get('monitoring_tol_rad',  1e-6)               # recommended value - decrease value if the mesh is coarse
    DictKeysGilesMxpl['monitoring_var'] = 'psta  pgen Tgen ux uy uz diffPgen diffTgen diffVel'
    DictKeysGilesMxpl['monitoring_period'] = kwargs.get('monitoring_period',  20)                   # recommended value   

    # define parameter for left and right interface
    
    LogRootName = f'Mxpl_{flag}_{flag+1}' # give a common LogRootName for the Mxpl interface (upstream and downstream)
    DictKeysGilesMxpl_left = DictKeysGilesMxpl.copy()
    DictKeysGilesMxpl_left['monitoring_flag'] = flag  # index gathering all BCs "left" for one given Mxpl interface. automatically computed, must be different from other Giles BC, including right BC of Mxpl
    DictKeysGilesMxpl_left['monitoring_file'] = f'{names.DIRECTORY_LOG}/{LogRootName}_{flag}'
    DictKeysGilesMxpl_right = DictKeysGilesMxpl.copy()
    DictKeysGilesMxpl_right['monitoring_flag'] = flag+1  # index gathering all BCs "right" for one given Mxpl interface. automatically computed, must be different from other Giles BC, including left BC of Mxpl
    DictKeysGilesMxpl_right['monitoring_file'] = f'{names.DIRECTORY_LOG}/{LogRootName}_{flag+1}'

    # set the BCs left with keys
    ListBCNodes_left = boundary_conditions.get_bc_nodes_from_family(workflow.tree, Family)
    for BCNode_left in ListBCNodes_left:
        BCNode_left.setParameters('.Solver#BC', **DictKeysGilesMxpl_left)

    # set the BCs right with keys
    ListBCNodes_right = boundary_conditions.get_bc_nodes_from_family(workflow.tree, LinkedFamily)
    for BCNode_right in ListBCNodes_right:
        BCNode_right.setParameters('.Solver#BC', **DictKeysGilesMxpl_right)

def _fix_point_range_in_gc(t):
    # The algorithm to build a structured globborder
    # does not work when a point range is "reversed", 
    # although it fits the CGNS standard (for this reason, it could be reversed by maia)
    # The function _restore_point_range_in_gc will reverse the ranges back to their original values
    # TODO Create an issue for etc
    for gc in t.group(Type='GridConnectivity1to1'):
        for ptr_node in gc.group(Type='IndexRange'):
            ptr = ptr_node.value()
            has_been_swapped = False
            old_ptr_node = ptr_node.copy(deep=True)
            for i, interval in enumerate(ptr):
                if interval[0] > interval[1]:
                    ptr[i][0], ptr[i][1] = ptr[i][1], ptr[i][0]
                    has_been_swapped = True
            if has_been_swapped:
                mola_logger.debug(f'swap node {gc.path()}')
                old_ptr_node.setName(f'{old_ptr_node.name()}_before_swap')
                old_ptr_node.setType('UserDefinedData')
                old_ptr_node.attachTo(gc)

def _restore_point_range_in_gc(t):
    # return to the previous state before function _fix_point_range_in_gc
    for gc in t.group(Type='GridConnectivity1to1'):
        for node in gc.group(Name='*_before_swap'):
            node.setName(node.name().replace('_before_swap', ''))
            swapped_node = gc.get(Name=node.name())
            assert swapped_node is not None
            node.setType(swapped_node.type())
            swapped_node.dettach()
            node.attachTo(gc)
