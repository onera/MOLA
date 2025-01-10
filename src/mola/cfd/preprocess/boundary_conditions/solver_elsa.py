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
from mola.logging import mola_logger, MolaException, MolaUserError, mute_stdout
from mola.cfd.preprocess.solver_specific_tools import solver_elsa
from mola.cfd.preprocess.motion import motion
from mola.cfd.preprocess.motion.solver_elsa import assert_rotation_axis_is_correct, translate_motion_to_elsa
from mola.cfd.preprocess.boundary_conditions import boundary_conditions
from mola.cfd.preprocess.mesh.families import get_zone_family_from_bc_or_gc_family
import mola.server as SV

def define_bc_family(workflow, Family, Value):
    familyNode = workflow.tree.get(Name=Family, Type='Family', Depth=2)
    familyNode.findAndRemoveNode(Name='.Solver#BC', Depth=1)
    familyNode.findAndRemoveNodes(Type='FamilyBC', Depth=1)
    cgns.Node( Name='FamilyBC', Value=Value, Type='FamilyBC', Parent=familyNode )
    return familyNode

def impose_bc_fields(workflow, bc_path, ImposedVariables, GridLocation='FaceCenter'):
    bc_node = workflow.tree.getAtPath(bc_path)
    BCDataSet = cgns.Node( Name='BCDataSet#Init', Value='Null', Type='BCDataSet', Parent=bc_node )
    cgns.Node(Name='GridLocation', Type='GridLocation', Value=GridLocation, Parent=BCDataSet)
    BCDataSet.setParameters('NeumannData', ContainerType='BCData', **ImposedVariables)

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
    wall = define_bc_family(workflow, Family, bctype_cgns)

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
        non_uniform_fields = boundary_conditions.apply_function_to_BCDataSet(workflow, Family, Motion)
        for bc_path, ImposedVariables in non_uniform_fields.items():
            assert list(ImposedVariables) == ['RotationSpeed'], f'list(ImposedVariables)={list(ImposedVariables)}'
            impose_bc_fields(workflow, bc_path, dict(omega = ImposedVariables['RotationSpeed']))

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
    define_bc_family(workflow, Family, 'BCSymmetryPlane')


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
        define_bc_family(workflow, Family, 'BCFarfield')
    else:
        variables_from_file = ['Density', 'MomentumX', 'MomentumY', 'MomentumZ', 'EnergyStagnationDensity']
        variables_from_file += list(workflow.Turbulence['Conservatives'])

        set_physical_boundary(workflow, Family, 
                            FamilyBC='BCFarfield', BCType='nref', interface_function=nref_interface,
                            variables_from_file=variables_from_file,
                            **kwargs
                            )

def inj1(workflow, Family, **kwargs):
    set_physical_boundary(workflow, Family, 
                          FamilyBC='BCInflowSubsonic', BCType='inj1', interface_function=inj1_interface,
                          **kwargs
                          )

def injmfr1(workflow, Family, **kwargs):
    set_physical_boundary(workflow, Family, 
                          FamilyBC='BCInflowSubsonic', BCType='injmfr1', interface_function=injmfr1_interface,
                          **kwargs
                          )

def outpres(workflow, Family, **kwargs):   
    set_physical_boundary(workflow, Family, 
                          FamilyBC='BCOutflowSubsonic', BCType='outpres', interface_function=outpres_interface,
                          **kwargs
                          )

def outsup(workflow, Family):
    define_bc_family(workflow, Family, 'BCOutflowSupersonic')

def outmfr2(workflow, Family, **kwargs):
    set_physical_boundary(workflow, Family, 
                          FamilyBC='BCOutflowSubsonic', BCType='outmfr2', interface_function=outmfr2_interface,
                          **kwargs
                          )
  


def nref_interface(workflow, **kwargs):
    conservatives = workflow.Flow['Conservatives'] + workflow.Turbulence['Conservatives']
    for key, value in kwargs.items():
        if key in conservatives:
            conservatives[key] = value
    return conservatives

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
    return ImposedVariables
       
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
    return ImposedVariables

def outpres_interface(workflow, **kwargs):
    ImposedVariables = dict(
        Pressure = kwargs.get('Pressure', workflow.Flow['Pressure'])
        )
    return ImposedVariables

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
        groupmassflow = groupmassflow,
        )
    return ImposedVariables

def outradeq_interface(workflow, Family, **kwargs):

    def _get_default_valve_ref_mflow():
        bcs = boundary_conditions.get_bc_nodes_from_family(workflow.tree, Family)
        bc = bcs[0]
        zone = bc.getParent(Type='Zone_t')
        row = zone.get(Type='FamilyName').value()
        try:
            rowParams = workflow.ApplicationContext['Rows'][row]
        except:
            raise MolaException('Worklow must have an attribute ApplicationContext with a dict named "Rows" inside.')
        fluxcoeff = rowParams['NumberOfBlades'] / float(rowParams['NumberOfBladesSimulated'])
        try:
            valve_ref_mflow = workflow.Flow['MassFlow'] / fluxcoeff
        except:
            raise MolaException('Miss MassFlow in Flow attribute')
        
        return valve_ref_mflow

    valve_type = kwargs.get('valve_type', 0)

    valve_ref_pres = kwargs.get('valve_ref_pres')
    if not valve_ref_pres:
        valve_ref_pres = kwargs.get('Pressure', workflow.Flow['Pressure'])
    
    if valve_type == 0:
        valve_ref_mflow = None
    else:
        valve_ref_mflow = kwargs.get('valve_ref_mflow')
        if not valve_ref_mflow:
            valve_ref_mflow = kwargs.get('MassFlow', _get_default_valve_ref_mflow())

    parameters = dict(
        valve_type = valve_type, 
        valve_ref_pres = valve_ref_pres,
        valve_ref_mflow = valve_ref_mflow, 
        valve_relax = kwargs.get('valve_relax', 0.1),
        indpiv = kwargs.get('indpiv', 1),
        dirorder = kwargs.get('dirorder', -1),
        )
    return parameters

def outradeqhyb_interface(workflow, Family, **kwargs):
    parameters = outradeq_interface(workflow, Family, **kwargs)
    parameters['nbband'] = kwargs.get('nbband', -1) # default value in etc, compute nbband based on mesh
    parameters['c'] = kwargs.get('c', 0.3) # default value in etc is 0.1
    return parameters

def set_physical_boundary(workflow, Family, 
                          FamilyBC, BCType, interface_function,
                          File=None, variableForInterpolation='ChannelHeight', 
                          **kwargs 
                          ):
    
    kwargs['Family'] = Family
    ImposedVariables = interface_function(workflow, **kwargs)

    if File is not None:

        input_data_from_file = boundary_conditions.get_fields_from_file(
            workflow.tree, Family, File, var2interp=list(ImposedVariables)
            )
        for bc, ImposedVariables in input_data_from_file.items():  
            setBCwithImposedVariables(
                workflow, 
                Family, 
                ImposedVariables,
                FamilyBC=FamilyBC, 
                BCType=BCType, 
                bc=bc,
                variableForInterpolation=variableForInterpolation
                )
    elif not all([np.ndim(v) == 0 and not callable(v) for v in ImposedVariables.values()]):
        for bc, ImposedVariables in boundary_conditions.get_bc_nodes_from_family(workflow.tree, Family):
            setBCwithImposedVariables(
                workflow, 
                Family, 
                ImposedVariables,
                FamilyBC=FamilyBC, 
                BCType=BCType, 
                bc=bc,
                variableForInterpolation=variableForInterpolation
                )
    else:
        setBCwithImposedVariables(
            workflow, 
            Family, 
            ImposedVariables,
            FamilyBC=FamilyBC, 
            BCType=BCType, 
            variableForInterpolation=variableForInterpolation
            )
        
def setBCwithImposedVariables(workflow, Family, ImposedVariables, FamilyBC, BCType,
    bc=None, BCDataSetName='BCDataSet#Init', BCDataName='DirichletData', variableForInterpolation='ChannelHeight'):
    '''
    Generic function to impose a Boundary Condition ``inj1``. The following
    functions are more specific:

    Parameters
    ----------

        workflow.tree : PyTree
            Tree to modify

        Family : str
            Name of the family on which the boundary condition will be imposed

        ImposedVariables : str
            When using a function to impose the radial profile of one or several quantities, 
            it defines the variable used as the argument of this function.
            Must be 'ChannelHeight' (default value) or 'Radius'.riable names and values must be either:

                * scalars: in that case they are imposed once for the
                  family **Family** in the corresponding ``Family_t`` node.

                * numpy arrays: in that case they are imposed for the ``BC_t``
                  node **bc**.

                * functions: in that case the function defined a profile depending on radius.
                  It is evaluated in each cell on the **bc**.
            
            They may be a combination of three.

        bc : PyTree
            ``BC_t`` node on which the boundary condition will be imposed. Must
            be :py:obj:`None` if the condition must be imposed once in the
            ``Family_t`` node.

        BCDataSetName : str
            Name of the created node of type ``BCDataSet_t``. Default value is
            'BCDataSet#Init'

        BCDataName : str
            Name of the created node of type ``BCData_t``. Default value is
            'DirichletData'
        
        variableForInterpolation : str
            When using a function to impose the radial profile of one or several quantities, 
            it defines the variable used as the argument of this function.
            Must be 'ChannelHeight' (default value), 'Radius', 'CoordinateX', 'CoordinateY' or 'CoordinateZ'.

    See also
    --------

    setBC_inj1, setBC_outpres, setBC_outmfr2

    '''
    FamilyNode = define_bc_family(workflow, Family, FamilyBC)

    if all([np.ndim(v)==0 and not callable(v) for v in ImposedVariables.values()]):
        checkVariables(ImposedVariables)
        ImposedVariables = solver_elsa.translate_to_elsa(ImposedVariables)
        FamilyNode.setParameters('.Solver#BC', type=BCType, **ImposedVariables)

    else:
        raise Exception('Not implemented yet')
        assert bc is not None
        J.set(bc, '.Solver#BC', type=BCType)

        zone = I.getParentFromType(workflow.tree, bc, 'Zone_t') 
        if variableForInterpolation in ['Radius', 'radius']:
            radius, theta = J.getRadiusTheta(zone)
        elif variableForInterpolation == 'ChannelHeight':
            radius = I.getValue(I.getNodeFromName(zone, 'ChannelHeight'))
        elif variableForInterpolation.startsWith('Coordinate'):
            radius = I.getValue(I.getNodeFromName(zone, variableForInterpolation))
        else:
            raise ValueError('varForInterpolation must be ChannelHeight, Radius, CoordinateX, CoordinateY or CoordinateZ')

        PointRangeNode = I.getNodeFromType(bc, 'IndexRange_t')
        if PointRangeNode:
            # Structured mesh
            PointRange = I.getValue(PointRangeNode)
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
                raise ValueError('Wrong BC shape {} in {}'.format(bc_shape, I.getPath(workflow.tree, bc)))
        
        else: 
            # Unstructured mesh
            PointList = I.getValue(I.getNodeFromType(bc, 'IndexArray_t'))
            bc_shape = PointList.size
            radius = radius[PointList-1]

        for var, value in ImposedVariables.items():
            if callable(value):
                ImposedVariables[var] = value(radius) 
            elif np.ndim(value)==0:
                # scalar value --> uniform data
                ImposedVariables[var] = value * np.ones(radius.shape)
            assert ImposedVariables[var].shape == bc_shape, \
                'Wrong shape for variable {}: {} (shape {} for {})'.format(
                    var, ImposedVariables[var].shape, bc_shape, I.getPath(workflow.tree, bc))
        
        checkVariables(ImposedVariables)

        BCDataSet = I.newBCDataSet(name=BCDataSetName, value='Null',
            gridLocation='FaceCenter', parent=bc)
        J.set(BCDataSet, BCDataName, childType='BCData_t', **ImposedVariables)


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

def getFamilyBCTypeFromFamilyBCName(t, FamilyBCName):
    '''
    Get the *BCType* of BCs defined by a given family BC name.

    Parameters
    ----------

        t : PyTree
            main CGNS tree

        FamilyBCName : str
            requested name of the *FamilyBC*

    Returns
    -------

        BCType : str
            the resulting *BCType*. Returns:py:obj:`None` if **FamilyBCName** is not
            found
    '''
    FamilyNode = I.getNodeFromNameAndType(t, FamilyBCName, 'Family_t')
    if not FamilyNode: return

    FamilyBCNode = I.getNodeFromName1(FamilyNode, 'FamilyBC')
    if not FamilyBCNode: return

    FamilyBCNodeType = I.getValue(FamilyBCNode)
    if FamilyBCNodeType != 'UserDefined': return FamilyBCNodeType

    SolverBC = I.getNodeFromName1(FamilyNode,'.Solver#BC')
    if SolverBC:
        SolverBCType = I.getNodeFromName1(SolverBC,'type')
        if SolverBCType:
            BCType = I.getValue(SolverBCType)
            return BCType

    SolverOverlap = I.getNodeFromName1(FamilyNode,'.Solver#Overlap')
    if SolverOverlap: return 'BCOverlap'

    BCnodes = I.getNodesFromType(t, 'BC_t')
    for BCnode in BCnodes:
        FamilyNameNode = I.getNodeFromName1(BCnode, 'FamilyName')
        if not FamilyNameNode: continue

        FamilyNameValue = I.getValue( FamilyNameNode )
        if FamilyNameValue == FamilyBCName:
            BCType = I.getValue( BCnode )
            if BCType != 'FamilySpecified': return BCType
            break

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

        ReferenceValues : :py:class:`dict` or :py:obj:`None`
            as produced by :py:func:`computeReferenceValues`

        TurboConfiguration : :py:class:`dict` or :py:obj:`None`
            as produced by :py:func:`getTurboConfiguration`

        method : optional, str
            Method used to compute the globborder. The default value is
            ``'globborder_dict'``, it corresponds to the ETC topological
            algorithm.
            Another possible value is ``'poswin'`` to use the geometrical
            algorithm in *turbo* (in this case, *turbo* environment must be
            sourced).

    '''
    if not workflow.tree.isStructured():
        raise MolaUserError(f'The boundary condition "outradeq" on Family {Family} is available only for structured mesh.')

    import etc.transform as trf
    t = workflow.tree

    params = outradeq_interface(workflow, Family, **kwargs)

    # Delete previous BC if it exists
    for bc in C.getFamilyBCs(t, Family):
        I._rmNodesByName(bc, '.Solver#BC')
    define_bc_family(workflow, Family, 'BCOutflowSubsonic')

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
    define_bc_family(workflow, Family, 'BCOutflowSubsonic')

    bc = trf.BCOutRadEqHyb(t, t.get(Name=Family, Type='Family'))
    bc.glob_border()
    bc.indpiv = params['indpiv']
    if params['valve_type'] == 0:
        bc.prespiv = params['valve_ref_pres']
    else:
        valve_law_dict = {1: 'SlopePsQ', 2: 'QTarget', 3: 'QLinear', 4: 'QHyperbolic'}
        bc.valve_law(valve_law_dict[params['valve_type']], params['valve_ref_pres'],
                    params['valve_ref_mflow'], valve_relax=params['valve_relax'], 
                    valve_file=f'prespiv_{Family}.log')
    bc.dirorder = params['dirorder']
    radius_filename = f'radius_{Family}.plt'
    radius = bc.repartition(filename=radius_filename, fileformat="bin_tp")
    radius.compute(t, nbband=params['nbband'], c=params['c'])
    radius.write()
    bc.create()
    workflow.tree = cgns.castNode(t)

    # Move radius files to the RunDirectory
    # HACK This will be outdated as soon as the radius distribution is written directly in the CGNS file
    # see https://elsa-e.onera.fr/issues/10541
    if Path(workflow.RunManagement['RunDirectory']).resolve() != Path.cwd():
        SV.copy_remote(
            source_path=radius_filename, 
            destination_path=Path(workflow.RunManagement['RunDirectory']) / Path(radius_filename), 
            destination_machine=workflow.RunManagement['Machine'],
            )
        SV.remove_path(radius_filename, machine='localhost')

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

    workflow.tree = trf.defineBCStageFromBC(workflow.tree, (Family, LinkedFamily))
    workflow.tree, stage = trf.newStageMxPlFromFamily(workflow.tree, Family, LinkedFamily)

    stage.jtype = 'nomatch_rad_line'
    stage.create()

    workflow.tree = cgns.castNode(workflow.tree)
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

    SectorPassagePeriod = stage_red_interface(workflow, Family, LinkedFamily, SectorPassagePeriod)

    # HACK: must change the type of all FamilyName to array
    # For a unknown reason, nodes FamilyName have value of type str instead of ndarray,
    # and that makes a bug in trf.defineBCStageFromBC (in CGU.getValueAsString(FamilyName))
    for FamilyName_node in workflow.tree.group(Type='FamilyName'):
        FamilyName_node.setValue(FamilyName_node.value())

    workflow.tree = trf.defineBCStageFromBC(workflow.tree, (Family, LinkedFamily))
    workflow.tree, stage = trf.newStageRedFromFamily(workflow.tree, Family, LinkedFamily, stage_ref_time=SectorPassagePeriod)

    stage.create()

    workflow.tree = cgns.castNode(workflow.tree)
    set_turbomachinery_interface_FamilyBC(workflow.tree, Family, LinkedFamily)
    # GC names must be unique to use globborders in elsa, otherwise the error "Error : duplicated object name!" will be raised
    I._correctPyTree(workflow.tree, level=4)

@mute_stdout
def stage_mxpl_hyb(workflow, Family, LinkedFamily, nbband=100, c=0.3):
    '''
    Set a hybrid mixing plane condition between families **Family** and **LinkedFamily**.

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
            SV.copy_remote(
                source_path=filename, 
                destination_path=Path(workflow.RunManagement['RunDirectory']) / Path(filename), 
                destination_machine=workflow.RunManagement['Machine'],
                )
            SV.remove_path(filename, machine='localhost')

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

    SectorPassagePeriod = stage_red_interface(workflow, Family, LinkedFamily, SectorPassagePeriod)

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

def stage_red_interface(workflow, Family, LinkedFamily, SectorPassagePeriod):
    '''
    see https://elsa-doc.onera.fr/restricted/MU_MT_tuto/latest/Tutos/Speciality/StageRed.html#numerical-parameters
    '''    
    if not SectorPassagePeriod:

        row1 = get_zone_family_from_bc_or_gc_family(workflow.tree, Family)
        row2 = get_zone_family_from_bc_or_gc_family(workflow.tree, LinkedFamily)

        LapPeriod = 2*np.pi / abs(workflow.ApplicationContext['ShaftRotationSpeed'])

        N1 = workflow.ApplicationContext['Rows'][row1]['NumberOfBlades']
        N2 = workflow.ApplicationContext['Rows'][row2]['NumberOfBlades']
        K1 = workflow.ApplicationContext['Rows'][row1]['NumberOfBladesSimulated']
        K2 = workflow.ApplicationContext['Rows'][row2]['NumberOfBladesSimulated']

        Dm = 2 / (K1/N1 + K2/N2)
        SectorPassagePeriod = LapPeriod / Dm

        msg = f'The reference time period for RNA interface is equal to {Dm}EO.'
        if np.isclose(Dm, 1) or np.isclose(Dm, K1/N1):
            mola_logger.info(msg)
        else:
            mola_logger.warning(msg)

    return SectorPassagePeriod

def chorochronic(workflow, Family, LinkedFamily, NumberOfHarmonicsForFamily=20., NumberOfHarmonicsForLinkedFamily=20., hybrid=True):
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

        NumberOfHarmonicsForFamily : float
            Number of harmonics of the first row.

        NumberOfHarmonicsForLinkedFamily : float
            Number of harmonics of the second row.
        
        hybrid : bool
            If True, use the `stage_choro_hyb` condition, else use `stage_choro`.
    '''   
    if hybrid:
        stage_choro_hyb(workflow, Family, LinkedFamily)
    else:
        stage_choro(workflow, Family, LinkedFamily)
    convert_periodic_to_chorochrono(workflow.tree)
    row1 = get_zone_family_from_bc_or_gc_family(workflow.tree, Family)
    row2 = get_zone_family_from_bc_or_gc_family(workflow.tree, LinkedFamily)
    choroParamsRow1, choroParamsRow2 = compute_choro_parameters(workflow.ApplicationContext, row1, row2, Nharm_Row1=NumberOfHarmonicsForFamily, Nharm_Row2=NumberOfHarmonicsForLinkedFamily)
    add_choro_data(workflow.tree, Family, **choroParamsRow1) 
    add_choro_data(workflow.tree, LinkedFamily, **choroParamsRow2) 

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
    set_turbomachinery_interface_FamilyBC(workflow.tree, Family, LinkedFamily)
    # GC names must be unique to use globborders in elsa, otherwise the error "Error : duplicated object name!" will be raised
    I._correctPyTree(workflow.tree, level=4)

@mute_stdout
def stage_choro_hyb(workflow, Family, LinkedFamily):
    '''
    Set a hybrid chorochronic interface condition between families **Family** and **LinkedFamily**.

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

    workflow.tree = trf.defineBCStageFromBC(workflow.tree, (Family, LinkedFamily))
    workflow.tree, stage = trf.newStageChoroHybFromFamily(workflow.tree, Family, LinkedFamily)

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
    for gc_node in t.group(Type='GridConnectivity*'):
        if gc_node.get(Type='Perdiodic'):
            gcnodes.append(gc_node)

    for gcnode in gcnodes:
        gc = trf.BCChoroChrono(t, gcnode, choro_file = 'None')
        gc.choro_file   = 'None'
        gc.file   = None
        gc.format = 'CGNS'
        gc.create()

def compute_choro_parameters(ApplicationContext, row1, row2, Nharm_Row1, Nharm_Row2, relax=1.0):
    '''
    Compute the parameters to run a chorochronic computation.
    '''       
    Nblade_Row1 = ApplicationContext['Rows'][row1]['NumberOfBlades']
    Nblade_Row2 = ApplicationContext['Rows'][row2]['NumberOfBlades']
    omega_Row1 = ApplicationContext['ShaftRotationSpeed'] if ApplicationContext['Rows'][row1]['IsRotating'] else 0.
    omega_Row2 = ApplicationContext['ShaftRotationSpeed'] if ApplicationContext['Rows'][row2]['IsRotating'] else 0.

    gcd = np.gcd(Nblade_Row1,Nblade_Row2)
    if Nharm_Row1 < Nblade_Row1/gcd:
        mola_logger.warning(f'The number of chorochronic harmonics for the first row is too low ({Nharm_Row1}). Recomputing...\n ')
        Nharm_Row1 = float(Nblade_Row2)

    if Nharm_Row2 < Nblade_Row2/gcd:
        mola_logger.warning(f'The number of chorochronic harmonics for the first row is too low ({Nharm_Row2}). Recomputing...\n ')
        Nharm_Row2 = float(Nblade_Row1)
        mola_logger.warning(f'New number of harmonics for row 2 : {Nharm_Row2}')

    mola_logger.info(f'Number of harmonics for {row1} : {Nharm_Row1}')
    mola_logger.info(f'Number of harmonics for {row2} : {Nharm_Row2}')

    choroParamsRow1 = dict(
        f_freq = Nblade_Row2*np.abs(omega_Row1-omega_Row2)/(2*np.pi), 
        f_omega = float(omega_Row1 - omega_Row2), 
        f_harm = float(Nharm_Row1), 
        f_relax = float(relax), 
        axis_ang_1 = Nblade_Row1, 
        axis_ang_2 = 1
        )
    choroParamsRow2 = dict(
        f_freq = Nblade_Row1*np.abs(omega_Row1-omega_Row2)/(2*np.pi), 
        f_omega = float(omega_Row1 - omega_Row2), 
        f_harm = float(Nharm_Row2), 
        f_relax = float(relax), 
        axis_ang_1 = Nblade_Row2, 
        axis_ang_2 = 1
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
    motion_node = fam_node.setParameters('.Solver#Motion', axis_ang_1=axis_ang_1, axis_ang_2=axis_ang_2)

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

