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

import numpy as np

import Converter.PyTree as C
import Converter.Internal as I

from treelab import cgns
from mola.logging import mola_logger, MolaException, mute_stdout
from mola.cfd.preprocess.solver_specific_tools import solver_elsa
from mola.cfd.preprocess.motion import motion
from mola.cfd.preprocess.motion.solver_elsa import assert_rotation_axis_is_correct, translate_motion_to_elsa
from mola.cfd.preprocess.boundary_conditions import boundary_conditions

def define_bc_family(workflow, Family, Value):
    familyNode = workflow.tree.get(Name=Family, Type='Family', Depth=2)
    familyNode.findAndRemoveNode(Name='.Solver#BC', Depth=1)
    familyNode.findAndRemoveNodes(Type='FamilyBC', Depth=1)
    cgns.Node( Name='FamilyBC', Value=Value, Type='FamilyBC', Parent=familyNode )
    return familyNode

def impose_bc_fields(workflow, bc_path, ImposedVariables):
    bc_node = workflow.tree.getAtPath(bc_path)
    BCDataSet = cgns.Node( Name='BCDataSet#Init', Value='Null', Type='BCDataSet', Parent=bc_node )
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

    if not motion.is_mobile(Motion):
        return

    if callable(Motion) or any([callable(v) for v in Motion.values()]):
        # Put global parameters in the family
        Motion_default = dict(RotationSpeed=workflow.ComponentAxis)
        motion.set_default_motion(Motion_default)
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

    Parameters
    ----------

        workflow.tree : PyTree
            Tree to modify

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

    '''
    wall(workflow, Family, Motion=Motion, bctype_cgns='BCWallInviscid', bctype_elsa='wallslip')

def walladia(workflow, Family, Motion=None):
    '''
    Set a viscous wall boundary condition.

    .. note:: see `elsA Tutorial about wall conditions <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/BCsTutorials/tutorial-BC.html#wall-conditions/>`_

    Parameters
    ----------

        workflow.tree : PyTree
            Tree to modify

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

    '''
    wall(workflow, Family, Motion=Motion, bctype_cgns='BCWallViscous', bctype_elsa='walladia')

def nref(workflow, Family):
    '''
    Set a nref boundary condition.

    Parameters
    ----------

        workflow.tree : PyTree
            Tree to modify

        Family : str
            Name of the family on which the boundary condition will be imposed

    '''
    define_bc_family(workflow, Family, 'BCFarfield')
 
def get_bcs(t, Family):
    bcs = []
    all_bcs = t.group(Type='BC')
    for bc in all_bcs:
        if bc.get('FamilyName') == Family:
            bcs.append(bc)
    return bc

def inj1(workflow, Family, ImposedVariables, bc=None, variableForInterpolation='ChannelHeight'):
    '''
    Generic function to impose a Boundary Condition ``inj1``. The following
    functions are more specific:

        * :py:func:`setBC_inj1_uniform`

        * :py:func:`setBC_inj1_interpFromFile`

    .. note::
        see `elsA Tutorial about inj1 condition <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/BCsTutorials/tutorial-BC.html#inj1/>`_

    Parameters
    ----------

        workflow.tree : PyTree
            Tree to modify

        Family : str
            Name of the family on which the boundary condition will be imposed

        ImposedVariables : dict
            Dictionary of variables to imposed on the boudary condition. Keys
            are variable names and values must be:

                * either scalars: in that case they are imposed once for the
                  family **FamilyName** in the corresponding ``Family_t`` node.

                * or numpy arrays: in that case they are imposed for the ``BC_t``
                  node **bc**.

        bc : PyTree
            ``BC_t`` node on which the boundary condition will be imposed. Must
            be :py:obj:`None` if the condition must be imposed once in the
            ``Family_t`` node.
        
        variableForInterpolation : str
            When using a function to impose the radial profile of one or several quantities, 
            it defines the variable used as the argument of this function.
            Must be 'ChannelHeight' (default value) or 'Radius'.

    See also
    --------

    setBC_inj1_uniform, setBC_inj1_interpFromFile
    '''
    if not bc and not all([np.ndim(v)==0 and not callable(v) for v in ImposedVariables.values()]):
        for bc in get_bcs(workflow.tree, Family):
            setBCwithImposedVariables(workflow, Family, ImposedVariables,
                FamilyBC='BCInflowSubsonic', BCType='inj1', bc=bc, variableForInterpolation=variableForInterpolation)
    else:
        setBCwithImposedVariables(workflow, Family, ImposedVariables,
            FamilyBC='BCInflowSubsonic', BCType='inj1', bc=bc, variableForInterpolation=variableForInterpolation)

def outpres(workflow, Family, Pressure, bc=None, variableForInterpolation='ChannelHeight'):
    '''
    Impose a Boundary Condition ``outpres``.

    .. note::
        see `elsA Tutorial about outpres condition <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/BCsTutorials/tutorial-BC.html#outpres/>`_

    Parameters
    ----------

        t : PyTree
            Tree to modify

        FamilyName : str
            Name of the family on which the boundary condition will be imposed

        Pressure : :py:class:`float` or :py:class:`numpy.ndarray` or :py:class:`dict`
            Value of pressure to impose on the boundary conditions. May be:

                * either a scalar: in that case it is imposed once for the
                  family **FamilyName** in the corresponding ``Family_t`` node.

                * or a numpy array: in that case it is imposed for the ``BC_t``
                  node **bc**.

            Alternatively, **Pressure** may be a :py:class:`dict` of the form:

            >>> Pressure = dict(Pressure=value)

            In that case, the same requirements that before stands for *value*.

        bc : PyTree
            ``BC_t`` node on which the boundary condition will be imposed. Must
            be :py:obj:`None` if the condition must be imposed once in the
            ``Family_t`` node.
        
        variableForInterpolation : str
            When using a function to impose the radial profile of one or several quantities, 
            it defines the variable used as the argument of this function.
            Must be 'ChannelHeight' (default value) or 'Radius'.

    '''
    ImposedVariables = dict(Pressure=Pressure)

    if not bc and not all([np.ndim(v) == 0 and not callable(v) for v in ImposedVariables.values()]):
        for bc in get_bcs(workflow.tree, Family):
            setBCwithImposedVariables(workflow, Family, ImposedVariables,
                                      FamilyBC='BCOutflowSubsonic', BCType='outpres', bc=bc, variableForInterpolation=variableForInterpolation)
    else:
        setBCwithImposedVariables(workflow, Family, ImposedVariables,
                                FamilyBC='BCOutflowSubsonic', BCType='outpres', bc=bc, variableForInterpolation=variableForInterpolation)


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
        if isinstance(value, np.ndarray): return np.all(value>0)
        else: return value>0

    def unitComponent(value):
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
def outradeq(workflow, FamilyName, valve_type=0, valve_ref_pres=None,
    valve_ref_mflow=None, valve_relax=0.1, indpiv=1):
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

    import etc.transform as trf
    t = workflow.tree

    if valve_ref_pres is None:
        try:
            valve_ref_pres = workflow.Flow['Pressure']
        except:
            raise MolaException('valve_ref_pres or ReferenceValues must be not None')
    if valve_type != 0 and valve_ref_mflow is None:
        try:
            bc = C.getFamilyBCs(t, FamilyName)[0]
            zone = I.getParentFromType(t, bc, 'Zone_t')
            row = I.getValue(I.getNodeFromType1(zone, 'FamilyName_t'))
            rowParams = workflow.ApplicationContext['Rows'][row]
            fluxcoeff = rowParams['NumberOfBlades'] / float(rowParams['NumberOfBladesSimulated'])
            valve_ref_mflow = workflow.Flow['MassFlow'] / fluxcoeff
        except:
            raise MolaException('Either valve_ref_mflow or both ReferenceValues and TurboConfiguration must be not None')

    # Delete previous BC if it exists
    for bc in C.getFamilyBCs(t, FamilyName):
        I._rmNodesByName(bc, '.Solver#BC')
    # Create Family BC
    family_node = I.getNodeFromNameAndType(t, FamilyName, 'Family_t')
    I._rmNodesByName(family_node, '.Solver#BC')
    I.newFamilyBC(value='BCOutflowSubsonic', parent=family_node)

    from etc.globborder.globborder_dict import globborder_dict
    gbd = globborder_dict(t, FamilyName, config="axial")

    for bcn in C.getFamilyBCs(t, FamilyName):
        bcpath = I.getPath(t, bcn)
        bc = trf.BCOutRadEq(t, bcn)
        bc.indpiv = indpiv
        bc.dirorder = -1
        # Valve laws:
        # <bc>.valve_law(valve_type, pref, Qref, valve_relax=relax, valve_file=None, valve_file_freq=1) # v4.2.01 pour valve_file*
        # valvelaws = [(1, 'SlopePsQ'),     # p(it+1) = p(it) + relax*( pref * (Q(it)/Qref) -p(it)) # relax = sans dim. # isoPs/Q
        #              (2, 'QTarget'),      # p(it+1) = p(it) + relax*pref * (Q(it)/Qref-1)         # relax = sans dim. # debit cible
        #              (3, 'QLinear'),      # p(it+1) = pref + relax*(Q(it)-Qref)                  # relax = Pascal    # lin en debit
        #              (4, 'QHyperbolic'),  # p(it+1) = pref + relax*(Q(it)/Qref)**2               # relax = Pascal    # comp. exp.
        #              (5, 'SlopePiQ')]     # p(it+1) = p(it) + relax*( pref * (Q(it)/Qref) -pi(it)) # relax = sans dim. # isoPi/Q
        # for law 5, pref = reference total pressure
        if valve_type == 0:
            bc.prespiv = valve_ref_pres
        else:
            valve_law_dict = {1: 'SlopePsQ', 2: 'QTarget',
                              3: 'QLinear', 4: 'QHyperbolic'}
            bc.valve_law(valve_law_dict[valve_type], valve_ref_pres,
                         valve_ref_mflow, valve_relax=valve_relax, valve_file=f'prespiv_{FamilyName}.log')
        globborder = bc.glob_border(current=FamilyName)
        globborder.i_poswin = gbd[bcpath]['i_poswin']
        globborder.j_poswin = gbd[bcpath]['j_poswin']
        globborder.glob_dir_i = gbd[bcpath]['glob_dir_i']
        globborder.glob_dir_j = gbd[bcpath]['glob_dir_j']
        globborder.azi_orientation = gbd[bcpath]['azi_orientation']
        globborder.h_orientation = gbd[bcpath]['h_orientation']
        bc.create()

    workflow.tree = cgns.castNode(t)

@mute_stdout
def stage_mxpl(workflow, left, right):
    '''
    Set a mixing plane condition between families **left** and **right**.

    .. important : This function has a dependency to the ETC module.

    Parameters
    ----------

        t : PyTree
            Tree to modify

        left : str
            Name of the family on the left side.

        right : str
            Name of the family on the right side.
    '''

    import etc.transform as trf

    # HACK: must change the type of all FamilyName to array
    def change_FamilyName_to_array():
        for bc in workflow.tree.group(Type='BC'):
            FamilyName = bc.get(Type='FamilyName')
            FamilyName.setValue(np.array(FamilyName.value()))
    def change_back_FamilyName_to_str():
        for FamilyName in workflow.tree.group(Type='FamilyName'):
            fam = FamilyName.value()
            if isinstance(fam, np.ndarray):
                FamilyName.setValue(FamilyName.value()[0])

    change_FamilyName_to_array()
    workflow.tree = trf.defineBCStageFromBC(workflow.tree, left)
    workflow.tree = trf.defineBCStageFromBC(workflow.tree, right)
    change_back_FamilyName_to_str()
    workflow.tree, stage = trf.newStageMxPlFromFamily(workflow.tree, left, right)

    stage.jtype = 'nomatch_rad_line'
    stage.create()

    workflow.tree = cgns.castNode(workflow.tree)

    set_turbomachinery_interface_FamilyBC(workflow.tree, left, right)


def set_turbomachinery_interface_FamilyBC(t, left, right):
    for gc in t.group(Type='GridConnectivity'):
        for FamilyBC in gc.group(Type='FamilyBC'):
            FamilyBC.remove()
    
    leftFamily = t.get(Name=left, Type='Family', Depth=2)
    rightFamily = t.get(Name=right, Type='Family', Depth=2)
    cgns.Node(Name='FamilyBC', Type='FamilyBC', Value='BCOutflow', Parent=leftFamily)
    cgns.Node(Name='FamilyBC', Type='FamilyBC', Value='BCInflow', Parent=rightFamily)

