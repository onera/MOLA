#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np

import Converter.PyTree as C
import Converter.Internal as I

import mola.cgns as c


def walladia(workflow, Family, Motion=None):
    '''
    Set a viscous wall boundary condition.

    .. note:: see `elsA Tutorial about wall conditions <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/BCsTutorials/tutorial-BC.html#wall-conditions/>`_

    Parameters
    ----------

        workflow : Workflow object

        Family : str
            Name of the family on which the boundary condition will be imposed

        Motion : dict
            Example:

            .. code-block:: python
                Motion = dict(
                    RotationSpeed = [1000., 0., 0.],
                    RotationAxisOrigin = [0., 0., 0.],
                    TranslationSpeed = [0., 0., 0.]
                    )

    '''
    wall = I.getNodeFromNameAndType(workflow.tree, Family, 'Family_t')
    I._rmNodesByName(wall, '.Solver#BC')
    I._rmNodesByType(wall, 'FamilyBC_t')
    I.newFamilyBC(value='BCWallViscous', parent=wall)
    c.castNode(wall)

    if Motion:
        # For elsA, the rotation must be around one axis only
        onlyOneRotationComponent = \
            (Motion['RotationSpeed'][0] == Motion['RotationSpeed'][1] == 0) \
        or (Motion['RotationSpeed'][0] == Motion['RotationSpeed'][2] == 0) \
        or (Motion['RotationSpeed'][1] == Motion['RotationSpeed'][2] == 0)
        
        assert onlyOneRotationComponent, 'For elsA, the rotation must be around one axis only'
        omega = sum(Motion['RotationSpeed'])

        if omega != 0. or any(Motion['TranslationSpeed']!=0.):
            wall.setParameters('.Solver#BC',
                                type='walladia',
                                data_frame='user',
                                omega=omega,
                                axis_pnt_x=Motion['RotationAxisOrigin'][0], 
                                axis_pnt_y=Motion['RotationAxisOrigin'][1], 
                                axis_pnt_z=Motion['RotationAxisOrigin'][2],
                                axis_vct_x=Motion['TranslationSpeed'][0], 
                                axis_vct_y=Motion['TranslationSpeed'][1], 
                                axis_vct_z=Motion['TranslationSpeed'][2]
                                )

def wallslip(workflow, Family):
    '''
    Set an inviscid wall boundary condition.

    .. note:: see `elsA Tutorial about wall conditions <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/BCsTutorials/tutorial-BC.html#wall-conditions/>`_

    Parameters
    ----------

        workflow.tree : PyTree
            Tree to modify

        Family : str
            Name of the family on which the boundary condition will be imposed

    '''
    wall = I.getNodeFromNameAndType(workflow.tree, Family, 'Family_t')
    I._rmNodesByName(wall, '.Solver#BC')
    I._rmNodesByType(wall, 'FamilyBC_t')
    I.newFamilyBC(value='BCWallInviscid', parent=wall)
    c.castNode(wall)

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
    farfield = I.getNodeFromNameAndType(workflow.tree, Family, 'Family_t')
    I._rmNodesByName(farfield, '.Solver#BC')
    I._rmNodesByType(farfield, 'FamilyBC_t')
    I.newFamilyBC(value='BCFarfield', parent=farfield)
    c.castNode(farfield)

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
        for bc in C.getFamilyBCs(workflow.tree, Family):
            setBCwithImposedVariables(workflow.tree, Family, ImposedVariables,
                FamilyBC='BCInflowSubsonic', BCType='inj1', bc=bc, variableForInterpolation=variableForInterpolation)
    else:
        setBCwithImposedVariables(workflow.tree, Family, ImposedVariables,
            FamilyBC='BCInflowSubsonic', BCType='inj1', bc=bc, variableForInterpolation=variableForInterpolation)



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
    FamilyNode = I.getNodeFromNameAndType(workflow.tree, Family, 'Family_t')
    I._rmNodesByName(FamilyNode, '.Solver#BC')
    I._rmNodesByType(FamilyNode, 'FamilyBC_t')
    I.newFamilyBC(value=FamilyBC, parent=FamilyNode)

    if all([np.ndim(v)==0 and not callable(v) for v in ImposedVariables.values()]):
        checkVariables(ImposedVariables)
        ImposedVariables = translateVariablesFromCGNS2Elsa(ImposedVariables)
        J.set(FamilyNode, '.Solver#BC', type=BCType, **ImposedVariables)
    else:
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

def translateVariablesFromCGNS2Elsa(Variables):
    '''
    Translate names in **Variables** from CGNS standards to elsA names for
    boundary conditions.

    Parameters
    ----------

        Variables : :py:class:`dict` or :py:class:`list` or :py:class:`str`
            Could be eiter:

                * a :py:class:`dict` with keys corresponding to variables names

                * a :py:class:`list` of variables names

                * a :py:class:`str` as a single variable name

    Returns
    -------

        NewVariables : :py:class:`dict` or :py:class:`list` or :py:class:`str`
            Depending on the input type, return the same object with variable
            names translated to elsA standards.

    '''
    CGNS2ElsaDict = dict(
        PressureStagnation       = 'stagnation_pressure',
        EnthalpyStagnation       = 'stagnation_enthalpy',
        TemperatureStagnation    = 'stagnation_temperature',
        Pressure                 = 'pressure',
        MassFlow                 = 'globalmassflow',
        SurfacicMassFlow         = 'surf_massflow',
        VelocityUnitVectorX      = 'txv',
        VelocityUnitVectorY      = 'tyv',
        VelocityUnitVectorZ      = 'tzv',
        TurbulentSANuTilde       = 'inj_tur1',
        TurbulentEnergyKinetic   = 'inj_tur1',
        TurbulentDissipationRate = 'inj_tur2',
        TurbulentDissipation     = 'inj_tur2',
        TurbulentLengthScale     = 'inj_tur2',
        VelocityCorrelationXX    = 'inj_tur1',
        VelocityCorrelationXY    = 'inj_tur2', 
        VelocityCorrelationXZ    = 'inj_tur3',
        VelocityCorrelationYY    = 'inj_tur4', 
        VelocityCorrelationYZ    = 'inj_tur5', 
        VelocityCorrelationZZ    = 'inj_tur6',
    )
    if 'VelocityCorrelationXX' in Variables:
        # For RSM models
        CGNS2ElsaDict['TurbulentDissipationRate'] = 'inj_tur7'

    elsAVariables = CGNS2ElsaDict.values()

    if isinstance(Variables, dict):
        NewVariables = dict()
        for var, value in Variables.items():
            if var == 'groupmassflow':
                NewVariables[var] = int(value)                    
            elif var in elsAVariables:
                NewVariables[var] = float(value)
            elif var in CGNS2ElsaDict:
                NewVariables[CGNS2ElsaDict[var]] = float(value)
            else:
                NewVariables[var] = float(value)
        return NewVariables
    elif isinstance(Variables, list):
        NewVariables = []
        for var in Variables:
            if var in elsAVariables:
                NewVariables.append(var)
            else:
                NewVariables.append(CGNS2ElsaDict[var])
        return NewVariables
    elif isinstance(Variables, str):
        if Variables in elsAVariables:
            return CGNS2ElsaDict[Variables]
    else:
        raise TypeError('Variables must be of type dict, list or string')

