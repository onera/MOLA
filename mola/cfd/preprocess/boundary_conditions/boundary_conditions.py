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

from mola import (misc, cgns)

BoundaryConditionsNames = dict(
    Farfield                     = dict(elsa='nref'),
    InflowStagnation             = dict(elsa='inj1'),
    InflowMassFlow               = dict(elsa='injmfr1'),
    OutflowPressure              = dict(elsa='outpres'),
    OutflowMassFlow              = dict(elsa='outmfr2'),
    OutflowRadialEquilibrium     = dict(elsa='outradeq'),
    MixingPlane                  = dict(elsa='stage_mxpl'),
    UnsteadyRotorStatorInterface = dict(elsa='stage_red'),
    WallViscous                  = dict(elsa='walladia'),
    WallViscousIsothermal        = dict(elsa='wallisoth'),
    WallInviscid                 = dict(elsa='wallslip'),
    SymmetryPlane                = dict(elsa='sym'),
)

# Shortcuts for already defined boundary conditions
BoundaryConditionsNames.update(
    dict(
        Wall = BoundaryConditionsNames['WallViscous']
    )
)


def apply(workflow):
    '''
    Set all boundary conditions for **workflow**.
    It transforms the tree attribute of the **workflow**.

    Parameters
    ----------
    workflow : Workflow object
    '''

    for bc in workflow.BoundaryConditions:
        
        bcName = bc['type']
        if bcName in BoundaryConditionsNames:
            # Define in the main MOLA preprocess, lower in this file
            MOLAGenericFunction = getattr('.', bcName)
            solverSpecificFunctionName = BoundaryConditionsNames[bc['type']]
            args, kwargs = MOLAGenericFunction(workflow, bc)
        else:
            # Defined only in the specific solver module
            solverSpecificFunctionName = bcName
            args, kwargs = bc['args'], bc['kwargs']

        solverModule = misc.load_source('solverModule', workflow.Solver)
        try:
            solverSpecificFunction = getattr(solverModule, solverSpecificFunctionName)
        except AttributeError:
            print(misc.RED+f'The function {solverSpecificFunctionName} does not exist for the solver {workflow.Solver}.'+misc.ENDC)
        else:
            solverSpecificFunction(workflow, *args, **kwargs)


def WallViscous(workflow, bc):
    RotationSpeed = bc.get('RotationSpeed', [0., 0., 0.])
    if isinstance(RotationSpeed, (int, float)):
        print(f'No rotation axis for WallViscous condition on {bc["Family"]}: set to x-axis by default.')
        RotationSpeed = [RotationSpeed, 0., 0.]
    RotationAxisOrigin = bc.get('RotationAxisOrigin', [0., 0., 0.])
    TranslationSpeed = bc.get('TranslationSpeed', [0., 0., 0.])

    Motion = dict(
        RotationSpeed = RotationSpeed,
        RotationAxisOrigin = RotationAxisOrigin,
        TranslationSpeed = TranslationSpeed
    )
    return [bc['Family']], dict(Motion=Motion) 

def WallInviscid(workflow, bc):
    return [bc['Family']], dict() 
    
def Farfield(workflow, bc):
    return [bc['Family']], dict() 



def set_boundary_conditions_OLD(t, BoundaryConditions, TurboConfiguration,
    FluidProperties, ReferenceValues, bladeFamilyNames=['BLADE','AUBE']):
    '''
    Set all BCs defined in the dictionary **BoundaryConditions**.

    .. important::

        Wall BCs are defined automatically given the dictionary **TurboConfiguration**


    Parameters
    ----------

        t : PyTree
            preprocessed tree as performed by :py:func:`prepareMesh4ElsA`

        BoundaryConditions : :py:class:`list` of :py:class:`dict`
            User-provided list of boundary conditions. Each element is a
            dictionary with the following keys:

                * FamilyName :
                    Name of the family on which the boundary condition will be imposed

                * type :
                  BC type among the following:

                  * Farfield

                  * InflowStagnation

                  * InflowMassFlow

                  * OutflowPressure

                  * OutflowMassFlow

                  * OutflowRadialEquilibrium

                  * MixingPlane

                  * UnsteadyRotorStatorInterface

                  * WallViscous

                  * WallViscousIsothermal

                  * WallInviscid

                  * SymmetryPlane

                  .. note::
                    elsA names are also available (``nref``, ``inj1``, ``injfmr1``,
                    ``outpres``, ``outmfr2``, ``outradeq``,
                    ``stage_mxpl``, ``stage_red``,
                    ``walladia``, ``wallisoth``, ``wallslip``,
                    ``sym``)

                * option (optional) : add a specification for type
                  InflowStagnation (could be 'uniform' or 'file')

                * other keys depending on type. They will be passed as an
                  unpacked dictionary of arguments to the BC type-specific
                  function.

        TurboConfiguration : dict
            as produced by :py:func:`getTurboConfiguration`

        FluidProperties : dict
            as produced by :py:func:`computeFluidProperties`

        ReferenceValues : dict
            as produced by :py:func:`computeReferenceValues`

        bladeFamilyNames : :py:class:`list` of :py:class:`str`
            list of patterns to find families related to blades.

    See also
    --------

    setBC_Walls, setBC_walladia, setBC_wallisoth, setBC_wallslip, setBC_sym,
    setBC_nref,
    setBC_inj1, setBC_inj1_uniform, setBC_inj1_interpFromFile,
    setBC_outpres, setBC_outmfr2,
    setBC_outradeq, setBC_outradeqhyb,
    setBC_stage_mxpl, setBC_stage_mxpl_hyb,
    setBC_stage_red, setBC_stage_red_hyb,
    setBCwithImposedVariables

    Examples
    --------

    The following list defines classical boundary conditions for a compressor
    stage:

    .. code-block:: python

        BoundaryConditions = [
            dict(type='InflowStagnation', option='uniform', FamilyName='row_1_INFLOW'),
            dict(type='OutflowRadialEquilibrium', FamilyName='row_2_OUTFLOW', valve_type=4, valve_ref_pres=0.75*Pt, valve_relax=0.3*Pt),
            dict(type='MixingPlane', left='Rotor_stator_10_left', right='Rotor_stator_10_right')
        ]

    Each type of boundary conditions currently available in MOLA is detailed below.

    **Wall boundary conditions**

    These BCs are automatically defined based on the rotation speeds in the
    :py:class:`dict` **TurboConfiguration**. There is a strong requirement on the
    names of families defining walls:

    * for the shroud: all family names must contain the pattern 'SHROUD' or 'CARTER'
      (in lower, upper or capitalized case)

    * for the hub: all family names must contain the pattern 'HUB' or 'MOYEU'
      (in lower, upper or capitalized case)

    * for the blades: all family names must contain the pattern 'BLADE' or 'AUBE'
      (in lower, upper or capitalized case). If names differ from that ones, it
      is still possible to give a list of patterns that are enought to find all
      blades (adding 'BLADE' or 'AUBE' if necessary). It is done with the
      argument **bladeFamilyNames** of :py:func:`prepareMainCGNS4ElsA`.

    If needed, these boundary conditions may be overwritten to impose other kinds
    of conditions. For instance, the following :py:class:`dict` may be used as
    an element of the :py:class:`list` **BoundaryConditions** to change the
    family 'SHROUD' into an inviscid wall:

    >>> dict(type='WallInviscid', FamilyName='SHROUD')

    The following py:class:`dict` change the family 'SHROUD' into a symmetry plane:

    >>> dict(type='SymmetryPlane', FamilyName='SHROUD')


    **Inflow boundary conditions**

    For the example, it is assumed that there is only one inflow family called
    'row_1_INFLOW'. The following types can be used as elements of the
    :py:class:`list` **BoundaryConditions**:

    >>> dict(type='Farfield', FamilyName='row_1_INFLOW')

    It defines a 'nref' condition based on the **ReferenceValues**
    :py:class:`dict`.

    >>> dict(type='InflowStagnation', option='uniform', FamilyName='row_1_INFLOW')

    It defines a uniform inflow condition imposing stagnation quantities ('inj1' in
    *elsA*) based on the **ReferenceValues**  and **FluidProperties**
    :py:class:`dict`. To impose values not based on **ReferenceValues**, additional
    optional parameters may be given (see the dedicated documentation for the function).

    >>> dict(type='InflowStagnation', option='file', FamilyName='row_1_INFLOW', filename='inflow.cgns')

    It defines an inflow condition imposing stagnation quantities ('inj1' in
    *elsA*) interpolating a 2D map written in the given file (must be given at cell centers, 
    in the container 'FlowSolution#Centers'). 

    >>> dict(type='InflowMassFlow', FamilyName='row_1_INFLOW')

    It defines a uniform inflow condition imposing the massflow ('inj1mfr1' in
    *elsA*) based on the **ReferenceValues**  and **FluidProperties**
    :py:class:`dict`. To impose values not based on **ReferenceValues**, additional
    optional parameters may be given (see the dedicated documentation for the function).
    In particular, either the massflow (``MassFlow``) or the surfacic massflow
    (``SurfacicMassFlow``) may be specified.

    **Outflow boundary conditions**

    For the example, it is assumed that there is only one outflow family called
    'row_2_OUTFLOW'. The following types can be used as elements of the
    :py:class:`list` **BoundaryConditions**:

    >>> dict(type='OutflowPressure', FamilyName='row_2_OUTFLOW', Pressure=20e3)

    It defines an outflow condition imposing a uniform static pressure ('outpres' in
    *elsA*).

    >>> dict(type='OutflowMassflow', FamilyName='row_2_OUTFLOW', Massflow=5.)

    It defines an outflow condition imposing the massflow ('outmfr2' in *elsA*).
    Be careful, **Massflow** should be the massflow through the given family BC
    *in the simulated domain* (not the 360 degrees configuration, except if it
    is simulated).
    If **Massflow** is not given, the massflow given in the **ReferenceValues**
    is automatically taken and normalized by the appropriate section.

    >>> dict(type='OutflowRadialEquilibrium', FamilyName='row_2_OUTFLOW', valve_type=4, valve_ref_pres=0.75*Pt, valve_ref_mflow=5., valve_relax=0.3*Pt)

    It defines an outflow condition imposing a radial equilibrium ('outradeq' in
    *elsA*). The arguments have the same names that *elsA* keys. Valve law types
    from 1 to 5 are available. The radial equilibrium without a valve law (with
    **valve_type** = 0, which is the default value) is also available. To be
    consistant with the condition 'OutflowPressure', the argument
    **valve_ref_pres** may also be named **Pressure**.


    **Interstage boundary conditions**

    For the example, it is assumed that there is only one interstage with both
    families 'Rotor_stator_10_left' and 'Rotor_stator_10_right'. The following
    types can be used as elements of the :py:class:`list` **BoundaryConditions**:

    >>> dict(type='MixingPlane', left='Rotor_stator_10_left', right='Rotor_stator_10_right')

    It defines a mixing plane ('stage_mxpl' in *elsA*).

    >>> dict(type='UnsteadyRotorStatorInterface', left='Rotor_stator_10_left', right='Rotor_stator_10_right', stage_ref_time=1e-5)

    It defines an unsteady interpolating interface (RNA interface, 'stage_red'
    in *elsA*). If **stage_ref_time** is not provided, it is automatically
    computed assuming a 360 degrees rotor/stator interface:

    >>> stage_ref_time = 2*np.pi / abs(TurboConfiguration['ShaftRotationSpeed'])

    '''
    PreferedBoundaryConditions = dict(
        Farfield                     = 'nref',
        InflowStagnation             = 'inj1',
        InflowMassFlow               = 'injmfr1',
        OutflowPressure              = 'outpres',
        OutflowMassFlow              = 'outmfr2',
        OutflowRadialEquilibrium     = 'outradeq',
        MixingPlane                  = 'stage_mxpl',
        UnsteadyRotorStatorInterface = 'stage_red',
        WallViscous                  = 'walladia',
        WallViscousIsothermal        = 'wallisoth',
        WallInviscid                 = 'wallslip',
        SymmetryPlane                = 'sym',
    )

    print(J.CYAN + 'set BCs at walls' + J.ENDC)
    setBC_Walls(t, TurboConfiguration, bladeFamilyNames=bladeFamilyNames)

    for BCparam in BoundaryConditions:

        BCkwargs = {key:BCparam[key] for key in BCparam if key not in ['type', 'option']}
        if BCparam['type'] in PreferedBoundaryConditions:
            BCparam['type'] = PreferedBoundaryConditions[BCparam['type']]

        if BCparam['type'] == 'nref':
            print(J.CYAN + 'set BC nref on ' + BCparam['FamilyName'] + J.ENDC)
            setBC_nref(t, **BCkwargs)

        elif BCparam['type'] == 'inj1':

            if 'option' not in BCparam:
                if 'bc' in BCkwargs:
                    BCparam['option'] = 'bc'
                else:
                    BCparam['option'] = 'uniform'

            if BCparam['option'] == 'uniform':
                print(J.CYAN + 'set BC inj1 (uniform) on ' + BCparam['FamilyName'] + J.ENDC)
                setBC_inj1_uniform(t, FluidProperties, ReferenceValues, **BCkwargs)

            elif BCparam['option'] == 'file':
                print('{}set BC inj1 (from file {}) on {}{}'.format(J.CYAN,
                    BCparam['filename'], BCparam['FamilyName'], J.ENDC))
                setBC_inj1_interpFromFile(t, ReferenceValues, **BCkwargs)

            elif BCparam['option'] == 'bc':
                print('set BC inj1 on {}'.format(J.CYAN, BCparam['FamilyName'], J.ENDC))
                setBC_inj1(t, ReferenceValues, **BCkwargs)

        elif BCparam['type'] == 'injmfr1':
            print(J.CYAN + 'set BC injmfr1 on ' + BCparam['FamilyName'] + J.ENDC)
            setBC_injmfr1(t, FluidProperties, ReferenceValues, **BCkwargs)

        elif BCparam['type'] == 'outpres':
            print(J.CYAN + 'set BC outpres on ' + BCparam['FamilyName'] + J.ENDC)
            setBC_outpres(t, **BCkwargs)

        elif BCparam['type'] == 'outmfr2':
            print(J.CYAN + 'set BC outmfr2 on ' + BCparam['FamilyName'] + J.ENDC)
            BCkwargs['ReferenceValues'] = ReferenceValues
            BCkwargs['TurboConfiguration'] = TurboConfiguration
            setBC_outmfr2(t, **BCkwargs)

        elif BCparam['type'] == 'outradeq':
            print(J.CYAN + 'set BC outradeq on ' + BCparam['FamilyName'] + J.ENDC)
            BCkwargs['ReferenceValues'] = ReferenceValues
            BCkwargs['TurboConfiguration'] = TurboConfiguration
            setBC_outradeq(t, **BCkwargs)

        elif BCparam['type'] == 'outradeqhyb':
            print(J.CYAN + 'set BC outradeqhyb on ' + BCparam['FamilyName'] + J.ENDC)
            BCkwargs['ReferenceValues'] = ReferenceValues
            BCkwargs['TurboConfiguration'] = TurboConfiguration
            setBC_outradeqhyb(t, **BCkwargs)

        elif BCparam['type'] == 'stage_mxpl':
            print('{}set BC stage_mxpl between {} and {}{}'.format(J.CYAN,
                BCparam['left'], BCparam['right'], J.ENDC))
            setBC_stage_mxpl(t, **BCkwargs)

        elif BCparam['type'] == 'stage_mxpl_hyb':
            print('{}set BC stage_mxpl_hyb between {} and {}{}'.format(J.CYAN,
                BCparam['left'], BCparam['right'], J.ENDC))
            setBC_stage_mxpl_hyb(t, **BCkwargs)

        elif BCparam['type'] == 'stage_red':
            print('{}set BC stage_red between {} and {}{}'.format(J.CYAN,
                BCparam['left'], BCparam['right'], J.ENDC))
            if not 'stage_ref_time' in BCkwargs:
                # Assume a 360 configuration
                BCkwargs['stage_ref_time'] = 2*np.pi / abs(TurboConfiguration['ShaftRotationSpeed'])
            setBC_stage_red(t, **BCkwargs)

        elif BCparam['type'] == 'stage_red_hyb':
            print('{}set BC stage_red_hyb between {} and {}{}'.format(J.CYAN,
                BCparam['left'], BCparam['right'], J.ENDC))
            if not 'stage_ref_time' in BCkwargs:
                # Assume a 360 configuration
                BCkwargs['stage_ref_time'] = 2*np.pi / abs(TurboConfiguration['ShaftRotationSpeed'])
            setBC_stage_red_hyb(t, **BCkwargs)

        elif BCparam['type'] == 'sym':
            print(J.CYAN + 'set BC sym on ' + BCparam['FamilyName'] + J.ENDC)
            setBC_sym(t, **BCkwargs)

        elif BCparam['type'] == 'walladia':
            print(J.CYAN + 'set BC walladia on ' + BCparam['FamilyName'] + J.ENDC)
            setBC_walladia(t, **BCkwargs)

        elif BCparam['type'] == 'wallslip':
            print(J.CYAN + 'set BC wallslip on ' + BCparam['FamilyName'] + J.ENDC)
            setBC_wallslip(t, **BCkwargs)

        elif BCparam['type'] == 'wallisoth':
            print(J.CYAN + 'set BC wallisoth on ' + BCparam['FamilyName'] + J.ENDC)
            setBC_wallisoth(t, **BCkwargs)

        else:
            raise AttributeError('BC type %s not implemented'%BCparam['type'])


def setBC_Walls(t, TurboConfiguration,
                    bladeFamilyNames=['BLADE', 'AUBE'],
                    hubFamilyNames=['HUB', 'SPINNER', 'MOYEU'],
                    shroudFamilyNames=['SHROUD', 'CARTER']):
    '''
    Set all the wall boundary conditions in a turbomachinery context, by making
    the following operations:

        * set BCs related to each blade.
        * set BCs related to hub. The intervals where the rotation speed is the
          shaft speed (for rotor platforms) are set in the following form:

            >>> TurboConfiguration['HubRotationSpeed'] = [(xmin1, xmax1), ..., (xminN, xmaxN)]

        * set BCs related to shroud. Rotation speed is set to zero.

    Parameters
    ----------

        t : PyTree
            Tree to modify

        TurboConfiguration : dict
            as produced :py:func:`getTurboConfiguration`

        bladeFamilyNames : :py:class:`list` of :py:class:`str`
            list of patterns to find families related to blades. Not sensible
            to string case. By default, search patterns 'BLADE' and 'AUBE'.

        hubFamilyNames : :py:class:`list` of :py:class:`str`
            list of patterns to find families related to hub. Not sensible
            to string case. By default, search patterns 'HUB' and 'MOYEU'.

        shroudFamilyNames : :py:class:`list` of :py:class:`str`
            list of patterns to find families related to shroud. Not sensible
            to string case. By default, search patterns 'SHROUD' and 'CARTER'.

    '''
    def extendListOfFamilies(FamilyNames):
        '''
        For each <NAME> in the list **FamilyNames**, add Name, name and NAME.
        '''
        ExtendedFamilyNames = copy.deepcopy(FamilyNames)
        for fam in FamilyNames:
            newNames = [fam.lower(), fam.upper(), fam.capitalize()]
            for name in newNames:
                if name not in ExtendedFamilyNames:
                    ExtendedFamilyNames.append(name)
        return ExtendedFamilyNames

    bladeFamilyNames = extendListOfFamilies(bladeFamilyNames)
    hubFamilyNames = extendListOfFamilies(hubFamilyNames)
    shroudFamilyNames = extendListOfFamilies(shroudFamilyNames)

    if 'PeriodicTranslation' in TurboConfiguration:
        # For linear cascade configuration: all blocks and wall are motionless
        wallFamily = []
        for wallFamily in bladeFamilyNames + hubFamilyNames + shroudFamilyNames:
            for famNode in I.getNodesFromNameAndType(t, '*{}*'.format(wallFamily), 'Family_t'):
                I._rmNodesByType(famNode, 'FamilyBC_t')
                I.newFamilyBC(value='BCWallViscous', parent=famNode)
        return

    def omegaHubAtX(x):
        omega = np.zeros(x.shape, dtype=float)
        for (x1, x2) in TurboConfiguration['HubRotationSpeed']:
            omega[(x1<=x) & (x<=x2)] = TurboConfiguration['ShaftRotationSpeed']
        return np.asfortranarray(omega)

    def getZoneFamilyNameWithFamilyNameBC(zones, FamilyNameBC):
        ZoneFamilyName = None
        for zone in zones:
            ZoneBC = I.getNodeFromType1(zone, 'ZoneBC_t')
            if not ZoneBC: continue
            FamiliesNames = I.getNodesFromType2(ZoneBC, 'FamilyName_t')
            for FamilyName_n in FamiliesNames:
                if I.getValue(FamilyName_n) == FamilyNameBC:
                    ZoneFamilyName = I.getValue(I.getNodeFromType1(zone, 'FamilyName_t'))
            if ZoneFamilyName: break

        assert ZoneFamilyName is not None, 'Cannot determine associated row for family {}. '.format(FamilyNameBC)
        return ZoneFamilyName
        
    # BLADES
    zones = I.getZones(t)
    families = I.getNodesFromType2(t,'Family_t')
    for blade_family in bladeFamilyNames:
        for famNode in I.getNodesFromNameAndType(t, '*{}*'.format(blade_family), 'Family_t'):
            famName = I.getName(famNode)
            if famName.startswith('F_OV_') or famName.endswith('Zones'): continue
            ZoneFamilyName = getZoneFamilyNameWithFamilyNameBC(zones, famName)
            family_with_bcwall, = [f for f in families if f[0]==ZoneFamilyName]
            solver_motion_data = J.get(family_with_bcwall,'.Solver#Motion')
            setBC_walladia(t, famName, omega=solver_motion_data['omega'])

    # HUB
    for hub_family in hubFamilyNames:
        for famNode in I.getNodesFromNameAndType(t, '*{}*'.format(hub_family), 'Family_t'):
            famName = I.getName(famNode)
            if famName.startswith('F_OV_') or famName.endswith('Zones'): continue
            setBC_walladia(t, famName, omega=0.)

            # TODO request initVars of BCDataSet
            wallHubBC = C.extractBCOfName(t, 'FamilySpecified:{0}'.format(famName))
            wallHubBC = C.node2Center(wallHubBC)
            for w in wallHubBC:
                xw = I.getValue(I.getNodeFromName(w,'CoordinateX'))
                zname, wname = I.getName(w).split(os.sep)
                znode = I.getNodeFromNameAndType(t,zname,'Zone_t')
                wnode = I.getNodeFromNameAndType(znode,wname,'BC_t')
                BCDataSet = I.newBCDataSet(name='BCDataSet#Init', value='Null',
                    gridLocation='FaceCenter', parent=wnode)
                J.set(BCDataSet, 'NeumannData', childType='BCData_t', omega=omegaHubAtX(xw))

    # SHROUD
    for shroud_family in shroudFamilyNames:
        for famNode in I.getNodesFromNameAndType(t, '*{}*'.format(shroud_family), 'Family_t'):
            famName = I.getName(famNode)
            if famName.startswith('F_OV_') or famName.endswith('Zones'): continue
            setBC_walladia(t, famName, omega=0.)


def setBC_wallisoth(t, FamilyName, Temperature, bc=None):
    '''
    Set an isothermal wall boundary condition.

    .. note:: see `elsA Tutorial about wall conditions <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/BCsTutorials/tutorial-BC.html#wall-conditions/>`_

    Parameters
    ----------

        t : PyTree
            Tree to modify

        FamilyName : str
            Name of the family on which the boundary condition will be imposed

        Temperature : :py:class:`float` or :py:class:`numpy.ndarray` or :py:class:`dict`
            Value of temperature to impose on the boundary condition. May be:

                * either a scalar: in that case it is imposed once for the
                  family **FamilyName** in the corresponding ``Family_t`` node.

                * or a numpy array: in that case it is imposed for the ``BC_t``
                  node **bc**.

            Alternatively, **Temperature** may be a :py:class:`dict` of the form:

            >>> Temperature = dict(wall_temp=value)

            In that case, the same requirements that before stands for *value*.

        bc : PyTree
            ``BC_t`` node on which the boundary condition will be imposed. Must
            be :py:obj:`None` if the condition must be imposed once in the
            ``Family_t`` node.

    '''
    if isinstance(Temperature, dict):
        assert 'wall_temp' in Temperature
        assert len(Temperature.keys() == 1)
        ImposedVariables = Temperature
    else:
        ImposedVariables = dict(wall_temp=Temperature)
    setBCwithImposedVariables(t, FamilyName, ImposedVariables,
        FamilyBC='BCWallViscousIsothermal', BCType='wallisoth', bc=bc)

def setBC_sym(t, FamilyName):
    '''
    Set a symmetry boundary condition.

    .. note:: see `elsA Tutorial about symmetry condition <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/BCsTutorials/tutorial-BC.html#symmetry/>`_

    Parameters
    ----------

        t : PyTree
            Tree to modify

        FamilyName : str
            Name of the family on which the boundary condition will be imposed

    '''
    symmetry = I.getNodeFromNameAndType(t, FamilyName, 'Family_t')
    I._rmNodesByName(symmetry, '.Solver#BC')
    I._rmNodesByType(symmetry, 'FamilyBC_t')
    I.newFamilyBC(value='BCSymmetryPlane', parent=symmetry)


def getPrimitiveTurbulentFieldForInjection(FluidProperties, ReferenceValues, **kwargs):
        '''
        Get the primitive (without the Density factor) turbulent variables (names and values) 
        to inject in an inflow boundary condition.

        For RSM models, see issue https://elsa.onera.fr/issues/5136 for the naming convention.

        Parameters
        ----------
        ReferenceValues : dict
            as obtained from :py:func:`computeReferenceValues`

        kwargs : dict
            Optional parameters, taken from **ReferenceValues** if not given.

        Returns
        -------
        dict
            Imposed turbulent variables
        '''
        TurbulenceLevel = kwargs.get('TurbulenceLevel', None)
        Viscosity_EddyMolecularRatio = kwargs.get('Viscosity_EddyMolecularRatio', None)
        if TurbulenceLevel and Viscosity_EddyMolecularRatio:
            ReferenceValuesForTurbulence = computeReferenceValues(FluidProperties,
                    kwargs.get('MassFlow'), ReferenceValues['PressureStagnation'],
                    kwargs.get('TemperatureStagnation'), kwargs.get('Surface'),
                    TurbulenceLevel=TurbulenceLevel,
                    Viscosity_EddyMolecularRatio=Viscosity_EddyMolecularRatio,
                    TurbulenceModel=ReferenceValues['TurbulenceModel'])
        else:
            ReferenceValuesForTurbulence = ReferenceValues

        turbDict = dict()
        for name, value in zip(ReferenceValuesForTurbulence['FieldsTurbulence'], ReferenceValuesForTurbulence['ReferenceStateTurbulence']):
            if name.endswith('Density'):
                name = name.replace('Density', '')
                value /= ReferenceValues['Density']
            elif name == 'ReynoldsStressDissipationScale':
                name = 'TurbulentDissipationRate'
                value /= ReferenceValues['Density']
            elif name.startswith('ReynoldsStress'):
                name = name.replace('ReynoldsStress', 'VelocityCorrelation')
                value /= ReferenceValues['Density']
            turbDict[name] = kwargs.get(name, value)
        return turbDict

def setBC_inj1_uniform(t, FluidProperties, ReferenceValues, FamilyName, **kwargs):
    '''
    Set a Boundary Condition ``inj1`` with uniform inflow values. These values
    are them in **ReferenceValues**.

    Parameters
    ----------

        t : PyTree
            Tree to modify

        FluidProperties : dict
            as obtained from :py:func:`computeFluidProperties`

        ReferenceValues : dict
            as obtained from :py:func:`computeReferenceValues`

        FamilyName : str
            Name of the family on which the boundary condition will be imposed

        kwargs : dict
            Optional parameters, taken from **ReferenceValues** if not given:
            PressureStagnation, TemperatureStagnation, EnthalpyStagnation,
            VelocityUnitVectorX, VelocityUnitVectorY, VelocityUnitVectorZ, 
            and primitive turbulent variables

    See also
    --------

    setBC_inj1, setBC_inj1_interpFromFile, setBC_injmfr1

    '''

    PressureStagnation    = kwargs.get('PressureStagnation', ReferenceValues['PressureStagnation'])
    TemperatureStagnation = kwargs.get('TemperatureStagnation', ReferenceValues['TemperatureStagnation'])
    EnthalpyStagnation    = kwargs.get('EnthalpyStagnation', FluidProperties['cp'] * TemperatureStagnation)
    VelocityUnitVectorX   = kwargs.get('VelocityUnitVectorX', ReferenceValues['DragDirection'][0])
    VelocityUnitVectorY   = kwargs.get('VelocityUnitVectorY', ReferenceValues['DragDirection'][1])
    VelocityUnitVectorZ   = kwargs.get('VelocityUnitVectorZ', ReferenceValues['DragDirection'][2])
    variableForInterpolation = kwargs.get('variableForInterpolation', 'ChannelHeight')   

    ImposedVariables = dict(
        PressureStagnation  = PressureStagnation,
        EnthalpyStagnation  = EnthalpyStagnation,
        VelocityUnitVectorX = VelocityUnitVectorX,
        VelocityUnitVectorY = VelocityUnitVectorY,
        VelocityUnitVectorZ = VelocityUnitVectorZ,
        **getPrimitiveTurbulentFieldForInjection(FluidProperties, ReferenceValues, **kwargs)
        )

    setBC_inj1(t, FamilyName, ImposedVariables, variableForInterpolation=variableForInterpolation)

def setBC_inj1_interpFromFile(t, FluidProperties, ReferenceValues, FamilyName, filename, fileformat=None):
    '''
    Set a Boundary Condition ``inj1`` using the field map in the file
    **filename**. It is expected to be a surface with the following variables
    defined at cell centers (in the container 'FlowSolution#Centers'):

        * the coordinates

        * the stagnation pressure ``'PressureStagnation'``

        * the stagnation enthalpy ``'EnthalpyStagnation'``

        * the three components of the unit vector for the velocity direction:
          ``'VelocityUnitVectorX'``, ``'VelocityUnitVectorY'``, ``'VelocityUnitVectorZ'``

        * the primitive turbulent variables (so not multiplied by density)
          comptuted from ``ReferenceValues['FieldsTurbulence']`` and
          depending on the turbulence model.
          For example: ``'TurbulentEnergyKinetic'`` and
          ``'TurbulentDissipationRate'`` for a k-omega model.

    Field variables will be extrapolated on the BCs attached to the family
    **FamilyName**, except if:

    * the file can be converted in a PyTree

    * with zone names like: ``<ZONE>\<BC>``, as obtained from function
      :py:func:`Converter.PyTree.extractBCOfName`

    * and all zone names and BC names are consistent with the current tree **t**

    In that case, field variables are just read in **filename** and written in
    BCs of **t**.

    Parameters
    ----------

        t : PyTree
            Tree to modify

        ReferenceValues : dict
            as obtained from :py:func:`computeReferenceValues`

        FamilyName : str
            Name of the family on which the boundary condition will be imposed

        filename : str
            name of the input filename

        fileformat : optional, str
            format of the input file to be passed to Converter.convertFile2PyTree
            Cassiopee function.

            .. note:: see `available file formats <http://elsa.onera.fr/Cassiopee/Converter.html?highlight=initvars#fileformats>`_

    See also
    --------

    setBC_inj1, setBC_inj1_uniform

    '''

    var2interp = ['PressureStagnation', 'EnthalpyStagnation',
        'VelocityUnitVectorX', 'VelocityUnitVectorY', 'VelocityUnitVectorZ']
    turbDict = getPrimitiveTurbulentFieldForInjection(FluidProperties, ReferenceValues)
    var2interp += list(turbDict)

    donor_tree = C.convertFile2PyTree(filename, format=fileformat)
    inlet_BC_nodes = C.extractBCOfName(t, 'FamilySpecified:{0}'.format(FamilyName))
    I._adaptZoneNamesForSlash(inlet_BC_nodes)
    for w in inlet_BC_nodes:
        bcLongName = I.getName(w)  # from C.extractBCOfName: <zone>\<bc>
        zname, wname = bcLongName.split('\\')
        znode = I.getNodeFromNameAndType(t, zname, 'Zone_t')
        bcnode = I.getNodeFromNameAndType(znode, wname, 'BC_t')

        print('Interpolate Inflow condition on BC {}...'.format(bcLongName))
        I._rmNodesByType(w, 'FlowSolution_t')
        donor_BC = P.extractMesh(donor_tree, w, mode='accurate')

        ImposedVariables = dict()
        for var in var2interp:
            varNode = I.getNodeFromName(donor_BC, var)
            if varNode:
                ImposedVariables[var] = np.asfortranarray(I.getValue(varNode))
            else:
                raise TypeError('variable {} not found in {}'.format(var, filename))

        setBC_inj1(t, FamilyName, ImposedVariables, bc=bcnode)

def setBC_injmfr1(t, FluidProperties, ReferenceValues, FamilyName, **kwargs):
    '''
    Set a Boundary Condition ``injmfr1`` with uniform inflow values. These values
    are them in **ReferenceValues**.

    Parameters
    ----------

        t : PyTree
            Tree to modify

        FluidProperties : dict
            as obtained from :py:func:`computeFluidProperties`

        ReferenceValues : dict
            as obtained from :py:func:`computeReferenceValues`

        FamilyName : str
            Name of the family on which the boundary condition will be imposed

        kwargs : dict
            Optional parameters, taken from **ReferenceValues** if not given:
            MassFlow, SurfacicMassFlow, Surface, TemperatureStagnation, EnthalpyStagnation,
            VelocityUnitVectorX, VelocityUnitVectorY, VelocityUnitVectorZ,
            TurbulenceLevel, Viscosity_EddyMolecularRatio and primitive turbulent variables

    See also
    --------

    setBC_inj1, setBC_inj1_interpFromFile

    '''
    Surface = kwargs.get('Surface', None)
    if not Surface:
        # Compute surface of the inflow BC
        zones = C.extractBCOfName(t, 'FamilySpecified:'+FamilyName)
        SurfaceTree = C.convertArray2Tetra(zones)
        SurfaceTree = C.initVars(SurfaceTree, 'ones=1')
        Surface = P.integ(SurfaceTree, var='ones')[0]

    MassFlow              = kwargs.get('MassFlow', ReferenceValues['MassFlow'])
    SurfacicMassFlow      = kwargs.get('SurfacicMassFlow', MassFlow / Surface)
    TemperatureStagnation = kwargs.get('TemperatureStagnation', ReferenceValues['TemperatureStagnation'])
    EnthalpyStagnation    = kwargs.get('EnthalpyStagnation', FluidProperties['cp'] * TemperatureStagnation)
    VelocityUnitVectorX   = kwargs.get('VelocityUnitVectorX', ReferenceValues['DragDirection'][0])
    VelocityUnitVectorY   = kwargs.get('VelocityUnitVectorY', ReferenceValues['DragDirection'][1])
    VelocityUnitVectorZ   = kwargs.get('VelocityUnitVectorZ', ReferenceValues['DragDirection'][2])
    variableForInterpolation = kwargs.get('variableForInterpolation', 'ChannelHeight')   

    ImposedVariables = dict(
        SurfacicMassFlow    = SurfacicMassFlow,
        EnthalpyStagnation  = EnthalpyStagnation,
        VelocityUnitVectorX = VelocityUnitVectorX,
        VelocityUnitVectorY = VelocityUnitVectorY,
        VelocityUnitVectorZ = VelocityUnitVectorZ,
        **getPrimitiveTurbulentFieldForInjection(FluidProperties, 
                                                 ReferenceValues,
                                                 Surface=Surface,
                                                 MassFlow=MassFlow,
                                                 TemperatureStagnation=TemperatureStagnation
                                                )
        )

    setBCwithImposedVariables(t, FamilyName, ImposedVariables,
        FamilyBC='BCInflowSubsonic', BCType='injmfr1', variableForInterpolation=variableForInterpolation)

def setBC_outpres(t, FamilyName, Pressure, bc=None, variableForInterpolation='ChannelHeight'):
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
    if isinstance(Pressure, dict):
        assert 'Pressure' in Pressure or 'pressure' in Pressure
        assert len(Pressure.keys() == 1)
        ImposedVariables = Pressure
    else:
        ImposedVariables = dict(Pressure=Pressure)

    if not bc and not all([np.ndim(v) == 0 and not callable(v) for v in ImposedVariables.values()]):
        for bc in C.getFamilyBCs(t, FamilyName):
            setBCwithImposedVariables(t, FamilyName, ImposedVariables,
                                      FamilyBC='BCOutflowSubsonic', BCType='outpres', bc=bc, variableForInterpolation=variableForInterpolation)
    else:
        setBCwithImposedVariables(t, FamilyName, ImposedVariables,
                                FamilyBC='BCOutflowSubsonic', BCType='outpres', bc=bc, variableForInterpolation=variableForInterpolation)

def setBC_outmfr2(t, FamilyName, MassFlow=None, groupmassflow=1, ReferenceValues=None, TurboConfiguration=None):
    '''
    Set an outflow boundary condition of type ``outmfr2``.

    .. note:: see `elsA Tutorial about outmfr2 condition <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/BCsTutorials/tutorial-BC.html#outmfr2/>`_

    Parameters
    ----------

        t : PyTree
            Tree to modify

        FamilyName : str
            Name of the family on which the boundary condition will be imposed

        MassFlow : :py:class:`float` or :py:obj:`None`
            Total massflow on the family (with the same **groupmassflow**).
            If :py:obj:`None`, the reference massflow in **ReferenceValues**
            divided by the appropriate fluxcoeff is taken.

            .. attention::
                It has to be the massflow through the simulated section only,
                not on the full 360 degrees configuration (except if the full
                circonference is simulated).

        groupmassflow : int
            Index used to link participating patches to this boundary condition.
            If several BC ``outmfr2`` are defined, **groupmassflow** has to be
            incremented for each family.

        ReferenceValues : :py:class:`dict` or :py:obj:`None`
            dictionary as obtained from :py:func:`computeReferenceValues`. Can
            be :py:obj:`None` only if **MassFlow** is not :py:obj:`None`.

        TurboConfiguration : :py:class:`dict` or :py:obj:`None`
            dictionary as obtained from :py:func:`getTurboConfiguration`. Can
            be :py:obj:`None` only if **MassFlow** is not :py:obj:`None`.

    '''
    if MassFlow is None:
        bc = C.getFamilyBCs(t, FamilyName)[0]
        zone = I.getParentFromType(t, bc, 'Zone_t')
        row = I.getValue(I.getNodeFromType1(zone, 'FamilyName_t'))
        rowParams = TurboConfiguration['Rows'][row]
        fluxcoeff = rowParams['NumberOfBlades'] / float(rowParams['NumberOfBladesSimulated'])
        MassFlow = ReferenceValues['MassFlow'] / fluxcoeff
    else:
        bc = None

    ImposedVariables = dict(globalmassflow=MassFlow, groupmassflow=groupmassflow)

    setBCwithImposedVariables(t, FamilyName, ImposedVariables,
        FamilyBC='BCOutflowSubsonic', BCType='outmfr2', bc=bc)


@J.mute_stdout
def setBC_stage_mxpl(t, left, right, method='globborder_dict'):
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

        method : optional, str
            Method used to compute the globborder. The default value is
            ``'globborder_dict'``, it corresponds to the ETC topological
            algorithm.
            Another possible value is ``'poswin'`` to use the geometrical
            algorithm in *turbo* (in this case, *turbo* environment must be
            sourced).
    '''

    import etc.transform.__future__ as trf

    if method == 'globborder_dict':
        t = trf.defineBCStageFromBC(t, left)
        t = trf.defineBCStageFromBC(t, right)
        t, stage = trf.newStageMxPlFromFamily(t, left, right)

    elif method == 'poswin':
        t = trf.defineBCStageFromBC(t, left)
        t = trf.defineBCStageFromBC(t, right)

        gbdu = computeGlobborderPoswin(t, left)
        # print("newStageMxPlFromFamily(up): gbdu = {}".format(gbdu))
        ups = []
        for bc in C.getFamilyBCs(t, left):
          bcpath = I.getPath(t, bc)
          bcu = trf.BCStageMxPlUp(t, bc)
          globborder = bcu.glob_border(left, opposite=right)
          globborder.i_poswin = gbdu[bcpath]['i_poswin']
          globborder.j_poswin = gbdu[bcpath]['j_poswin']
          globborder.glob_dir_i = gbdu[bcpath]['glob_dir_i']
          globborder.glob_dir_j = gbdu[bcpath]['glob_dir_j']
          ups.append(bcu)

        # Downstream BCs declaration
        gbdd = computeGlobborderPoswin(t, right)
        # print("newStageMxPlFromFamily(down): gbdd = {}".format(gbdd))
        downs = []
        for bc in C.getFamilyBCs(t, right):
          bcpath = I.getPath(t, bc)
          bcd = trf.BCStageMxPlDown(t, bc)
          globborder = bcd.glob_border(right, opposite=left)
          globborder.i_poswin = gbdd[bcpath]['i_poswin']
          globborder.j_poswin = gbdd[bcpath]['j_poswin']
          globborder.glob_dir_i = gbdd[bcpath]['glob_dir_i']
          globborder.glob_dir_j = gbdd[bcpath]['glob_dir_j']
          downs.append(bcd)

        # StageMxpl declaration
        stage = trf.BCStageMxPl(t, up=ups, down=downs)
    else:
        raise Exception

    stage.jtype = 'nomatch_rad_line'
    stage.create()

    setRotorStatorFamilyBC(t, left, right)


@J.mute_stdout
def setBC_stage_mxpl_hyb(t, left, right, nbband=100, c=0.3):
    '''
    Set a hybrid mixing plane condition between families **left** and **right**.

    .. important : This function has a dependency to the ETC module.

    Parameters
    ----------

        t : PyTree
            Tree to modify

        left : str
            Name of the family on the left side.

        right : str
            Name of the family on the right side.

        nbband : int
            Number of points in the radial distribution to compute.

        c : float
            Parameter for the distribution of radial points.

    '''

    import etc.transform.__future__ as trf

    t = trf.defineBCStageFromBC(t, left)
    t = trf.defineBCStageFromBC(t, right)

    t, stage = trf.newStageMxPlHybFromFamily(t, left, right)
    stage.jtype = 'nomatch_rad_line'
    stage.hray_tolerance = 1e-16
    for stg in stage.down:
        filename = "state_radius_{}_{}.plt".format(right, nbband)
        radius = stg.repartition(mxpl_dirtype='axial',
                                 filename=filename, fileformat="bin_tp")
        radius.compute(t, nbband=nbband, c=c)
        radius.write()
    for stg in stage.up:
        filename = "state_radius_{}_{}.plt".format(left, nbband)
        radius = stg.repartition(mxpl_dirtype='axial',
                                 filename=filename, fileformat="bin_tp")
        radius.compute(t, nbband=nbband, c=c)
        radius.write()
    stage.create()

    setRotorStatorFamilyBC(t, left, right)


@J.mute_stdout
def setBC_stage_red(t, left, right, stage_ref_time):
    '''
    Set a RNA condition between families **left** and **right**.

    .. important : This function has a dependency to the ETC module.

    Parameters
    ----------

        t : PyTree
            Tree to modify

        left : str
            Name of the family on the left side.

        right : str
            Name of the family on the right side.

        stage_ref_time : float
            Reference period on the simulated azimuthal extension.
    '''

    import etc.transform.__future__ as trf

    t = trf.defineBCStageFromBC(t, left)
    t = trf.defineBCStageFromBC(t, right)

    t, stage = trf.newStageRedFromFamily(
        t, left, right, stage_ref_time=stage_ref_time)
    stage.create()

    setRotorStatorFamilyBC(t, left, right)


@J.mute_stdout
def setBC_stage_red_hyb(t, left, right, stage_ref_time):
    '''
    Set a hybrid RNA condition between families **left** and **right**.

    .. important : This function has a dependency to the ETC module.

    Parameters
    ----------

        t : PyTree
            Tree to modify

        left : str
            Name of the family on the left side.

        right : str
            Name of the family on the right side.

        stage_ref_time : float
            Reference period on the simulated azimuthal extension.

    '''

    import etc.transform.__future__ as trf

    t = trf.defineBCStageFromBC(t, left)
    t = trf.defineBCStageFromBC(t, right)

    t, stage = trf.newStageRedHybFromFamily(
        t, left, right, stage_ref_time=stage_ref_time)
    stage.create()

    for gc in I.getNodesFromType(t, 'GridConnectivity_t'):
        I._rmNodesByType(gc, 'FamilyBC_t')


@J.mute_stdout
def setBC_outradeq(t, FamilyName, valve_type=0, valve_ref_pres=None,
    valve_ref_mflow=None, valve_relax=0.1, indpiv=1, 
    ReferenceValues=None, TurboConfiguration=None, method='globborder_dict'):
    '''
    Set an outflow boundary condition of type ``outradeq``.

    .. important : This function has a dependency to the ETC module.

    Parameters
    ----------

        t : PyTree
            Tree to modify

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

    import etc.transform.__future__ as trf

    if valve_ref_pres is None:
        try:
            valve_ref_pres = ReferenceValues['Pressure']
        except:
            MSG = 'valve_ref_pres or ReferenceValues must be not None'
            raise Exception(J.FAIL+MSG+J.ENDC)
    if valve_type != 0 and valve_ref_mflow is None:
        try:
            bc = C.getFamilyBCs(t, FamilyName)[0]
            zone = I.getParentFromType(t, bc, 'Zone_t')
            row = I.getValue(I.getNodeFromType1(zone, 'FamilyName_t'))
            rowParams = TurboConfiguration['Rows'][row]
            fluxcoeff = rowParams['NumberOfBlades'] / \
                float(rowParams['NumberOfBladesSimulated'])
            valve_ref_mflow = ReferenceValues['MassFlow'] / fluxcoeff
        except:
            MSG = 'Either valve_ref_mflow or both ReferenceValues and TurboConfiguration must be not None'
            raise Exception(J.FAIL+MSG+J.ENDC)

    # Delete previous BC if it exists
    for bc in C.getFamilyBCs(t, FamilyName):
        I._rmNodesByName(bc, '.Solver#BC')
    # Create Family BC
    family_node = I.getNodeFromNameAndType(t, FamilyName, 'Family_t')
    I._rmNodesByName(family_node, '.Solver#BC')
    I.newFamilyBC(value='BCOutflowSubsonic', parent=family_node)

    # Outflow (globborder+outradeq, valve 4)
    if method == 'globborder_dict':
        from etc.globborder.globborder_dict import globborder_dict
        gbd = globborder_dict(t, FamilyName, config="axial")
    elif method == 'poswin':
        gbd = computeGlobborderPoswin(t, FamilyName)
    else:
        raise Exception
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
                         valve_ref_mflow, valve_relax=valve_relax)
        globborder = bc.glob_border(current=FamilyName)
        globborder.i_poswin = gbd[bcpath]['i_poswin']
        globborder.j_poswin = gbd[bcpath]['j_poswin']
        globborder.glob_dir_i = gbd[bcpath]['glob_dir_i']
        globborder.glob_dir_j = gbd[bcpath]['glob_dir_j']
        globborder.azi_orientation = gbd[bcpath]['azi_orientation']
        globborder.h_orientation = gbd[bcpath]['h_orientation']
        bc.create()


@J.mute_stdout
def setBC_outradeqhyb(t, FamilyName, valve_type=0, valve_ref_pres=None,
                      valve_ref_mflow=None, valve_relax=0.1, indpiv=1, nbband=100, c=0.3, 
                      ReferenceValues=None, TurboConfiguration=None):
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

    import etc.transform.__future__ as trf

    if valve_ref_pres is None:
        try:
            valve_ref_pres = ReferenceValues['Pressure']
        except:
            MSG = 'valve_ref_pres or ReferenceValues must be not None'
            raise Exception(J.FAIL+MSG+J.ENDC)
    if valve_type != 0 and valve_ref_mflow is None:
        try:
            bc = C.getFamilyBCs(t, FamilyName)[0]
            zone = I.getParentFromType(t, bc, 'Zone_t')
            row = I.getValue(I.getNodeFromType1(zone, 'FamilyName_t'))
            rowParams = TurboConfiguration['Rows'][row]
            fluxcoeff = rowParams['NumberOfBlades'] / \
                float(rowParams['NumberOfBladesSimulated'])
            valve_ref_mflow = ReferenceValues['MassFlow'] / fluxcoeff
        except:
            MSG = 'Either valve_ref_mflow or both ReferenceValues and TurboConfiguration must be not None'
            raise Exception(J.FAIL+MSG+J.ENDC)

    # Delete previous BC if it exists
    for bc in C.getFamilyBCs(t, FamilyName):
        I._rmNodesByName(bc, '.Solver#BC')
    # Create Family BC
    family_node = I.getNodeFromNameAndType(t, FamilyName, 'Family_t')
    I._rmNodesByName(family_node, '.Solver#BC')
    I.newFamilyBC(value='BCOutflowSubsonic', parent=family_node)

    bc = trf.BCOutRadEqHyb(
        t, I.getNodeFromNameAndType(t, FamilyName, 'Family_t'))
    bc.glob_border()
    bc.indpiv = indpiv
    valve_law_dict = {1: 'SlopePsQ', 2: 'QTarget',
                      3: 'QLinear', 4: 'QHyperbolic'}
    bc.valve_law(valve_law_dict[valve_type], valve_ref_pres,
                 valve_ref_mflow, valve_relax=valve_relax)
    bc.dirorder = -1
    radius_filename = "state_radius_{}_{}.plt".format(FamilyName, nbband)
    radius = bc.repartition(filename=radius_filename, fileformat="bin_tp")
    radius.compute(t, nbband=nbband, c=c)
    radius.write()
    bc.create()


def setRotorStatorFamilyBC(t, left, right):
    for gc in I.getNodesFromType(t, 'GridConnectivity_t'):
        I._rmNodesByType(gc, 'FamilyBC_t')

    leftFamily = I.getNodeFromNameAndType(t, left, 'Family_t')
    rightFamily = I.getNodeFromNameAndType(t, right, 'Family_t')
    I.newFamilyBC(value='BCOutflow', parent=leftFamily)
    I.newFamilyBC(value='BCInflow', parent=rightFamily)


def computeGlobborderPoswin(tree, win):
    from turbo.poswin import computePosWin
    gbd = computePosWin(tree, win)
    for path, obj in gbd.items():
        gbd.pop(path)
        bc = I.getNodeFromPath(tree, path)
        gdi, gdj = getGlobDir(tree, bc)
        gbd['CGNSTree/'+path] = dict(glob_dir_i=gdi, glob_dir_j=gdj,
                                     i_poswin=obj.i, j_poswin=obj.j,
                                     azi_orientation=gdi, h_orientation=gdj)
    return gbd


def getGlobDir(tree, bc):
    # Remember: glob_dir_i is the opposite of theta, which is positive when it goes from Y to Z
    # Remember: glob_dir_j is as the radius, which is positive when it goes from hub to shroud

    # Check if the BC is in i, j or k constant: need pointrage of BC
    ptRi = I.getValue(I.getNodeFromName(bc, 'PointRange'))[0]
    ptRj = I.getValue(I.getNodeFromName(bc, 'PointRange'))[1]
    ptRk = I.getValue(I.getNodeFromName(bc, 'PointRange'))[2]
    x, y, z = J.getxyz(I.getParentFromType(tree, bc, 'Zone_t'))
    y0 = y[0, 0, 0]
    z0 = z[0, 0, 0]

    if ptRi[0] == ptRi[1]:
        dir1 = 2  # j
        dir2 = 3  # k
        y1 = y[0, -1, 0]
        z1 = z[0, -1, 0]
        y2 = y[0, 0, -1]
        z2 = y[0, 0, -1]

    elif ptRj[0] == ptRj[1]:
        dir1 = 1  # i
        dir2 = 3  # k
        y1 = y[-1, 0, 0]
        z1 = z[-1, 0, 0]
        y2 = y[0, 0, -1]
        z2 = y[0, 0, -1]

    elif ptRk[0] == ptRk[1]:
        dir1 = 1  # i
        dir2 = 2  # j
        y1 = y[-1, 0, 0]
        z1 = z[-1, 0, 0]
        y2 = y[0, -1, 0]
        z2 = y[0, -1, 0]

    rad0 = np.sqrt(y0**2+z0**2)
    rad1 = np.sqrt(y1**2+z1**2)
    rad2 = np.sqrt(y2**2+z2**2)
    tet0 = np.arctan2(z0, y0)
    tet1 = np.arctan2(z1, y1)
    tet2 = np.arctan2(z2, y2)

    ang1 = np.arctan2(rad1-rad0, rad1*tet1-rad0*tet0)
    ang2 = np.arctan2(rad2-rad0, rad2*tet2-rad0*tet0)

    if abs(np.sin(ang2)) < abs(np.sin(ang1)):
        # dir2 is more vertical than dir1
        # => globDirJ = +/- dir2
        if np.cos(ang1) > 0:
            # dir1 points towards theta>0
            globDirI = -dir1
        else:
            # dir1 points towards thetaw0
            globDirI = dir1
        if np.sin(ang2) > 0:
            # dir2 points towards r>0
            globDirJ = dir2
        else:
            # dir2 points towards r<0
            globDirJ = -dir2
    else:
        # dir1 is more vertical than dir2
        # => globDirJ = +/- dir1
        if np.cos(ang2) > 0:
            # dir2 points towards theta>0
            globDirI = -dir2
        else:
            # dir2 points towards thetaw0
            globDirI = dir2
        if np.sin(ang1) > 0:
            # dir1 points towards r>0
            globDirJ = dir1
        else:
            # dir1 points towards r<0
            globDirJ = -dir1

    print('  * glob_dir_i = %s\n  * glob_dir_j = %s' % (globDirI, globDirJ))
    assert globDirI != globDirJ
    return globDirI, globDirJ

