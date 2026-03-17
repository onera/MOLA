import numpy as np

from .LiftingLine import getLiftingLines, checkComponentKind
from .Global import *

from mola.pytree import InternalShortcuts as J
import Converter.PyTree as C
import Converter.Internal as I
import Transform.PyTree as T

def addPitch(LiftingLine, pitch=0.0):
    '''
    Given a Lifting-Line (at any arbitrary position) add a pitch angle (in degrees),
    resulting in a rotation of the LiftingLine and its meaningful fields

    Parameters
    ----------

        LiftingLine : PyTree, base, zone or list of zones
            A Lifting Line object, as generated from
            function :py:func:`buildLiftingLine`.

            .. note:: 
                **LitingLine** is modified in-place

        pitch : float
            angle in degrees of the rotation around `PitchAxis` passing through
            `PitchRelativeCenter`, previously defined in :py:func:`buildLiftingLine`
    '''
    for LL in getLiftingLines(LiftingLine):
        Kin_n = I.getNodeFromName(LL,'.Kinematics')
        rc = I.getValue(I.getNodeFromName1(Kin_n,'RotationCenter'))

        PitchCtr = J.getVars(LL,
                        ['PitchRelativeCenterX','PitchRelativeCenterY','PitchRelativeCenterZ'])


        PitchAxis = J.getVars(LL, ['PitchAxisX','PitchAxisY','PitchAxisZ'])

        PitchCtr_pt = (PitchCtr[0][0]+rc[0], PitchCtr[1][0]+rc[1], PitchCtr[2][0]+rc[2])
        PitchAxis_vec = (PitchAxis[0][0], PitchAxis[1][0], PitchAxis[2][0])
        T._rotate(LL, PitchCtr_pt, PitchAxis_vec, pitch, 
                vectors=NamesOfChordSpanThickwiseFrameNoTangential)
        
def resetPitch(LiftingLine, ZeroPitchRelativeSpan=0.75, modifyLiftingLine=True):
    '''
    Given an existing LiftingLine object, reset the pitch taking
    as reference the value in attribute **ZeroPitchRelativeSpan**,
    which modifies in-place the **LiftingLine** object (update of ``Twist``
    field) applying a *DeltaTwist* value such that the resulting Twist
    yields ``0`` degrees at **ZeroPitchRelativeSpan**.
    The value of *DeltaTwist* is returned by the function.

    Parameters
    ----------

        LiftingLine : zone
            the lifting line zone

            .. note:: zone **LiftingLine** is modified if **modifyLiftingLine**
                = :py:obj:`True`

        ZeroPitchRelativeSpan : float
            the relative span location where zero twist must be placed.

        modifyLiftingLine : bool
            if :py:obj:`True`, modify the ``Twist`` field of the
            **LiftingLine** accordingly.

    Returns
    -------

        DeltaTwist : float
            Value required to be added to ``Twist`` field in order
            to verify :math:`Twist=0` at the location requested by **ZeroPitchRelativeSpan**
    '''
    r, Twist = J.getVars(LiftingLine,['Span','Twist'])
    DeltaTwist = J.interpolate__(np.array([ZeroPitchRelativeSpan]), r/r.max(), Twist)
    if modifyLiftingLine: Twist -= DeltaTwist

    return DeltaTwist

def setRPM(LiftingLines, newRPM):
    for LiftingLine in getLiftingLines(LiftingLines):
        if not checkComponentKind(LiftingLine,'LiftingLine'): continue
        Kin_n = I.getNodeFromName1(LiftingLine,'.Kinematics')
        if Kin_n:
            RPM = I.getNodeFromName1(Kin_n,'RPM')

            if RPM:
                RPMvalue = np.atleast_1d(RPM[1])
                RPMvalue[:] = newRPM
            else:
                I.createNode('RPM','DataArray_t',
                             value=np.atleast_1d(np.array(newRPM,dtype=np.float64)),
                             parent=Kin_n)
        else:
            J.set(LiftingLine,'.Kinematics',RPM=np.atleast_1d(np.array(newRPM,dtype=np.float64)))

def setKinematicsUsingConstantRotationAndTranslation(LiftingLines, RotationCenter=[0,0,0],
                                  RotationAxis=[1,0,0], RPM=2500.0, RightHandRuleRotation=True,
                                  VelocityTranslation=[0,0,0], TorqueOrigin=[0,0,0]):
    '''
    This function is a convenient wrap used for setting the ``.Kinematics`` node of **LiftingLine** object.

    .. note:: information contained in ``.Kinematics`` node
        is used by :py:func:`moveLiftingLines` and :py:func:`computeKinematicVelocity`
        functions.

    Parameters
    ----------

        LiftingLines : PyTree, base, zone or list of zones
            Container with Lifting lines where ``.Kinematics`` node is to be set

            .. note:: zones contained in **LiftingLines** are modified

        RotationCenter : :py:class:`list` of 3 :py:class:`float`
            Rotation Center of the motion :math:`(x,y,z)` components

        RotationAxis : :py:class:`list` of 3 :py:class:`float`
            Rotation axis of the motion :math:`(x,y,z)` components

        RPM : float
            revolution per minute. Angular speed of the motion.

        RightHandRuleRotation : bool
            if :py:obj:`True`, the motion is done around
            the **RotationAxis** following the right-hand-rule convention.
            :py:obj:`False` otherwise.

        VelocityTranslation : :py:class:`list` of 3 :py:class:`float`
            Constant velocity translation of the LiftingLine
            along the :math:`(x,y,z)` coordinates in :math:`(m/s)`

    '''

    for LiftingLine in I.getZones( LiftingLines ):
        J.set(LiftingLine,'.Kinematics',
                RotationCenter=np.array(RotationCenter,dtype=np.float64),
                RotationAxis=np.array(RotationAxis,dtype=np.float64),
                RPM=np.atleast_1d(np.array(RPM,dtype=np.float64)),
                RightHandRuleRotation=RightHandRuleRotation,
                VelocityTranslation=np.array(VelocityTranslation,dtype=np.float64),
                TorqueOrigin=np.array(TorqueOrigin,dtype=np.float64))

def getRotationAxisCenterAndDirFromKinematics(LiftingLine):
    '''

    .. note:: this is a private function

    Extract **RotationAxis**, **RotationCenter** and rotation direction from
    ``.Kinematics`` node.

    Parameters
    ----------

        LiftingLine : zone
            Lifting-line with ``.Kinematics`` node defined
            using :py:func:`setKinematicsUsingConstantRotationAndTranslation`

            .. note:: zone **LiftingLine** is modified
    '''
    Kinematics_n = I.getNodeFromName(LiftingLine,'.Kinematics')
    if not Kinematics_n:
        raise ValueError('missing ".Kinematics" node')


    RotationAxis_n = I.getNodeFromName1(Kinematics_n,'RotationAxis')
    if not RotationAxis_n:
        raise ValueError('missing "RotationAxis" node in ".Kinematics"')
    RotationAxis = I.getValue( RotationAxis_n )

    RotationCenter_n = I.getNodeFromName1(Kinematics_n,'RotationCenter')
    if not RotationCenter_n:
        raise ValueError('missing "RotationCenter" node in ".Kinematics"')
    RotationCenter = I.getValue( RotationCenter_n )

    Dir_n = I.getNodeFromName1(Kinematics_n,'RightHandRuleRotation')
    if not Dir_n:
        raise ValueError('missing "RightHandRuleRotation" node in ".Kinematics"')
    Dir = I.getValue( Dir_n )
    if not Dir: Dir = -1


    return RotationAxis, RotationCenter, Dir

def computeKinematicVelocity(t):
    '''
    Compute or update ``VelocityKinematicX``, ``VelocityKinematicY`` and
    ``VelocityKinematicZ`` fields of LiftingLines provided to function using
    information contained in ``.Kinematics`` node attached to each LiftingLine.

    Parameters
    ----------

        t : Pytree, base, list of zones, zone
            container with LiftingLines

            .. note:: LiftingLines contained in **t** are modified.
    '''

    RequiredFieldNames = ['VelocityKinematicX',
                          'VelocityKinematicY',
                          'VelocityKinematicZ',]

    LiftingLines = [z for z in I.getZones(t) if checkComponentKind(z,'LiftingLine')]
    for LiftingLine in LiftingLines:
        Kinematics = J.get(LiftingLine,'.Kinematics')
        VelocityTranslation = Kinematics['VelocityTranslation']
        RotationCenter = Kinematics['RotationCenter']
        RotationAxis = Kinematics['RotationAxis']
        RPM = Kinematics['RPM']
        Dir = 1 if Kinematics['RightHandRuleRotation'] else -1
        Omega = RPM*np.pi/30.
        x,y,z = J.getxyz(LiftingLine)
        ExistingFieldNames = C.getVarNames(LiftingLine,excludeXYZ=True)[0]
        v = dict()
        for fieldname in RequiredFieldNames:
            if fieldname in ExistingFieldNames:
                v[fieldname] = J.getVars(LiftingLine,[fieldname])[0]
            else:
                v[fieldname] = J.invokeFields(LiftingLine,[fieldname])[0]

        NPts = len(x)
        # TODO vectorize this
        for i in range(NPts):
            rvec = np.array([x[i] - RotationCenter[0],
                             y[i] - RotationCenter[1],
                             z[i] - RotationCenter[2]],dtype=np.float64)

            VelocityKinematic = np.cross( Dir * Omega * RotationAxis, rvec) + VelocityTranslation

            v['VelocityKinematicX'][i] = VelocityKinematic[0]
            v['VelocityKinematicY'][i] = VelocityKinematic[1]
            v['VelocityKinematicZ'][i] = VelocityKinematic[2]

def assembleAndProjectVelocities(t):
    '''
    This function updates a series of veolicity (and other fields) of
    LiftingLines provided a given kinematic and flow conditions.

    The new or updated fields are the following :

    * ``VelocityRelativeX`` ``VelocityRelativeY`` ``VelocityRelativeZ``
        Three components of the VelocityInduced + VelocityFreestream + VelocityPerturbation - 
            VelocityKinematic, i.e. of the Relative Velocity.

    * ``VelocityX`` ``VelocityY`` ``VelocityZ``
        Three components of the VelocityInduced + VelocityFreestream + VelocityPerturbation.

    * ``VelocityInducedX`` ``VelocityInducedY`` ``VelocityInducedZ``
        Three components of the Velocity Induced by the Flow.

    * ``VelocityPerturbationX`` ``VelocityPerturbationY`` ``VelocityPerturbationZ``
        Three components of the Perturbation Velocity given by a Perturbation Field.

    * ``VelocityKinematicX`` ``VelocityKinematicY`` ``VelocityKinematicZ``
        Three components of the Kinematic Velocity due to the kinematic movement of the Lifting 
        Lines.

    * ``VelocityAxial``
        Relative velocity in -RotationAxis direction.

    * ``VelocityTangential``
        Relative velocity in the rotation plane direction.

    * ``VelocityChordwise``
        This is the projection of the VelocityRelative upon the Chordwise direction.

    * ``VelocityThickwise``
        This is the projection of the VelocityRelative upon the Thickwise direction.

    * ``VelocityMagnitudeLocal``
        Magnitude of the local velocity neglecting the radial contribution.

    * ``Chord``
        Chord of the local sections.

    * ``ChordVirtualWithSweep``
        Chord of the local sections with the sweep angle correction as
        ``Chord * np.cos(SweepAngleDeg)``

    * ``AoA``
        Local angle-of-attack of the blade section in degree.
        ``np.rad2deg(np.arctan2( VelocityThickwise, VelocityChordwise ))``

    * ``phiRad``
        Local angle-of-attack of the blade section with respect to rotation plane.

    * ``Mach``
        Mach number neglecting the radial contribution.

    * ``Reynolds``
        Reynolds number neglecting the radial contribution.

    * ``ChordwiseX`` ``ChordwiseY`` ``ChordwiseZ``
        Three components of the local Chord direction of each section.

    * ``ThickwiseX`` ``ThickwiseY`` ``ThickwiseZ``
        Three components of the local direction normal to the chord of each section.

    * ``TangentialwiseX`` ``TangentialwiseY`` ``TangentialwiseZ``
        Three components of the local direction tangential to rotation trajectory.

    * ``SweepAngleDeg``
        Local sweep angle (in degree) of the blade section relative to the Lifting Line.

    * ``DihedralAngleDeg``
        Local Dihedral angle (in degree) of the blade section relative to the Lifting Line.

    .. attention:: please note that this function requires the LiftingLine to
        have the fields: ``VelocityKinematicX``, ``VelocityKinematicY``, ``VelocityKinematicZ``,
        ``VelocityInducedX``,   ``VelocityInducedY``,   ``VelocityInducedZ``,
        if they are not found, then they are created (with zero values).

    Parameters
    ----------

        t : PyTree, base, list of zones, zone
            container with LiftingLines.

            .. note:: Lifting-lines contained in **t** are modified.

    '''
    RequiredFieldNames = [
                          'VelocityRelativeX',
                          'VelocityRelativeY',
                          'VelocityRelativeZ',
                          'VelocityX',
                          'VelocityY',
                          'VelocityZ',
                          'VelocityInducedX',
                          'VelocityInducedY',
                          'VelocityInducedZ',
                          'VelocityPerturbationX',
                          'VelocityPerturbationY',
                          'VelocityPerturbationZ',
                          'VelocityKinematicX',
                          'VelocityKinematicY',
                          'VelocityKinematicZ',
                          'VelocityAxial',
                          'VelocityTangential',
                          'VelocityChordwise',
                          'VelocityThickwise',
                          'VelocityMagnitudeLocal',
                          'Chord', 'ChordVirtualWithSweep',
                          'AoA','phiRad','Mach','Reynolds',
                          'ChordwiseX','ChordwiseY','ChordwiseZ',
                          'ThickwiseX','ThickwiseY','ThickwiseZ',
                          'TangentialX','TangentialY','TangentialZ',
                          'SweepAngleDeg', 'DihedralAngleDeg'
                          ]

    LiftingLines = [z for z in I.getZones(t) if checkComponentKind(z,'LiftingLine')]
    for LiftingLine in LiftingLines:
        #Collecting prescribed polar corrections
        Correc_n = I.getNodeFromName(LiftingLine,'Corrections3D')
        SweepCorrection = I.getValue(I.getNodeFromName(Correc_n,'Sweep'))
        DihedralCorrection = I.getValue(I.getNodeFromName(Correc_n,'Dihedral'))

        Conditions = J.get(LiftingLine,'.Conditions')
        Temperature = Conditions['Temperature']
        Density = Conditions['Density']
        VelocityFreestream = Conditions['VelocityFreestream']
        Kinematics = J.get(LiftingLine,'.Kinematics')
        RotationAxis = Kinematics['RotationAxis']
        dir = 1 if Kinematics['RightHandRuleRotation'] else -1
        Mu = Mus*np.sqrt(Temperature/Ts)*(1. + Cs/Ts)/(1. + Cs/Temperature)
        SoundSpeed = np.sqrt(Gamma * Rgp * Temperature)
        
        ExistingFieldNames = C.getVarNames(LiftingLine, excludeXYZ = True)[0]
        v = dict()
        for fieldname in RequiredFieldNames:
            if fieldname in ExistingFieldNames:
                v[fieldname] = J.getVars(LiftingLine,[fieldname])[0]
            else:
                v[fieldname] = J.invokeFields(LiftingLine,[fieldname])[0]

        VelocityInduced = np.vstack([v['VelocityInduced' + i] for i in 'XYZ'])
        VelocityPerturbation = np.vstack([v['VelocityPerturbation' + i] for i in 'XYZ'])
        VelocityKinematic = np.vstack([v['VelocityKinematic' + i] for i in 'XYZ'])
        ChordwiseDirection = np.vstack([v['Chordwise' + i] for i in 'XYZ'])
        ThickwiseDirection = np.vstack([v['Thickwise' + i] for i in 'XYZ'])
        TangentialDirection = np.vstack([v['Tangential' + i] for i in 'XYZ'])

        Velocity          = (VelocityInduced + VelocityPerturbation).T + VelocityFreestream
        v['VelocityX'][:] = Velocity[:, 0]
        v['VelocityY'][:] = Velocity[:, 1]
        v['VelocityZ'][:] = Velocity[:, 2]

        VelocityRelative          = Velocity - VelocityKinematic.T
        v['VelocityRelativeX'][:] = VelocityRelative[:, 0]
        v['VelocityRelativeY'][:] = VelocityRelative[:, 1]
        v['VelocityRelativeZ'][:] = VelocityRelative[:, 2]

        v['VelocityAxial'][:]      = VelocityRelative.dot(-RotationAxis)
        v['VelocityTangential'][:] = np.diag(VelocityRelative.dot(TangentialDirection))

        Vchord = np.diag(VelocityRelative.dot(ChordwiseDirection))
        Vthick = np.diag(VelocityRelative.dot(ThickwiseDirection))

        if SweepCorrection:    Vchord = Vchord*np.cos(np.deg2rad(v['SweepAngleDeg']))
        if DihedralCorrection: Vthick = Vthick*np.cos(np.deg2rad(v['DihedralAngleDeg']))

        v['VelocityChordwise'][:] = Vchord
        v['VelocityThickwise'][:] = Vthick
        # Updating the Angle of Attack considering the new velocity components.
        v['AoA'][:] = np.rad2deg(np.arctan2(Vthick, Vchord))

        # NOTE the absence of radial velocity contribution to Velocity Magnitude, Mach and Reynolds
        v['VelocityMagnitudeLocal'][:] = W = np.linalg.norm(np.vstack([Vchord, Vthick]), axis = 0)
        v['Mach'][:] = W/SoundSpeed
        if SweepCorrection: v['Reynolds'][:] = Density[0]/Mu*W*v['ChordVirtualWithSweep']
        else:               v['Reynolds'][:] = Density[0]/Mu*W*v['Chord']

        V2D = (Vchord*ChordwiseDirection + Vthick*ThickwiseDirection).T
        V2Da = V2D.dot(-RotationAxis)
        V2Dt = dir*np.diag(V2D.dot(TangentialDirection))
        v['phiRad'][:] = np.arctan2(V2Da, V2Dt) #Used for Tip-Loss corrections

def moveLiftingLines(t, TimeStep):
    '''
    Move the lifting lines following their ``.Kinematics`` law.

    It also updates the local frame quantities of the lifting lines and
    updates the kinematic velocity.

    Parameters
    ----------

        t : PyTree, base, list of zones, zone
            container with LiftingLines.

            .. note:: Lifting-lines contained in **t** are modified.

        TimeStep : float
            time step for the movement of the lifting-lines in [s]
    '''
    LiftingLines = [z for z in I.getZones(t) if checkComponentKind(z,'LiftingLine')]
    for LiftingLine in LiftingLines:
        Kinematics = J.get(LiftingLine,'.Kinematics')
        VelocityTranslation = Kinematics['VelocityTranslation']
        RotationCenter = Kinematics['RotationCenter']
        RotationAxis = Kinematics['RotationAxis']
        RightHandRuleRotation = Kinematics['RightHandRuleRotation']
        RPM = Kinematics['RPM']
        Omega = RPM * np.pi / 30.
        Dpsi = np.rad2deg( Omega * TimeStep )
        try: Dpsi = Dpsi[0]
        except: pass
        if not RightHandRuleRotation: Dpsi *= -1

        if Dpsi: T._rotate(LiftingLine, RotationCenter, RotationAxis, Dpsi,
                           vectors=NamesOfChordSpanThickwiseFrame)

        if VelocityTranslation.any():
            T._translate(LiftingLine, TimeStep*VelocityTranslation)
            RotationCenter += TimeStep*VelocityTranslation

        computeKinematicVelocity(LiftingLine)

def moveObject(t, TimeStep):
    '''
    Move the lifting lines following their ``.Kinematics`` law.

    It also updates the local frame quantities of the lifting lines and
    updates the kinematic velocity.

    Parameters
    ----------

        t : PyTree, base, list of zones, zone
            container with LiftingLines.

            .. note:: Lifting-lines contained in **t** are modified.

        TimeStep : float
            time step for the movement of the lifting-lines in [s]
    '''

    for LiftingLine in I.getZones(t):
        Kinematics = J.get(LiftingLine,'.Kinematics')
        VelocityTranslation = Kinematics['VelocityTranslation']
        RotationCenter = Kinematics['RotationCenter']
        RotationAxis = Kinematics['RotationAxis']
        RightHandRuleRotation = Kinematics['RightHandRuleRotation']
        RPM = Kinematics['RPM']
        Omega = RPM * np.pi / 30.
        Dpsi = np.rad2deg( Omega * TimeStep )
        try: Dpsi = Dpsi[0]
        except: pass
        if not RightHandRuleRotation: Dpsi *= -1

        if Dpsi: T._rotate(LiftingLine, RotationCenter, RotationAxis, Dpsi,
                           vectors=NamesOfChordSpanThickwiseFrame)

        if VelocityTranslation.any():
            T._translate(LiftingLine, TimeStep*VelocityTranslation)
            RotationCenter += TimeStep*VelocityTranslation

