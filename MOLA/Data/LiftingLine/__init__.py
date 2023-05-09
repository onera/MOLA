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

'''
Main subpackage for LiftingLine related operations

06/05/2022 - L. Bernardos - first creation
'''
from ..Core import np,RED,GREEN,WARN,PINK,CYAN,ENDC,interpolate
from ..Mesh import Curve
from ..Node import Node
from ..Zone import Zone
from ... import __version__
from . import Polars
import scipy.integrate as sint
import scipy.optimize as so

class LiftingLine(Curve):
    """docstring for LiftingLine"""
    def __init__(self, SpanMin=None, SpanMax=None, N=None, SpanwiseDistribution=dict(),
                       GeometricalLaws=dict(), AirfoilPolarsFilenames=[],
                       *args, **kwargs):


        if isinstance(SpanMin,float) or isinstance(SpanMin,int):
            super().__init__(*args, **kwargs)
            Span = X = np.linspace(SpanMin, SpanMax, N)
            Y = np.zeros(N, dtype=float)
            Z = np.zeros(N, dtype=float)
            GridCoordinates = Node(Parent=self,
                   Name='GridCoordinates', Type='GridCoordinates_t',
                   Children=[Node(Name='CoordinateX', Value=X, Type='DataArray_t'),
                             Node(Name='CoordinateY', Value=Y, Type='DataArray_t'),
                             Node(Name='CoordinateZ', Value=Z, Type='DataArray_t')])

            self.setValue(np.array([[N,N-1,0]],dtype=np.int32,order='F'))

            s = self.abscissa()

            LLx, LLy, LLz = self.xyz()
            SpecialTreatment = ['Airfoils','Span','s']
            Variables2Invoke = [v for v in GeometricalLaws if v not in SpecialTreatment]
            LLDict = self.newFields(Variables2Invoke+['Span','s'], return_type='dict')
            LLDict['Span'][:] = X
            RelSpan = Span/SpanMax

            InterpLaws = {}
            for GeomParam in LLDict:
                if GeomParam in SpecialTreatment: continue
                InterpLaws[GeomParam+'_law']=GeometricalLaws[GeomParam]['InterpolationLaw']
                try: InterpOptions = GeometricalLaws[GeomParam]['InterpolationOptions']
                except KeyError: InterpOptions = dict()
                LLDict[GeomParam][:] = interpolate(RelSpan,
                                        GeometricalLaws[GeomParam]['RelativeSpan'],
                                        GeometricalLaws[GeomParam][GeomParam],
                                        InterpLaws[GeomParam+'_law'],
                                        **InterpOptions)

            LLx[:] = Span
            if 'Sweep' in LLDict:    LLy[:] = -LLDict['Sweep']
            if 'Dihedral' in LLDict: LLz[:] =  LLDict['Dihedral']

            # Add Airfoils node
            if 'RelativeSpan' in GeometricalLaws['Airfoils']:
                AbscissaPolar = interpolate(GeometricalLaws['Airfoils']['RelativeSpan'],
                                                                         RelSpan, s)
                GeometricalLaws['Airfoils']['Abscissa'] = AbscissaPolar
            elif 'Abscissa' in GeometricalLaws['Airfoils']:
                AbscissaPolar = GeometricalLaws['Airfoils']['Abscissa']
            else:
                raise ValueError("Attribute Polars (dict) must contain 'RelativeSpan' or 'Abscissa' key")
            GeometricalLaws['Airfoils']['AirfoilPolarsFilenames'] = AirfoilPolarsFilenames

            self.newFields(['AoA', 'Mach', 'Reynolds']+Polars.DEFAULT_INTERPOLATORS_FIELDS)

            self.setParameters('.Component#Info', kind='LiftingLine',
                                                  MOLAversion=__version__,
                                                  GeometricalLaws=GeometricalLaws)

            if SpanwiseDistribution: self.discretize( **SpanwiseDistribution )

        elif isinstance(SpanMin, list):
            NodeLikeList = SpanMin # first argument is supposed to be a NodeLikeList
            super().__init__(NodeLikeList)
        elif SpanMin is None:
            # is being copied or empty-declarated
            super().__init__(*args, **kwargs)
            return
        else:
            ERRMSG = 'did not recognize first argument SpanMin %s'%type(SpanMin)
            raise TypeError(RED+ERRMSG+ENDC)

        self.constructAttributes()

    def constructAttributes(self):
        AirfoilPolarsFilenames = self.get('AirfoilPolarsFilenames')
        if not AirfoilPolarsFilenames: return
        AirfoilPolarsFilenames = AirfoilPolarsFilenames.value()
        self.setAirfoilPolarInterpolator(AirfoilPolarsFilenames)
        Kinematics = self.childNamed('Kinematics')
        if Kinematics:
            self.RPM = np.atleast_1d(Kinematics.childNamed('RPM').value())
            self.Pitch = np.atleast_1d(Kinematics.childNamed('Pitch').value())
            self.RotationCenter = np.atleast_1d(Kinematics.childNamed('RotationCenter').value())
            self.RotationAxis = np.atleast_1d(Kinematics.childNamed('RotationAxis').value())
            self.VelocityTranslation = np.atleast_1d(Kinematics.childNamed('VelocityTranslation').value())
            self.RightHandRuleRotation = np.atleast_1d(Kinematics.childNamed('RightHandRuleRotation').value())

        else:
            self.RPM = np.array([0.],dtype=np.float64)
            self.Pitch = np.array([0.],dtype=np.float64)
            self.RotationCenter = np.array([0.,0.,0.],dtype=np.float64)
            self.RotationAxis = np.array([0.,0.,0.],dtype=np.float64)
            self.VelocityTranslation = np.array([0.,0.,0.],dtype=np.float64)
            self.RightHandRuleRotation = np.array([1.],dtype=np.int32)
            self.setKinematicsUsingConstantRotationAndTranslation()


    def copy(self, deep=False):
        ValueIsNumpy = isinstance(self[1], np.ndarray)
        ValueCopy = self[1].copy(order='K') if deep and ValueIsNumpy else self[1]
        CopiedNode = self.__class__()
        CopiedNode.setName( self[0] )
        CopiedNode.setValue( ValueCopy )
        CopiedNode.setType( self[3] )
        for child in self[2]: CopiedNode.addChild( child.copy(deep) )
        CopiedNode.constructAttributes()

        return CopiedNode

    def setAirfoilPolarInterpolator(self, FilenamesOrPolarInterpoltorDict,
                             InterpFields=Polars.DEFAULT_INTERPOLATORS_FIELDS):
        if isinstance(FilenamesOrPolarInterpoltorDict, dict):
            self.AirfoilPolarInterpolator = FilenamesOrPolarInterpoltorDict
        else:
            self.AirfoilPolarInterpolator = Polars.loadInterpolators(FilenamesOrPolarInterpoltorDict,
                                                     InterpFields=InterpFields)

    def computeSectionalCoefficients(self):

        InterpFields = Polars.DEFAULT_INTERPOLATORS_FIELDS
        Fields = ['AoA', 'Mach', 'Reynolds', 's']+InterpFields
        DictOfVars = self.fields(Fields, return_type='dict')

        # Get Airfoils data
        PolarInfo= self.childNamed('.Component#Info').childNamed('GeometricalLaws'
                        ).childNamed('Airfoils')
        Abscissa         = PolarInfo.childNamed('Abscissa').value()
        PyZonePolarNames = PolarInfo.childNamed('PyZonePolarNames').value()
        InterpolationLaw = PolarInfo.childNamed('InterpolationLaw').value()

        # Interpolates IntField (Cl, Cd,...) from polars to LiftingLine
        NVars   = len(InterpFields)
        VarArrays = {}
        for IntField in InterpFields:
            VarArrays[IntField] = []

        for PolarName in PyZonePolarNames:
            ListOfVals = self.AirfoilPolarInterpolator[PolarName](DictOfVars['AoA'],
                DictOfVars['Mach'],
                DictOfVars['Reynolds'])

            for i in range(NVars):
                VarArrays[InterpFields[i]] += [ListOfVals[i]]

        for IntField in InterpFields:
            VarArrays[IntField] = np.vstack(VarArrays[IntField])
            Res = interpolate(DictOfVars['s'],Abscissa,VarArrays[IntField],
                              Law=InterpolationLaw, axis=0)
            DictOfVars[IntField][:] = np.diag(Res)

    def getRotationAxisCenterAndDirFromKinematics(self):
        Kinematics_n = self.childNamed('Kinematics')
        if not Kinematics_n:
            raise ValueError('missing "Kinematics" node')


        RotationAxis_n = Kinematics_n.childNamed('RotationAxis')
        if not RotationAxis_n:
            raise ValueError('missing "RotationAxis" node in "Kinematics"')
        RotationAxis = RotationAxis_n.value()

        RotationCenter_n = Kinematics_n.childNamed('RotationCenter')
        if not RotationCenter_n:
            raise ValueError('missing "RotationCenter" node in "Kinematics"')
        RotationCenter = RotationCenter_n.value()

        Dir_n = Kinematics_n.childNamed('RightHandRuleRotation')
        if not Dir_n:
            raise ValueError('missing "RightHandRuleRotation" node in "Kinematics"')
        Dir = 1 if Dir_n.value() else -1

        return RotationAxis, RotationCenter, Dir

    def updateFrame(self):
        RequiredFieldNames = ['tx','ty','tz',
                              'nx','ny','nz',
                              'bx','by','bz',
                              'tanx','tany','tanz']

        tx,ty,tz,nx,ny,nz,bx,by,bz,tanx,tany,tanz = self.fields(RequiredFieldNames)
        x,y,z = self.xyz()
        xyz = np.vstack((x,y,z))
        RotationAxis, RotationCenter, Dir = self.getRotationAxisCenterAndDirFromKinematics()
        norm = np.linalg.norm

        # Compute rotation plane direction
        rvec = (xyz.T - RotationCenter).T
        TangentialDirection = Dir * np.cross(RotationAxis, rvec, axisb=0).T
        tan_norm = np.atleast_1d(norm(TangentialDirection, axis=0))
        pointAlignedWithRotationAxis = np.where(tan_norm==0)[0]
        if pointAlignedWithRotationAxis:
            if len(pointAlignedWithRotationAxis) > 1:
                raise ValueError('several points are aligned with rotation axis')
            if pointAlignedWithRotationAxis > 0:
                TangentialDirection[:,pointAlignedWithRotationAxis] = TangentialDirection[:,pointAlignedWithRotationAxis-1]
                tan_norm[pointAlignedWithRotationAxis] = tan_norm[pointAlignedWithRotationAxis-1]
            else:
                TangentialDirection[:,pointAlignedWithRotationAxis] = TangentialDirection[:,pointAlignedWithRotationAxis+1]
                tan_norm[pointAlignedWithRotationAxis] = tan_norm[pointAlignedWithRotationAxis+1]
        TangentialDirection /= tan_norm
        tanx[:] = TangentialDirection[0,:]
        tany[:] = TangentialDirection[1,:]
        tanz[:] = TangentialDirection[2,:]

        # COMPUTE TANGENTS
        # Central difference O(2)
        txyz = 0.5*(np.diff(xyz[:,:-1],axis=1)+np.diff(xyz[:,1:],axis=1))
        # Uncentered at bounds O(1)
        txyz = np.hstack(((xyz[:,1]-xyz[:,0])[np.newaxis].T,txyz,(xyz[:,-1]-xyz[:,-2])[np.newaxis].T))
        txyz /= norm(txyz, axis=0)
        tx[:] = txyz[0,:]
        ty[:] = txyz[1,:]
        tz[:] = txyz[2,:]

        # Determine normal vectors using Rotation Axis
        RAxyz = np.vstack((RotationAxis[0],RotationAxis[1],RotationAxis[2]))

        # Determine binormal using cross product
        bxyz = np.cross(txyz,RAxyz,axisa=0,axisb=0,axisc=0)
        bxyz /= norm(bxyz, axis=0)
        bx[:] = bxyz[0,:]
        by[:] = bxyz[1,:]
        bz[:] = bxyz[2,:]

        nxyz = np.cross(bxyz,txyz,axisa=0,axisb=0,axisc=0)
        nxyz /= norm(nxyz, axis=0)
        nx[:] = nxyz[0,:]
        ny[:] = nxyz[1,:]
        nz[:] = nxyz[2,:]

        return txyz, nxyz, bxyz

    def setKinematicsUsingConstantRotationAndTranslation(self,
            RotationCenter=[0,0,0], RotationAxis=[0,0,1], RPM=0.0, Pitch=0.,
            RightHandRuleRotation=True, VelocityTranslation=[0,0,0]):
        '''
        This function is a convenient wrap used for setting the ``Kinematics``
        node of **LiftingLine** object.

        .. note:: information contained in ``Kinematics`` node
            is used by :py:func:`moveLiftingLines` and :py:func:`computeKinematicVelocity`
            functions.

        Parameters
        ----------

            LiftingLines : PyTree, base, zone or list of zones
                Container with Lifting lines where ``Kinematics`` node is to be set

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


        self.setRPM(RPM)
        self.setPitch(Pitch)
        self.setRotationCenter(RotationCenter)
        self.setRotationAxis(RotationAxis)
        self.setRightHandRuleRotation(RightHandRuleRotation)
        self.setVelocityTranslation(VelocityTranslation)
        self.setParameters('Kinematics',
                            RotationCenter=self.RotationCenter,
                            RotationAxis=self.RotationAxis,
                            RPM=self.RPM,
                            Pitch=self.Pitch,
                            RightHandRuleRotation=self.RightHandRuleRotation,
                            VelocityTranslation=self.VelocityTranslation,)

    def setConditions(self, VelocityFreestream=[0,0,0], Density=1.225,
                      Temperature=288.15):
        '''
        This function is a convenient wrap used for setting the ``Conditions``
        node of **LiftingLine** object.

        .. note:: information contained in ``Conditions`` node
            is used for computation of Reynolds and Mach number, as well as other
            required input of methods such that Vortex Particle Method.

        Parameters
        ----------

            LiftingLines : PyTree, base, zone or list of zones
                Container with Lifting lines where ``Conditions`` node is to be set

                .. note:: zones contained in **LiftingLines** are modified

            VelocityFreestream : :py:class:`list` of 3 :py:class:`float`
                Components :math:`(x,y,z)` of the freestream velocity, in [m/s].

            Density : float
                air density in [kg/m3]

            Temperature : float
                air temperature in [K]
        '''
        self.setParameters('Conditions',
                  VelocityFreestream=np.array(VelocityFreestream,dtype=float),
                  Density=np.atleast_1d(float(Density)),
                  Temperature=np.atleast_1d(float(Temperature)))

    def assembleAndProjectVelocities(self):
        '''
        This function updates a series of veolicity (and other fields) of
        LiftingLines provided a given kinematic and flow conditions.

        The new or updated fields are the following :

        * ``VelocityX`` ``VelocityY`` ``VelocityZ``
            Three components of the VelocityInduced + VelocityFreestream

        * ``VelocityAxial``
            Relative velocity in -RotationAxis direction

        * ``VelocityTangential``
            Relative velocity in the rotation plane direction

        * ``VelocityNormal2D``
            This is the normal-wise (in ``nx`` ``ny`` ``nz`` direction)
            of the 2D velocity

        * ``VelocityTangential2D``
            This is the tangential (in ``bx`` ``by`` ``bz`` direction)
            of the 2D velocity

        * ``phiRad``
            Angle of the flow with respect to rotation plane as
            ``np.arctan2( VelocityNormal2D, VelocityTangential2D )``

        * ``AoA``
            Local angle-of-attack of the blade section

        * ``VelocityMagnitudeLocal``
            Magnitude of the local velocity neglecting the radial contribution

        * ``Mach``
            Mach number neglecting the radial contribution

        * ``Reynolds``
            Reynolds number neglecting the radial contribution

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
        RequiredFieldNames = ['VelocityKinematicX',
                              'VelocityKinematicY',
                              'VelocityKinematicZ',
                              'VelocityInducedX',
                              'VelocityInducedY',
                              'VelocityInducedZ',
                              'VelocityInducedAxial',
                              'VelocityInducedTangential',
                              'VelocityX',
                              'VelocityY',
                              'VelocityZ',
                              'VelocityAxial',
                              'VelocityTangential',
                              'VelocityNormal2D',
                              'VelocityTangential2D',
                              'VelocityMagnitudeLocal',
                              'Chord','Twist','AoA','phiRad','Mach','Reynolds',
                              'tx','ty','tz',
                              'nx','ny','nz',
                              'bx','by','bz',
                              'tanx','tany','tanz',
                              ]

        Conditions = self.getParameters('Conditions')
        Temperature = Conditions['Temperature']
        Density = Conditions['Density']
        VelocityFreestream = Conditions['VelocityFreestream']
        RotationAxis, RotationCenter, dir = self.getRotationAxisCenterAndDirFromKinematics()
        Mu = computeViscosityMolecular(Temperature)
        SoundSpeed = computeSoundSpeed(Temperature)
        self.updateFrame()
        NPts = self.numberOfPoints()

        f = self.fields(RequiredFieldNames, return_type='dict')
        VelocityKinematic = np.vstack([f['VelocityKinematic'+i] for i in 'XYZ'])
        VelocityInduced = np.vstack([f['VelocityInduced'+i] for i in 'XYZ'])
        TangentialDirection = np.vstack([f['tan'+i] for i in 'xyz'])
        nxyz = np.vstack([f['n'+i] for i in 'xyz'])
        bxyz = np.vstack([f['b'+i] for i in 'xyz'])
        VelocityRelative = (VelocityInduced.T + VelocityFreestream - VelocityKinematic.T).T
        f['VelocityX'][:] = VelocityInduced[0,:] + VelocityFreestream[0]
        f['VelocityY'][:] = VelocityInduced[1,:] + VelocityFreestream[1]
        f['VelocityZ'][:] = VelocityInduced[2,:] + VelocityFreestream[2]
        f['VelocityAxial'][:] = Vax = ( VelocityRelative.T.dot(-RotationAxis) ).T
        f['VelocityTangential'][:] = Vtan = np.diag(VelocityRelative.T.dot( TangentialDirection))
        f['VelocityInducedAxial'][:] = ( VelocityInduced.T.dot(-RotationAxis) ).T
        f['VelocityInducedTangential'][:] = np.diag(VelocityInduced.T.dot( TangentialDirection))

        # note the absence of radial velocity contribution to 2D flow
        V2D = np.vstack((Vax * RotationAxis[0] + Vtan * TangentialDirection[0,:],
                         Vax * RotationAxis[1] + Vtan * TangentialDirection[1,:],
                         Vax * RotationAxis[2] + Vtan * TangentialDirection[2,:]))
        f['VelocityNormal2D'][:] = V2Dn = np.diag( V2D.T.dot( nxyz) )
        f['VelocityTangential2D'][:] = V2Dt = dir * np.diag( V2D.T.dot( bxyz) )
        f['phiRad'][:] = phi = np.arctan2( V2Dn, V2Dt )
        f['AoA'][:] = (f['Twist'] + self.Pitch) - np.rad2deg(phi)
        f['VelocityMagnitudeLocal'][:] = W = np.sqrt( V2Dn**2 + V2Dt**2 )
        # note the absence of radial velocity contribution to Mach and Reynolds
        f['Mach'][:] = W / SoundSpeed
        f['Reynolds'][:] = Density[0] * W * f['Chord'] / Mu

    def computeKinematicVelocity(self):
        '''
        Compute or update ``VelocityKinematicX``, ``VelocityKinematicY`` and
        ``VelocityKinematicZ`` fields of LiftingLines provided to function using
        information contained in ``Kinematics`` node attached to each LiftingLine.

        Parameters
        ----------

            t : Pytree, base, list of zones, zone
                container with LiftingLines

                .. note:: LiftingLines contained in **t** are modified.
        '''

        RequiredFieldNames = ['VelocityKinematicX',
                              'VelocityKinematicY',
                              'VelocityKinematicZ',]

        Kinematics = self.getParameters('Kinematics')
        VelocityTranslation = Kinematics['VelocityTranslation']
        RotationCenter = Kinematics['RotationCenter']
        RotationAxis = Kinematics['RotationAxis']
        RPM = Kinematics['RPM']
        Dir = 1 if Kinematics['RightHandRuleRotation'] else -1
        Omega = RPM*np.pi/30.
        xyz = np.vstack( self.xyz() )
        rvec = xyz - np.atleast_2d(RotationCenter).T
        f = self.fields(RequiredFieldNames, return_type='dict')
        OmegaVector = Dir * Omega * RotationAxis
        VelocityTranslationT = np.atleast_2d(VelocityTranslation).T
        VelocityKinematic = VelocityTranslationT + np.cross(OmegaVector, rvec,
                                                       axisa=0, axisb=0,axisc=0)

        f['VelocityKinematicX'][:] = VelocityKinematic[0,:]
        f['VelocityKinematicY'][:] = VelocityKinematic[1,:]
        f['VelocityKinematicZ'][:] = VelocityKinematic[2,:]

    def computeLoads(self, NumberOfBlades=1.):
        '''
        This function is used to compute local and integral arrays of a lifting line
        with general orientation and shape (including sweep and dihedral).

        .. important:: the flow's contribution to the efforts in the tangential
            direction of the lifting-line is neglected. This means that radial
            contribution on rotors is neglected.

        Parameters
        ----------

            t : PyTree, base, zone, list of zones
                container with Lifting Line zones.
                Each LiftingLine must contain the minimum required fields:

                * ``phiRad``
                    Local angle of the flow in radians

                * ``Cl`` ``Cd`` ``Cm``
                    Local aerodynamic coefficients of the lifting line sections

                * ``Chord``
                    Local chord of the sections

                * ``VelocityMagnitudeLocal``
                    velocity magnitude employed for computing the fluxes, moments
                    and local bound circulation.

                * ``s``
                    curvilinear abscissa

                * ``Span``
                    local span, which is the cylindric distance from **RotationAxis**
                    (information contained in ``.Kinematics`` node) to each section

                Required Frenet fields (if absent, they will be computed):

                .. see also:: :py:func:`updateLocalFrame`

                * ``tx`` ``ty`` ``tz``
                    unitary vector pointing towards the local abscissa direction of
                    the lifting line curve.

                * ``bx`` ``by`` ``bz``
                    unitary vector normal to the local lifting line curve
                    and contained in the rotation plane of the blade.

                * ``nx`` ``ny`` ``nz``
                    unitary vector normal to the local lifting line curve
                    forming a right-hand-rule frame with the aforementioned vectors

                * ``tanx`` ``tany`` ``tanz``
                    unitary vector of the section's local direction
                    tangent to the rotation plane and perpendicular to the rotation
                    axis. This is employed for computing torque and power.

                New fields are created as a result of this function call:

                * ``fx`` ``fy`` ``fz``
                    local linear forces at each lifting line's section
                    in [N/m]. Each component :math:`(x,y,z)` corresponds to absolute
                    coordinate frame (same as ``GridCoordinates``)

                * ``fa`` ``ft``
                    local linear forces projected onto axial and tangential
                    directions. ``fa`` contributes to Thrust. ``ft`` contributes to Torque.
                    They have dimensions of [N/m]

                * ``fn`` ``fb``
                    local linear forces projected onto 2D frame defined by
                    ``nx`` ``ny`` ``nz`` direction and ``bx`` ``by`` ``bz`` direction, respectively.
                    They have dimensions of [N/m]

                * ``mx`` ``my`` ``mz``
                    local linear moments in :math:`(x,y,z)` frame. Dimensions are [N]
                    The moments are applied on 1/4 chord (at LiftingLine's nodes)

                * ``m0x`` ``m0y`` ``m0z``
                    local linear moments in :math:`(x,y,z)` frame applied at
                    rotation center of the blade. Dimensions are [N]

                * ``Lx`` ``Ly`` ``Lz`` ``La`` ``Lt`` ``Ln`` ``Lb``
                    Respectively linear Lift contribution following
                    the directions :math:`(x,y,z)` axial, tangential normal and
                    binormal. [N/m]

                * ``Dx`` ``Dy`` ``Dz`` ``Da`` ``Dt`` ``Dn`` ``Db``
                    linear Drag contribution following
                    the directions :math:`(x,y,z)` axial, tangential normal and
                    binormal. [N/m]

                * ``Gamma``
                    circulation magnitude of the blade section following the
                    Kutta-Joukowski theorem

                * ``GammaX`` ``GammaY`` ``GammaZ``
                    circulation vector of the blade section following the
                    Kutta-Joukowski theorem

                .. note::
                    LiftingLine zones contained in **t** are modified
            NumberOfBlades : float
                Multiplication factor of integral arrays
        '''

        FrenetFields = ['tx','ty','tz','nx','ny','nz','bx','by','bz',
            'tanx','tany','tanz']
        MinimumRequiredFields = ['phiRad','Cl','Cd','Cm','Chord',
            'VelocityMagnitudeLocal','s','Span']
        NewFields = ['fx','fy','fz', 'fa','ft','fn','fb',
            'mx','my','mz',
            'm0x','m0y','m0z',
            'Lx','Ly','Lz','La','Lt','Ln','Lb',
            'Dx','Dy','Dz','Da','Dt','Dn','Db',
            'Gamma',
            'GammaX','GammaY','GammaZ']

        RotationAxis, TorqueOrigin, dir = self.getRotationAxisCenterAndDirFromKinematics()

        Kinematics = self.getParameters('Kinematics')
        TorqueOrigin = RotationCenter = Kinematics['RotationCenter']
        RotationAxis = Kinematics['RotationAxis']
        RightHandRuleRotation = Kinematics['RightHandRuleRotation']
        RPM = self.RPM
        Conditions = self.getParameters('Conditions')
        Temperature = Conditions['Temperature']
        Density = Conditions['Density']
        VelocityFreestream = Conditions['VelocityFreestream']

        f = self.fields(FrenetFields+MinimumRequiredFields+NewFields,return_type={})
        x,y,z = self.xyz()
        xyz = np.vstack((x,y,z))
        rx = x - TorqueOrigin[0]
        ry = y - TorqueOrigin[1]
        rz = z - TorqueOrigin[2]


        # ----------------------- COMPUTE LINEAR FORCES ----------------------- #
        FluxC = 0.5*Density*f['VelocityMagnitudeLocal']**2*f['Chord']
        Lift = FluxC*f['Cl']
        Drag = FluxC*f['Cd']

        f['Ln'][:] = Lift*np.cos(f['phiRad'])
        f['Lb'][:] = Lift*np.sin(f['phiRad'])

        f['Dn'][:] =-Drag*np.sin(f['phiRad'])
        f['Db'][:] = Drag*np.cos(f['phiRad'])

        f['Lx'][:] = f['Ln']*f['nx'] + dir*f['Lb']*f['bx']
        f['Ly'][:] = f['Ln']*f['ny'] + dir*f['Lb']*f['by']
        f['Lz'][:] = f['Ln']*f['nz'] + dir*f['Lb']*f['bz']
        f['Dx'][:] = f['Dn']*f['nx'] + dir*f['Db']*f['bx']
        f['Dy'][:] = f['Dn']*f['ny'] + dir*f['Db']*f['by']
        f['Dz'][:] = f['Dn']*f['nz'] + dir*f['Db']*f['bz']

        f['La'][:] = f['Lx']*RotationAxis[0] + \
                     f['Ly']*RotationAxis[1] + \
                     f['Lz']*RotationAxis[2]

        f['Da'][:] = f['Dx']*RotationAxis[0] + \
                     f['Dy']*RotationAxis[1] + \
                     f['Dz']*RotationAxis[2]

        f['Lt'][:] = f['Lx']*f['tanx'] + \
                     f['Ly']*f['tany'] + \
                     f['Lz']*f['tanz']

        f['Dt'][:] = f['Dx']*f['tanx'] + \
                     f['Dy']*f['tany'] + \
                     f['Dz']*f['tanz']

        f['fa'][:] = f['La'] + f['Da']
        f['ft'][:] = f['Lt'] + f['Dt']

        f['fx'][:] = f['Lx'] + f['Dx']
        f['fy'][:] = f['Ly'] + f['Dy']
        f['fz'][:] = f['Lz'] + f['Dz']

        # ----------------------- COMPUTE LINEAR TORQUE ----------------------- #
        FluxM = FluxC*f['Chord']*f['Cm']
        f['mx'][:] = dir * FluxM * f['tx']
        f['my'][:] = dir * FluxM * f['ty']
        f['mz'][:] = dir * FluxM * f['tz']
        f['m0x'][:] = f['mx'] + ry*f['fz'] - rz*f['fy']
        f['m0y'][:] = f['my'] + rz*f['fx'] - rx*f['fz']
        f['m0z'][:] = f['mz'] + rx*f['fy'] - ry*f['fx']

        # Compute linear bound circulation using Kutta-Joukowski
        # theorem:  Lift = Density * ( Velocity x Gamma )
        w = f['VelocityMagnitudeLocal']
        FluxKJ = Lift/Density
        Flowing = abs(w)>0
        FluxKJ[Flowing] /= w[Flowing]
        FluxKJ[~Flowing] = 0.
        f['GammaX'][:] = dir * FluxKJ * f['tx']
        f['GammaY'][:] = dir * FluxKJ * f['ty']
        f['GammaZ'][:] = dir * FluxKJ * f['tz']
        f['Gamma'][:] = FluxKJ
        # ------------------------- INTEGRAL LOADS ------------------------- #
        DimensionalAbscissa = self.length() * self.abscissa()

        # Integrate linear axial force <fa> to get Thrust
        FA = Thrust = sint.simps(f['fa'], DimensionalAbscissa)
        FT =          sint.simps(f['ft'], DimensionalAbscissa)
        FX =          sint.simps(f['fx'], DimensionalAbscissa)
        FY =          sint.simps(f['fy'], DimensionalAbscissa)
        FZ =          sint.simps(f['fz'], DimensionalAbscissa)
        MX =          -sint.simps(f['m0x'], DimensionalAbscissa)
        MY =          -sint.simps(f['m0y'], DimensionalAbscissa)
        MZ =          -sint.simps(f['m0z'], DimensionalAbscissa)

        # # Integrate tangential moment <ft>*Span to get Power
        # Torque = sint.simps(f['ft']*f['Span'],DimensionalAbscissa) # equivalent
        Torque = MX*RotationAxis[0]+MY*RotationAxis[1]+MZ*RotationAxis[2]
        Power  = dir*(RPM*np.pi/30.)*Torque


        # Store computed integral Loads
        IntegralData = self.setParameters('.Loads',
                                          Thrust=NumberOfBlades*Thrust,
                                          Power=NumberOfBlades*Power,
                                          Torque=NumberOfBlades*Torque,
                                          ForceTangential=NumberOfBlades*FT,
                                          ForceX=NumberOfBlades*FX,
                                          ForceY=NumberOfBlades*FY,
                                          ForceZ=NumberOfBlades*FZ,
                                          TorqueX=NumberOfBlades*MX,
                                          TorqueY=NumberOfBlades*MY,
                                          TorqueZ=NumberOfBlades*MZ)

        return IntegralData

    def setRotationAxis(self, newRotationAxis):
        self.RotationAxis[:] = newRotationAxis

    def setRotationCenter(self, newRotationCenter):
        self.RotationCenter[:] = newRotationCenter

    def setVelocityTranslation(self, newVelocityTranslation):
        self.VelocityTranslation[:] = newVelocityTranslation

    def setRPM(self, newRPM): self.RPM[0] = newRPM

    def setPitch(self, newPitch): self.Pitch[0] = newPitch

    def setRightHandRuleRotation(self, newRightHandRuleRotation):
        self.RightHandRuleRotation[0] = newRightHandRuleRotation

    def computeTipLossFactor(self, NumberOfBlades, model='Adkins'):
        r, phi = self.fields(['Span','phiRad'])
        Rmax = r.max()
        xi = r/Rmax
        # phiEff avoids possible overflow (division by zero)
        f_max = 100.0
        a = (NumberOfBlades/2.)*(1-xi)
        phi_min = np.arctan(a/(xi*f_max))
        phiEff = np.maximum(np.abs(phi),np.maximum(phi_min,1e-4))
        if model == 'Adkins':
            # This version is also used by M. Drela
            f = a/(xi*np.tan(phiEff))
            f = np.maximum(f,0)
            F = (2./np.pi)*np.arccos(np.minimum(np.exp(-f),1))

        elif model == 'Glauert':
            f = a/(xi * np.sin(phiEff))
            F = (2./np.pi)*np.arccos(np.minimum(np.exp(-f),1))

        elif model == 'Prandtl':
            Omega = self.RPM * np.pi / 30.
            VelocityFreestream = self.childNamed('Conditions').childNamed(
                'VelocityFreestream').value()
            Velocity = np.linalg.norm(-VelocityFreestream.dot(self.RotationAxis))
            f = a*np.sqrt(1-(Omega*r/Velocity)**2)
            F = (2./np.pi)*np.arccos(np.minimum(np.exp(-f),1))
        else:
            raise ValueError('TipLosses=%s not recognized.'%kind)

        return F

    def setOptimumAngleOfAttack(self,
            Aim='Cl', AimValue=0.5, AoASearchBounds=(-2,6),
            SpecificSections=None):
        """
        Update ``AoA`` field with the optimum angle of attack based on a given
        **Aim**, using the provided **PolarsInterpolatorDict** as well as the existing
        ``Reynolds`` and ``Mach`` number values contained in ``FlowSolution`` container.

        Parameters
        ----------

            LiftingLine : zone
                the lifting line where ``AoA`` field will be updated

                .. note:: zone **LiftingLine** is modified

            PolarsInterpolatorDict : dict
                dictionary of
                interpolator functions of 2D polars, as obtained from
                :py:func:`buildPolarsInterpolatorDict` function.

            Aim : str
                can be one of:

                * ``'Cl'``
                    aims a requested ``Cl`` (:math:`c_l`) value (provided by argument
                    **AimValue**) throughout the entire lifting line

                * ``'minCd'``
                    aims the minimum ``Cd`` value, :math:`\min (c_d)`

                * ``'maxClCd'``
                    aims the maximum ``Cl/Cd`` value, :math:`\max (c_l / c_d)`

            AimValue : float
                Specifies the aimed value for corresponding relevant
                Aim types.

                .. note:: currently, only relevant for **Aim** = ``'Cl'``

            AoASearchBounds : :py:class:`tuple` of 2 :py:class:`float`
                Since there may exist multiple angle-of-attack (*AoA*) values
                verifying the requested conditions, this argument constraints the
                research interval of angle-of-attack of valid candidates.

            SpecificSections : :py:class:`list` of :py:class:`int`
                If specified (not :py:obj:`None`), only the
                indices corresponding to the user-provided sections are updated.

            ListOfEquations : :py:class:`list` of :py:class:`str`
                list of equations compatible with
                the syntax allowed in :py:func:`Converter.PyTree.initVars`
                (``FlowSolution`` located at vertex), in order to tune or correct
                the aerodynamic coefficients.
        """


        AoA, Cl, Cd, Mach, Reynolds = self.fields(['AoA','Cl','Cd','Mach', 'Reynolds'])

        if SpecificSections is None: SpecificSections = range(len(AoA))

        if Aim == 'Cl':
            for i in SpecificSections:
                def searchAoA(x,i):
                    AoA[i] = x
                    self.computeSectionalCoefficients()
                    Residual = Cl[i]-AimValue
                    return Residual

                sol=so.root_scalar(searchAoA, bracket=AoASearchBounds, x0=AoA[i],
                        args=(i),  method='toms748')

                if sol.converged:
                    searchAoA(sol.root,i)
                else:
                    print(WARN+"Not found optimum AoA at section %d"%i)
                    print(sol)
                    print(ENDC)
                    continue
        elif Aim == 'maxClCd':
            for i in SpecificSections:
                def searchAoA(x,i):
                    AoA[i] = x
                    self.computeSectionalCoefficients()
                    MinimizeThis = -Cl[i]/Cd[i]
                    return MinimizeThis

                sol=so.minimize_scalar(searchAoA, bracket=[0,2], args=(i),
                            method='Golden', options=dict(xtol=0.01))


                if not sol.success:
                    print(WARN+"Not found optimum AoA at section %d"%i)
                    print(sol)
                    print(ENDC)
                    continue
        elif Aim == 'minCd':
            for i in SpecificSections:
                def searchAoA(x,i):
                    AoA[i] = x
                    self.computeSectionalCoefficients()
                    MinimizeThis = Cd[i]
                    return MinimizeThis

                sol=so.minimize_scalar(searchAoA, bracket=AoASearchBounds,
                            args=(i),  method='Golden', options=dict(xtol=0.01))

                if not sol.success:
                    print(WARN+"Not found optimum AoA at section %d"%i)
                    print(sol)
                    print(ENDC)
                    continue

    def resetTwist(self, ZeroPitchRelativeSpan=0.75, modifyLiftingLine=True):
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
        r, Twist = self.fields(['Span','Twist'])
        DeltaTwist = interpolate(np.array([ZeroPitchRelativeSpan]), r/r.max(), Twist)
        if modifyLiftingLine: Twist -= DeltaTwist
        self.setPitch(DeltaTwist)

        return DeltaTwist

    def computeAxialSpeed(self):
        Kinematics = self.getParameters('Kinematics')
        Conditions = self.getParameters('Conditions')
        VelocityTranslation = Kinematics['VelocityTranslation']
        VelocityFreestream = Conditions['VelocityFreestream']
        RotationAxis = Kinematics['RotationAxis']
        AxialSpeed = (VelocityTranslation-VelocityFreestream).dot(RotationAxis)
        
        return AxialSpeed

    def addPropellerLoads(self):
        n = self.RPM/60.
        loads = self.getParameters('.Loads')
        if not loads:
            raise ValueError('must call computeLoads before addPropellerLoads')
        AdvanceVelocityNorm = self.computeAxialSpeed()
        Span = self.fields('Span')
        d = 2*Span.max() 
        Conditions = self.getParameters('Conditions')
        CTpropeller = loads['Thrust'] / (Conditions['Density'] * n**2 * d**4)
        CPpropeller = loads['Power']  / (Conditions['Density'] * n**3 * d**5)
        Jparam = AdvanceVelocityNorm / (n*d)
        FigureOfMerit = np.sqrt(2./np.pi)* np.maximum(CTpropeller,0)**1.5 / \
                                           np.maximum(CPpropeller,1e-12)
        PropEff = AdvanceVelocityNorm*loads['Thrust']/np.abs(loads['Power'])
        loads['CTpropeller']=CTpropeller
        loads['CPpropeller']=CPpropeller
        loads['Jparam']=Jparam
        loads['FigureOfMerit']=FigureOfMerit
        loads['PropulsiveEfficiency']=PropEff

        IntegralData = self.setParameters('.Loads',**loads)

        return IntegralData

    def addHelicopterRotorLoads(self):
        Omega = self.RPM*np.pi/30.
        loads = self.getParameters('.Loads')
        if not loads:
            raise ValueError('must call computeLoads before addHelicopterRotorLoads')
        Span = self.fields('Span')
        R = 2*Span.max() 
        Conditions = self.getParameters('Conditions')
        T = loads['Thrust']
        P = loads['Power']
        rho = Conditions['Density']
        CT = 2*T/(rho*(Omega*R)**2*np.pi*R**2)
        CP = 2*P/(rho*(Omega*R)**3*np.pi*R**2)
        FigureOfMeritHeli = 1./np.sqrt(2.)/CP * CT**1.5
        loads['CTheli']=CT
        loads['CPheli']=CP
        loads['FigureOfMeritHeli']=FigureOfMeritHeli

        IntegralData = self.setParameters('.Loads',**loads)

        return IntegralData


def computeViscosityMolecular(Temperature):
    # TODO replace with atmospheric module
    Mus, Cs, Ts= 1.711e-5, 110.4, 273.0 # Sutherland const.
    return Mus*((Temperature/Ts)**0.5)*((1.+Cs/Ts)/(1.+Cs/Temperature))

def computeSoundSpeed(Temperature):
    # TODO replace with atmospheric module
    Gamma, Rgp = 1.4, 287.058
    return np.sqrt(Gamma * Rgp * Temperature)
