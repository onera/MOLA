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

            self.setValue(np.array([[N,N-1,0]],dtype=np.int,order='F'))

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

        AirfoilPolarsFilenames = self.get('AirfoilPolarsFilenames').value()
        self.setAirfoilPolarInterpolator(AirfoilPolarsFilenames)

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
        Kinematics_n = self.childNamed('.Kinematics')
        if not Kinematics_n:
            raise ValueError('missing ".Kinematics" node')


        RotationAxis_n = Kinematics_n.childNamed('RotationAxis')
        if not RotationAxis_n:
            raise ValueError('missing "RotationAxis" node in ".Kinematics"')
        RotationAxis = RotationAxis_n.value()

        RotationCenter_n = Kinematics_n.childNamed('RotationCenter')
        if not RotationCenter_n:
            raise ValueError('missing "RotationCenter" node in ".Kinematics"')
        RotationCenter = RotationCenter_n.value()

        Dir_n = Kinematics_n.childNamed('RightHandRuleRotation')
        if not Dir_n:
            raise ValueError('missing "RightHandRuleRotation" node in ".Kinematics"')
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
            RotationCenter=[0,0,0], RotationAxis=[0,0,1], RPM=2500.0,
            RightHandRuleRotation=True, VelocityTranslation=[0,0,0]):
        '''
        This function is a convenient wrap used for setting the ``.Kinematics``
        node of **LiftingLine** object.

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

        self.setParameters('.Kinematics',
                RotationCenter=np.array(RotationCenter,dtype=np.float),
                RotationAxis=np.array(RotationAxis,dtype=np.float),
                RPM=np.atleast_1d(np.array(RPM,dtype=np.float)),
                RightHandRuleRotation=np.atleast_1d(np.array(RightHandRuleRotation,dtype=np.int32)),
                VelocityTranslation=np.array(VelocityTranslation,dtype=np.float),)

    def setConditions(self, VelocityFreestream=[0,0,0], Density=1.225,
                      Temperature=288.15):
        '''
        This function is a convenient wrap used for setting the ``.Conditions``
        node of **LiftingLine** object.

        .. note:: information contained in ``.Conditions`` node
            is used for computation of Reynolds and Mach number, as well as other
            required input of methods such that Vortex Particle Method.

        Parameters
        ----------

            LiftingLines : PyTree, base, zone or list of zones
                Container with Lifting lines where ``.Conditions`` node is to be set

                .. note:: zones contained in **LiftingLines** are modified

            VelocityFreestream : :py:class:`list` of 3 :py:class:`float`
                Components :math:`(x,y,z)` of the freestream velocity, in [m/s].

            Density : float
                air density in [kg/m3]

            Temperature : float
                air temperature in [K]
        '''
        self.setParameters('.Conditions',
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

        Conditions = self.getParameters('.Conditions')
        Temperature = Conditions['Temperature']
        Density = Conditions['Density']
        VelocityFreestream = Conditions['VelocityFreestream']
        RotationAxis, RotationCenter, dir = self.getRotationAxisCenterAndDirFromKinematics()
        Mu = computeViscosityMolecular(Temperature)
        SoundSpeed = computeSoundSpeed(Temperature)
        self.updateFrame()
        NPts = self.numberOfPoints()

        v = self.fields(RequiredFieldNames, return_type='dict')
        VelocityKinematic = np.vstack([v['VelocityKinematic'+i] for i in 'XYZ'])
        VelocityInduced = np.vstack([v['VelocityInduced'+i] for i in 'XYZ'])
        TangentialDirection = np.vstack([v['tan'+i] for i in 'xyz'])
        nxyz = np.vstack([v['n'+i] for i in 'xyz'])
        bxyz = np.vstack([v['b'+i] for i in 'xyz'])
        VelocityRelative = (VelocityInduced.T + VelocityFreestream - VelocityKinematic.T).T
        v['VelocityX'][:] = VelocityInduced[0,:] + VelocityFreestream[0]
        v['VelocityY'][:] = VelocityInduced[1,:] + VelocityFreestream[1]
        v['VelocityZ'][:] = VelocityInduced[2,:] + VelocityFreestream[2]
        v['VelocityAxial'][:] = Vax = ( VelocityRelative.T.dot(-RotationAxis) ).T
        v['VelocityTangential'][:] = Vtan = np.diag(VelocityRelative.T.dot( TangentialDirection))
        # note the absence of radial velocity contribution to 2D flow
        V2D = np.vstack((Vax * RotationAxis[0] + Vtan * TangentialDirection[0,:],
                         Vax * RotationAxis[1] + Vtan * TangentialDirection[1,:],
                         Vax * RotationAxis[2] + Vtan * TangentialDirection[2,:]))
        v['VelocityNormal2D'][:] = V2Dn = np.diag( V2D.T.dot( nxyz) )
        v['VelocityTangential2D'][:] = V2Dt = dir * np.diag( V2D.T.dot( bxyz) )
        v['phiRad'][:] = phi = np.arctan2( V2Dn, V2Dt )
        v['AoA'][:] = v['Twist'] - np.rad2deg(phi)
        v['VelocityMagnitudeLocal'][:] = W = np.sqrt( V2Dn**2 + V2Dt**2 )
        # note the absence of radial velocity contribution to Mach and Reynolds
        v['Mach'][:] = W / SoundSpeed
        v['Reynolds'][:] = Density[0] * W * v['Chord'] / Mu

    def computeKinematicVelocity(self):
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

        Kinematics = self.getParameters('.Kinematics')
        VelocityTranslation = Kinematics['VelocityTranslation']
        RotationCenter = Kinematics['RotationCenter']
        RotationAxis = Kinematics['RotationAxis']
        RPM = Kinematics['RPM']
        Dir = 1 if Kinematics['RightHandRuleRotation'] else -1
        Omega = RPM*np.pi/30.
        x,y,z = self.xyz()
        v = self.fields(RequiredFieldNames, return_type='dict')
        NPts = len(x)
        # TODO vectorize this
        for i in range(NPts):
            rvec = np.array([x[i] - RotationCenter[0],
                             y[i] - RotationCenter[1],
                             z[i] - RotationCenter[2]],dtype=np.float)

            VelocityKinematic = np.cross( Dir * Omega * RotationAxis, rvec) + VelocityTranslation

            v['VelocityKinematicX'][i] = VelocityKinematic[0]
            v['VelocityKinematicY'][i] = VelocityKinematic[1]
            v['VelocityKinematicZ'][i] = VelocityKinematic[2]


def computeViscosityMolecular(Temperature):
    # TODO replace with atmospheric module
    Mus, Cs, Ts= 1.711e-5, 110.4, 273.0 # Sutherland const.
    return Mus*((Temperature/Ts)**0.5)*((1.+Cs/Ts)/(1.+Cs/Temperature))

def computeSoundSpeed(Temperature):
    # TODO replace with atmospheric module
    Gamma, Rgp = 1.4, 287.058
    return np.sqrt(Gamma * Rgp * Temperature)
