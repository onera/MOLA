'''
Main subpackage for Blade-Element Momentum Theory (BEMT) computations

13/06/2022 - L. Bernardos - first creation
'''

from .. import load as file_load
from ..Core import np,RED,GREEN,WARN,PINK,CYAN,ENDC,interpolate,secant
from ..LiftingLine import (LiftingLine, Polars,
                           computeViscosityMolecular, computeSoundSpeed)
import scipy.optimize as so
import scipy.integrate as sint
import scipy.signal as ss

def compute(LL, NumberOfBlades=2, RPM=1500.,
        Pitch=0., AxialVelocity=0., Density=1.225, Temperature=288.15,
        model='Drela', TipLossesModel='Adkins',
        TrimQuantity=None, TrimCommand='Pitch', TrimValue=0.,
        TrimValueTolerance=0.1,
        TrimCommandGuess=[[0.8,1.2],[0.5,1.5],[0.25,1.75]]):

    RotationAxis, RotationCenter, Dir = LL.getRotationAxisCenterAndDirFromKinematics()
    Velocity = - RotationAxis * AxialVelocity
    AdvanceVelocityNorm = np.linalg.norm(Velocity)
    LL.setConditions(VelocityFreestream=Velocity, Density=Density,
                     Temperature=Temperature)
    LL.setPitch(Pitch)
    LL.setRPM(RPM)
    LL.updateFrame()
    LL.computeKinematicVelocity()
    LL.assembleAndProjectVelocities()
    NPts = LL.numberOfPoints()

    commandDict = {'Pitch':Pitch,'RPM':RPM}

    NewFields = ['VelocityKinematicAxial','VelocityKinematicTangential','psi']

    f = LL.fields(NewFields)
    f = LL.allFields()
    TangentialDirection = np.vstack([f['tan'+i] for i in 'xyz'])
    f['VelocityKinematicAxial'][:] = f['VelocityAxial'][:]
    f['VelocityKinematicTangential'][:] = -f['VelocityTangential'][:]
    nxyz = np.array([f['nx'],f['ny'],f['nz']], dtype=np.float64, order='F')
    bxyz = np.array([f['bx'],f['by'],f['bz']], dtype=np.float64, order='F')


    Mu = computeViscosityMolecular(Temperature)
    SoundSpeed = computeSoundSpeed(Temperature)
    sigma = NumberOfBlades * f['Chord'] / (2*np.pi*f['Span']) # Blade solidity
    Rmax = f['Span'].max()
    d = 2*Rmax  # diameter

    def modelHeene(vi, SectionIndex):
        i = SectionIndex

        VkAxial = f['VelocityKinematicAxial'][i]
        VkTan = f['VelocityKinematicTangential'][i]

        via = f['VelocityInducedAxial'][i]      = vi[0]
        vit = f['VelocityInducedTangential'][i] = vi[1]


        f['VelocityAxial'][i]      = Vax  =  VkAxial + via
        f['VelocityTangential'][i] = Vtan = -VkTan   + vit

        V2D = Vax * LL.RotationAxis  +  Vtan * TangentialDirection[:,i]

        f['VelocityNormal2D'][i]     = V2Dn = V2D.dot( nxyz[:,i] )
        f['VelocityTangential2D'][i] = V2Dt = V2D.dot( bxyz[:,i] )
        f['phiRad'][i] = phi = np.arctan2( V2Dn, V2Dt )
        f['AoA'][i] = (f['Twist'][i] + LL.Pitch) - np.rad2deg(phi)

        # NB: W == Vp (Fig 2.2)
        f['VelocityMagnitudeLocal'][i] = W = np.sqrt( V2Dn**2 + V2Dt**2 )

        f['Mach'][i] = W / SoundSpeed
        f['Reynolds'][i] = Density * W * f['Chord'][i] / Mu

        # Compute tip losses factor, F
        F = LL.computeTipLossFactor(NumberOfBlades, model=TipLossesModel)

        LL.computeSectionalCoefficients()

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cL = f['Cl'][i]
        cD = f['Cd'][i]
        Cy = cL * cos_phi - cD * sin_phi
        Cx = cL * sin_phi + cD * cos_phi

        # Non-linear functions f1, f2, to solve (Eqns 2.38-2.39)
        f1 = 0.5*sigma[i]*W**2*Cy - 2*Vax*via*F[i]
        f2 = 0.5*sigma[i]*W**2*Cx - 2*Vax*vit*F[i]

        Residual = [f1, f2]

        return Residual

    def modelDrela(x,SectionIndex):
        # Current section index is variable <i>
        i = SectionIndex
        # Drela's dummy variable (Fig.4) (radians)
        f['psi'][i]= psi = x


        # Apply Drela's Eqns (17--27)
        ua = ut = 0 # TODO: Extend for multi-propellers
        Ua = AxialVelocity + ua
        Omega = LL.RPM*np.pi/30.
        Ut = Omega * f['Span'][i] - ut
        U = np.sqrt(Ua**2+Ut**2)
        f['VelocityAxial'][i] = Wa = 0.5 * (Ua + U * np.sin(psi))
        f['VelocityTangential'][i] = Wt = 0.5 * (Ut + U * np.cos(psi))
        f['phiRad'][i] = phi = np.arctan2(Wa,Wt) # (useful to store)
        f['VelocityInducedAxial'][i] = va = Wa - Ua
        f['VelocityInducedTangential'][i] = vt = Ut - Wt
        f['AoA'][i] = (f['Twist'][i] + LL.Pitch) - np.rad2deg(phi)

        f['VelocityMagnitudeLocal'][i] = W = np.sqrt(Wa**2 + Wt**2)
        f['Reynolds'][i] = Density * W * f['Chord'][i] / Mu
        f['Mach'][i] = W / SoundSpeed

        # Compute tip losses factor, F
        F = LL.computeTipLossFactor(NumberOfBlades, model=TipLossesModel)

        LL.computeSectionalCoefficients()

        # Local wake advance ratio (Eqn.9)
        r = f['Span'][i]
        lambda_w = (r/Rmax)*Wa/Wt

        # Total circulation from Momentum theory
        # (Eqn.31)
        GammaMom = np.sign(Wa) * vt * (4*np.pi*r/NumberOfBlades) \
            *F[i]*np.sqrt(1+ (4*lambda_w*Rmax/(np.pi*NumberOfBlades*r))**2)

        # Total circulation from Blade Element theory
        # (Eqn.16)
        GammaBE  = 0.5 * W * f['Chord'][i] * f['Cl'][i]

        # Both circulations shall be equal (Eqn.32)
        Residual = GammaMom - GammaBE

        return Residual

    def psi2AoAFirstSection(psi):
        Ua = AxialVelocity
        Omega = RPM*np.pi/30.
        Ut = Omega * f['Span'][0]
        U = np.sqrt(Ua**2+Ut**2)
        Wa = 0.5 * (Ua + U * np.sin(psi))
        Wt = 0.5 * (Ut + U * np.cos(psi))
        phi = np.arctan(Wa/Wt)
        AoA = (f['Twist'][0]+LL.Pitch) - np.rad2deg(phi)
        return AoA


    def includePropellerLoads(DictOfIntegralData):
        n = LL.RPM/60.
        CTpropeller = DictOfIntegralData['Thrust'] / (Density * n**2 * d**4)
        CPpropeller = DictOfIntegralData['Power']  / (Density * n**3 * d**5)
        Jparam = AdvanceVelocityNorm / (n*d)
        FigureOfMeritPropeller = np.sqrt(2./np.pi)* np.maximum(CTpropeller,0)**1.5 / np.maximum(CPpropeller,1e-12)
        PropEff = AdvanceVelocityNorm*DictOfIntegralData['Thrust']/np.abs(DictOfIntegralData['Power'])
        DictOfIntegralData['CTpropeller']=CTpropeller
        DictOfIntegralData['CPpropeller']=CPpropeller
        DictOfIntegralData['Jparam']=Jparam
        DictOfIntegralData['FigureOfMeritPropeller']=FigureOfMeritPropeller
        DictOfIntegralData['PropulsiveEfficiency']=PropEff


    # These flags will be further used for fast conditioning
    ModelIsDrela = ModelIsHeene = False
    if model == 'Heene':
        IterationVariables = ['VelocityInducedAxial','VelocityInducedTangential']
        ModelIsHeene = True
        BEMTsolver = modelHeene
        # Set initial guess:
        f['VelocityInducedAxial'][:] = 1.0
        f['VelocityInducedTangential'][:] = 0.0 * f['VelocityInducedAxial'][:]
    elif model == 'Drela':
        IterationVariables = ['psi']
        ModelIsDrela = True
        BEMTsolver = modelDrela
        # Set initial guess:
        f['psi'][:] = 0.0

    else:
        raise AttributeError(RED+'Attribute model="%s" not recognized.'%model+ENDC)
    Nunk = len(IterationVariables)

    def singleShot__(cmd):

        if TrimCommand == 'Pitch':
            cmd = np.clip(cmd,-90,+90)
            LL.setPitch(cmd)
        elif TrimCommand == 'RPM':
            LL.setRPM(cmd)
        else:
            raise AttributeError("TrimCommand '%s' not recognized. Please use 'Pitch' or 'RPM'."%TrimCommand)

        # Initial guess calculation
        v_Pred = np.zeros(Nunk,dtype=np.float64)
        v_Corr = np.zeros(Nunk,dtype=np.float64)
        for i,vn in enumerate(IterationVariables):
            v_Pred[i]=f[vn][0]

        FirstSection = True
        for i in range(NPts-1):
            # predict Guess
            for iu in range(1,Nunk): v_Pred[iu] = f[IterationVariables[iu]][i-1]

            # correct Guess
            Guess = v_Pred + v_Corr

            if ModelIsHeene:
                # Solve the 2-eqn non-linear system
                sol=so.root(BEMTsolver,Guess, args=(i), method='hybr')
                success = sol.success
                root = sol.x

            elif ModelIsDrela:
                if FirstSection:
                    # Override guess based on psi(AoA=0)
                    sol = secant(psi2AoAFirstSection,x0=0.5,x1=1.0, ftol=1.e-07,
                            bounds=(-np.pi/2.,+np.pi/2.), maxiter=50)
                    Guess=sol['root'] if sol['converged'] else 1.

                # Solve the non-linear 1-eqn
                EpsStep = 2.*np.pi/180.
                sol = secant(BEMTsolver,x0=Guess-EpsStep,x1=Guess+EpsStep,
                            ftol=1.e-07, bounds=(-np.pi/2.,+np.pi/2.),
                            maxiter=50,args=(i,))
                success = sol['converged']
                root = sol['root']

            # Compute correctors
            for iu in range(Nunk):
                v_Corr[iu] = f[IterationVariables[iu]][i] - v_Pred[iu]

            BEMTsolver(root,i)

            FirstSection = False

        # Extrapole last section's unknowns (singularity)
        r = f['Span']
        for vn in IterationVariables:
            Slope = (f[vn][-2]-f[vn][-3])/(r[-2]-r[-3])
            Shift = f[vn][-2] - Slope*r[-2]
            f[vn][-1] = Slope*r[-1] + Shift

        tipsol = [f[vn][-1] for vn in IterationVariables]
        if len(tipsol)==1: tipsol = tipsol[0]

        # Solve last section
        BEMTsolver(tipsol,NPts-1)

        # Compute the arrays
        DictOfIntegralData = LL.computeLoads(NumberOfBlades=NumberOfBlades)
        includePropellerLoads(DictOfIntegralData)

        if TrimQuantity:
            print('command %s = %g , current %s=%g , objective %s=%g'%(TrimCommand,
                cmd, TrimQuantity, DictOfIntegralData[TrimQuantity],
                TrimQuantity, TrimValue))
            return DictOfIntegralData[TrimQuantity]-TrimValue
        else:
            return 0.


    if TrimQuantity:
        # Trim attempts
        success = False
        BestCmd = np.array([0.])
        CurrentAccuracy = TrimValueTolerance*100.
        for attempt in range(len(TrimCommandGuess)):
            print('Trim attempt: %d'%attempt)
            bounds = commandDict[TrimCommand]*np.array(TrimCommandGuess[attempt])
            wg0 = 1./3.
            wg1 = 2./3.
            x0  = (1-wg0)*bounds[0] + wg0*bounds[1]
            x1  = (1-wg1)*bounds[0] + wg1*bounds[1]
            TrimSol = secant(singleShot__, ftol=TrimValueTolerance, x0=x0, x1=x1,
                                           bounds=bounds, maxiter=20)

            if TrimSol['converged']:
                success = True
                BestCmd[0] = TrimSol['root']
                break

            if TrimSol['froot']<CurrentAccuracy:
                BestCmd[0]      = TrimSol['root']
                CurrentAccuracy = TrimSol['froot']

        singleShot__(BestCmd[0])

        if TrimCommand == 'RPM':
            LL.setRPM(BestCmd[0])
        elif TrimCommand == 'Pitch':
            LL.setPitch(BestCmd[0])

    else:
        attempt = 0
        singleShot__(commandDict[TrimCommand])

        success = True

    LL.computeSectionalCoefficients()
    DictOfIntegralData = LL.computeLoads(NumberOfBlades=NumberOfBlades)
    DictOfIntegralData['Converged'] = success
    DictOfIntegralData['Attempts']  = attempt
    DictOfIntegralData['Pitch']     = LL.Pitch
    DictOfIntegralData['RPM']       = LL.RPM
    includePropellerLoads(DictOfIntegralData)

    return DictOfIntegralData



def design(LL, NumberOfBlades=2, RPM=1500., AxialVelocity=0., Density=1.225,
              Temperature=288.15, TipLossesModel='Adkins',
              Constraint='Thrust', ConstraintValue=1000.,
              AirfoilAim='maxClCd', AimValue=None, AoASearchBounds=(-2,8),
              SmoothAoA=False, itMaxAoAsearch=3):
    '''
    This function performs a minimum induced loss design of a propeller
    in axial flight conditions.

    Parameters
    ----------

        LiftingLine : zone
            lifting-line used for design (values of chord and
            twist can be arbitrary).

            .. note:: this **LiftingLine** will be modified as a result of design

        PolarsInterpolatorDict : dict
            dictionary of interpolator functions of 2D polars, as obtained from
            :py:func:`MOLA.LiftingLine.buildPolarsInterpolatorDict`

        NBlades : int
            the number of blades of the design candidate

        Velocity : float
            the advance velocity of the propeller [m/s]

        RPM : float
            the rotational velocity of the propeller [rev / minute]

        Temperature : float
            the air temperature [K]

        Density : float
            the air density [kg / m3]

        TipLosses : str
            the model of tip-losses. Can be one of:
            ``'Adkins'``, ``'Glauert'`` or ``'Prandtl'``.

            .. note:: we recommend using ``'Adkins'``.

        ListOfEquations : str
            :py:func:`Converter.PyTree.initVars`-compatible equations to be applied
            after each lifting-line's polar interpolation operation.

        Constraint : str
            constraint type for the optimization. Can be: ``'Thrust'`` or ``'Power'``

        ConstraintValue : float
            the value of the constraint for the design

        Aim : str
            can be one of:

            * ``'Cl'``
                aims a given ``Cl``, :math:`c_l` , value (provided by argument **AimValue**)
                throughout the entire lifting line

            * ``'minCd'``
                aims the minimum ``Cd``, :math:`\min(c_d)` , value

            * ``'maxClCd'``
                aims the maximum aero efficiency :math:`\max(c_l/c_d)` value

        AimValue : float
            Specifies the aimed value for corresponding relevant **Aim** types
            (e.g. ``'Cl'``).

        AoASearchBounds : :py:class:`tuple` of 2 :py:class:`float`
            As there may exist multiple AoA values
            verifying the requested conditions, this argument constraints the
            research interval of angle-of-attack of valid candidates.

        SmoothAoA : bool
            if :py:obj:`True`, perform a post-smoothing of angle of
            attack spanwise distribution after optimization.

            .. tip:: this is useful in order to avoid "shark-teeth" shaped
                distribution.

        itMaxAoAsearch : int
            maximum number of iterations allowed for reaching the desired AoA
            distribution type

    Returns
    -------

        DictOfIntegralData : dict
            dictionary containing LiftingLine arrays.

            .. note:: for design result, please refer to input argument
                **LiftingLine**, which is modified (``Chord`` and ``Twist``
                values are updated).
    '''

    AoA = LL.fields('AoA')
    if AirfoilAim == 'AoA': AoA_imposed = np.copy(AoA)

    RotationAxis, RotationCenter, Dir = LL.getRotationAxisCenterAndDirFromKinematics()
    VelocityVector = - RotationAxis * AxialVelocity
    Velocity = AxialVelocity
    LL.setConditions(VelocityFreestream=VelocityVector, Density=Density,
                     Temperature=Temperature)
    LL.setRPM(RPM)
    LL.updateFrame()
    AoA = LL.fields('AoA')
    LL.computeKinematicVelocity()
    LL.assembleAndProjectVelocities()
    if AirfoilAim == 'AoA': AoA[:] = AoA_imposed

    NewFields = ['VelocityKinematicAxial','VelocityKinematicTangential','psi',
                 'a','aP']

    f = LL.fields(NewFields)
    f = LL.allFields()
    f['VelocityKinematicAxial'][:] = f['VelocityAxial'][:]
    f['VelocityKinematicTangential'][:] = -f['VelocityTangential'][:]

    # Compute some constants:
    # Sutherland's law
    Mu = computeViscosityMolecular(Temperature)
    SoundSpeed = computeSoundSpeed(Temperature)
    r = f['Span']
    Rmax = f['Span'].max()
    ChordMin = 0.001*Rmax # TODO make parameter ?


    # Declare additional variables (not stored in LiftingLine)
    xi    = r/Rmax
    Omega = LL.RPM*np.pi/30.
    lambd = Velocity / (Omega * Rmax)

    # Compute Approximate Reynolds, Mach, and AoA,
    # based on initial "guess" of Chord distribution
    Wapprox = ((Omega*r)**2+Velocity**2)**0.5
    f['Reynolds'][:] = Density * Wapprox * f['Chord'] / Mu
    f['Mach'][:]  = Wapprox / SoundSpeed # Approximate

    if AirfoilAim == 'AoA':
        LL.computeSectionalCoefficients()
    else:
        f['AoA'][:] = 0.
        LL.setOptimumAngleOfAttack(Aim=AirfoilAim, AimValue=AimValue,
                                   AoASearchBounds=AoASearchBounds,)

    if SmoothAoA:
        WindowFilter = int(len(f['AoA'])*0.6)
        if WindowFilter%2 ==0: WindowFilter -=1
        FilterOrder  = 4
        f['AoA'][:] = ss.savgol_filter(f['AoA'],WindowFilter,FilterOrder)
        LL.computeSectionalCoefficients()

    def computeDistributionsFromZeta(zeta):

        xfact = Omega * r / Velocity

        f['phiRad'][:] = phi = np.arctan( (1+zeta/2.)/xfact ) # Eqn. 8
        cos_phi = np.cos(f['phiRad'])
        sin_phi = np.sin(f['phiRad'])
        tan_phi = sin_phi / cos_phi


        F = LL.computeTipLossFactor(NumberOfBlades, model=TipLossesModel)

        G = F * xfact * cos_phi * sin_phi # Eqn. 5

        Wc = 4*np.pi*lambd*G*Velocity*Rmax*zeta/(f['Cl']*NumberOfBlades) # Eqn. 16

        f['Reynolds'][:] = Density * Wc / Mu

        # Mach[:]  = Wc / (SoundSpeed * Chord) # Exact but bugged
        f['Mach'][:]  = Wapprox / SoundSpeed # Approximate

        if AirfoilAim == 'AoA' or it >= itMaxAoAsearch:
            LL.computeSectionalCoefficients()
        else:
            LL.setOptimumAngleOfAttack(Aim=AirfoilAim, AimValue=AimValue,
                                       AoASearchBounds=AoASearchBounds,)


        if SmoothAoA and it < itMaxAoAsearch:
            f['AoA'][:] = ss.savgol_filter(f['AoA'],WindowFilter,FilterOrder)
            LL.computeSectionalCoefficients()

        # Determine interference factors (Eqn.7)
        varepsilon = f['Cd'] / f['Cl']
        f['a'][:] = (zeta/2.)*cos_phi*cos_phi*(1-varepsilon*tan_phi)
        f['aP'][:] = (zeta/(2.*xfact))*cos_phi*sin_phi*(1+varepsilon/tan_phi)


        # Determine actual LocalVelocity (Eqn.17)
        f['VelocityMagnitudeLocal'][:] = W = Velocity*(1+f['a'])/sin_phi

        f['VelocityAxial'][:]  = Velocity*(1+f['a'])
        f['VelocityTangential'][:] = Omega*r*(1 - f['aP'])

        f['VelocityInducedAxial'][:]  = Velocity*(f['a'])
        f['VelocityInducedTangential'][:]  = Omega*r*(- f['aP'])

        # Update Chord distribution
        f['Chord'][:] = np.maximum(Wc/W,ChordMin)

        # Update blade twist
        f['Twist'][:] = f['AoA'] + np.rad2deg(f['phiRad'])

        # Determine derivatives (Eqns.11 a--d)
        I1 = 4*xi*G*(1-varepsilon*tan_phi)
        I2 = lambd*(0.5*I1/xi)*(1+varepsilon/tan_phi)*sin_phi*cos_phi
        J1 = 4*xi*G*(1+varepsilon/tan_phi)
        J2 = 0.5*J1*(1-varepsilon*tan_phi)*cos_phi*cos_phi

        # Integrate derivatives in order to get Power, Thrust and new zeta
        I1int = sint.simps(I1,xi)
        I2int = sint.simps(I2,xi)
        J1int = sint.simps(J1,xi)
        J2int = sint.simps(J2,xi)

        if Constraint == 'Thrust':
            Thrust = ConstraintValue
            Tc = 2*Thrust/(Density*Velocity**2*np.pi*Rmax**2)
            NewZeta = (0.5*I1int/I2int)-np.sqrt((0.5*I1int/I2int)**2-Tc/I2int)
            Pc = J1int*NewZeta+J2int*NewZeta**2
            Power = 0.5*Density*Velocity**3*np.pi*Rmax**2*Pc
        elif Constraint == 'Power':
            Power = ConstraintValue
            Pc = 2*Power/(Density*Velocity**3*np.pi*Rmax**2)
            NewZeta = -(0.5*J1int/J2int)+np.sqrt((0.5*J1int/J2int)**2 + (Pc/J2int))
            Tc = I1int*NewZeta - I2int*NewZeta**2
            Thrust = Tc*0.5*Density*Velocity**2*np.pi*Rmax**2

        Residual = NewZeta - zeta
        PropulsiveEfficiency = Velocity*Thrust/Power
        return Residual, Power, Thrust, PropulsiveEfficiency


    InitialGuess = f['Chord']

    # Direct iterations like this may work
    itmax = 100
    L2Residual, L2ResidualTol, L2minVariation = 1e8,1e-8, 1e-10
    zeta0 = InitialGuess
    it = -1
    L2Prev = 1.e10
    L2Variation = 1.e10
    while it<itmax and L2Residual>L2ResidualTol and L2Variation > L2minVariation:
        it += 1
        Residual, Power, Thrust, PropulsiveEfficiency = computeDistributionsFromZeta(zeta0)
        L2Residual = np.sqrt(Residual.dot(Residual))
        L2Variation = L2Prev-L2Residual
        L2Prev = L2Residual
        print ('it=%d | Thrust=%g, Power=%g, PropulsiveEfficiency=%g | L2 res = %g'%(it,Thrust,Power,PropulsiveEfficiency, L2Residual))
        loads = LL.computeLoads(NumberOfBlades=NumberOfBlades)
        print(WARN+'integrated | Thrust=%g, Power=%g'%(loads['Thrust'],loads['Power'])+ENDC)
        zeta1 = Residual+zeta0
        RelaxFactor = 0.
        zeta0 = (1.0-RelaxFactor)*zeta1+RelaxFactor*zeta0

    # Prepare output
    DesignPitch = LL.resetTwist()

    DictOfIntegralData = LL.computeLoads(NumberOfBlades=NumberOfBlades)
    DictOfIntegralData['Pitch'] = DesignPitch

    return DictOfIntegralData

def designHover(LL, NumberOfBlades=2, RPM=1500., AxialVelocity=0., Density=1.225,
              Temperature=288.15, TipLossesModel='Adkins',
              Constraint='Thrust', ConstraintValue=1000.,
              AirfoilAim='maxClCd', AimValue=None, AoASearchBounds=(-2,8),
              SmoothAoA=False, itMaxAoAsearch=3):
    '''
    This function performs a minimum induced loss design of a propeller
    in axial flight conditions.

    Parameters
    ----------

        LiftingLine : zone
            lifting-line used for design (values of chord and
            twist can be arbitrary).

            .. note:: this **LiftingLine** will be modified as a result of design

        PolarsInterpolatorDict : dict
            dictionary of interpolator functions of 2D polars, as obtained from
            :py:func:`MOLA.LiftingLine.buildPolarsInterpolatorDict`

        NBlades : int
            the number of blades of the design candidate

        Velocity : float
            the advance velocity of the propeller [m/s]

        RPM : float
            the rotational velocity of the propeller [rev / minute]

        Temperature : float
            the air temperature [K]

        Density : float
            the air density [kg / m3]

        TipLosses : str
            the model of tip-losses. Can be one of:
            ``'Adkins'``, ``'Glauert'`` or ``'Prandtl'``.

            .. note:: we recommend using ``'Adkins'``.

        ListOfEquations : str
            :py:func:`Converter.PyTree.initVars`-compatible equations to be applied
            after each lifting-line's polar interpolation operation.

        Constraint : str
            constraint type for the optimization. Can be: ``'Thrust'`` or ``'Power'``

        ConstraintValue : float
            the value of the constraint for the design

        Aim : str
            can be one of:

            * ``'Cl'``
                aims a given ``Cl``, :math:`c_l` , value (provided by argument **AimValue**)
                throughout the entire lifting line

            * ``'minCd'``
                aims the minimum ``Cd``, :math:`\min(c_d)` , value

            * ``'maxClCd'``
                aims the maximum aero efficiency :math:`\max(c_l/c_d)` value

        AimValue : float
            Specifies the aimed value for corresponding relevant **Aim** types
            (e.g. ``'Cl'``).

        AoASearchBounds : :py:class:`tuple` of 2 :py:class:`float`
            As there may exist multiple AoA values
            verifying the requested conditions, this argument constraints the
            research interval of angle-of-attack of valid candidates.

        SmoothAoA : bool
            if :py:obj:`True`, perform a post-smoothing of angle of
            attack spanwise distribution after optimization.

            .. tip:: this is useful in order to avoid "shark-teeth" shaped
                distribution.

        itMaxAoAsearch : int
            maximum number of iterations allowed for reaching the desired AoA
            distribution type

    Returns
    -------

        DictOfIntegralData : dict
            dictionary containing LiftingLine arrays.

            .. note:: for design result, please refer to input argument
                **LiftingLine**, which is modified (``Chord`` and ``Twist``
                values are updated).
    '''

    AoA = LL.fields('AoA')
    if AirfoilAim == 'AoA': AoA_imposed = np.copy(AoA)

    RotationAxis = LL.getRotationAxisCenterAndDirFromKinematics()[0]
    VelocityVector = - RotationAxis * AxialVelocity
    Velocity = AxialVelocity
    LL.setConditions(VelocityFreestream=VelocityVector, Density=Density,
                     Temperature=Temperature)
    LL.setRPM(RPM)
    LL.updateFrame()
    AoA = LL.fields('AoA')
    LL.computeKinematicVelocity()
    LL.assembleAndProjectVelocities()
    if AirfoilAim == 'AoA': AoA[:] = AoA_imposed

    NewFields = ['VelocityKinematicAxial','VelocityKinematicTangential','psi',
                 'a','aP']

    f = LL.fields(NewFields)
    f = LL.allFields()
    f['VelocityKinematicAxial'][:] = f['VelocityAxial'][:]
    f['VelocityKinematicTangential'][:] = -f['VelocityTangential'][:]

    # Compute some constants:
    # Sutherland's law
    Mu = computeViscosityMolecular(Temperature)
    SoundSpeed = computeSoundSpeed(Temperature)
    r = f['Span']
    Rmax = f['Span'].max()
    ChordMin = 0.001*Rmax # TODO make parameter ?


    # Declare additional variables (not stored in LiftingLine)
    xi    = r/Rmax
    Omega = LL.RPM*np.pi/30.
    lambd = Velocity / (Omega * Rmax)

    # Compute Approximate Reynolds, Mach, and AoA,
    # based on initial "guess" of Chord distribution
    Wapprox = ((Omega*r)**2+Velocity**2)**0.5
    f['Reynolds'][:] = Density * Wapprox * f['Chord'] / Mu
    f['Mach'][:]  = Wapprox / SoundSpeed # Approximate

    if AirfoilAim == 'AoA':
        LL.computeSectionalCoefficients()
    else:
        f['AoA'][:] = 0.
        LL.setOptimumAngleOfAttack(Aim=AirfoilAim, AimValue=AimValue,
                                   AoASearchBounds=AoASearchBounds,)

    if SmoothAoA:
        WindowFilter = int(len(f['AoA'])*0.6)
        if WindowFilter%2 ==0: WindowFilter -=1
        FilterOrder  = 4
        f['AoA'][:] = ss.savgol_filter(f['AoA'],WindowFilter,FilterOrder)
        LL.computeSectionalCoefficients()

    def computeDistributions(x):

        # advance
        # zeta = x
        # xfact = Omega * r / Velocity
        # hover 
        vp = x



        # advance
        # f['phiRad'][:] = phi = np.arctan( (1+zeta/2.)/xfact ) # Eqn. 8
        # hover
        phi_t = np.arctan(vp/(2*Omega*Rmax)) 
        f['phiRad'][:] = phi = np.arctan(np.tan(phi_t)/xi) # Eqn. 21

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        tan_phi = sin_phi / cos_phi


        F = LL.computeTipLossFactor(NumberOfBlades, model=TipLossesModel)

        # advance
        # G = F * xfact * cos_phi * sin_phi # Eqn. 5

        # advance
        # Wc = 4*np.pi*lambd*G*Velocity*Rmax*zeta/(f['Cl']*NumberOfBlades) # Eqn. 16
        # hover
        Wc = 4.*np.pi*xi*Rmax*F*vp*sin_phi*cos_phi/(f['Cl']*NumberOfBlades)

        f['Reynolds'][:] = Density * Wc / Mu

        # Mach[:]  = Wc / (SoundSpeed * Chord) # Exact but bugged
        f['Mach'][:]  = Wapprox / SoundSpeed # Approximate

        if AirfoilAim == 'AoA' or it >= itMaxAoAsearch:
            LL.computeSectionalCoefficients()
        else:
            LL.setOptimumAngleOfAttack(Aim=AirfoilAim, AimValue=AimValue,
                                       AoASearchBounds=AoASearchBounds,)


        if SmoothAoA and it < itMaxAoAsearch:
            f['AoA'][:] = ss.savgol_filter(f['AoA'],WindowFilter,FilterOrder)
            LL.computeSectionalCoefficients()

        # Determine interference factors (Eqn.7)
        varepsilon = f['Cd'] / f['Cl']

        # advance
        # f['a'][:] = (zeta/2.)*cos_phi*cos_phi*(1-varepsilon*tan_phi)
        # f['aP'][:] = (zeta/(2.*xfact))*cos_phi*sin_phi*(1+varepsilon/tan_phi)
        # hover
        f['aP'][:] = vp/(2*Omega*xi*Rmax)*cos_phi*sin_phi*(1+varepsilon/tan_phi)


        # Determine actual LocalVelocity (Eqn.17)
        # # advance
        # f['VelocityMagnitudeLocal'][:] = W = Velocity*(1+f['a'])/sin_phi
        # f['VelocityAxial'][:]  = Velocity*(1+f['a'])
        # f['VelocityTangential'][:] = Omega*r*(1 - f['aP'])
        # hover
        f['VelocityAxial'][:]=Va= vp/2.*cos_phi**2.*(1.-varepsilon*tan_phi)
        f['VelocityTangential'][:]= Omega*r*(1 - f['aP'])
        f['VelocityMagnitudeLocal'][:] = W = Va/sin_phi # is equivalent to np.sqrt(Va**2+Vt**2)

        # advance
        # f['VelocityInducedAxial'][:]  = Velocity*(f['a'])
        # hover
        f['VelocityInducedAxial'][:]  = f['VelocityAxial']


        f['VelocityInducedTangential'][:]  = Omega*r*(- f['aP'])

        # Update Chord distribution
        f['Chord'][:] = np.maximum(Wc/W,ChordMin)

        # Update blade twist
        f['Twist'][:] = f['AoA'] + np.rad2deg(f['phiRad'])

        # advance
        # # Determine derivatives (Eqns.11 a--d)
        # I1 = 4*xi*G*(1-varepsilon*tan_phi)
        # J1 = 4*xi*G*(1+varepsilon/tan_phi)
        # rotor
        I1 = 2*xi*F/(Omega**2*Rmax**3)*(cos_phi**2*(1-varepsilon*tan_phi))**2.
        J1 = 2*xi*F/(Omega**2*Rmax**4)* cos_phi**3*(1-varepsilon*tan_phi)*sin_phi*(1+varepsilon/tan_phi)

        I2 = lambd*(0.5*I1/xi)*(1+varepsilon/tan_phi)*sin_phi*cos_phi
        J2 = 0.5*J1*(1-varepsilon*tan_phi)*cos_phi*cos_phi

        # Integrate derivatives in order to get Power, Thrust and new zeta
        I1int = sint.simps(I1,xi)
        J1int = sint.simps(J1,xi)
        I2int = sint.simps(I2,xi)
        J2int = sint.simps(J2,xi)


        # advance
        # if Constraint == 'Thrust':
        #     Thrust = ConstraintValue
        #     Tc = 2*Thrust/(Density*Velocity**2*np.pi*Rmax**2)
        #     NewZeta = (0.5*I1int/I2int)-np.sqrt((0.5*I1int/I2int)**2-Tc/I2int)
        #     Pc = J1int*NewZeta+J2int*NewZeta**2
        #     Power = 0.5*Density*Velocity**3*np.pi*Rmax**2*Pc
        # elif Constraint == 'Power':
        #     Power = ConstraintValue
        #     Pc = 2*Power/(Density*Velocity**3*np.pi*Rmax**2)
        #     NewZeta = -(0.5*J1int/J2int)+np.sqrt((0.5*J1int/J2int)**2 + (Pc/J2int))
        #     Tc = I1int*NewZeta - I2int*NewZeta**2
        #     Thrust = Tc*0.5*Density*Velocity**2*np.pi*Rmax**2
        # Residual = NewZeta - zeta
        # PropulsiveEfficiency = Velocity*Thrust/Power
        # return Residual, Power, Thrust, PropulsiveEfficiency
        # rotor
        if Constraint == 'Thrust':
            Thrust = ConstraintValue
            Tc = 2*Thrust/(Density*(Omega*Rmax)**2*np.pi*Rmax**2)
            new_vp = np.sqrt(Tc/I1int)
            Pc = J1int*new_vp**2
            Power = 0.5*Density*(Omega*Rmax)**3*np.pi*Rmax**2*Pc
            print('Tc=%g ; Pc=%g'%(Tc,Pc))

        elif Constraint == 'Power':
            raise ValueError('Constrint="Power" not implemented yet')
        Residual = new_vp - vp
        FigureOfMerit = 1./np.sqrt(2.)/Pc * Tc**1.5


        return Residual, Power, Thrust, FigureOfMerit


    # advance
    # InitialGuess = f['Chord']
    # hover
    InitialGuess = 0.1

    # Direct iterations like this may work
    itmax = 100
    L2Residual, L2ResidualTol, L2minVariation = 1e8,1e-8, 1e-10
    x0 = InitialGuess
    it = -1
    L2Prev = 1.e10
    L2Variation = 1.e10
    while it<itmax and L2Residual>L2ResidualTol and L2Variation > L2minVariation:
        it += 1
        Residual, Power, Thrust, Efficiency = computeDistributions(x0)
        # advance
        # L2Residual = np.sqrt(Residual.dot(Residual))
        # hover
        L2Residual = Residual
        L2Variation = L2Prev-L2Residual
        L2Prev = L2Residual
        print('it=%d | Thrust=%g, Power=%g, Efficiency=%g | L2 res = %g'%(it,Thrust,Power,Efficiency, L2Residual))
        loads = LL.computeLoads(NumberOfBlades=NumberOfBlades)
        print(WARN+'integrated | Thrust=%g, Power=%g'%(loads['Thrust'],loads['Power'])+ENDC)
        x1 = Residual+x0
        RelaxFactor = 0.
        x0 = (1.0-RelaxFactor)*x1+RelaxFactor*x0

    # Prepare output

    LL.computeLoads(NumberOfBlades=NumberOfBlades)
    DictOfIntegralData = LL.addHelicopterRotorLoads()
    DesignPitch = LL.resetTwist()
    DictOfIntegralData['Pitch'] = DesignPitch

    return DictOfIntegralData

def optimalDesignByThrustAndFixedPitch(number_of_blades=2, Rmin=0.1, Rmax=1.0,
        max_mach_tip=0.8, min_mach_tip=0.35, nb_envelope_points=11, ChordMax=0.15,
        Temperature=288.15, Density=1.225,
        VelocityAxial=10., NominalRequiredThrust=100.,
        VelocityAxialForMaxThrust=5., MaximumRequiredThrust=200.,
        AirfoilPolarsFilenames='/stck/lbernard/MOLA/v1.13/EXAMPLES/BEMT/ROTOR_DESIGN/Polar.cgns',
        PolarsRelativeSpan=[0,1],
        PolarsNames=['OA309', 'OA309'], FAILED_VALUE=np.nan):


    print(CYAN)
    print('number_of_blades = %g'%number_of_blades)
    print('Rmax = %g'%Rmax)
    print('Rmin = %g'%Rmin)
    print('max_mach_tip = %g'%max_mach_tip)
    print('VelocityAxial = %g'%VelocityAxial)
    print('NominalRequiredThrust = %g'%NominalRequiredThrust)
    print('VelocityAxialForMaxThrust = %g'%VelocityAxialForMaxThrust)
    print('MaximumRequiredThrust = %g'%MaximumRequiredThrust)
    print('Temperature = %g'%Temperature)
    print('Density = %g'%Density)
    print(ENDC+'\n')
    SpanTotal = Rmax-Rmin
    LL = LiftingLine(SpanMin=Rmin, SpanMax=Rmax, N=50,
             SpanwiseDistribution=dict(kind='bitanh',first=0.05*SpanTotal, last=0.0016*SpanTotal),
             GeometricalLaws=dict(
                Chord=dict(RelativeSpan=[0, 1],
                           Chord=[0.1, 0.1],
                           InterpolationLaw='interp1d_linear'),
                Twist=dict(RelativeSpan=[0, 1],
                           Twist=[0., 0.],
                           InterpolationLaw='interp1d_linear'),
                Sweep=dict(RelativeSpan=[0, 1],
                           Sweep=[0., 0.],
                           InterpolationLaw='interp1d_linear'),
                Dihedral=dict(RelativeSpan=[0, 1],
                           Dihedral=[0., 0.],
                           InterpolationLaw='interp1d_linear'),
                Airfoils=dict(RelativeSpan=PolarsRelativeSpan,
                              PyZonePolarNames=PolarsNames,
                              InterpolationLaw='interp1d_linear'),
                ),
             AirfoilPolarsFilenames=AirfoilPolarsFilenames,
             Name='TestLiftingLine')

    SpeedOfSound = computeSoundSpeed(Temperature)

    max_RPM = (30/np.pi)*max_mach_tip*SpeedOfSound/Rmax

    mach_tip = np.linspace(min_mach_tip,max_mach_tip,nb_envelope_points)

    is_max_thrust_possible = np.zeros(nb_envelope_points, dtype=bool)
    maximum_powers = np.zeros(nb_envelope_points, dtype=float)
    maximum_rpms = np.zeros(nb_envelope_points, dtype=float)
    nominal_rpms = np.zeros(nb_envelope_points, dtype=float)
    nominal_powers = np.zeros(nb_envelope_points, dtype=float)
    DesignLiftingLines = []

    for i in range(nb_envelope_points):

        RPM = (30/np.pi)*mach_tip[i]*SpeedOfSound/Rmax

        # design the propeller
        AdvanceParameterNorm = abs(VelocityAxial / (LL.RPM/60.*(2*Rmax)))
        if AdvanceParameterNorm > 0.1:
            NominalDict = design(LL, AirfoilAim='maxClCd',
                NumberOfBlades=number_of_blades, AxialVelocity=VelocityAxial,
                RPM=RPM, Temperature=Temperature, Density=Density,
                Constraint='Thrust',ConstraintValue=NominalRequiredThrust)
        else:
            NominalDict = designHover(LL, AirfoilAim='maxClCd',
                NumberOfBlades=number_of_blades, AxialVelocity=0,
                RPM=RPM, Temperature=Temperature, Density=Density,
                Constraint='Thrust',ConstraintValue=NominalRequiredThrust)                

        r, AoA, Chord, Twist = LL.fields(['Span','AoA','Chord','Twist'])

        Chord[:] = np.minimum(ChordMax,Chord)
        AoA_Step2a, Chord_Step2a = AoA*1., Chord*1.  # (produce copies)
        AoA_Root = np.mean(0.75*AoA[:int(len(AoA)/4)])
        AoA_Tip  = np.mean(0.75*AoA[int(3*len(AoA)/4):])
        AoA[:] = np.linspace(AoA_Root,AoA_Tip,len(AoA))

        print(PINK)
        print('AoA_Root = %g'%AoA_Root)
        print('AoA_Tip = %g'%AoA_Tip)
        print('RPM = %g'%RPM)
        print(ENDC)

        AdvanceParameterNorm = abs(VelocityAxial / (LL.RPM/60.*(2*Rmax)))
        if AdvanceParameterNorm > 0.1:
            NominalDict = design(LL, AirfoilAim='AoA',
                NumberOfBlades=number_of_blades, AxialVelocity=VelocityAxial,
                RPM=RPM, Temperature=Temperature, Density=Density,
                Constraint='Thrust',ConstraintValue=NominalRequiredThrust)
        else:
            NominalDict = designHover(LL, AirfoilAim='AoA',
                NumberOfBlades=number_of_blades, AxialVelocity=VelocityAxial,
                RPM=RPM, Temperature=Temperature, Density=Density,
                Constraint='Thrust',ConstraintValue=NominalRequiredThrust)

        Chord[:] = np.minimum(ChordMax,Chord)
        print('design Chord')
        print(Chord)
        print('design Twist')
        print(Twist + LL.Pitch)

        print('analysis...')
        NominalDict = compute(LL, model='Drela',
             NumberOfBlades=number_of_blades, Density=Density, Temperature=Temperature,
             RPM=LL.RPM,Pitch=NominalDict['Pitch'], AxialVelocity=VelocityAxial,
             TrimCommand='RPM', TrimQuantity='Thrust', TrimValue=NominalRequiredThrust)

        CopiedLL = LL.copy(deep=True)
        print("CopiedLL.Pitch",CopiedLL.Pitch)
        DesignLiftingLines.append( CopiedLL )

        nominal_rpms[i] = NominalDict['RPM']
        nominal_powers[i] = NominalDict['Power']

        print('nominal_rpm = %g'%nominal_rpms[i])
        print('nominal_power = %g'%nominal_powers[i])
        print('Thrust = %g'%NominalDict['Thrust'])
        print('')

        # verify if we can reach the required maximum thrust at max_RPM
        FullThrottleDict = compute(LL, model='Drela',
            AxialVelocity=VelocityAxialForMaxThrust,
            RPM=max_RPM, Pitch=NominalDict['Pitch'],
            NumberOfBlades=number_of_blades,
            Density=Density, Temperature=Temperature)

        print('thrust_at_max_rpm = %g'%FullThrottleDict['Thrust'])
        print('power_at_max_rpm = %g'%FullThrottleDict['Power'])
        print('')


        is_max_thrust_possible[i] = False

        if FullThrottleDict['Thrust'] > MaximumRequiredThrust:

            # trim RPM for maximum required Thrust condition
            is_max_thrust_possible[i] = True

            MaxThrustDict = compute(LL, model='Drela',
                NumberOfBlades=number_of_blades, Density=Density, Temperature=Temperature,
                RPM=LL.RPM,Pitch=NominalDict['Pitch'], AxialVelocity=VelocityAxial,
                TrimCommand='RPM', TrimQuantity='Thrust', TrimValue=MaximumRequiredThrust,
                TrimCommandGuess=[[0.8,1.03],[0.5,1.03]])

            maximum_powers[i] = MaxThrustDict['Power']
            maximum_rpms[i] = MaxThrustDict['RPM']

            print(GREEN)
            print('max_req_thrust = %g'%MaxThrustDict['Thrust'])
            print('power_at_max_req_thrust = %g'%MaxThrustDict['Power'])
            print('rpm_at_max_req_thrust = %g'%MaxThrustDict['RPM'])
            print(ENDC+'')


    # discard designs that cannot reach the maximum required thrust
    nominal_powers=nominal_powers[is_max_thrust_possible]
    nominal_rpms=nominal_rpms[is_max_thrust_possible]
    maximum_powers=maximum_powers[is_max_thrust_possible]
    DesignLiftingLines= [l for i,l in enumerate(DesignLiftingLines) if is_max_thrust_possible[i]]

    outputs = {'pressure_field':FAILED_VALUE,
               'airfoil':FAILED_VALUE,
               'chord':FAILED_VALUE,
               'radius':FAILED_VALUE,
               'twist':FAILED_VALUE,
               'max_power':FAILED_VALUE,
               'max_rpm':FAILED_VALUE,
               'nominal_power':FAILED_VALUE,
               'nominal_rpm':FAILED_VALUE}

    if len(nominal_powers) == 0:
        # NO DESIGN CAN REACH MAXIMUM REQUIRED THRUST
        return None, outputs

    else:
        print('NOMINAL POWERS')
        print(nominal_powers)
        print('MAX POWERS')
        print(maximum_powers)
        best_design_index = np.argmin(nominal_powers)
        maximum_power = maximum_powers[best_design_index]
        maximum_rpm = maximum_rpms[best_design_index]
        nominal_rpm = nominal_rpms[best_design_index]
        nominal_power = nominal_powers[best_design_index]
        DesignLiftingLine = DesignLiftingLines[best_design_index]

        chord, radius, twist = DesignLiftingLine.fields(['Chord','Span','Twist'])

        outputs['pressure_field'] = FAILED_VALUE
        outputs['airfoil'] = FAILED_VALUE
        outputs['chord'] = chord
        outputs['radius'] = radius
        outputs['twist'] = twist
        outputs['max_power'] = maximum_power
        outputs['max_rpm'] = maximum_rpm
        outputs['nominal_power'] = nominal_power
        outputs['nominal_rpm'] = nominal_rpm
        outputs['pitch'] = DesignLiftingLine.Pitch

        return DesignLiftingLine, outputs


def load(filename):
    t = file_load( filename )
    LLs = [ z for z in t.zones() if isinstance(z,LiftingLine) ]
    NbOfLiftingLines = len(LLs)
    if NbOfLiftingLines == 1:
        return LLs[0]
    elif NbOfLiftingLines == 0:
        raise IOError('no lifting line found in file %s'%filename)
    else:
        raise IOError('multiple lifting lines (%d) found in file %s'%(NbOfLiftingLines,filename))
