'''
MOLA - PropellerAnalysis.py

PROPELLER ANALYSIS MODULE

Collection of routines and functions designed for performing
rapid Propeller Analysis.

19/07/2019 - L. Bernardos - Creation
'''

# System modules
import sys, os
if sys.version_info[0] > 2: range = range # guarantee backwards compatibility
import numpy as np
import timeit
import itertools

import Converter.PyTree as C
import Converter.Internal as I

from . import InternalShortcuts as J
from . import Wireframe as W
from . import LiftingLine as LL

try:
    silence = J.OutputGrabber()
    with silence:
        import PUMA
        import PUMA.Fact
except:
    pass

# Global constants
# -> Fluid constants
Gamma, Rgp = 1.4, 287.053
Mus, Cs, Ts= 1.711e-5, 110.4, 273.0 # Sutherland const.

# -> Blade references
ZeroPitchRelativeSpan=0.75

# Some variables
GREEN = '\033[92m'
FAIL  = '\033[91m'
WARN  = '\033[93m'
ENDC  = '\033[0m'

DEBUG = True

CostFoilInterp = np.array([0.0])
CostApplyPolar = np.array([0.0])

# Plot options
__PLOT_STIX_LATEX__ = True

def computePUMA(NBlades, Velocity, RPM, Temperature, Density,
    Constraint='Thrust',
    ConstraintValue=1000.,
    PerturbationField=None,
    PrescribedWake=False,
    FroudeVelocity=True,
    Restart=False,
    GeomBladeFile='GeomBlade.py',
    PUMADir='PUMADir',
    Spinner=None,
    ExtractVolume=None,
    SpanDiscretization=None,
    StopDeltaTtol=None,
    saveKimFiles=True,
    AdvancedOptions=dict(
        AdvanceRatioHoverLim=0.01,
        BoundMach=0.95,
        Boundaries=[False,True],
        WakeRevs=5.,
        MinimumRevsAfterWakeRevs=5.,
        ReynoldsFormula="Cd=Cd*(%f/%f)**(-0.11)",
        MaximumNrevs=10.,
        Dpsi=10.,
        NbSections=None,
        InitialPitch=0.),
    ):
    '''
    This is a macro-function used for making a simple isolated propeller PUMA
    simulation in axial flight.

    Parameters
    ----------

        NBlades : int
            number of blades of the propeller

        Velocity : float
            axial advancing velocity of the propeller [m/s]

        RPM : float
            rotational speed of the propeller [revolutions per minute]

        Temperature : float
            external air temperature [K]

        Density : float
            air density [kg / m3]

        Constraint : str
            one of: ``'Thrust'``, ``'Power'`` or :py:obj:`None`. Specifies the
            trim constraint

        ConstraintValue : float
            if trim **Constraint** type is provided, then this
            argument specifies its value.

        PerturbationField : zone or :py:obj:`None`
            zone containing ``VelocityX``, ``VelocityY`` and ``VelocityZ``
            perturbation fields or None if no perturbation is imposed

        PrescribedWake : bool
            if :py:obj:`True`, wake is not free, but prescribed.

        FroudeVelocity : bool
            if :py:obj:`True`, Froude velocity is computed. This
            shall be set to :py:obj:`True` if **PrescribedWake** == :py:obj:`True`
            for consistency.

        Restart : bool
            If :py:obj:`True`, restart from fields ``Restart.plt``.

            .. warning:: this feature is bugged in PUMA in its last tested version

        GeomBladeFile : str
            name of the file containing blade geometry of PUMA

        PUMADir : str
            name of the compute directory of PUMA

        Spinner : :py:obj:`None`, zone or :py:class:`str`
            * if :py:obj:`None`, then no spinner modeling

            * if zone, then use provided surface for modeling spinner

            * if :py:class:`str`, then it can be ``'Standard'``, then a default
                spinner is constructed

        ExtractVolume : :py:obj:`None`, zone or :py:class:`str`
            * if :py:obj:`None`, then no volume data is extracted

            * if zone, then use provided zone for extracting volume fields

            * if :py:class:`str` then it can be one of:

                * ``"cartesian2D"``
                    use a default 2D cartesian surface

                * ``"cartesian3D"``
                    use a default 3D cartesian block

                * ``"cylinder"``
                    use a default 3D cylinder

        SpanDiscretization : :py:obj:`None` or :py:class:`str`
            kind of discretization law for blades ``'sqrt'`` or ``'linear'``

        StopDeltaTtol : :py:obj:`None` or :py:class:`float`
            if provided, sets the threshold difference
            of thrust used to determine if convergence is reached [N]

        saveKimFiles : bool
            if :py:obj:`True`, save ``kim.geom`` and kim.autre files

    Returns
    -------

        DictOfIntegralData : :py:class:`dict`
            dictionary including predictions

        Prop.Loads.Data : PUMA object

        SectionalLoadsLL : PUMA object
    '''
    import Generator.PyTree as G
    import Transform.PyTree as T
    import Geom.PyTree as D
    try: import CPlot.PyTree as CPlot
    except: CPlot = None

    import PUMA # PUMA Requires appropriate environment

    ao = AdvancedOptions

    if not os.path.isdir(PUMADir): os.makedirs(PUMADir)

    # Disable <StopDeltaTtol> if not provided
    if StopDeltaTtol is None: StopDeltaTtol = -1.0

    # Blade Geometry
    BladeGeom = J.load_source('Geom',GeomBladeFile)
    BladeDef = BladeGeom.GetBladeDef()
    Rmax = np.array(BladeDef['Span']).max()
    Rmin = np.array(BladeDef['Span']).min()

    # Advance ratio
    AdvRatioJ = Velocity / ((RPM/60.)*(2*Rmax))

    # Boolean specifying special Hover treatment
    PseudoHOVER = True if AdvRatioJ < ao['AdvanceRatioHoverLim'] else False

    # Blade Options
    BOpt=dict(Interpolate='linear',Correction3D='Mach_Wind',
        Boundaries=ao['Boundaries'],BoundMach=ao['BoundMach'])
    if SpanDiscretization is None:
        BOpt['SpanDiscretization'] = 'sqrt' if PseudoHOVER else 'linear'
    else:
        BOpt['SpanDiscretization'] = SpanDiscretization

    def WriteAll3DGeom(FilenameOutput):
        """
        Input : Filename of result (e.g. "MyDirectory/Geometry.cgns")
        Output : PyTree with Geometry
        """
        # Trees = map(lambda body: body.getGeom3D(),Pb.Fluid.objects['LiftingElement'])

        Trees = [body.getGeom3D() for body in Pb.Fluid.objects['LiftingElement']]

        Surfaces = []
        for t in Trees:
            # Sections = map(lambda zn: I.getNodeFromNameAndType(t,zn,'Zone_t'),('TE','QCUP','LE','QCDOWN','TE'))
            Sections = [I.getNodeFromNameAndType(t,zn,'Zone_t') for zn in ('TE','QCUP','LE','QCDOWN')]
            Nj = len(Sections)
            Ni = C.getNPts(Sections[0])
            Surface = G.cart((0,0,0),(1,1,1),(Ni,Nj,1))
            BaseName = I.getNodeFromType(t,'CGNSBase_t')[0]
            Surface[0] = BaseName
            SurfX = I.getNodeFromName2(Surface,'CoordinateX')[1]
            SurfY = I.getNodeFromName2(Surface,'CoordinateY')[1]
            SurfZ = I.getNodeFromName2(Surface,'CoordinateZ')[1]
            for j in range(Nj):
                SecX = I.getNodeFromName2(Sections[j],'CoordinateX')[1]
                SecY = I.getNodeFromName2(Sections[j],'CoordinateY')[1]
                SecZ = I.getNodeFromName2(Sections[j],'CoordinateZ')[1]
                SurfX[:,j] = SecX
                SurfY[:,j] = SecY
                SurfZ[:,j] = SecZ
            Surfaces += [Surface]

        tOut = C.newPyTree(['BaseGeometry',Surfaces])
        C.convertPyTree2File(tOut,FilenameOutput)
        return tOut

    # ---------------------------------------------------- #

    # Set PUMA Problem
    Pb = PUMA.Problem('Propeller')
    Pb.set('OutputDir',PUMADir)
    Pb.set('FreeStream',{'VelocityX':-Velocity,'VelocityY':0.,'VelocityZ':0,'Density':Density,'Temperature':Temperature})

    # Wake model
    Wake=Pb.add_Wake('Wake',Prescribed_Opt=PrescribedWake)
    NbAges = int((360./ao['Dpsi'])*ao['WakeRevs'])
    Wake.set('NbAges',NbAges)
    # Wake.set('RegCoefP',1.0)
    # Wake.set('RegCoefL',0.5)
    if PseudoHOVER:
        Wake.set('RegCoefP',1.8)
        Wake.set('OrderLength',[1,4,1])

    # Support of the Propeller
    ModelSupport=Pb.add_Root('ModelSupport',State={'Phi':0.,'Theta':0.,'Psi':0.})
    ModelSupport.Cmds.set('Motion',{'VelocityX':0.,'VelocityY':0.,'VelocityZ':0.})

    # Propeller
    Prop = ModelSupport.add_Propeller('Prop',[[0.,0.,0.],[0.,0.,0.]],Direct=1)
    if ao['NbSections'] is None: ao['NbSections']=len(BladeDef['Span'])
    Prop.add_Blades(NBlades,Aerodynamics={'Definition':BladeDef,'NbSections':ao['NbSections'],'Options':BOpt,'IndVeloModel':Wake})
    Prop.Cmds.set('Omega',RPM) # in RPM

    if Spinner is not None:
        if isinstance(Spinner,list):
            typeOfElmnt = I.isStdNode(Spinner)
            if typeOfElmnt == 0:
                raise AttributeError('Spinner variable is a list of nodes. This is not implemented yet')
            elif typeOfElmnt == -1:
                SpinnerSurface = Spinner
                Pb.add_SimpleSurface('Spinner',SpinnerSurface)
            else:
                raise AttributeError('Spinner variable type not recognized. Please provide an allowed keyword or a PyTree zone')
        elif isinstance(Spinner,str):
            from . import RotatoryWings as RW
            import Transform.PyTree as T
            import Generator.PyTree as G

            if Spinner == 'Standard':
                Height = 2*Rmin
                Width  = 0.9*Rmin
                Length = 3*Height
                SpinnerProfile,_ = RW.makeSimpleSpinner(Height, Width, Length, NptsTop=20, NptsBottom=30)
                SpinnerZones,_ = RW.makeHub(SpinnerProfile,
                                                AxeCenter=(0,0,0),
                                                AxeDir=(1,0,0),
                                                NumberOfAzimutalPoints=91,
                                                BladeNumberForPeriodic=None,
                                                LeadingEdgeAbscissa=0.25,
                                                TrailingEdgeAbscissa=0.75,
                                                SmoothingParameters={'eps':0.50,'niter':300,'type':2})

                tAux = C.newPyTree(['Base',I.getZones(SpinnerZones)])
                I._rmNodesByType(tAux,'FlowSolution_t')
                SpinnerUnstr = C.convertArray2Hexa(tAux)
                SpinnerSurface = T.join(I.getZones(SpinnerUnstr))
                G._close(SpinnerSurface)
                SpinnerSurface[0] = 'SpinnerSurface'
                SpinnerSurface=T.reorderAll(SpinnerSurface)
                C.convertPyTree2File(SpinnerSurface,'mySpinner.cgns')

            else:
                # Spinner are Keywords of the form:
                # 'Height=5*Rmin,Width=Rmin,Length=3*Height'
                raise AttributeError('Spinner attribute of kind "Height=5*Rmin,Width=Rmin,Length=3*Height" to be implemented')

            Pb.add_SimpleSurface('Spinner',SpinnerSurface)


    # Numerics
    Num=Pb.get_Numerics()
    Num.set('TimeStep',float(ao['Dpsi'])/(360.*RPM/60.))

    if 'NumConv' in ao: Num.set('Convergence',ao['NumConv'])
    else: Num.set('Convergence',1.e-3)

    if 'NumRelax' in ao: Num.set('Relaxation',ao['NumRelax'])
    else: Num.set('Relaxation',0.1)


    if 'NbSubIterations' in ao: Num.set('NbSubIterations',ao['NbSubIterations'])
    else: Num.set('NbSubIterations',100)


    if PseudoHOVER: Num.set('Relaxation',0.05)

    if Spinner is not None:
        Num.set('Linear_NbSubIterations',200)
        Num.set('Linear_Solver','minres') # 'minres' or 'lgmres'
        Num.set('Linear_Convergence',1e-4)


    MinimumNiters = int(NbAges+(360./ao['Dpsi'])*ao['MinimumRevsAfterWakeRevs'])
    MaximumNiters = int((360./ao['Dpsi'])*ao['MaximumNrevs'])
    Niters=np.maximum(MinimumNiters,MaximumNiters)

    # Trim
    if Constraint == 'Thrust' or Constraint == 'Power':
        MakeTrim = True
        Trim = Pb.add_Trim('%sImposed'%Constraint)

        FactorOfRevsWithoutTrim = 0.2
        FactorOfRevsTrimConv = 0.1

        TrimValueConvergence = ConstraintValue * 1.e-5
        Trim.set('IterationsDelay', int( FactorOfRevsWithoutTrim*Niters ))
        Trim.set('Relaxation',0.15)
        Trim.set('Objective','Objective_%s'%Constraint,ConstraintValue,Prop.Loads,Constraint,int(FactorOfRevsTrimConv*Niters),TrimValueConvergence)
        Trim.set('Command','RecalagePitch',Prop.Cmds,'Pitch',0.05)
        '''
        # TODO: implement This:
        # Pour trimer toutes les N it:
        Trim.set('TrimPeriod',10)
        # Pour trimer en utilisant une moyenne sur N it
        Trim.set('TrimAveraged',10)
        '''
    else:
        MakeTrim = False

    if not not PerturbationField:
        Pb.set('PerturbationField',PerturbationField)

    if Restart:
        try:
            WakeTree = C.convertFile2PyTree('%s/Restart.plt'%PUMADir)
            print('Using restart wake from file.')
        except IOError:
            print('Using standard initialization.')
            Pb.initialize()
            pass
        else:
            Wake.setState(WakeTree)

    if PrescribedWake:
        Vfroude={'VelocityX':0.,'VelocityY':0.,'VelocityZ':0.}
        Wake.attachFroudeVelocity(Vfroude)
    Pb.initialize()

    if saveKimFiles: saveKimFilesFromPUMA(Pb)

    PitchValue = ConstraintValue if not MakeTrim else ao['InitialPitch']
    Prop.Cmds.set('Pitch',PitchValue)
    Thrust = Prop.Loads.Data['Thrust']

    if PrescribedWake:
        if FroudeVelocity:
            for it in range(Niters):
                Vfroude=Prop.CalculateFroudeVelocity(Radius=Rmax,verbose=True)
                Wake.attachFroudeVelocity(Vfroude)
                Pb.advance(Num.get('TimeStep'))
                ThrustAbsDiff=np.abs(Prop.Loads.Data['Thrust']-Thrust)
                Thrust = Prop.Loads.Data['Thrust']
                Power = Prop.Loads.Data['Power']
                print('it %d/%d : Thrust %g N, Power %g W, |Delta(T)| %g N'%(it,Niters-1,Thrust,Power,ThrustAbsDiff))
                if ThrustAbsDiff<=StopDeltaTtol and (it >= MinimumNiters):
                    print('Reached StopDeltaTtol convergence criterion')
                    break


        else:
            for it in range(Niters):
                Pb.advance(Num.get('TimeStep'))
                ThrustAbsDiff=np.abs(Prop.Loads.Data['Thrust']-Thrust)
                Thrust = Prop.Loads.Data['Thrust']
                Power = Prop.Loads.Data['Power']
                print('it %d/%d : Thrust %g N, Power %g W, |Delta(T)| %g N'%(it,Niters-1,Thrust,Power,ThrustAbsDiff))
                if ThrustAbsDiff<=StopDeltaTtol and (it >= MinimumNiters):
                    print('Reached StopDeltaTtol convergence criterion')
                    break

    else:
        for it in range(Niters):
            Pb.advance(Num.get('TimeStep'))
            ThrustAbsDiff=np.abs(Prop.Loads.Data['Thrust']-Thrust)
            Thrust = Prop.Loads.Data['Thrust']
            Power = Prop.Loads.Data['Power']
            print('it %d/%d : Thrust %g N, Power %g W, |Delta(T)| %g N'%(it,Niters-1,Thrust,Power,ThrustAbsDiff))
            if ThrustAbsDiff<=StopDeltaTtol and (it >= MinimumNiters):
                print('Reached StopDeltaTtol convergence criterion')
                break


    # Write geometry
    tGeom = WriteAll3DGeom('%s/Geometry.cgns'%PUMADir)


    C.convertPyTree2File(Wake.getState(),'%s/Restart.plt'%PUMADir)


    # Merge Wake and Geometry:
    ListOfTrees = [Wake.getState(),tGeom]
    t = I.merge(ListOfTrees)
    BaseWake = I.getNodeFromNameAndType(t,'Wake','CGNSBase_t')
    BaseGeom = I.getNodeFromNameAndType(t,'BaseGeometry','CGNSBase_t')
    if CPlot:
        for z in BaseWake[2]:
            CPlot._addRender2Zone(z, material='Glass', color='Blue', blending=1., meshOverlay=None, shaderParameters=(1.43,2.0))
        if Spinner is not None: I.addChild(BaseGeom,SpinnerSurface)
        for z in BaseGeom[2]: CPlot._addRender2Zone(z, material='Solid', color='Red')
    C.convertPyTree2File(t, '%s/WakeAndGeometry.cgns'%PUMADir)


    # Extract integral arrays
    Thrust = Prop.Loads.Data['Thrust']
    Torque = Prop.Loads.Data['Torque']
    Power  = Prop.Loads.Data['Power']

    n = RPM/60.
    d = Rmax*2
    CTpropeller = Thrust / (Density * n**2 * d**4) # = Prop.Loads.Data['Ct']
    CPpropeller = Power  / (Density * n**3 * d**5) # = Prop.Loads.Data['Cp']
    FigureOfMeritPropeller = np.sqrt(2./np.pi)* CTpropeller**1.5 / CPpropeller

    PropEff = Velocity*Thrust/Power # = Prop.Loads.Data['Efficiency']

    DictOfIntegralData = dict(Thrust=Thrust,Power=Power,PropulsiveEfficiency=PropEff,J=AdvRatioJ,FigureOfMeritPropeller=FigureOfMeritPropeller)
    if MakeTrim: DictOfIntegralData['Pitch']=Trim.extractor.Data['RecalagePitch']


    # Extract Blade sectional data
    ListOfBlades = Prop.Blades.Blades

    BladesTrees = [blade.Fluid.BladeLLSectionalLoads.getDataAsTree() for blade in ListOfBlades]
    C.convertPyTree2File(BladesTrees,'%s/SectionalLoads.cgns'%PUMADir)

    Pb.finalize() # Make general extractions

    # Make volume extraction
    if ExtractVolume is not None:
        print('PUMA: making Flowfield extraction...')
        if isinstance(ExtractVolume,str):
            if ExtractVolume == "cartesian2D":
                Ni,Nj,Nk  = 400, 1, 200
                Lx,Ly,Lr  = 6*Rmax, 3*Rmax, 3*Rmax
                Flowfield = G.cart((-(Lx-4.*Rmin),-1.5*Rmax*0,-1.5*Rmax),(Lx/(Ni-1),1.,Lr/(Nk-1)),(Ni,Nj,Nk))
            elif ExtractVolume == "cartesian3D":
                Ni,Nj,Nk  = 400, 200, 200
                Lx,Ly,Lr  = 6*Rmax, 3*Rmax, 3*Rmax
                Flowfield = G.cart((-(Lx-4.*Rmin),-1.5*Rmax*-1.5*Rmax,-1.5*Rmax),(Lx/(Ni-1),Ly/(Nk-1),Lr/(Nk-1)),(Ni,Nj,Nk))

            elif ExtractVolume == "cylinder":

                Lx = 6*Rmax

                DistrX = D.getDistribution(D.line((0,0,0),(Lx,0,0),400))
                DistrR = D.getDistribution(D.line((0.5*Rmin,0,0),(1.5*Rmax,0,0),100))
                DistrTh= D.getDistribution(D.line((0,0,0),(1.,0,0),int(360/ao['Dpsi'])))

                Flowfield = G.cylinder2((0,0,0),0.5*Rmin,1.5*Rmax,180.,-180.,Lx,DistrR, DistrTh,DistrX)
                T._rotate(Flowfield,(0,0,0),(0,1,0),90.)
                T._translate(Flowfield,(-Lx,0,0))

            else:
                raise AttributeError("Kind of ExtractVolume (%s) not recognized"%ExtractVolume)
        else:
            typeOfElmnt = I.isStdNode(ExtractVolume)
            if typeOfElmnt == 0:
                raise AttributeError('ExtractVolume variable is a list of nodes. This is not implemented yet')
            elif typeOfElmnt == -1:
                Flowfield = ExtractVolume
            else:
                raise AttributeError('Spinner variable type not recognized. Please provide an allowed keyword or a PyTree zone')

        C.convertPyTree2File(Pb.getFlowField(Flowfield),Pb.getOutputFilePath('Flowfield.cgns'))

    SectionalLoadsLL = BladesTrees[0]
    return DictOfIntegralData, Prop.Loads.Data, SectionalLoadsLL

def computeQprop(QpropFile, Velocity, Pitch, RPM=None, Thrust=None, Power=None,
        Temperature = 288., Density=1.225,
        QpropExecutable="/stck/lbernard/Outils/Qprop/bin/qprop"):
    """
    Launch Qprop computation based on provided QpropFile,
    (see :py:func:`buildQpropFile` function to know how to generate
    the file) and operating conditions. Possible inputs are
    as expected by Qprop:

    ::

        qprop propfile motorfile Vel Rpm [ Volt dBeta Thrust Torque Amps Pele ]

    This means that **Velocity** and **Pitch** arguments shall
    *ALWAYS* be specified by user. However, the rest of
    variables are optional, and Qprop will attempt to find
    a mechanical equilibrium if at least one of (**RPM**, **Thrust** or
    **Power**) is also specified.

    Note that Motor parameters is not implemented, so
    motorfile, *Volt*, *Amps* and *Pele* are never used.

    Note that for the moment only single-point computations are possible.
    Multi-point behavior can be mimicked by using this function inside a loop.

    Parameters
    ----------

        Velocity : float
            Velocity of advance of propeller (m/s)

        Pitch : float
            Pitch angle (degrees)

        RPM : :py:class:`float` or :py:obj:`None`
            Rotational speed in rev / minute

        Thrust : :py:class:`float` or :py:obj:`None`
            Required Thrust (N)

        Power : :py:class:`float` or :py:obj:`None`
            Required Power (W)

        Temperature : float
            Temperature (Kelvin)

        Density : float
            Air density (kg / m^3)

        QpropExecutable : str
            Location of qprop executable

    Returns
    -------

        Result : str
            To be printed or post-processed with :py:class:`extractQprop` function
    """

    import subprocess


    # Write a local file specifying air constants
    # Compute some constants:
    # Sutherland's law
    Mu=Mus*((Temperature/Ts)**0.5)*((1.+Cs/Ts)/(1.+Cs/Temperature))
    Pressure = Density * Rgp * Temperature # State
    SoundSpeed = (Gamma * Rgp * Temperature)**0.5

    with open('qcon.def','w') as f:
        f.write(' %g    ! rho (kg/m^3)   density\n'%Density)
        f.write(' %g    ! mu  (kg/m-s)   dynamic viscosity\n'%Mu)
        f.write(' %g    ! a   (m/s)      speed of sound\n'%SoundSpeed)


    if Thrust == 0:
        Thrust = 1e-5 # Very small
    elif Thrust is None:
        Thrust = 0 # Qprop's "None" equivalent
    if Power == 0:
        Power = 1e-5 # Very small
    elif Power is None:
        Power = 0 # Qprop's "None" equivalent

    Omega = RPM*np.pi/30.
    Torque = Power / Omega

    result = subprocess.check_output("%s %s /stck/lbernard/Outils/Qprop/runs/nomotorfile %g %g 0 %g %g %g"%(QpropExecutable,QpropFile, Velocity, RPM, Pitch, Thrust, Torque), shell=True)

    return result

def buildQpropFile(LiftingLine, PyZonePolars, OutputQpropFilename='prop.qprop',
        NBlades=6,Velocity=27.77,RPM=500., Temperature = 288-4.5+20.,
        Density=1.225*0.73, LinearFitAoACLmaxFactor=0.5,
        REexps=(-0.7,-0.5), RElow=1.e5):
    """
    Build a Drela's Qprop-compliant propeller file based on
    a LiftingLine and PyZonePolar objects.

    In order to produce Qprop-compliant airfoil's polar
    information, the user shall provide a reference flight
    condition (advance Velocity, RPM, Temperature, Density)
    so that (Reynolds, Mach) can be calculated and used
    as reference for the Qprop's analytical fit at each
    required section.

    Parameters
    ----------

        LiftingLine : zone
            liftin-line object

        PyZonePolars : list
            list of CGNS zone containing 2D polars

        OutputQpropFilename : str
            output file name of qprop

        NBlades : int
            number of blades

        Velocity : float
            advance velocity (m/s)

        RPM : float
            rotation speed of the propeller (revolutions per minute)

        Temperature : float
            Temperature of air (Kelvin)

        Density : float
            Density of air (kg /m3)

        LinearFitAoACLmaxFactor : float
            maximum value of CL to consider for linear fitting of :math:`c_L(\\alpha)`
            linear function employed by QProp

        REexps : :py:class:`list` of 2 :py:class:`float`
            Employ first Reynolds exponent if Reynolds is  less than **RElow**,
            or employ second Reynolds exponent otherwise

        RElow : float
            determine the threshold of usage of **REexps**

    Returns
    -------

        None : None
            write qprop file named after **OutputQpropFilename**

    """
    import scipy.optimize as so

    # Build required interpolator objects
    PolarsInterpolatorDict = LL.buildPolarsInterpolatorDict(PyZonePolars)

    # Get some existing variables:
    r, Chord, Twist, AoA, Mach, Reynolds, s, \
    Cl, Cd = J.getVars(LiftingLine,['Span','Chord','Twist','AoA', 'Mach', 'Reynolds', 's', 'Cl', 'Cd'])

    # Compute some constants:
    # Sutherland's law
    Mu=Mus*((Temperature/Ts)**0.5)*((1.+Cs/Ts)/(1.+Cs/Temperature))
    Pressure = Density * Rgp * Temperature # State
    Omega = RPM*np.pi/30.   # rotational speed
    Rmax = r.max()          # Tip radius
    SoundSpeed = (Gamma * Rgp * Temperature)**0.5


    # Approximate Reynolds and Mach numbers
    Wapprox = ((Omega*r)**2+Velocity**2)**0.5
    Reynolds[:] = Density * Wapprox * Chord / Mu
    Mach[:]  = Wapprox / SoundSpeed

    """ # DEPRECATED : TODO -> REMOVE !
    # Declare the arrays of Radius, Chord and Twist used by
    # Qprop at the section positions given by the airfoils
    # defined at Airfoils node contained in LiftingLine
    SectionsAbscissa = I.getNodeFromName(LiftingLine,'Abscissa')[1]
    QpropRadius = np.interp(SectionsAbscissa, s, r)
    QpropChord  = np.interp(SectionsAbscissa, s, Chord)
    QpropTwist  = np.interp(SectionsAbscissa, s, TwistDeg)
    QpropReynolds  = np.interp(SectionsAbscissa, s, Reynolds)
    QpropMach  = np.interp(SectionsAbscissa, s, Mach)
    """

    # Determines the minimal AoA range based upon the
    # PyZonePolar information. This range is used for
    # fitting the analytical form (linear for CL,
    # parabolic for CD) that will be used in Qprop
    NodeStr = I.getNodeFromName(LiftingLine,'PyZonePolarNames')[1]
    NodeStr = [k.decode("utf-8") for k in NodeStr]
    PyZonePolarNames = ''.join(NodeStr).split(' ')
    AoAMins, AoAMaxs = [], []
    for PolarName in PyZonePolarNames:
        PyZonePolar = [z for z in PyZonePolars if z[0]==PolarName][0]
        AoARange = I.getNodeFromName(PyZonePolar,'AngleOfAttack')[1]
        AoAMins += [np.array(AoARange).min()]
        AoAMaxs += [np.array(AoARange).max()]
    AoAMin, AoAMax = np.array(AoAMins).min(), np.array(AoAMaxs).max()

    # Determines the array of values used as "scattered"
    # data for the curve fitting
    NPtsAoA = 20
    AoARange = np.linspace(AoAMin, AoAMax, NPtsAoA)

    # At each station, compute the analytical form of the
    # airfoil's polar by fitting curve data
    SectionLines = []
    for i in range(len(r)):

        # --------- FIT LIFT-COEFFICIENT CURVE --------- #
        # Qprop uses the following analytical form,
        #
        #    CL(alpha)=(CL0+CL_a*alpha)/sqrt(1-Mach**2)
        #
        # where sqrt(1-Mach**2) is Prandtl-Meyer
        # compressibility factor, and (CL0,CL_a) are the
        # _incompressible_ line-fit of CL(alpha) polar.
        #
        # As here we provide already _compressible_ polars
        # at reference (Reynolds, Mach), we rather fit the
        # following curve:
        #
        #    CL(alpha)=(CL0+CL_a*alpha)*sqrt(1-Mach**2)
        #
        # so that the obtained CL0 and CL_a are indeed the
        # incompressible-equivalent coefficients such that
        # Prandtl-Meyer correction can be used by Qprop.

        # Prandtl-Meyer compressibility factor
        CompFac = np.sqrt(1-Mach[i]**2)

        # Declare containers for polar data
        CLcomp = AoARange * 0
        CDcomp = AoARange * 0

        # Compute compressible polar
        for ialpha in range(NPtsAoA):
            AoA[:] = AoARange[ialpha]
            LL._applyPolarOnLiftingLine(LiftingLine,PolarsInterpolatorDict)
            [C._initVars(LiftingLine, eq) for eq in ListOfEquations]

            CLcomp[ialpha] = Cl[i]
            CDcomp[ialpha] = Cd[i]

        # Incompressible-equivalent polar
        CL = CLcomp * CompFac
        CD = CDcomp * CompFac

        # Find CLmin, CLmax
        iCLmax = np.argmax(CL)
        CLmin, CLmax = CL.min(), CL[iCLmax]


        # FIND CL0, CLa :
        # Compute linear regression clipped to a user-
        # provided threshold w.r.t stall angle of attack
        linearizableData = AoARange < AoARange[iCLmax]*LinearFitAoACLmaxFactor
        for ild in range(len(linearizableData)):
            if not linearizableData[ild]:
                linearizableData[ild:] = False
                break

        # Linear CL function:
        def linearCLfunction(x,CL0,CLa):
            # x is the angle of attack in RADIANS
            return CL0 + CLa*x

        # Residual of the linear CL function:
        def residualCLfunction(coefs, x, y):
            # y is the CL
            return y - linearCLfunction(x,*coefs)

        InitialGuess = [0, 2*np.pi]
        AoARadLinearizable = AoARange[linearizableData]*np.pi/180.
        CLLinearizable = CL[linearizableData]
        (CL0, CLa), Cov = so.leastsq(residualCLfunction, InitialGuess, args=(AoARadLinearizable,CLLinearizable))


        # --------- FIT DRAG-COEFFICIENT CURVE --------- #
        # Qprop uses the following analytical form,
        # CD(CL,Re)=(CD0+CD2*(CL-CLCD0)**2)*(Re/REref)**REexp
        # where (Re/REref)**REexp is the Reynolds-correction
        # factor which includes:
        #  REref -> Reference Reynolds of the obtained polar
        #  REexp -> Usually takes the following values:
        #   REexpH=-0.7 -> most usual Reynolds conditions
        #   REexpL=-0.5 -> low-Reynolds conditions
        #   REexp2bis=-0.2 -> very high-turbulence conditns.
        #                  (large propellers)
        #   the user provides REexp=(REexpH,REexpL) as
        #   argument together with a low-Reynolds number
        #   threshold (RElow), used to switch from REexpH
        #   to REexpL.

        # FIND CD0, CD2, CLCD0
        # Parabolic CD(CL) function
        def parabolicCDfunction(x,CD0, CD2u, CD2l, CLCD0):
            # x is CL
            CD2 = x*0
            CD2[x>CLCD0]  = CD2u
            CD2[x<=CLCD0] = CD2l
            return CD0+CD2*(x-CLCD0)**2

        # Residual of the parabolic CD function:
        def residualCDfunction(coefs, x, y):
            # y is the CD
            return y -parabolicCDfunction(x,*coefs)


        InitialGuess = [0.032, 0.06,0.010,0.6]
        CDLinearizable = CD[linearizableData]
        (CD0,CD2u,CD2l,CLCD0), Cov = so.leastsq(residualCDfunction, InitialGuess, args=(CLLinearizable,CDLinearizable))

        # Determines REexp
        REexp = REexps[0] if Reynolds[i] > RElow else REexps[1]




        # DEBUG : Plot CL, CD curves:
        """
        import matplotlib.pyplot as plt
        fig, (ax,ax2) = plt.subplots(1,2,dpi=200)
        # CL
        ax.plot(AoARange,CL,'o',mfc='None')
        CLanalytic = (CL0+CLa*AoARadLinearizable)*np.sqrt(1-Mach[i]**2)
        ax.plot(AoARadLinearizable*180/np.pi,CLanalytic,'.-')
        ax.set_xlabel('Angle of Attack (deg)')
        ax.set_ylabel('$C_L$')
        ax.grid()
        # CD
        ax2.plot(CD,CL,'o',mfc='None')
        ax2.plot(parabolicCDfunction(CLanalytic,CD0, CD2u, CD2l, CLCD0),CLanalytic,'.-')
        ax2.set_xlabel('$C_D$')
        ax2.grid()
        plt.title('radius = %g m'%r[i])
        plt.show()
        """


        # All information is ready. Store the file line:
        #                  r  chord  beta [  CL0  CL_a   CLmin CLmax  CD0   CD2u   CD2l   CLCD0  REref  REexp ]
        # SectionLines += [" %0.12E  %0.12E  %0.12E  %0.12E  %0.12E  %0.12E  %0.12E  %0.12E  %0.12E  %0.12E  %0.12E  %0.12E  %0.12E \n"%(r[i],Chord[i],Twist[i], CL0, CLa, CLmin, CLmax, CD0, CD2u, CD2l, CLCD0, Reynolds[i],REexp)]
        SectionLines += [" %0.3f %0.3f %0.2f  %0.2f %0.3f %0.2f %0.2f %0.3f %0.3f %0.3f %0.3f %0.5E %g \n"%(r[i],Chord[i],Twist[i], CL0, CLa, CLmin, CLmax, CD0, CD2u, CD2l, CLCD0, Reynolds[i],REexp)]

    # Write the output file:
    with open(OutputQpropFilename,'w') as f:

        # Title
        f.write('Qprop file built from LiftingLine named: "%s"\n\n'%LiftingLine[0])

        # Tells number of blades and Rmax
        f.write(' %g     %0.3f   ! Nblades  [ R ] \n\n'%(NBlades,r.max()))

        # Default CL and CD polars. In theory this is never
        # used, as polars informations are given at each
        # section
        f.write(' 0  %g   ! CL0     CL_a\n'%(2*np.pi))
        f.write(' -0.3  1.2   ! CLmin   CLmax\n\n')

        f.write(' 0.028  0.050 0.020  0.5   !  CD0    CD2u   CD2l   CLCD0\n')
        f.write(' 70000   -0.7              !  REref  REexp\n\n')


        # Conversion units. This is never used, as we use SI
        f.write(' 1.0     1.0      1.0  !  Rfac   Cfac   Bfac\n')
        f.write(' 0.      0.       0.   !  Radd   Cadd   Badd\n\n')


        # This is actually the interesting part:
        # Definition of geometry and polar coefficients
        f.write('#  r  chord  beta [  CL0  CL_a   CLmin CLmax  CD0   CD2u   CD2l   CLCD0  REref  REexp ]\n')
        for sl in SectionLines: f.write(sl)

    return None

def extractQprop(QpropResult):
    """
    Given a Qprop result string (as obtained from :py:func:`computeQprop`),
    build a results dictionary

    Parameters
    ----------

        QpropResult : str
            as obtained from :py:func:`computeQprop`

    Returns
    -------

        Results : dict
            dictionary containing all predictions of QProp
    """

    import re
    def scan(line,OutputType=float, RegExpr=r'[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?'):
        scanned = re.findall(RegExpr,line)
        return [OutputType(item) for item in scanned]


    Lines = QpropResult.splitlines()

    Lines = [l.decode("utf-8") for l in Lines]

    ResultDict = {} # Here the results will be stored

    # Get integral efforts
    for il in range(len(Lines)):
        if "#  V(m/s)    rpm      Dbeta      T(N)       Q(N-m)    Pshaft(W)" in Lines[il]:
            DeltaLine = 2 if "Voltage convergence failed" in Lines[il+1] else 1
            IntegralQties = scan(Lines[il+DeltaLine],float)

            if len(IntegralQties) != 15:
                raise ValueError('The amount of integral quantities is %d, and expected 15.'%len(IntegralQties))
            ResultDict['Velocity'] = IntegralQties[0]
            ResultDict['RPM']      = IntegralQties[1]
            ResultDict['Pitch']    = IntegralQties[2]
            ResultDict['Thrust']   = IntegralQties[3]
            ResultDict['Torque']   = IntegralQties[4]
            ResultDict['Power']    = IntegralQties[5]
            ResultDict['PropulsiveEfficiency'] = IntegralQties[7]
            ResultDict['AdvanceRatio'] = IntegralQties[8]
            ResultDict['CT']           = IntegralQties[9]
            ResultDict['CP']           = IntegralQties[10]
            ResultDict['cl_avg']       = IntegralQties[13]
            ResultDict['cd_avg']       = IntegralQties[14]

            break

    Data = []
    for il in range(il+1,len(Lines)):
        if '#  radius   chord   beta' in Lines[il]:
            for ill in range(il+1,len(Lines)):
                Data += [scan(Lines[ill],float)]
            Data = np.array(Data)

    VariablesNames = ['Radius', 'Chord', 'Twist', 'Cl', 'Cd', 'Reynolds', 'Mach', 'LocalInducedEfficiency', 'LocalPropulsiveEfficiency', 'VelocityAxial', 'Aswirl', 'AdvanceWakeVelocity']
    for i in range(len(VariablesNames)):
        # print  (i)
        # print (VariablesNames[i])
        # print (Data)
        # print (Data.shape)
        ResultDict[VariablesNames[i]] = Data[:,i]


    return ResultDict

def designPropellerAdkins(LiftingLine, PolarsInterpolatorDict, NBlades=None,
                          Velocity=None, RPM=None, Temperature=None,
                          Density=None, TipLosses='Adkins',
                          ListOfEquations=[], Constraint='Thrust',
                          ConstraintValue=1000., AirfoilAim='maxClCd',
                          AimValue=None, AoASearchBounds=(-2,8), SmoothAoA=False,
                          itMaxAoAsearch=3):
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

    NPts = C.getNPts(LiftingLine)

    # Read existing flight conditions.
    # For each user-provided not None condition,
    # replace existing one in LiftingLine object
    ProvidedFlightVars = dict(NBlades=NBlades, Velocity=Velocity, RPM=RPM, Density=Density, Temperature=Temperature)
    FlightVars = J.get(LiftingLine,'.Conditions')
    for fvname in ProvidedFlightVars:
        if ProvidedFlightVars[fvname] is not None:
            FlightVars[fvname] = ProvidedFlightVars[fvname]
    J.set(LiftingLine, '.Conditions', **FlightVars)

    # Use local variable names based on
    # updated FlightVars
    NBlades     = FlightVars['NBlades']
    Velocity    = FlightVars['Velocity']
    RPM         = FlightVars['RPM']
    Density     = FlightVars['Density']
    Temperature = FlightVars['Temperature']

    # invoke variables in Lifting Line
    ListOfVariables = ['phiRad','VelocityMagnitudeLocal', 'VelocityAxial', 'VelocityTangential', 'VelocityInducedAxial','VelocityInducedTangential','a','aP','F','dFx','dMx']
    v = J.invokeFieldsDict(LiftingLine,ListOfVariables)

    # Get some existing variables:
    r, Chord, TwistDeg, AoADeg, Mach, Reynolds, \
    Cl, Cd = J.getVars(LiftingLine,['Span','Chord','Twist','AoA', 'Mach', 'Reynolds', 'Cl', 'Cd'])

    # Complementary variables
    v2 = J.getVars2Dict(LiftingLine,['Span', 'Chord', 'Twist', 'Cl', 'Cd'])
    v.update(v2)

    # Compute some constants:
    # Sutherland's law
    Mu=Mus*((Temperature/Ts)**0.5)*((1.+Cs/Ts)/(1.+Cs/Temperature))
    Pressure = Density * Rgp * Temperature # State
    Omega = RPM*np.pi/30.   # rotational speed
    Rmax = r.max()          # Tip radius
    SoundSpeed = (Gamma * Rgp * Temperature)**0.5
    ChordMin = 0.001*Rmax


    # Declare additional variables (not stored in LiftingLine)
    xi    = r/Rmax
    lambd = Velocity / (Omega * Rmax)

    # Compute Approximate Reynolds, Mach, and AoA,
    # based on initial "guess" of Chord distribution
    Wapprox = ((Omega*r)**2+Velocity**2)**0.5
    Reynolds[:] = Density * Wapprox * Chord / Mu
    Mach[:]  = Wapprox / SoundSpeed # Approximate

    if AirfoilAim == 'AoA':
        LL._applyPolarOnLiftingLine(LiftingLine,PolarsInterpolatorDict)
        [C._initVars(LiftingLine, eq) for eq in ListOfEquations]
    else:
        AoADeg[:] = 0.
        LL._findOptimumAngleOfAttackOnLiftLine(LiftingLine,PolarsInterpolatorDict, Aim=AirfoilAim, AimValue=AimValue, AoASearchBounds=AoASearchBounds, ListOfEquations=ListOfEquations)



    if SmoothAoA:
        WindowFilter = int(len(AoADeg)*0.6)
        if WindowFilter%2 ==0: WindowFilter -=1
        FilterOrder  = 4

        import scipy.signal as ss
        AoADeg[:] = ss.savgol_filter(AoADeg,WindowFilter,FilterOrder)
        LL._applyPolarOnLiftingLine(LiftingLine,PolarsInterpolatorDict)
        [C._initVars(LiftingLine, eq) for eq in ListOfEquations]

    def computeDistributionsFromZeta(zeta):

        xfact = Omega * r / Velocity

        v['phiRad'][:] = phi = np.arctan( (1+zeta/2.)/xfact ) # Eqn. 8
        # v['phiRad'][:] = phi =  np.arctan( lambd*(1+zeta/2.)/xi ) # Eqns. 20 and 21

        # Compute tip losses factor, F
        v['F'][:] = F = TipLossFactor(NBlades,Velocity,Omega,phi,r,Rmax, kind=TipLosses)

        """ # Not good result: minimum chord
        # By Luis:
        W = v['VelocityMagnitudeLocal']
        Gmin = (W*ChordMin*Cl*NBlades)/(4*np.pi*lambd*Velocity*Rmax*zeta)
        G = np.maximum(F * xfact * np.cos(v['phiRad']) * np.sin(v['phiRad']),Gmin)
        """
        G = F * xfact * np.cos(v['phiRad']) * np.sin(v['phiRad']) # Eqn. 5

        Wc = 4*np.pi*lambd*G*Velocity*Rmax*zeta/(Cl*NBlades) # Eqn. 16
        Reynolds[:] = Density * Wc / Mu
        if any(Reynolds) <= 0:
            print ('zeta')
            print (zeta)
            print ('G')
            print (G)
            print ('Wc')
            print (Wc)
            print ('Reynolds')
            print (Reynolds)
            raise ValueError('Negative Reynolds found')

        # Mach[:]  = Wc / (SoundSpeed * Chord) # Exact but bugged
        Mach[:]  = Wapprox / SoundSpeed # Approximate

        if AirfoilAim == 'AoA' or it >= itMaxAoAsearch:
            LL._applyPolarOnLiftingLine(LiftingLine,PolarsInterpolatorDict)
            [C._initVars(LiftingLine, eq) for eq in ListOfEquations]

        else:
            LL._findOptimumAngleOfAttackOnLiftLine(LiftingLine,PolarsInterpolatorDict, Aim=AirfoilAim, AimValue=AimValue, AoASearchBounds=AoASearchBounds, ListOfEquations=ListOfEquations)

        if SmoothAoA and it < itMaxAoAsearch:
            # import matplotlib.pyplot as plt
            # plt.plot(Chord,AoADeg)
            import scipy.signal as ss
            AoADeg[:] = ss.savgol_filter(AoADeg,WindowFilter,FilterOrder)
            LL._applyPolarOnLiftingLine(LiftingLine,PolarsInterpolatorDict)
            [C._initVars(LiftingLine, eq) for eq in ListOfEquations]


        if any(np.isnan(zeta)):
            print ('"zeta" is NaN')
            C.convertPyTree2File(LiftingLine,'debug.cgns')
            sys.exit()


        # Determine interference factors (Eqn.7)
        varepsilon = Cd / Cl
        if any(np.isnan(varepsilon)):
            print ('"Cd/Cl" is NaN')
            C.convertPyTree2File(LiftingLine,'debug.cgns')
            sys.exit()

        v['a'][:] = (zeta/2.)*np.cos(v['phiRad'])*np.cos(v['phiRad'])*(1-varepsilon*np.tan(v['phiRad']))
        v['aP'][:] = (zeta/(2.*xfact))*np.cos(v['phiRad'])*np.sin(v['phiRad'])*(1+varepsilon/np.tan(v['phiRad']))

        if any(np.isnan(v['a'])):
            print ('Interference factor "a" is NaN')
            C.convertPyTree2File(LiftingLine,'debug.cgns')
            sys.exit()

        # Determine actual LocalVelocity (Eqn.17)
        v['VelocityMagnitudeLocal'][:] = W = Velocity*(1+v['a'])/np.sin(v['phiRad'])

        v['VelocityAxial'][:]  = Velocity*(1+v['a'])
        v['VelocityTangential'][:] = Omega*r*(1 - v['aP'])

        v['VelocityInducedAxial'][:]  = Velocity*(v['a'])
        v['VelocityInducedTangential'][:]  = Omega*r*(- v['aP'])

        # Update Chord distribution
        Chord[:] = np.maximum(Wc/W,ChordMin)

        # Update blade twist
        TwistDeg[:] = AoADeg + np.rad2deg(v['phiRad'])

        # Determine derivatives (Eqns.11 a--d)
        I1 = 4*xi*G*(1-varepsilon*np.tan(v['phiRad']))
        I2 = lambd*(0.5*I1/xi)*(1+varepsilon/np.tan(v['phiRad']))*np.sin(v['phiRad'])*np.cos(v['phiRad'])
        J1 = 4*xi*G*(1+varepsilon/np.tan(v['phiRad']))
        J2 = 0.5*J1*(1-varepsilon*np.tan(v['phiRad']))*np.cos(v['phiRad'])*np.cos(v['phiRad'])

        # Integrate derivatives in order to get Power, Thrust and new zeta
        I1int = np.trapz(I1,xi)
        I2int = np.trapz(I2,xi)
        J1int = np.trapz(J1,xi)
        J2int = np.trapz(J2,xi)

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

        if np.isnan(Power):
            print ('Power is NaN')
            C.convertPyTree2File(LiftingLine,'debug.cgns')
            sys.exit()
        Residual = NewZeta - zeta
        PropulsiveEfficiency = Velocity*Thrust/Power
        return Residual, Power, Thrust, PropulsiveEfficiency


    InitialGuess = Chord

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
        if DEBUG: print ('it=%d | Thrust=%g, Power=%g, PropulsiveEfficiency=%g | L2 res = %g'%(it,Thrust,Power,PropulsiveEfficiency, L2Residual))
        zeta1 = Residual+zeta0
        RelaxFactor = 0.
        zeta0 = (1.0-RelaxFactor)*zeta1+RelaxFactor*zeta0

    # Prepare output
    DesignPitch = LL.resetPitch(LiftingLine)

    DictOfIntegralData = computeAxialLoads(VariablesDict=v, ConditionsDict=FlightVars)
    DictOfIntegralData['Pitch'] = DesignPitch
    J.set(LiftingLine,'.Loads',**DictOfIntegralData)

    return DictOfIntegralData

def designPropellerPatterson(LiftingLine, NBlades, Velocity, RPM, PyZonePolars=None,
        Temperature = 273.+15., Density=1.225, TipLosses='Adkins',
        RadiusCoordinate='x', Constraint='Thrust', ConstraintValue=1000.,
        OptionalStep1=False, OptionalStep2=False,
        MaximumChangeTangentialFactor=1.25, AirfoilAim='maxClCd', AimValue=None,
        AoASearchBounds=(-2,8), SmoothAoA=False):
    """
    .. warning:: little advantage has been obtained by using Patterson's
        algorithm for designing propellers. This function will documentation will
        not be updated. However, most of its parameters can be understood by
        similarity to :py:func:`designPropellerAdkins` function.
    """
    import scipy.optimize as so

    # Build Polars' interpolator functions
    PolarsInterpolatorDict=LL.buildPolarsInterpolatorDict(PyZonePolars)

    # Get radius coordinate from LiftingLine
    if RadiusCoordinate.lower()[-1] == 'x':
        r = J.getx(LiftingLine)
    elif RadiusCoordinate.lower()[-1] == 'y':
        r = J.gety(LiftingLine)
    elif RadiusCoordinate.lower()[-1] == 'z':
        r = J.getz(LiftingLine)
    else:
        raise ValueError('RadiusCoordinate==%s not recognized. Use "x", "y", or "z"'%RadiusCoordinate)

    # Compute some constants:
    # Sutherland's law
    Mu=Mus*((Temperature/Ts)**0.5)*((1.+Cs/Ts)/(1.+Cs/Temperature))
    Pressure = Density * Rgp * Temperature # State
    Omega = RPM*np.pi/30.   # rotational speed
    Rmax = r.max()          # Tip radius
    SoundSpeed = (Gamma * Rgp * Temperature)**0.5

    # invoke variables in Lifting Line
    ListOfVariables = ['phiRad','VelocityMagnitudeLocal', 'VelocityAxial', 'VelocityTangential', 'VelocityInducedAxial','VelocityInducedTangential','a','aP','F']
    v = J.invokeFieldsDict(LiftingLine,ListOfVariables)

    # Get some existing variables:
    Chord, TwistDeg, AoADeg, Mach, Reynolds, \
    Cl, Cd, Cm = J.getVars(LiftingLine,['Chord','Twist','AoA', 'Mach', 'Reynolds', 'Cl', 'Cd', 'Cm'])


    xi = r / Rmax


    if AirfoilAim == 'AoA':
        LL._applyPolarOnLiftingLine(LiftingLine,PolarsInterpolatorDict)
        [C._initVars(LiftingLine, eq) for eq in ListOfEquations]
    else:
        LL._findOptimumAngleOfAttackOnLiftLine(LiftingLine,PolarsInterpolatorDict, Aim=AirfoilAim, AimValue=AimValue, AoASearchBounds=AoASearchBounds)



    if SmoothAoA:
        WindowFilter = int(len(AoADeg)*0.6)
        if WindowFilter%2 ==0: WindowFilter -=1
        FilterOrder  = 4

        import scipy.signal as ss
        AoADeg[:] = ss.savgol_filter(AoADeg,WindowFilter,FilterOrder)
        LL._applyPolarOnLiftingLine(LiftingLine,PolarsInterpolatorDict)
        [C._initVars(LiftingLine, eq) for eq in ListOfEquations]




    def computeDesignIteration(phi,aValue):

        v['phiRad'][:] = phi

        # Compute tip losses factor, F
        v['F'][:] = F = TipLossFactor(NBlades,Velocity,Omega,phi,r,
            Rmax*1.035, # Patterson's bidouille (cf AIAA2016 p.7)
            kind=TipLosses)

        v['a'][:] = aValue

        # OptionalStep1=True (as proposed by Patterson) has
        # little sense, as avoids having uniform profile in
        # many cases !
        if OptionalStep1: v['a'][:] = v['a'][:]/np.maximum(F,1e-3) # Eqn. 15.

        a = v['a'][:]

        # Eqns. 11--12
        aP = v['aP'][:]
        Discr = 1- (4*Velocity**2*(1+a)*a/(Omega**2*r**2))
        for i in range(len(aP)):
            if Discr[i] < 0 or aP[i]>0.5:
                aP[i] = 0.5
                a[i] = (-1 + np.sqrt((1+ (4*Omega**2*r[i]**2*(1-aP[i])*aP[i])/(Velocity**2) )))/2.
            else:
                aP[i] =  (1-np.sqrt(Discr[i])) / 2.

            if aP[i] > 0.5:
                raise ValueError('Unexpected 2.')

        if OptionalStep2:
            for i in range(1,len(r)):
                Slope_aP = (aP[i]-aP[i-1])/(xi[i]-xi[i-1])
                if Slope_aP > MaximumChangeTangentialFactor:
                    aP[i] = aP[i-1]+MaximumChangeTangentialFactor*(xi[i]-xi[i-1])
                    a[i] = (-1 + np.sqrt((1+ (4*Omega**2*r[i]**2*(1-aP[i])*aP[i])/(Velocity**2) )))/2.


        # Determine actual LocalVelocity
        v['VelocityMagnitudeLocal'][:] = W = Velocity*(1+v['a'])/np.sin(v['phiRad'])

        v['VelocityAxial'][:]  = Velocity*(1+v['a'])
        v['VelocityTangential'][:] = Omega*r*(1 - v['aP'])

        v['VelocityInducedAxial'][:]  = Velocity*(v['a'])
        v['VelocityInducedTangential'][:]  = Omega*r*(- v['aP'])

        oldPhi = np.copy(phi,order='F')
        newPhi = np.maximum(0.1 * np.pi/180.,np.arctan( v['VelocityAxial'] / v['VelocityTangential'] ))
        v['phiRad'][:] = newPhi

        # Update blade twist
        TwistDeg[:] = AoADeg + np.rad2deg(v['phiRad'])

        #  BEWARE : here Chord is not up-to-date !
        Reynolds[:] = Density * W * Chord / Mu
        Mach[:] = W / SoundSpeed


        if AirfoilAim == 'AoA':
            LL._applyPolarOnLiftingLine(LiftingLine,PolarsInterpolatorDict)
            [C._initVars(LiftingLine, eq) for eq in ListOfEquations]

        else:
            LL._findOptimumAngleOfAttackOnLiftLine(LiftingLine,PolarsInterpolatorDict, Aim=AirfoilAim, AimValue=AimValue, AoASearchBounds=AoASearchBounds, ListOfEquations=ListOfEquations)

        if SmoothAoA:

            # import matplotlib.pyplot as plt
            # plt.plot(Chord,AoADeg)

            import scipy.signal as ss
            AoADeg[:] = ss.savgol_filter(AoADeg,WindowFilter,FilterOrder)
            LL._applyPolarOnLiftingLine(LiftingLine,PolarsInterpolatorDict)
            [C._initVars(LiftingLine, eq) for eq in ListOfEquations]


        # Compute Chord (Eqn. 20)
        v['F'][:] = F = TipLossFactor(NBlades,Velocity,Omega,phi,r,Rmax,kind=TipLosses)
        Chord[:] = 8*np.pi*r*Velocity**2*(1+a)*a*F/ (NBlades*W**2*(Cl*np.cos(newPhi)-Cd*np.sin(newPhi)))


        dFx = 0.5*Density*W**2*NBlades*Chord*(Cl*np.cos(v['phiRad'])-Cd*np.sin(v['phiRad']))
        dMx = 0.5*Density*W**2*NBlades*Chord*(Cl*np.sin(v['phiRad'])+Cd*np.cos(v['phiRad']))*r
        Thrust = Fx = np.trapz(dFx,r)
        Torque = Mx = np.trapz(dMx,r)
        Power = Omega*Mx

        n = RPM/60.
        d = Rmax*2
        CTpropeller = Thrust / (Density * n**2 * d**4)
        CPpropeller = Power  / (Density * n**3 * d**5)
        Jparam = Velocity / (n*d)
        FigureOfMeritPropeller = np.sqrt(2./np.pi)* CTpropeller**1.5 / CPpropeller

        PropEff = Velocity*Fx/Power

        DictOfIntegralData = dict(Thrust=Thrust,Power=Power,PropulsiveEfficiency=PropEff,J=Jparam,FigureOfMeritPropeller=FigureOfMeritPropeller)


        Residual = newPhi - oldPhi

        return Residual, DictOfIntegralData




    # Initial Guess
    v['phiRad'] = np.arctan( Velocity/(Omega * r) )


    def designGivenInducedFactor(aValue):
        itmax = 100
        L2Residual, L2ResidualTol = 1e8,1e-8
        phi0 = v['phiRad']
        it = -1
        while it<itmax and L2Residual>L2ResidualTol:
            it += 1
            Residual, res = computeDesignIteration(phi0,aValue)
            L2Residual = np.sqrt(Residual.dot(Residual))
            # print('it=%d | Thrust=%g N; Power=%g W; eta=%g | L2 res = %g'%(it,res['Thrust'],res['Power'],res['PropulsiveEfficiency'], L2Residual))
            phi1 = Residual+phi0
            phi0 = phi1

        EffortResidual = res[Constraint]-ConstraintValue

        return EffortResidual


    x0 = 0.1
    x1 = 0.2

    sol=so.root_scalar(designGivenInducedFactor, x0=x0, x1=x1, method='secant')
    if sol.converged:

        Residual, res = computeDesignIteration(v['phiRad'],sol.root)
        print('Patterson Design: Thrust=%g N; Power=%g W; eta=%g'%(res['Thrust'],res['Power'],res['PropulsiveEfficiency']))
    else:
        print ("Not converged at section %d"%i)
        print (sol)


    return LiftingLine

def designDirectBEMT(LiftingLine, PolarsInterpolatorsDict,
            LiftingLineInterpolator=None,
            NBlades=None, Velocity=None, RPM=None, Density=None,
            Temperature=None, ListOfEquations=[], TipLosses='Adkins',
            Constraint='Thrust',ConstraintValue=1000.,
            Objective='PropulsiveEfficiency', ChordPairs=None, TwistPairs=None,
            bounds=None, method='L-BFGS-B',OptimOptions=None,
            AttemptCommandGuess=[[5.,15.],[0.,20.],[0.,25.],[0.,30.]],
            ValueTol=10., AcceptableTol=50., TwistScale=0.1,
            makeOnlyInitialGuess=False, stdout=None):
    """
    .. warning:: this function is experimental. Poor convergence rates has been
        obtained using scipy's optimizers.

    **ChordPairs** and **TwistPairs** are :math:`2 \\times Nparam` numpy arrays.
    The second row stands for the relative span position
    and the first row stands for its corresponding value.
    E.g :

    ::

        ChordPairs = np.array([[0.2, 0.1],  # VALUES
                               [0.1, 1.0]]) # REL SPAN POSIT.

    means that Chord distribution has 2 parameters,
    located at :math:`r/R=0.1` whith initial value of
    :math:`c=0.2`, and the other one located at blade tip :math:`r/R=1`
    with initial value of :math:`c=0.1`.

    bounds is a tuple of tuples. Each tuple is a two-float
    standing for the minimum and maximum value of each design
    parameter.

    """
    import scipy.optimize as so

    # Invoke stdout file (overrides existing one, if any)
    if isinstance(stdout,str):
        with open(stdout,'w') as f:
            f.write('')

    # Read existing flight conditions.
    # For each user-provided not None condition,
    # replace existing one in LiftingLine object
    ProvidedFlightVars = dict(NBlades=NBlades, Velocity=Velocity, RPM=RPM, Density=Density, Temperature=Temperature)
    FlightVars = J.get(LiftingLine,'.Conditions')
    for fvname in ProvidedFlightVars:
        if ProvidedFlightVars[fvname] is not None:
            FlightVars[fvname] = ProvidedFlightVars[fvname]
    J.set(LiftingLine, '.Conditions', **FlightVars)

    # Use local variable names based on
    # updated FlightVars
    NBlades     = FlightVars['NBlades']
    Velocity    = FlightVars['Velocity']
    RPM         = FlightVars['RPM']
    Density     = FlightVars['Density']
    Temperature = FlightVars['Temperature']


    v = J.getVars2Dict(LiftingLine,['s','Span','Chord','Twist', 'MinimumThickness','RelativeThickness'])

    isChordLimited = True if v['MinimumThickness'] is not None and v['RelativeThickness'] is not None else False

    if ChordPairs is not None:
        NchordPars = len(ChordPairs[0,:])
    else:
        NchordPars = 0

    if TwistPairs is not None:
        NtwistPars = len(TwistPairs[0,:])
    else:
        NtwistPars = 0

    if bounds is not None:
        Nbound = len(bounds)
        if Nbound != NchordPars + NtwistPars:
            raise AttributeError('Mismatch between number of bounds and number of design parameters.')
        # Scale Twist bounds
        for b in bounds[NchordPars:]:
            b[0] *= TwistScale
            b[1] *= TwistScale


    def _deformBlade__():

        # Deform chord
        if ChordPairs is not None:
            r = v['Span']
            RelSpan = r/r.max()

            ChordVals = ChordPairs[0,:]
            ChordSpan = ChordPairs[1,:]

            if isChordLimited:
                RelThick = J.interpolate__(ChordSpan, RelSpan, v['RelativeThickness'], Law='interp1d_linear')
                MinThick = J.interpolate__(ChordSpan, RelSpan, v['MinimumThickness'], Law='interp1d_linear')
                ChordVals = np.maximum(ChordVals,MinThick/RelThick)

            NewChord = J.interpolate__(RelSpan, ChordSpan, ChordVals, Law='akima')

            # Make sure constraint is well respected
            if isChordLimited:
                v['Chord'][:] = np.maximum(NewChord,v['MinimumThickness']/v['RelativeThickness'])
            else:
                v['Chord'][:] = NewChord

        # Deform twist
        if TwistPairs is not None:
            TwistVals = TwistPairs[0,:]
            TwistSpan = TwistPairs[1,:]

            # # Enable this for forcing a pole on reference
            # TwistBase = TwistSpan < ZeroPitchRelativeSpan
            # TwistTip  = TwistSpan > ZeroPitchRelativeSpan
            # TwistVals = np.hstack((TwistVals[TwistBase],0.0,TwistVals[TwistTip]))
            # TwistSpan = np.hstack((TwistSpan[TwistBase],ZeroPitchRelativeSpan,TwistSpan[TwistTip]))

            NewTwist = J.interpolate__(RelSpan, TwistSpan, TwistVals, Law='akima')

            v['Twist'][:] = NewTwist

            LL.resetPitch(LiftingLine)

        return None


    def costFunction__(x):

        if NchordPars > 0: ChordPairs[0,:] = x[:NchordPars]
        if NtwistPars > 0: TwistPairs[0,:] = x[NchordPars:NtwistPars+NchordPars]/TwistScale

        _deformBlade__()

        Results = computeBEMT(LiftingLine, PolarsInterpolatorsDict,  kind='Drela', TipLosses=TipLosses, Constraint=Constraint, ConstraintValue=ConstraintValue, AttemptCommandGuess=AttemptCommandGuess, ValueTol=ValueTol)

        cost = -Results[Objective]

        Acceptable = True if abs(Results[Constraint]-ConstraintValue) < AcceptableTol else False

        Str = ''
        for ix in range(NchordPars):
            Str += ' %6.3f'%x[ix]
        for ix in range(NchordPars,NchordPars+NtwistPars):
            Str += ' %6.3f'%(x[ix]/TwistScale)

        PRNT = '%8s  %10.2f  %8.6f  %2d  %5.3f x: %s'%(Acceptable,Results[Constraint],Results[Objective],Results['Attempts'], Results['Pitch'], Str)
        if Acceptable:
            if BestObjec[0] <= Results[Objective]:
                BestObjec[0] = Results[Objective]
                BestTrim[0]  = Results['Pitch']
                BestGuess[:] = x
                PRNT = '%s%s%s'%(GREEN,PRNT,ENDC)
        else:
            cost = 0.
            PRNT = '%s%s%s'%(FAIL,PRNT,ENDC)


        if isinstance(stdout,str):
            with open(stdout,'a') as f:
                f.write('%s\n'%PRNT)

        print(PRNT)


        return cost

    # Launch optimization process
    x0 = np.zeros(NchordPars+NtwistPars,dtype=np.float64)

    # Global variables
    BestTrim  = np.array([20.0])
    BestObjec = np.array([0.0])
    BestGuess = x0*1.

    if NchordPars > 0: x0[:NchordPars] = ChordPairs[0,:]
    if NtwistPars > 0: x0[NchordPars:NtwistPars+NchordPars] = TwistPairs[0,:]*TwistScale

    if makeOnlyInitialGuess:
        if NchordPars > 0: ChordPairs[0,:] = x0[:NchordPars]
        if NtwistPars > 0: TwistPairs[0,:] = x0[NchordPars:NtwistPars+NchordPars]/TwistScale

    else:
        NcharObj = len(Objective)
        LabelObjective = '%s.'%Objective[:11] if NcharObj >= 11 else Objective

        DesignLabels = 'Acceptable  %s  %s  Att.  Pitch'%(Constraint,LabelObjective)
        if isinstance(stdout,str):
            with open(stdout,'a') as f:
                f.write('%s\n'%DesignLabels)
        print(DesignLabels)

        solution = so.minimize(costFunction__,x0,
            method=method,
            bounds=bounds,
            options=OptimOptions,
            )

        # compute solution point for final output
        if NchordPars > 0: ChordPairs[0,:] = solution.x[:NchordPars]
        if NtwistPars > 0: TwistPairs[0,:] = solution.x[NchordPars:NtwistPars+NchordPars]/TwistScale


        SolutionPrint = 'Solution:\n%s'%str(solution.x)
        if isinstance(stdout,str):
            with open(stdout,'a') as f:
                f.write('%s\n'%SolutionPrint)
        print(SolutionPrint)

    _deformBlade__()

    Results = computeBEMT(LiftingLine, PolarsInterpolatorsDict,  kind='Drela', TipLosses=TipLosses, Constraint=Constraint, ConstraintValue=ConstraintValue, AttemptCommandGuess=AttemptCommandGuess, ValueTol=ValueTol)


    SolutionPrint = 'BestGuess:\n%s\nBestObjec:\n%s'%(str(BestGuess),str(BestObjec))
    if isinstance(stdout,str):
        with open(stdout,'a') as f:
            f.write('%s\n'%SolutionPrint)
    print(SolutionPrint)

    # Use best candidate if guessed solution is poor
    if BestObjec > Results[Objective] and not makeOnlyInitialGuess:
        MSG = 'Poor optimization converged result (%g). Using BestGuess instead (%g)'%(Results[Objective],BestObjec)
        if isinstance(stdout,str):
            with open(stdout,'a') as f:
                f.write('%s\n'%MSG)
        print(MSG)

        if NchordPars > 0: ChordPairs[0,:] = BestGuess[:NchordPars]
        if NtwistPars > 0: TwistPairs[0,:] = BestGuess[NchordPars:NtwistPars+NchordPars]/TwistScale

        _deformBlade__()

        Results = computeBEMT(LiftingLine, PolarsInterpolatorsDict,  kind='Drela', TipLosses=TipLosses, Constraint=Constraint, ConstraintValue=ConstraintValue, AttemptCommandGuess=AttemptCommandGuess, ValueTol=ValueTol)

    return Results

def computeBEMT(LiftingLine, PolarsInterpolatorDict, model='Adkins',
                LiftingLineInterpolator=None, NBlades=None,
                Velocity=None, RPM=None, Density=None,Temperature=None,
                ListOfEquations=[], TipLosses='Adkins', Constraint='Pitch',
                ConstraintValue=0., ValueTol=1.0,
                AttemptCommandGuess=[[0.,25.],[5.,40.]], CommandType='Pitch',
                PitchIfTrimCommandIsRPM=0., FailedAsNaN=False):
    """
    .. danger:: this function is beeing DEPRECATED. Please use:
        :py:func:`computeBEMTaxial3D` function instead.

        In future, both functions will be totally merged.
    """
    import scipy.optimize as so

    LiftingLine, = I.getZones(LiftingLine)

    NPts = C.getNPts(LiftingLine)

    if Constraint == 'Pitch':
        Pitch = ConstraintValue
        CommandType = 'Pitch'
        Trim = False
    elif CommandType == 'RPM':
        Pitch = PitchIfTrimCommandIsRPM
        Trim = True
    else:
        Pitch = 0.
        Trim = True

    # Read existing flight conditions.
    # For each user-provided not None condition,
    # replace existing one in LiftingLine object
    ProvidedFlightVars = dict(NBlades=NBlades, Velocity=Velocity, RPM=RPM,
                        Density=Density, Temperature=Temperature, Pitch=Pitch)
    FlightVars = J.get(LiftingLine,'.Conditions')
    for fvname in ProvidedFlightVars:
        if ProvidedFlightVars[fvname] is not None:
            FlightVars[fvname] = ProvidedFlightVars[fvname]
    J.set(LiftingLine, '.Conditions', **FlightVars)
    FlightVars = J.get(LiftingLine,'.Conditions')
    # Use local variable names based on
    # updated FlightVars


    NBlades     = FlightVars['NBlades']
    Velocity    = FlightVars['Velocity']
    RPM         = FlightVars['RPM']
    Pitch       = FlightVars['Pitch']
    Density     = FlightVars['Density']
    Temperature = FlightVars['Temperature']

    # invoke variables in Lifting Line
    ListOfVariables = [
    'psi', # Drela's angle in rad - see QProp's theory Fig. 4
    'a',   # axial interference factor
    'aP',  # tangential interference factor

    # -------------- see references:  HEENE
    'phiRad',                    # phi (Eqn. 2.1)
    'VelocityMagnitudeLocal',    # Vp (Fig. 2.2)
    'VelocityAxial',             # Vx (Fig. 2.2)
    'VelocityTangential',        # Vt (Fig. 2.2)
    'VelocityInducedAxial',      # va (Eqn. 2.7)
    'VelocityInducedTangential', # vr (Eqn. 2.8)
    'F',   # Tip loss factor ( see function TipLossFactor() )
    'dFx', # section's axial force  - for Thrust computation
    'dMx', # section's axial moment - for Power computation
    ]
    v = J.invokeFieldsDict(LiftingLine,ListOfVariables)

    # Get some existing variables:
    r, Chord, TwistDeg, AoADeg, Mach, Reynolds, \
    Cl, Cd = J.getVars(LiftingLine,['Span','Chord','Twist','AoA', 'Mach', 'Reynolds', 'Cl', 'Cd'])

    # Complementary variables
    v2 = J.getVars2Dict(LiftingLine,['Span', 'Chord', 'Twist', 'Cl', 'Cd','Cm'])
    v.update(v2)


    # Compute some constants:
    # Sutherland's law
    Mu=Mus*((Temperature/Ts)**0.5)*((1.+Cs/Ts)/(1.+Cs/Temperature))
    Pressure = Density * Rgp * Temperature # State
    Rmax = r.max()          # Tip radius
    SoundSpeed = np.sqrt(Gamma * Rgp * Temperature)

    # Declare additional variables (not stored in LiftLine)
    sigma = NBlades * Chord / (2*np.pi*r) # Blade solidity
    Omega = RPM[0]*np.pi/30.

    # Initialize Mach and Reynolds number with kinematics
    KinematicV  = np.sqrt( Velocity**2 + (Omega*r)**2 )
    Mach[:]     = KinematicV / SoundSpeed
    Reynolds[:] = Density * KinematicV * Chord / Mu

    def psi2AoA(psi,SectionIndex):
        i = SectionIndex
        Ua = Velocity
        Omega = FlightVars['RPM']*np.pi/30.
        Ut = Omega * r[i]
        U = np.sqrt(Ua**2+Ut**2)
        Wa = 0.5 * (Ua + U * np.sin(psi))
        Wt = 0.5 * (Ut + U * np.cos(psi))
        phi = np.arctan(Wa/Wt)
        AoA = TwistDeg[i] - np.rad2deg(phi)
        return AoA

    def solveHeene(x,SectionIndex):
        i = SectionIndex
        v['VelocityInducedAxial'][i]      = x[0]
        v['VelocityInducedTangential'][i] = x[1]

        Omega = RPM[0]*np.pi/30.


        # MISTAKE on ALPHA notation:
        # phi, alphaRad = solveAlphaPhiCoupling(Velocity[0],x[0],x[1],Omega,r[i],
        #                         np.deg2rad(TwistDeg[i]), tol=1.e-8, maxiter=100)
        # v['phiRad'][i] = phi
        # AoADeg[i] = np.rad2deg(alphaRad)
        # v['VelocityAxial'][i] = Velocity[0]*np.cos(alphaRad)+x[0]
        # v['VelocityTangential'][i] = Omega*r[i]-Velocity[0]*np.sin(alphaRad)-x[0]
        # # Eqn. 2.7
        # VxP = v['VelocityAxial'][i]-v['VelocityInducedAxial'][i]

        # Correction Luis 26/04/2021
        v['VelocityAxial'][i] = Velocity[0]+x[0]
        v['VelocityTangential'][i] = Omega*r[i]-x[1]
        VxP = Velocity[0]
        v['phiRad'][i] = phi = np.arctan2(v['VelocityAxial'][i],
                                    v['VelocityTangential'][i])
        AoADeg[i] = TwistDeg[i] - np.rad2deg(phi)


        # NB: W == Vp (Fig 2.2)
        v['VelocityMagnitudeLocal'][i] = W = np.sqrt(v['VelocityAxial'][i]**2 +
                                                v['VelocityTangential'][i]**2)


        Mach[i] = W / SoundSpeed
        Reynolds[i] = Density[0] * W * Chord[i] / Mu

        # Compute tip losses factor, F
        F = TipLossFactor(NBlades[0],Velocity[0],Omega,phi,r[i],Rmax, kind=TipLosses)
        v['F'][i] = F

        LL._applyPolarOnLiftingLine(LiftingLine,PolarsInterpolatorDict,
                                    InterpFields=['Cl', 'Cd'])
        # if i==0:
        #     print('Cl=%g , AoA=%g, Mach=%g | Va=%g , Vt=%g'%(Cl[i],AoADeg[i],Mach[i],v['VelocityAxial'][i],v['VelocityTangential'][i]))
        [C._initVars(LiftingLine, eq) for eq in ListOfEquations]


        Cy = Cl[i]*np.cos(phi)-Cd[i]*np.sin(phi)
        Cx = Cl[i]*np.sin(phi)+Cd[i]*np.cos(phi)

        # Non-linear functions f1, f2, to solve (Eqns 2.38-2.39)
        #
        f1 = 0.5*sigma[i]*W**2*Cy - 2*(VxP+x[0])*x[0]*F
        f2 = 0.5*sigma[i]*W**2*Cx - 2*(VxP+x[0])*x[1]*F

        Residual = [f1, f2]


        return Residual

    def solveDrela(x,SectionIndex):
        # Current section index is variable <i>
        i = SectionIndex
        # Drela's dummy variable (Fig.4) (radians)
        try:
            v['psi'][i]= psi = x
        except ValueError:
            print(x)
            v['psi'][i]= psi = x
            sys.exit()

        # Apply Drela's Eqns (17--27)
        ua = ut = 0 # TODO: Extend for multi-propellers
        Ua = Velocity[0] + ua
        Omega = RPM[0]*np.pi/30.
        Ut = Omega * r[i] - ut
        U = np.sqrt(Ua**2+Ut**2)
        v['VelocityAxial'][i] = Wa = 0.5 * (Ua + U * np.sin(psi))
        v['VelocityTangential'][i] = Wt = 0.5 * (Ut + U * np.cos(psi))
        # v['phiRad'][i] = phi = np.arctan(Wa/Wt) # (useful to store)
        v['phiRad'][i] = phi = np.arctan2(Wa,Wt) # (useful to store)
        v['VelocityInducedAxial'][i] = va = Wa - Ua
        v['VelocityInducedTangential'][i] = vt = Ut - Wt
        phiDeg = np.rad2deg(phi)
        AoADeg[i] = TwistDeg[i] - phiDeg

        v['VelocityMagnitudeLocal'][i] = W = np.sqrt(Wa**2 + Wt**2)
        Reynolds[i] = Density[0] * W * Chord[i] / Mu
        Mach[i] = W / SoundSpeed

        # Compute tip losses factor, F
        F = TipLossFactor(NBlades[0],Velocity[0],Omega,phi,r[i],Rmax, kind=TipLosses)
        v['F'][i] = F


        LL._applyPolarOnLiftingLine(LiftingLine,PolarsInterpolatorDict)
        [C._initVars(LiftingLine, eq) for eq in ListOfEquations]

        # Local wake advance ratio (Eqn.9)
        lambda_w = (r[i]/Rmax)*Wa/Wt

        # Total circulation from Momentum theory
        # (Eqn.31)
        GammaMom = vt * (4*np.pi*r[i]/NBlades[0])*F*np.sqrt(1+ (4*lambda_w*Rmax/(np.pi*NBlades[0]*r[i]))**2)

        # Total circulation from Blade Element theory
        # (Eqn.16)
        GammaBE  = 0.5 * W * Chord[i] * Cl[i]

        # Both circulations shall be equal (Eqn.32)
        Residual = GammaMom - GammaBE

        if i==0:
            print('Cl=%g , AoA=%g, Mach=%g | %sphi=%g%s | Va=%g , Vt=%g'%(Cl[i],AoADeg[i],Mach[i],J.GREEN,np.rad2deg(phi),J.ENDC,v['VelocityAxial'][i],v['VelocityTangential'][i]))

        return Residual

    def solveAdkins(x,SectionIndex):
        i = SectionIndex
        v['a'][i]  = x[0]#np.maximum(x[0],0)
        v['aP'][i] = x[1]#np.maximum(x[1],0)

        Omega = RPM[0]*np.pi/30.

        # W is airfoil's local velocity magnitude, and
        # Wa and Wt its components
        v['VelocityAxial'][i] = Wa = Velocity[0]*(1+v['a'][i])
        v['VelocityTangential'][i]= Wt =Omega*r[i]*(1-v['aP'][i])

        v['VelocityInducedAxial'][i]  = Velocity[0]*(v['a'][i])
        v['VelocityInducedTangential'][i]  = Omega*r[i]*(- v['aP'][i])

        v['VelocityMagnitudeLocal'][i] = W = (Wa**2 + Wt**2) ** 0.5
        v['phiRad'][i] =phi= np.arctan(Wa/Wt)  # Adkins' Eqn.(25)

        AoADeg[i] = TwistDeg[i] - np.rad2deg(phi)
        Mach[i] = W / SoundSpeed
        Reynolds[i] = Density[0] * W * Chord[i] / Mu

        # Compute tip losses factor, F
        F = TipLossFactor(NBlades[0],Velocity[0],Omega,phi,r[i],Rmax, kind=TipLosses)
        v['F'][i] = F

        LL._applyPolarOnLiftingLine(LiftingLine,PolarsInterpolatorDict)
        [C._initVars(LiftingLine, eq) for eq in ListOfEquations]


        Cy = Cl[i]*np.cos(phi)-Cd[i]*np.sin(phi)
        Cx = Cl[i]*np.sin(phi)+Cd[i]*np.cos(phi)

        # Non-linear functions f1, f2, to solve:
        #
        f1 = 0.5*W**2*NBlades[0]*Chord[i]*Cy - 2*np.pi*r[i]*Velocity[0]*(1+v['a'][i])*(2*Velocity[0]*v['a'][i]*F)

        f2 = 0.5*W**2*NBlades[0]*Chord[i]*Cx - 2*np.pi*r[i]*Velocity[0]*(1+v['a'][i])*(2*Omega*r[i]*v['aP'][i]*F)

        Residual = [f1, f2]

        return Residual


    # These flags will be further used for fast conditioning
    ModelIsDrela = ModelIsHeene = ModelIsAdkins = False
    if model == 'Drela':
        IterationVariables = ['psi']
        ModelIsDrela = True
        BEMTsolver = solveDrela
    elif model == 'Heene':
        IterationVariables = ['VelocityInducedAxial','VelocityInducedTangential']
        ModelIsHeene = True
        BEMTsolver = solveHeene
    elif model == 'Adkins':
        IterationVariables = ['a','aP']
        ModelIsAdkins = True
        BEMTsolver = solveAdkins
    else:
        raise AttributeError(FAIL+'Attribute model="%s" not recognized, please use one of these: "Drela", "Heene" or "Adkins".'%model+ENDC)
    Nunk = len(IterationVariables)



    def singleShot__(cmd):

        if CommandType == 'Pitch':
            Pitch[0] = np.clip(cmd,-90,+90)
        elif CommandType == 'RPM':
            RPM[0] = cmd
        else:
            raise AttributeError("CommandType '%s' not recognized. Please use 'Pitch' or 'RPM'."%CommandType)

        TwistDeg[:] = TwistDeg+Pitch # Add Pitch


        # Initial guess calculation
        v_Pred = np.zeros(Nunk,dtype=np.float64)
        v_Corr = np.zeros(Nunk,dtype=np.float64)
        for vn in IterationVariables: v[vn][-1] = 0.0

        FirstSection = True
        for i in range(NPts-1):
            # predict Guess
            for iu in range(Nunk): v_Pred[iu] = v[IterationVariables[iu]][i-1]

            # correct Guess
            Guess = v_Pred #+ v_Corr

            if ModelIsHeene:
                # Solve the 2-eqn non-linear system
                sol=so.root(BEMTsolver,Guess, args=(i), method='hybr')
                success = sol.success
                root = sol.x

            elif ModelIsDrela:
                if FirstSection:
                    # Override guess based on psi(AoA=0)
                    sol = J.secant(psi2AoA,x0=0.5,x1=1.0, ftol=1.e-07, bounds=(-np.pi/2.,+np.pi/2.), maxiter=50,args=(i,))
                    Guess=sol['root'] if sol['converged'] else 1.

                # Solve the non-linear 1-eqn
                EpsStep = 2.*np.pi/180.
                sol = J.secant(BEMTsolver,x0=Guess-EpsStep,x1=Guess+EpsStep, ftol=1.e-07, bounds=(-np.pi/2.,+np.pi/2.), maxiter=50,args=(i,))
                success = sol['converged']
                root = sol['root']

            elif ModelIsAdkins:
                # Solve the 2-eqn non-linear system
                sol=so.root(BEMTsolver,Guess, args=(i), method='hybr')
                success = sol.success
                root = sol.x

            # Compute correctors
            for iu in range(Nunk): v_Corr[iu] = v[IterationVariables[iu]][i] - v_Pred[iu]

            if DEBUG and not success:
                MSG = FAIL+'Section %d/%d failed (Re=%g, M=%g)'%(i+1,NPts,Reynolds[i],Mach[i])+ENDC
                print(sol)
                C.convertPyTree2File(LiftingLine,'debug.cgns')
                print(MSG)
                LL._plotAoAAndCl([LiftingLine])
                print(WARN+'phi angle (degree)'+ENDC)
                print(WARN+str(np.rad2deg(v['phiRad']))+ENDC)
                print(WARN+'AoA (degree)'+ENDC)
                print(WARN+str(np.v['AoA'])+ENDC)
                raise ValueError(MSG)
            else:
                BEMTsolver(root,i)

            FirstSection = False

        # Extrapole last section's unknowns (singularity)
        for vn in IterationVariables:
            Slope = (v[vn][-2]-v[vn][-3])/(r[-2]-r[-3])
            Shift = v[vn][-2] - Slope*r[-2]
            v[vn][-1] = Slope*r[-1] + Shift

        tipsol = [v[vn][-1] for vn in IterationVariables]
        if len(tipsol)==1: tipsol = tipsol[0]

        # Solve last section
        BEMTsolver(tipsol,NPts-1)

        TwistDeg[:] = TwistDeg-Pitch # Reset Pitch

        # Compute the arrays
        DictOfIntegralData = computeAxialLoads(VariablesDict=v, ConditionsDict=FlightVars)

        if Constraint != 'Pitch':
            return DictOfIntegralData[Constraint]-ConstraintValue
        else:
            return 0.

    if Trim:
        # Trim attempts
        success = False
        BestCmd = np.array([0.])
        CurrentAccuracy = ValueTol*100.
        for attempt in range(len(AttemptCommandGuess)):
            # print('Attempt: %d'%attempt)
            bounds = AttemptCommandGuess[attempt]
            wg0 = 1./3.
            wg1 = 2./3.
            x0  = (1-wg0)*bounds[0] + wg0*bounds[1]
            x1  = (1-wg1)*bounds[0] + wg1*bounds[1]
            TrimSol = J.secant(singleShot__, ftol=ValueTol, x0=x0, x1=x1, bounds=bounds, maxiter=20)

            if TrimSol['converged']:
                success = True
                BestCmd[0] = TrimSol['root']
                break



            if TrimSol['froot']<CurrentAccuracy:
                BestCmd[0]      = TrimSol['root']
                CurrentAccuracy = TrimSol['froot']

        singleShot__(BestCmd[0])

        FlightVars[CommandType][0] = BestCmd[0]

    else:
        attempt = 0
        singleShot__(Pitch)

        success = True

    if not success and FailedAsNaN:
        print(TrimSol)
        DictOfIntegralData = dict(Thrust=np.nan,Power=np.nan,PropulsiveEfficiency=np.nan,J=np.nan,FigureOfMeritPropeller=np.nan,Converged=success, Attempts=attempt,Pitch=np.nan,RPM=np.nan,)
        J.set(LiftingLine,'.Loads',**DictOfIntegralData)
        return DictOfIntegralData

    # Compute arrays
    DictOfIntegralData = computeAxialLoads(VariablesDict=v, ConditionsDict=FlightVars)
    DictOfIntegralData['Converged'] = success
    DictOfIntegralData['Attempts']  = attempt
    DictOfIntegralData['Pitch']     = Pitch
    DictOfIntegralData['RPM']       = RPM[0]
    J.set(LiftingLine,'.Loads',**DictOfIntegralData)

    return DictOfIntegralData


def computeBEMTaxial3D(LiftingLine, PolarsInterpolatorDict,
                NBlades=None,
                Velocity=[0.,0.,0.], RPM=None, Density=None,Temperature=None,
                RotationCenter=[0.,0.,0.],RotationAxis=[0.,0.,1.],
                RightHandRuleRotation=True, model='Heene',
                ListOfEquations=[], TipLosses='Adkins', Constraint='Pitch',
                ConstraintValue=0., ValueTol=1.0,
                AttemptCommandGuess=[[0.,25.],[5.,40.]], CommandType='Pitch',
                PitchIfTrimCommandIsRPM=0., FailedAsNaN=False):
    """
    Theory reference document : `Mario Heene's Master Thesis <https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjb9pWhquDyAhWfQkEAHXJkDKYQFnoECAUQAQ&url=http%3A%2F%2Fkth.diva-portal.org%2Fsmash%2Fget%2Fdiva2%3A559083%2FFULLTEXT01.pdf&usg=AOvVaw3HEbEZVJfHL1tCtLg2AYim>`_.
    Governing equations are (2.38) and (2.39).

    Perform a blade-element momentum-theory (BEMT) computation of a propeller
    flying in axial flight.

    Returns a dictionary of integral arrays.

    Updates relevant spanwise quantities on LiftingLine zone.

    Parameters
    ----------

        LiftingLine : zone
            Lifting-line modeling the blades of the propeller.
            The LiftingLine is modified.

        PolarsInterpolatorDict : dict
            dictionary of interpolator functions of 2D polars, as obtained from
            :py:func:`MOLA.LiftingLine.buildPolarsInterpolatorDict` function.

        NBlades : int
            number of blades of the propeller

        Velocity : :py:class:`list` of 3 :py:class:`float`
            Velocity components.

            .. warning:: in current implementation, **Velocity** *must be*
                aligned with **RotationAxis**.

            .. hint:: for canonical placement of LiftingLine, use:

                >>> Velocity = [0,0,-Value]

        RPM :  float
            rotational speed of the propeller [rev / minute]

        Density :  float
            air density [kg/m3]

        Temperature :  float
            air temperature [K]

        RotationCenter : :py:class:`list` of 3 :py:class:`float`
            coordinates of the rotation point [m] in math:`(x,y,z)`

        RotationAxis : :py:class:`list` of 3 :py:class:`float`
            unitary vector pointing towards the rotation axis (which is the
            desired direction of thrust).

        RightHandRuleRotation : bool
            if :py:obj:`True`, propeller rotates around **RotationAxis**
            following the right-hand-rule rotation. :py:obj:`False` otherwise.

        model : str
            .. important:: currently, only ``'Heene'`` model is available.

        ListOfEquations : str
            converter-compatible equations to be applied
            after each lifting-line's polar interpolation operation.

        TipLosses : str
            the model of tip-losses. Can be one of:
            ``'Adkins'``, ``'Glauert'`` or ``'Prandtl'``.

            .. hint:: we recommend ``'Adkins'``.

        Constraint : str
            can be ``'Thrust'``, ``'Power'`` or ``'Pitch'``.

        ConstraintValue : float
            value of the constraint to be verified *(dimensions are contextual)*

        ValueTol : float
            threshold tolerance for determining if Trim condition
            is satisfied with respect to **ConstraintValue**.

        AttemptCommandGuess : :py:class:`list` of 2-:py:class:`float` lists
            each item of the list establishes the pairs of minimum and maximum
            boundaries of trim command.
            Each item is a command guess. In case of failure, the next candidate is
            tested and so on.

            .. hint:: use as many different lists of ``[min, max]`` as the number
                of trials for reaching the requested trimmed condition

        CommandType : str
            can be ``'Pitch'`` or ``'RPM'``

        PitchIfTrimCommandIsRPM : float
            if **CommandType** is ``'RPM'`` and **Constraint** is not ``'Pitch'``,
            then this argument sets the employed pitch.

        FailedAsNaN : bool
            if :py:obj:`True`, if trim fails, then returns ``NaN`` values as
            integral arrays

    Returns
    -------

        DictOfIntegralData : dict
            contains the integral arrays of the propeller
    """
    import scipy.optimize as so

    LiftingLine, = I.getZones(LiftingLine)

    NPts = C.getNPts(LiftingLine)

    if Constraint == 'Pitch':
        Pitch = ConstraintValue
        CommandType = 'Pitch'
        Trim = False
    elif CommandType == 'RPM':
        Pitch = PitchIfTrimCommandIsRPM
        Trim = True
    else:
        Pitch = 0.
        Trim = True

    # Read existing flight conditions.
    # For each user-provided not None condition,
    # replace existing one in LiftingLine object
    Velocity = np.array(Velocity,dtype=np.float, order='F')
    RotationAxis = np.array(RotationAxis,dtype=np.float, order='F')
    RotationCenter = np.array(RotationCenter,dtype=np.float, order='F')
    ProvidedFlightVars = dict(NBlades=NBlades, VelocityFreestream=Velocity, RPM=RPM,
                        Density=Density, Temperature=Temperature, Pitch=Pitch,
                        RotationAxis=RotationAxis, RotationCenter=RotationCenter,
                        RightHandRuleRotation=True)
    FlightVars = J.get(LiftingLine,'.Conditions')
    for fvname in ProvidedFlightVars:
        if ProvidedFlightVars[fvname] is not None:
            FlightVars[fvname] = ProvidedFlightVars[fvname]
    Conditions = dict(VelocityFreestream=FlightVars['VelocityFreestream'],
                      Density=FlightVars['Density'],
                      Temperature=FlightVars['Temperature'])
    J.set(LiftingLine, '.Conditions', **Conditions)
    Kinematics = dict(RPM=FlightVars['RPM'],
                      NBlades=FlightVars['NBlades'],
                      Pitch=FlightVars['Pitch'],
                      RotationAxis=FlightVars['RotationAxis'],
                      RotationCenter=FlightVars['RotationCenter'],
                      RightHandRuleRotation=FlightVars['RightHandRuleRotation'])
    J.set(LiftingLine, '.Kinematics', **Kinematics)
    FlightVars = J.get(LiftingLine,'.Conditions')
    FlightVarsKin = J.get(LiftingLine,'.Kinematics')
    FlightVars.update(FlightVarsKin)
    # Use local variable names based on
    # updated FlightVars

    NBlades     = FlightVars['NBlades']
    Velocity    = FlightVars['VelocityFreestream']
    RPM         = FlightVars['RPM']
    Pitch       = FlightVars['Pitch']
    Density     = FlightVars['Density']
    Temperature = FlightVars['Temperature']

    # invoke variables in Lifting Line
    ListOfVariables = [
    'a',   # axial interference factor
    'aP',  # tangential interference factor
    'Cm',
    # -------------- see references:  HEENE
    'phiRad',                    # phi (Eqn. 2.1)
    'VelocityMagnitudeLocal',    # Vp (Fig. 2.2)
    'VelocityAxial',             # Vx (Fig. 2.2)
    'VelocityTangential',        # Vt (Fig. 2.2)
    'VelocityInducedAxial',      # va (Eqn. 2.7)
    'VelocityInducedTangential', # vr (Eqn. 2.8)
    'VelocityKinematicX',
    'VelocityKinematicY',
    'VelocityKinematicZ',
    'VelocityNormal2D',
    'VelocityTangential2D',
    'F',   # Tip loss factor ( see function TipLossFactor() )
    'dFx', # section's axial force  - for Thrust computation
    'dMx', # section's axial moment - for Power computation

    'tx','ty','tz', # Local frame
    'nx','ny','nz',
    'bx','by','bz',

    'tanx','tany','tanz', # tangential vector
    ]
    v = J.invokeFieldsDict(LiftingLine,ListOfVariables)
    W.addDistanceRespectToLine(LiftingLine, RotationCenter, RotationAxis,
                                FieldNameToAdd='Span')

    # Get some existing variables:
    r, Chord, TwistDeg, AoADeg, Mach, Reynolds, \
    Cl, Cd = J.getVars(LiftingLine,['Span','Chord','Twist','AoA', 'Mach',
                                                        'Reynolds', 'Cl', 'Cd'])
    x,y,z = J.getxyz(LiftingLine)
    LL.updateLocalFrame(LiftingLine)

    # Complementary variables
    v2 = J.getVars2Dict(LiftingLine,['Span', 'Chord', 'Twist', 'Cl', 'Cd'])
    v.update(v2)
    v.update(FlightVars)

    Dir = +1 if RightHandRuleRotation else -1

    # Declare additional variables (not stored in LiftLine)
    sigma = NBlades * Chord / (2*np.pi*r) # Blade solidity
    Omega = RPM[0] * np.pi / 30.

    # Compute constant fields
    for i in range(NPts):
        rvec = np.array([x[i] - RotationCenter[0],
                         y[i] - RotationCenter[1],
                         z[i] - RotationCenter[2]],dtype=np.float)

        VelocityKinematic = -Velocity + np.cross(Dir*Omega*RotationAxis,rvec)
        v['VelocityKinematicX'][i] = VelocityKinematic[0]
        v['VelocityKinematicY'][i] = VelocityKinematic[1]
        v['VelocityKinematicZ'][i] = VelocityKinematic[2]



    # Compute some constants:
    # Sutherland's law
    Mu=Mus*((Temperature/Ts)**0.5)*((1.+Cs/Ts)/(1.+Cs/Temperature))
    Pressure = Density * Rgp * Temperature # State
    Rmax = r.max()          # Tip radius
    SoundSpeed = np.sqrt(Gamma * Rgp * Temperature)


    def solveHeene(vi,SectionIndex):
        i = SectionIndex

        Vk = np.array([v['VelocityKinematicX'][i],
                       v['VelocityKinematicY'][i],
                       v['VelocityKinematicZ'][i]],
                      dtype=np.float, order='F')

        TangentialDirection = np.array([v['tanx'][i],
                                        v['tany'][i],
                                        v['tanz'][i]],
                                        dtype=np.float, order='F')

        VkAxial = Vk.dot(RotationAxis)
        VkTan = Vk.dot(TangentialDirection)

        # via = v['VelocityInducedAxial'][i]      = np.maximum(np.minimum(vi[0],abs(VkAxial)),0)
        # vit = v['VelocityInducedTangential'][i] = np.maximum(np.minimum(vi[1],abs(VkTan)),0)

        via = v['VelocityInducedAxial'][i]      = vi[0]
        vit = v['VelocityInducedTangential'][i] = vi[1]


        nxyz = np.array([v['nx'][i],v['ny'][i],v['nz'][i]],
                        dtype=np.float, order='F')
        bxyz = np.array([v['bx'][i],v['by'][i],v['bz'][i]],
                        dtype=np.float, order='F')


        v['VelocityAxial'][i]      = Vax  =  VkAxial + via
        v['VelocityTangential'][i] = Vtan = -VkTan   - vit

        V2D = Vax * RotationAxis  +  Vtan * TangentialDirection

        v['VelocityNormal2D'][i]     = V2Dn = V2D.dot( nxyz )
        v['VelocityTangential2D'][i] = V2Dt = V2D.dot( bxyz )
        v['phiRad'][i] = phi = np.arctan2( V2Dn, V2Dt )
        AoADeg[i] = TwistDeg[i] - np.rad2deg(phi)


        # NB: W == Vp (Fig 2.2)
        v['VelocityMagnitudeLocal'][i] = W = np.sqrt( V2Dn**2 + V2Dt**2 )

        Mach[i] = W / SoundSpeed
        Reynolds[i] = Density[0] * W * Chord[i] / Mu

        # Compute tip losses factor, F
        F = TipLossFactor(NBlades[0],VkAxial,Omega,phi,r[i],Rmax, kind=TipLosses)
        v['F'][i] = F

        LL._applyPolarOnLiftingLine(LiftingLine,PolarsInterpolatorDict, InterpFields=['Cl', 'Cd'])
        [C._initVars(LiftingLine, eq) for eq in ListOfEquations]


        Cy = Cl[i]*np.cos(phi)-Cd[i]*np.sin(phi)
        Cx = Cl[i]*np.sin(phi)+Cd[i]*np.cos(phi)

        # Non-linear functions f1, f2, to solve (Eqns 2.38-2.39)
        #
        f1 = 0.5*sigma[i]*W**2*Cy - 2*Vax*via*F
        f2 = 0.5*sigma[i]*W**2*Cx - 2*Vax*vit*F

        Residual = [f1, f2]

        return Residual



    # These flags will be further used for fast conditioning
    ModelIsDrela = ModelIsHeene = ModelIsAdkins = False
    if model == 'Heene':
        IterationVariables = ['VelocityInducedAxial','VelocityInducedTangential']
        ModelIsHeene = True
        BEMTsolver = solveHeene
        # Set initial guess:
        v['VelocityInducedAxial'][:] = 0.0
        v['VelocityInducedTangential'][:] = 0.5 * v['VelocityInducedAxial'][:]
    else:
        raise AttributeError(FAIL+'Attribute model="%s" not recognized.'%model+ENDC)
    Nunk = len(IterationVariables)



    def singleShot__(cmd):

        if CommandType == 'Pitch':
            Pitch[0] = np.clip(cmd,-90,+90)
        elif CommandType == 'RPM':
            RPM[0] = cmd
        else:
            raise AttributeError("CommandType '%s' not recognized. Please use 'Pitch' or 'RPM'."%CommandType)

        TwistDeg[:] = TwistDeg+Pitch # Add Pitch


        # Initial guess calculation
        v_Pred = np.zeros(Nunk,dtype=np.float64)
        v_Corr = np.zeros(Nunk,dtype=np.float64)
        for i,vn in enumerate(IterationVariables):
            v_Pred[i]=v[vn][0]
        # print(v_Pred) # Show initial guess

        FirstSection = True
        for i in range(NPts-1):
            # predict Guess
            for iu in range(1,Nunk): v_Pred[iu] = v[IterationVariables[iu]][i-1]

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
                    sol = J.secant(psi2AoA,x0=0.5,x1=1.0, ftol=1.e-07, bounds=(-np.pi/2.,+np.pi/2.), maxiter=50,args=(i,))
                    Guess=sol['root'] if sol['converged'] else 1.

                # Solve the non-linear 1-eqn
                EpsStep = 2.*np.pi/180.
                sol = J.secant(BEMTsolver,x0=Guess-EpsStep,x1=Guess+EpsStep, ftol=1.e-07, bounds=(-np.pi/2.,+np.pi/2.), maxiter=50,args=(i,))
                success = sol['converged']
                root = sol['root']

            elif ModelIsAdkins:
                # Solve the 2-eqn non-linear system
                sol=so.root(BEMTsolver,Guess, args=(i), method='hybr')
                success = sol.success
                root = sol.x

            # Compute correctors
            for iu in range(Nunk): v_Corr[iu] = v[IterationVariables[iu]][i] - v_Pred[iu]

            MatrixComputationRequired = any([not success,
                                             AoADeg[i] > 15.,
                                             AoADeg[i] < -15.])

            if DEBUG and MatrixComputationRequired:
                print('\ndid not succeed at index %d. Using response matrix...'%i)
                V = np.maximum(5,np.sqrt(Velocity.dot(Velocity)))
                Nmsh = 21
                via = np.linspace(0,5*V,Nmsh)
                vit = np.linspace(0, V,Nmsh)
                RespMatrixf1 = np.zeros((Nmsh, Nmsh))
                RespMatrixf2 = np.zeros((Nmsh, Nmsh))
                AoAMatrix = np.zeros((Nmsh, Nmsh))
                for ir, jc in itertools.product(range(Nmsh), range(Nmsh)):
                    residual = BEMTsolver((via[ir], vit[jc]), i)
                    RespMatrixf1[ir,jc] = residual[0]
                    RespMatrixf2[ir,jc] = residual[1]
                    AoAMatrix[ir,jc] = AoADeg[i]

                RespMatrix = RespMatrixf1 + RespMatrixf2

                iMinRes = np.argmin(np.abs(RespMatrix).ravel(order='K'))
                iOpt, jOpt = np.unravel_index(iMinRes, RespMatrix.shape, order='C')
                MinRes = RespMatrix[iOpt, jOpt]
                OptAoA = AoAMatrix[iOpt, jOpt]

                root = via[iOpt], vit[jOpt]
                print('Minimum Resiudal = %g at AoA= %g deg'%(MinRes, OptAoA))
                print('With induced velocities axial %g and tangential %g'%root)
                #
                # MSG = FAIL+'Section %d/%d failed (Re=%g, M=%g)'%(i+1,NPts,Reynolds[i],Mach[i])+ENDC
                # print(sol)
                # C.convertPyTree2File(LiftingLine,'debug.cgns')
                # print(MSG)
                # LL._plotAoAAndCl([LiftingLine])
                # print(WARN+'phi angle (degree)'+ENDC)
                # print(WARN+str(np.rad2deg(v['phiRad']))+ENDC)
                # print(WARN+'AoA (degree)'+ENDC)
                # print(WARN+str(AoADeg)+ENDC)
                # raise ValueError(MSG)

            BEMTsolver(root,i)

            FirstSection = False

        # Extrapole last section's unknowns (singularity)
        for vn in IterationVariables:
            Slope = (v[vn][-2]-v[vn][-3])/(r[-2]-r[-3])
            Shift = v[vn][-2] - Slope*r[-2]
            v[vn][-1] = Slope*r[-1] + Shift

        tipsol = [v[vn][-1] for vn in IterationVariables]
        if len(tipsol)==1: tipsol = tipsol[0]

        # Solve last section
        BEMTsolver(tipsol,NPts-1)

        TwistDeg[:] = TwistDeg-Pitch # Reset Pitch

        # Compute the arrays
        DictOfIntegralData = LL.computeGeneralLoadsOfLiftingLine(LiftingLine,
                                                                NBlades=NBlades)

        if Constraint != 'Pitch':
            return DictOfIntegralData[Constraint]-ConstraintValue
        else:
            return 0.

    if Trim:
        # Trim attempts
        success = False
        BestCmd = np.array([0.])
        CurrentAccuracy = ValueTol*100.
        for attempt in range(len(AttemptCommandGuess)):
            # print('Attempt: %d'%attempt)
            bounds = AttemptCommandGuess[attempt]
            wg0 = 1./3.
            wg1 = 2./3.
            x0  = (1-wg0)*bounds[0] + wg0*bounds[1]
            x1  = (1-wg1)*bounds[0] + wg1*bounds[1]
            TrimSol = J.secant(singleShot__, ftol=ValueTol, x0=x0, x1=x1, bounds=bounds, maxiter=20)

            if TrimSol['converged']:
                success = True
                BestCmd[0] = TrimSol['root']
                break



            if TrimSol['froot']<CurrentAccuracy:
                BestCmd[0]      = TrimSol['root']
                CurrentAccuracy = TrimSol['froot']

        singleShot__(BestCmd[0])

        v[CommandType][0] = BestCmd[0]

    else:
        attempt = 0
        singleShot__(Pitch)

        success = True

    if not success and FailedAsNaN:
        print(TrimSol)
        DictOfIntegralData = dict(Thrust=np.nan,Power=np.nan,PropulsiveEfficiency=np.nan,J=np.nan,FigureOfMeritPropeller=np.nan,Converged=success, Attempts=attempt,Pitch=np.nan,RPM=np.nan,)
        J.set(LiftingLine,'.Loads',**DictOfIntegralData)
        return DictOfIntegralData

    # Compute arrays
    LL._applyPolarOnLiftingLine(LiftingLine, PolarsInterpolatorDict,
                                InterpFields=['Cl', 'Cd','Cm'])
    [C._initVars(LiftingLine, eq) for eq in ListOfEquations]

    DictOfIntegralData = LL.computeGeneralLoadsOfLiftingLine(LiftingLine,
                                                            NBlades=NBlades)
    DictOfIntegralData['Converged'] = success
    DictOfIntegralData['Attempts']  = attempt
    DictOfIntegralData['Pitch']     = Pitch
    DictOfIntegralData['RPM']       = RPM[0]
    AdvanceVelocityModule = np.sqrt(Velocity.dot(Velocity))
    n = RPM/60.       # rev / second
    d = v['Span'].max()*2  # diameter
    CTpropeller = DictOfIntegralData['Thrust'] / (Density * n**2 * d**4)
    CPpropeller = DictOfIntegralData['Power']  / (Density * n**3 * d**5)
    Jparam = AdvanceVelocityModule / (n*d)
    FigureOfMeritPropeller = np.sqrt(2./np.pi)* np.maximum(CTpropeller,0)**1.5 / np.maximum(CPpropeller,1e-12)
    PropEff = AdvanceVelocityModule*DictOfIntegralData['Thrust']/np.abs(DictOfIntegralData['Power'])
    DictOfIntegralData['CTpropeller']=CTpropeller
    DictOfIntegralData['CPpropeller']=CPpropeller
    DictOfIntegralData['Jparam']=Jparam
    DictOfIntegralData['FigureOfMeritPropeller']=FigureOfMeritPropeller
    DictOfIntegralData['PropulsiveEfficiency']=PropEff


    J.set(LiftingLine,'.Loads',**DictOfIntegralData)

    return DictOfIntegralData



def TipLossFactor(NBlades,Velocity,Omega,phi,r,Rmax, kind='Adkins'):
    '''
    Compute the tip loss factor, F, for use with Blade Element Theory code.

    .. hint:: all input parameters may be arrays of the same length, in this
        case the returned value is also an array

    Parameters
    ----------

        NBlades : int
            Number of Blades

        Velocity : float
            Advance velocity [m/s]

        Omega : float
            Rotation velocity [rad / s]

        phi : float
            Geometric flow angle [rad]

        r : float
            section Radius [m]

        Rmax : float
            Blade tip radius [m]

        kind : str
            May be one of:

            * ``"Adkins"``
                :math:`\\frac{2}{\pi} \\arccos{\left(\exp{-\\frac{(1-\\frac{r}{R}) N_b/2}{\\frac{r}{R}\\tan{\phi}}}\\right)}`

            * ``"Glauert"``
                :math:`\\frac{2}{\pi} \\arccos{\left(\exp{-\\frac{(1-\\frac{r}{R}) N_b/2}{\\frac{r}{R}\\sin{\phi}}}\\right)}`

            * ``"Prandtl"``
                :math:`\\frac{2}{\pi} \\arccos{\left(\exp{-(1-\\frac{r}{R}) N_b/2} \sqrt{1 - (\\frac{\Omega r}{V})^2} \\right)}`

            .. note:: we recommend ``"Adkins"`` formulation, which is more
                appropriate for highly-loaded propellers, as discussed by
                `Mark Drela <https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiplIKBsODyAhXViVwKHaeAAcMQFnoECAUQAQ&url=http%3A%2F%2Fweb.mit.edu%2Fdrela%2FPublic%2Fweb%2Fqprop%2Fqprop_theory.pdf&usg=AOvVaw2pOfpH6zAeAbDA94_XElRg>`_.

    Returns
    -------

        F : float
            Tip loss factor

    '''

    xi = r/Rmax
    # phiEff avoids possible overflow (division by zero)
    f_max = 100.0
    a = (NBlades/2.)*(1-xi)
    phi_min = np.arctan(a/(xi*f_max))

    phiEff = np.maximum(np.abs(phi),phi_min)
    if kind == 'Adkins':
        # This version is also used by M. Drela
        f = a/(xi*np.tan(phiEff))
        f = np.maximum(f,0)
        F = (2./np.pi)*np.arccos(np.minimum(np.exp(-f),1))

    elif kind == 'Glauert':
        f = a/(xi * np.sin(phiEff))
        F = (2./np.pi)*np.arccos(np.minimum(np.exp(-f),1))

    elif kind == 'Prandtl':
        f = a*np.sqrt(1-(Omega*r/Velocity)**2)
        F = (2./np.pi)*np.arccos(np.minimum(np.exp(-f),1))
    else:
        raise ValueError('TipLosses=%s not recognized.'%kind)

    return F

def computeAxialLoads(LiftingLine=None, VariablesDict=None, ConditionsDict=None):
    '''

    .. danger:: THIS FUNCTION IS BEING DEPRECATED, PLEASE USE:
        :py:func:`MOLA.LiftingLine.computeGeneralLoadsOfLiftingLine` instead

    If **LiftingLine** object is provided, then the arguments
    **VariablesDict** and **ConditionsDict** are IGNORED.
    In this case, Integral data is also stored in the LiftingLine
    object in ``.Loads`` node.

    **LiftingLine** object may be omitted if BOTH **VariablesDict**
    and **ConditionsDict** are provided (useful for numerical
    performance)

    If **VariablesDict** is provided, it must contain:
    ``['VelocityMagnitudeLocal','phiRad', 'Span', 'Chord', 'Cl', 'Cd', 'dFx', 'dMx']``

    If **ConditionsDict** is provided, it must contain:
    ``['Density', 'NBlades', 'RPM']``
    '''

    import scipy.integrate as sint

    if VariablesDict is None or ConditionsDict is None:
        J._invokeFields(LiftingLine,['dFx', 'dMx'])
        v = J.getVars2Dict(LiftingLine,['VelocityMagnitudeLocal','phiRad', 'Span', 'Chord', 'Cl', 'Cd', 'dFx', 'dMx'])
        c = J.get(LiftingLine,'.Conditions')
    else:
        v = VariablesDict
        c = ConditionsDict

    v['dFx'] = 0.5*c['Density']*v['VelocityMagnitudeLocal']**2*c['NBlades']*v['Chord']*(v['Cl']*np.cos(v['phiRad'])-v['Cd']*np.sin(v['phiRad']))
    # Thrust = np.trapz(v['dFx'],v['Span'])
    Thrust = sint.simps(v['dFx'],v['Span'])

    v['dMx'] = 0.5*c['Density']*v['VelocityMagnitudeLocal']**2*c['NBlades']*v['Chord']*(v['Cl']*np.sin(v['phiRad'])+v['Cd']*np.cos(v['phiRad']))*v['Span']
    # Torque = np.trapz(v['dMx'],v['Span'])
    Torque = sint.simps(v['dMx'],v['Span'])
    Power  = (c['RPM']*np.pi/30.)*Torque

    try:
        AdvanceVelocityModule = c['Velocity']
    except KeyError:
        AdvanceVelocityModule = np.sqrt(c['VelocityFreestream'].dot(c['VelocityFreestream']))

    n = c['RPM']/60.       # rev / second
    d = v['Span'].max()*2  # diameter
    CTpropeller = Thrust / (c['Density'] * n**2 * d**4)
    CPpropeller = Power  / (c['Density'] * n**3 * d**5)
    Jparam = AdvanceVelocityModule / (n*d)
    FigureOfMeritPropeller = np.sqrt(2./np.pi)* np.maximum(CTpropeller,0)**1.5 / np.maximum(CPpropeller,1e-12)

    PropEff = AdvanceVelocityModule*Thrust/np.abs(Power)

    DictOfIntegralData = dict(Thrust=Thrust,Power=Power,PropulsiveEfficiency=PropEff,J=Jparam,FigureOfMeritPropeller=FigureOfMeritPropeller)

    if LiftingLine is not None: J.set(LiftingLine,'.Loads',**DictOfIntegralData)

    return DictOfIntegralData


def kinematicReynoldsAndMach(LiftingLine,Velocity,RPM,Density=1.225,Temperature=288.):
    '''
    Given a **LiftingLine** object and operating conditions,
    compute the kinematic Reynolds and Mach.
    This function updates LiftingLine's ``Reynolds`` and ``Mach`` fields located
    at container ``FlowSolution`` and returns its corresponding 1D numpy arrays.

    Parameters
    ----------

        LiftingLine : PyTree
            a lifting-line as generated by :py:func:`buildLiftingLine`.
            Does not need ``Airfoils`` node.

            .. note:: zone **LiftingLine** is modified (fields are updated)

        Velocity : float
            Advance velocity in [m/s]

        RPM : float
            Rotational speed in [rev/minute]

        Density : float
            Air density in [kg/m3]

        Temperature : float
            Air temperature in [Kelvin]

    Returns
    -------

        Reynolds : numpy.ndarray
            1d numpy copy of Reynolds array

        Mach : numpy.ndarray
            1d numpy copy of Mach array
    '''


    # Get some existing variables:
    r, Chord, Mach, Reynolds = J.getVars(LiftingLine,['Span','Chord','Mach', 'Reynolds'])

    # Sutherland's law
    Mu=Mus*((Temperature/Ts)**0.5)*((1.+Cs/Ts)/(1.+Cs/Temperature))
    Omega = RPM*np.pi/30.   # rotational speed
    SoundSpeed = (Gamma * Rgp * Temperature)**0.5

    # Initialize Mach and Reynolds number with kinematics
    KinematicV = np.sqrt( Velocity**2 + (Omega*r)**2 )
    Mach[:] = KinematicV / SoundSpeed
    Reynolds[:] = Density * KinematicV * Chord / Mu

    return Reynolds*1.0, Mach*1.0 # *1.0 to easily make copies

def getPolarOperatingRanges(LiftingLine, Velocity, RPM, Density, Temperature,
        Labels=[], OverlapFactor=1., QhullNPts=20, QhullScale=1.2, grading=0.1,
        rescale=True, includeQhull=True, showplot=False, saveplot=False):
    '''
    Compute the Reynolds and Mach expected operating ranges
    of a **LiftingLine** with ``Airfoils`` node based on a set of
    operating conditions. This function is useful for determining
    the Reynolds and Mach that are to be passed to a Polar
    computation workflow *(CFD, XFoil, both...)*

    .. attention:: the number of elements contained in lists **Velocity**,
        **RPM**, **Density**, **Temperature** and **Labels** must be all the SAME

    Parameters
    ----------

        LiftingLine : zone
            LiftingLine object as built from :py:func:`buildLiftingLine` function.

        Velocity : :py:class:`list` of :py:class:`float`
            Set of velocities to consider as operating conditions (m/s)

        RPM : :py:class:`list` of :py:class:`float`
            Set of rotational speeds to consider as operating conditions (rev/minute)

        Density : :py:class:`list` of :py:class:`float`
            Set of air densities to consider as operating conditions (kg/m3)

        Temperature : :py:class:`list` of :py:class:`float`
            Set of Temperatures to consider as operating conditions (Kelvin)

        Labels : :py:class:`list` of :py:class:`str`
            List used to provide a user-defined label to each operating condition.
            This argument has several purposes:

            *   It is used to appropriately store each operating
                condition inside Results Python dictionary. The Labels
                argument are the keys of such Result dictionary.

            *   If argument showplot==True, then Labels are also used for
                providing a legend on the plot figure.

            .. note:: if **Labels** is not provided (an empty list), then it is
                automatically transformed into a integer range with as
                many elements as operating conditions.


        OverlapFactor : float
            Factor that controls the polar's
            overlapping in the spanwise direction. Increasing this
            value increases the required Reynolds and Mach bounds for
            each airfoil, but improves the interpolation trust region
            when using final polar interpolator objects.
            A value of 0 would yield no-overlap, minimizing the
            amount of Reynolds and Mach computing points, but
            may deteriorate the LiftingLine interpolation precision
            if few polar sections are used and high order
            interpolation objects are used. Maximum interpolation
            precision for an interpolator object of order Q is
            obtained if OverlapFactor = Q. A value of 1 would yield
            maximum interpolation precision for objects using linear
            interpolation (RECOMMENDED). A value of 2 would yield
            maximum interpolation precision for objects using
            quadratic interpolation, and so on.

        includeQhull : bool
            If :py:obj:`True`, include ``"qhull"`` key in
            Results dictionary, with the ``Reynolds(Mach)`` vector
            establishing the convex hull of the operating envelope
            of the propeller.

        showplot : bool
            if :py:obj:`True`, then produce a matplotlib
            figure for easy data assessment.

        saveplot : :py:class:`str` or :py:class:`bool` or :py:obj:`None`
            If a string is provided, then saves the figure using saveplot as
            output filename.

    Returns
    -------

        ResultsDict : dict
            Stores the results of the kinematic computation for each operating
            condition:

            >>> ResultsDict['<label>'][<'Reynolds' or 'Mach'>]

            * ``'<label>'`` : :py:class:`str`
                shall be repaced by one of the user-provided **Labels**
                corresponding to a given operating condition.

            * ``<'Reynolds' or 'Mach'>`` : :py:class:`str`
                shall be ``"Reynolds"`` or ``"Mach"``, and
                provides access to the resulting 1D numpy array along
                the **LiftingLine**.

        SamplesDict : dict
            Stores the interpolated space of requested samples, and may be used
            as input for each airfoil's set of Reynolds and Mach for
            computing polars. This dictionary is organized as:

            >>> SamplesDict[<AirfoilIdentifier>][<'Reynolds' or 'Mach'>]

            .. note:: Airfoil identifier is skipped if ``Airfoils`` node does
                not exist

            * ``<'Reynolds' or 'Mach'>`` : :py:class:`str`
                shall be ``"Reynolds"`` or ``"Mach"``, and
                provides access to the resulting 1D numpy array of elements on
                the flight envelope.

    '''

    # Perform some input verifications
    nVel = len(Velocity)
    nRPM = len(RPM)
    nRho = len(Density)
    nTem = len(Temperature)
    nLab = len(Labels)
    if nLab==0:
        Labels = np.arange(nVel)
        nLab = nVel
    if not (nVel == nRPM == nRho == nTem == nLab):
        raise AttributeError('Number of elements of each operating condition parameter and label must be equal.\nProvided: Velocity (%d), RPM (%d), Density (%d), Temperature (%d), Labels (%d)'%(nVel, nRPM, nRho, nTem, nLab))

    # Check if Airfoils node exist
    PolarInfoNode = I.getNodeFromName(LiftingLine,'Airfoils')
    if PolarInfoNode is not None:
        AirfoilIdentifiers = I.getValue(I.getNodeFromName1(PolarInfoNode,'PyZonePolarNames')).split(' ')
        AirfoilPositions = I.getNodeFromName1(PolarInfoNode,'Abscissa')[1]
    else:
        # Trick to have general procedure
        AirfoilIdentifiers = ['DummyFirst','DummySecond']
        AirfoilPositions   = np.array([0.0,1.0])

    RequestSpan=100
    RequestOperating=30

    Nfoils = len(AirfoilIdentifiers)
    # Get some variables of LiftingLine
    r, s = J.getVars(LiftingLine,['Span','s'])
    Rmax = r.max()

    # Construct ResultsDict
    ResultsDict = {}
    for i in range(nLab):
        ResultsDict[Labels[i]] = {}
        Reynolds, Mach = kinematicReynoldsAndMach(LiftingLine,Velocity[i],RPM[i],Density[i],Temperature[i])
        ResultsDict[Labels[i]]['Reynolds'] = Reynolds
        ResultsDict[Labels[i]]['Mach']     = Mach

    AllReynolds, AllMach = [],[]
    for lab in Labels:
        AllReynolds += [ResultsDict[lab]['Reynolds']]
        AllMach += [ResultsDict[lab]['Mach']]
    AllMach     = np.hstack(AllMach)
    AllReynolds = np.hstack(AllReynolds)
    ResultsDict['AllMach'] = AllMach
    ResultsDict['AllReynolds'] = AllReynolds

    # Construct convex hull
    if includeQhull:
        Qhull,_,_ = J.get2DQhullZone__(AllMach, AllReynolds)
        QhullMach, QhullRe = J.getxy(Qhull)
        ResultsDict['Qhull'] = dict(Mach=QhullMach, Reynolds=QhullRe)

    # Construct SamplesDict
    SamplesDict = {}
    for i in range(Nfoils):
        foilname = AirfoilIdentifiers[i]
        SamplesDict[foilname] = {}
        WeightFactor= 0.5*(OverlapFactor+1.)
        # Find minimum spanwise bound in curvilinear abscissa
        if i == 0:
            MinBound = s.min()
        else:
            MinBound =    WeightFactor*AirfoilPositions[i-1]+ \
                     (1.-WeightFactor)*AirfoilPositions[i]

        # Find maximum spanwise bound in curvilinear abscissa
        if i == Nfoils-1:
            MaxBound = s.max()
        else:
            MaxBound =    WeightFactor*AirfoilPositions[i+1]+ \
                     (1.-WeightFactor)*AirfoilPositions[i]

        # Construct new abscissa sampling vector
        SamplingAbs = np.linspace(MinBound,MaxBound,RequestSpan)

        # Allocate arrays where sampling will be stored
        SamplesDict[foilname]['Reynolds'] = np.zeros((RequestOperating,RequestSpan),dtype=np.float64,order='F')
        SamplesDict[foilname]['Mach'] = np.zeros((RequestOperating,RequestSpan),dtype=np.float64,order='F')
        SamplesDict[foilname]['Span'] = J.interpolate__(SamplingAbs, s, r)

        # For each new abscissa element, find the (min, max)
        # pair for Reynolds and Mach variations in all the
        # operating points envelope
        for jSample in range(RequestSpan):
            sample = SamplingAbs[jSample]
            samReynolds, samMach = [], []
            for lab in Labels:
                samReynolds += [J.interpolate__([sample], s, ResultsDict[lab]['Reynolds'])[0]]
                samMach += [J.interpolate__([sample], s, ResultsDict[lab]['Mach'])[0]]
            samReynolds = np.array(samReynolds,dtype=np.float64, order='F')
            samMach = np.array(samMach,dtype=np.float64, order='F')

            # Store sampling
            SamplesDict[foilname]['Reynolds'][:,jSample] = np.linspace(samReynolds.min(), samReynolds.max(),RequestOperating)
            SamplesDict[foilname]['Mach'][:,jSample] = np.linspace(samMach.min(), samMach.max(),RequestOperating)

        # These are the dense 2D sampling:
        SamplesDict[foilname]['Mach'] = SamplesDict[foilname]['Mach'].flatten()
        SamplesDict[foilname]['Reynolds'] = SamplesDict[foilname]['Reynolds'].flatten()

        # Now, perform qhull-based sampling:
        NewMach, NewRe, Qhull = J.sampleIn2DQhull__(SamplesDict[foilname]['Mach'],SamplesDict[foilname]['Reynolds'],QhullNPts=QhullNPts,QhullScale=QhullScale, grading=grading, rescale=rescale)

        # Store result
        SamplesDict[foilname]['Mach'] = np.maximum(NewMach,0.0)
        SamplesDict[foilname]['Reynolds'] = np.maximum(NewRe,0.0)

        # including qhull:
        if includeQhull:
            QhullMach, QhullRe = J.getxy(Qhull)
            SamplesDict[foilname]['Qhull'] = dict(Mach=QhullMach, Reynolds=QhullRe)

    if showplot or isinstance(saveplot,str):
        import matplotlib.pyplot as plt

        if __PLOT_STIX_LATEX__:
            plt.rc('text', usetex=True)
            plt.rc('text.latex', preamble=r'\usepackage[notextcomp]{stix}')
            plt.rc('font',family='stix')

        from matplotlib.ticker import AutoMinorLocator

        fig, ax1 = plt.subplots(1,1,figsize=(4.75,4.25))
        FontSize = 10.
        ax2 = ax1.twinx()
        for label in Labels:
            ax1.plot(r/Rmax,ResultsDict[label]['Reynolds'],'-',label=label)
            ax2.plot(r/Rmax,ResultsDict[label]['Mach'],'--',label=label)

        minLocX = AutoMinorLocator()
        ax1.xaxis.set_minor_locator(minLocX)
        minLocY1 = AutoMinorLocator()
        ax1.yaxis.set_minor_locator(minLocY1)
        minLocY2 = AutoMinorLocator()
        ax2.yaxis.set_minor_locator(minLocY2)

        ax1.set_xlabel('$r/R$', fontsize=FontSize)
        ax1.set_ylabel(r'$Re_c^{(k)}$  (-)', fontsize=FontSize)
        ax2.set_ylabel('$M^{(k)}$ (- - -)', fontsize=FontSize)
        ax1.grid(which='major', axis='x')
        ax1.legend(loc="lower right",fontsize=FontSize-1.)

        if PolarInfoNode is not None:
            ax3 = ax1.twiny()
            SpanFoils = J.interpolate__(AirfoilPositions, s, r/Rmax)
            ax3.set_xticks(SpanFoils)
            ax3.set_xbound(ax1.get_xbound())
            ax3.set_xticklabels(AirfoilIdentifiers,rotation=20.)
            ax3.set_zorder(-100)

        ax2.format_coord = LL._makeFormat__(ax2, ax1)
        plt.tight_layout()

        if isinstance(saveplot,str):
            filename = saveplot
            print('Saving %s ...'%filename)
            plt.savefig(filename)
            print('ok')
        if showplot:     plt.show()

    return ResultsDict, SamplesDict

def plotOperatingRanges(ResultsDict=None, SamplesDict=None, PyZonePolars=None,
                        saveplot=None, showplot=True):
    '''
    Convenient macro-function used to plot at least one of
    the user-provided arguments.

    Parameters
    ----------

        ResultsDict : dict
            first output of :py:func:`getPolarOperatingRanges`

        SamplesDict : dict
            second output of :py:func:`getPolarOperatingRanges`

        PyZonePolars : list
            list of zones *PyZonePolar* containing 2D polar data

        saveplot : :py:class:`str` or :py:obj:`None`
            if a :py:class:`str` is provided, then this function saves
            PDF files with the corresponding curves

        showplot : bool
            if :py:obj:`True`, then the function opens a graphical window
            with the matplotlib figure plot

    .. note:: if **showplot** = :py:obj:`False` and **saveplot** = :py:obj:`None`
        then this function does not do anything useful

    '''
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator


    if __PLOT_STIX_LATEX__:
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage[notextcomp]{stix}')
        plt.rc('font',family='stix')

    # Declare figure
    fig, ax = plt.subplots(1,1,figsize=(4.75,4.25))

    # Plot ResultsDict
    if ResultsDict is not None:
        if 'Qhull' in ResultsDict:

            plt.plot(ResultsDict['Qhull']['Mach'], ResultsDict['Qhull']['Reynolds'],
                ls='-',color='k')

        ax.plot(ResultsDict['AllMach'], ResultsDict['AllReynolds'],
            ls='None',marker='x',color='grey',
            label='oper. points')

    # Plot Samples Dict
    if SamplesDict is not None:
        for foilname in SamplesDict:
            Mach = SamplesDict[foilname]['Mach'].ravel()
            Reynolds = SamplesDict[foilname]['Reynolds'].ravel()
            ax.plot(Mach,Reynolds,
                ls='None',marker='o',
                label='samples %s (%d)'%(foilname,len(Mach)))

    # Plot PyZonePolars
    if PyZonePolars is not None:
        for pzp in PyZonePolars:
            Reynolds = I.getValue(I.getNodeFromName(pzp,'Reynolds')).ravel()
            Mach = I.getValue(I.getNodeFromName(pzp,'Mach')).ravel()

            ax.plot(Mach,Reynolds,
                ls='None',marker='s',
                label='polar %s (%d)'%(pzp[0], len(Mach)))


    minLocX = AutoMinorLocator()
    ax.xaxis.set_minor_locator(minLocX)
    minLocY1 = AutoMinorLocator()
    ax.yaxis.set_minor_locator(minLocY1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('Mach')
    ax.set_ylabel('Reynolds')
    ax.legend(loc='upper left')
    plt.tight_layout()
    if isinstance(saveplot,str):
        filename = saveplot
        print('Saving %s ...'%filename)
        plt.savefig(filename)
        print('ok')
    if showplot: plt.show()


def saveKimFilesFromPUMA(Pb):
    '''

    **Original author: R. Boisard**
    **Included to MOLA by M. Balmaseda on 17/08/2021**

    .. warning:: Only a unique propeller is studied -> The anisotropies are not
        considered! Convergence is supposed as reached in the last iteration!
        Propeller must rotate around X axis, such that local blade's incidence
        can be computed as :math:`atan(y,x)`

    Parameters
    ----------

        Pb : problem object of PUMA
            the compatible problem object of PUMA

    Returns
    -------

        None : None
            only files kim.geom and kim.autre are writen
    '''

    Root=Pb.getPObject(PUMA.Fact.Root)
    NbProp=len(Root.Propellers)

    with open(Pb.getOutputFilePath('kim.geom'),'w') as f:
        for oneprop in range(NbProp):
            locprop=Root.Propellers[oneprop]
            NbBlades=locprop.NbBlades
            Blades=locprop.Blades

            for bladeno in range(NbBlades):
                locblade=locprop.Blades.getBlade(bladeno)
                bladegeom=locblade.Fluid.Aerodynamics['Definition']
                nbsec=len(bladegeom['Span'])
                localpitch=locprop.Direct*locblade.MBS.BladePitch.DoF[-1]*180./np.pi
                bladeazim=locblade.Azimuth
                bladepos=locblade.MBS.BladeJoin.Link.Origin

                f.write('Helice n. {:>3d}\n'.format(oneprop))
                f.write('Rmin   Rmax     Pas     %empilage     Azimuth     Xpos\n')
                f.write("%10.5f\t%10.5f\t%9.4f\t%8.4f\t%10.5f\t%10.5f\n"%(bladegeom['Span'][0],bladegeom['Span'][-1],localpitch,0.25,bladeazim,bladepos[0]))
                f.write('Nbsec   NbPale   NoPale\n')
                f.write("%d\t%d\t%d\n"%(nbsec,locprop.NbBlades,bladeno))
                for item in range(nbsec):
                    locpolar=bladegeom['Airfoil'][item]
                    if locpolar is None:
                        secname='xxxx'
                    else:
                        secname=locpolar.Name
                    f.write("%12.10e\t%12.10e\t%12.10e\t%12.10e\t%12.10e\t%s\n"%(bladegeom['Span'][item],bladegeom['Sweep'][item],
                                              bladegeom['Chord'][item],bladegeom['Twist'][item],bladegeom['Dihedral'][item],secname))


    num=Pb.get_Numerics()
    timestep=num.get('TimeStep')
    lastiter=Pb.Fluid.Iteration

    with open(Pb.getOutputFilePath('kim.autre'),'w') as f:
        f.write('Temperature [K] :\n'+str(Pb.Fluid.RefState['Temperature'])+'\n')
        f.write('densite [Kg/m3] :\n'+str(Pb.Fluid.RefState['Density'])+'\n')
        f.write('Incidence [deg] :\n'+str(np.arctan2(Pb.Fluid.RefState['VelocityY'],-Pb.Fluid.RefState['VelocityX'])*180./np.pi)+'\n')
        f.write('Derapage [deg]  :\n'+str(np.arctan2(Pb.Fluid.RefState['VelocityZ'],-Pb.Fluid.RefState['VelocityX'])*180./np.pi)+'\n')
        f.write('Mach avancement :\n'+str(Pb.Fluid.RefState['Mach'])+'\n')
        for oneprop in range(NbProp):
            rpm=locprop.Cmds.get('Omega')
            nbiterpartour=60./rpm/timestep
            pasdeg=rpm/30.*timestep*180.

            locprop=Root.Propellers[oneprop]
            f.write('Omega'+str(oneprop+1)+' [RPM] :\n'+str(rpm)+'\n')
            f.write('Pas azimuthal rotor '+str(oneprop+1)+' [deg] :\n'+str(pasdeg)+'\n')
            f.write('Nombre azimuth par tour rotor '+str(oneprop+1)+' :\n'+str(nbiterpartour)+'\n')
            f.write('Azimuth initial :\n'+str((lastiter-nbiterpartour)*pasdeg)+'\n')
            f.write('Azimuth final :\n'+str(lastiter*pasdeg)+'\n')

        f.write('Iteration a convergence :\n'+str(lastiter)+'\n')
