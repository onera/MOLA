import sys, os
import numpy as np
import Converter.PyTree   as C
import Converter.Internal as I
import MOLA.InternalShortcuts  as J
import MOLA.Wireframe  as W
import MOLA.PropellerAnalysis  as PA
import MOLA.LiftingLine  as LL


# Build a simple blade
ChordDict = dict(
RelativeSpan = [0.25,   0.45,  0.6,  1.0],
Chord =        [0.08,  0.12, 0.12, 0.03],
InterpolationLaw = 'akima', 
)


TwistDict = dict(
RelativeSpan = [0.25,  0.6,  1.0],
Twist        = [20.,  6.0, -1.0],
InterpolationLaw = 'akima',
)


PolarsDict = dict(RelativeSpan     = [  0.2,         1.000],
                  PyZonePolarNames = ['NACA4416','NACA4416'],
                  InterpolationLaw = 'interp1d_linear',)

Rmin = 0.15 # minimum radius of blade 
Rmax = 0.6  # maximum radius of blade
NPts =  50  # number of points discretizing the blade

# Non-uniform adapted discretization is recommended
RootSegmentLength = 0.0500 * Rmax
TipSegmentLength  = 0.0016 * Rmax
BladeDiscretization = dict(P1=(Rmin,0,0),P2=(Rmax,0,0),
                           N=NPts, kind='tanhTwoSides',
                           FirstCellHeight=RootSegmentLength,
                           LastCellHeight=TipSegmentLength)

LiftingLine = LL.buildLiftingLine(BladeDiscretization,
                                  Polars=PolarsDict,
                                  Chord =ChordDict,
                                  Twist =TwistDict,)
LL.resetPitch(LiftingLine, ZeroPitchRelativeSpan=0.75)

PolarInfoNode = I.getNodeFromName1(LiftingLine,'.Polar#Info')

C.convertPyTree2File(LiftingLine,'LiftingLine.cgns')

LL.prepareComputeDirectoryPUMA('LiftingLine.cgns', 'PolarCorrected.cgns', 
        DIRECTORY_PUMA='PUMA_DIR', GeomBladeFilename='GeomBlade.py',
        OutputFileNamePreffix='HOST_')

def computePUMA(NBlades,Velocity,RPM,Temperature,Density,
    Constraint='Pitch',
    ConstraintValue=0.,
    PerturbationField=None,
    PrescribedWake=False,
    FroudeVelocity=False,
    Restart=False,
    GeomBladeFile='PUMA_DIR/GeomBlade.py',
    PUMADir='PUMA_DIR', 
    Spinner=None,
    ExtractVolume=None,
    SpanDiscretization=None,
    StopDeltaTtol=None,
    AdvancedOptions=dict(
        AdvanceRatioHoverLim=0.01,
        BoundMach=0.95,
        Boundaries=[False,True],
        WakeRevs=5.,
        MinimumRevsAfterWakeRevs=5.,
        MaximumNrevs=10.,
        Dpsi=10.,
        NbSections=None,
        InitialPitch=0.),
    ):

    import imp
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
    print(GeomBladeFile)
    BladeGeom = imp.load_source('Geom',GeomBladeFile)
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
                SpinnerZones,_ = RW.makeHub(SpinnerProfile,AxeCenter=(0,0,0),AxeDir=(1,0,0),NPsi=91,BladeNumberForPeriodic=None,LeadingEdgeAbscissa=0.25,TrailingEdgeAbscissa=0.75,SmoothingParameters={'eps':0.50,'niter':300,'type':2},SplitLEind=None,SplitTEind=None)
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


    CurrentLiftingLine = Pb.Fluid.objects['LiftingElement'][0]

    LLgeom = CurrentLiftingLine.getGeom3D()
    LLgeom = I.copyTree(I.getNodeFromName(LLgeom,'QC'))
    LLfields = I.copyTree(Prop.Blades.Blades[0].Fluid.BladeLLSectionalLoads.getDataAsTree())
    FlowSol_n = I.getNodeFromName(LLfields,'FlowSolution')
    LLgeom[2].append(FlowSol_n)
    LLgeom[2].append(PolarInfoNode)


    NewVars = ['Density', 'VelocityTangential',
               'VelocityMagnitudeLocal', 'Reynolds',
               ]
    v = J.invokeFieldsDict(LLgeom, NewVars)
    
    FxMBS,FzMBS,VYLL,VX,VY,VZ,Chord=J.getVars(LLgeom,['Fx_Rotor_MBS_Prop',
                                       'Fz_Rotor_MBS_Prop',
                                       'VY_LL',
                                       'VX_SEC',
                                       'VY_SEC',
                                       'VZ_SEC',
                                       'Chord'])


    v['Density'][:] = Density 
    v['VelocityTangential'][:] = np.abs(VYLL)
    v['VelocityMagnitudeLocal'][:] = np.sqrt(VX**2 + VY**2 + VZ**2)
    mu = 1.795e-5 # CAVEAT
    v['Reynolds'][:] = Density * v['VelocityMagnitudeLocal'] * Chord / mu

    J.set(LLgeom, '.Conditions', NBlades=NBlades, Temperature=Temperature,
            Density=Density, RPM=RPM)


    PitchField, Span = J.getVars(LLgeom, ['Pitch', 'Span'])
    Twist, = J.invokeFields(LLgeom, ['Twist'])
    Twist[:] = J.interpolate__(Span, BladeDef['Span'], BladeDef['Twist'])
    Twist += PitchField

    I._renameNode(LLgeom, 'Alpha', 'AoA')

    C.convertPyTree2File(LLgeom,'LiftingLine.cgns')



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


    # Extract integral loads
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





computePUMA(3,15.0,3000.,288.,1.225,
    Constraint='Pitch',
    ConstraintValue=10.,
    )

