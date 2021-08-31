'''
ISOLATED PROPELLER PERIODIC CHANNEL WORKFLOW
for use with MOLA v1.5.

This script assumes that the configuration is a single 
propeller with no side-slip (only advancing), with rotation
of the propeller around the (-X) axis, and freestream advance
velocity towards the  (+X) axis.

The blade is positioned towards (+Z) axis.

                 |--------------------|
                 |File user inputs :  |
                 |                    |
                 |blade.cgns          |
                 |SpinnerProfile.cgns |
                 |--------------------|


General user inputs (see script for specific details):
    
    1. BLADE SURFACE : blade.cgns file. Mesh units shall be meters.
    Blade surface, where main blade surface is single-bloc,
    with outwards-pointing normal and such that jmax boundary is 
    located at the tip.
    The tip shall be closed with additional surfaces.
    The root shall be totally contained INSIDE the spinner.


    2. SPINNER GEOMETRY : SpinnerProfile.cgns. Mesh units shall be
    meters.
    It MUST be placed at the XZ (y=0) plane. It MUST start from
    Leading Edge.


    3. DISCRETIZATION PARAMETERS
    Discretization parameters of spinner, extrusions...
    Also the smoothing parameters of the extrusion operation


    4. FLIGHT CONDITIONS
    This is used in order to set appropriate ReferenceState
    and other CGNS-related data.


    5. SPLIT INFORMATION
    In order to split the mesh and distribute to processors.



History of current file:
v1.0 - 17/01/2020 - L. Bernardos - Creation by recycling.
'''


import sys, timeit
import numpy as np

# Import Cassiopee
import Converter.PyTree as C
import Converter.elsAProfile as EP
import Converter.Internal as I
import Connector.PyTree as X
import Transform.PyTree as T
import Generator.PyTree as G
import Geom.PyTree as D
import Post.PyTree as P
import Distributor2.PyTree as D2

# Import MOLA (MOdule pour des Logiciels en Aerodynamique)
import InternalShortcuts as J
import Wireframe as W
import GenerativeShapeDesign as GSD
import GenerativeVolumeDesign as GVD
import RotatoryWings as RW

tic = timeit.default_timer()

# ===================================================== #
# ==========         MAIN PARAMETERS         ========== #
# ===================================================== #

# ~~~ Execution of Script ~~~ #
# Blade:
BuildBladeSurface   = True #
ExtrudeBlade        = True #
# Spinner:
BuildSpinnerSurface = True #
BuildPeriodicSurfs  = True #
ExtrudeSpinner      = True #
# Assembly
AssembleSplitConnect= True  #
AddOversetData      = True  #
OptimizeOverlap     = True  #
MotionRefStateExtrct= True  #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

CreateInitialSolution= False #


# Prints, plots...
printIters=True
plotIters =True


# --------- Flight conditions (for CGNS setup) --------- #

Condition = 'CR_FL100'

if Condition == 'CR_FL100':
    # Altitude= 10000. * 0.3048 # M
    Temperature = 268.338
    Pressure    = 69681.7
    VitesseKTS = 90.0
    Vitesse=(VitesseKTS*1.852)/3.6  # M/S
    RPM=2350.0 # Revolutions per minute. 
    Pitch=11.0 # RELATIVE PITCH CHANGE (degree)

RPM   = -RPM         # Revolutions per minute. 
Uinf  = Vitesse      # Propeller advance velocity (m/s)
Pinf  = Pressure     # Pressure at freestream (Pa)
Tinf  = Temperature  # Temperature at freestream (K)
Tuinf = 0.01 * 0.01  # Turbulence intensity at freestream
ReTinf= 0.1          # Turbulence Reynolds number at freest.
# ------------------------------------------------------ # 


# ------------ GENERAL GEOMETRY INFORMATION ------------ #
NBlades = 3        # Number of Blades
Rmin    = 0.107    # Blade Root radius at definition (meters)
Rmax    = 0.760    # Blade Tip radius at definition (meters)
Pitch   = Pitch    # Rotation of blade w.r.t +Z axis (degree)
# ------------------------------------------------------ #

def checkRepeatedZoneNames(t):
    zones = I.getZones(t)
    ZonesNames = [I.getName(z) for z in zones]
    RepeatedZoneNames = []
    for zn in ZonesNames:
        Multiplicity = ZonesNames.count(zn)
        if Multiplicity > 1 and zn not in RepeatedZoneNames:
            RepeatedZoneNames += [zn]
    if len(RepeatedZoneNames) > 0:
        print ('WARNING : Repeated zone names:\n%s'%str(RepeatedZoneNames))

    return RepeatedZoneNames


# ----------------- SPLIT  INFORMATION ----------------- #
ProcPointsLoad = 2e5  # Points approx. load per processor

NPtsTrailingEdge=9
WallCellHeight = 1.68e-6
BladeRootCellWidth = 0.02



if BuildBladeSurface:

    print('Building Blade surface...')
    # Read Blade
    BladeClosed = C.convertFile2PyTree('blade.cgns')
    BladeSurf = I.getZones(BladeClosed)[0]

    # Apply pitch
    T._rotate(BladeClosed,(0,0,0),(0,0,1),Pitch) # Apply Pitch

    C.convertPyTree2File(BladeClosed,'MainBladeGeometry.cgns')
else:
    BladeClosed = C.convertFile2PyTree('MainBladeGeometry.cgns')
    BladeSurf = I.getZones(BladeClosed)[0]


# Up to here the main blade geometry, including tip, is
# built and discretized. Now we proceed to build the spinner
# surface from a given profile.


# ~~~~~~~~~~~~~~~~ BUILD SPINNER SURFACE ~~~~~~~~~~~~~~~~ # 

# For optimum result, the profile shape of the spinner (which
# is a 3D planar curve) shall be supported on the plane that
# bisects the blade. The profile shall be a single structured
# curve, starting at Leading Edge (the leading edge extrema
# shall coincide with blade's rotation axis).
# Trailing Edge may not necessarily coincide with rotation
# axis (for modeling e.g. infinite spinner body).

# For example: For a blade positioned towards positive Z axis
# that is expected to rotate on the right-hand-side positive
# X axis, its Spinner profile shall be supported on the plane
# defined by Z and X axis (hence, all CoordinateY are zero),
# and the first point, defining the Leading Edge, is placed
# at the minimum X coordinate and Z=0.

# SPINNER PARAMETERS
NPtsAzimut = 101 # MUST BE ODD
SpinnerLeadingEdgeTFIzoneAbscissa = 0.02
SpinnerSurfaceSmoothing = dict(eps=0.5, niter=10, type=0)
SpinnerCentralCellSize = 0.001
SpinnerExtrusionDistance = 10*Rmax
DownstramDistance = SpinnerExtrusionDistance
SpinnerProfileDiscretizations = [
dict(kind='tanhTwoSides', N=100, FirstCellHeight=0.0001,LastCellHeight=SpinnerCentralCellSize,BreakPoint=0.7),
dict(kind='tanhTwoSides', N=100, FirstCellHeight=SpinnerCentralCellSize,LastCellHeight=SpinnerCentralCellSize,BreakPoint=1.),]

if BuildSpinnerSurface:
    print('Building Spinner surface...')    
    # Build spinner surface:
    SpinnerProfile = C.convertFile2PyTree('SpinnerProfile.cgns')
    SpinnerProfile = I.getZones(SpinnerProfile)[0]

    SpinnerProfile = W.polyDiscretize(SpinnerProfile, SpinnerProfileDiscretizations)


    x,y,z = J.getxyz(SpinnerProfile)
    y[0],z[0] = 0., 0.
    DwnLine = D.line((x[-1],y[-1],z[-1]),(x[-1]+DownstramDistance,y[-1],z[-1]),2)
    DwnLine = W.polyDiscretize(DwnLine, 
        [dict(kind='tanhTwoSides',N=120,FirstCellHeight=SpinnerCentralCellSize,LastCellHeight=0.05*DownstramDistance,BreakPoint=1.),])

    SpinnerProfile = T.join(SpinnerProfile,DwnLine)


    C.convertPyTree2File(SpinnerProfile,'SpinnerProfileWithCylinder.cgns')

    # Fine-tune the azimutal position orientation
    T._rotate(SpinnerProfile,(0,0,0),(1,0,0),0.)

    # Build Spinner surface
    SpinnerSurf, PeriodicProfiles = RW.makeHub(SpinnerProfile,AxeCenter=(0,0,0),AxeDir=(1,0,0),NPsi=NPtsAzimut,BladeNumberForPeriodic=NBlades,LeadingEdgeAbscissa=SpinnerLeadingEdgeTFIzoneAbscissa,TrailingEdgeAbscissa=None,
        SmoothingParameters=SpinnerSurfaceSmoothing)

    # Nota Bene:
    # SpinnerSurf is a PyTree
    # PeriodicProfiles are two Zones

    SpinnerSurfZones = I.getZones(SpinnerSurf)
    SpinnerTree = C.newPyTree(['SpinnerSurface',SpinnerSurfZones,'SpinnerPeriodicBounds',PeriodicProfiles])


    C.convertPyTree2File(SpinnerTree,'SpinnerSurfaceWithPeriodicProfiles.cgns')
else:
    SpinnerTree = C.convertFile2PyTree('SpinnerSurfaceWithPeriodicProfiles.cgns')
    SpinnerBase = I.getNodeFromName1(SpinnerTree,'SpinnerSurface')
    PreiodicProfilesBase = I.getNodeFromName1(SpinnerTree,'SpinnerPeriodicBounds')
    SpinnerZones     = I.getZones(SpinnerBase)
    PeriodicProfiles = I.getZones(PreiodicProfilesBase)
    SpinnerSurf = C.newPyTree(['SpinnerSurface',SpinnerZones])

# ~~~~~~~~~~~~~~~~ EXTRUDE BLADE SURFACE ~~~~~~~~~~~~~~~~ #

# Build Blade Body-fitted distributions
NPtsBladeNormal       = 100
RootExtrusionDistance = 0.07
TipExtrusionDistance  = RootExtrusionDistance*0.9
NPtsExtrapolateRegion = 100
LastCellHeight        = 0.005

RootContour = GSD.getBoundary(BladeSurf,'jmin')
if ExtrudeBlade:
    print('Extruding Blade...')

    # Define the blade's normal extrusion distribution at root
    RCx, RCy, RCz = J.getxyz(RootContour)
    RootDistribution = W.linelaw(
        P1=(RCx[0], RCy[0], RCz[0]),
        P2=(RCx[0]+RootExtrusionDistance, RCy[0], RCz[0]),
        N=NPtsBladeNormal,
        Distribution=dict(kind='tanhTwoSides',
                          FirstCellHeight=WallCellHeight,
                          LastCellHeight=LastCellHeight))

    # RootDistribution extrusion parameters
    nf, gf, nit, git = J.invokeFields(RootDistribution,['normalfactor','growthfactor','normaliters','growthiters'])
    #----Growth factor and iterations
    gf[:]  = np.linspace(0.1,10,len(gf))#0.1
    git[:] = np.linspace(300,100,len(gf))#100
    #----Normals smoothing factor and iterations
    nf[:]  = 100
    nit[:] = np.linspace(50,100,len(gf))#100


    # Define the blade's normal extrusion distribution at tip
    TipContour = GSD.getBoundary(BladeSurf,'jmax')
    TCx, TCy, TCz = J.getxyz(TipContour)

    TipDistribution = W.linelaw(
        P1=(TCx[0], TCy[0], TCz[0]),
        P2=(TCx[0]+TipExtrusionDistance, TCy[0], TCz[0]),
        N=NPtsBladeNormal,
        Distribution=dict(kind='tanhTwoSides',
                          FirstCellHeight=WallCellHeight,
                          LastCellHeight=LastCellHeight))

    # TipDistribution extrusion parameters
    nf, gf, nit, git = J.invokeFields(TipDistribution,['normalfactor','growthfactor','normaliters','growthiters'])

    
    #----Growth factor and iterations
    gf[:]  = np.linspace(0.1,0.4,len(gf))#0.1
    git[:] = np.linspace(300,200,len(gf))#100
    #----Normals smoothing factor and iterations
    nf[:]  = 100
    nit[:] = np.linspace(20,50,len(gf))#100

    growthEquation='nodes:dH={nodes:dH}*minimum(maximum(1.+tanh(-{nodes:growthfactor}*{nodes:divs}),0.9),1.1)'



    BladeExtruded = GVD.extrudeWingOnSupport(BladeClosed,SpinnerSurf,[RootDistribution, TipDistribution],SupportMode='intersection',extrapolatePoints=NPtsExtrapolateRegion, InterMinSep=0.005,CollarRelativeRootDistance=0.1, extrudeOptions=dict(ExtrusionAuxiliarCellType='TRI', printIters=printIters, plotIters=plotIters, growthEquation=growthEquation))

    C.convertPyTree2File(BladeExtruded,'BladeExtruded.cgns')

else:
    BladeExtruded = C.convertFile2PyTree('BladeExtruded.cgns')


# Get the main blade zone
BladeExtrudedZones = I.getZones(BladeExtruded)
MainBladeExtruded,_  = GSD.getNearestZone(BladeExtrudedZones,(0,0,0))




# ~~~~~~~~~~~~~~~~~~~ SPINNER EXTRUSION ~~~~~~~~~~~~~~~~~~~ #

# Build the distribution. Choose a strategy:
# 'Prescribed': Use a prescribed distribution
# 'BladeBased': Use the blade's spanwise distribution to
#               extrude the spinner, then make a constant
#               growth ratio expansion.

Strategy = 'Prescribed' 

if Strategy == 'Prescribed':
    # Make a single distribution:
    MaximumAdditionalDomainLength  = 10*Rmax
    AdditionalPointsAfterBlade     = 150 
    AdditionalGrowthRateAfterBlade = 1.08
    JoinPoint = (0,0,0.95*Rmax)
    #                v<<< This factor is important <<<<<
    #                v
    JoinCellLength = 0.0003

    Nspan = I.getZoneDim(BladeSurf)[2]
    
    SpanwiseDistributionBlade = W.linelaw(P1=(0,0,Rmin),P2=JoinPoint,N=Nspan+NPtsExtrapolateRegion,
        Distribution=dict(kind='tanhTwoSides',FirstCellHeight=WallCellHeight, LastCellHeight=JoinCellLength))

    AdditionalDistribution = W.linelaw(
        P1=JoinPoint,
        P2=(0,0,0.95*Rmax+MaximumAdditionalDomainLength),
        N=AdditionalPointsAfterBlade,
        Distribution=dict(kind='ratio',
            FirstCellHeight=JoinCellLength,
            growth=AdditionalGrowthRateAfterBlade)
        )

    SpinnerExtrusionDistribution = T.join(SpanwiseDistributionBlade, AdditionalDistribution)    


elif Strategy == 'BladeBased':
    # Employ the blade spanwise  distribution in order to 
    # construct the spinner normal extrusion distribution.
    MaximumAdditionalDomainLength  = 10*Rmax
    AdditionalPointsAfterBlade     = 80
    AdditionalGrowthRateAfterBlade = 1.08

    _,Ni,Nj,Nk,_ = I.getZoneDim(MainBladeExtruded)
    SpanwiseDistributionBlade = T.subzone(MainBladeExtruded, (int(NPtsTrailingEdge/2),1,Nk),(int(NPtsTrailingEdge/2),Nj,Nk))
    SDBx, SDBy, SDBz = J.getxyz(SpanwiseDistributionBlade)

    JoinCellLength = ((SDBx[-1]-SDBx[-2])**2 +
                     (SDBy[-1]-SDBy[-2])**2 +
                     (SDBz[-1]-SDBz[-2])**2)**0.5

    AdditionalDistribution = W.linelaw(
        P1=(SDBx[-1], SDBy[-1], SDBz[-1]),
        P2=(SDBx[-1], SDBy[-1], SDBz[-1]+MaximumAdditionalDomainLength),
        N=AdditionalPointsAfterBlade,
        Distribution=dict(kind='ratio',
            FirstCellHeight=JoinCellLength,
            growth=AdditionalGrowthRateAfterBlade)
        )

    SpinnerExtrusionDistribution = T.join(SpanwiseDistributionBlade, AdditionalDistribution)    

else: 
    raise ValueError('Strategy %s not recognized'%Strategy)



# Smoothing parameters of Spinner Extrusion
NPtsDomain = C.getNPts(SpinnerExtrusionDistribution)
nf, gf, nit, git = J.invokeFields(SpinnerExtrusionDistribution,['normalfactor','growthfactor','normaliters','growthiters'])

gf[:]  =np.linspace(0.7,0.0,len(gf))
git[:] =np.linspace(100,600,len(gf))
nf[:]  =1000 
nit[:] =np.linspace(20,150,len(gf)) 


if BuildPeriodicSurfs:
    print('Extruding Periodic surfaces...')    

    # Extrude the periodic profiles
    growthEquation='nodes:dH={nodes:dH}*maximum(1.+tanh(-0.2*{nodes:divs}),0.9)*(1+{nodes:growthfactor}*tanh(min({nodes:vol})/({nodes:vol})))'

    FirstPeriodicSurf, SecondPeriodicSurf = RW.extrudePeriodicProfiles(PeriodicProfiles,Distributions=[SpinnerExtrusionDistribution],NBlades=NBlades,AxeDir=(1,0,0),extrudeOptions=dict(ExtrusionAuxiliarCellType='ORIGINAL', printIters=printIters, plotIters=plotIters,growthEquation=growthEquation)
        )
    C.convertPyTree2File([FirstPeriodicSurf, SecondPeriodicSurf], 'PeriodicSurfs.cgns')
    # from "else"
    ps = C.convertFile2PyTree('PeriodicSurfs.cgns')
    FirstPeriodicSurf, SecondPeriodicSurf = I.getZones(ps)

else:
    ps = C.convertFile2PyTree('PeriodicSurfs.cgns')
    FirstPeriodicSurf, SecondPeriodicSurf = I.getZones(ps)


# ~~~~ Extrude Domain grid ~~~~ #

FirstProfile, SecondProfile = PeriodicProfiles[:2]
Constraints = [
dict(kind='Match',curve=FirstProfile, surface=FirstPeriodicSurf,MatchDir=None),
dict(kind='Match',curve=SecondProfile, surface=SecondPeriodicSurf),
]

if ExtrudeSpinner:
    print('Extruding Spinner...')    
    # Use auxiliar ExtractMesh strategy
    ExtractMesh = D.axisym(FirstPeriodicSurf, (0,0,0),(1,0,0), angle=360./NBlades, Ntheta=NPtsAzimut)
    ExtractMesh[0]='ExtractMesh'
    sx, sy, sz, dH = J.invokeFields(ExtractMesh,['sx','sy','sz','dH'])
    _,Ni,Nj,Nk,_=I.getZoneDim(ExtractMesh)
    x,y,z = J.getxyz(ExtractMesh)
    dH[:,1:,:] = ((x[:,1:,:]-x[:,:-1,:])**2+(y[:,1:,:]-y[:,:-1,:])**2+(z[:,1:,:]-z[:,:-1,:])**2)**0.5
    sx[:,1:,:] = ( x[:,1:,:] - x[:,:-1,:]) / dH[:,1:,:]
    sy[:,1:,:] = ( y[:,1:,:] - y[:,:-1,:]) / dH[:,1:,:]
    sz[:,1:,:] = ( z[:,1:,:] - z[:,:-1,:]) / dH[:,1:,:]

    # Constraints = []
    SpinnerExtrusionTree = GVD.extrude(SpinnerSurf,[SpinnerExtrusionDistribution],Constraints,
        extractMesh=ExtractMesh, 
        extractMeshHardSmoothOptions=dict(eps=0.1, niter=100, type=2, HardSmoothLayerProtection=70, FactordH=2.),
        modeSmooth='ignore',
        printIters=printIters, plotIters=plotIters)
    ExtrudedVolume = I.getNodeFromName1(SpinnerExtrusionTree,'ExtrudedVolume')

    C.convertPyTree2File(ExtrudedVolume,'DomainGrid.cgns')

    # From "else"
    ExtrudedVolume = C.convertFile2PyTree('DomainGrid.cgns')
else:
    ExtrudedVolume = C.convertFile2PyTree('DomainGrid.cgns')

DomainZones = I.getZones(ExtrudedVolume)


# --------------------------------------------------------- #
# Right here, the mesh is COMPLETED. Now, assemble the mesh,
# name it properly, create families for elsA, perform the
# Overset operations, Distribute the Tree, and connect it.
# --------------------------------------------------------- #
# Defines a function to add UserDefinedData_t  node and its DataArray_t children (or type1 and type2)
def _addSetOfNodes(parent, name, ListOfNodes, type1='UserDefinedData_t', type2='DataArray_t'):
    '''
    parent : Parent node
    name : name of the node
    ListOfNodes : First element is the node name, 
    and the second element is its value...
    ... -> [[nodename1, value1],[nodename2, value2],etc...]
    '''
    children = []
    for e in ListOfNodes:
        typeOfNode = type2 if len(e) == 2 else e[2]
        children += [I.createNode(e[0],typeOfNode,value=e[1])]
    node = I.createNode(name,type1, children=children)
    I.addChild(parent, node)

    return None



if AssembleSplitConnect:
    print('Assembling, splitting and connecting mesh...')    
    # Assemble the Tree
    for z in BladeExtrudedZones: z[0] = 'Blade_'+z[0]
    for z in DomainZones: z[0] = 'Bckgnd_'+z[0]


    t = C.newPyTree(['BaseBlade',BladeExtrudedZones, 'BaseBackground', DomainZones])
    C.registerZoneNames(t)
    BaseBlade = I.getNodeFromName1(t,'BaseBlade')
    BaseBackground = I.getNodeFromName1(t,'BaseBackground')

    # Check the existance of negative-volume cells
    '''
    tCheck = G.getVolumeMap(t)
    for zchk in I.getZones(tCheck):
        vol, = J.getVars(zchk,['vol'],'FlowSolution#Centers')
        NegCells = vol<0
        if np.any(NegCells):
            x,y,z = J.getxyz(zchk)
            xNeg = x[NegCells][0]
            yNeg = y[NegCells][0]
            zNeg = z[NegCells][0]
            print("WARNING : Zone %s has negative cells around point (%g,%g,%g)"%(zchk[0],xNeg,yNeg,zNeg))
    '''
    # Define some boundary conditions
    for z in BladeExtrudedZones:
        C._addBC2Zone(z,'BladeBCWall','FamilySpecified:BLADE','kmin')
        C._addBC2Zone(z,'BladeBCOverlap','BCOverlap','kmax')
    MainBladeZone,_ = GSD.getNearestZone(BladeExtrudedZones,(0,0,0))
    C._addBC2Zone(MainBladeZone,'BladeBCWall','FamilySpecified:BLADE_ON_SPINNER','jmin')


    for z in DomainZones:
        C._addBC2Zone(z,'SpinnerBCWall','FamilySpecified:SPINNER','kmin')
        C._addBC2Zone(z,'SpinnerBCFarfield','FamilySpecified:NREF','kmax')


    # Add Families
    C._addFamily2Base(BaseBlade,'BLADE',bndType='BCWallViscous')
    C._addFamily2Base(BaseBlade,'BLADE_ON_SPINNER',bndType='BCWallViscous')
    C._addFamily2Base(BaseBackground,'SPINNER',bndType='BCWallViscous')
    C._addFamily2Base(BaseBackground,'NREF',bndType='BCFarfield')
    C._addFamily2Base(BaseBackground,'PERIODIC_BLOCK')
    C._addFamily2Base(BaseBlade,'BLADE_BLOCK')
    C._tagWithFamily(DomainZones, 'PERIODIC_BLOCK')
    C._tagWithFamily(BladeExtrudedZones, 'BLADE_BLOCK')


    # Split and distribute 
    MeshTotalNPts = 0
    for zone in I.getZones(t): MeshTotalNPts += C.getNPts(zone)
    print ('Mesh total number of points : %g'%MeshTotalNPts)
    Nprocs = int(np.round(MeshTotalNPts / float(ProcPointsLoad)))

    if Nprocs > 1:
        # Perform Splitting and distribution
        print ("Splitting...")
        t = T.splitNParts(t, Nprocs, multigrid=0, dirs=[1,2,3], recoverBC=True)

        # Force break all connectivity
        I._rmNodesByType(t,'GridConnectivity1to1_t')

        # Force check multiply-defined zone names
        RepeatedZoneNames = checkRepeatedZoneNames(t)
        if len(RepeatedZoneNames) > 0: t = I.correctPyTree(t,level=3)

        # Re-Connect the resulting blocks
        t = X.connectMatch(t, dim=3, tol=1e-9)

        print ("Distributing...")
        t,stats=D2.distribute(t, Nprocs, useCom=0)

        # Check if all procs have at least one block assigned
        zones = I.getZones(t)
        ProcDistributed = [I.getValue(I.getNodeFromName(z,'proc')) for z in zones]

        for p in range(max(ProcDistributed)):
            if p not in ProcDistributed:
                raise ValueError('Bad proc distribution! rank %d is empty'%p)


    # Set Periodicity basic information

    print ("Making Periodic Match...")
    t = X.connectMatchPeriodic(t,
        rotationCenter=[0,0,0],
        rotationAngle=[360/float(NBlades),0,0],
        unitAngle='Degree',
        tol=1.e-7,
        dim=3)    

    print ('Adapting Periodic Match...')
    EP._adaptPeriodicMatch(t,clean=True)
    C._fillEmptyBCWith(t,'SpinnerBCFarfield','FamilySpecified:NREF',dim=3)

    C.convertPyTree2File(t,'Case_Assembled.cgns')
    # from "else"
    t = C.convertFile2PyTree('Case_Assembled.cgns')

else:
    t = C.convertFile2PyTree('Case_Assembled.cgns')

# ======================================================== #
# =================== OVERSET ASSEMBLY =================== #
# ======================================================== #

if AddOversetData:
    print('Performing Overset Assembly...')    

    DEPTH, DIM = 2, 3

    # Set cellN=2 to centers near overlap BCs
    t = X.applyBCOverlaps(t, depth=DEPTH)


    # ~~ Get bodies of each base

    # Make Blade Watertight surface
    BladeRootOpenTRI = C.convertArray2Tetra(BladeClosed)
    RootContourTRI   = G.delaunay(RootContour)
    BladeClosedTRI   = T.join(BladeRootOpenTRI, RootContourTRI)
    G._close(BladeClosedTRI)
    BladeClosedTRI = I.getZones(BladeClosedTRI)[0]
    # # Make Spinner Watertight surface
    # SpinnerBody = D.axisym(PeriodicProfiles[0], (0,0,0),(1,0,0), angle=360., Ntheta=NPtsAzimut*NBlades) 
    # SpinnerBody = C.convertArray2Tetra(SpinnerBody)
    # G._close(SpinnerBody)
    # # Close the Spinner Body if it is open at Trailing Edge
    # try:
    #     OpenTEContour = P.exteriorFaces(SpinnerBody)
    #     ClosedTEContour = G.delaunay(OpenTEContour)
    #     SpinnerBody = T.join(SpinnerBody,ClosedTEContour)
    #     G._close(SpinnerBody)
    # except:
    #     pass

    # Order of this list is important: shall be consistent with
    # bases, as this order is used in Blanking Matrix
    bodies = [[BladeClosedTRI]]

    # c = C.getFields(I.__GridCoordinates__, bodies[0])
    # sys.exit()
    # blanking matrix.
    # The blanking matrix BM is a numpy array of size nbases x nbodies.
    # BM(i,j)=1 means that ith basis is blanked by jth body.

    #            body 
    #            Blade
    BM = np.array([[0],   # Blade  base
                   [1]])  # Spinner/Background base


    # # blanking
    # t = X.blankCells(t, bodies, BM, depth=DEPTH, dim=DIM)

    if DIM == 2:
        t = X.blankCells(t, bodies, BM, depth=DEPTH, dim=2)
    else:
        t = X.blankCellsTri(t,[bodies], BM, blankingType='cell_intersect')
        # t = X.blankCells(t, [bodies], BM, depth=DEPTH, dim=3, blankingType=-2)

    # set interpolated points around blanked points
    t = X.setHoleInterpolatedPoints(t, depth=DEPTH)

    # Overlap optimization with a high priority to blade
    if OptimizeOverlap:
        print('Optimizing overlap...')
        t = X.optimizeOverlap(t, double_wall=1,
            priorities=['BaseBlade',0,'BaseBackground',1])

    print('Maximizing Blanked Cells...')
    t = X.maximizeBlankedCells(t, depth=DEPTH)


    print('Computing interpolation coefficients...')
    t = X.setInterpolations(t, loc='cell', sameBase=0, double_wall=1, 
        # planarTol=10*WallCellHeight,
        )

    # for interpolated interface (only for depth = 1)
    if DEPTH == 1: t = X.setInterpolations(t, loc='face', sameBase=0)


    # DEBUG ORPHAN POINTS
    t2 = X.chimeraInfo(t, type='orphan')
    OrphanPoints = X.extractChimeraInfo(t2, type='orphan', loc='centers')
    C.convertPyTree2File(OrphanPoints,'orphans.cgns')

    # Convert tree to elsAxdt profile (for rereading by elsAxdt)
    EP._convert2elsAxdt(t)


    # compute OversetHoles nodes
    t = X.cellN2OversetHoles(t)

    # Remove spurious PointList of orphan points
    OPL_ns = I.getNodesFromName(t,'OrphanPointList')
    for opl in OPL_ns:
        ID_node, _ = I.getParentOfNode(t,opl)
        I.rmNode(t,ID_node)


    C.convertPyTree2File(t,'Case_AssembledOverset.cgns')
    # from "else"
    t=C.convertFile2PyTree('Case_AssembledOverset.cgns')
else:
    t=C.convertFile2PyTree('Case_AssembledOverset.cgns')





if MotionRefStateExtrct:
    print('Adding elsA nodes of Motion, RefState and Extractions...')    
    # Set Motion and Reference State information
    BaseBlade = I.getNodeFromName1(t,'BaseBlade')
    BaseBackground = I.getNodeFromName1(t,'BaseBackground')
    DomainZones = I.getZones(BaseBackground)
    for MotionBlock in ['PERIODIC_BLOCK', 'BLADE_BLOCK']:
        PerBlk_n = I.getNodeFromName2(t,MotionBlock)

        # Add Motion data
        children = [
        ['motion',1],
        ['omega',float(RPM*np.pi/30.)], 
        ['axis_pnt_x',0.],
        ['axis_pnt_y',0.],
        ['axis_pnt_z',0.],
        ['axis_vct_x',1.],
        ['axis_vct_y',0.],
        ['axis_vct_z',0.],
        ['transl_vct_x',1.],
        ['transl_vct_y',0.],
        ['transl_vct_z',0.],
        ['transl_speed',0.],
        ]
        _addSetOfNodes(PerBlk_n,'.Solver#Motion',children)

    # Add .Solver#BC with motion data to surfaces
    FamiliesBCWithMotion = ['BLADE', 'SPINNER', 'BLADE_ON_SPINNER']
    for famname in FamiliesBCWithMotion:
        children = [
        ['mobile_coef',-1]
        ]

        if   famname == 'BLADE':   children += [['family', 101]]
        elif famname == 'SPINNER': children += [['family', 102]]
        elif famname == 'BLADE_ON_SPINNER': children += [['family', 102]]

        fam_n = I.getNodeFromName2(t,famname)
        _addSetOfNodes(fam_n,'.Solver#BC',children)    


    # Add ReferenceState
    Gamma, Rgp = 1.4, 287.058
    Mus, Cs, Ts= 1.711e-5, 110.4, 273.0

    SoundSpeed = (Gamma*Rgp*Tinf)**0.5 
    Mach = Uinf/ SoundSpeed
    T_tot = Tinf * (1+0.5*(Gamma-1.)*Mach**2)
    Muinf  = Mus*((Tinf/Ts)**0.5)*((1.+Cs/Ts)/(1.+Cs/Tinf))
    einf  = Rgp*Tinf/(Gamma - 1.)
    Einf  = einf + 0.5*Uinf**2

    Ro = Pinf/((Gamma - 1.) * einf)
    RoU, RoV, RoW = Ro * Uinf, 0., 0.
    RoE = Ro * Einf
    RoK = 3.*0.5*(Tuinf**2)*(Uinf**2)*Ro
    RoO = Ro * RoK/( ReTinf*Muinf )

    MachTip = (RoU**2 + (Rmax* (RPM*np.pi/30.))**2) ** 0.5 / SoundSpeed

    procs = [pr[1] for pr in I.getNodesFromName(t,'proc')]
    Nprocs = np.max(np.array(procs))+1
    EP._addReferenceState(t,conservative=[float(Ro), float(RoU), float(RoV), float(RoW), float(RoE), float(RoK), float(RoO)], turbmod='komega', name='ReferenceState', comments='General information:\n\nAdvanceMach = %g\nMachTip = %g\nPitch = %g deg\nNprocs = %g'%(Mach,MachTip,Pitch,Nprocs))



    Variables2Extract = [
    'Density',
    'MomentumX','MomentumY','MomentumZ',
    'EnergyStagnationDensity',
    'TurbulentEnergyKineticDensity',
    'TurbulentDissipationRateDensity',
    'Pressure', 'Mach', 'Temperature',
    'Viscosity_EddyMolecularRatio',
    # 'q_criterion', # Not available in elsA v4.1.01
    ]

    EP._addFlowSolutionEoR(t, name='', variables=Variables2Extract, governingEquations='NSTurbulent', writingFrame='relative', addBCExtract=False, protocol='end')

    # Initiates a solution
    if CreateInitialSolution:
        print("Creating initial solution...")
        I._renameNode(t,'FlowSolution#Centers','FlowSolution#Overset')

        C._initVars(t,'centers:Density',Ro)
        C._initVars(t,'centers:MomentumX',RoU)
        C._initVars(t,'centers:MomentumY',RoV)
        C._initVars(t,'centers:MomentumZ',RoW)
        C._initVars(t,'centers:EnergyStagnationDensity',RoE)
        C._initVars(t,'centers:TurbulentEnergyKineticDensity',RoK)
        C._initVars(t,'centers:TurbulentDissipationRateDensity',RoO)
        I._renameNode(t,'FlowSolution#Centers','FlowSolution#Init')
        I._renameNode(t,'FlowSolution#Overset','FlowSolution#Centers')


# Final adjustments before saving
I._rmNodesByName(t,'source')
I._rmNodesByName(t,'loc2glob')


# ======================================================= #
# ============== SAVE FINAL .HDF ELSA FILE ============== #
# ======================================================= #


C.convertPyTree2File(t,'Case_Nprocs%d'%Nprocs+'.hdf')

toc = timeit.default_timer()

print ("ELAPSED TIME = %g minutes"%((toc-tic)/60.))
