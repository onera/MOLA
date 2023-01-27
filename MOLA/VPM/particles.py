import sys
import numpy as np
import Converter.PyTree as C
import Converter.Internal as I
import Generator.PyTree as G
import Transform.PyTree as T
import Geom.PyTree as D
from .. import LiftingLine as LL
from .. import InternalShortcuts as J
from .. import Wireframe as W
from . import user as VPM
from time import time, sleep

def addParticles(t, PolarsInterpolatorDict, IterationInfo = {}):
    timeLL = time()
    LiftingLines = LL.getLiftingLines(t)
    ShieldsBase = I.getNodeFromName2(t, 'ShieldsBase')
    ShieldBoxes = I.getZones(ShieldsBase)
    ParticlesBase = I.getNodeFromName2(t, 'ParticlesBase')
    Particles = VPM.pickParticlesZone(t)

    if not Particles: raise ValueError('"Particles" zone not found in ParticlesTree')

    solverParams = J.get(Particles, 'SolverParameters')

    LiftingLinesM1 = [I.copyTree(ll) for ll in LiftingLines]
    ratio = solverParams['SmoothingRatio'][0]
    h = solverParams['Resolution']
    Sigma0 = solverParams['Sigma0']
    dt = solverParams['TimeStep']
    U0 = solverParams['VelocityFreestream']
    MaskShedParticles = solverParams['ShedParticlesIndex']
    for LiftingLineM1 in LiftingLinesM1:
        x, y, z = J.getxyz(LiftingLineM1)
        ui, vi, wi = J.getVars(LiftingLineM1,['VelocityInduced'+i for i in 'XYZ'])
        x += dt*(U0[0] + ui)
        y += dt*(U0[1] + vi)
        z += dt*(U0[2] + wi)

    LL.computeKinematicVelocity(LiftingLinesM1)
    LL.moveLiftingLines(LiftingLines, dt)
    LL.assembleAndProjectVelocities(LiftingLines)
    #LL.moveObject(ShieldBoxes, dt)
    #VPM.maskParticlesInsideShieldBoxes(t, ShieldBoxes)

    AllAbscissaSegments, SigmaDistributionOnLiftingLine, SigmaDistribution = [], [], []
    for LiftingLine in LiftingLines:
        VPM_Parameters = J.get(LiftingLine,'.VPM#Parameters')
        AllAbscissaSegments += [VPM_Parameters['ParticleDistribution']]
        SigmaDistributionOnLiftingLine.extend(np.array(VPM_Parameters['SigmaDistributionOnLiftingLine'], order='F', dtype=np.float64))
        SigmaDistribution.extend(np.array(VPM_Parameters['SigmaDistribution'], order='F', dtype=np.float64))

    Sources = LL.buildVortexParticleSourcesOnLiftingLine(LiftingLines, AbscissaSegments=AllAbscissaSegments, IntegralLaw='linear')
    SourcesM1 = LL.buildVortexParticleSourcesOnLiftingLine(LiftingLinesM1, AbscissaSegments=AllAbscissaSegments, IntegralLaw='linear')
    SigmaDistributionOnLiftingLine = np.hstack(np.array(SigmaDistributionOnLiftingLine, dtype=np.float64))
    WakeInducedVelocity = VPM.getInducedVelocityFromWake(t, LiftingLines, SigmaDistributionOnLiftingLine)

    CoordinateX = I.getValue(I.getNodeFromName3(Particles, 'CoordinateX')); CoordinateY = I.getValue(I.getNodeFromName3(Particles, 'CoordinateY')); CoordinateZ = I.getValue(I.getNodeFromName3(Particles, 'CoordinateZ'))
    NumberParticlesShedPerStation, NewSigma = [], []
    NumberSource = 0
    for Source in Sources:
        SourceX = I.getValue(I.getNodeFromName3(Source, 'CoordinateX')); SourceY = I.getValue(I.getNodeFromName3(Source, 'CoordinateY')); SourceZ = I.getValue(I.getNodeFromName3(Source, 'CoordinateZ'))
        TrailingDistance = np.array([(((SourceX[i + 1] + SourceX[i])/2. - CoordinateX[MaskShedParticles[i + NumberSource]])**2 + \
                                      ((SourceY[i + 1] + SourceY[i])/2. - CoordinateY[MaskShedParticles[i + NumberSource]])**2 + \
                                      ((SourceZ[i + 1] + SourceZ[i])/2. - CoordinateZ[MaskShedParticles[i + NumberSource]])**2)**0.5 for i in range(len(SourceX) - 1)])
        NumberParticlesShedPerStation += [[max(round(dy/h[0] - 0.95), 0) for dy in TrailingDistance]]
        for i in range(len(SourceX) - 1): NewSigma += [SigmaDistribution[i + NumberSource]]*NumberParticlesShedPerStation[-1][i]

        NumberSource += len(SourceX) - 1

    NumberParticlesShedPerStation = np.hstack(np.array(NumberParticlesShedPerStation, dtype=np.int32, order = 'F'))
    NumberParticlesShed = np.sum(NumberParticlesShedPerStation)
    N0 = Particles[1][0][0] * 1

    VPM._roll(Particles, N0 - NumberSource)
    VPM._extend(Particles, NumberParticlesShed)


    Particles[1][0][0] = N0
    KinematicViscosity, Volume, Sigma = J.getVars(Particles, ['Nu', 'Volume', 'Sigma'])
    Sigma[N0 - NumberSource:] = SigmaDistribution + NewSigma
    GammaOld = [I.getNodeFromName3(Source, 'Gamma')[1] for Source in Sources]

    GammaThreshold = solverParams['CirculationThreshold']
    GammaRelax = solverParams['CirculationRelaxation']
    GammaError = GammaThreshold + 1.

    ni = 0
    for Ni in range(int(solverParams['MaxLiftingLineSubIterations'])):
        Particles[1][0][0] = N0
        #updateParticlesStrength(Particles, MaskShedParticles, Sources, SourcesM1, NumberParticlesShedPerStation, NumberSource)
        VPM.addParticlesFromLiftingLineSources(t, Sources, SourcesM1, NumberParticlesShedPerStation, NumberSource)
        VPM.computeInducedVelocityOnLiftinLines(t, NumberParticlesShed + NumberSource, LiftingLines, SigmaDistributionOnLiftingLine, WakeInducedVelocity)
        LL.assembleAndProjectVelocities(LiftingLines)
        LL._applyPolarOnLiftingLine(LiftingLines, PolarsInterpolatorDict, ['Cl', 'Cd'])
        IntegralLoads = LL.computeGeneralLoadsOfLiftingLine(LiftingLines)
        Sources = LL.buildVortexParticleSourcesOnLiftingLine(LiftingLines, AbscissaSegments=AllAbscissaSegments, IntegralLaw='linear')
        GammaError = _relaxCirculationAndGetImbalance(GammaOld, GammaRelax, Sources)

        ni += 1
        if GammaError < GammaThreshold: break


    Particles[1][0][0] = N0 + NumberParticlesShed
    LL.computeGeneralLoadsOfLiftingLine(LiftingLines,
            UnsteadyData={'IterationNumber':solverParams['CurrentIteration'],
                          'Time':solverParams['Time'],
                          'CirculationSubiterations':ni,
                          'CirculationError':GammaError},
                            UnsteadyDataIndependentAbscissa='IterationNumber')

    AlphaXYZ = np.vstack(J.getVars(Particles, ['Alpha'+i for i in 'XYZ']))
    AlphaNorm = np.linalg.norm(AlphaXYZ[:, N0 - NumberSource:],axis=0)
    StrengthMagnitude = J.getVars(Particles, ['StrengthMagnitude'])[0]
    StrengthMagnitude[N0 - NumberSource:] = AlphaNorm[:]

    Volume[N0 - NumberSource:] = 4./3.*np.pi*Sigma[N0 - NumberSource:]**3
    KinematicViscosity[N0 - NumberSource:] = solverParams['KinematicViscosity'][0] + \
        (Sigma0[0]*solverParams['EddyViscosityConstant'][0])**2*2**0.5*\
        AlphaNorm[:]/Volume[N0 - NumberSource:]

    VorticityX, VorticityY, VorticityZ = J.getVars(Particles, ['Vorticity'+i for i in 'XYZ'])
    VorticityX[N0 - NumberSource:] = AlphaXYZ[0,N0 - NumberSource:]/Volume[N0 - NumberSource:]
    VorticityY[N0 - NumberSource:] = AlphaXYZ[1,N0 - NumberSource:]/Volume[N0 - NumberSource:]
    VorticityZ[N0 - NumberSource:] = AlphaXYZ[2,N0 - NumberSource:]/Volume[N0 - NumberSource:]
    VorticityXYZ = np.vstack([VorticityX, VorticityY, VorticityZ])
    VorticityNorm = np.linalg.norm(VorticityXYZ[:, N0 - NumberSource:],axis=0)
    VorticityMagnitude = J.getVars(Particles, ['VorticityMagnitude'])[0]
    VorticityMagnitude[N0 - NumberSource:] = VorticityNorm[:]

    Ns = 0
    posNumberParticlesShed = 0
    for Source in Sources:
        SourceX = I.getValue(I.getNodeFromName(Source, 'CoordinateX'))
        for i in range(Ns, Ns + len(SourceX) - 1):
            if NumberParticlesShedPerStation[posNumberParticlesShed]:
                MaskShedParticles[i] = sum(NumberParticlesShedPerStation[:posNumberParticlesShed]) + NumberSource
            else: 
                MaskShedParticles[i] += NumberParticlesShed

            posNumberParticlesShed += 1
        Ns += len(SourceX) - 1

    VPM._roll(Particles, NumberSource + NumberParticlesShed)
    
    IterationInfo['Circulation Error'] = GammaError
    IterationInfo['Number of sub-iterations'] = ni
    IterationInfo['Number of shed particles'] = NumberParticlesShed
    IterationInfo['Lifting Line time'] = time() - timeLL
    return IterationInfo

def _initialiseLiftingLinesAndGetShieldBoxes(LiftingLines, PolarsInterpolatorDict, Resolution):
    LL.computeKinematicVelocity(LiftingLines)
    LL.assembleAndProjectVelocities(LiftingLines)
    LL._applyPolarOnLiftingLine(LiftingLines, PolarsInterpolatorDict, ['Cl', 'Cd'])
    LL.computeGeneralLoadsOfLiftingLine(LiftingLines)
    ShieldBoxes = _buildShieldBoxesAroundLiftingLines(LiftingLines, Resolution)

    return ShieldBoxes

def _buildShieldBoxesAroundLiftingLines(LiftingLines, Resolution):
    ShieldBoxes = []
    h = 2*Resolution
    for LiftingLine in I.getZones(LiftingLines):
        tx,ty,tz,bx,by,bz,nx,ny,nz = J.getVars(LiftingLine,
            ['tx','ty','tz','bx','by','bz','nx','ny','nz'])
        x,y,z = J.getxyz(LiftingLine)
        quads = []
        for i in range(len(tx)):
            quad = G.cart((-h/2.,-h/2.,0),(h,h,1),(2,2,1))
            T._rotate(quad,(0,0,0), ((1,0,0),(0,1,0),(0,0,1)),
                ((nx[i],ny[i],nz[i]), (bx[i],by[i],bz[i]), (tx[i],ty[i],tz[i])))
            T._translate(quad,(x[i],y[i],z[i]))
            quads += [ quad ]
        I._correctPyTree(quads, level=3)
        ShieldBox = G.stack(quads)
        ShieldBox[0] = LiftingLine[0] + '.shield'

        # in theory, this get+set is a copy by reference (e.g.: in-place
        # modification of RPM in LiftingLine will produce a modification of the
        # RPM of its associated ShieldBoxes)
        for paramsName in ['.Conditions','.Kinematics']:
            params = J.get(LiftingLine, paramsName)
            J.set(ShieldBox, paramsName, **params)
        ShieldBoxes += [ ShieldBox ]

    return ShieldBoxes

def _relaxCirculationAndGetImbalance(GammaOld , GammaRelax, Sources):
    GammaError = 0
    for i in range(len(Sources)):
        GammaNew, = J.getVars(Sources[i],['Gamma'])
        GammaError = max(GammaError, max(abs(GammaNew - GammaOld[i]))/max(1e-12,np.mean(abs(GammaNew))))
        GammaNew[:] = (1. - GammaRelax)*GammaOld[i] + GammaRelax*GammaNew
        GammaOld[i][:] = GammaNew
    return GammaError

def updateParticlesStrength(Particles, MaskShedParticles, Sources, SourcesM1, NumberParticlesShedPerStation, NumberSource):
    Np = Particles[1][0]
    CoordinateX       = I.getNodeFromName3(Particles, "CoordinateX")
    CoordinateY       = I.getNodeFromName3(Particles, "CoordinateY")
    CoordinateZ       = I.getNodeFromName3(Particles, "CoordinateZ")
    AlphaX            = I.getNodeFromName3(Particles, "AlphaX")
    AlphaY            = I.getNodeFromName3(Particles, "AlphaY")
    AlphaZ            = I.getNodeFromName3(Particles, "AlphaZ")

    Ns = 0
    posEmbedded = Np[0] - NumberSource
    for k in range(len(Sources)):
        LLXj                  = I.getValue(I.getNodeFromName3(Sources[k], "CoordinateX"))
        LLYj                  = I.getValue(I.getNodeFromName3(Sources[k], "CoordinateY"))
        LLZj                  = I.getValue(I.getNodeFromName3(Sources[k], "CoordinateZ"))
        Gamma                = I.getValue(I.getNodeFromName3(Sources[k], "Gamma"))
        V2DXj                 = I.getValue(I.getNodeFromName3(Sources[k], "Velocity2DX"))
        V2DYj                 = I.getValue(I.getNodeFromName3(Sources[k], "Velocity2DY"))
        V2DZj                 = I.getValue(I.getNodeFromName3(Sources[k], "Velocity2DZ"))
        GammaM1              = I.getValue(I.getNodeFromName3(SourcesM1[k], "Gamma"))

        NsCurrent = len(LLXj)

        for i in range(NsCurrent - 1):
            Nshed = NumberParticlesShedPerStation[Ns + i] + 1
            vecj2D = -np.array([V2DXj[i + 1] + V2DXj[i], V2DYj[i + 1] + V2DYj[i], V2DZj[i + 1] + V2DZj[i]])
            dy = np.linalg.norm(vecj2D)
            vecj2D /= dy

            veci = np.array([(LLXj[i + 1] - LLXj[i]), (LLYj[i + 1] - LLYj[i]), (LLZj[i + 1] - LLZj[i])])
            dx = np.linalg.norm(veci)
            veci /= dx

            pos = 0.5*np.array([LLXj[i + 1] + LLXj[i], LLYj[i + 1] + LLYj[i], LLZj[i + 1] + LLZj[i]])
            vecj = np.array([CoordinateX[1][MaskShedParticles[Ns + i] - NumberSource], CoordinateY[1][MaskShedParticles[Ns + i] - NumberSource], CoordinateZ[1][MaskShedParticles[Ns + i] - NumberSource]]) - pos
            dy = np.linalg.norm(vecj)
            vecj /= Nshed
            
            GammaTrailing = (Gamma[i + 1] - Gamma[i])*dy/Nshed
            GammaShedding = (GammaM1[i + 1] + GammaM1[i] - (Gamma[i + 1] + Gamma[i]))/2.*dx/Nshed
            GammaBound = [GammaTrailing*vecj2D[0] + GammaShedding*veci[0], GammaTrailing*vecj2D[1] + GammaShedding*veci[1], GammaTrailing*vecj2D[2] + GammaShedding*veci[2]]
            CoordinateX[1][posEmbedded] = pos[0]
            CoordinateY[1][posEmbedded] = pos[1]
            CoordinateZ[1][posEmbedded] = pos[2]
            AlphaX[1][posEmbedded] = GammaBound[0]
            AlphaY[1][posEmbedded] = GammaBound[1]
            AlphaZ[1][posEmbedded] = GammaBound[2]
            posEmbedded += 1
            for j in range(1, Nshed):
                CoordinateX[1][Np[0]] = pos[0] + (j + 0.)*vecj[0]
                CoordinateY[1][Np[0]] = pos[1] + (j + 0.)*vecj[1]
                CoordinateZ[1][Np[0]] = pos[2] + (j + 0.)*vecj[2]
                AlphaX[1][Np[0]] = GammaBound[0]
                AlphaY[1][Np[0]] = GammaBound[1]
                AlphaZ[1][Np[0]] = GammaBound[2]

                Np[0] += 1

        Ns += NsCurrent - 1
        '''
        for i in range(NsCurrent - 1):
            Nshed = NumberParticlesShedPerStation[Ns + i] + 1
            vecj2D = -dt*0.5*np.array([V2DXj[i + 1] + V2DXj[i], V2DYj[i + 1] + V2DYj[i], V2DZj[i + 1] + V2DZj[i]])
            dy = np.linalg.norm(vecj2D)
            vecj2D /= dy

            veci = np.array([(LLXj[i + 1] - LLXj[i]), (LLYj[i + 1] - LLYj[i]), (LLZj[i + 1] - LLZj[i])])
            dx = np.linalg.norm(veci)
            veci /= dx

            pos = 0.5*np.array([LLXj[i + 1] + LLXj[i], LLYj[i + 1] + LLYj[i], LLZj[i + 1] + LLZj[i]])
            vecj = np.array([CoordinateX[1][MaskShedParticles[Ns + i] - NumberSource], CoordinateY[1][MaskShedParticles[Ns + i] - NumberSource], CoordinateZ[1][MaskShedParticles[Ns + i] - NumberSource]]) - pos
            dy = np.linalg.norm(vecj)
            vecj /= dy
            
            GammaTrailing = (Gamma[i + 1] - Gamma[i])*dy
            GammaShedding = (GammaM1[i + 1] + GammaM1[i] - (Gamma[i + 1] + Gamma[i]))/2.*dx
            MeanWeightX = (GammaTrailing*vecj2D[0] + GammaShedding*veci[0] + AlphaX[1][MaskShedParticles[Ns + i] - NumberSource])/(Nshed + 1)
            slopeX = 2.*(AlphaX[1][MaskShedParticles[Ns + i] - NumberSource] - MeanWeightX)/dy
            heightX = 2.*MeanWeightX - AlphaX[1][MaskShedParticles[Ns + i] - NumberSource]
            MeanWeightY = (GammaTrailing*vecj2D[1] + GammaShedding*veci[1] + AlphaY[1][MaskShedParticles[Ns + i] - NumberSource])/(Nshed + 1)
            slopeY = 2.*(AlphaY[1][MaskShedParticles[Ns + i] - NumberSource] - MeanWeightY)/dy
            heightY = 2.*MeanWeightY - AlphaY[1][MaskShedParticles[Ns + i] - NumberSource]
            MeanWeightZ = (GammaTrailing*vecj2D[2] + GammaShedding*veci[2] + AlphaZ[1][MaskShedParticles[Ns + i] - NumberSource])/(Nshed + 1)
            slopeZ = 2.*(AlphaZ[1][MaskShedParticles[Ns + i] - NumberSource] - MeanWeightZ)/dy
            heightZ = 2.*MeanWeightZ - AlphaZ[1][MaskShedParticles[Ns + i] - NumberSource]
            CoordinateX[1][posEmbedded] = pos[0]
            CoordinateY[1][posEmbedded] = pos[1]
            CoordinateZ[1][posEmbedded] = pos[2]
            AlphaX[1][posEmbedded] = 0.*slopeX + heightX
            AlphaY[1][posEmbedded] = 0.*slopeY + heightY
            AlphaZ[1][posEmbedded] = 0.*slopeZ + heightZ
            posEmbedded += 1
            for j in range(1, Nshed):
                CoordinateX[1][Np[0]] = pos[0] + j*dy/Nshed*vecj[0]
                CoordinateY[1][Np[0]] = pos[1] + j*dy/Nshed*vecj[1]
                CoordinateZ[1][Np[0]] = pos[2] + j*dy/Nshed*vecj[2]
                AlphaX[1][Np[0]] = j*dy/Nshed*slopeX + heightX
                AlphaY[1][Np[0]] = j*dy/Nshed*slopeY + heightY
                AlphaZ[1][Np[0]] = j*dy/Nshed*slopeZ + heightZ
                Np[0] += 1
        Ns += NsCurrent - 1
        '''
