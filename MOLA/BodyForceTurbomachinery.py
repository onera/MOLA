'''
MOLA - BodyForceTurbomachinery.py

File history:
8/09/2022 - T. Bontemps - Creation
'''


import numpy as np

import Converter.PyTree as C
import Converter.Internal as I
import Geom.PyTree as D
import Transform.PyTree as T

import MOLA.InternalShortcuts as J


def addBodyForce(t, ReferenceValues):

    # Initialize Source terms
    I.__FlowSolutionCenters__ = 'FlowSolution#SourceTerm'
    for FieldName in ReferenceValues['Fields']:
        C._initVars(t, 'centers:%s' % FieldName, 0.)
    

    # Compute data source terms
    I.__FlowSolutionCenters__ = 'FlowSolution#DataSourceTerm'
    addDataSourceTerms_Hall(t)
    
    I.__FlowSolutionCenters__ = 'FlowSolution#Centers'

def addDataSourceTerms_Hall(t):
    variables = ['nx','nr','nt','xc','b','gradxb','gradrb','chord','chordx','delta0','isf']


def updateBodyForce(t, iteri=1., iterf=300., relax=0.):
    coeff_eff = J.rampFunction(iteri, iterf, 0., 1.)

    for zone in I.getZones(t):

        NewSourceTerms = computeBodyForce_Hall(zone)

        BLProtectionSourceTerms = computeProtectionSourceTerms(zone)
        for key, value in BLProtectionSourceTerms.items():
            NewSourceTerms[key] += value

        CurrentSourceTermNode = I.getNodeFromName1(zone, 'FlowSolution#SourceTerm')
        for name, newSourceTerm in NewSourceTerms.items():
            node = I.getNodeFromName1(CurrentSourceTermNode, name)
            currentSourceTerm = I.getValue(node) 
            currentSourceTerm = coeff_eff * ((1-relax) * newSourceTerm + relax * currentSourceTerm)


def computeBlockageSourceTerms(zone, tol=1e-5):

    FlowSolution    = J.getVars2Dict(zone, C.getVarNames(zone), Container='FlowSolution#Init')
    DataSourceTerms = J.getVars2Dict(zone, ['b', 'gradxb', 'gradrb', 'nt'], Container='FlowSolution#DataSourceTerm')

    radius, theta = J.getRadiusTheta(zone)    
    Blockage = DataSourceTerms['b']

    Density = np.maximum(FlowSolution['Density'], tol)
    EnthalpyStagnation = (FlowSolution['EnergyStagnationDensity'] + FlowSolution['Pressure']) / Density
    Vx, Vy, Vz = FlowSolution['MomentumX']/Density, FlowSolution['MomentumY'] / Density, FlowSolution['MomentumZ']/Density
    Vr = Vy * np.cos(theta) + Vz * np.sin(theta)

    Sb = -(Vx * DataSourceTerms['gradxb'] + Vr *DataSourceTerms['gradrb']) / Blockage

    NewSourceTerms = dict(
        Density          = Sb,
        MomentumX        = Sb * Vx,
        MomentumY        = Sb * Vy,
        MomentumZ        = Sb * Vz,
        EnergyStagnation = Sb * EnthalpyStagnation
    )

    return NewSourceTerms


def computeBodyForce_Hall(zone, FluidProperties, TurboConfiguration, tol=1e-5):

    rowName = I.getValue(I.getNodeFromType1(zone, 'FamilyName_t'))
    NumberOfBlades = TurboConfiguration['Rows'][rowName]['NumberOfBlades']
    RotationSpeed = TurboConfiguration['Rows'][rowName]['RotationSpeed']

    FlowSolution    = J.getVars2Dict(zone, C.getVarNames(zone), Container='FlowSolution#Init')
    DataSourceTerms = J.getVars2Dict(zone, C.getVarNames(zone), Container='FlowSolution#DataSourceTerm')

    # Coordinates
    radius, theta = J.getRadiusTheta(zone)
    cosTheta = np.cos(theta)
    sinTheta = np.sin(theta)

    Density = np.maximum(FlowSolution['Density'], tol)
    Vx, Vy, Vz = FlowSolution['MomentumX']/Density, FlowSolution['MomentumY']/Density, FlowSolution['MomentumZ']/Density
    Wx, Wr, Wt = Vx, Vy*cosTheta+Vz*sinTheta, -Vy*sinTheta+Vz*cosTheta-radius*RotationSpeed
    Vmag, Wmag = (Vx**2+Vy**2+Vz**2)**0.5, np.maximum(tol, (Wx**2+Wr**2+Wt**2)**0.5)
    Temperature = np.maximum(tol, (FlowSolution['EnergyStagnationDensity']/Density-0.5*Vmag**2.)/FluidProperties['cp'])
    Mrel = Wmag/(FluidProperties['Gamma']*FluidProperties['IdealGasConstant']*Temperature)**0.5

    # Vitesses normales et parallele au squelette
    wpn_bf = Wx*DataSourceTerms['nx']+Wr*DataSourceTerms['nr']+Wt*DataSourceTerms['nt']
    Wnx, Wnr, Wnt        = wpn_bf*DataSourceTerms['nx'], wpn_bf*DataSourceTerms['nr'], wpn_bf*DataSourceTerms['nt']
    Wpx, Wpr, Wpt = Wx-Wnx, Wr-Wnr, Wt-Wnt
    Wp = np.maximum(tol, (Wpx**2+Wpr**2+Wpt**2)**0.5)

    # Compressibility correction 
    CompressibilityCorrection = 3. * np.ones(Density.shape)
    subsonic_bf, supersonic_bf = np.less_equal(Mrel,0.99), np.greater_equal(Mrel,1.01)
    CompressibilityCorrection[subsonic_bf]  = np.clip(1.0/(1-Mrel[subsonic_bf]**2)**0.5, 0.0, 3.0)
    CompressibilityCorrection[supersonic_bf]= np.clip(4.0/(2*np.pi)/(Mrel[supersonic_bf]**2-1)**0.5, 0.0, 3.0)

    blade2BladeDistance = 2*np.pi*radius / NumberOfBlades * np.absolute(DataSourceTerms['nt']) * DataSourceTerms['b']
    incidence = np.arcsin(wpn_bf/Wmag)

    # Force normal to the chord
    fn = -0.5*Wmag**2. * CompressibilityCorrection * 2*np.pi*incidence / blade2BladeDistance * DataSourceTerms['isf']

    # Friction on blade
    Viscosity = FluidProperties['SutherlandViscosity']*np.sqrt(Temperature/FluidProperties['SutherlandTemperature'])*(1+FluidProperties['SutherlandConstant'])/(1+FluidProperties['SutherlandConstant']*FluidProperties['SutherlandTemperature']/Temperature)
    Re_x = Density*DataSourceTerms['xc']*DataSourceTerms['chordx'] * Wmag / Viscosity
    cf = 0.0592*Re_x**(-0.2)

    # Force parallel to the chord
    fp = -0.5*Wmag**2. * (2*cf + 2*np.pi*(incidence - DataSourceTerms['delta0'])**2.) / blade2BladeDistance * DataSourceTerms['isf']

    fx = fn*(np.cos(incidence)*DataSourceTerms['nx']-np.sin(incidence)*Wpx/Wp) + fp*Wx/Wmag
    fr = fn*(np.cos(incidence)*DataSourceTerms['nr']-np.sin(incidence)*Wpr/Wp) + fp*Wr/Wmag
    ft = fn*(np.cos(incidence)*DataSourceTerms['nt']-np.sin(incidence)*Wpt/Wp) + fp*Wt/Wmag
    fy = -sinTheta * ft + cosTheta * fr
    fz =  cosTheta * ft + sinTheta * fr

    NewSourceTerms = dict(
        Density          = 0.,
        MomentumX        = Density * fx,
        MomentumY        = Density * fy,
        MomentumZ        = Density * fz,
        EnergyStagnation = Density * radius * RotationSpeed * ft
    )

    # Add blockage terms
    BlockageSourceTerms = computeBlockageSourceTerms(zone)
    for key, value in BlockageSourceTerms.items():
        NewSourceTerms[key] += value

    return NewSourceTerms

def computeProtectionSourceTerms(zone):
    BLProtectionSourceTerms = dict()
    return BLProtectionSourceTerms