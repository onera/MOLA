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

import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator, Akima1DInterpolator, CubicSpline

import Converter.PyTree as C
import Converter.Internal as I

import MOLA.InternalShortcuts as J
import MOLA.Coprocess as CO
import MOLA.BodyForceTurbomachinery as BF
import MOLA.Postprocess as POST

def BodyForceModel_HallThollet(t, BodyForceParameters):
    NewSourceTermsGlobal = BodyForceModel_blockage(t, BodyForceParameters)
    BF.addDictionaries(NewSourceTermsGlobal, BodyForceModel_hall_without_blockage(t, BodyForceParameters))
    return NewSourceTermsGlobal


def BodyForceModel_blockage(t, BodyForceParameters):
    """
    Compute actualized source terms corresponding to blockage.

    Parameters
    ----------

        t : PyTree
            input tree (or zone, or list of zones) involved in source terms computations

        BodyForceParameters : dict
            Additional parameters for this body-force model.
            For that model, available parameters are:
              * tol (float, default=1e-5): minimum value for quantities used as a denominator.

    Returns
    -------

        NewSourceTermsGlobal : dict
            Computed source terms.
    """
    tol = BodyForceParameters.get('tol', 1e-05)

    NewSourceTermsGlobal = dict()
    for zone in I.getZones(t):

        FlowSolution = J.getVars2Dict(zone, Container='FlowSolution#Init')
        DataSourceTerms = J.getVars2Dict(zone, ['theta', 'blockage', 'gradxb', 'gradrb', 'nt'], Container='FlowSolution#DataSourceTerm')
        
        Blockage = DataSourceTerms['blockage']
        Density = np.maximum(FlowSolution['Density'], tol)
        EnthalpyStagnation = (FlowSolution['EnergyStagnationDensity'] + FlowSolution['Pressure']) / Density
        Vx, Vy, Vz = FlowSolution['MomentumX'] / Density, FlowSolution['MomentumY'] / Density, FlowSolution['MomentumZ'] / Density
        Vr = Vy * np.cos(DataSourceTerms['theta']) + Vz * np.sin(DataSourceTerms['theta'])
        Sb = -Density * (Vx * DataSourceTerms['gradxb'] + Vr * DataSourceTerms['gradrb']) / Blockage

        NewSourceTerms = dict(
            Density                 = Sb,
            MomentumX               = Sb * Vx,
            MomentumY               = Sb * Vy,
            MomentumZ               = Sb * Vz,
            EnergyStagnationDensity = Sb * EnthalpyStagnation
            )
        NewSourceTermsGlobal[I.getName(zone)] = NewSourceTerms
    
    return NewSourceTermsGlobal


def BodyForceModel_blockage_correction(t, BodyForceParameters):
    """
    Compute actualized source terms corresponding to blockage.

    Parameters
    ----------

        t : PyTree
            input tree (or zone, or list of zones) involved in source terms computations
        
        BodyForceParameters : dict
            Additional parameters for this body-force model.
            For that model, available parameters are:
              * tol (float, default=1e-5): minimum value for quantities used as a denominator.

    Returns
    -------

        NewSourceTermsGlobal : dict
            Computed source terms.
    """
    tol = BodyForceParameters.get('tol', 1e-05)

    for zone in I.getZones(t):
            
        FlowSolution = J.getVars2Dict(zone, Container='FlowSolution#Init')
        DataSourceTerms = J.getVars2Dict(zone, ['theta', 'blockage', 'gradxb', 'gradrb', 'nt'], Container='FlowSolution#DataSourceTerm')
        
        Blockage = DataSourceTerms['blockage']
        Density = np.maximum(FlowSolution['Density'], tol)
        EnthalpyStagnation = (FlowSolution['EnergyStagnationDensity'] + FlowSolution['Pressure']) / Density
        Vx, Vy, Vz = FlowSolution['MomentumX'] / Density, FlowSolution['MomentumY'] / Density, FlowSolution['MomentumZ'] / Density
        Vr = Vy * np.cos(DataSourceTerms['theta']) + Vz * np.sin(DataSourceTerms['theta'])
        Sb = -Density * (Vx * DataSourceTerms['gradxb'] + Vr * DataSourceTerms['gradrb']) / Blockage

        J.set(zone, 'FlowSolution#tmpBlockage', childType='FlowSolution_t', 
            Sb=Sb,
            Vx=Vx,
            Vy=Vy,
            Vz=Vz,
            EnthalpyStagnation=EnthalpyStagnation
            )
        tmpBlockageNode = I.getNodeFromName(zone, 'FlowSolution#tmpBlockage')
        I.createChild(tmpBlockageNode, 'GridLocation', 'GridLocation_t', value='CellCenter')
        
    averagedData = CO.volumicAverage(t, container='FlowSolution#tmpBlockage', localComm=BodyForceParameters['communicator'])
    NewSourceTermsGlobal = dict()
    for zone in I.getZones(t):
        FStmp = J.getVars2Dict(zone, Container='FlowSolution#tmpBlockage')
        Sb = FStmp['Sb'] - averagedData['Sb']
        NewSourceTerms = dict(
            Density                 = Sb,
            MomentumX               = Sb * FStmp['Vx'],
            MomentumY               = Sb * FStmp['Vy'],
            MomentumZ               = Sb * FStmp['Vz'],
            EnergyStagnationDensity = Sb * FStmp['EnthalpyStagnation']
            )
        NewSourceTermsGlobal[I.getName(zone)] = NewSourceTerms
    
    return NewSourceTermsGlobal


def BodyForceModel_hall_without_blockage(t, BodyForceParameters):
    r"""
    Compute actualized source terms corresponding to the Hall model.

    Parameters
    ----------

        t : PyTree
            input tree (or zone, or list of zones) involved in source terms computations

        BodyForceParameters : dict
            Additional parameters for this body-force model.
            For that model, available parameters are:

                * FluidProperties (dict): as read in `setup.py`

                * TurboConfiguration (dict): as read in `setup.py`

                * incidenceLoss (bool, default=False): apply or not the source term related to loss due to 
                  the deviation of the flow compared with the reference angle: 
                  :math:`2 \pi (\delta - \delta_0)^2 / H` 
                  where H is the blade to blade distance.

                * tol (float, default=1e-5): minimum value for quantities used as a denominator.

    Returns
    -------

        NewSourceTermsGlobal : dict
            Computed source terms.
    """
    FluidProperties = BodyForceParameters.get('FluidProperties')
    TurboConfiguration = BodyForceParameters.get('TurboConfiguration')
    incidenceLoss = BodyForceParameters.get('incidenceLoss', False)
    tol = BodyForceParameters.get('tol', 1e-05)

    NewSourceTermsGlobal = dict()
    for zone in I.getZones(t):

        rowName = I.getValue(I.getNodeFromType1(zone, 'FamilyName_t'))
        RotationSpeed = TurboConfiguration['Rows'][rowName]['RotationSpeed']
        FlowSolution = J.getVars2Dict(zone, Container='FlowSolution#Init')
        DataSourceTerms = J.getVars2Dict(zone, Container='FlowSolution#DataSourceTerm')
        tmpMOLAFlow = BF.getAdditionalFields(zone, FluidProperties, RotationSpeed, tol)

        # Compressibility correction 
        CompressibilityCorrection = 3.0 * np.ones(FlowSolution['Density'].shape)
        subsonic_bf, supersonic_bf = np.less_equal(tmpMOLAFlow['Mrel'], 0.99), np.greater_equal(tmpMOLAFlow['Mrel'], 1.01)
        CompressibilityCorrection[subsonic_bf] = np.clip(1.0 / (1 - tmpMOLAFlow['Mrel'][subsonic_bf] ** 2) ** 0.5, 0.0, 3.0)
        CompressibilityCorrection[supersonic_bf] = np.clip(4.0 / (2 * np.pi) / (tmpMOLAFlow['Mrel'][supersonic_bf] ** 2 - 1) ** 0.5, 0.0, 3.0)
        
        # Friction on blade
        Viscosity = FluidProperties['SutherlandViscosity'] * np.sqrt(tmpMOLAFlow['Temperature'] / FluidProperties['SutherlandTemperature']) * (1 + FluidProperties['SutherlandConstant']) / (1 + FluidProperties['SutherlandConstant'] * FluidProperties['SutherlandTemperature'] / tmpMOLAFlow['Temperature'])
        Re_x = FlowSolution['Density'] * DataSourceTerms['AbscissaFromLE'] * tmpMOLAFlow['Wmag'] / Viscosity
        cf = 0.0592 * Re_x ** (-0.2)
        
        # Force normal to the chord
        fn = -0.5 * tmpMOLAFlow['Wmag'] ** 2.0 * CompressibilityCorrection * 2 * np.pi * tmpMOLAFlow['incidence'] / DataSourceTerms['blade2BladeDistance']
        
        # Force parallel to the chord
        delta0 = DataSourceTerms.get('delta0', 0.0)
        fp = 0.5 * tmpMOLAFlow['Wmag'] ** 2.0 * (2 * cf + incidenceLoss * 2 * np.pi * (tmpMOLAFlow['incidence'] - delta0) ** 2.0) / DataSourceTerms['blade2BladeDistance']
        
        # Get force in the cartesian frame
        fx, fy, fz, fr, ft = BF.getForceComponents(fn, fp, tmpMOLAFlow)

        NewSourceTerms = dict(
            Density                 = np.zeros(FlowSolution['Density'].shape),
            MomentumX               = FlowSolution['Density'] * fx,
            MomentumY               = FlowSolution['Density'] * fy,
            MomentumZ               = FlowSolution['Density'] * fz,
            EnergyStagnationDensity = FlowSolution['Density'] * DataSourceTerms['radius'] * RotationSpeed * ft
            )
        
        NewSourceTermsGlobal[I.getName(zone)] = NewSourceTerms
    
    return NewSourceTermsGlobal


def BodyForceModel_EndWallsProtection(t, BodyForceParameters):
    ''' 
    Protection of the boudary layer ofr body-force modelling, as explain in the appendix D of 
    W. Thollet PdD manuscrit.

    Parameters
    ----------

        t : PyTree
            input tree (or zone, or list of zones) involved in source terms computations

        BodyForceParameters : dict
            Additional parameters for this body-force model.
            For that model, available parameters are:

                * FluidProperties (dict): as read in `setup.py`

                * TurboConfiguration (dict): as read in `setup.py`

                * ProtectedHeight (float, default=0.05): Height of the channel flow corresponding to the boundary layer. 

                * EndWallsCoefficient (float, default=10.): Multiplicative factor to apply on source terms. 

    Returns
    -------

        BLProtectionSourceTerms : dict
            Source terms to protect the boundary layer. The keys are Density (=0), MomentumX, 
            MomentumY, MomentumZ and EnergyStagnation (=0).
    
    '''
    TurboConfiguration = BodyForceParameters.get('TurboConfiguration')
    ProtectedHeight = BodyForceParameters.get('ProtectedHeight', 0.05)
    EndWallsCoefficient = BodyForceParameters.get('EndWallsCoefficient', 10.)

    NewSourceTermsGlobal = dict()
    for zone in I.getZones(t):

        zoneCopy = I.copyTree(zone)

        rowName = I.getValue(I.getNodeFromType1(zoneCopy, 'FamilyName_t'))
        RotationSpeed = TurboConfiguration['Rows'][rowName]['RotationSpeed']

        FlowSolution = J.getVars2Dict(zoneCopy, Container='FlowSolution#Init')
        DataSourceTerms = J.getVars2Dict(zoneCopy, Container='FlowSolution#DataSourceTerm')

        # Extract Boundary layers edges, based on ProtectedHeightPercentage
        I._renameNode(zoneCopy, 'FlowSolution#Init', 'FlowSolution#Centers')
        I._renameNode(zoneCopy, 'FlowSolution#Height', 'FlowSolution')
        BoundarayLayerEdgeAtHub    = POST.isoSurface(zoneCopy, 'ChannelHeight', value=ProtectedHeight, container='FlowSolution#Centers')
        BoundarayLayerEdgeAtShroud = POST.isoSurface(zoneCopy, 'ChannelHeight', value=1-ProtectedHeight, container='FlowSolution#Centers')
        C._node2Center__(zoneCopy, 'ChannelHeight')
        h, = J.getVars(zoneCopy, VariablesName=['ChannelHeight'], Container='FlowSolution#Centers')

        # Coordinates
        radius, theta = DataSourceTerms['radius'], DataSourceTerms['theta']
        cosTheta = np.cos(theta)
        sinTheta = np.sin(theta)
        # Flow data
        Vx = FlowSolution['MomentumX'] / FlowSolution['Density']
        Vy = FlowSolution['MomentumY'] / FlowSolution['Density']
        Vz = FlowSolution['MomentumZ'] / FlowSolution['Density']
        Wr =  Vy*cosTheta + Vz*sinTheta 
        Wt = -Vy*sinTheta + Vz*cosTheta - radius*RotationSpeed
        Wmag = (Vx**2+Wr**2+Wt**2)**0.5

        def get_mean_W_and_gradP(z):
            if not z: 
                return Wmag, 0, 0

            C._initVars(z, 'radius=({CoordinateY}**2+{CoordinateZ}**2)**0.5')
            C._initVars(z, 'theta=arctan2({CoordinateZ}, {CoordinateY})')
            C._initVars(z, 'Wx={MomentumX}/{Density}')
            C._initVars(z, 'Vy={MomentumY}/{Density}')
            C._initVars(z, 'Vz={MomentumZ}/{Density}')
            C._initVars(z, 'Wr={Vy}*cos({theta})+{Vz}*sin({theta})')
            C._initVars(z, 'Wt=-{{Vy}}*sin({{theta}})+{{Vz}}*cos({{theta}})-{{radius}}*{}'.format(RotationSpeed))
            C._initVars(z, 'Wmag=({Wx}**2+{Wr}**2+{Wt}**2)**0.5')
            C._initVars(z, 'DpDr=cos({theta})*{DpDy}+sin({theta})*{DpDz}')

            meanWmag = C.getMeanValue(z, 'Wmag')
            meanDpDx = C.getMeanValue(z, 'DpDx')
            meanDpDr = C.getMeanValue(z, 'DpDr')
            return meanWmag, meanDpDx, meanDpDr

        W_HubEdge, DpDx_HubEdge, DpDr_HubEdge = get_mean_W_and_gradP(BoundarayLayerEdgeAtHub)
        W_ShroudEdge, DpDx_ShroudEdge, DpDr_ShroudEdge = get_mean_W_and_gradP(BoundarayLayerEdgeAtShroud)

        zeros = np.zeros(Wmag.shape)
        # Source terms
        S_BL_Hub_x    = np.minimum(zeros, (1. - (Wmag /    W_HubEdge)**2)) * EndWallsCoefficient * DpDx_HubEdge
        S_BL_Hub_r    = np.minimum(zeros, (1. - (Wmag /    W_HubEdge)**2)) * EndWallsCoefficient * DpDr_HubEdge
        S_BL_Shroud_x = np.minimum(zeros, (1. - (Wmag / W_ShroudEdge)**2)) * EndWallsCoefficient * DpDx_ShroudEdge
        S_BL_Shroud_r = np.minimum(zeros, (1. - (Wmag / W_ShroudEdge)**2)) * EndWallsCoefficient * DpDr_ShroudEdge

        # Terms applied only in the BL protected area
        S_BL_x = S_BL_Hub_x * (h<ProtectedHeight) + S_BL_Shroud_x * (h>1-ProtectedHeight)
        S_BL_r = S_BL_Hub_r * (h<ProtectedHeight) + S_BL_Shroud_r * (h>1-ProtectedHeight)

        BLProtectionSourceTerms = dict(
            Density                 = zeros,
            MomentumX               = S_BL_x,
            MomentumY               = cosTheta * S_BL_r,
            MomentumZ               = sinTheta * S_BL_r,
            EnergyStagnationDensity = zeros
        )
        NewSourceTermsGlobal[I.getName(zone)] = BLProtectionSourceTerms

    return NewSourceTermsGlobal


def BodyForceModel_ThrustSpread(t, BodyForceParameters):
    """
    Compute actualized source terms corresponding to the Tspread model.

    Parameters
    ----------

        t : PyTree
            input tree (or zone, or list of zones) involved in source terms computations

        BodyForceParameters : dict
            Additional parameters for this body-force model.
            For that model, available parameters are:
              * tol (float, default=1e-5): minimum value for quantities used as a denominator.
              * Thust (float): Value of the total thrust to spread on the control volume.

    Returns
    -------

        NewSourceTermsGlobal : dict
            Computed source terms.
    """
    tol = BodyForceParameters.get('tol', 1e-05)

    NewSourceTermsGlobal = dict()
    for zone in I.getZones(t):

        FlowSolution = J.getVars2Dict(zone, Container='FlowSolution#Init')
        DataSourceTerms = J.getVars2Dict(zone, Container='FlowSolution#DataSourceTerm')

        Density = np.maximum(FlowSolution['Density'], tol)
        Vx, Vy, Vz = FlowSolution['MomentumX'] / Density, FlowSolution['MomentumY'] / Density, FlowSolution['MomentumZ'] / Density
        Vmag = np.maximum(tol, (Vx ** 2 + Vy ** 2 + Vz ** 2) ** 0.5)
        f = BodyForceParameters['Thrust'] / DataSourceTerms['totalVolume']

        NewSourceTerms = dict(
            Density                 = np.zeros(Vmag.shape),
            MomentumX               = Density * f * Vx / Vmag,
            MomentumY               = Density * f * Vy / Vmag,
            MomentumZ               = Density * f * Vz / Vmag,
            EnergyStagnationDensity = Density * f * Vmag
        )
        NewSourceTermsGlobal[I.getName(zone)] = NewSourceTerms
    
    return NewSourceTermsGlobal


def BodyForceModel_constant(t, BodyForceParameters):
    """
    Compute constant source terms (reshape given source terms if they are uniform).
        
        SourceTerms : dict
            Source terms to apply. For each value, the given value may be a 
            float or a numpy array (the shape must correspond to the zone shape).

    Parameters
    ----------

        t : PyTree
            input tree (or zone, or list of zones) involved in source terms computations

        BodyForceParameters : dict
            Additional parameters for this body-force model.
            For that model, available parameters are the source terms to apply. 
            For each value, the given value may be a float or a numpy array (the shape must 
            correspond to the zone shape).

    Returns
    -------

        NewSourceTermsGlobal : dict
            Computed source terms.
    """
    ReferenceValues = BodyForceParameters.get('ReferenceValues')

    SourceTerms = dict()
    for key in ReferenceValues['Fields']:

        if key in BodyForceParameters:
            SourceTerms[key] = BodyForceParameters[key]

        NewSourceTermsGlobal = dict()
        for zone in I.getZones(t):

            FlowSolution = J.getVars2Dict(zone, Container='FlowSolution#Init')
            ones = np.ones(FlowSolution['Density'].shape)

            if 'MomentumTheta' in BodyForceParameters or 'MomentumR' in BodyForceParameters:
                DataSourceTerms = J.getVars2Dict(zone, Container='FlowSolution#DataSourceTerm')
                theta = DataSourceTerms['theta']
                BodyForceParameters.setdefault('MomentumTheta', 0.)
                BodyForceParameters.setdefault('MomentumR', 0.)
                SourceTerms['MomentumY'] = np.cos(theta) * BodyForceParameters['MomentumTheta'] + np.sin(theta) * BodyForceParameters['MomentumR']
                SourceTerms['MomentumZ'] =-np.cos(theta) * BodyForceParameters['MomentumR'] + np.sin(theta) * BodyForceParameters['MomentumTheta']

            NewSourceTerms = dict()
            for key, value in SourceTerms.items():
                if isinstance(value, (float, int)):
                    NewSourceTerms[key] = value * ones
                else:
                    NewSourceTerms[key] = value
            
            NewSourceTermsGlobal[I.getName(zone)] = NewSourceTerms

    return NewSourceTermsGlobal


def BodyForceModel_ShockWaveLoss(t, BodyForceParameters):
    """
    Compute the volumic force parallel to the flow (and in the opposite direction) corresponding 
    to shock wave loss. 

    .. danger::
        
        WORK IN PROGRESS, DO NOT USE THIS FUNCTION !

    .. note::
        
        See the following reference for details on equations:

        Pazireh and Defoe, 
        A New Loss Generation Body Force Model for Fan/Compressor Blade Rows: Application to Uniform 
        and Non-Uniform Inflow in Rotor 67,
        Journal of Turbomachinery (2022)

    Parameters
    ----------

        t : PyTree
            input tree (or zone, or list of zones) involved in source terms computations


        BodyForceParameters : dict
            Additional parameters for this body-force model.
            For that model, available parameters are:

                * FluidProperties (dict): as read in `setup.py`

                * TurboConfiguration (dict): as read in `setup.py`

                * tol (float, default=1e-5): minimum value for quantities used as a denominator.

    Returns
    -------

        NewSourceTermsGlobal : dict
            Computed source terms.
    """
    FluidProperties = BodyForceParameters.get('FluidProperties')
    TurboConfiguration = BodyForceParameters.get('TurboConfiguration')
    tol = BodyForceParameters.get('tol', 1e-05)

    cv = FluidProperties['cv']
    R = FluidProperties['IdealGasConstant']
    gamma = FluidProperties['Gamma']

    FlowSolutionAtLE = BF.getFieldsAtLeadingEdge(t)

    NewSourceTermsGlobal = dict()
    for zone in I.getZones(t):

        rowName = I.getValue(I.getNodeFromType1(zone, 'FamilyName_t'))
        RotationSpeed = TurboConfiguration['Rows'][rowName]['RotationSpeed']

        Density, = J.getVars(zone, ['Density'], Container='FlowSolution#Init')
        DataSourceTerms = J.getVars2Dict(zone, Container='FlowSolution#DataSourceTerm')
        tmpMOLAFlow = BF.getAdditionalFields(zone, FluidProperties, RotationSpeed, tol)

        Ptrel = FlowSolutionAtLE['Pressure'] * (1 + (gamma - 1) / 2.0 * tmpMOLAFlow['Mrel'] ** 2) ** (gamma / (gamma - 1))
        PressureStagnationLossRatio = cv / R * 2 * gamma * (gamma - 1) / 3.0 / (gamma + 1) ** 2 * (tmpMOLAFlow['Mrel'] ** 2 - 1) ** 3
        
        fp = Ptrel * PressureStagnationLossRatio / DataSourceTerms['blade2BladeDistance']
        fx, fy, fz, fr, ft = BF.getForceComponents(0.0, fp, tmpMOLAFlow)

        NewSourceTerms = dict(
            Density                 = np.zeros(Density.shape),
            MomentumX               = Density * fx,
            MomentumY               = Density * fy,
            MomentumZ               = Density * fz,
            EnergyStagnationDensity = Density * DataSourceTerms['radius'] * RotationSpeed * ft
        )
        NewSourceTermsGlobal[I.getName(zone)] = NewSourceTerms
    
    return NewSourceTermsGlobal


def spreadPressureLossAlongChord(t, BodyForceParameters):
    '''
    Warning: Careful to the definition of PressureLossCoefficient !!
            Denominator is Pt-Ps or 0.5*rho*W**2  ??
    '''

    from ..Coprocess import printCo, rank

    FluidProperties = BodyForceParameters.get('FluidProperties')
    TurboConfiguration = BodyForceParameters.get('TurboConfiguration')

    PressureLoss = BodyForceParameters.get('PressureLoss')
    PressureLossCoefficient = BodyForceParameters.get('PressureLossCoefficient')
    Distribution = BodyForceParameters.get('Distribution', 'uniform')

    for zone in I.getZones(t):
        rowName = I.getValue(I.getNodeFromType1(zone, 'FamilyName_t'))
        RotationSpeed = TurboConfiguration['Rows'][rowName]['RotationSpeed']
        tmpMOLAFlow = BF.getAdditionalFields(zone, FluidProperties, RotationSpeed)

    # Extract Leading edge revolution surface, with flow quantities
    LeadingEdgeSurface = BF.getFieldsAtLeadingEdge(t)

    # Create interpolators based on their values at the leading edge
    variablesToInterp = ['Temperature', 'PressureStagnationRel']
    if PressureLoss is None:
        if PressureLossCoefficient is not None or 'PressureLossCoefficient' in tmpMOLAFlow:
            # In this case, the dynamic pressure will be needed in 2D
            # --> Add 'PressureDynamicRel' to the list of interpolated values
            variablesToInterp.append('PressureDynamicRel')
    interpDict = BF.getInterpolatorsInHeightAndTheta(LeadingEdgeSurface, variablesToInterp)

    NewSourceTermsGlobal = dict()
    for zone in I.getZones(t):

        rowName = I.getValue(I.getNodeFromType1(zone, 'FamilyName_t'))
        RotationSpeed = TurboConfiguration['Rows'][rowName]['RotationSpeed']
        FlowSolution = J.getVars2Dict(zone, Container='FlowSolution#Init')
        DataSourceTerms = J.getVars2Dict(zone, Container='FlowSolution#DataSourceTerm')
        tmpMOLAFlow = BF.getAdditionalFields(zone, FluidProperties, RotationSpeed)

        Temperature_LE = interpDict['Temperature'](DataSourceTerms['ChannelHeight'], DataSourceTerms['theta'])
        Ptel_LE = interpDict['PressureStagnationRel'](DataSourceTerms['ChannelHeight'], DataSourceTerms['theta'])

        # # Extract value at LE (to be replaced)
        # Temperature_LE    = FlowSolution['Temperature'][0,:,:]
        # Ptel_LE = FlowSolution['PressureStagnation'][0,:,:]
        # # stack this 2D field into a 3d field (uniform in x)
        # Temperature_LE_old = np.broadcast_to(Temperature_LE, FlowSolution['Temperature'].shape)
        # Ptel_LE_old = np.broadcast_to(Ptel_LE, FlowSolution['Temperature'].shape)

        # delta = I.copyTree(zone)
        # FS = I.newFlowSolution('delta', 'CellCenter', parent=delta)
        # I.newDataArray('delta_Temperature', value=Temperature_LE - Temperature_LE_old, parent=FS)
        # I.newDataArray('delta_PressureStagnationRel', value=Ptel_LE - Ptel_LE_old, parent=FS)

        # J.save(delta, f'error_on_zone_{zone[0]}.cgns')
        
        if PressureLoss is None:
            if PressureLossCoefficient is not None:
                PressureDynamicRel_LE = interpDict['PressureDynamicRel'](DataSourceTerms['ChannelHeight'], DataSourceTerms['theta'])
                PressureLoss = PressureDynamicRel_LE * PressureLossCoefficient
            else:
                if 'PressureLoss' in tmpMOLAFlow:
                    PressureLoss = tmpMOLAFlow['PressureLoss']
                elif 'PressureLossCoefficient' in tmpMOLAFlow:
                    PressureDynamicRel_LE = interpDict['PressureDynamicRel'](DataSourceTerms['ChannelHeight'], DataSourceTerms['theta'])
                    PressureLoss = PressureDynamicRel_LE * tmpMOLAFlow['PressureLossCoefficient']
                else:
                    raise Exception('Either PressureLoss or PressureLossCoefficient must be provided')
                
        if Distribution == 'uniform':
            distributionFunction = 1 / DataSourceTerms['ChordX']
        else:
            raise ValueError(f"Distribution='{Distribution}' is not implemented. Only 'uniform' is available.")   

        gradPt = PressureLoss * distributionFunction
        Wm = ( tmpMOLAFlow['Wx']**2 + tmpMOLAFlow['Wr']**2 )**0.5
        beta = np.arctan2(tmpMOLAFlow['Wt'], Wm)  
        # TODO Add the term on Temperature Stagnation for rotors
        fp = FluidProperties['IdealGasConstant'] * Temperature_LE / Ptel_LE * np.cos(beta) * gradPt

        # printCo(f'mean fp = {np.mean(fp)}', proc=rank)

        # Get force in the cartesian frame
        fx, fy, fz, fr, ft = BF.getForceComponents(0., fp, tmpMOLAFlow)

        NewSourceTerms = dict(
            Density                 = np.zeros(FlowSolution['Density'].shape),
            MomentumX               = FlowSolution['Density'] * fx,
            MomentumY               = FlowSolution['Density'] * fy,
            MomentumZ               = FlowSolution['Density'] * fz,
            EnergyStagnationDensity = FlowSolution['Density'] * DataSourceTerms['radius'] * RotationSpeed * ft
            )
        
        NewSourceTermsGlobal[I.getName(zone)] = NewSourceTerms
    
    return NewSourceTermsGlobal

def PtLossModel_Roberts1988_RotorTipClearance(r, rShroud, delta1, TipClearance, BladeSpan, alpha=0.01):
    '''
    Loss model (in total pressure) for endwall losses due to the rotor tip clearence.

    From: Roberts, Serovy and Sandercock, 1988, "Design Point Variation of Three-Dimensional Loss and Deviation for Axial Compressor Middle Stages"

    See equations (4), (5), (6) in this paper.

    Parameters
    ----------
        r : np.ndarray
            Radius values, in the current zone of interest for instance.

        rShroud : float
            Radius at Shroud (tip)

        delta1 : np.ndarray
            Displacement thickness normalized by blade span

        TipClearance : float
            Tip clearance at rotor tip, normalized by blade span

        BladeSpan : float
            Equal to the radius at shroud minus the radius at hub.

        alpha : float, optional
            Floor value to define the extent of the gaussian function, by default 0.01

    Returns
    -------
        np.ndarray
            Total pressure loss, with the same shape as the input radius **r**
    '''
    # We define a gaussian function of radius with its maximum value, its location and its width at alpha percents of the maximum
    PtLossMaximum  = 0.25 * np.tanh(np.sqrt(delta1*TipClearance*1e3))   # eq (4) in the reference paper
    PtLossDistanceOfMaxFromWall = 0.125 * np.tanh(np.sqrt(delta1*TipClearance*1e3)) * BladeSpan  # eq (5) in the reference paper
    PtLossExtent   = 2.5 * PtLossDistanceOfMaxFromWall  # eq (6) in the reference paper
    rmaxLoss = rShroud - PtLossDistanceOfMaxFromWall
    sigma = PtLossExtent**2 / (-4*np.log(alpha))
    PtLossDistrib = PtLossMaximum * np.exp( -(r-rmaxLoss)**2 / sigma )
    return PtLossDistrib

def PtLossModel_Roberts1988_EndWallWithoutClearance(r, rWall, WallType, delta1, Camber, ARc, Solidity, BladeSpan, alpha=0.01):
    '''
    Loss model (in total pressure) for endwall losses at rotor hub or stator endwalls.

    From: Roberts, Serovy and Sandercock, 1988, "Design Point Variation of Three-Dimensional Loss and Deviation for Axial Compressor Middle Stages"

    See equations (8), (9) and equation on Fig. 12 in this paper. Be careful, equation (7) seems to be wrong, prefer Fig. 12.

    Parameters
    ----------
        r : np.ndarray
            Radius values, in the current zone of interest for instance.

        rWall : float
            Radius at the endwall (hub or shroud, depending on **WallType**)
        
        WallType : str
            Must be 'hub' or 'shroud'.

        delta1 : np.ndarray
            Displacement thickness normalized by blade span

        Camber : float
            Camber of the blade in degrees (noted phi in the reference paper)
        
        ARc : float
            Channel Aspect Ratio = blade span / blade spacing at mid span
        
        Solidity : float
            Solidity = blade chord / blade spacing (noted sigma in the reference paper)

        BladeSpan : float
            Equal to the radius at shroud minus the radius at hub.

        alpha : float, optional
            Floor value to define the extent of the gaussian function, by default 0.01

    Returns
    -------
        np.ndarray
            Total pressure loss, with the same shape as the input radius **r**
    '''
    PtLossMaximum  = 0.20 * np.tanh( np.sqrt( 15 * abs(Camber) * delta1**2 / (ARc * Solidity) ) )  # error in eq (7) in the reference paper, see eq on Fig. 12
    PtLossDistanceOfMaxFromWall = 0.1 * BladeSpan  # eq (8) in the reference paper
    PtLossExtent   = 2.5 * PtLossDistanceOfMaxFromWall  # eq (9) in the reference paper

    if WallType.lower() == 'hub':
        rmaxLoss = rWall + PtLossDistanceOfMaxFromWall
    elif WallType.lower() == 'shroud':
        rmaxLoss = rWall - PtLossDistanceOfMaxFromWall
    else:
        raise ValueError('WallType must be hub or shroud')
    
    sigma = PtLossExtent**2 / (-4*np.log(alpha))
    PtLossDistrib = PtLossMaximum * np.exp( -(r-rmaxLoss)**2 / sigma )
    return PtLossDistrib

def DeviationModel_Roberts1988_StatorEndWalls(r, rHub, rShroud, delta1AtHub, delta1AtShroud, CamberAtHub, CamberAtShroud, SolidityAtHub, SolidityAtShroud, ARc, BladeSpan):
    '''
    Deviation model based on correlations at stator endwalls.

    From: 
        [1] Roberts, Serovy and Sandercock, 1988, "Design Point Variation of Three-Dimensional Loss and Deviation for Axial Compressor Middle Stages"
        [2] Roberts, Serovy and Sandercock, 1986, "Modeling the 3-D Flow effects on Deviation angle for Axial Compressor Middle Stages"

    Parameters
    ----------
        r : np.ndarray
            Radius values, in the current zone of interest for instance.

        rHub : float
            Radius at the hub
        
        rShroud : float
            Radius at the shroud

        delta1AtHub : np.ndarray
            Displacement thickness at hub normalized by blade span
        
        delta1AtShroud : np.ndarray
            Displacement thickness at shroud normalized by blade span

        CamberAtHub : float
            Camber of the blade at hub in degrees (noted phi in the reference paper)
        
        CamberAtShroud : float
            Camber of the blade at shroud in degrees (noted phi in the reference paper)
        
        SolidityAtHub : float
            Solidity at hub = blade chord / blade spacing (noted sigma in the reference paper)

        SolidityAtShroud : float
            Solidity at shroud = blade chord / blade spacing (noted sigma in the reference paper)
                
        ARc : float
            Channel Aspect Ratio = blade span / blade spacing at mid span

        BladeSpan : float
            Equal to the radius at shroud minus the radius at hub.

    Returns
    -------
        np.ndarray
            Underturning with respect to midspan flow angle, with the same shape as the input radius **r**
    '''
    # Same equations at hub and shroud, but different inputs
    # At hub
    arg1 = 0.5 * CamberAtHub * delta1AtHub**2 / (ARc*SolidityAtHub) * 1e3  
    UnderturningMaximumNearHub = 15 * np.tanh( arg1 )  # eq (1-Rob.,'86) in [1]
    rmaxNearHub = rHub + 0.125 * BladeSpan  # eq (2) in [2]
    UnderturningMaximumMinusAtHubWall = 20 * np.tanh( arg1 )  # eq (2-Rob.,'86) in [1]

    # At shroud
    arg2 = 0.5 * CamberAtShroud * delta1AtShroud**2 / (ARc*SolidityAtShroud) * 1e3  
    UnderturningMaximumNearShroud = 15 * np.tanh( arg2 )  # eq (1-Rob.,'86) in [1]
    rmaxNearShroud = rHub + 0.875 * BladeSpan  # eq (2) in [2]
    UnderturningMaximumMinusAtShroudWall = 20 * np.tanh( arg2 )  # eq (2-Rob.,'86) in [1]

    # # Same equations at hub and shroud, but different inputs
    # # At hub
    # UnderturningMaximumNearHub = 420 * (CamberAtHub * delta1AtHub**2 / (ARc*SolidityAtHub))**0.75  # eq (1) in [1]
    # rmaxNearHub = rHub + 0.125 * BladeSpan  # eq (2) in [2]
    # UnderturningMaximumMinusAtHubWall = 570 * (CamberAtHub * delta1AtHub**2 / (ARc*SolidityAtHub))**0.75  # eq (3) in [1]

    # # At shroud
    # UnderturningMaximumNearShroud = 420 * (CamberAtShroud * delta1AtShroud**2 / (ARc*SolidityAtShroud))**0.75  # eq (1) in [1]
    # rmaxNearShroud = rHub + 0.875 * BladeSpan  # eq (2) in [2]
    # UnderturningMaximumMinusAtShroudWall = 570 * (CamberAtShroud * delta1AtShroud**2 / (ARc*SolidityAtShroud))**0.75  # eq (3) in [1]

    Underturning = np.zeros(r.shape)
    rmidspan = (rHub + rShroud) / 2
    # Near hub
    # Between mid span and max underturning location
    where = (rmaxNearHub < r) & (r < rmidspan)
    Underturning[where] = UnderturningMaximumNearHub * np.exp( -(6*(r[where]-rmaxNearHub)/BladeSpan)**2 )
    # Between max underturning location and the endwall
    where = r < rmaxNearHub
    Underturning[where] = UnderturningMaximumNearHub - UnderturningMaximumMinusAtHubWall * np.exp( -(20*(r[where]-rHub)/BladeSpan)**2 )
    
    # Near shroud
    # Between mid span and max underturning location
    where = (rmidspan < r) & (r < rmaxNearShroud)
    Underturning[where] = UnderturningMaximumNearShroud * np.exp( -(6*(r[where]-rmaxNearShroud)/BladeSpan)**2 )
    # Between max underturning location and the endwall
    where = rmaxNearShroud < r
    Underturning[where] = UnderturningMaximumNearShroud - UnderturningMaximumMinusAtShroudWall * np.exp( -(20*(r[where]-rShroud)/BladeSpan)**2 )

    return Underturning

def DeviationModel_Roberts1988_RotorEndWalls(r, rHub, rShroud, delta1AtShroud, TipClearance, BladeSpan):
    '''
    Deviation model based on correlations at rotor endwalls.

    From: 
        [1] Roberts, Serovy and Sandercock, 1988, "Design Point Variation of Three-Dimensional Loss and Deviation for Axial Compressor Middle Stages"
        [2] Roberts, Serovy and Sandercock, 1986, "Modeling the 3-D Flow effects on Deviation angle for Axial Compressor Middle Stages"

    Parameters
    ----------
        r : np.ndarray
            Radius values, in the current zone of interest for instance.

        rHub : float
            Radius at the hub
        
        rShroud : float
            Radius at the shroud
        
        delta1AtShroud : np.ndarray
            Displacement thickness at shroud normalized by blade span

        TipClearance : float
            Tip clearance of the rotor normalized by blade span.

        BladeSpan : float
            Equal to the radius at shroud minus the radius at hub.

    Returns
    -------
        np.ndarray
            Underturning with respect to midspan flow angle, with the same shape as the input radius **r**
    '''
    # Only for rotor
    UnderturningAtTip = 30 * np.tanh( delta1AtShroud * TipClearance * 1e3 )  # eq (4-Rob.,'86) in [1]
    UnderturningAtTipLocation = rShroud - 0.05 * BladeSpan   # eq (2) in [2]
    OverturningAtTip = -1.25 # deg
    OverturningAtTipLocation = rShroud - 0.15 * BladeSpan
    OverturningAtHub = -2.0 # deg
    OverturningAtHubLocation = rHub + 0.05 * BladeSpan
    StartOverturningAtHubLocation = rHub + 0.15 * BladeSpan

    Underturning = np.zeros(r.shape)
    rmidspan = (rHub + rShroud) / 2
    # # Between tip and max underturning location
    # f = Akima1DInterpolator([OverturningAtTipLocation, UnderturningAtTipLocation], [OverturningAtTip, UnderturningAtTip])
    # where = OverturningAtTipLocation < r
    # Underturning[where] = f(r[where]) # smooth, through UnderturningAtTip
    # # Between max underturning location and mid span
    # where = (rmidspan < r) & (r < UnderturningAtTipLocation)
    # Underturning[where] = OverturningAtTip * np.exp( -(6*(r[where]-OverturningAtTipLocation)/BladeSpan)**2 )
    # # Between midspan and StartOverturningAtHubLocation
    # # Underturning stays null
    # # Between StartOverturningAtHubLocation to the hub
    # # Underturning[r<StartOverturningAtHubLocation] = ... # smooth curve. Quasi-linear ? What is the value at the hub ? 

    deviationInterpolator = PchipInterpolator(
        [OverturningAtHubLocation, StartOverturningAtHubLocation, rmidspan, OverturningAtTipLocation, UnderturningAtTipLocation], 
        [OverturningAtHub        ,                             0,        0, OverturningAtTip        , UnderturningAtTip        ]
        )
    Underturning = deviationInterpolator(r)
    
    return Underturning

def BodyForceModel_Roberts1988(t, BodyForceParameters):
    r"""
    Compute source terms corresponding to end wall lossses and flow deviation.

    From: 
        [1] Roberts, Serovy and Sandercock, 1988, "Design Point Variation of Three-Dimensional Loss and Deviation for Axial Compressor Middle Stages"
        [2] Roberts, Serovy and Sandercock, 1986, "Modeling the 3-D Flow effects on Deviation angle for Axial Compressor Middle Stages"

    Parameters
    ----------

        t : PyTree
            input tree (or zone, or list of zones) involved in source terms computations

        BodyForceParameters : dict
            Additional parameters for this body-force model.

    Returns
    -------

        NewSourceTermsGlobal : dict
            Computed source terms.
    """
    rShroud = BodyForceParameters['ShroudRadius']
    rHub = BodyForceParameters['HubRadius']
    BladeSpan = rShroud - rHub
    ChannelAspectRatio = BodyForceParameters['ChannelAspectRatio'] # blade span / blade spacing at mid span
    Solidity = BodyForceParameters['Solidity'] # chord / blade spacing
    Camber = BodyForceParameters['Camber'] # In degrees

    TurboConfiguration = BodyForceParameters.get('TurboConfiguration')

    NewSourceTermsGlobal = dict()
    for zone in I.getZones(t):

        rowName = I.getValue(I.getNodeFromType1(zone, 'FamilyName_t'))
        RotationSpeed = TurboConfiguration['Rows'][rowName]['RotationSpeed']

        DataSourceTerms = J.getVars2Dict(zone, Container='FlowSolution#DataSourceTerm')
        r = DataSourceTerms['radius']

        #####################################################################
        # For Hub

        # 1. Extract the BC
        # 2. Get the displacement thickness
        # Maybe first just take an average and broadcast it in zones
        DisplacementThickness = ...
        DisplacementThicknessAdim = DisplacementThickness / BladeSpan

        # At hub        
        PtLossAtHub = PtLossModel_Roberts1988_EndWallWithoutClearance(r, rHub, 'Hub', HubDisplacementThickness, Camber, ChannelAspectRatio, Solidity, BladeSpan)

        # At Shroud
        if RotationSpeed == 0.:
            # Only for stator
            PtLossAtShroud = PtLossModel_Roberts1988_EndWallWithoutClearance(r, rShroud, 'Shroud', ShroudDisplacementThickness, Camber, ChannelAspectRatio, Solidity, BladeSpan)
        else:
            # For rotor with tip clearance
            PtLossAtShroud = PtLossModel_Roberts1988_RotorTipClearance(r, rShroud, ShroudDisplacementThickness, TipClearance, BladeSpan)

        tmpMOLAFlowNode = I.getNodeFromName(zone, 'FlowSolution#tmpMOLAFlow')
        I.createChild(tmpMOLAFlowNode, 'PressureLossCoefficient', 'DataArray_t', value=PtLossAtHub + PtLossAtShroud)

        ####################################################################
        # Deviation with respect to flow angle at mid span
        if RotationSpeed == 0.:
            # Only for stators
            Underturning = DeviationModel_Roberts1988_StatorEndWalls(r, rHub, rShroud, delta1AtHub, delta1AtShroud, CamberAtHub, CamberAtShroud, SolidityAtHub, SolidityAtShroud, ARc, BladeSpan)
        else:
            # Only for rotors
            Underturning = DeviationModel_Roberts1988_RotorEndWalls(r, rHub, rShroud, delta1AtShroud, TipClearance, BladeSpan)

        # Add deviation to the midspan angle, and convert that into a force...


    LossSourceTerms = spreadPressureLossAlongChord(t, BodyForceParameters)
    BF.addDictionaries(NewSourceTermsGlobal, LossSourceTerms)

    DeviationSourceTerms = ...
    BF.addDictionaries(NewSourceTermsGlobal, DeviationSourceTerms)
    
    return NewSourceTermsGlobal