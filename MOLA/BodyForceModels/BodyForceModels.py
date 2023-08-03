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

    return BLProtectionSourceTerms


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
    SourceTerms = dict()
    for key in BodyForceParameters['ReferenceValues']['Fields']:

        if key in BodyForceParameters:
            SourceTerms[key] = BodyForceParameters[key]

        NewSourceTermsGlobal = dict()
        for zone in I.getZones(t):

            FlowSolution = J.getVars2Dict(zone, Container='FlowSolution#Init')
            ones = np.ones(FlowSolution['Density'].shape)

            NewSourceTerms = dict()
            for key, value in SourceTerms.items():
                NewSourceTerms[key] = value * ones
            else:
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


def spreadLossAlongChord(t, BodyForceParameters):
    FluidProperties = BodyForceParameters.get('FluidProperties')
    TurboConfiguration = BodyForceParameters.get('TurboConfiguration')
    R = FluidProperties['IdealGasConstant']

    NewSourceTermsGlobal = dict()
    for zone in I.getZones(t):

        rowName = I.getValue(I.getNodeFromType1(zone, 'FamilyName_t'))
        RotationSpeed = TurboConfiguration['Rows'][rowName]['RotationSpeed']
        FlowSolution = J.getVars2Dict(zone, Container='FlowSolution#Init')
        DataSourceTerms = J.getVars2Dict(zone, Container='FlowSolution#DataSourceTerm')
        tmpMOLAFlow = BF.getAdditionalFields(zone, FluidProperties, RotationSpeed)

        gradPt = ... # Depends on the loss coefficent provided to the function
        fp = R * tmpMOLAFlow['Temperature'] / tmpMOLAFlow['Ptrel'] * np.cos(tmpMOLAFlow['incidence']) * gradPt

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
