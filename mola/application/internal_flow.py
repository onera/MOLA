#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

def compute_reference_values(FluidProperties, PressureStagnation,
                           TemperatureStagnation, Surface, MassFlow=None, Mach=None, TurbulenceLevel=0.001,
        Viscosity_EddyMolecularRatio=0.1, TurbulenceModel='Wilcox2006-klim',
        TurbulenceCutoff=1e-8, TransitionMode=None, CoprocessOptions={},
        Length=1.0, TorqueOrigin=[0., 0., 0.],
        FieldsAdditionalExtractions=['ViscosityMolecular', 'Viscosity_EddyMolecularRatio', 'Pressure', 'Temperature', 'PressureStagnation', 'TemperatureStagnation', 'Mach', 'Entropy'],
        BCExtractions=dict(
            BCWall = ['normalvector', 'frictionvector','psta', 'bl_quantities_2d', 'yplusmeshsize'],
            BCInflow = ['convflux_ro'],
            BCOutflow = ['convflux_ro']),
        AngleOfAttackDeg=0.,
        YawAxis=[0.,0.,1.],
        PitchAxis=[0.,1.,0.]):
    '''
    This function is the Compressor's equivalent of :func:`MOLA.Preprocess.computeReferenceValues`.
    The main difference is that in this case reference values are set through
    ``MassFlow``, total Pressure ``PressureStagnation``, total Temperature
    ``TemperatureStagnation`` and ``Surface``.

    You can also give the Mach number instead of massflow (but not both).

    Please, refer to :func:`MOLA.Preprocess.computeReferenceValues` doc for more details.
    '''
    # Fluid properties local shortcuts
    Gamma   = FluidProperties['Gamma']
    IdealGasConstant = FluidProperties['IdealGasConstant']
    cv      = FluidProperties['cv']
    cp      = FluidProperties['cp']

    # Compute variables
    assert not(MassFlow and Mach), 'MassFlow and Mach cannot be given together in ReferenceValues. Choose one'
    if MassFlow:
        Mach  = machFromMassFlow(MassFlow, Surface, Pt=PressureStagnation,
                                Tt=TemperatureStagnation, r=IdealGasConstant,
                                gamma=Gamma)
    else:
        MassFlow  = massflowFromMach(Mach, Surface, Pt=PressureStagnation,
                            Tt=TemperatureStagnation, r=IdealGasConstant,
                            gamma=Gamma)
    Temperature  = TemperatureStagnation / (1. + 0.5*(Gamma-1.) * Mach**2)
    Pressure  = PressureStagnation / (1. + 0.5*(Gamma-1.) * Mach**2)**(Gamma/(Gamma-1))
    Density = Pressure / (Temperature * IdealGasConstant)
    SoundSpeed  = np.sqrt(Gamma * IdealGasConstant * Temperature)
    Velocity  = Mach * SoundSpeed

    # REFERENCE VALUES COMPUTATION
    mus = FluidProperties['SutherlandViscosity']
    Ts  = FluidProperties['SutherlandTemperature']
    S   = FluidProperties['SutherlandConstant']
    ViscosityMolecular = mus * (Temperature/Ts)**1.5 * ((Ts + S)/(Temperature + S))

    # if not 'AveragingIterations' in CoprocessOptions:
    #     CoprocessOptions['AveragingIterations'] = 1000

    TurboStatistics = ['rsd-{}'.format(var) for var in ['MassFlowIn', 'MassFlowOut',
        'PressureStagnationRatio', 'TemperatureStagnationRatio', 'EfficiencyIsentropic',
        'PressureStagnationLossCoeff']]
    try:
        RequestedStatistics = CoprocessOptions['RequestedStatistics']
        for stat in TurboStatistics:
            if stat not in CoprocessOptions:
                RequestedStatistics.append( stat )
    except KeyError:
        CoprocessOptions['RequestedStatistics'] = TurboStatistics

    CoprocessOptions.setdefault('BodyForceComputeFrequency', 1)

    ReferenceValues = PRE.computeReferenceValues(FluidProperties,
        Density=Density,
        Velocity=Velocity,
        Temperature=Temperature,
        AngleOfAttackDeg=AngleOfAttackDeg,
        AngleOfSlipDeg = 0.0,
        YawAxis=YawAxis,
        PitchAxis=PitchAxis,
        TurbulenceLevel=TurbulenceLevel,
        Surface=Surface,
        Length=Length,
        TorqueOrigin=TorqueOrigin,
        TurbulenceModel=TurbulenceModel,
        Viscosity_EddyMolecularRatio=Viscosity_EddyMolecularRatio,
        TurbulenceCutoff=TurbulenceCutoff,
        TransitionMode=TransitionMode,
        CoprocessOptions=CoprocessOptions,
        FieldsAdditionalExtractions=FieldsAdditionalExtractions,
        BCExtractions=BCExtractions)

    addKeys = dict(
        PressureStagnation = PressureStagnation,
        TemperatureStagnation = TemperatureStagnation,
        MassFlow = MassFlow,
        )

    ReferenceValues.update(addKeys)

    return ReferenceValues


def massflowFromMach(Mx, S, Pt=101325.0, Tt=288.25, r=287.053, gamma=1.4):
    '''
    Compute the massflow rate through a section.

    Parameters
    ----------

        Mx : :py:class:`float`
            Mach number in the normal direction to the section.

        S : :py:class:`float`
            Surface of the section.

        Pt : :py:class:`float`
            Stagnation pressure of the flow.

        Tt : :py:class:`float`
            Stagnation temperature of the flow.

        r : :py:class:`float`
            Specific gas constant.

        gamma : :py:class:`float`
            Ratio of specific heats of the gas.


    Returns
    -------

        massflow : :py:class:`float`
            Value of massflow through the section.
    '''
    return S * Pt * (gamma/r/Tt)**0.5 * Mx / (1. + 0.5*(gamma-1.) * Mx**2) ** ((gamma+1) / 2 / (gamma-1))

def machFromMassFlow(massflow, S, Pt=101325.0, Tt=288.25, r=287.053, gamma=1.4):
    '''
    Compute the Mach number normal to a section from the massflow rate.

    Parameters
    ----------

        massflow : :py:class:`float`
            MassFlow rate through the section.

        S : :py:class:`float`
            Surface of the section.

        Pt : :py:class:`float`
            Stagnation pressure of the flow.

        Tt : :py:class:`float`
            Stagnation temperature of the flow.

        r : :py:class:`float`
            Specific gas constant.

        gamma : :py:class:`float`
            Ratio of specific heats of the gas.


    Returns
    -------

        Mx : :py:class:`float`
            Value of the Mach number in the normal direction to the section.
    '''
    if isinstance(massflow, (list, tuple, np.ndarray)):
        Mx = []
        for i, MF in enumerate(massflow):
            Mx.append(machFromMassFlow(MF, S, Pt=Pt, Tt=Tt, r=r, gamma=gamma))
        if isinstance(massflow, np.ndarray):
            Mx = np.array(Mx)
        return Mx
    else:
        # Check that massflow is lower than the chocked massflow
        chocked_massflow = massflowFromMach(1., S, Pt=Pt, Tt=Tt, r=r, gamma=gamma)
        assert massflow < chocked_massflow, "MassFlow ({:6.3f}kg/s) is greater than the chocked massflow ({:6.3f}kg/s)".format(massflow, chocked_massflow)
        # MassFlow as a function of Mach number
        f = lambda Mx: massflowFromMach(Mx, S, Pt, Tt, r, gamma)
        # Objective function
        g = lambda Mx: f(Mx) - massflow
        # Search for the corresponding Mach Number between 0 and 1
        Mx = scipy.optimize.brentq(g, 0, 1)
        return Mx
