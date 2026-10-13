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

from treelab import cgns

cgns_to_elsa_bc_field_name = dict(
        PressureStagnation       = 'stagnation_pressure',
        EnthalpyStagnation       = 'stagnation_enthalpy',
        TemperatureStagnation    = 'stagnation_temperature',
        Pressure                 = 'pressure',
        MassFlow                 = 'globalmassflow',
        SurfacicMassFlow         = 'surf_massflow',
        VelocityUnitVectorX      = 'txv',
        VelocityUnitVectorY      = 'tyv',
        VelocityUnitVectorZ      = 'tzv',
        TurbulentSANuTilde       = 'inj_tur1',
        TurbulentEnergyKinetic   = 'inj_tur1',
        TurbulentDissipationRate = 'inj_tur2',
        TurbulentDissipation     = 'inj_tur2',
        TurbulentLengthScale     = 'inj_tur2',
        VelocityCorrelationXX    = 'inj_tur1',
        VelocityCorrelationXY    = 'inj_tur2', 
        VelocityCorrelationXZ    = 'inj_tur3',
        VelocityCorrelationYY    = 'inj_tur4', 
        VelocityCorrelationYZ    = 'inj_tur5', 
        VelocityCorrelationZZ    = 'inj_tur6'
)

cgns_to_elsa_extraction_name = dict(
    CoordinateX = 'x',
    CoordinateY = 'y',
    CoordinateZ = 'z',
    Density = 'ro',
    EnergyStagnationDensity = 'roE',
    Enthalpy = 'enthalpy',
    EnthalpyStagnation = 'stagnation_enthalpy',
    Entropy = 'entropy',
    IntermittencyDensity = 'rotrans1',
    Mach = 'mach',
    MomentumThicknessReynoldsDensity = 'rotrans2',
    MomentumX = 'rovx',
    MomentumY = 'rovy',
    MomentumZ = 'rovz',
    Pressure = 'psta',
    PressureStagnation = 'pgen',
    ReynoldsStressDissipationScale = 'roscale',
    ReynoldsStressXX = 'rouu',
    ReynoldsStressXY = 'rouv',
    ReynoldsStressXZ = 'rouw',
    ReynoldsStressYY = 'rovv',
    ReynoldsStressYZ = 'rovw',
    ReynoldsStressZZ = 'roww',
    SkinFrictionMagnitude = 'frictionmodulus',
    SkinFrictionX = 'frictionvectorx',
    SkinFrictionY = 'frictionvectory',
    SkinFrictionZ = 'frictionvectorz',
    SpecificHeatPressure = 'cp',
    SpecificHeatRatio = 'gamma',
    SpecificHeatVolume = 'cv',
    SpectralFluxProdTransDensity = 'rof1',
    SpectralFluxTransDissDensity = 'rof2',
    Temperature = 'tsta',
    TemperatureStagnation = 'tgen',
    TurbulentDissipation = 'eps',
    TurbulentDissipationDensity = 'roeps',
    TurbulentDissipationRate = 'omega',
    TurbulentDissipationRateDensity = 'roeps',  # yes, roeps for ρω
    TurbulentDistance = 'walldistance',
    TurbulentDistanceIndex = 'wallglobalindex',
    TurbulentEnergyKinetic = 'k',
    TurbulentEnergyKineticDensity = 'rok',
    TurbulentEnergyKineticPLS = 'kl',
    TurbulentEnergyKineticPLSDensity = 'rokl',
    TurbulentEnergyKineticPZDensity = 'rok1',
    TurbulentEnergyKineticTZDensity = 'rok2',
    TurbulentLengthScale = 'l',
    TurbulentLengthScaleDensity = 'rol',
    TurbulentSANuTildeDensity = 'ronutilde',
    TurbulentTimeScaleVar = 'phi',
    TurbulentTimeScaleVarDensity = 'rophi',
    VelocityCorrelationXX = 'uu',
    VelocityCorrelationXY = 'uv',
    VelocityCorrelationXZ = 'uw',
    VelocityCorrelationYY = 'vv',
    VelocityCorrelationYZ = 'vw',
    VelocityCorrelationZZ = 'ww',
    VelocitySound = 'soundspeed',
    VelocityUnitVectorX = 'd0x',
    VelocityUnitVectorY = 'd0y',
    VelocityUnitVectorZ = 'd0z',
    VelocityX = 'u',
    VelocityY = 'v',
    VelocityZ = 'w',
    ViscosityEddy = 'viscturb',
    ViscosityMolecular = 'visclam',
    Viscosity_EddyMolecularRatio = 'viscrapp',
    VorticityX = 'vorticity_x',
    VorticityY = 'vorticity_y',
    VorticityZ = 'vorticity_z',
    v10 = 'rotur5',
    v11 = 'rotur6',
    v12 = 'rotur7',
    v6 = 'rotur1',
    v7 = 'rotur2',
    v8 = 'rotur3',
    v9 = 'rotur4',
    q_criterion = 'q_criterion',
    cellN = 'cellnf', # does not seem official
    ChimeraCellType = 'cellnf', # does not seem official
)

ElsaCGNS2MOLA = dict(
    IterationNumber = 'Iteration',
    MomentumXFlux = 'ForceX',
    MomentumYFlux = 'ForceY',
    MomentumZFlux = 'ForceZ',
    convflux_ro   = 'MassFlow',
)

# Integral data (flux_*) and MOLA shortcuts (not official CGNS names)
cgns_to_elsa_extraction_name.update(dict(
    Coordinates              = 'xyz',
    BoundaryLayer            = 'bl_quantities_2d bl_quantities_3d bl_ue_vector',
    NormalVector             = 'normalvector',
    Velocity                 = 'vx vy vz',
    Momentum                 = 'rovx rovy rovz',
    SkinFriction             = 'frictionvector', 
    SkinFrictionX            = 'frictionvectorx',
    SkinFrictionY            = 'frictionvectory',
    SkinFrictionZ            = 'frictionvectorz',
    yPlus                    = 'yplusmeshsize',
    
    Force                    = 'flux_rou flux_rov flux_row',
    ForceX                   = 'flux_rou',
    ForceY                   = 'flux_rov',
    ForceZ                   = 'flux_row',
    Torque                   = 'torque_rou torque_rov torque_row',
    TorqueX                  = 'torque_rou',
    TorqueY                  = 'torque_rov',
    TorqueZ                  = 'torque_row',
    MassFlow                 = 'convflux_ro',
))


def translate_to_elsa(Variables, type='node'):
    '''
    Translate names in **Variables** from CGNS standards to elsA names for
    boundary conditions.

    Parameters
    ----------

        Variables : :py:class:`dict` or :py:class:`list` or :py:class:`str`
            Could be eiter:

                * a :py:class:`dict` with keys corresponding to variables names

                * a :py:class:`list` of variables names

                * a :py:class:`str` as a single variable name
        
        type : 'node' or 'var'
            Context to translate CGNS names to elsA names.

    Returns
    -------

        NewVariables : :py:class:`dict` or :py:class:`list` or :py:class:`str`
            Depending on the input type, return the same object with variable
            names translated to elsA standards.

    '''
    if type == 'node':
        CGNS2ElsaDict = cgns_to_elsa_bc_field_name.copy()  # ensure not to modify the reference dict
        if isinstance(Variables, (dict, list)) and 'VelocityCorrelationXX' in Variables:
            # For RSM models
            CGNS2ElsaDict['TurbulentDissipationRate'] = 'inj_tur7'
    else:
        CGNS2ElsaDict = cgns_to_elsa_extraction_name.copy()  # ensure not to modify the reference dict

    if isinstance(Variables, dict):
        NewVariables = dict()
        for var, value in Variables.items():
            if var in CGNS2ElsaDict:
                NewVariables[CGNS2ElsaDict[var]] = value
            else:
                NewVariables[var] = value
        return NewVariables
    elif isinstance(Variables, list):
        NewVariables = []
        for var in Variables:
            if var in CGNS2ElsaDict:
                NewVariables.append(CGNS2ElsaDict[var])
            else:
                NewVariables.append(var)                    
        return NewVariables
    elif isinstance(Variables, str):
        if Variables in CGNS2ElsaDict:
            return CGNS2ElsaDict[Variables]
    else:
        raise TypeError('Variables must be of type dict, list or string')

def translate_elsa_CGNS_field_names_to_MOLA(container_node : cgns.Node):

    for node in container_node.children():
        node_name = node.name()
        if node_name in ElsaCGNS2MOLA:
            node.setName( ElsaCGNS2MOLA[node_name] )

