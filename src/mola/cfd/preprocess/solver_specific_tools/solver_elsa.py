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

def translate_to_elsa(Variables):
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

    Returns
    -------

        NewVariables : :py:class:`dict` or :py:class:`list` or :py:class:`str`
            Depending on the input type, return the same object with variable
            names translated to elsA standards.

    '''
    CGNS2ElsaDict = dict(
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
        VelocityCorrelationZZ    = 'inj_tur6',
        
        BoundaryLayer            = 'bl_quantities_2d bl_quantities_3d bl_ue',
        NormalVector             = 'normalvector',
        Friction                 = 'frictionvector', 
        yPlus                    = 'yplusmeshsize',
        MomentumFlux             = 'flux_rou flux_rov flux_row',
        TorqueFlux               = 'torque_rou torque_rov torque_row',
    )

    if isinstance(Variables, (dict, list)) and 'VelocityCorrelationXX' in Variables:
        # For RSM models
        CGNS2ElsaDict['TurbulentDissipationRate'] = 'inj_tur7'

    elsAVariables = CGNS2ElsaDict.values()

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
            if var in elsAVariables:
                NewVariables.append(var)
            else:
                NewVariables.append(CGNS2ElsaDict[var])
        return NewVariables
    elif isinstance(Variables, str):
        if Variables in CGNS2ElsaDict:
            return CGNS2ElsaDict[Variables]
    else:
        raise TypeError('Variables must be of type dict, list or string')

 