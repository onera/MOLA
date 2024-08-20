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

from mola.logging import mola_logger, MolaException
from mola.cfd.preprocess.boundary_conditions.boundary_conditions import BoundaryConditionsNames
from mola.cfd.preprocess.motion.solver_sonics import translate_motion_to_sonics

BoundaryConditionsNamesInSONICS = set(v['sonics'] for v in BoundaryConditionsNames.values() if 'sonics' in v)

# For each boundary condition, this generic function does the job
def function_generator(name):
    def set_bc(workflow, **kwargs):
        import miles

        Family = kwargs.pop('Family')
        kwargs = translate_motion(kwargs)
        
        interface = None
        try:
            # use the dedicated interface if it exists to prepared kwargs (parameters)
            interface = globals()[f'{name}_interface']  # interface is a function in this file named "<SonicsBCName>_interface"
        except:
            # no interface exists for this BC
            mola_logger.debug(f"  No interface function for BC {name}_interface")
            pass

        if interface is not None:
            kwargs = interface(workflow, Family=Family, **kwargs)

        miles.bcfactory(workflow.tree, name, Family, **kwargs)
    return set_bc

# Define functions with the write name to be called from .boundary_conditions
for fun_name in BoundaryConditionsNamesInSONICS:
    locals()[fun_name] = function_generator(fun_name)

def translate_motion(kwargs):
    if 'Motion' in kwargs:
        # put elements of dict Motion directly in kwargs (remove the "level" Motion)
        motion = kwargs.pop('Motion')
        motion = translate_motion_to_sonics(motion)
        kwargs['motion'] = motion
    return kwargs

def BCInflowSubsonicPressure_interface(workflow, **kwargs):
    '''
    This interface function must return a dict with the variables expected by Miles
    '''
    PressureStagnation    = kwargs.get('PressureStagnation', workflow.Flow['PressureStagnation'])
    TemperatureStagnation = kwargs.get('TemperatureStagnation', workflow.Flow['TemperatureStagnation'])
    EnthalpyStagnation    = kwargs.get('EnthalpyStagnation', workflow.Fluid['cp'] * TemperatureStagnation)
    VelocityUnitVectorX   = kwargs.get('VelocityUnitVectorX', workflow.Flow['Direction'][0])
    VelocityUnitVectorY   = kwargs.get('VelocityUnitVectorY', workflow.Flow['Direction'][1])
    VelocityUnitVectorZ   = kwargs.get('VelocityUnitVectorZ', workflow.Flow['Direction'][2])
    turb_values = workflow.Turbulence['Conservatives']
    for key in turb_values.keys():
        if key in kwargs:
            turb_values[key] = kwargs[key]

    ImposedVariables = dict(
        PressureStagnation  = PressureStagnation,
        EnthalpyStagnation  = EnthalpyStagnation,
        VelocityUnitVectorX = VelocityUnitVectorX,
        VelocityUnitVectorY = VelocityUnitVectorY,
        VelocityUnitVectorZ = VelocityUnitVectorZ,
        **turb_values
        )
    return ImposedVariables

def BCInflowSubsonicMassFlow_interface(workflow, **kwargs):
    Surface = kwargs.get('Surface')
    if not Surface:
        from mola.cfd.preprocess.mesh.tools import get_surface_of_family
        Surface = get_surface_of_family(workflow.tree, kwargs['Family'])
        try:
            Surface *= workflow.ApplicationContext['NormalizationCoefficient'][kwargs['Family']]['FluxCoef']
        except:
            pass

    MassFlow = kwargs.get('MassFlow')
    if MassFlow is None:
        try:
            MassFlow = workflow.Flow['MassFlow']
        except:
            MolaException('Error for InflowMassFlow boundary condition: '
                          'MassFlow is neither given by user as a boundary parameter, '
                          'nor foundable in workflow Flow attribute.')
    
    SurfacicMassFlow      = kwargs.get('SurfacicMassFlow', MassFlow / Surface)

    TemperatureStagnation = kwargs.get('TemperatureStagnation', workflow.Flow['TemperatureStagnation'])
    EnthalpyStagnation    = kwargs.get('EnthalpyStagnation', workflow.Fluid['cp'] * TemperatureStagnation)
    VelocityUnitVectorX   = kwargs.get('VelocityUnitVectorX', workflow.Flow['Direction'][0])
    VelocityUnitVectorY   = kwargs.get('VelocityUnitVectorY', workflow.Flow['Direction'][1])
    VelocityUnitVectorZ   = kwargs.get('VelocityUnitVectorZ', workflow.Flow['Direction'][2])
    turb_values = workflow.Turbulence['Conservatives']
    for key in turb_values.keys():
        if key in kwargs:
            turb_values[key] = kwargs[key]

    ImposedVariables = dict(
        MassFlow            = SurfacicMassFlow,
        EnthalpyStagnation  = EnthalpyStagnation,
        VelocityUnitVectorX = VelocityUnitVectorX,
        VelocityUnitVectorY = VelocityUnitVectorY,
        VelocityUnitVectorZ = VelocityUnitVectorZ,
        **turb_values
        )
    return ImposedVariables

def BCOutflowSubsonic_interface(workflow, **kwargs):
    ImposedVariables = dict(
        Pressure = kwargs.get('Pressure', workflow.Flow['Pressure'])
        )
    return ImposedVariables
