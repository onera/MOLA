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

from mola.logging import mola_logger, MolaException
from mola.cfd.preprocess.boundary_conditions.boundary_conditions import BoundaryConditionsNames, get_turbulent_primitives, get_bc_nodes_from_family
from mola.cfd.preprocess.motion.solver_sonics import translate_motion_to_sonics

BoundaryConditionsNamesInSONICS = set(v['sonics'] for v in BoundaryConditionsNames.values() if 'sonics' in v)


# For each boundary condition, this generic function does the job
def function_generator(bc_type):

    if bc_type.startswith('GC'):
        def set_gc(workflow, **kwargs):
            import miles

            Family = kwargs.pop('Family')
            LinkedFamily = kwargs.pop('LinkedFamily')
            miles.set_gc(workflow.tree, bc_type, Family, LinkedFamily)
            workflow.tree = cgns.castNode(workflow.tree)
            
        return set_gc
    
    else:
        def set_bc(workflow, **kwargs):
            import miles

            Family = kwargs.pop('Family')
            kwargs = mola_to_miles(workflow, Family, bc_type, kwargs)
            miles.set_bc(workflow.tree, bc_type, Family, **kwargs)
            workflow.tree = cgns.castNode(workflow.tree)
            
        return set_bc

# Define functions with the write name to be called from .boundary_conditions
for fun_name in BoundaryConditionsNamesInSONICS:
    locals()[fun_name] = function_generator(fun_name)


def mola_to_miles(workflow, Family, bc_type, kwargs):
    kwargs = translate_motion(kwargs)
    
    interface = None
    try:
        # use the dedicated interface if it exists to prepared kwargs (parameters)
        interface = globals()[f'{bc_type}_interface']  # interface is a function in this file named "<SonicsBCName>_interface"
    except:
        # no interface exists for this BC
        mola_logger.debug(f"  No interface function for BC {bc_type}_interface")
        pass

    if interface is not None:
        kwargs = interface(workflow, Family=Family, **kwargs)
    
    return kwargs

def translate_motion(kwargs):
    from mola.cfd.preprocess.motion.motion import update_motion_with_defaults
    if 'Motion' in kwargs:
        # put elements of dict Motion directly in kwargs (remove the "level" Motion)
        motion = kwargs.pop('Motion')
        update_motion_with_defaults(motion)
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

    ImposedVariables = dict(
        PressureStagnation  = PressureStagnation,
        EnthalpyStagnation  = EnthalpyStagnation,
        VelocityUnitVectorX = VelocityUnitVectorX,
        VelocityUnitVectorY = VelocityUnitVectorY,
        VelocityUnitVectorZ = VelocityUnitVectorZ,
        **get_turbulent_primitives(workflow, **kwargs)
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

    ImposedVariables = dict(
        MassFlow            = SurfacicMassFlow,
        EnthalpyStagnation  = EnthalpyStagnation,
        VelocityUnitVectorX = VelocityUnitVectorX,
        VelocityUnitVectorY = VelocityUnitVectorY,
        VelocityUnitVectorZ = VelocityUnitVectorZ,
        **get_turbulent_primitives(workflow, **kwargs)
        )
    return ImposedVariables

def BCOutflowSubsonic_interface(workflow, **kwargs):
    ImposedVariables = dict(
        Pressure = kwargs.get('Pressure', workflow.Flow['Pressure'])
        )
    return ImposedVariables

def BCOutflowRadialEquilibrium_interface(workflow, **kwargs):

    AVAILABLE_VALVE_LAWS = [None, 'BCValveLawSlopePsQ', 'BCValveLawQTarget', 'BCValveLawQHyperbolic'] # respectively laws 1, 2 and 4

    parameters = dict(
        Pressure = kwargs.get('Pressure', workflow.Flow['Pressure']),
        PivotPercenthH = kwargs.get('PivotPercenthH', 0.),
        )
    
    valve_type = kwargs.get('valve_type')
    assert valve_type in AVAILABLE_VALVE_LAWS
    if valve_type is None:
        return parameters

    def _get_default_valve_ref_mflow():
        bcs = get_bc_nodes_from_family(workflow.tree, kwargs['Family'])
        bc = bcs[0]
        zone = bc.getParent(Type='Zone_t')
        row = zone.get(Type='FamilyName').value()
        try:
            rowParams = workflow.ApplicationContext['Rows'][row]
        except:
            raise MolaException('Worklow must have an attribute ApplicationContext with a dict named "Rows" inside.')
        fluxcoeff = rowParams['NumberOfBlades'] / float(rowParams['NumberOfBladesSimulated'])
        try:
            valve_ref_mflow = workflow.Flow['MassFlow'] / fluxcoeff
        except:
            raise MolaException('Miss MassFlow in Flow attribute')
        
        return valve_ref_mflow

    valve_ref_mflow = kwargs.get('valve_ref_mflow')
    if not valve_ref_mflow:
        valve_ref_mflow = kwargs.get('MassFlow', _get_default_valve_ref_mflow())

    parameters.update(
        dict(
            valve_type = valve_type, 
            valve_ref_mflow = valve_ref_mflow, 
            valve_relax = kwargs.get('valve_relax', 0.1),
        )
    )
    
    return parameters

def get_valve_law_trigger(config, bc, niter, hardware_target='cpu', period=10):
    from sonics.toolkit.triggers import valve_law_trigger as VLT

    valve_law_trigger = VLT.ValveLawRadialEquilibrium(
        config, 
        hardware_target, 
        bc['Family'], 
        bc['valve_ref_pres'], 
        bc['valve_ref_mflow'], 
        niter, 
        valve_law=bc['valve_type'], 
        valve_relax=bc['valve_relax'], 
        period=period
        )
    
    return valve_law_trigger
