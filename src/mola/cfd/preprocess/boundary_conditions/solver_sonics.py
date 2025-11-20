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

from mola.logging import mola_logger, MolaException, MolaUserError
from mola.cfd.preprocess.boundary_conditions.boundary_conditions import (
    get_turbulent_primitives, 
    get_fluxcoeff_on_bc, 
    OutflowRadialEquilibrium_interface,
)
from mola.cfd.preprocess.motion.solver_sonics import translate_motion_to_sonics


from mola.cfd.preprocess.boundary_conditions.boundary_conditions_dispatcher_sonics import BoundaryConditionsDispatcherSonics
bc_dispatcher = BoundaryConditionsDispatcherSonics()
BoundaryConditionsNamesInSONICS = bc_dispatcher.get_all_specific_names()


# For each boundary condition, this generic function does the job
def function_generator(bc_type):

    def set_bc(workflow, **kwargs):
        import miles

        Family = kwargs.pop('Family')
        kwargs = mola_to_miles(workflow, Family, bc_type, kwargs)

        if bc_type.startswith('GC'):

            # ### Check on MPI size, until mixing plane is made available for NumberOfProcessors>1
            # if bc_type == 'GCMixingPlane' and workflow.RunManagement['NumberOfProcessors']>1:
            #     raise MolaException('For now, MixingPlane in SoNICS is available only for a simulation on one MPI rank.')
            # ### End of MPI check

            LinkedFamily = kwargs.pop('LinkedFamily')
            miles.set_gc(workflow.tree, bc_type, Family, LinkedFamily, **kwargs)
        
        else:
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

    if 'PressureAtHub' in kwargs:
        Pressure = kwargs.get('PressureAtHub')
        PivotPercenthH = 0.
    elif 'PressureAtShroud' in kwargs:
        Pressure = kwargs.get('PressureAtShroud')
        PivotPercenthH = 1.
    elif 'PressureAtSpecifiedHeight' in kwargs:
        Pressure = kwargs.get('PressureAtSpecifiedHeight')
        PivotPercenthH = kwargs.get('Height')
        if PivotPercenthH is None:
            raise MolaUserError((
                'Height must be provided if PressureAtSpecifiedHeight is given. '
                'Otherwise, consider giving directly PressureAtHub or PressureAtShroud.'
            ))
    else:
        # a valve law is used, but default parameters must still be provided anyway
        Pressure = kwargs.get('PressureAtSpecifiedHeight', workflow.Flow['Pressure'])
        PivotPercenthH = kwargs.get('Height', 0.)

    parameters = dict(
        Pressure = Pressure,
        PivotPercenthH = PivotPercenthH,
        )

    return parameters

def valve_law_interface(workflow, **kwargs):

    # Default values, will be updated below depending on the valve law
    valve_ref_pres = workflow.Flow['Pressure']
    valve_ref_mflow = workflow.Flow['MassFlow']
    valve_relax = 0.1
    valve_period = kwargs.get('valve_period', 10)

    ValveLaw = kwargs.get('ValveLaw')
    if 'MassFlow' in kwargs:
        valve_type = 'BCValveLawQTarget'
        fluxcoeff = get_fluxcoeff_on_bc(workflow, kwargs['Family'])
        valve_ref_mflow = kwargs['MassFlow'] / fluxcoeff
        pref = kwargs.get('PressureRef')
        if pref is not None:
            valve_ref_pres = pref 
        
    elif ValveLaw['Type'] == 'Linear':
        valve_type = 'BCValveLawSlopePsQ'
        fluxcoeff = get_fluxcoeff_on_bc(workflow, kwargs['Family'])
        valve_ref_pres = ValveLaw['PressureRef']
        valve_ref_mflow = ValveLaw['MassFlowRef'] / fluxcoeff
        valve_relax = ValveLaw['RelaxationCoefficient']

    elif ValveLaw['Type'] == 'Quadratic':
        valve_type = 'BCValveLawQHyperbolic'
        fluxcoeff = get_fluxcoeff_on_bc(workflow, kwargs['Family'])
        valve_ref_pres = ValveLaw['PressureRef']
        valve_ref_mflow = ValveLaw['MassFlowRef'] / fluxcoeff
        valve_relax = ValveLaw['ValveCoefficient'] * workflow.Flow['PressureStagnation']

    else:
        raise MolaUserError(f"Valve law {ValveLaw['Type']} is not available with SoNICS. Available laws are 'Linear' and 'Quadratic'.")

    parameters = dict(
        valve_type = valve_type, 
        valve_ref_pres = valve_ref_pres,
        valve_ref_mflow = valve_ref_mflow, 
        valve_relax = valve_relax,
        valve_period = valve_period,
        )

    return parameters

def get_valve_law_trigger(workflow, config, bc, hardware_target='cpu'):
    from sonics.toolkit.triggers import valve_law_trigger as VLT

    OutflowRadialEquilibrium_interface(workflow, bc)
    valve_params = valve_law_interface(workflow, **bc)

    valve_law_trigger = VLT.ValveLawRadialEquilibrium(
        config, 
        hardware_target, 
        family=bc['Family'], 
        valve_ref_pressure=valve_params['valve_ref_pres'], 
        valve_ref_massflow=valve_params['valve_ref_mflow'], 
        niter=workflow.Numerics['NumberOfIterations'], 
        valve_law=valve_params['valve_type'], 
        valve_relax=valve_params['valve_relax'], 
        period=valve_params['valve_period']
        )
    
    return valve_law_trigger


def adapt_workflow_for_sonics(w):
    will_define_connection = _will_define_connection(w)
    will_define_families = _will_define_families(w)
    
    if will_define_connection or will_define_families:
        _read_mesh(w)    

        if will_define_connection:
            _preset_connection(w)

        if will_define_families:
            _preset_families(w)

    _preset_splitting_and_distribution(w)

def _read_mesh(workflow):
    from mola.cfd.preprocess.mesh import io
    io.read(workflow)
    workflow.RawMeshComponents[0]['Source'] = workflow.tree

def _will_define_connection(workflow) -> bool:
    return 'Connection' in workflow.RawMeshComponents[0]

def _will_define_families(workflow) -> bool:
    return 'Families' in workflow.RawMeshComponents[0]

def _preset_families(workflow):
    from mola.cfd.preprocess.mesh import families
    families.apply(workflow)
    workflow.RawMeshComponents[0].pop('Families')

def _preset_connection(workflow):
    from mola.cfd.preprocess.mesh import connect
    connect.apply(workflow)
    workflow.RawMeshComponents[0].pop('Connection')

def _preset_splitting_and_distribution(workflow):
    workflow.SplittingAndDistribution = dict(Splitter='maia', Strategy='AtComputation')