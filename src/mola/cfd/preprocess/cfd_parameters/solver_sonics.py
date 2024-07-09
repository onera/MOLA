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

from treelab import cgns

# TODO Check the correspondance of models in SoNICS
TURBULENCE_SONICS_KEYS = {

    'SA': dict(
        name = 'SpalartStandard',
    ),

    'SA-QCR2000': dict(
        name = 'SpalartQCR2020', 
    ),

    'SST': dict(
        name        = 'KOmegaMenterSST',
        prodK_type = 'from_S', 
    ),

    'SST-V': dict(
        name        = 'KOmegaMenterSST',
        prodK_type = 'from_W', 
    ),

    'BSL': dict(
        name = 'KOmegaMenterBSL',
        prodK_type = 'from_S',      
        # kprod_limiter = 20.,
    ),

    'BSL-V': dict(
        name = 'KOmegaMenterBSL',
        prodK_type = 'from_W',   
    ),

}

def apply_to_solver(workflow):

    import miles

    TurbulenceSetup = TURBULENCE_SONICS_KEYS[workflow.Turbulence['Model']]
    TurbulenceSetup['cutvars'] = get_turbulence_cutoff_setup(workflow.Turbulence)

    my_config = miles.solver.config.Configuration(workflow.tree, pure_cgns_mode=False)
    # my_config.add_template(configuration_templates.mobile)
    # my_config.add_template(configuration_templates.steady_spectral_ssor)

    my_config.add_template(
        dict(
            sonics = dict(
                formulation = dict(fvm="cell_center"),

                motion = "mobile",

                model = dict(
                    eos = "perfect_gas",
                    viscosity = "sutherland_law",
                    primitive_model = "primitive_model_from_temperature",
                ),

                execution = dict(hardware = "cpu"),
            ),

            GasModel = dict(name = 'PerfectGas', SpecificHeatRatio = workflow.Fluid['Gamma']),

            TurbulenceModel = TurbulenceSetup,
        )
    )

    my_config.add_template(get_turbulence_template(workflow.Turbulence))
    my_config.add_template(get_spatial_fluxes_template(workflow.Numerics))
    my_config.add_template(get_time_marching_template(workflow.Numerics))
    my_config.set_numerics(CFL=workflow.Numerics['CFL'])

    # FIXME pctrad=0.1 currently

    configuration = my_config.apply()

    configuration.update(
        dict(
            output_folder = ".",
            niter = workflow.Numerics['NumberOfIterations'],
            niter_period = 1,
            extracts = {'*': ['conservatives', 'LaminarViscosity', 'TurbulentViscosity','TurbulentViscosity', 'TurbulentDistance', 'Mach', 'primitives']},
            code_generation = "none",
            fcfl = workflow.Numerics['CFL'],
        )
    )

    # convert to dict to be able to write in cgns tree with treelab
    configuration['conf']  = configuration['conf'].to_dict(configuration['conf'])
    del configuration['hpc_conf'] 

    workflow.SolverParameters['configuration'] = configuration
    workflow.tree = cgns.castNode(workflow.tree)

def get_spatial_fluxes_template(Numerics):
    # from miles.solver import configuration_templates

    # Convective flux 
    if Numerics['Scheme'] == 'Roe':
        # template = configuration_templates.roe_second_order_none_cf
        template = dict(
            sonics = dict(
                numeric = dict(
                    scheme = dict(
                        upwind_scheme = dict(
                            upwind_grad_kind = "classic",
                            upwind_fxc = "roe",
                            upwind_limiter = "upwind_limiter_vanalbada",
                            upwind_order = 2,
                            upwind_sensor = "upwind_sensor_none",
                        ),
                    ),
                )
            )
        )
    else:
        raise MolaException(f"Scheme={Numerics['Scheme']} is not available for solver sonics")
    
    return template

def get_time_marching_template(Numerics):
    TimeMarchingSetup = dict(
        time_algo = 'steady',
        ode = "explicit",
        grad_scheme = "green_gauss", # shouldn't it be optional ???
        time_step = "local", # shouldn't it be optional ???
        viscous_flux = "vf5p_cor", # shouldn't it be optional ???
    )

    if Numerics['TimeMarching'] != 'Steady':
        raise MolaException(f"Only Steady simulations are implemented yet for soNICS with MOLA")

    template = dict(
        sonics = dict(numeric = TimeMarchingSetup)
    )

    return template

def get_turbulence_template(Turbulence):
    # from miles.solver import configuration_templates

    if Turbulence['Model'] == 'SA':
        # template = configuration_templates.spalart_standard
        template = dict(
            sonics = dict(
                model = dict(
                    physical_model = dict(
                        nstur = dict(
                            turbulence_closure = dict(spalart='spalart_standard')
                        )
                    )
                )
            )
        )
    else:
        raise MolaException(f"Scheme={Turbulence['Model']} is not available for solver sonics")

    return template

def get_turbulence_cutoff_setup(Turbulence):
    # Definition of cut-off values for turbulence 
    turbValues = list(Turbulence['Conservatives'].values())
    if len(turbValues) == 7:  # RSM
        cutoffs = [Turbulence['TurbulenceCutOffRatio'] * turbValues[i] for i in [0, 3, 5, 6]]
    elif len(turbValues) > 4: # unsupported 
        raise MolaException('Unsupported number of turbulent fields')
    else:
        cutoffs = [Turbulence['TurbulenceCutOffRatio'] * v for v in turbValues]

    return cutoffs