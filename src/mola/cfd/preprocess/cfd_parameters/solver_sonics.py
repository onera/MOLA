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
from mola.cfd.preprocess.cfd_parameters import cfd_parameters

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
    from miles.solver import configuration_templates

    TurbulenceSetup = TURBULENCE_SONICS_KEYS[workflow.Turbulence['Model']]
    TurbulenceCutOffSetup = cfd_parameters.get_turbulence_cutoff_setup(workflow.Turbulence)
    TurbulenceSetup['cutvars'] = TurbulenceCutOffSetup.values()

    my_config = miles.solver.config.Configuration(workflow.tree, pure_cgns_mode=False)
    my_config.add_template(configuration_templates.mobile)
    my_config.add_template(configuration_templates.steady_spectral_ssor)
    my_config.add_template(get_spatial_fluxes_template(workflow.Numerics))
    my_config.set_turbulence_model(**TurbulenceSetup)
    configuration = my_config.apply()

    configuration.update(
        dict(
            output_folder = ".",
            niter = workflow.Numerics['NumberOfIterations'],
            niter_period = 1,
            extracts = {'*': ['conservatives', 'LaminarViscosity', 'TurbulentViscosity','TurbulentViscosity', 'TurbulentDistance',"Mach","primitives"]},
            code_generation = "none",
            fcfl = workflow.Numerics['CFL'],
        )
    )

    workflow.SolverParameters['configuration'] = configuration
    workflow.tree = cgns.castNode(workflow.tree)

def get_spatial_fluxes_template(Numerics):
    from miles.solver import configuration_templates

    # Convective flux 
    if Numerics['Scheme'] == 'Roe':
        template = configuration_templates.roe_second_order_none_cf
    else:
        raise MolaException(f"Scheme={Numerics['Scheme']} is not available for solver sonics")
    
    return template

