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

import copy
from pprint import pprint

from treelab import cgns
import mola.naming_conventions as names
from mola.logging import mola_logger, MolaException
from mola.cfd.preprocess.cfd_parameters.cfd_parameters import deep_update

TURBULENCE_SONICS_KEYS = {

    'SA': dict(
        features = ['spalart_standard'],
        parameters = dict(),
    ),

    # 'Wilcox2006-klim': dict(
    #     turbmod        = 'komega_kok',
    #     kok_diff_cor   = 'wilcox2006',
    #     sst_cor        = 'active',
    #     sst_version    = 'wilcox2006',
    #     k_prod_limiter = 20.,
    #     k_prod_compute = 'from_sij',
    #     zhenglim       = 'inactive',
    #     omega_prolong  = 'linear_extrap',
    # ),
    
    # 'Wilcox2006-klim-V': dict(
    #     turbmod        = 'komega_kok',
    #     kok_diff_cor   = 'wilcox2006',
    #     sst_cor        = 'active',
    #     sst_version    = 'wilcox2006',
    #     k_prod_limiter = 20.,
    #     k_prod_compute = 'from_vorticity',
    #     zhenglim       = 'inactive',
    #     omega_prolong  = 'linear_extrap',
    # ),

    # 'Wilcox2006': dict(
    #     turbmod        = 'komega_kok',
    #     kok_diff_cor   = 'wilcox2006',
    #     sst_cor        = 'active',
    #     sst_version    = 'wilcox2006',
    #     k_prod_compute = 'from_sij',
    #     zhenglim       = 'inactive',
    #     omega_prolong  = 'linear_extrap',
    # ),
    
    # 'Wilcox2006-V': dict(
    #     turbmod        = 'komega_kok',
    #     kok_diff_cor   = 'wilcox2006',
    #     sst_cor        = 'active',
    #     sst_version    = 'wilcox2006',
    #     k_prod_compute = 'from_vorticity',
    #     zhenglim       = 'inactive',
    #     omega_prolong  = 'linear_extrap',
    # ),

    'SST-2003': dict(
        features = ['sst/std_sij', 'k_prod/from_sij'],
        parameters = dict(k_prod_limiter=10.),
    ),

    'SST-V2003': dict(
        features = ['sst/std_sij', 'k_prod/from_vorticity'],
        parameters = dict(k_prod_limiter=10.),   
    ),

    'SST': dict(
        features = ['sst/std_vort', 'k_prod/from_sij'],
        parameters = dict(k_prod_limiter=20.),
    ),

    'SST-V': dict(
        features = ['sst/std_vort', 'k_prod/from_vorticity'],
        parameters = dict(k_prod_limiter=20.),
    ),

    'BSL': dict(
        features = ['bsl', 'k_prod/from_sij'],
        parameters = dict(k_prod_limiter=20.),     
    ),

    'BSL-V': dict(
        features = ['bsl', 'k_prod/from_vorticity'],
        parameters = dict(k_prod_limiter=20.),     
    ),

    'smith': dict(
        turbmod        = 'kl_smith',
        k_prod_compute = 'from_sij',
    ),

    'smith-V': dict(
        turbmod        = 'kl_smith',
        k_prod_compute = 'from_vorticity',
    ),
}

for model in ['SST-2003', 'SST-V2003']:
    TURBULENCE_SONICS_KEYS[f'{model}-LM2009'] = dict(
        features = TURBULENCE_SONICS_KEYS[model]['features'] + ['transition_menter'],
        parameters = TURBULENCE_SONICS_KEYS[model]['parameters'],
    )


def apply_to_solver(workflow):

    import miles

    fluid_features, fluid_parameters = get_fluid_template(workflow.Fluid)
    turb_features, turb_parameters = get_turbulence_template(workflow.Turbulence)
    flux_features, flux_parameters = get_spatial_fluxes_template(workflow.Numerics)
    time_features, time_parameters = get_time_marching_template(workflow.Numerics)

    my_config = miles.solver.config.Configuration(workflow.tree)
    my_config.update(
        "motion/mobile",
        *fluid_features,
        *turb_features,
        *flux_features,
        *time_features,
    )
    my_config.set(
        **fluid_parameters,
        **turb_parameters, 
        **flux_parameters, 
        **time_parameters,
    )
    user_given_parameters = update_config_with_user_parameters(my_config, workflow)

    configuration = my_config.apply()
    configuration.update(
        dict(
            output_folder = names.DIRECTORY_LOG,
            niter = workflow.Numerics['NumberOfIterations'],
        )
    )

    del configuration['configuration']
    del configuration['hpc_conf'] 

    workflow.SolverParameters['configuration'] = nested_dict_from_keys(configuration)
    workflow.tree = cgns.castNode(workflow.tree)
    deep_update(workflow.SolverParameters, user_given_parameters) 

def update_config_with_user_parameters(my_config, workflow):
    if 'features' in workflow.SolverParameters:
        features = workflow.SolverParameters.pop('features')
        workflow.tree.getAtPath(
            Path=f'CGNSTree/{workflow._workflow_parameters_container_}/SolverParameters/features'
            ).remove()
        my_config.update(*features)
    if 'parameters' in workflow.SolverParameters:
        parameters = workflow.SolverParameters.pop('parameters')
        workflow.tree.getAtPath(
            Path=f'CGNSTree/{workflow._workflow_parameters_container_}/SolverParameters/parameters'
            ).remove()
        my_config.set(**parameters)
    user_given_parameters = copy.copy(workflow.SolverParameters) 
    return user_given_parameters

def get_spatial_fluxes_template(Numerics):
    scheme = Numerics['Scheme']
    if Numerics['Scheme'] != 'Roe':
        mola_logger.warning(f'sonics Scheme={scheme} not implemented, using Roe instead')
    Numerics['Scheme'] = 'Roe'

    features = []
    parameters = dict()

    # Convective flux 
    if Numerics['Scheme'] == 'Roe':
        features = [
            "roe",
            "upwind_order:2",
            "upwind_limiter_vanalbada",
        ]
    else:
        raise MolaException(f"Scheme={Numerics['Scheme']} is not available for solver sonics")

    features.append("viscous_flux/vf5p_cor") 
    features.append("grad_scheme/green_gauss") 

    parameters['pctrad'] = 0.01
    
    return features, parameters

def get_time_marching_template(Numerics):
    features = [
        "time_algo/steady",
        "ode/implicit",
        "time_step/spectral",
    ]

    parameters = dict()

    if Numerics['TimeMarching'] != 'Steady':
        raise MolaException(f"Only Steady simulations are implemented yet for soNICS with MOLA")
    
    # CFL setting
    if isinstance(Numerics['CFL'], float):
        parameters['CFL'] = Numerics['CFL']
    else:
        parameters['CFL'] = Numerics['CFL']['EndValue']

    return features, parameters

def get_turbulence_template(Turbulence):

    try:
        turb_dict = TURBULENCE_SONICS_KEYS[Turbulence['Model']]
        features = turb_dict['features']
        parameters = turb_dict.get('parameters', dict())
    except:
        raise MolaException(f"Scheme={Turbulence['Model']} is not available for solver sonics")
    
    parameters['cutvars'] = get_turbulence_cutoff_setup(Turbulence)

    return features, parameters

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

def get_fluid_template(Fluid):
    features = ['viscosity']
    parameters = dict()

    translate_to_miles = dict(
        Gamma = 'SpecificHeatRatio',
        cv = 'SpecificHeatVolume',
        cp = 'SpecificHeatPressure',
        SutherlandViscosity = 'ViscosityMolecularReference',
        SutherlandTemperature = 'TemperatureReference',
        SutherlandConstant = 'SutherlandLawConstant',
    )
    for key, value in Fluid.items():
        if key in translate_to_miles:
            key = translate_to_miles[key]
        parameters[key] = value
    return features, parameters
        
def get_cfl_function(cfl):
    if isinstance(cfl, dict):
        if cfl['EndIteration'] <= cfl['StartIteration'] \
            or cfl['EndValue'] <= cfl['StartValue']:
            CFLfunction = lambda iteration: cfl
        else:
            a = (cfl['EndValue']-cfl['StartValue']) / (cfl['EndIteration']-cfl['StartIteration'])
            linear_ramp = lambda iteration: cfl['StartValue'] + a * (iteration - cfl['StartIteration'])
            CFLfunction = lambda iteration: min(linear_ramp(iteration), cfl['EndValue'])
    else:
        CFLfunction = lambda iteration: cfl
    return CFLfunction

def nested_dict_from_keys(d):
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            # If the value is a dictionary, apply the function recursively
            result[key] = nested_dict_from_keys(value)
        elif '/' in key:
            # If the key contains a '/', transform it into a nested dictionary
            keys = key.split('/')
            temp_dict = result
            for k in keys[:-1]:  # Traverse all keys except the last one
                temp_dict = temp_dict.setdefault(k, {})
            temp_dict[keys[-1]] = value  # Set the value for the last key
        else:
            # Otherwise, keep the key-value pair as is
            result[key] = value
    return result

