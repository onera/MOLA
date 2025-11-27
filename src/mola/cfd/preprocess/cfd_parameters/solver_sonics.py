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
from mola.cfd.preprocess.motion.motion import all_families_are_fixed

TURBULENCE_SONICS_KEYS = {

    'SA': dict(
        features = ['spalart_standard'],  # in fact it is this model: https://doi.org/10.1016/S1270-9638(02)01148-3
        parameters = dict(),
    ),

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
        features = ['kl_smith', 'k_prod/from_sij'],
        parameters = dict(k_prod_limiter=20.),
    ),

    'smith-V': dict(
        features = ['kl_smith', 'k_prod/from_vorticity'],
        parameters = dict(k_prod_limiter=20.),
    ),
}

for model in ['SST-2003', 'SST-V2003']:
    TURBULENCE_SONICS_KEYS[f'{model}-LM2009'] = dict(
        features = TURBULENCE_SONICS_KEYS[model]['features'] + ['transition_menter'],
        parameters = TURBULENCE_SONICS_KEYS[model]['parameters'],
    )


def apply_to_solver(workflow):

    my_config = get_sonics_config(workflow)
    my_config.to_cgns_base(workflow.tree)

    workflow.tree = cgns.castNode(workflow.tree)

def get_sonics_config(workflow):

    import miles

    fluid_features, fluid_parameters = get_fluid_template(workflow.Fluid, workflow.Turbulence['Model'])
    turb_features, turb_parameters = get_turbulence_template(workflow.Turbulence)
    flux_features, flux_parameters = get_spatial_fluxes_template(workflow.Numerics, workflow.Turbulence['Model'])
    time_features, time_parameters = get_time_marching_template(workflow.Numerics)

    my_config = miles.Configuration()

    if all_families_are_fixed(workflow):
        my_config.update("motion/fixed")
    else:
        my_config.update("motion/mobile")
        
    my_config.update(
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
    if workflow.ProblemDimension == 2:
        my_config.set(use_cache_blocking=False)  # HACK Segmentation fault if not
        
    update_config_with_user_parameters(my_config, workflow)

    return my_config

def update_config_with_user_parameters(my_config, workflow):
    if 'features' in workflow.SolverParameters:
        features = workflow.SolverParameters.pop('features')
        if not isinstance(features, list): 
            features = [features]  # Force type as list, if SolverParameters has been read from a file (workflow.cgns)
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

def get_spatial_fluxes_template(Numerics, TurbulenceModel):
    scheme = Numerics['Scheme']
    if Numerics['Scheme'] != 'Roe':
        mola_logger.user_warning(f'sonics Scheme={scheme} not implemented, using Roe instead')
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

    if TurbulenceModel != 'Euler':
        features.append("viscous_flux/vf5p_cor") 
    features.append("grad_scheme/green_gauss") 

    parameters['pctrad'] = 0.01
    
    return features, parameters

def get_time_marching_template(Numerics):
    features = [
        "time_algo/steady",
        "ode/implicit",
        # "time_step/spectral",
    ]

    parameters = dict()

    if Numerics['TimeMarching'] != 'Steady':
        raise MolaException(f"Only Steady simulations are implemented yet for soNICS with MOLA")
    
    # # CFL setting
    # if isinstance(Numerics['CFL'], float):
    #     parameters['CFL'] = Numerics['CFL']
    # else:
    #     parameters['CFL'] = Numerics['CFL']['EndValue']

    return features, parameters

def get_turbulence_template(Turbulence):

    if Turbulence['Model'] == 'Euler':
        features = ['euler']
        parameters = dict()

    elif Turbulence['Model'] == 'Laminar':
        features = ['nslam']
        parameters = dict()
    
    else:
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

def get_fluid_template(Fluid, TurbulenceModel):
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
    
    if TurbulenceModel == 'Euler':
        try: parameters.pop('Prandtl') 
        except: pass
    if TurbulenceModel in ['DNS', 'ILES', 'Laminar', 'Euler']:
        try: parameters.pop('PrandtlTurbulent') 
        except: pass

    return features, parameters
        
def get_cfl_function(cfl):
    # NOTE: Careful, for miles, the argument in the lambda function must be named "it"
    if isinstance(cfl, dict):
        if cfl['EndIteration'] <= cfl['StartIteration'] \
            or cfl['EndValue'] <= cfl['StartValue']:
            CFLfunction = lambda it: cfl
        else:
            a = (cfl['EndValue']-cfl['StartValue']) / (cfl['EndIteration']-cfl['StartIteration'])
            linear_ramp = lambda it: cfl['StartValue'] + a * (it - cfl['StartIteration'])
            CFLfunction = lambda it: min(linear_ramp(it), cfl['EndValue'])
    else:
        CFLfunction = lambda it: cfl
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

