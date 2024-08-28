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

from pprint import pprint
import mola.naming_conventions as names
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

    # TurbulenceSetup = TURBULENCE_SONICS_KEYS[workflow.Turbulence['Model']]
    # TurbulenceSetup['cutvars'] = get_turbulence_cutoff_setup(workflow.Turbulence)

    # TODO take into account a dict for the CFL
    # For now, it must be a float
    if not isinstance(workflow.Numerics['CFL'], float):
        raise MolaException(f"CFL must be a float (now it is a {type(workflow.Numerics['CFL'])})")
    
    my_config = miles.solver.config.Configuration(workflow.tree)
    # print(my_config.get_feature_diagram())
    my_config.update(
        "motion/mobile",
        *get_turbulence_template(workflow.Turbulence)[0],
        *get_spatial_fluxes_template(workflow.Numerics)[0],
        *get_time_marching_template(workflow.Numerics)[0],
    )
    my_config.set(
        # TurbulenceModel = TurbulenceSetup,
        CFL = workflow.Numerics['CFL'],
        pctrad = 0.01,
        cutvars = get_turbulence_cutoff_setup(workflow.Turbulence),
    )
    
    update_fluid_model(my_config, workflow.Fluid)

    configuration = my_config.apply()

    # pprint( my_config.get_active_features())
    # pprint(my_config.parameters)
    # pprint(my_config.default_parameters)
    # pprint(configuration)

    configuration.update(
        dict(
            output_folder = names.DIRECTORY_LOG,
            niter = workflow.Numerics['NumberOfIterations'],
            # niter_period = 1,
            # extracts = {'*': ['conservatives', 'LaminarViscosity', 'TurbulentViscosity','TurbulentViscosity', 'TurbulentDistance', 'Mach', 'primitives']},
            # code_generation = "none",
            CFL = workflow.Numerics['CFL'],
            # fcfl = lambda iteration: workflow.Numerics['CFL'],
        )
    )

    # convert to dict to be able to write in cgns tree with treelab
    # configuration['conf']  = configuration['conf'].to_dict(configuration['conf'])
    del configuration['hpc_conf'] 

    workflow.SolverParameters['configuration'] = nested_dict_from_keys(configuration)
    # pprint(workflow.SolverParameters['configuration'])

    workflow.tree = cgns.castNode(workflow.tree)

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

    features.append("viscous_flux/vf5p_cor") # shouldn't it be optional ???
    # features.append("grad_scheme/green_gauss") # shouldn't it be optional ???
    
    return features, parameters

def get_time_marching_template(Numerics):
    features = [
        "time_algo/steady",
        "ode/explicit",
        "time_step/spectral",
    ]

    parameters = dict()

    if Numerics['TimeMarching'] != 'Steady':
        raise MolaException(f"Only Steady simulations are implemented yet for soNICS with MOLA")

    return features, parameters

def get_turbulence_template(Turbulence):

    features = []
    parameters = dict()

    if Turbulence['Model'] == 'SA':
        features = ['spalart_standard']
    else:
        raise MolaException(f"Scheme={Turbulence['Model']} is not available for solver sonics")

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

def update_fluid_model(config, Fluid):
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
        config.set(**{key: value})
        

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
