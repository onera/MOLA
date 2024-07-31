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

from mola.logging import mola_logger, MolaUserError

from mola.cfd.preprocess.cfd_parameters.cfd_parameters import deep_update

keys_to_store_in_bases = [
    "temporal_scheme",
    "ss_iteration",
    "modulo_verif", ]


keys_to_store_in_zones = [
    'scheme',
    'slope',
    'senseurType',
    'coef_hyper',
    'motion',
    'time_step',
    'time_step_nature',
    'epsi_newton',
    'inj1_newton_tol',
    'inj1_newton_nit',
    'psiroe',
    'cfl',
    'model',
    'prandtltb',
    'ransmodel',
    'DES',
    'DES_debug',
    'sgsmodel',
    'extract_res',
    'source',
    'ratiom',
    
    # not documented:
    "ssdom_IJK",
    "nb_relax",
    ]


def apply_to_solver(workflow):

    # https://fast.onera.fr/Fast.html#Fast.PyTree.setNum2Base
    # https://fast.onera.fr/Fast.html#Fast.PyTree.setNum2Zones

    set_model(workflow) 
    set_numerics(workflow)
    

def set_model(workflow):
    deep_update( workflow.SolverParameters, get_fluid_setup(workflow.Fluid) )
    deep_update( workflow.SolverParameters, get_turbulence_setup(workflow.Turbulence) )


def set_numerics(workflow):

    deep_update( workflow.SolverParameters, get_spatial_fluxes(workflow.Numerics) )
    deep_update( workflow.SolverParameters, get_time_marching_setup(workflow.Numerics) )


def get_fluid_setup( Fluid : dict ) -> dict:
    
    Parameters = dict(Num2Base={}, 
                      Num2Zones={'prandtltb': Fluid['PrandtlTurbulent']})
    
    return Parameters

def get_turbulence_setup( Turbulence : dict ) -> dict:
    
    Parameters = dict(Num2Base={}, Num2Zones={})
    
    requested_model = Turbulence['Model']
    if requested_model in ['LES', 'ILES', 'DNS', 'Laminar']:

        if requested_model=='LES':
            # https://doi.org/10.1002/(SICI)1097-0363(20000229)32:4<369::AID-FLD943>3.0.CO;2-6
            Parameters['Num2Zones']['sgsmodel'] = 'smsm'


    elif requested_model.startswith('ZDES'):
        zdes_mode = requested_model.split('-')[1]
        Parameters['Num2Zones']['DES'] = 'zdes'+zdes_mode
    
    else: # RANS modeling
        if requested_model != 'SA':
            mola_logger.warning("RANS model %s not implemented in Fast. Switching to 'SA'"%requested_model)
            Turbulence['Model'] = 'SA' 
        
        Parameters['Num2Zones']['ransmodel'] = 'SA'
        Parameters['Num2Zones']['ratiom'] = 1e4 

    return Parameters

def get_spatial_fluxes(Numerics : dict):

    Parameters = dict(Num2Base={}, Num2Zones={})

    if Numerics['Scheme'] == 'Jameson':
        mola_logger.warning("Jameson scheme not implemented in Fast. Switching to Roe.")
        Numerics['Scheme'] = "Roe"

    # Convective flux 
    if Numerics['Scheme'] == 'ausm+':
        Parameters['Num2Zones'].update( dict(
        scheme             = "ausmpred", # "ausmpred", "roe_min", "senseur"
        slope              = "o3",
        ))
    elif Numerics['Scheme'] == 'Roe':
        Parameters['Num2Zones'].update( dict(
        scheme = 'roe_min',
        psiroe = 0.01,
        slope = "o3", # "minmod" or "o3"
        ))
    else:
        raise MolaUserError(f'Numerical scheme {Numerics["Scheme"]} not recognized for the solver fast')
    
    return Parameters


def get_time_marching_setup(Numerics):

    Parameters = dict(
        Num2Base =dict(ss_iteration=5,
                       temporal_scheme = "implicit"),
    
        Num2Zones=dict( # ssdom_IJK=[10000,10000,10000], # TODO investigate this key
                       nb_relax=1, # newton
                       epsi_newton=0.01
                       ))

    if Numerics['TimeMarching'] == 'Steady':
        
        Parameters['Num2Base'].update(dict(
            ss_iteration=1,
            modulo_verif=10,
        ))
        
        Parameters['Num2Zones'].update(dict(
            time_step_nature = "local",
            time_step = 1e-6, # must exist even if ignored ?
        ))
        
        Parameters['Num2Zones'].update(get_cfl_setup(Numerics['CFL']))

    else:

        Parameters['Num2Zones'].update(dict(
            time_step          = Numerics['TimeStep'],
            time_step_nature   = "global",
        ))

        # TODO include 1st or 2nd order time marching ?

    return Parameters


def get_cfl_setup(cfl):
    if isinstance(cfl, dict):
        CFLSetup = dict(cfl=cfl['EndValue'])
    else:
        CFLSetup = dict(cfl=cfl)
    return CFLSetup
