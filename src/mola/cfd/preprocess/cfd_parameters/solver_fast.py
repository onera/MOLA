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

from mola.logging import mola_logger, MolaException, MolaUserError


from treelab import cgns
from mola.cfd.preprocess.cfd_parameters import cfd_parameters

keys_to_store_in_bases = [
    "temporal_scheme",
    "ss_iteration",
    "modulo_verif", ]


keys_to_store_in_zones = [
    "time_step",
    "scheme",
    "time_step_nature",
    "ssdom_IJK",
    "psiroe",
    "cfl",
    "nb_relax",
    "epsi_newton", ]

def apply_to_solver(workflow):

    set_numerics(workflow)
    

def set_numerics(workflow):

    workflow.SolverParameters['numerics'] = dict(
        **get_spatial_fluxes(workflow.Numerics),
        **get_time_marching_setup(workflow.Numerics),
    )
    put_numerics_in_tree(workflow.SolverParameters['numerics'], workflow.tree)
    workflow.tree = cgns.castNode(workflow.tree)

def get_spatial_fluxes(Numerics):

    if Numerics['Scheme'] == 'Jameson':
        mola_logger.warning("Jameson scheme not implemented in Fast. Switching to Roe.")
        Numerics['Scheme'] = "Roe"

    # Convective flux 
    if Numerics['Scheme'] == 'ausm+':
        SchemeSetup = dict(
        scheme             = "ausmpred", # "ausmpred", "roe_min", "senseur"
        slope              = "o3",
        )
    elif Numerics['Scheme'] == 'Roe':
        SchemeSetup = dict(
        scheme = 'roe_min',
        psiroe = 0.01,
        slope = "o3", # "minmod" or "o3"
        )
    else:
        raise MolaUserError(f'Numerical scheme {Numerics["Scheme"]} not recognized for the solver fast')
    
    return SchemeSetup

def get_time_marching_setup(Numerics):
    TimeMarchingSetup = dict(
        ssdom_IJK=[10000,10000,10000],
        epsi_newton=0.01, 
        nb_relax=1, # newton
        ss_iteration=5,
    )

    if Numerics['TimeMarching'] == 'Steady':

        TimeMarchingSetup.update({
            "temporal_scheme": "implicit", # or "explicit"
            "time_step_nature": "local",
            "ss_iteration":1,
            "time_step": 1e-6, # must exist even in steady
            "modulo_verif":10,
        })

        TimeMarchingSetup.update(get_cfl_setup(Numerics['CFL']))

    else:

        TimeMarchingSetup.update(dict(
            time_step          = Numerics['TimeStep'],
            time_step_nature   = "global",
            temporal_scheme    = "implicit",
        ))

        # TODO include 1st or 2nd order time marching ?

    return TimeMarchingSetup

def put_numerics_in_tree(fast_numerics, tree):
    import Fast.PyTree as Fast # TODO setParameters ?
    num_base = dict()
    for k_base in keys_to_store_in_bases:
        if k_base in fast_numerics:
            num_base[k_base] = fast_numerics[k_base]

    num_zone = dict()
    for kzone in keys_to_store_in_zones:
        if kzone in fast_numerics:
            num_zone[kzone] = fast_numerics[kzone]

    for base in tree.bases():
        Fast._setNum2Base(base, num_base) # TODO setParameters ?
        Fast._setNum2Zones(base, num_zone) # TODO setParameters ?
    tree.setParameters('.Solver#define',**num_base) 
    tree = cgns.castNode(tree)


def get_cfl_setup(cfl):
    if isinstance(cfl, dict):
        CFLSetup = dict(cfl=cfl['EndValue'])
    else:
        CFLSetup = dict(cfl=cfl)
    return CFLSetup
