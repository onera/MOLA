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

# ----------------------- IMPORT SYSTEM MODULES ----------------------- #
import numpy as np
from mpi4py import MPI
comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
NumberOfProcessors = comm.Get_size()

from treelab import cgns
import mola.naming_conventions as names
from mola.cfd.compute.read_cfd_files import read_cfd_files
from mola.cfd.preprocess.motion.motion import any_mobile

def apply_to_solver(workflow):

    import FastS.PyTree as FastS
    from mola.cfd.coprocess.manager import CoprocessManager

    read_cfd_files.apply(workflow)
    workflow._coprocess_manager = CoprocessManager(workflow)

    inititer, niter = get_range_of_iterations(workflow)

    # time-marching loop
    for it in range( inititer-1, inititer+niter-1 ):
    
        # Numbering in MOLA starts at iteration 1, and starts at 0 for Fast
        workflow._iteration = it
        workflow._status = 'RUNNING_BEFORE_ITERATION'
        workflow._coprocess_manager.run_iteration()

        # LB: isn't it done through infos_ale at warmup call ?
        # if any_mobile(workflow.Motion):
        #     apply_motion(workflow)

        FastS._compute(workflow.tree,
                       workflow._fast_metrics,
                       it,
                       workflow._treeAtCenters,
                       workflow._fast_graph)

        FastS.display_temporal_criteria(workflow.tree, workflow._fast_metrics, it, format='store') 
        FastS._calc_global_convergence(workflow.tree) # should work now: https://github.com/onera/Fast/issues/14

        workflow._iteration = it + 1  # we are after the method Fast._compute

        # TODO : split run_iteration in two ?
        # workflow._iteration = it
        # workflow._status = 'RUNNING_AFTER_ITERATION'
        # workflow._coprocess_manager.run_iteration()

        if workflow._coprocess_manager.status == 'TO_FINALIZE': break

    workflow.tree = cgns.castNode(workflow.tree)
    workflow._coprocess_manager.finalize()
    del workflow._coprocess_manager
    

def get_range_of_iterations(workflow):
    inititer = workflow.Numerics['IterationAtInitialState']
    niter = workflow.Numerics['NumberOfIterations']

    return inititer, niter

def apply_motion(workflow):
    import FastC.PyTree as FastC
    theta, omega = get_theta_and_omega(workflow, workflow._iteration)
    FastC._motionlaw(workflow.tree, theta, omega)

def get_theta_and_omega(workflow, iteration):
    from mola.cfd.preprocess.motion.motion import (get_first_found_rotation_speed_vector_at_motion,
                                                   get_rotation_axis_from_first_found_motion)

    rotation_vector = get_rotation_axis_from_first_found_motion(workflow.Motion)
    omega_vector = get_first_found_rotation_speed_vector_at_motion(workflow.Motion)
    omega = np.linalg.norm(omega_vector)
    time = workflow.Numerics['TimeAtInitialState'] + (iteration - workflow.Numerics['IterationAtInitialState']) * workflow.Numerics['TimeStep']
    # Motion is applied at iteration n + 1/2
    time_ale = time + 0.5*workflow.Numerics['TimeStep']
    theta = omega * time_ale

    return theta, omega
