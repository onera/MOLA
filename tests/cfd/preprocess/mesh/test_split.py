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

import pytest
import numpy as np
from treelab import cgns

from mola.cfd.preprocess.mesh import split

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_splitAndDistributeUsingNPartsAndNProcsWithCassiopee():

    class FakeWorkflow():
        def __init__(self) -> None:
            self.tree = get_cart_block(10,10,10)

            self.RawMeshComponents = [dict(Name='Base')]

            self.SplittingAndDistribution = dict(
                Splitter='cassiopee',
                Distributor='cassiopee',
                CoresPerNode=1,
                ComponentsToSplit='all'
            )

    w = FakeWorkflow()
    nb_parts = 2
    nb_aimed_procs = 2
    split._splitAndDistributeUsingNPartsAndNProcsWithCassiopee(
        w, nb_parts, nb_aimed_procs, raise_error=True)
    
@pytest.mark.unit
@pytest.mark.cost_level_0
def test_distribute_with_cassiopee():
    tree = get_cart_block(10,10,10)
    nb_aimed_procs = 1
    cores_per_node = 1
    split._distribute_with_cassiopee(tree, nb_aimed_procs, cores_per_node)

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_splitAndDistributeUsingNPartsAndNProcsWithCassiopee_only_distribute():

    class FakeWorkflow():
        def __init__(self) -> None:
            self.tree = get_cart_block(10,10,10)

            self.RawMeshComponents = [dict(Name='Base')]

            self.SplittingAndDistribution = dict(
                Splitter='cassiopee',
                Distributor='cassiopee',
                CoresPerNode=1,
                ComponentsToSplit=[]
            )

    w = FakeWorkflow()
    nb_parts = 2  # won't be used
    nb_aimed_procs = 1
    split._splitAndDistributeUsingNPartsAndNProcsWithCassiopee(
        w, nb_parts, nb_aimed_procs, raise_error=True)


# --------------------------- fixtures and helpers --------------------------- #
def get_cart_block(ni, nj, nk):
    x, y, z = np.meshgrid( np.linspace(0,1,ni),
                           np.linspace(0,1,nj),
                           np.linspace(0,1,nk), indexing='ij')
    block = cgns.newZoneFromArrays( 'block', ['x','y','z'], [ x,  y,  z ])
    tree = cgns.Tree(Base=block)

    return tree