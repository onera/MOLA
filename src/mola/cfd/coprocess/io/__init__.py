#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute self.iteration and/or modify
#    self.iteration under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that self.iteration will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

import shutil
import mola.naming_conventions as names
from mola.logging import GREEN, CYAN, ENDC
from .. import mola_logger, rank, comm
from .cassiopee import save_with_cassiopee, load_skeleton
from .pypart import save_with_pypart

def save(t, filename, coprocess_manager=None, tagWithIteration=False):
    '''
    Generic function to save a PyTree **t** in parallel. Works whatever the
    dimension of the PyTree. Use it to save ``'fields.cgns'``,
    ``'surfaces.cgns'`` or ``'arrays.cgns'``.

    .. important::
        If the mesh was split with PyPart and if the function is called to save
        *FILE_FIELDS*, the tree is automatically merged and saved using PyPart.
        In that case, the variable **PyPartBase** should be defined (normally,
        in ``compute.py``)

    Parameters
    ----------

        t : PyTree
            tree to save

        filename : str
            Name of the file

        tagWithIteration : bool
            if :py:obj:`True`, adds a suffix ``_AfterIter<iteration>``
            to the saved filename (creates a copy)
    '''
    mola_logger.info(f'{CYAN}saving {filename}...{ENDC}', rank=0)

    # is_PyPart_used = coprocess_manager is not None and coprocess_manager.workflow.SplittingAndDistribution['Splitter'].lower() == 'pypart'
    # is_3d_field_to_save = filename.endswith(names.FILE_OUTPUT_3D) or filename.endswith(names.FILE_OUTPUT_RESTART)
    
    # if  is_PyPart_used and is_3d_field_to_save:
    #     save_with_pypart(t, filename, coprocess_manager.PyPartBase)
    # else:
    #     save_with_cassiopee(t, filename)

    from mola.cfd.preprocess.mesh.io.writer import write
    write(coprocess_manager.workflow, t, filename)
    
    mola_logger.info(f'{GREEN}saving {filename}... OK{ENDC}', rank=0)

    if coprocess_manager is not None:
        if tagWithIteration and rank == 0: 
            copyOutputFiles(coprocess_manager.iteration, filename)
    comm.barrier()

def copyOutputFiles(iteration, *files2copy):
    '''
    Copy the files provided as input *(comma-separated variables)* by addding to
    their name ``'_AfterIter<X>'`` where ``<X>`` will be replaced with the
    corresponding interation

    Parameters
    ----------

        iteration : int
            current iteration

        file2copy : comma-separated :py:class:`str`
            file(s) name(s) to copy at ``OUTPUT`` directory.

    Examples
    --------

    ::

        copyOutputFiles('surfaces.cgns','arrays.cgns')

    '''
    for file2copy in files2copy:
        f2cSplit = file2copy.split('.')
        name = '.'.join(f2cSplit[:-1])
        fmt = f2cSplit[-1]
        newFileName = f'{name}_AfterIter{iteration}.{fmt}'
        try:
            shutil.copy2(file2copy, newFileName)
        except:
            pass

