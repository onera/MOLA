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

'''
Use this script to apply turbo postprocessing to all COMPLETED cases  under the current directory.
Here is a typical application case. Something went wrong during the postprocessing of a isospeedline, 
launched with the function `launchIsoSpeedLines`.
You may place the current script in the root directory, which has the following sub-directories and files: 
.
├── DISPATCHER
│   ├── dispatcherJob.sh
│   ├── Dispatch-err.log
│   ├── Dispatch-out.log
│   ├── dispatch.py
│   ├── ...
└── rotor37_isospeed0
    ├── 20266.00_rotor37_isospeed0
    ├── 25332.50_rotor37_isospeed0
    ├── 30399.00_rotor37_isospeed0
    ├── 35465.50_rotor37_isospeed0
    ├── 40532.00_rotor37_isospeed0
    ├── 45598.50_rotor37_isospeed0
    ├── 50665.00_rotor37_isospeed0
    ├── 55731.50_rotor37_isospeed0
    ├── 60798.00_rotor37_isospeed0
    ├── 65864.50_rotor37_isospeed0
    ├── error.545620.log
    ├── job.sh
    └── output.545620.log

Executing the current script with python will found all the cases (here the *_rotor37_isospeed0 directories)
where there is a file named COMPLETED. For each of these cases, the postprocessing with turbo will be performed
and the files surfaces.cgns and arrays.cgns will be updated.    
'''
import os

import MOLA.InternalShortcuts as J
import MOLA.Coprocess as CO

def postprocess_one_case(path, SURFACES_EXTRACTION=False):

    previous_path = os.getcwd()
    os.chdir(path)

    CO.setup = J.load_source('setup', CO.FILE_SETUP)
    arrays = CO.invokeArrays()
    # Write again COMPLETED because it iis removed by CO.invokeArrays
    with open('COMPLETED','w') as f: f.write('COMPLETED')

    if SURFACES_EXTRACTION:
        # Redo surfaces extraction
        t = J.load(os.path.join(CO.DIRECTORY_OUTPUT, CO.FILE_FIELDS))
        surfaces = CO.extractSurfaces(t, CO.setup.Extractions, arrays=arrays)
    else:
        surfaces = J.load(os.path.join(CO.DIRECTORY_OUTPUT, CO.FILE_SURFACES))

    # Do postprocessing
    surfaces = CO._extendSurfacesWithWorkflowQuantities(surfaces, arrays=arrays)

    CO.save(surfaces, os.path.join(CO.DIRECTORY_OUTPUT, CO.FILE_SURFACES))
    arraysTree = CO.arraysDict2PyTree(arrays)
    CO.save(arraysTree, os.path.join(CO.DIRECTORY_OUTPUT,CO.FILE_ARRAYS))

    os.chdir(previous_path)


def recursive_postprocess(current_path='.', SURFACES_EXTRACTION=False):
    for filename in os.listdir(path=current_path):

        path = os.path.join(current_path, filename)
        if not os.path.isdir(path):
            continue

        print(path)

        if os.path.isfile(f'{path}/{CO.FILE_SETUP}') and os.path.isdir(f'{path}/{CO.DIRECTORY_OUTPUT}'):
            # It is a case directory

            if not os.path.isfile(f'{path}/COMPLETED'):
                # Not completed case, we don't postprocess it
                print(J.WARN + '    This case is not completed.' + J.ENDC)
                continue

            print(J.CYAN + '    > postprocessing this case...' + J.ENDC)
            postprocess_one_case(path, SURFACES_EXTRACTION=SURFACES_EXTRACTION)
        
        else:
            # Recursive search from this path
            recursive_postprocess(current_path=path, SURFACES_EXTRACTION=SURFACES_EXTRACTION)


recursive_postprocess()