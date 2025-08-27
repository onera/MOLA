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

import sys
import os
import glob
from timeit import default_timer as tic
import importlib
from packaging.version import Version

from mola import __version__, __MOLA_PATH__
from mola.logging import RED, GREEN, YELLOW, ENDC

def print_environment():

    machine = os.getenv('MOLA_MACHINE', 'UNKNOWN')
    archi = os.getenv('ARCH', '-')
    mola_version = __version__
    if Version(mola_version).is_devrelease:
        mola_version = YELLOW + mola_version + ENDC
    print(f"MOLA version {mola_version} at {machine} ({archi})")

    vpython = sys.version_info
    print(f' --> Python {vpython.major}.{vpython.minor}.{vpython.micro}')
    
    print_module_version('treelab')
    print_module_version('KCore', 'Cassiopee')
    print_module_version('maia')
    # print_module_version('VULCAINS')
    print_module_version('turbo')
    print_module_version('Ersatz')

    print_solver_version()
    print_status_on_mola_version()

def print_module_version(module_name, printed_name=None):
    toc = tic()
    try:
        module = importlib.import_module(module_name)
        v = module.__version__
    except:
        v = RED + 'UNAVAILABLE' + ENDC 
    if not printed_name:
        printed_name = module_name
    tag = f' --> {printed_name} '
    print(tag+v.ljust(20-len(tag))+print_time(toc))

def print_solver_version():
    solver = os.getenv('MOLA_SOLVER', 'UNKNOWN')

    if solver == 'elsa':
        vELSA = os.getenv('ELSAVERSION', 'UNAVAILABLE')
        if vELSA == 'UNAVAILABLE':
            vELSA = RED + vELSA + ENDC
        print(' --> elsA '+vELSA)

        # elsA tools chain
        tag = '     with ETC '
        print(tag,end='')
        toc = tic()
        try:
            import etc
        except:
            v = RED + 'UNAVAILABLE' + ENDC
        else:
            v = YELLOW + 'UNKNOWN' + ENDC
            for vatt in ('__version__', 'version'):
                if hasattr(etc, vatt):
                    v = getattr(etc,vatt)
                    break
        print(v.ljust(20-len(tag))+print_time(toc))

    elif solver == 'sonics':
        vSONICS = os.getenv('SONICSVERSION', 'UNAVAILABLE')
        if vSONICS == 'UNAVAILABLE': 
            vSONICS = RED + vSONICS + ENDC
        print(' --> SoNICS '+vSONICS)
        print_module_version('miles')

    elif solver == 'fast':
        print_module_version('Fast')

    else:
        print(YELLOW+'WARNING: unknown solver'+ENDC)    

def print_status_on_mola_version():

    def gather_mola_versions():
        ALL_MOLAS_DIR = os.path.sep+os.path.join(*__MOLA_PATH__.split(os.path.sep)[:-1])+os.path.sep
        ALL_MOLAS_VER = [v.replace(ALL_MOLAS_DIR,'') for v in glob.glob(os.path.join(ALL_MOLAS_DIR,'*'))]
        versions = []
        for ver in ALL_MOLAS_VER:
            if not ver.startswith('v'): 
                continue
            versions.append(Version(ver))
        return versions

    mola_version = Version(__version__)
    if mola_version.is_devrelease:
        print(YELLOW+'WARNING: you are using an UNSTABLE version of MOLA.\nConsider using a stable version.'+ENDC)
    else:
        AllVersions = gather_mola_versions()
        most_updated_version = Version('0.0.0')
        for v in AllVersions:
            if mola_version.major != v.major:
                continue
            if mola_version < v:
                if most_updated_version < v:
                    most_updated_version = v

        if most_updated_version > Version('0.0.0'):
            print(YELLOW+f'WARNING: a most updated version exist: {most_updated_version}'+ENDC)
        else:
            print(GREEN+'You are using the latest version of MOLA'+ENDC)

def print_time(toc):
    ElapsedTime = tic() - toc
    if ElapsedTime < 0.1: 
        return ''
    elif ElapsedTime < 0.5: 
        return ' (took %g s)'%ElapsedTime
    elif ElapsedTime < 1.0: 
        return YELLOW+' (took %g s)'%ElapsedTime+ENDC
    else:
        return RED+' (took %g s : too long)'%ElapsedTime+ENDC