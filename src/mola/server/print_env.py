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

    machine = os.getenv('MAC', 'UNKNOWN')
    archi = os.getenv('ARCH', '-')
    mola_version = __version__
    if Version(mola_version).is_devrelease:
        mola_version = YELLOW + mola_version + ENDC
    print(f"MOLA version {mola_version} at {machine} ({archi}")

    vpython = sys.version_info
    print(f' --> Python {vpython.major}.{vpython.minor}.{vpython.micro}')
    
    print_module_version('treelab')
    print_module_version('KCore', 'Cassiopee')
    print_module_version('maia')
    print_module_version('VULCAINS')
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
    print(tag+v.ljust(20-len(tag))+printTime(toc))

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
        print(v.ljust(20-len(tag))+printTime(toc))

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

    def getMajorMinorMicro(version_string):
        version_string = version_string.replace('v','')
        MajorMinorMicro = version_string.split('.')
        try:
            Major, Minor, Micro = MajorMinorMicro
        except ValueError:
            Major, Minor = MajorMinorMicro
            Micro = '0'
        return int(Major), int(Minor), int(Micro)

    def gatherMOLAversions():
        ALL_MOLAS_DIR = os.path.sep+os.path.join(*__MOLA_PATH__.split(os.path.sep)[:-1])+os.path.sep
        ALL_MOLAS_VER = [v.replace(ALL_MOLAS_DIR,'') for v in glob.glob(os.path.join(ALL_MOLAS_DIR,'*'))]
        v = {}
        for ver in ALL_MOLAS_VER:
            if not ver.startswith('v'): continue
            M, m, n = getMajorMinorMicro(ver)
            ver_format = Version(ver)
            M, m, n = ver_format.major, ver_format.minor, ver_format.micro
            ver.major, ver.minor, ver.micro
            if M not in v:
                v.update({M:{m:[n]}})
            elif m not in v[M]:
                v[M][m] = [n]
            else:
                v[M][m].append(n)

        return v

    mola_version = Version(__version__)
    if mola_version.is_devrelease:
        print(YELLOW+'WARNING: you are using an UNSTABLE version of MOLA.\nConsider using a stable version.'+ENDC)
    else:
        AllVersions = gatherMOLAversions()
        most_updated_version = Version('0.0.0')
        for v in AllVersions:
            v = Version(v)
            if mola_version.major != v.major:
                continue
            if mola_version < v:
                if most_updated_version < v:
                    most_updated_version = v

        if most_updated_version > Version('0.0.0'):
            print(YELLOW+f'WARNING: a most updated version exist: {most_updated_version}'+ENDC)
        else:
            print(GREEN+'You are using the latest version of MOLA'+ENDC)

def printTime(toc):
    ElapsedTime = tic() - toc
    if ElapsedTime < 0.1: return ''
    if ElapsedTime < 0.5: return ' (took %g s)'%ElapsedTime
    if ElapsedTime < 1.0: return YELLOW+' (took %g s)'%ElapsedTime+ENDC
    return RED+' (took %g s : too long)'%ElapsedTime+ENDC