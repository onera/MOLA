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

from mola import __version__, __MOLA_PATH__
from mola.logging import RED, GREEN, YELLOW, ENDC

def print_environment():

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
            if M not in v:
                v.update({M:{m:[n]}})
            elif m not in v[M]:
                v[M][m] = [n]
            else:
                v[M][m].append(n)

        return v

    def mostUpToDateVersion(AllVersions):
        Major = max(list(AllVersions))
        Minor = max(list(AllVersions[Major]))
        Micro = max(list(AllVersions[Major][Minor]))
        MajorMinorMicro = [str(Major),str(Minor)]
        if Micro > 0: MajorMinorMicro.append( str(Micro) )
        return 'v'+'.'.join(MajorMinorMicro)

    def mostUpToDateMicroVersion(AllVersions, Major, Minor):
        return max(AllVersions[Major][Minor])

    def fullStringOfMostUpToDateMicroVersion(AllVersions, Major, Minor):
        Micro = mostUpToDateMicroVersion(AllVersions, Major, Minor)
        fullString = 'v'+'.'.join([str(Major),str(Minor)])
        if Micro != '0': fullString += '.'+str(Micro)
        return fullString

    def microVersionIsUpToDate(AllVersions, Major, Minor, Micro):
        mostUpToDateMicro = mostUpToDateMicroVersion(AllVersions, Major, Minor)
        return mostUpToDateMicro == Micro

    def usingMostUpToDateVersion(AllVersions, Major, Minor, Micro):
        latestVersion = mostUpToDateVersion(AllVersions)
        MajorMinorMicro = [str(Major),str(Minor)]
        if Micro > 0: MajorMinorMicro.append( str(Micro) )
        usedVersion =  'v'+'.'.join(MajorMinorMicro)
        return usedVersion == latestVersion

    def printTime(toc):
        ElapsedTime = tic() - toc
        if ElapsedTime < 0.1: return ''
        if ElapsedTime < 0.5: return ' (took %g s)'%ElapsedTime
        if ElapsedTime < 1.0: return YELLOW+' (took %g s)'%ElapsedTime+ENDC
        return RED+' (took %g s : too long)'%ElapsedTime+ENDC

    machine = os.getenv('MAC', 'UNKNOWN')
    solver = os.getenv('MOLA_SOLVER', 'UNKNOWN')
    totoV = __version__
    if totoV in ['Dev','master']:
        vMOLA = YELLOW + totoV + ENDC
    else:
        vMOLA = totoV
    print('MOLA version '+vMOLA+' at '+machine+' (%s)'%os.getenv('ARCH', '-'))
    print(' --> Python '+sys.version.split(' ')[0])

    # treelab
    tag = ' --> treelab '
    print(tag,end='')
    toc = tic()
    try:
        import treelab
        v = treelab.__version__
    except:
        v = RED + 'UNAVAILABLE' + ENDC 
    print(v.ljust(20-len(tag))+printTime(toc))

    # Cassiopee
    tag = ' --> Cassiopee ' + os.getenv('OWNCASSREV','') + ' '
    print(tag,end='')
    toc = tic()
    try:
        import KCore as K
        v = K.__version__
    except:
        v = RED + 'UNAVAILABLE' + ENDC
    print(v.ljust(20-len(tag))+printTime(toc))

    # maia
    tag = ' --> maia '
    print(tag,end='')
    toc = tic()
    try:
        import maia
        try:
            v = maia.__version__
        except:
            v = 'dev'
    except:
        v = RED+'UNAVAILABLE'+ENDC
    print(v.ljust(20-len(tag))+printTime(toc))

    # Vortex Particle Method
    tag = ' --> Vulcains (VPM) '
    print(tag,end='')
    toc = tic()
    try:
        from VULCAINS.__init__ import __version__ as v
    except:
        v = RED + 'UNAVAILABLE' + ENDC
    print(v.ljust(20-len(tag))+printTime(toc))

    # turbo
    tag = ' --> turbo '
    print(tag,end='')
    toc = tic()
    try:
        import turbo
        v = turbo.__version__
    except:
        v = RED + 'UNAVAILABLE' + ENDC 
    print(v.ljust(20-len(tag))+printTime(toc))

    # ErsatZ
    tag = ' --> Ersatz '
    print(tag,end='')
    toc = tic()
    try:
        import Ersatz
        v = Ersatz.__version__
    except:
        v = RED + 'UNAVAILABLE' + ENDC 
    print(v.ljust(20-len(tag))+printTime(toc))

    if solver == 'elsa':
        # elsA
        vELSA = os.getenv('ELSAVERSION', 'UNAVAILABLE')
        if vELSA == 'UNAVAILABLE': vELSA = RED + vELSA + ENDC
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
            for vatt in ('__version__','version'):
                if hasattr(etc, vatt):
                    v = getattr(etc,vatt)
                    break
        print(v.ljust(20-len(tag))+printTime(toc))
    elif solver == 'sonics':
        # sonics
        vSONICS = os.getenv('SONICSVERSION', 'UNAVAILABLE')
        if vSONICS == 'UNAVAILABLE': vSONICS = RED + vSONICS + ENDC
        print(' --> SoNICS '+vSONICS)

        # Miles
        tag = '     with Miles '
        print(tag,end='')
        toc = tic()
        try:
            import miles
        except:
            v = RED + 'UNAVAILABLE' + ENDC
        else:
            try:
                from mola.misc import load_source
                import miles
                path = miles.__path__[0]
                path = os.path.sep.join(path.split(os.path.sep)[:-1])
                setup = load_source('setup', os.path.join(path, 'configure_setup.py'))
                v = f'{setup.VERSION_MAJOR}.{setup.VERSION_MINOR}'
            except:
                v = YELLOW + 'UNKNOWN' + ENDC
        print(v.ljust(20-len(tag))+printTime(toc))

    else:
        print(YELLOW+'WARNING: unknown solver'+ENDC)


    if totoV in ['Dev', 'master'] or 'dev' in totoV.lower():
        print(YELLOW+'WARNING: you are using an UNSTABLE version of MOLA.\nConsider using a stable version.'+ENDC)
    else:
        Major, Minor, Micro = getMajorMinorMicro(totoV)
        AllVersions = gatherMOLAversions()
        if not microVersionIsUpToDate(AllVersions, Major, Minor, Micro):
            print(YELLOW+'WARNING: a most updated micro version exist: '+fullStringOfMostUpToDateMicroVersion(AllVersions, Major, Minor)+ENDC)
        if not usingMostUpToDateVersion(AllVersions,Major, Minor, Micro):
            print('INFO: a most updated version exist: '+mostUpToDateVersion(AllVersions)+ENDC)
        else:
            print(GREEN+'You are using the latest version of MOLA'+ENDC)

