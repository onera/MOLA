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
MOLA - _cpmv_.py

AUXILIARY COPY/MOVE MODULE

Python wrapper of convenient copy and move operations for
files and directories.

It supports multi-server operations (local, spiro and sator).

May be used in MODULE MODE or in TERMINAL MODE.


Example: Move an entire directory from sator to spiro.

-------------- Example of usage in MODULE MODE --------------
python3
>>> import MOLA._cpmv_ as cpmv
>>> cpmv.cpmvWrap4MultiServer('mv',
'/tmp_user/sator/username/sandbox/',
'/scratchm/username/sandbox/')


------------- Example of usage in TERMINAL MODE -------------
# REMEMBER: environment variables must be properly set
alias cpmv='python3 $MOLA/_cpmv_.py'

# Then in terminal one may tape this:
cpmv mv /tmp_user/sator/username/sandbox/ /scratchm/username/sandbox/

-------------------------- IMPORTANT --------------------------
This module must import standard python3 libraries only !!!
Otherwise, calling this module in TERMINAL MODE will produce
an error. Remember that even a usage as a MODULE will lead to
a usage in TERMINAL MODE because of the function cpmvWrap4MultiServer
---------------------------------------------------------------

First creation:
28/07/2020 - L. Bernardos - creation
'''

FAIL  = '\033[91m'
GREEN = '\033[92m'
WARN  = '\033[93m'
ENDC  = '\033[0m'

import sys
import os
import time
import timeit
import shutil
import subprocess
import socket
import getpass
import pprint

from distutils.dir_util import copy_tree

def cpmv(mode, In, Out='none'):
    '''
    Copy or move file or directory from <In> to <Out>, or remove <In>.
    Both <In> and <Out> must be accessible (i.e. a previous
    server connection may be required for achieving this).

    Users should rather use cpmvWrap4MultiServer() in case
    of doubt.

    INPUTS:
    mode (str) - 'mv', 'cp', 'rm', for moving, copying or removing respectively

    In (str) - Input (source) path (folder or file). If mode=='rm', only <In>
               is deleted

    Out (str) - Output (destination) path (folder or file) or None if mode=='rm'
    '''
    modeLowercase = mode.lower()[:2]

    if In == Out: return
    if not os.path.exists(In): return

    if modeLowercase == 'mv':
        if not os.path.isdir(Out):
            if sys.version_info.major == 3 and sys.version_info.minor > 2:
                os.makedirs(os.path.dirname(Out), exist_ok=True)
                shutil.move(In,Out)
            else:
                try:
                    shutil.move(In,Out)
                except:
                    os.makedirs(os.path.dirname(Out))
                    shutil.move(In,Out)

    elif modeLowercase == 'cp':
        if os.path.isdir(In):
            # avoid shutil.copytree(In,Out,symlinks=True) for existing dir
            copy_tree(In,Out)
        else:
            if sys.version_info.major == 3 and sys.version_info.minor > 2:
                os.makedirs(os.path.dirname(Out), exist_ok=True)
                shutil.copy2(In,Out,follow_symlinks=True)
            else:
                try:
                    shutil.copy2(In,Out)
                except IOError as io_err:
                        os.makedirs(os.path.dirname(Out))
                        shutil.copy2(In,Out)
                finally:
                    shutil.copy2(In,Out)

    elif modeLowercase == 'rm':
        if os.path.isdir(In):
            for PathName in os.listdir(In):
                file_path = os.path.join(In, PathName)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete file %s. Reason: %s' % (file_path, e))

            try:
                shutil.rmtree(In)
            except Exception as e:
                print('Failed to delete directory %s. Reason: %s' % (In, e))

        elif os.path.isfile(In) or os.path.islink(In):
            try:
                os.unlink(In)
            except Exception as e:
                print('FAILED in deleting %s. Error: %s'%(In, e))

    else:
        raise AttributeError("Mode %s not recognized. Must be 'cp' or 'mv'"%mode)

def whichServer(PathElts):
    '''
    Analyze the input path elements in order to determine if
    server connection is required for accessing the path.
    It also returns the username.

    INPUTS

    PathElts - (list of strings) - each element is a directory of a path,
        except the last one, which may correspond to a file

    OUTPUTS

    ServerName - (string) - Name of the involved machine.
        For example: 'sator' or 'spiro'.

    UserName - (string) - attempt to detect the username where the path
        is located
    '''
    PathElts = getAsPathElts(PathElts)
    if 'scratch' in PathElts[0]:
        return 'spiro', PathElts[1]
    elif (PathElts[0]=='tmp_user' and PathElts[1]=='sator'):
        return 'sator', PathElts[2]
    else:
        return '', ''

def whichHost():
    '''
    Returns the host name. For example:   'sator', 'spiro', 'visio' or 'celeste'
    '''
    HostName = socket.gethostname()
    PossibleMachineNamesInHostName = ('sator','spiro','visio','celeste')
    for name in PossibleMachineNamesInHostName:
        if name in HostName: return name
        if HostName.startswith('n'): return 'sator'
    UserName = getpass.getuser()
    if not os.path.exists(os.path.join(os.path.sep,'stck',UserName)):
        return 'StckInvisible'

    return HostName

def _launchSubprocess(Host,CMD):
    '''
    Wrapper for launching a subprocess in a remote server.

    INPUTS

    Host - (string) - hostname where subprocess will be submitted. For example,
        'username@sator' or 'spiro-daaa'

    CMD - (string) - command to submit to server. For multiple lines, use the
        following syntax: '"command1; command2; command3"'
    '''
    ssh = subprocess.Popen(
        'ssh %s %s'%(Host,CMD),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
        )
    ssh.wait()
    Error = readStderr(ssh)
    Output = readStdout(ssh)
    if len(Output)>0:
        print('\n'.join(Output))
    if len(Error)>0:
        WillRaise = False
        for e in Error:
            if 'warning:' in e:
                if not 'bind:' in e:
                    print(WARN+str(e)+ENDC)
            else:
                WillRaise = True
                print(FAIL+str(e)+ENDC)

        if WillRaise:
            msg = f'Host: {Host}\nCMD={CMD}\nerror message:\n' + '\n'.join(Error)
            raise IOError(FAIL+msg+ENDC)

def readStderr(ssh):
    '''
    Read the standard error from the object **ssh* obtained from
    subprocess.Popen

    Parameters
    ----------

        ssh : object
            returned by subprocess.Popen

    Returns
    -------

        Error : :py:class:`list` of :py:class:`str`
            error lines

    See also
    --------
    readStdout
    '''
    Error = ssh.stderr.readlines()
    for i, e in enumerate(Error):
        if isinstance(e, bytes):
            Error[i] = e.decode('utf-8')
    return Error

def readStdout(ssh):
    '''
    Read the standard output from the object **ssh* obtained from
    subprocess.Popen

    Parameters
    ----------

        ssh : object
            returned by subprocess.Popen

    Returns
    -------

        Output : :py:class:`list` of :py:class:`str`
            output lines

    See also
    --------
    readStderr
    '''
    Output = ssh.stdout.readlines()
    for i, o in enumerate(Output):
        if isinstance(o, bytes):
            Output[i] = o.decode('utf-8')
    return Output

def cpmvWrap4MultiServer(mode,In,Out='none'):
    '''
    Copy or move file or directory from <In> to <Out>.

    This function attempts server connections if <In> and/or
    <Out> are located in paths only accessible by servers. In
    this case, calls of function cpmv() are sent from server using
    a subprocess.

    INPUTS:
    mode (str) - 'mv' or 'cp', for moving or copying respectively

    In (str) - Input (source) path (folder or file)

    Out (str) - Output (destination) path (folder or file)
    '''
    if In == Out: return
    cwd = os.getcwd()

    if In[0] != os.path.sep: In = os.path.join(cwd,In)
    if Out[0] != os.path.sep: Out = os.path.join(cwd,Out)

    PathInElts = list(filter(None,In.split(os.path.sep)))
    PathOutElts = list(filter(None,Out.split(os.path.sep)))


    if len(mode) > 2:    # cases: 'cp_forced', 'mv_forced'
        cpmv(mode,In,Out)
    elif len(mode) == 2: # cases: 'cp', 'mv'
        InAtServer,  usernameIn  = whichServer(PathInElts)
        OutAtServer, usernameOut = whichServer(PathOutElts)
        usernameIn = getpass.getuser()
        HostServer = whichHost()

        if HostServer == 'StckInvisible':
            if PathInElts[0] == 'stck' or PathOutElts[0] == 'stck':
                print('WARNING: Requested %s\ntowards %s\nalthough stck connection is impossible'%(In,Out))
            cpmv(mode,In,Out)
        elif InAtServer == OutAtServer == HostServer:
            cpmv(mode,In,Out)
        elif HostServer == 'sator' and 'spiro' not in (InAtServer, OutAtServer):
            cpmv(mode,In,Out)
        elif HostServer == 'spiro' and 'sator' not in (InAtServer, OutAtServer):
            cpmv(mode,In,Out)
        elif all([InAtServer,OutAtServer]):
            # Need to use a temporary auxiliar folder on local
            AuxiliaryDir = os.path.join(os.path.sep,'stck',usernameIn,'.tmpFolder4cpmv')
            OutAuxiliary = os.path.join(AuxiliaryDir,PathInElts[-1])

            if   InAtServer == 'spiro': Host = usernameIn+'@'+'spiro-daaa'
            elif InAtServer == 'sator': Host = usernameIn+'@'+'sator'
            else: raise ValueError('BAD CONDITIONING')
            CMD = 'python3 $MOLA/MOLA/_cpmv_.py %s_forced %s %s'%(mode,In,OutAuxiliary)
            _launchSubprocess(Host,CMD)

            if   OutAtServer == 'spiro': Host = usernameIn+'@'+'spiro-daaa'
            elif OutAtServer == 'sator': Host = usernameIn+'@'+'sator'
            else: raise ValueError('BAD CONDITIONING')
            if In[-1] == os.path.sep: OutAuxiliary += os.path.sep
            CMD = 'python3 $MOLA/MOLA/_cpmv_.py mv_forced %s %s'%(OutAuxiliary,Out)
            _launchSubprocess(Host,CMD)

        elif any([InAtServer,OutAtServer]):
            Server = max([InAtServer,OutAtServer])
            User   = max([usernameIn,usernameOut])
            if Server == 'spiro': Server += '01'
            Host = User+'@'+Server

            CMD = 'python3 $MOLA/MOLA/_cpmv_.py %s_forced %s %s'%(mode,In,Out)
            _launchSubprocess(Host,CMD)

        else:
            cpmv(mode,In,Out) # No server is involved, do regular operation
    else:
        raise AttributeError('Mode %s not recognized'%mode)

def wait4FileFromServer(filename, requestInterval=0.5, timeout=60.):
    '''
    This function is employed to determine if a file exist on a given path.
    The algorithm will check for file existence every <requestInterval> seconds,
    up to a limit of <timeout> seconds.
    As soon as file is detected, the function returns True. Otherwise, if the
    timeout is reached, the function returns False and raises a warning.

    INPUTS

    filename - (string) - full path of the element to check existance

    requestInterval - (float) - seconds to wait between two consecutive checks

    timeout - (float) - maximum total waiting time, after which the function
        will return False.

    OUTPUTS

    pathfound - (boolean) - True if the path is found before timeout. False
        otherwise
    '''
    tic = timeit.default_timer()
    directory_path = os.path.dirname(filename)
    if not directory_path: directory_path = '.'
    os.listdir(directory_path)
    ElapsedTime = 0.
    if os.path.exists(filename): return True
    while ElapsedTime < timeout:
        time.sleep(requestInterval)
        ElapsedTime = timeit.default_timer() - tic
        os.listdir(directory_path)
        if os.path.exists(filename): return True
    if ElapsedTime >= timeout:
        print(WARN+'Warning: timeout reached.'+ENDC)
        return False

def getAsPathElts(PathElts):
    if isinstance(PathElts,str):
        PathElts = PathElts.split(os.path.sep)
        if not PathElts[0] and len(PathElts) > 1: PathElts = PathElts[1:]

    return PathElts

if __name__ == '__main__':
    mode = sys.argv[1] # 'cp', 'mv', 'cp_forced', 'mv_forced'
    In   = sys.argv[2]
    Out  = sys.argv[3]
    cpmvWrap4MultiServer(mode, In, Out)
