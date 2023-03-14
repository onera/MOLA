#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import sys
import os
import pprint

RED  = '\033[91m'
GREEN = '\033[92m'
WARN  = '\033[93m'
PINK  = '\033[95m'
CYAN  = '\033[96m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
ENDC  = '\033[0m'

AutoGridLocation = {'FlowSolution':'Vertex',
                    'FlowSolution#Centers':'CellCenter',
                    'FlowSolution#Height':'Vertex',
                    'FlowSolution#EndOfRun':'CellCenter',
                    'FlowSolution#Init':'CellCenter',
                    'FlowSolution#SourceTerm':'CellCenter',
                    'FlowSolution#EndOfRun#Coords':'Vertex'}

CoordinatesShortcuts = dict(CoordinateX='CoordinateX',
                            CoordinateY='CoordinateY',
                            CoordinateZ='CoordinateZ',
                            x='CoordinateX',
                            y='CoordinateY',
                            z='CoordinateZ',
                            X='CoordinateX',
                            Y='CoordinateY',
                            Z='CoordinateZ')

def sortListsUsingSortOrderOfFirstList(*arraysOrLists):
    '''
    This function accepts an arbitrary number of lists (or arrays) as input.
    It sorts all input lists (or arrays) following the ordering of the first
    list after sorting.

    Returns all lists with new ordering.

    Parameters
    ----------

        arraysOrLists : comma-separated arrays or lists
            Arbitrary number of arrays or lists

    Returns
    -------

        NewArrays : list
            list containing the new sorted arrays or lists following the order
            of first the list or array (after sorting).

    Examples
    --------

    ::

        import numpy as np
        import MOLA.Data.Core as C

        First = [5,1,6,4]
        Second = ['a','c','f','h']
        Third = np.array([10,20,30,40])

        NewFirst, NewSecond, NewThird = C.sortListsUsingSortOrderOfFirstList(First,Second,Third)
        print(NewFirst)
        print(NewSecond)
        print(NewThird)

    will produce

    ::

        [1, 4, 5, 6]
        ['c', 'h', 'a', 'f']
        [20, 40, 10, 30]

    '''
    SortInd = np.argsort(arraysOrLists[0])
    NewArrays = []
    for a in arraysOrLists:
        if type(a) == 'ndarray':
            NewArray = np.copy(a,order='K')
            for i in SortInd:
                NewArray[i] = a[i]

        else:
            NewArray = [a[i] for i in SortInd]

        NewArrays.append( NewArray )

    return NewArrays


def writeFileFromModuleObject(settings, filename='.MOLA.py'):
    Lines = '#!/usr/bin/python\n'

    for Item in dir(settings):
        if not Item.startswith('_'):
            Lines+=Item+"="+pprint.pformat(getattr(settings, Item))+"\n\n"

    with open(filename,'w') as f: f.write(Lines)

    try: os.remove(filename+'c')
    except: pass


def reload_source(module):
    '''
    Reload a python module guaranteeing intercompatibility between
    different Python versions

    Parameters
    ----------

        module : module
            pointer towards the previously loaded module
    '''

    import importlib
    importlib.reload(module)

def mute_stdout(func):
    '''
    This is a decorator to redirect standard output to /dev/null.
    '''
    def wrap(*args, **kwargs):
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            res = func(*args, **kwargs)
            sys.stdout = old_stdout
        return res
    return wrap

def mute_stderr(func):
    '''
    This is a decorator to redirect standard error to /dev/null.
    '''
    def wrap(*args, **kwargs):
        with open(os.devnull, 'w') as devnull:
            old_stderr = sys.stderr
            sys.stderr = devnull
            res = func(*args, **kwargs)
            sys.stderr = old_stderr
        return res
    return wrap

class OutputGrabber(object):
    """
    Class used to grab standard output or another stream.
    """
    escape_char = "\b"

    def __init__(self, stream=None, threaded=False):
        self.origstream = stream
        self.threaded = threaded
        if self.origstream is None:
            self.origstream = sys.stdout
        self.origstreamfd = self.origstream.fileno()
        self.capturedtext = ""
        # Create a pipe so the stream can be captured:
        self.pipe_out, self.pipe_in = os.pipe()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def start(self):
        """
        Start capturing the stream data.
        """
        self.capturedtext = ""
        # Save a copy of the stream:
        self.streamfd = os.dup(self.origstreamfd)
        # Replace the original stream with our write pipe:
        os.dup2(self.pipe_in, self.origstreamfd)
        if self.threaded:
            # Start thread that will read the stream:
            self.workerThread = threading.Thread(target=self.readOutput)
            self.workerThread.start()
            # Make sure that the thread is running and os.read() has executed:
            time.sleep(0.01)

    def stop(self):
        """
        Stop capturing the stream data and save the text in `capturedtext`.
        """
        # Print the escape character to make the readOutput method stop:
        self.origstream.write(self.escape_char)
        # Flush the stream to make sure all our data goes in before
        # the escape character:
        self.origstream.flush()
        if self.threaded:
            # wait until the thread finishes so we are sure that
            # we have until the last character:
            self.workerThread.join()
        else:
            self.readOutput()
        # Close the pipe:
        os.close(self.pipe_in)
        os.close(self.pipe_out)
        # Restore the original stream:
        os.dup2(self.streamfd, self.origstreamfd)
        # Close the duplicate stream:
        os.close(self.streamfd)

    def readOutput(self):
        """
        Read the stream data (one byte at a time)
        and save the text in `capturedtext`.
        """
        while True:
            if sys.version_info.major == 3:
                char = os.read(self.pipe_out, 1).decode(self.origstream.encoding)
            else:
                char = os.read(self.pipe_out, 1)
            if not char or self.escape_char in char:
                break
            self.capturedtext += char
