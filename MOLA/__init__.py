import os
__version__ = 'Dev'
__MOLA_PATH__ = os.path.sep.join(__file__.split(os.path.sep)[:-2])

try:
    def getSHA():
        with open('{}/.git/HEAD'.format(__MOLA_PATH__), 'r') as HEAD:
            line = HEAD.readlines()[0]
            ref = line.split('ref: ')[-1].rstrip('\n')
        with open('{}/.git/{}'.format(__MOLA_PATH__, ref), 'r') as REF:
            SHA = REF.readlines()[0].rstrip('\n')
        return SHA
    __SHA__ = getSHA()
except:
    __SHA__ = 'unknown'
