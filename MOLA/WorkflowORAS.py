'''
MOLA - WorkflowORAS.py

WORKFLOW ORAS

Collection of functions designed for CFD simulations of Open Rotor And Stator (ORAS)

File history:
01/04/2022 - M. Balmaseda - Creation
'''

import os
import numpy as np

import Converter.PyTree    as C
import Converter.Internal  as I
import Distributor2.PyTree as D2
import Post.PyTree         as P
import Generator.PyTree    as G
import Transform.PyTree    as T
import Connector.PyTree    as X

from . import InternalShortcuts as J
from . import Preprocess        as PRE
from . import JobManager        as JM
from . import WorkflowCompressor as WC

def prepareMesh4ElsA(mesh, **kwargs):
    '''
    Exactly like :py:func:`MOLA.Preprocess.prepareMesh4ElsA`
    '''
    return WC.prepareMesh4ElsA(mesh, **kwargs)

def prepareMainCGNS4ElsA(mesh='mesh.cgns', ReferenceValuesParams={},
        NumericalParams={}, Extractions={},
        writeOutputFields=True, Initialization={'method':'uniform'}, TurboConfiguration = {},
        BoundaryConditions = [],bladeFamilyNames =['Blade'],
        FULL_CGNS_MODE=False):
    '''
    This is mainly a function similar to :func:`MOLA.Preprocess.prepareMainCGNS4ElsA`
    but adapted to ORAS mono-chanel computations. Its purpose is adapting
    the CGNS to elsA.

    Parameters
    ----------

        mesh : :py:class:`str` or PyTree
            if the input is a :py:class:`str`, then such string specifies the
            path to file (usually named ``mesh.cgns``) where the result of
            function :py:func:`prepareMesh4ElsA` has been writen. Otherwise,
            **mesh** can directly be the PyTree resulting from :func:`prepareMesh4ElsA`

        ReferenceValuesParams : dict
            Python dictionary containing the
            Reference Values and other relevant data of the specific case to be
            run using elsA. For information on acceptable values, please
            see the documentation of function :func:`computeReferenceValues`.

            .. note:: internally, this dictionary is passed as *kwargs* as follows:

                >>> MOLA.Preprocess.computeReferenceValues(arg, **ReferenceValuesParams)

        NumericalParams : dict
            dictionary containing the numerical
            settings for elsA. For information on acceptable values, please see
            the documentation of function :func:`MOLA.Preprocess.getElsAkeysNumerics`

            .. note:: internally, this dictionary is passed as *kwargs* as follows:

                >>> MOLA.Preprocess.getElsAkeysNumerics(arg, **NumericalParams)

        RPM : float
            revolutions per minute of the blade

        Extractions : :py:class:`list` of :py:class:`dict`
            List of extractions to perform during the simulation. See
            documentation of :func:`MOLA.Preprocess.prepareMainCGNS4ElsA`

        writeOutputFields : bool
            if :py:obj:`True`, write initialized fields overriding
            a possibly existing ``OUTPUT/fields.cgns`` file. If :py:obj:`False`, no
            ``OUTPUT/fields.cgns`` file is writen, but in this case the user must
            provide a compatible ``OUTPUT/fields.cgns`` file to elsA (for example,
            using a previous computation result).

        Initialization : dict
            dictionary defining the type of initialization, using the key
            **method**. See documentation of :func:`MOLA.Preprocess.initializeFlowSolution`

        FULL_CGNS_MODE : bool
            if :py:obj:`True`, put all elsA keys in a node ``.Solver#Compute``
            to run in full CGNS mode.

    Returns
    -------

        files : None
            A number of files are written:

            * ``main.cgns``
                main CGNS file to be read directly by elsA

            * ``OUTPUT/fields.cgns``
                file containing the initial fields (if ``writeOutputFields=True``)

            * ``setup.py``
                ultra-light file containing all relevant info of the simulation
    '''

    def addFieldExtraction(fieldname):
        try:
            FieldsExtr = ReferenceValuesParams['FieldsAdditionalExtractions']
            if fieldname not in FieldsExtr.split():
                FieldsExtr += ' '+fieldname
        except:
            ReferenceValuesParams['FieldsAdditionalExtractions'] = fieldname

    if isinstance(mesh,str):
        t = C.convertFile2PyTree(mesh)
    elif I.isTopTree(mesh):
        t = mesh
    else:
        raise ValueError('parameter mesh must be either a filename or a PyTree')

    FluidProperties = PRE.computeFluidProperties()
    if not 'Surface' in ReferenceValuesParams:
        ReferenceValuesParams['Surface'] = 1.0

    MainDirection = np.array([1,0,0]) # Strong assumption here
    YawAxis = np.array([0,0,1])
    PitchAxis = np.cross(YawAxis, MainDirection)
    ReferenceValuesParams.update(dict(PitchAxis=PitchAxis, YawAxis=YawAxis))

    ReferenceValues = PRE.computeReferenceValues(FluidProperties, **ReferenceValuesParams)
    

    if I.getNodeFromName(t, 'proc'):
        NProc = max([I.getNodeFromName(z,'proc')[1][0][0] for z in I.getZones(t)])+1
        ReferenceValues['NProc'] = int(NProc)
        ReferenceValuesParams['NProc'] = int(NProc)
        Splitter = None
    else:
        ReferenceValues['NProc'] = 0
        Splitter = 'PyPart'

    elsAkeysCFD      = PRE.getElsAkeysCFD(nomatch_linem_tol=1e-4)
    elsAkeysModel    = PRE.getElsAkeysModel(FluidProperties, ReferenceValues)
    elsAkeysNumerics = PRE.getElsAkeysNumerics(ReferenceValues, **NumericalParams)

    PRE.initializeFlowSolution(t, Initialization, ReferenceValues)

    WC.setBoundaryConditions(t, BoundaryConditions, TurboConfiguration,
                            FluidProperties,ReferenceValues, bladeFamilyNames=bladeFamilyNames)

    
    
    
    
    
    
    
    
    
    WC.computeFluxCoefByRow(t, ReferenceValues, TurboConfiguration)

    AllSetupDics = dict(FluidProperties=FluidProperties,
                        ReferenceValues=ReferenceValues,
                        elsAkeysCFD=elsAkeysCFD,
                        elsAkeysModel=elsAkeysModel,
                        elsAkeysNumerics=elsAkeysNumerics,
                        TurboConfiguration=TurboConfiguration,
                        Extractions=Extractions,
                        Splitter=Splitter)

    PRE.addTrigger(t)
    PRE.addExtractions(t, AllSetupDics['ReferenceValues'],
                          AllSetupDics['elsAkeysModel'], extractCoords=False)

    if elsAkeysNumerics['time_algo'] != 'steady':
        PRE.addAverageFieldExtractions(t, AllSetupDics['ReferenceValues'],
            AllSetupDics['ReferenceValues']['CoprocessOptions']['FirstIterationForAverage'])

    PRE.addReferenceState(t, AllSetupDics['FluidProperties'],
                         AllSetupDics['ReferenceValues'])
    dim = int(AllSetupDics['elsAkeysCFD']['config'][0])
    PRE.addGoverningEquations(t, dim=dim)
    AllSetupDics['ReferenceValues']['NProc'] = int(max(PRE.getProc(t))+1)
    PRE.writeSetup(AllSetupDics)

    if FULL_CGNS_MODE:
        PRE.addElsAKeys2CGNS(t, [AllSetupDics['elsAkeysCFD'],
                                 AllSetupDics['elsAkeysModel'],
                                 AllSetupDics['elsAkeysNumerics']])

    PRE.saveMainCGNSwithLinkToOutputFields(t,writeOutputFields=writeOutputFields)

    if not Splitter:
        print('REMEMBER : configuration shall be run using %s%d%s procs'%(J.CYAN,
                                                   ReferenceValues['NProc'],J.ENDC))
    else:
        print('REMEMBER : configuration shall be run using %s'%(J.CYAN + \
            Splitter + J.ENDC))

