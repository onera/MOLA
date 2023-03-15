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

'''
CouplingScript.py template
For coupling between elsA and Zset using CWIPI
Adapted to Dirichlet-Robin problems.
'''

iteration = elsAxdt.iteration()
e_state   = elsAxdt.state()

SentVariables = ['NormalHeatFlux', 'Temperature']
ReceivedVariables = ['Temperature']
VariablesForAlpha = ['Density', 'thrm_cndy_lam', 'hpar', 'ViscosityMolecular', 'Viscosity_EddyMolecularRatio']

stepForCwipiCommunication = setup.ReferenceValues['CoprocessOptions']['UpdateCWIPICouplingFrequency']
if 'timestep' in setup.elsAkeysNumerics:
    timestep = setup.elsAkeysNumerics['timestep']
    dtCoupling = timestep * stepForCwipiCommunication
RequestedStatistics = CO.getOption('RequestedStatistics', default=[])

print("SCRIPT CWIPI+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print('elsA >> iteration {}, state {}'.format(iteration, e_state))
AllNeededVariables = SentVariables + VariablesForAlpha

if e_state in [2, 16] and iteration % stepForCwipiCommunication == 0 and iteration > 0:

    outputTree  = elsAxdt.get(elsAxdt.OUTPUT_TREE)
    elsAxdt.free(elsAxdt.OUTPUT_TREE)
    outputTree = I.merge([outputTree, t])

    for CPLsurf, cplList in pyC2Connections.items():
        CO.printCo('CWIPI coupling on {}'.format(CPLsurf), 0, color=CO.MAGE)
        #___________________________________________________________________________
        # Get all needed data at coupled BCs
        #___________________________________________________________________________
        BCDataSet = dict(zip(AllNeededVariables, [[]]*len(AllNeededVariables)))
        for BCnode in C.getFamilyBCs(outputTree, CPLsurf):
            for var in AllNeededVariables:
                varNode = I.getNodeFromName(BCnode, var)
                if varNode:
                    BCDataSet[var] = BCDataSet[var] + list(I.getValue(varNode).flatten())
        for var in AllNeededVariables:
            BCDataSet[var] = np.array(BCDataSet[var]).transpose()

        #___________________________________________________________________________
        # SEND DATA
        #___________________________________________________________________________
        maxSentValues = dict()
        for var in SentVariables:
            print('Sending...')
            r = cplList.publish2(BCDataSet[var], iteration=iteration, stride=1, tag=100)
            cplList.wait_issend(r)
            print("Send {}, with mean value = {}".format(var, np.mean(BCDataSet[var])))

            maxSentValues[var] = np.amax(BCDataSet[var])

        maxSentValues['IterationNumber'] = iteration #-1,  # Because extraction before current iteration (next_state=16)

        CO.appendDict2Arrays(arrays, maxSentValues, 'SEND_{}'.format(CPLsurf))
        CO._extendArraysWithStatistics(arrays, 'SEND_{}'.format(CPLsurf), RequestedStatistics)
        arraysTree = CO.arraysDict2PyTree(arrays)
        CO.save(arraysTree, os.path.join(DIRECTORY_OUTPUT, FILE_ARRAYS))

        #___________________________________________________________________________
        # Compute alpha_opt
        #___________________________________________________________________________
        BCDataSet['cp'] = setup.FluidProperties['cp']

        if 'timestep' not in setup.elsAkeysNumerics:
            localTimestep = WAT.computeLocalTimestep(BCDataSet, setup)
            dtCoupling = localTimestep * stepForCwipiCommunication

        alphaOpt = WAT.computeOptimalAlpha(BCDataSet, dtCoupling)
        alphaOpt *= WAT.rampFunction(100, 100+10*stepForCwipiCommunication, 1e5, 10.)(iteration)
        print('alphaOpt = {}'.format(np.mean(alphaOpt)))
        r = cplList.publish2(alphaOpt, iteration=iteration, stride=1, tag=100)
        cplList.wait_issend(r)


    #___________________________________________________________________________
    # RECEIVE DATA
    #___________________________________________________________________________
    remote_data = dict()
    for CPLsurf, cplList in pyC2Connections.items():
        print('Receiving...')
        (r2, remote_data[CPLsurf]) = cplList.retrieve2(iteration=iteration, stride=len(ReceivedVariables), tag=100)
        cplList.wait_irecv(r2)
        print("Received data with mean value = {}".format(np.mean(remote_data[CPLsurf])))

    #___________________________________________________________________________
    # MODIFY BC IN ELSA
    #___________________________________________________________________________
    for CPLsurf, cplList in pyC2Connections.items():
        size1 = 0
        for var in ReceivedVariables:
            for BCnode in C.getFamilyBCs(outputTree, CPLsurf):
                size = WAT.getNumberOfNodesInBC(BCnode)
                varNode = I.getNodeFromName(BCnode, var)
                if varNode:
                    size2 = size1 + size
                    varNode[1] = remote_data[CPLsurf][size1:size2].copy()
            size1 = size2

    #___________________________________________________________________________
    # UPDATE RUNTIME TREE
    #___________________________________________________________________________
    elsAxdt.xdt(elsAxdt.PYTHON,(elsAxdt.RUNTIME_TREE, outputTree, 1))
