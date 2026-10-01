from mola.mpi import get_mpi_elements
comm, rank, size, nb_digs = get_mpi_elements()

from pprint import pformat
import etc.pypart.PyPart as PPA

PyPartBase = PPA.PyPart('main.cgns',
                        lksearch=['.'],
                        loadoption='partial',
                        mpicomm=comm,
                        LoggingInFile=False, 
                        LoggingFile='partTree',
                        LoggingVerbose=40  
                        )
PartTree = PyPartBase.runPyPart(method=2, partN=1, reorder=[6, 2])
PyPartBase.finalise(PartTree, savePpart=True, method=1)

Distribution = PyPartBase.getDistribution()
with open(f'distribution_rank_{rank}.py','w') as f:
    f.write(pformat(Distribution))

PyPartBase.mergeAndSave(PartTree, 'PyPart_fields', cgns_standard=True)

# eventually, read file using Cassiopee
import Converter.Mpi as Cmpi
import Distributor2.PyTree as D2
t = Cmpi.convertFile2SkeletonTree('PyPart_fields_all.hdf')
t, stats = D2.distribute(t, size, useCom=0, algorithm='fast')
t = Cmpi.readZones(t, 'PyPart_fields_all.hdf', rank=Cmpi.rank)
