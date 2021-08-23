import Converter.PyTree as C
import MOLA.WorkflowAirfoil as WF

t = C.convertFile2PyTree('Polar.cgns')

WF.correctPolar(t, useBigRangeValuesIf_StdCLisHigherThan=0.0005,
                   Fields2Correct=['Cl','Cd','Cm'])

C.convertPyTree2File(t, 'PolarCorrected.cgns')