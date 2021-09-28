import MOLA.WorkflowAirfoil as WA

t = WA.C.convertFile2PyTree('Polar.cgns')
WA.correctPolar(t, useBigRangeValuesIf_StdCLisHigherThan=0.0005,
                   Fields2Correct=['Cl','Cd','Cm'])
WA.C.convertPyTree2File(t, 'PolarCorrected.cgns')
