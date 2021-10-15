import MOLA.WorkflowAirfoil as WA

config = WA.JM.loadJobsConfiguration()
PolarsDict, PyZonePolar = WA.buildPolar(config, PolarName='NACA4416')
WA.C.convertPyTree2File(PyZonePolar, 'Polar.cgns')
