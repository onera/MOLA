import MOLA.WorkflowAirfoil as WF

DIRECTORY_WORK = '/tmp_user/sator/lbernard/POLARS/NACA4416/'

PolarsDict, PyZonePolar = WF.buildPolar(DIRECTORY_WORK, 'NACA4416')
WF.C.convertPyTree2File(PyZonePolar, 'Polar.cgns')
