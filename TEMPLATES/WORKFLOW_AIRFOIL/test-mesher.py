import MOLA.WorkflowAirfoil as WF

meshParams = WF.getMeshingParameters()
meshParams['References']['DeltaYPlus'] = 0.5
meshParams['References']['Reynolds']   = 1e6

WF.buildMesh('psu94097.txt', save_meshParams=False, save_mesh=True,
                         meshParams=meshParams,)
