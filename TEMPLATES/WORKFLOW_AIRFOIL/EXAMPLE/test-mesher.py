import MOLA.WorkflowAirfoil as WF

meshParams = WF.getMeshingParameters()
meshParams['References']['DeltaYPlus'] = 0.5
meshParams['References']['Reynolds']   = 333333.33

WF.buildMesh('e562.dat', save_meshParams=False, save_mesh=True,
                         meshParams=meshParams,)