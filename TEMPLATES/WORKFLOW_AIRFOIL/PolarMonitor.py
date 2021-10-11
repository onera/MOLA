import pprint
import MOLA.WorkflowAirfoil as WF

DIRECTORY_WORK = '/tmp_user/sator/lbernard/MYPOLAR/'

WF.printConfigurationStatus(DIRECTORY_WORK, useLocalConfig=False)

AoA = 0.0
Mach = 0.6
config = WF.getPolarConfiguration(DIRECTORY_WORK)
CASE_LABEL = WF.getCaseLabelFromAngleOfAttackAndMach(config, AoA, Mach)
Reynolds = WF.getReynoldsFromCaseLabel(config, CASE_LABEL)
IntegralLoads = WF.getCaseLoads(config, CASE_LABEL)
print(pprint.pformat(IntegralLoads))
DistributedLoads = WF.getCaseDistributions(config, CASE_LABEL)
WF.getCaseFields(config, CASE_LABEL)
#
# WF.compareAgainstXFoil('naca 4416', config, CASE_LABEL, DistributedLoads)
