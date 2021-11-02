import pprint
import MOLA.WorkflowAirfoil as WA

config = WA.JM.loadJobsConfiguration()

WA.printConfigurationStatus(config.DIRECTORY_WORK)

AoA = 6.0
Mach = 0.2
CASE_LABEL = WA.getCaseLabelFromAngleOfAttackAndMach(config, AoA, Mach)
Reynolds = WA.getReynoldsFromCaseLabel(config, CASE_LABEL)
IntegralLoads = WA.JM.getCaseArrays(config, CASE_LABEL)
print(pprint.pformat(IntegralLoads))
DistributedLoads = WA.getCaseDistributions(config, CASE_LABEL)
WA.getCaseFields(config, CASE_LABEL)
#
# WA.compareAgainstXFoil('naca 4416', config, CASE_LABEL, DistributedLoads)
