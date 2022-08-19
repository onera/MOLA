import MOLA.WorkflowAirfoil as WF

AER = '28643013F'
FILE_GEOMETRY = 'psu94097.txt'

# To modify for each polar computation set:
PREFIX_JOB = 'a'
DIRECTORY_WORK = '/tmp_user/sator/lbernard/MYPOLAR/'

AoARange    = [0,1,2,3,4,-1,-2,-3,-4]
MachRange  =  [0.6,0.7,0.8]
ReynoldsOverMach = 1e6/0.8

WF.launchBasicStructuredPolars(PREFIX_JOB, FILE_GEOMETRY, AER, 'sator',
                          DIRECTORY_WORK, AoARange, MachRange, ReynoldsOverMach)
