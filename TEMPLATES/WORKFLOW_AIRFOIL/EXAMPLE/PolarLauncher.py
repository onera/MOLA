import MOLA.WorkflowAirfoil as WF

PREFIX_JOB = 'c' # MUST BE UNIQUE for each airfoil
AER = '31447034F'
machine = 'sator'
DIRECTORY_WORK = '/tmp_user/sator/lbernard/POLARS/NEWTEST/'
GeomPath = 'NACA4416'

AoARange    = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,-1,-2,-3,-4,-5,-6,-7,-8]
MachRange  =  [0.3, 0.4]
ReynoldsOverMach = 2e6 / 0.6

WF.launchBasicStructuredPolars(PREFIX_JOB, GeomPath, AER, machine,
                          DIRECTORY_WORK, AoARange, MachRange, ReynoldsOverMach)
