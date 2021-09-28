import MOLA.WorkflowAirfoil as WF

AER = '31447034F'
AirfoilPath = '/home/ffalissa/H2T/ETUDES/MOTUS/FLUX_2/POLAIRES/PROFILS/Airfoil_20.tp'

# To modify for each polar computation set:
PREFIX_JOB = 'j' # MUST BE UNIQUE
DIRECTORY_WORK = '/tmp_user/sator/lbernard/MYPOLAR/' # MUST BE UNIQUE

AoARange    = [0,1,2,4,6,10,12,14,16,-1,-2,-4]
MachRange  =  [0.5, 0.6]
ReynoldsOverMach = 600000.0

WF.launchBasicStructuredPolars(PREFIX_JOB, AirfoilPath, AER, 'sator',
                          DIRECTORY_WORK, AoARange, MachRange, ReynoldsOverMach)
