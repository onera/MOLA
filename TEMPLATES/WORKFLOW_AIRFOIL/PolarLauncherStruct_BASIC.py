import MOLA.WorkflowAirfoil as WF

PREFIX_JOB = 'j' # MUST BE UNIQUE for each airfoil
AER = '31447034F'
machine = 'sator'
DIRECTORY_WORK = '/tmp_user/sator/lbernard/MYPOLARS/'

# Airfoil must be placed in XY plane and be clockwise oriented starting from
# trailing edge. It can also be in selig / lednicer ASCII format 
GeomPath = '/home/ffalissa/H2T/ETUDES/MOTUS/FLUX_2/POLAIRES/PROFILS/Airfoil_20.tp'

AoARange    = [0,1,2,4,6,10,12,14,16,-1,-2,-4]
MachRange  =  [0.5, 0.6]
ReynoldsOverMach = 600000.0

WF.launchBasicStructuredPolars(PREFIX_JOB, GeomPath, AER, machine,
                          DIRECTORY_WORK, AoARange, MachRange, ReynoldsOverMach)