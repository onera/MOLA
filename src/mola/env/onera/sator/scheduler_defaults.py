from mola import solver

MOLA_PATH = '/tmp_user/sator/$USER/MOLA/Dev/src'

JOB_SCHEDULER = 'SLURM'

JOB_SCHEDULER_OPTIONS = {
    'time' : '15:00:00',
}

if solver == 'sonics':
    JOB_SCHEDULER_OPTIONS['constraint'] = 'spr'


MOLA_TO_SCHEDULER = {
    'AER' : 'comment',
}

AER_FOR_TEST = '37753003F'  # RG MOLA 2026
