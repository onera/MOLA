MOLA_PATH = '/tmp_user/sator/$USER/MOLA/Dev/src'

JOB_SCHEDULER = 'SLURM'

JOB_SCHEDULER_OPTIONS = {
    'time' : '15:00:00',
    # 'constraint' : 'csl',
}

MOLA_TO_SCHEDULER = {
    'AER' : 'comment',
}

AER_FOR_TEST = '34790003F'  # PDEV MOLA 2025 
