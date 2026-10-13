import os
from mola import solver, __version__, __MOLA_PATH__


if any(__MOLA_PATH__.startswith(p) for p in (
    '/stck/mola',
    '/tmp_user/juno/mola',
    '/tmp_user/sator/mola',
    )):

    # stable release or pre-release

    ci_tag = os.environ.get('CI_COMMIT_TAG')
    if ci_tag:
        # we are in ci/cd we suppose mola was correctly installed
        MOLA_PATH = f'/tmp_user/sator/mola/{ci_tag}/src'
    else:
        # we are in a stable release
        version_name = os.path.basename(os.path.dirname(__MOLA_PATH__))
        if __version__ != version_name:
            print(f'warning: __version__={__version__} != {version_name}')

        MOLA_PATH = f'/tmp_user/sator/mola/{version_name}/src'

else:
    # we are in developer mode, we use conventional install path:
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

MOLA_PMI2_LIB = '/usr/lib64/libpmi2.so'