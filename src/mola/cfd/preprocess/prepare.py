from pathlib import Path
import traceback
import mola.naming_conventions as names
from mola.workflow import read_workflow
from mola.logging import MolaException

def prepare_from_workflow_file():

    if not Path(names.FILE_INPUT_WORKFLOW).exists():
        return

    preprocessing_marker = Path(names.FILE_PREPROCESSING)
    try:
        preprocessing_marker.touch()
        w = read_workflow(names.FILE_INPUT_WORKFLOW)
        w.prepare()
        w.write_cfd_files(items='except_existing_job')

        Path(names.FILE_INPUT_WORKFLOW).unlink()
    except Exception as e:
        with open(names.FILE_ERROR_PREPARING_WORKFLOW, 'w') as f: 
            f.write(traceback.format_exc())
        raise MolaException('An error occured during preprocess') from e
    finally:
        # A failed preprocess is represented by ERROR_PREPARING_WORKFLOW;
        # leave no transient marker that could hide that terminal state.
        preprocessing_marker.unlink(missing_ok=True)

if __name__ == "__main__":
    prepare_from_workflow_file()
