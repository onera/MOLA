from mola.workflow import read_workflow
import mola.naming_conventions as names
workflow = read_workflow(names.FILE_INPUT_SOLVER)
workflow.compute()