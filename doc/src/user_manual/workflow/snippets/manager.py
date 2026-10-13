from mola.workflow.rotating_component import turbomachinery

w = turbomachinery.Workflow(...)

# For a single case
w.prepare()
w.write_cfd_files()
w.submit()

## For multiple cases
manager = turbomachinery.WorkflowManager(w, '/tmp_user/sator/$USER/test/multi_run')

for solver in ['elsa', 'sonics']:
    manager.new_job(solver)
    for throttle in [0.95e5, 0.98e5, 1e5]:
        manager.add_variations(
            [
                ('Solver', solver),
                ('RunManagement|JobName', f'{solver}_{throttle:.2f}'),
                ('RunManagement|RunDirectory', f'Pressure_{throttle:.2f}'),
                ('BoundaryConditions|Family=R37_OUTFLOW|Pressure', throttle),
            ]
        )

manager.prepare()
manager.submit()
