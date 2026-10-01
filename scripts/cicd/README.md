# MOLA CI/CD Scripts

This directory contains CI/CD scripts for MOLA.

## Public Scripts (Available on GitHub)

These scripts are available in both ONERA GitLab and GitHub:

- `README.md` - This file
- `CI_CD_VARIABLES.md` - CI/CD configuration guide
- `monitor_slurm.sh` - Generic SLURM job monitoring utility

## ONERA-Specific Scripts (Internal Only)

The following scripts are ONERA-specific and are **not** published to GitHub:

- `deploy.sh` - SSH-based deployment to ONERA infrastructure
- `deploy_as_mola.sh` - Execution script with cleanup
- `deploy_doc.sh` - Stage and deploy documentation to /stck/mola/html via SSH
- `deploy_doc_as_mola.sh` - Privileged documentation deployment implementation
- `promote_release.sh` - Promote a validated release candidate
- `promote_release_as_mola.sh` - Privileged promotion implementation

These internal scripts are versioned in ONERA GitLab but excluded from GitHub by
`scripts/sync_to_github.sh`.

## Using the CI/CD Examples Runner

The main CI/CD tool is now a Python module: `mola.cicd.examples_runner`

### Usage

```bash
# Source the MOLA environment
source /path/to/mola/env/onera/env.sh elsa

# Submit all marked examples
python -m mola.cicd.examples_runner submit --pipeline-id 123 --version v2.0.1

# Check status
python -m mola.cicd.examples_runner status

# Remove local state and legacy generated example files
python -m mola.cicd.examples_runner clear --states-only

# Remove remote directories for failed cases, preserving state files
python -m mola.cicd.examples_runner clear --failed-only

# Remove all local state/legacy files and all remote example directories
python -m mola.cicd.examples_runner clear --all

# Move completed examples into the versioned Juno/Sator result spaces
python -m mola.cicd.examples_runner repatriate --full --tag v2.0.1-rc1

# Run comparison and visualization scripts on each machine
python -m mola.cicd.examples_runner postprocess --tag v2.0.1-rc1

# Publish signals, numbered coprocess logs, and generated images to /stck
python -m mola.cicd.examples_runner repatriate --light --tag v2.0.1-rc1

# Validate results
python -m mola.cicd.examples_runner validate
```

### Available Commands

| Command | Description |
|---------|-------------|
| `submit` | Submit all examples marked with `# MOLA_CI_MODE` |
| `status` | Check status of submitted examples |
| `clear` | Clean local CI state and/or remote example directories |
| `repatriate` | Move full results or publish light results (`--light` or `--full`) |
| `postprocess` | Run `compare.py` and `visualize.py` on each HPC machine |
| `validate` | Validate results from completed examples (exits with code 0/1/2) |

### Options

```bash
python -m mola.cicd.examples_runner submit --help

Options:
  --pipeline-id ID   CI/CD pipeline ID
  --version VER      MOLA version being tested
  --failed-only      Submit only examples marked as FAILED
```

The `clear` command requires exactly one cleanup scope:

```bash
python -m mola.cicd.examples_runner clear --states-only
python -m mola.cicd.examples_runner clear --failed-only
python -m mola.cicd.examples_runner clear --all
```

`clear --failed-only` preserves the local state files so they can be used by
`submit --failed-only`. The `--all` mode removes the local `.release`
directory, any legacy generated `.mola_run_directory`,
`workflow_manager.yaml`/`gather_results.py` files, and all remote run
directories recorded in the state files.

Submission state files are the source of truth for job IDs, run directories,
status checks, cleanup, and repatriation. The runner does not require
`.mola_run_directory` or `workflow_manager.yaml` files.

### Validate Command Exit Codes

The `validate` command is designed for CI/CD pipelines and exits with specific codes:

| Exit Code | Meaning | CI/CD Behavior |
|-----------|---------|----------------|
| `0` | All examples completed successfully | Pipeline continues to next stage |
| `1` | All examples completed, but some failed | Pipeline fails (red cross) |
| `2` | Some examples still running/pending | Job fails, manual retry required |

## Marking Examples for CI/CD

To include an example in CI/CD testing, add a marker at the end of the file:

```python
# Your example code here
w.prepare()
w.submit()

# MOLA_CI_MODE: mono
```

Use `# MOLA_CI_MODE: multi` for a `WorkflowManager` submission.

## GitLab CI/CD Workflow

The release-candidate pipeline consists of the following stages:

```yaml
stages:
  - release-candidate
  - test-examples-submit
  - test-examples-monitor
  - test-examples-repatriate
  - test-examples-postprocess
  - test-examples-publish
```

The final tag pipeline promotes the already validated release candidate, builds
the versioned documentation, and synchronizes the public GitHub tree.

### Job Flow

1. **`release_candidate_prepare`** (manual)
   - Verifies environment and version

2. **`test_examples_submit`** (manual, 30 min timeout)
   - Submits all marked examples
   - Creates state files in `.release/example_states/`
   - Uses 4 SLURM tasks for OpenMP parallelism

3. **`test_examples_monitor`** (manual)
   - Refreshes and validates all examples
   - If still running: Job fails, click "Retry" button
   - If all success: Pipeline continues
   - If some failed: Pipeline fails (red cross)

4. **`test_examples_repatriate`**
   - Moves full results to `/tmp_user/juno/mola/<tag>/examples` and
     `/tmp_user/sator/mola/<tag>/examples`

5. **`test_examples_postprocess`**
   - Runs postprocessors on each machine, so comparisons remain machine-local

6. **`test_examples_publish`**
   - Publishes only `OUTPUT/signals.cgns`, `LOGS/coprocess-*.log`, and images to
     `/stck/mola/<tag>/examples`

7. **`final_release`**
   - Promotes the validated `-rcX` directories to the final tag
   - Builds versioned documentation and synchronizes GitHub

### Syncing to GitHub

For ONERA maintainers, use the sync script to create a clean GitHub release:

```bash
./scripts/sync_to_github.sh v2.0.1
```

This creates a clean version without internal files while maintaining the same tag.

## Testing

Run unit tests for the examples runner:

```bash
pytest tests/cicd/test_examples_runner.py -v
```

## Example Output

### Status Command
```
============================================================
EXAMPLE STATUS SUMMARY
============================================================
Run directory                                                   Status
--------------------------------------------------------------------------------------------------------------------
/remote/examples/run_elsa                                       SUCCESS
/remote/examples/run_sator                                      RUNNING
/remote/examples/run_sator_isospeed_line                        SUCCESS
============================================================
```

### Validate Command (Success)
```
============================================================
EXAMPLE STATUS SUMMARY
============================================================
Run directory                                                   Status
--------------------------------------------------------------------------------------------------------------------
/remote/examples/run_elsa                                       SUCCESS
/remote/examples/run_sator                                      SUCCESS
/remote/examples/run_sator_isospeed_line                        SUCCESS
============================================================

✅ All examples completed successfully!
Pipeline validation passed. Proceeding to next stages.
```

### Validate Command (Still Running)
```
============================================================
EXAMPLE STATUS SUMMARY
============================================================
Run directory                                                   Status
--------------------------------------------------------------------------------------------------------------------
/remote/examples/run_elsa                                       SUCCESS
/remote/examples/run_sator                                      RUNNING
/remote/examples/run_sator_isospeed_line                        SUCCESS
============================================================

⚠️  Some examples are still running or pending.
Please retry this job manually when all examples complete.
```
