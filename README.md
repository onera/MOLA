# MOLA ![logo](DOCUMENTATION/src_sphinx/FIGURES/favicon.ico)

MOLA is an ONERA Python code that implements user-level workflows and tools for aerodynamic analysis. 

## Source MOLA for ONERA users

Please use one of the latests **stable versions** of MOLA : `source /stck/lbernard/MOLA/vX.Y/src/env/onera.env.sh`

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install MOLA.

```bash
pip install mola-workflow
```

## Usage

```python
from mola.workflow import Workflow

# set workflow parameters
params = dict(...)  # see the doc section

# create the workflow
workflow = Workflow(Solver='elsa', **params)  # user parameters don't change with the solver
workflow.prepare()

# run the simulation
workflow.run_simulation()
```

## Documentation
For documentation, examples, major changes, please consult the [documentation site](https://gitlab.onera.net/numerics/mola/-/wikis/home)

## Contributing

See ![CONTRIBUTING](CONTRIBUTING.md)

## License
See ![LICENSE](LICENSE)
