# MOLA

MOLA (Modular Orchestration Library for Aerodynamics) is an ONERA Python code that implements user-level workflows and tools for aerodynamic analysis. 

[![Python 3](https://img.shields.io/static/v1?label=Python&logo=Python&color=3776AB&message=3)](https://www.python.org/)
![License-LGPL3](https://img.shields.io/badge/license-LGPLv3-blue.svg)

## Documentation

For documentation, examples, major changes, please consult the [documentation site](http://numerics.gitlab-pages.onera.net/mola/)


## Source MOLA for ONERA users

Please use one of the latest **stable versions** of MOLA: 

```bash
source /stck/mola/vX.Y.Z/env_mola.sh "solver"
```

The argument `"solver"` is the name (lowercase) of the solver you want to use. 
For instance, to use MOLA for elsA, use the command: 

```bash
source /stck/mola/vX.Y.Z/env_mola.sh elsa
```

## Installation

Refer to the [dedicated section of the documentation](https://numerics.gitlab-pages.onera.net/mola/latest/developer_manual/deployment.html).

## Usage

```python
from mola.workflow import Workflow  # choose a suitable Workflow for your application

workflow = Workflow(...)  # user parameters don't change with the solver
workflow.submit() # handles remote cluster job submission including preprocess
```

## Contributing

See ![CONTRIBUTING](CONTRIBUTING.md)

## License
See ![LICENSE](LICENSE)
