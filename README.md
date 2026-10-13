# MOLA

MOLA (Modular Orchestration Library for Aerodynamics) is an ONERA Python code that implements user-level workflows and tools for aerodynamic analysis. 

[![Python 3](https://img.shields.io/static/v1?label=Python&logo=Python&color=3776AB&message=3)](https://www.python.org/)
![License-LGPL3](https://img.shields.io/badge/license-LGPLv3-blue.svg)

## Source MOLA for ONERA users

Please use one of the latest **stable versions** of MOLA: 

`source /stck/mola/vX.Y.Z/src/mola/env/onera/env.sh <SOLVER>`

The argument `<SOLVER>` is the name (lowercase) of the solver you want to use. 
For instance, to use MOLA for elsA, use the command: 

`source /stck/mola/vX.Y.Z/src/mola/env/onera/env.sh elsa`

## Installation

Refer to the [documentation site](https://numerics.gitlab-pages.onera.net/mola/latest/developer_manual/deployment.html).

## Usage

```python
from mola.workflow import Workflow  # choose a suitable Workflow for your application

workflow = Workflow(...)  # user parameters don't change with the solver
workflow.prepare()
workflow.write_cfd_files()
workflow.submit()
```

## Documentation

For documentation, examples, major changes, please consult the [documentation site](http://numerics.gitlab-pages.onera.net/mola/)

## Contributing

See ![CONTRIBUTING](CONTRIBUTING.md)

## License
See ![LICENSE](LICENSE)
