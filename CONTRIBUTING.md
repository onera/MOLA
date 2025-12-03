Table of contents
=================
- [Table of contents](#table-of-contents)
- [Getting the sources](#getting-the-sources)
- [Contributing](#contributing)
    - [1. Create a *new branch* for your development](#1-create-a-new-branch-for-your-development)
    - [2. Make your developments, and regularly update your sources onto GitLab:](#2-make-your-developments-and-regularly-update-your-sources-onto-gitlab)
    - [3. Check your developments and add examples](#3-check-your-developments-and-add-examples)
    - [4. Push your development](#4-push-your-development)
- [Guidelines](#guidelines)
  - [Syntax](#syntax)
  - [Architecture](#architecture)
  - [Development](#development)
  - [Tests](#tests)

Getting the sources
===================

Clone the sources from GitLab:

```bash
git clone git@gitlab.onera.net:numerics/mola.git
```
If never done, configure your `git` using **your** personal informations:

```bash
git config --global user.name "Georges Guynemer"
git config --global user.email georges.guynemer@onera.fr
```


Contributing
============

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

For making contributions, please follow these general rules:

### 1. Create a *new branch* for your development

```bash
git branch <mydevname>
git checkout <mydevname>
```

### 2. Make your developments, and regularly update your sources onto GitLab:

Make your developments following [guidelines](#guidelines).

Associate a commit short message to your major modifications:
```bash
git add <files>
git commit -m "this is a commit message"
```

Before commit, you could run pytest to check that nothing is broken (see section [Tests](#tests) for more details):
```bash
pytest $MOLA
```

Update regularly your sources towards GitLab:
```bash
git push origin <mydevname>
```

### 3. Check your developments and add examples

Before asking for the integration of your new developments into the `master` branch of MOLA:

* you **MUST** run pytest to check that nothing is broken.
* Create *preferrably light* new examples using `EXAMPLES` arborescence.
* Relaunch the cases contained in `EXAMPLES` *(specially LIGHT ones)* in order to verify that nothing is broken.

### 4. Push your development

:warning: Before pushing a development, you must be sure that it is compatible with ![MOLA licence](LICENSE), 
and above all **check that it can be disseminated freely**.

After `commit` + `push`, request a merge towards `master` branch using GitLab's web interface. 
The merge-request allows reviewing your development, discussing and possibly requesting modifications. 
In this case, you may fix the issue, commit again and push again your branch on GitLab. 
The merge-request will be automatically updated.

You will be automatically notified by e-mail once MOLA maintainer has integrated your contribution.

You can update your own branch sources using master's branch with:

```bash
git pull origin master
```

This is specially recommended once your development has been merged by MOLA's maintainer, or after major bug fixes.



Guidelines
==========

Except files handling environment, documentation, and GitLab/GitHub related features, the source code of MOLA is exclusively in Python.

As a first general advice about coding style, make your best to follow [PEP 8](https://pep8.org/). 
May the [Zen of Python](https://peps.python.org/pep-0020/#the-zen-of-python) be an inspiration for your developments !


Syntax
------

* **Files** names follow **snake-case** :snake: convention, like 'my_new_file.py'.

* **Functions** and **methods** names follow **snake-case** :snake: convention, like 'specific_function()'.

* **Classes** names follow **camel-case** :camel: convention, like 'WorkflowPropeller()'.

* For **variables** names, there is no global recommandation for style. However, for physical quantities, follow the [CGNS standard](http://cgns.github.io/CGNS_docs_current/sids/dataname.html) if possible. In all cases, use [meaningful names](https://ashishmd.medium.com/summary-of-clean-code-by-robert-c-martin-part-2-meaningful-names-5b5baaa5b3c6).


Architecture
------------

* Code lines specific to one solver should be written in files called `solver_<SOLVER_NAME>.py`, in the folder dedicated to the current feature. For instance, functions that specify boundary conditions for the elsA solver are in ``mola/cfd/preprocess/boundary_conditions/solver_elsa.py``. The name of the solver should be in lower case ('elsa', not 'elsA'; 'sonics', not 'SoNICS'). 

Development
-----------

* Documentation and information files (``README.md``, ``CONTRIBUTING.md``, ...), are written in [Markdown](https://www.markdownguide.org/cheat-sheet/).

* The "HACK" tag in source code indicates lines that make a workaround for an issue that rather should be handled by 
  another software. Normally, the issue should be reported to the support team of this software, and the lines marked
  with the "HACK" tag in MOLA should be removed once the issue is solved.

Tests
-----

* Tests are done using [pytest](https://docs.pytest.org/en/8.0.x/).

* There must be written in parallel of the development, in a `test` repository in the current module to be tested. To test functions in the file `file_with_bugs.py`, the test file must be called `test_file_with_bugs.py`. For instance, to test the functions or methods in ``mola/workflow/workflow.py``, the test file should be ``mola/workflow/test/test_workflow.py``.

* Tests are categorized usings markers to easily run some tests specifically (short tests, tests available for one solver only, etc.). To tag a test with a marker, be sure that `pytest` module is imported in the header of the file and add one or several markers as decorators of the test function :
```python
import pytest

@pytest.mark.unit
@pytest.mark.elsa
@pytest.mark.cost_level_1
def test_some_elsa_feature():
    pass
```
Main categories of tests are the following: 
* one of `unit`, `integration` or `user_case`
* optionally, one or several among solvers (in lower case). If there is none of them, the test will be run in every environment. If there is one or several markers for solvers, the test will be run only in these solver environments. 
* optionally, `cost_level_<N>`, with N in {0,1,2,3,4}. It defines a range of duration of the test (see these ranges in conftest.py). The aim of these markers is to easily raise a change in the duration of the test (if for some reason it is longer to run after some development).

To run tests choosing only some markers, use the option `-m` of pytest, like:
```
pytest $MOLA -m unit
```
Notice that the marker of the solver corresponding to the sourced environment is applied automatically.
