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
git branch $USER-mydevname
git checkout $USER-mydevname
```

### 2. Make your developments, and regularly update your sources onto GitLab:

Make your developments following [guidelines](#guidelines).

Associate a commit short message to your major modifications:
```bash
git commit -m "this is a commit message"
```

Before commit, you could run pytest to check that nothing is broken:
```bash
pytest $MOLA/mola
```

Update regularly your sources towards GitLab:
```bash
git push origin $USER-mydevname
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

You will be automatically notified by e-mail once MOLA's maintainer has integrated your contribution.

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

* For **variables** names, there is no global recommandation. However, for physical quantities, follow the [CGNS standard](http://cgns.github.io/CGNS_docs_current/sids/dataname.html) if possible.


Architecture
------------

* Code lines specific to one solver should be written in files called `solver_<SOLVER_NAME>.py`, in the folder dedicated to the current feature. For instance, functions that specify boundary conditions for the elsA solver are in ``mola/cfd/preprocess/boundary_conditions/solver_elsa.py``. The name of the solver should be in lower case ('elsa', not 'elsA'; 'sonics', not 'SoNICS'). 

Development
-----------

* In parallel of the development, unit tests must be written in a `test` repository in the current module to be tested. To test functions in the file `file_with_bugs.py`, the test file must be called `test_file_with_bugs.py`. For instance, to test the functions or methods in ``mola/workflow/workflow.py``, the test file should be ``mola/workflow/test/test_workflow.py``.
  Tests must be written to work with [pytest](https://docs.pytest.org/en/8.0.x/).

* Documentation and information files (``README.md``, ``CONTRIBUTING.md``, ...), are written in [Markdown](https://www.markdownguide.org/cheat-sheet/).
  