For documentation, examples, major changes, please consult the [documentation site](https://gitlab.onera.net/numerics/mola/-/wikis/Documentation-link)

Information for users
=====================

Please use one of the latests **stable versions** of MOLA : `/stck/lbernard/MOLA/v1.13/env_MOLA.sh` 


Information for developers
==========================


Getting the sources
-------------------

1. create a local directory named `MOLA` and go into it:

```bash
mkdir /stck/%USER/MOLA
cd /stck/%USER/MOLA
```

2. clone the sources from GitLab:

```bash
git clone git@gitlab.onera.net:numerics/mola.git
```

Make sure that you have correctly set your public ssh Key onto your profile's preferences of ONERA GitLab.

3. replace the newly created directory `mola` by `Dev`:

```bash
mv mola Dev
```

4. replace `lbernard` username by **yours** on files `env_MOLA.sh` and `TEMPLATES/job_template.sh`

5. Copy MOLA directory on your `sator` space:

```bash
cp -r /stck/$USER/MOLA /tmp_user/sator/$USER/MOLA
```


Contributing
------------

For making contributions, please follow these general rules:

0. If never done, please configure your `git` using **your** personal informations:

```bash
git config --global user.name "Georges Guynemer"
git config --global user.email georges.guynemer@onera.fr
```

1. create a *new branch* for your development:

```bash
git branch $USER-mydevname
git checkout $USER-mydevname
```

2. make your developments, and regularly update your sources onto GitLab:

Associate a commit short message to your major modifications:
```
git commit -m "this is a commit message"
```

Update your sources towards GitLab:
```
git push origin $USER-mydevname
```

3. For each development, update your **sator** sources using:

```
rsync -var /stck/$USER/MOLA/Dev /tmp_user/sator/$USER/MOLA/
```

4. Create *preferrably light* new examples using `EXAMPLES` arborescence

5. Before asking for the integration of your new developments into the `master` branch of MOLA, please 
   relaunch the cases contained in `EXAMPLES` *(specially LIGHT ones)* in order to verify that nothing
   is broken. 

6. After `commit` + `push`, request a merge towards `master` branch using GitLab's web interface.
   You will be automatically notified by e-mail once MOLA's maintainer has integrated your contribution.

7. You can update your sources using master's branch using:

```bash
git pull origin master
```

This is specially recommended once your development has been merged by MOLA's maintainer, or after major bug fixes.




