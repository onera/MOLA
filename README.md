For documentation, examples, major changes, please consult the [documentation site](https://gitlab.onera.net/numerics/mola/-/wikis/home)

Information for users
=====================

Please use one of the latest **stable versions** of MOLA : `/stck/lbernard/MOLA/v1.15/env_MOLA.sh`


Information for developers
==========================


Getting the sources
-------------------

1. create a local directory named `MOLA` and go into it:

```bash
mkdir /stck/$USER/MOLA
cd /stck/$USER/MOLA
```

2. clone the sources from GitLab:

```bash
git clone git@gitlab.onera.net:numerics/mola.git
```

Make *sure* that you have correctly set your public ssh Key onto your profile's preferences of ONERA GitLab.

3. replace the newly created directory `mola` by `Dev`:

```bash
mv mola Dev
```

4. Replace paths directing to ``lbernard`` by **yours** on files `env_MOLA.sh` and `TEMPLATES/job_template.sh`
   (except the paths `EXTPYLIB` and `EXTPYLIBSATOR`).

You may adapt this script to simplify this operation (just replace `myMOLA` and `myMOLASATOR`):

```bash
#!/usr/bin/sh
# Execute this script once when getting MOLA sources to adapt the files
# env_MOLA.sh and job_template.sh to your paths

# Do not commit these files !
# Before pushing your local branch to the GitLab, use the following command
# to get back these files in their original state :
#   git checkout env_MOLA.sh && git checkout TEMPLATES/job_template.sh && chmod a+r env_MOLA.sh TEMPLATES/job_template.sh
# You amy create an alias for this command in your .bash_aliases

# MOLA version
MOLAVER='Dev'

# Custom paths for my dev version - TO MODIFY
myMOLA='/stck/tbontemp/softs/MOLA/$MOLAVER'
myMOLASATOR='/tmp_user/sator/tbontemp/MOLA/$MOLAVER'


################################################################################
################################################################################
# DO NOT MODIFY THE LINES BELOW

eval "myMOLAPATH=$myMOLA"
eval "myMOLASATORPATH=$myMOLASATOR"

# To be able to use your dev version on sator computation nodes, you must first
# to copy your dev done on /stck on /tmp_user/sator
# You may use the following command :
#   rsync -rav $myMOLAPATH $myMOLASATORPATH

# MOLA paths of for the master version
MOLA='/stck/lbernard/MOLA/$MOLAVER'
MOLASATOR='/tmp_user/sator/lbernard/MOLA/$MOLAVER'

# Modify paths to my custom MOLA dev version
sed -i $myMOLAPATH/env_MOLA.sh -e "s|MOLA=$MOLA|MOLA=$myMOLA|"
sed -i $myMOLAPATH/env_MOLA.sh -e "s|MOLASATOR=$MOLASATOR|MOLASATOR=$myMOLASATOR|"
# Restore external librairies
sed -i $myMOLAPATH/env_MOLA.sh -e 's|MOLAext=/stck/lbernard/MOLA/$MOLAVER/ext|MOLAext=/stck/lbernard/MOLA/Dev/ext|'
sed -i $myMOLAPATH/env_MOLA.sh -e 's|MOLASATORext=/tmp_user/sator/lbernard/MOLA/$MOLAVER/ext|MOLASATORext=/tmp_user/sator/lbernard/MOLA/Dev/ext|'
# Modify paths to my custom MOLA dev version in the job template
sed -i $myMOLAPATH/TEMPLATES/job_template.sh -e "s|$MOLA|$myMOLA|"
sed -i $myMOLAPATH/TEMPLATES/job_template.sh -e "s|$MOLASATOR|$myMOLASATOR|"
```

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
```bash
git commit -m "this is a commit message"
```

**Important**: Do not commit the files `env_MOLA.sh` and `TEMPLATES/job_template.sh`.
Before pushing your local branch to the GitLab, use the following command
to get back these files in their original state:
```bash
git checkout env_MOLA.sh && git checkout TEMPLATES/job_template.sh && chmod a+r env_MOLA.sh TEMPLATES/job_template.sh
```
You may create an alias for this command in your ``.bash_aliases``.

Update your sources towards GitLab:
```bash
git push origin $USER-mydevname
```

3. For each development, update your **sator** sources using:

```bash
rsync -var /stck/$USER/MOLA/Dev /tmp_user/sator/$USER/MOLA/
```

4. Create *preferrably light* new examples using `EXAMPLES` arborescence

5. Before asking for the integration of your new developments into the `master` branch of MOLA, please
   relaunch the cases contained in `EXAMPLES` *(specially LIGHT ones)* in order to verify that nothing
   is broken.

6. After `commit` + `push`, request a merge towards `master` branch using GitLab's web interface.
   You will be automatically notified by e-mail once MOLA's maintainer has integrated your contribution.

7. You can update your own branch sources using master's branch with:

```bash
git pull origin master
```

This is specially recommended once your development has been merged by MOLA's maintainer, or after major bug fixes.
