#!/bin/bash

git fetch /home/tbontemp/softs/MOLA/Dev/ dev_2.0:ThomasDev
git merge ThomasDev
git branch -d ThomasDev
