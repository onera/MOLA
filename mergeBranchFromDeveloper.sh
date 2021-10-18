#!/bin/bash

git fetch /home/tbontemp/softs/MOLA/Dev/ master:ThomasDev
git merge ThomasDev
git branch -d ThomasDev
