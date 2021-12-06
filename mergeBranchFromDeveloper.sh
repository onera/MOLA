#!/bin/bash

git fetch /stck/tbontemp/softs/MOLA/Dev/ master:ThomasDev
git merge ThomasDev
git branch -d ThomasDev
