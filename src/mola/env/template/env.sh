#! /bin/sh
#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

# Detection machine
KC=`uname -n`
EL8=`uname -r|grep el8`
MAC0=$(echo $KC | grep 'cluster'); if [ "$MAC0" != "" ]; then export MAC="cluster"; fi
MAC0=$(echo $KC | grep 'node'); if [ "$MAC0" != "" ]; then export MAC="cluster"; fi
MAC0=$(echo $KC | grep 'local'); if [ "$MAC0" != "" ]; then export MAC="local"; fi

if [ "$1" = "" ]; then
    export MOLA_SOLVER=mola
else
    export MOLA_SOLVER=$1
fi

SCRIPT_DIR=$( \cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/network.sh

# source the environment associated to the current machine and MOLA_SOLVER
echo "source $MOLA/mola/env/template/$MAC/$MOLA_SOLVER.sh"
source $MOLA/mola/env/template/$MAC/$MOLA_SOLVER.sh &>/dev/null || echo 'Error: Cannot source this environment!'
