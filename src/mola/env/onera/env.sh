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

SCRIPT_DIR=$( \cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Detection machine
HOSTNAME=`hostname`
# Extract lines containing the Python dict PatternsToEnvironments in network.py
python_data=$(sed -n '/PatternsToEnvironments/,/}/p' $SCRIPT_DIR/network.py)
# Extract keys and values from PatternsToEnvironments
keys=$(echo "$python_data" | grep -oP "'[^']+'\ *:" | sed "s/'//g;s/://g")
values=$(echo "$python_data" | grep -oP ":\ *'[^']+'" | sed "s/://;s/'//g")
i=1
for pattern in $keys; do
    if [[ $HOSTNAME == $pattern ]]; then 
        export MOLA_MACHINE=$(echo $values | cut -d ' ' -f $i)
        break
    fi
    i=$((i+1))
done

if [ "$1" = "" ]; then
    export MOLA_SOLVER=mola
else
    export MOLA_SOLVER=$1
fi

source $SCRIPT_DIR/network.sh

# source the environment associated to the current machine and MOLA_SOLVER
echo "source $MOLA/mola/env/onera/$MOLA_MACHINE/$MOLA_SOLVER.sh"
source $MOLA/mola/env/onera/$MOLA_MACHINE/$MOLA_SOLVER.sh &>/dev/null || { echo 'Error: Cannot source this environment!' >&2; return 1; }
