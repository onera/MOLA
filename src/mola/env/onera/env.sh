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
export MOLA_MACHINE=$(python3 $SCRIPT_DIR/network.py)

if [ "$1" = "" ]; then
    export MOLA_SOLVER=elsa  # default solver
else
    export MOLA_SOLVER=$1
fi

source "$SCRIPT_DIR/network.sh" || {
    rc=$?
    echo "Failed to source: $SCRIPT_DIR/network.sh (status $rc)" >&2
    return "$rc"
}

if [ -z "${MOLA:-}" ]; then
    echo "MOLA is unset after sourcing network.sh" >&2
    return 1
fi

ENV_FILE="$MOLA/mola/env/onera/$MOLA_MACHINE/$MOLA_SOLVER.sh"
printf 'Sourcing environment file: %s\n' "$ENV_FILE"

if [ ! -r "$ENV_FILE" ]; then
    echo "Environment file is missing or unreadable: $ENV_FILE" >&2
    return 1
fi

source "$ENV_FILE" || {
    rc=$?
    echo "Failed to source: $ENV_FILE (status $rc)" >&2
    return "$rc"
}