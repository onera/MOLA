#!/bin/bash
dpi=150
for f in *.pdf; do
    len=${#f}-4
    echo Will convert $f into ${f:0:$len}.png at $dpi dpi
    pdftoppm -r $dpi $f ${f:0:$len} -png -f 1 #-singlefile
done
