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

from treelab import cgns
from .reader import read

def reader(w, component):
    
    mesh = read(w, component['Source'])

    if isinstance(mesh, cgns.Zone):
        base = cgns.Base(Name=component['Name'], Children=[mesh])
        mesh = cgns.Tree()
        mesh.addChild(base)
    elif isinstance(mesh, cgns.Base):
        mesh.setName(component['Name'])
        t = cgns.Tree()
        t.addChild(mesh)
        mesh = t
    nb_of_bases = len(mesh.bases())
    if nb_of_bases != 1:
        msg = f"component {component['Name']} must have exactly 1 base (got {nb_of_bases})"
        raise ValueError(msg)
    base = mesh.bases()[0]
    base.setName( component['Name'] )

    return base

