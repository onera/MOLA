#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

import mola.cgns as cgns
import Connector.PyTree as X

def apply(workflow):
    t = workflow.tree
    for base in t.bases():
        component = workflow.get_component(base.name())

        if 'Connection' not in component: continue

        for operation in component['Connection']:
            ConnectionType = operation['Type']
            print(f'connecting type {ConnectionType} at base {base.name()}')
            try: tolerance = operation['Tolerance']
            except KeyError:
                print('connection tolerance not defined. Using tol=1e-8')
                tolerance = 1e-8
            
            if ConnectionType == 'Match':
                X.connectMatch(base, tol=tolerance, dim=base.dim())

            elif ConnectionType == 'NearMatch':
                try: ratio = operation['Ratio']
                except KeyError:
                    print('NearMatch ratio was not defined. Using ratio=2')
                    ratio = 2

                X.connectNearMatch(base, ratio=ratio,
                                         tol=tolerance,
                                         dim=base.dim())

            elif ConnectionType == 'PeriodicMatch':
                try: rotationCenter = operation['RotationCenter']
                except: rotationCenter = [0., 0., 0.]
                try: rotationAngle = operation['RotationAngle']
                except: rotationAngle = [0., 0., 0.]
                try: translation = operation['Translation']
                except: translation = [0., 0., 0.]
                t = X.connectMatchPeriodic(t,
                                            rotationCenter=rotationCenter,
                                            rotationAngle=rotationAngle,
                                            translation=translation,
                                            tol=tolerance,
                                            dim=base.dim())
            else:
                ERRMSG = f'Connection type {ConnectionType} not implemented'
                raise AttributeError(ERRMSG)

    workflow.tree = cgns.castNode(t)

