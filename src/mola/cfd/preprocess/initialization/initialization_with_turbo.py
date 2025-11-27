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

import os
from treelab import cgns
from mola.logging import mola_logger, MolaException, redirect_streams_to_null


def initialize_flow_with_turbo(workflow, FlowSolution_name):
    '''
    Initialize the flow solution with the module ``turbo``.
    The initial flow is computed analytically in the 2D-throughflow plane
    based on:

    * radial equilibrium in the radial direction.

    * Euler theorem between rows in the axial direction.

    The values **FlowAngleAtRoot** and **FlowAngleAtTip** (relative angles 'beta')
    must be provided for each row in **TurboConfiguration**.

    .. note::
        See also documentation of the related function in ``turbo`` module
        `<file:///stck/jmarty/TOOLS/turbo/doc/html/initial.html>`_

    .. important::
        Dependency to ``turbo``

    .. danger::
        Rows in TurboConfiguration must be list in the downstream order.

    '''
    mola_logger.info(' - initialize flow with turbo', rank=0)
    
    if workflow.Name != 'WorkflowTurbomachinery' or workflow.Solver != 'elsa':
        raise MolaException('Initialization with turbo is available only with WorkflowTurbomachinery and for elsA solver.')
    
    import turbo.initial as TI
    import Converter.PyTree as C

    t = workflow.tree

    if os.path.isfile('mask.cgns'):
        mask = cgns.load('mask.cgns')
    else: 
        mask = compute_mask(t)

    def getPlane(t, row, rowParams, Extractions, plane_type):
        try:
            return rowParams['InletPlane']
        
        except KeyError:
            try:
                for ext in Extractions:
                    try:
                        Type = ext['Type']
                        IsoSurfaceField = ext['IsoSurfaceField']
                        IsoSurfaceValue = ext['IsoSurfaceValue']
                        ReferenceRow = ext['OtherOptions']['ReferenceRow']
                        tag = ext['OtherOptions']['tag']
                    except:
                        continue

                    if Type == 'IsoSurface' and ReferenceRow == row and tag == plane_type:
                        if IsoSurfaceField in ['CoordinateX', 'x', 'X']:
                            plane_points = [[IsoSurfaceValue, -999.],[IsoSurfaceValue, 999.]]
                            return plane_points
                        elif IsoSurfaceField in ['CoordinateR', 'r', 'R', 'radius', 'Radius']:
                            plane_points = [[-999., IsoSurfaceValue], [999., IsoSurfaceValue]]
                            return plane_points
                    
                raise KeyError

            except KeyError:
                zones = C.getFamilyZones(t, row)
                return C.getMinValue(zones, 'CoordinateX')

    def getInletPlane(t, row, rowParams, Extractions):
        return getPlane(t, row, rowParams, Extractions, 'InletPlane')

    def getOutletPlane(t, row, rowParams, Extractions):
        return getPlane(t, row, rowParams, Extractions, 'OutletPlane')

    class RefState():
        def __init__(self):
            self.Gamma = workflow.Fluid['Gamma']
            self.Rgaz  = workflow.Fluid['IdealGasConstant']
            self.Pio   = workflow.Flow['PressureStagnation']
            self.Tio   = workflow.Flow['TemperatureStagnation']
            self.roio  = self.Pio / self.Tio / self.Rgaz
            self.aio   = (self.Gamma * self.Rgaz * self.Tio)**0.5
            self.Lref  = 1.

    planes_data = []
    config = 'axial'  # by default

    row, rowParams = list(workflow.ApplicationContext['Rows'].items())[0]
    plane_points = getInletPlane(t, row, rowParams, workflow.Extractions)
    alpha = 0.  # workflow.ApplicationContext['AngleOfAttackDeg']
    planes_data.append(
        dict(
            omega = 0.,
            beta = [alpha, alpha],
            Pt = 1.,
            Tt = 1.,
            massflow = workflow.Flow['MassFlow'],
            plane_points = plane_points,
            plane_name = '{}_InletPlane'.format(row)
        )
    )

    for row, rowParams in workflow.ApplicationContext['Rows'].items():
        plane_points = getOutletPlane(t, row, rowParams, workflow.Extractions)
        if plane_points[0][0] == -999. and plane_points[1][0] == 999.:
            config = 'centrifugal'
        omega = workflow.ApplicationContext['ShaftRotationSpeed'] if rowParams['IsRotating'] else 0.
        beta1 = rowParams.get('FlowAngleAtRootDeg', 0.)
        beta2 = rowParams.get('FlowAngleAtTipDeg', 0.)
        Csir = 1. if omega == 0 else 0.95  # Total pressure loss is null for a rotor, 5% for a stator
        planes_data.append(
            dict(
                omega = omega,
                beta = [beta1, beta2],
                Csir = Csir,
                plane_points = plane_points,
                plane_name = '{}_OutletPlane'.format(row)
                )
        )

    if workflow.Name == 'WorkflowLinearCascade':
        config = 'linear'
        lin_axis = workflow.ApplicationContext['lin_axis']
    else:
        lin_axis = 'XY'  # whatever, this value is not used

    # > Initialization
    with redirect_streams_to_null():
        t = TI.initialize(t, mask, RefState(), planes_data,
                config=config,
                lin_axis = lin_axis,
                constant_data=workflow.Turbulence['Conservatives'],
                turbvarsname=list(workflow.Turbulence['Conservatives']),
                velocity='absolute',
                useSI=True,
                keepTmpVars=False,
                keepFS=True,  # To conserve other FlowSolution_t nodes, as FlowSolution#Height
                fsname=FlowSolution_name,
                )

    workflow.tree = cgns.castNode(t)
 

def compute_mask(t, lin_axis=None, method=2):
    import turbo.height as TH

    if not lin_axis:
        endlinesTree = TH.generateHLinesAxial(t, filename='shroud_hub_lines.plt', method=method)
        with redirect_streams_to_null():
            m = TH.generateMaskWithChannelHeight(t, 'shroud_hub_lines.plt')
        os.remove('shroud_hub_lines.plt')
    else:
        m = TH.generateMaskWithChannelHeightLinear(t, lin_axis=lin_axis)
    return m
