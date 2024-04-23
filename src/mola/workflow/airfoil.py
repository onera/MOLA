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

import numpy as np

from treelab import cgns
from . import Workflow


class WorkflowAirfoil(Workflow):

    def __init__(self, **UserParameters):

        self.ApplicationContext.setdefault('AngleOfAttackDeg', 0.0) 
        self.ApplicationContext.setdefault('AngleOfSlipDeg', 0.0) 
        self.ApplicationContext.setdefault('YawAxis', [0.,0.,1.]) 
        self.ApplicationContext.setdefault('PitchAxis', [0.,-1.,0.]) 
        self.set_flow_directions()

        self.ApplicationContext.setdefault('Chord', 1.)
        self.ApplicationContext.setdefault('Surface', 1.)

        super(WorkflowAirfoil, self).__init__(**UserParameters)

    def set_flow_directions(self):
        DragDirection, SideDirection, LiftDirection = self.get_flow_directions(
            self.ApplicationContext['AngleOfAttackDeg'], 
            self.ApplicationContext['AngleOfSlipDeg'], 
            self.ApplicationContext['YawAxis'], 
            self.ApplicationContext['PitchAxis']
            )
        
        self.ApplicationContext.update(
            dict(DragDirection=DragDirection, SideDirection=SideDirection, LiftDirection=LiftDirection)
        )

        self.Flow['Direction'] = self.ApplicationContext['DragDirection']

    @staticmethod
    def get_flow_directions(AngleOfAttackDeg, AngleOfSlipDeg, YawAxis, PitchAxis):
        '''
        Compute the main flow directions from angle of attack and slip and aircraft
        yaw and pitch axis. The resulting directions can be used to impose inflow
        conditions and to compute aero-forces (Drag, Side, Lift) by projection of
        cartesian (X, Y, Z) forces onto the corresponding Flow Direction.

        Parameters
        ----------

            AngleOfAttackDeg : float
                Angle-of-attack in degree. A positive
                angle-of-attack has an analogous impact as making a rotation of the
                aircraft around the **PitchAxis**, and this will likely contribute in
                increasing the Lift force component.

            AngleOfSlipDeg : float
                Angle-of-attack in degree. A positive
                angle-of-slip has an analogous impact as making a rotation of the
                aircraft around the **YawAxis**, and this will likely contribute in
                increasing the Side force component.

            YawAxis : array of 3 :py:class:`float`
                Vector indicating the Yaw-axis of the
                aircraft, which commonly points towards the top side of the aircraft.
                A positive rotation around **YawAxis** is commonly produced by applying
                left-pedal rudder (rotation towards the left side of the aircraft).
                This left-pedal rudder application will commonly produce a positive
                angle-of-slip and thus a positive side force.

            PitchAxis : array of 3 :py:class:`float`
                Vector indicating the Pitch-axis of the
                aircraft, which commonly points towards the right side of the
                aircraft. A positive rotation around **PitchAxis** is commonly produced
                by pulling the elevator, provoking a rotation towards the top side
                of the aircraft. By pulling the elevator, a positive angle-of-attack
                is created, which commonly produces an increase of Lift force.

        Returns
        -------

            DragDirection : array of 3 :py:class:`float`
                Vector indicating the main flow
                direction. The Drag force is obtained by projection of the absolute
                (X, Y, Z) forces onto this vector. The inflow vector for reference
                state is also obtained by projection of the momentum magnitude onto
                this vector.

            SideDirection : array of 3 :py:class:`float`
                Vector normal to the main flow
                direction pointing towards the Side direction. The Side force is
                obtained by projection of the absolute (X, Y, Z) forces onto this
                vector.

            LiftDirection : array of 3 :py:class:`float`
                Vector normal to the main flow
                direction pointing towards the Lift direction. The Lift force is
                obtained by projection of the absolute (X, Y, Z) forces onto this
                vector.
        '''
        import Geom.PyTree as D
        import Transform.PyTree as T

        def getDirectionFromLine(line):
            x,y,z = line.xyz()
            Direction = np.array([x[-1]-x[0], y[-1]-y[0], z[-1]-z[0]])
            Direction /= np.sqrt(Direction.dot(Direction))
            return Direction

        # Yaw axis must be exact
        YawAxis    = np.array(YawAxis, dtype=np.float64)
        YawAxis   /= np.sqrt(YawAxis.dot(YawAxis))

        # Pitch axis may be approximate
        PitchAxis  = np.array(PitchAxis, dtype=np.float64)
        PitchAxis /= np.sqrt(PitchAxis.dot(PitchAxis))

        # Roll axis is inferred
        RollAxis  = np.cross(PitchAxis, YawAxis)
        RollAxis /= np.sqrt(RollAxis.dot(RollAxis))

        # correct Pitch axis
        PitchAxis = np.cross(YawAxis, RollAxis)
        PitchAxis /= np.sqrt(PitchAxis.dot(PitchAxis))

        # FlowLines are used to infer the final flow direction
        DragLine = cgns.castNode(D.line((0,0,0),(1,0,0),2))
        SideLine = cgns.castNode(D.line((0,0,0),(0,1,0),2))
        LiftLine = cgns.castNode(D.line((0,0,0),(0,0,1),2))
        FlowLines = [DragLine, SideLine, LiftLine]

        # Put FlowLines in Aircraft's frame
        zero = (0,0,0)
        InitialFrame =  [       [1,0,0],         [0,1,0],       [0,0,1]]
        AircraftFrame = [list(RollAxis), list(PitchAxis), list(YawAxis)]
        T._rotate(FlowLines, zero, InitialFrame, AircraftFrame)

        # Apply Flow angles with respect to Airfraft's frame
        T._rotate(FlowLines, zero, list(PitchAxis), -AngleOfAttackDeg)
        T._rotate(FlowLines, zero,   list(YawAxis),  AngleOfSlipDeg)

        DragDirection = getDirectionFromLine(DragLine)
        SideDirection = getDirectionFromLine(SideLine)
        LiftDirection = getDirectionFromLine(LiftLine)

        return DragDirection, SideDirection, LiftDirection

    def compute_flow_and_turbulence(self):
        super().compute_flow_and_turbulence()
        self.set_reference_values()

    def set_reference_values(self):
        self.ApplicationContext['FluxCoef'] = 1./ (self.Flow['PressureDynamic'] * self.ApplicationContext['Surface'])
        self.ApplicationContext['TorqueCoef'] = self.ApplicationContext['FluxCoef'] / self.ApplicationContext['Chord']
        self.Flow['Reynolds'] = self.Flow['Density'] * self.Flow['VelocityUsedForScalingAndTurbulence'] * self.ApplicationContext['Chord'] / self.Flow['ViscosityMolecular']

    def set_TransitionZones(self):
        if self.Turbulence['TransitionMode'] is not None:
            self.Turbulence['TransitionZones'] = dict(
                TopOrigin                   = 0.002,
                BottomOrigin                = 0.010,
                TopLaminarImposedUpTo       = 0.001,
                TopLaminarIfFailureUpTo     = 0.2,
                TopTurbulentImposedFrom     = 0.995,
                BottomLaminarImposedUpTo    = 0.001,
                BottomLaminarIfFailureUpTo  = 0.2,
                BottomTurbulentImposedFrom  = 0.995,
            )
