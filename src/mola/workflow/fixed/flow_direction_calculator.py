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

import mola.math_tools as mt

def from_two_angles_and_aircraft_yaw_pitch_axis(
        angle_of_attack_deg : float, angle_of_slip_deg : float,
        yaw_axis : np.ndarray , pitch_axis : np.ndarray ):

        '''
        Compute the main flow directions from angle of attack and slip and aircraft
        yaw and pitch axis. The resulting directions can be used to impose inflow
        conditions and to compute aero-forces (Drag, Side, Lift) by projection of
        cartesian (X, Y, Z) forces onto the corresponding Flow Direction.

        Parameters
        ----------

            angle_of_attack_deg : float
                Angle-of-attack in degree. A positive
                angle-of-attack has an analogous impact as making a rotation of the
                aircraft around the **pitch_axis**, and this will likely contribute in
                increasing the Lift force component.

            angle_of_slip_deg : float
                Angle-of-attack in degree. A positive
                angle-of-slip has an analogous impact as making a rotation of the
                aircraft around the **yaw_axis**, and this will likely contribute in
                increasing the Side force component.

            yaw_axis : array of 3 :py:class:`float`
                Vector indicating the Yaw-axis of the
                aircraft, which commonly points towards the top side of the aircraft.
                A positive rotation around **yaw_axis** is commonly produced by applying
                left-pedal rudder (rotation towards the left side of the aircraft).
                This left-pedal rudder application will commonly produce a positive
                angle-of-slip and thus a positive side force.

            pitch_axis : array of 3 :py:class:`float`
                Vector indicating the Pitch-axis of the
                aircraft, which commonly points towards the right side of the
                aircraft. A positive rotation around **pitch_axis** is commonly produced
                by pulling the elevator, provoking a rotation towards the top side
                of the aircraft. By pulling the elevator, a positive angle-of-attack
                is created, which commonly produces an increase of Lift force.

        Returns
        -------

            drag_direction : array of 3 :py:class:`float`
                Vector indicating the main flow
                direction. The Drag force is obtained by projection of the absolute
                (X, Y, Z) forces onto this vector. The inflow vector for reference
                state is also obtained by projection of the momentum magnitude onto
                this vector.

            side_direction : array of 3 :py:class:`float`
                Vector normal to the main flow
                direction pointing towards the Side direction. The Side force is
                obtained by projection of the absolute (X, Y, Z) forces onto this
                vector.

            lift_direction : array of 3 :py:class:`float`
                Vector normal to the main flow
                direction pointing towards the Lift direction. The Lift force is
                obtained by projection of the absolute (X, Y, Z) forces onto this
                vector.
        '''

        
        yaw_axis    = np.array(yaw_axis, dtype=np.float64) # must be exact
        mt.normalize_numpy_vector(yaw_axis)

        pitch_axis  = np.array(pitch_axis, dtype=np.float64) # may be approximate
        mt.normalize_numpy_vector(pitch_axis)

        roll_axis  = np.cross(pitch_axis, yaw_axis) # is inferred
        mt.normalize_numpy_vector(roll_axis)

        pitch_axis = np.cross(yaw_axis, roll_axis) # correction
        mt.normalize_numpy_vector(pitch_axis)

        rotator = mt.rotate_3d_vector_from_axis_and_angle_in_degrees

        def get_new_axis_after_rotations(axis):
            new_axis = rotator(axis, pitch_axis, -angle_of_attack_deg)
            return rotator(new_axis, yaw_axis, -angle_of_slip_deg)
              

        drag_direction = get_new_axis_after_rotations(roll_axis)
        lift_direction = get_new_axis_after_rotations(yaw_axis)
        side_direction = get_new_axis_after_rotations(-pitch_axis)

        return drag_direction, side_direction, lift_direction
