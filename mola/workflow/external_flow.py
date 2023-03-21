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


class FlowGenerator(object):

    def __init__(self, workflow):
        # Set attributes
        self.Fluid = workflow.Fluid
        self.Flow = workflow.Flow
        self.Turbulence = workflow.Turbulence
        
        if hasattr(workflow, 'Length'):
            self.Length = workflow.Length  # Mandatory to compute the Reynolds number
        else:
            self.Length = 1.

        # Set default values 
        self.Flow.setdefault('Density', 1.225)
        self.Flow.setdefault('Temperature', 288.15)
        self.Flow.setdefault('Velocity', 0.)
        self.Flow.setdefault('VelocityUsedForScalingAndTurbulence', None)
        self.Flow.setdefault('AngleOfAttackDeg', 0.)
        self.Flow.setdefault('AngleOfSlipDeg', 0.)
        self.Flow.setdefault('YawAxis', [0.,0.,1.])
        self.Flow.setdefault('PitchAxis', [0.,-1.,0.])

    def generate(self):
        # Compute flow and turbulence properties
        self.set_flow_properties()
        self.set_turbulence_properties()
        self.Flow['ReferenceState'] = dict(**self.Flow['Conservatives'], **self.Turbulence['FieldsTurbulence'])

        # TODO Move these values ?
        # If yes, Surface can be removed from the attributes of this class
        self.Flow.update(dict(
            Length          = Length,
            Surface         = Surface,
            FluxCoef        = 1./(self.Flow['PressureDynamic'] * Surface),
            TorqueCoef      = 1./(self.Flow['PressureDynamic'] * self.Surface*self.Length),
        ))

    def set_flow_properties(self):

        FreestreamIsTooLow = np.abs(self.Flow['Velocity']) < 1e-5
        if FreestreamIsTooLow and self.Flow['VelocityUsedForScalingAndTurbulence'] is None:
            ERRMSG = f'Velocity is too low ({self.Flow["Velocity"]}).'
            ERRMSG+= 'You must provide a non-zero value for VelocityUsedForScalingAndTurbulence'
            raise ValueError(J.FAIL+ERRMSG+J.ENDC)

        if self.Flow['VelocityUsedForScalingAndTurbulence'] is not None:
            if self.Flow['VelocityUsedForScalingAndTurbulence'] <= 0:
                ERRMSG = 'VelocityUsedForScalingAndTurbulence must be positive'
                raise ValueError(J.FAIL+ERRMSG+J.ENDC)
        else:
            self.Flow['VelocityUsedForScalingAndTurbulence'] = np.abs(self.Flow['Velocity'])

        # TODO Put ViscosityMolecular in the Fluid attribute ?
        def SutherlandLaw(T, mus, Ts, S):
            return mus * (T/Ts)**1.5 * ((Ts + S)/(T + S))
       
        ViscosityMolecular = SutherlandLaw(self.Flow['Temperature'], self.Fluid['SutherlandViscosity'], self.Fluid['SutherlandTemperature'], self.Fluid['SutherlandConstant'])

        Mach = self.Flow['VelocityUsedForScalingAndTurbulence'] /np.sqrt( self.Fluid['Gamma'] * self.Fluid['IdealGasConstant'] * self.Flow['Temperature'] )
        Reynolds = self.Flow['Density'] * self.Flow['VelocityUsedForScalingAndTurbulence'] * self.Length / ViscosityMolecular
        Pressure = self.Flow['Density'] * self.Fluid['IdealGasConstant'] * self.Flow['Temperature']
        PressureDynamic = 0.5 * self.Flow['Density'] * self.Flow['VelocityUsedForScalingAndTurbulence'] **2

        # Reference state (farfield)
        FlowDir = self.get_flow_directions(self.Flow['AngleOfAttackDeg'],
                                           self.Flow['AngleOfSlipDeg'],
                                           self.Flow['YawAxis'],
                                           self.Flow['PitchAxis']
                                           )
        DragDirection, SideDirection, LiftDirection= FlowDir
        MomentumX =  self.Flow['Density'] * self.Flow['Velocity'] * DragDirection[0]
        MomentumY =  self.Flow['Density'] * self.Flow['Velocity'] * DragDirection[1]
        MomentumZ =  self.Flow['Density'] * self.Flow['Velocity'] * DragDirection[2]
        EnergyStagnationDensity = self.Flow['Density'] * ( self.Fluid['cv'] * self.Flow['Temperature'] + 0.5 * self.Flow['Velocity'] **2)

        self.Flow['Conservatives'] = dict(
            Density = self.Flow['Density'],
            MomentumX = float(MomentumX),
            MomentumY = float(MomentumY),
            MomentumZ = float(MomentumZ),
            EnergyStagnationDensity = float(EnergyStagnationDensity)
        )

        # Update ReferenceValues dictionary
        self.Flow.update(dict(
            Reynolds                = Reynolds,
            Mach                    = Mach,
            DragDirection           = list(DragDirection),
            SideDirection           = list(SideDirection),
            LiftDirection           = list(LiftDirection),
            Pressure                = Pressure,
            PressureDynamic         = PressureDynamic,
            ViscosityMolecular      = ViscosityMolecular,
            ViscosityEddy           = self.Flow['Viscosity_EddyMolecularRatio'] * ViscosityMolecular,
            MomentumX               = MomentumX,
            MomentumY               = MomentumY,
            MomentumZ               = MomentumZ,
            EnergyStagnationDensity = EnergyStagnationDensity,
        ))

    def set_turbulence_properties(self):

        # -> for SA model
        def computeTurbulentSANuTilde(Nu, Nut_Nu):
            def computeEddyViscosityFromNuTilde(Nu, NuTilde):
                    '''
                    Compute cinematic ViscosityEddy using Eqn. (A1) of DOI:10.2514/6.1992-439
                    '''
                    Cnu1 = 7.1
                    f_nu1 = (NuTilde/Nu)**3 / ((NuTilde/Nu)**3 + Cnu1**3)
                    CinematicViscosityEddy = NuTilde * f_nu1
                    return CinematicViscosityEddy

            def residualEddyViscosityRatioFromGivenNuTilde(NuTilde):
                return Nut_Nu - computeEddyViscosityFromNuTilde(Nu, NuTilde) / Nu

            sol = J.secant(residualEddyViscosityRatioFromGivenNuTilde, x0=Nut_Nu*Nu, x1=1.5*Nut_Nu*Nu, ftol=Nut_Nu*0.001, bounds=(1e-14,1.e6))
            return sol['root']

        self.Turbulence['TurbulentSANuTilde'] = computeTurbulentSANuTilde(
                                                    Nu=self.Flow['ViscosityMolecular']/self.Flow['Density'],
                                                    Nut_Nu=self.Flow['Viscosity_EddyMolecularRatio']
                                                    )

        # -> for k-omega models
        TurbulentEnergyKineticDensity   = self.Flow['Density']*1.5*(self.Turbulence['Level']**2)*(self.Flow['VelocityUsedForScalingAndTurbulence']**2)
        TurbulentDissipationRateDensity = self.Flow['Density'] * TurbulentEnergyKineticDensity / (self.Flow['Viscosity_EddyMolecularRatio'] * self.Flow['ViscosityMolecular'])
        
        # -> for Smith k-l model
        k = TurbulentEnergyKineticDensity / self.Flow['Density']
        omega = TurbulentDissipationRateDensity / self.Flow['Density']
        TurbulentLengthScaleDensity = self.Flow['Density'] * k * 18.0**(1./3.) / (np.sqrt(2*k)*omega),
        
        # -> for k-kL model
        TurbulentEnergyKineticPLSDensity = TurbulentLengthScaleDensity*k

        # -> for Menter-Langtry assuming acceleration factor F(lambda_theta)=1
        IntermittencyDensity = self.Flow['Density'] * 1.0
        if TurbulenceLevel*100 <= 1.3:
            MomentumThicknessReynoldsDensity = self.Flow['Density'] * (1173.51 - 589.428*(self.Turbulence['Level']*100) + 0.2196*(self.Turbulence['Level']*100)**(-2.))
        else:
            MomentumThicknessReynoldsDensity = self.Flow['Density'] * ( 331.50*(self.Turbulence['Level']*100-0.5658)**(-0.671) )

        # -> for RSM models
        ReynoldsStressXX = ReynoldsStressYY = ReynoldsStressZZ = (2./3.) * TurbulentEnergyKineticDensity
        ReynoldsStressXY = ReynoldsStressXZ = ReynoldsStressYZ = 0.

        self.Turbulence.update(dict(
            TurbulentSANuTilde               = TurbulentSANuTilde,
            TurbulentEnergyKineticDensity    = TurbulentEnergyKineticDensity,
            TurbulentDissipationRateDensity  = TurbulentDissipationRateDensity,
            TurbulentLengthScaleDensity      = TurbulentLengthScaleDensity,
            TurbulentEnergyKineticPLSDensity = TurbulentEnergyKineticPLSDensity,
            IntermittencyDensity             = IntermittencyDensity,
            MomentumThicknessReynoldsDensity = MomentumThicknessReynoldsDensity,
            ReynoldsStressXX                 = ReynoldsStressXX,
            ReynoldsStressYY                 = ReynoldsStressYY,
            ReynoldsStressZZ                 = ReynoldsStressZZ,
            ReynoldsStressXY                 = ReynoldsStressXY,
            ReynoldsStressXZ                 = ReynoldsStressXZ,
            ReynoldsStressYZ                 = ReynoldsStressYZ,
            ReynoldsStressDissipationScale   = TurbulentDissipationRateDensity,
        ))

        if TurbulenceModel == 'SA':
            self.Turbulence['Conservatives'] = dict(
                TurbulentSANuTildeDensity = self.Turbulence['TurbulentSANuTilde'] * self.Flow['Density']
            )

        elif TurbulenceModel in K_OMEGA_TWO_EQN_MODELS:
            self.Turbulence['Conservatives'] = dict(
                TurbulentEnergyKineticDensity   = self.Turbulence['TurbulentEnergyKineticDensity'],
                TurbulentDissipationRateDensity = self.Turbulence['TurbulentDissipationRateDensity'],
            )

        elif TurbulenceModel == 'smith':
            self.Turbulence['Conservatives'] = dict(
                TurbulentEnergyKineticDensity   = self.Turbulence['TurbulentEnergyKineticDensity'],
                TurbulentLengthScaleDensity = self.Turbulence['TurbulentLengthScaleDensity'],
            )

        elif 'LM2009' in TurbulenceModel:
            self.Turbulence['Conservatives'] = dict(
                TurbulentEnergyKineticDensity    = self.Turbulence['TurbulentEnergyKineticDensity'],
                TurbulentDissipationRateDensity  = self.Turbulence['TurbulentDissipationRateDensity'],
                IntermittencyDensity             = self.Turbulence['IntermittencyDensity'],
                MomentumThicknessReynoldsDensity = self.Turbulence['MomentumThicknessReynoldsDensity'],
            )

        elif TurbulenceModel == 'SSG/LRR-RSM-w2012':
            self.Turbulence['Conservatives'] = dict(
                ReynoldsStressXX               = self.Turbulence['ReynoldsStressXX'],
                ReynoldsStressXY               = self.Turbulence['ReynoldsStressXY'],
                ReynoldsStressXZ               = self.Turbulence['ReynoldsStressXZ'],
                ReynoldsStressYY               = self.Turbulence['ReynoldsStressYY'],
                ReynoldsStressYZ               = self.Turbulence['ReynoldsStressYZ'],
                ReynoldsStressZZ               = self.Turbulence['ReynoldsStressZZ'],
                ReynoldsStressDissipationScale = self.Turbulence['ReynoldsStressDissipationScale'],
            )

        else:
            raise AttributeError(f'Turbulence model {self.Turbulence["Model"]} not implemented in workflow. Must be in: {AvailableTurbulenceModels}')

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

        import Transform.PyTree as T

        def getDirectionFromLine(line):
            x,y,z = J.getxyz(line)
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
        DragLine = D.line((0,0,0),(1,0,0),2)
        SideLine = D.line((0,0,0),(0,1,0),2)
        LiftLine = D.line((0,0,0),(0,0,1),2)
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

            
