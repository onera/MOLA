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


from ..interface import WorkflowRotatingComponentInterface, Union, np


class WorkflowPropellerInterface(WorkflowRotatingComponentInterface):

    def __init__(self, workflow, tree=None, **kwargs):
        super().__init__(workflow, tree, **kwargs)
        if tree is None:
            # FIXME elsa extraction of Pressure in WallInviscid
            # self.add_to_Extractions_BC(Source='WallViscous', 
            #     Fields=['Pressure', #'BoundaryLayer', 'yPlus'
            #             ])
            self.add_to_Extractions_Integral(
                Source='Wall*',
                Fields=['Force', 'Torque'],
                PostprocessOperations=[
                    dict(Type="compute_propeller_coefficients", AtEndOfRunOnly=False),
                    dict(Type='avg', Variable='Thrust', AtEndOfRunOnly=False),
                    dict(Type='std', Variable='Thrust', AtEndOfRunOnly=False),
                    dict(Type='avg', Variable='Power', AtEndOfRunOnly=False),
                    dict(Type='std', Variable='Power', AtEndOfRunOnly=False),
                ]
            )

    def set_ApplicationContext(self, 
        ShaftAxis : Union[list,
                        tuple,
                        np.ndarray] = [1,0,0],
        ShaftRotationSpeedUnit : str = 'rpm', 
        HubRotationIntervals : list = None,
        Surface : float = 1.0,
        Length : float = 1.0,
        NormalizationCoefficient : dict = None,
        NumberOfBladesSimulated : int = 1,
        NumberOfBladesInInitialMesh : int = 1,
        IsRotating : bool = True,
        TurbulenceSetAtRelativeRadius : float = 0.75,
        Rows : dict = dict(),
        *,        
        NumberOfBlades : int,
        ShaftRotationSpeed : Union[float, int] = None,
        ):

        Rows = dict( Propeller = dict(
                        IsRotating = IsRotating,
                        NumberOfBladesSimulated = NumberOfBladesSimulated,
                        NumberOfBladesInInitialMesh = NumberOfBladesInInitialMesh,
                        NumberOfBlades = NumberOfBlades,
                        )
        )

        ShaftAxis = np.array(ShaftAxis,dtype=float)
        ShaftAxis /= np.linalg.norm(ShaftAxis)

        kwargs = self.get_default_values_from_local_signature()
        self.ApplicationContext = self._get_comp(WorkflowPropellerInterface.set_ApplicationContext, kwargs)

        for key, row_parameters in Rows.items():
            self.add_Row_to_ApplicationContext(_Key=key, **row_parameters)

        self.apply_ShaftRotationSpeedUnit(default_ShaftRotationSpeedUnit=kwargs["ShaftRotationSpeedUnit"])
        if 'HubRotationIntervals' in self.ApplicationContext:
            self.set_HubRotationIntervals()


    def set_SplittingAndDistribution(self,
            Strategy    : str = 'AtComputation',
            Splitter    : str = 'PyPart',
            Distributor : str = 'PyPart',
            **kwargs):

        super().set_SplittingAndDistribution(
            **self.get_default_values_from_local_signature(),
            **kwargs)
