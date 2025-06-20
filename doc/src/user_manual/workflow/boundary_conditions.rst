###################
Boundary conditions
###################

On this page, all boundary condition types are listed, with their corresponding dictionary structure.
For each, an example is given. The ``Type`` keyword is always required, and identifies the type of boundary condition.
The ``Family`` keyword is also required to indicate on which CGNS Family the boundary condition applies. In following
examples, it is set to ``myFamily``.

.. contents:: Table of contents

************************
Wall boundary conditions
************************

Wall
====

This condition is a shortcut to :ref:`WallInviscid` or :ref:`WallViscous` depending on the physics model selected.
If `Turbulence['Model']` is `Euler`, then this condition is equivalent to :ref:`WallInviscid`.
Otherwise, it corresponds to :ref:`WallViscous`. 

>>> dict(Type='Wall', Family='myFamily')

WallInviscid
============

A slip condition at wall:

>>> dict(Type='WallInviscid', Family='myFamily')

WallViscous
===========

An adiabatic, no-slip condition at wall:

>>> dict(Type='WallViscous', Family='myFamily')

SymmetryPlane
============

>>> dict(Type='SymmetryPlane', Family='myFamily')



*****************
Inflow conditions
*****************

Farfield
========

This condition imposes a reference state (taken from the parameters in ``Workflow.Flow``).

>>> dict(Type='Farfield', Family='myFamily')

InflowStagnation
================

This condition imposes total pressure, total enthalpy, flow direction and turbulence quantities.

>>> dict(Type='InflowStagnation', Family='myFamily')

Editable quantities are: `PressureStagnation`, `EnthalpyStagnation` (or `TemperatureStagnation` for an ideal gas),
`VelocityUnitVectorX`,  `VelocityUnitVectorY`,  `VelocityUnitVectorZ`. 

If a model of turbulence is activated, the following quantities are also editable: 
`TurbulenceLevel`, `Viscosity_EddyMolecularRatio` 
and model depending quantities (e.g., `TurbulentEnergyKinetic`, `TurbulentDissipationRate`, `TurbulentLengthScale`, etc).

InflowMassFlow
==============

This condition imposes a given MassFlow through the boundary, plus total enthalpy, flow direction and turbulence quantities.

>>> dict(Type='InflowMassFlow', Family='myFamily', MassFlow=10.)


In particular, either the massflow (``MassFlow``) or the surfacic massflow
(``SurfacicMassFlow``) may be specified.


If not given, `MassFlow` is taken from the attribute `Flow` of `Workflow`.

Alternatively, `SurfacicMassFlow` may be specified. 
In this case, `Surface` may be also given, Otherwise it is automatically computed.

Other editable quantities are: `EnthalpyStagnation` (or `TemperatureStagnation` for an ideal gas),
`VelocityUnitVectorX`,  `VelocityUnitVectorY`,  `VelocityUnitVectorZ`. 

If a model of turbulence is activated, the following quantities are also editable: 
`TurbulenceLevel`, `Viscosity_EddyMolecularRatio` 
and model depending quantities (e.g., `TurbulentEnergyKinetic`, `TurbulentDissipationRate`, `TurbulentLengthScale`, etc).


******************
Outflow conditions
******************

OutflowPressure
===============

>>> dict(Type='OutflowPressure', Family='myFamily')

The only editable quantity is `Pressure`.

OutflowMassFlow
===============

>>> dict(Type='OutflowMassFlow', Family='myFamily')

The only editable quantity is `MassFlow`.


OutflowRadialEquilibrium
========================

.. code-block:: python
    dict(Type='OutflowRadialEquilibrium', Family='myFamily', 
        valve_type=4, valve_ref_pres=0.75*Pt, valve_ref_mflow=5., valve_relax=0.3*Pt
        )

It defines an outflow condition imposing a radial equilibrium ('outradeq' in
*elsA*). The arguments have the same names that *elsA* keys. Valve law types
from 1 to 5 are available. The radial equilibrium without a valve law (with
**valve_type** = 0, which is the default value) is also available. To be
consistant with the condition 'OutflowPressure', the argument
**valve_ref_pres** may also be named **Pressure**.


***************************
Imposing data on a boundary
***************************

For inflow and outflow boudaries, imposed quantities are taken using from 
`Fluid`, `Flow` and `Turbulence` workflow attributes by default.

It is also permitted to give specific values, to impose uniform field, 1D varing field or a 2D mapping.

The behavior being the same for all inflow and outflow boundaries, it is detailed
below by taking the condition `InflowStagnation` and the parameter `PressureStagnation` for instance.


To prescribe a uniform value, different from the default value:

>>> dict(Type='InflowStagnation', Family='myFamily', PressureStagnation=100000.)

To prescribe a radial profile:

>>> dict(Type='InflowStagnation', Family='myFamily', PressureStagnation=funPt)

Same that before but imposing a radial profile of PressureStagnation given by the function 'funPt'
(must be defined before by the user). The function may be an analytical function or a interpoland 
computed by the user from a data set. If not given, the function argument is the variable 'ChannelHeight'.
Otherwise, the function argument has to be precised with the optional argument `variableForInterpolation`.
It must be one of 'ChannelHeight' (default value), 'Radius', 'CoordinateX', 'CoordinateY' or 'CoordinateZ'.

To prescribe a 2D map from a given file:

>>> dict(Type='InflowStagnation',  Family='myFamily', File='inflow.cgns')

It defines an inflow condition imposing stagnation quantities ('inj1' in
*elsA*) given a 2D map written in the given file (must be given at cell centers, 
in the container 'FlowSolution#Centers'). The flow field will be just copied, there is no 
interpolation (if needed, the user has to done that before and provide a ready-to-copy file).


*********************
Interstage conditions
*********************

For the example, it is assumed that there is only one interstage with both
families 'InterfaceUpstreamSide' and 'InterfaceDownstreamSide'. 

.. note:: Autogrid named by default these conditions 'Rotor_stator_10_left' and 'Rotorstator__10_right'. 

Mixing plane 
============

>>> dict(Type='MixingPlane', Family='InterfaceUpstreamSide', LinkedFamily='InterfaceDownstreamSide')


UnsteadyRotorStatorInterface 
============================

>>> dict(Type='UnsteadyRotorStatorInterface', Family='InterfaceUpstreamSide', LinkedFamily='InterfaceDownstreamSide')

It defines an unsteady interpolating interface using a sliding mesh techinque.
If **SectorPassagePeriod** is not provided, it is automatically computed using the formula:

.. math::
    D_m &= \frac{2}{\frac{K_1}{B_1}+\frac{K_2}{N_2}} \\
    T_{rot} &= \frac{2 \pi}{|\Omega|} \\
    SectorPassagePeriod &= \frac{T_{rot}}{D_m}

where:
    * :math:`\Omega` is `workflow.ApplicationContext['ShaftRotationSpeed']`
    * :math:`N_i` is `workflow.ApplicationContext['Rows'][<RowFamily_i>]['NumberOfBlades']`
    * :math:`K_i` is `workflow.ApplicationContext['Rows'][<RowFamily_i>]['NumberOfBladesSimulated']`

.. note:: 
    It correspond to an RNA interface in elsA (`stage_red_hybrid`)

ChorochronicInterface 
=====================

>>> dict(Type='ChorochronicInterface', Family='InterfaceUpstreamSide', LinkedFamily='InterfaceDownstreamSide') 

It defines a chorochronic interface ('stage_choro' in *elsA*), and update azimuthal periodic conditions
as chorochronic conditions.
The number of harmonics `NumberOfHarmonicsForFamily` and `NumberOfHarmonicsForLinkedFamily`
can be modified and are set to 20 by default. 

The parameter `hybrid` (:py:obj:`True` by default) indicates if structured or hybrid version of the condition is used.



