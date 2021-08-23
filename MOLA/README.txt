-----------------------------------------------------------
******* MOdules pour de Logiciels en Aerodynamique ********
-----------------------------------------------------------

Version: Dev

Contents:

    Coprocess.py
    _cpmv_.py
    GenerativeShapeDesign.py
    GenerativeVolumeDesign.py
    InternalShortcuts.py
    ! Interpolate.py # not fully functional yet
    LiftingLine.py
    ! Pilot.py # deleted
    Postprocess.py
    Preprocess.py
    PropellerAnalysis.py
    RotatoryWings.py
    Visu.py
    ! VortexParticleMethodPrototype.py # to be deprecated
    Wireframe.py
    WorkflowAirfoil.py
    XFoil.py


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
INSTRUCTIONS FOR USING MOLA :
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


1/ MOLA requires Cassiopee environment (NOT necessarily elsA)

Some default environment source files can be found at

/home/lbernard/MOLA/1.11/ENVIRONMENTS/



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Changelog
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

v1.11 - 20/07/2020 - Major improvements on most modules. Introduction
    of stable versions of Workflows: Standard and Airfoil.

v1.10 - 19/03/2020 - Temporary working version of MOLA. Workflow Standard and
    Preprocess modules were introduced in this version

v1.9 - 23/11/2020 - Major developments and bug fixes in most modules. Major
    changes on the structure of MOLA. Important note: now, user has to import
    MOLA modules with prefix like: "import MOLA.TheModuleToBeImported".
    New WorkflowAirfoil module included.

v1.8 - 23/05/2020 - Major developments and bug fixes in
    PropellerAnalysis.py, including improved Adkins' and Drela
    algorithms, trim based on RPM, unsteady LiftingLine prototype
    functions, improvements of OutOfRange handling with structured
    PyZonePolars. Major bug fix of Drela algorithm: back to local
    sectional converged algorithm (more robust). Examples invite
    user to employ adapted-grid strategy.
    Corrections in InternalShortcuts J.secant() and J.get().

v1.7 - 19/04/2020 - Major developments in PropellerAnalysis.py
    including: enhanced Drela's algorithm, Trim capacities,
    enhanced MIL design, direct constrained design, support
    of unstructured Polars data, support of foil-distribution
    data on polar, creation of postLiftingLine2Surface(),
    new interface computeBEMT() with regression-impact of
    arguments (due to new design of the BEMT function).
    Creation of XFoil wrapper and generalization of airfoil's
    polar generation.
    New general-purpose interpolation macro in InternalShorcuts
    and scalar root finding algorithm (secant).
    Homogenization of interpolation macros on GSD.
v1.6 - 27/01/2020 - Bug corrections in PropellerAnalysis.py.
    Improved GSD.scanWing() function.
v1.5 - 05/12/2019 - Correction of a series of bugs
v1.4 - 08/11/2019 - MOLA adapted to Python v3
v1.3 - 30/10/2019 - Improvements to RotatoryWings,
    PropellerAnalysis, InternalShortcuts, GVD and GSD
v1.2 - 03/10/2019 - Correction of bugs and improvements on GVD and RW.
    - Addition of PropellerAnalysis.py
v1.1 - 18/07/2019 - Minor correction bugs and improvements on GVD.
v1.0 - 17/07/2019 - First functional version of MOLA
