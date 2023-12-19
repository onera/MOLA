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

'''
The main function is computeBodyForce, called by Coprocess.updateBodyForce for coprocessing.
Other functions in this file are useful functions for several body-force models.

File history:

8/09/2022 - T. Bontemps - Creation

19/12/2023 - T. Bontemps - Split in several files
'''

import MOLA

if not MOLA.__ONLY_DOC__:
    import numpy as np
    import scipy

    import Converter.PyTree as C
    import Converter.Internal as I

import MOLA.InternalShortcuts as J
from . import Models as BFM

AvailableBodyForceModels = dict(
    blockage = BFM.BodyForceModel_blockage,
    blockage_correction = BFM.BodyForceModel_blockage_correction,
    hall_without_blockage = BFM.BodyForceModel_hall_without_blockage,
    HallThollet = BFM.BodyForceModel_HallThollet,
    ThrustSpread = BFM.BodyForceModel_ThrustSpread,
    constant = BFM.BodyForceModel_constant,
    ShockWaveLoss = BFM.BodyForceModel_ShockWaveLoss,
    EndWallsProtection = BFM.BodyForceModel_EndWallsProtection,
    spreadPressureLossAlongChord = BFM.spreadPressureLossAlongChord,
    Roberts1988 = BFM.BodyForceModel_Roberts1988,
)

def computeBodyForce(t, BodyForceParameters):
    '''
    Compute Body force source terms.

    Parameters
    ----------
        t : PyTree

            Tree in which the source terms will be compute
        
        BodyForceParameters : dict
            Body force parameters for the current family.

    Returns
    -------
        dict
            New source terms to apply. Should be for example : 

            >>> TotalSourceTermsGloblal['zoneName'] = dict(Density=ndarray, MomentumX=ndarray, ...)

            For each zone, Density, MomentumX, MomentumY, MomentumZ and EnergyStagnationDensity are 
            body force source terms (corresponding to a volumic force, in N/m3)

    '''
    # Get the list of source terms to compute
    if not isinstance(BodyForceParameters, list):
        BodyForceParameters = [BodyForceParameters]

    # Compute and gather all the required source terms
    TotalSourceTermsGlobal = dict()
    for modelParameters in BodyForceParameters:
        model = modelParameters.pop('model')
        NewSourceTermsGlobal = AvailableBodyForceModels[model](t, modelParameters)
        # Add the computed source terms to the total source terms
        addDictionaries(TotalSourceTermsGlobal, NewSourceTermsGlobal)

    return TotalSourceTermsGlobal


def addDictionaries(d1, d2):
    '''
    Update **d1** by adding values of **d2** to values in **d1**

    .. important:: 

        Dictionaries must have two levels like that:

        >>> d1['zoneName']['Density'] = np.ndarray(...)

    Parameters
    ----------

        d1 : dict
            Dictionary that will be updated
        
        d2 : dict
            Dictionary that will be added to **d1**

    '''
    for zone in d2:
        if not zone in d1:
            d1[zone] = d2[zone]
        else:
            for key, value in d2[zone].items():
                if key in d1[zone]:
                    d1[zone][key] += value
                else:
                    d1[zone][key] = value

def getAdditionalFields(zone, FluidProperties, RotationSpeed, tol=1e-5):
    '''
    Compute additional flow quantities used in body-force models, and store them into a
    temporary node in **zone** to have access to them later if the function is called more 
    than once.

    Parameters
    ----------

        zone : PyTree
            Current zone
        
        FluidProperties : dict
            as read in `setup.py`

        RotationSpeed : float
            Rotation speed of the current zone

        tol : float
            minimum value for quantities used as a denominator.

    Returns
    -------
    
        dict

            Newly computed quantities
    '''
    tmpMOLAFlowNode = I.getNodeFromName(zone, 'FlowSolution#tmpMOLAFlow')
    
    if tmpMOLAFlowNode:
        return J.getVars2Dict(zone, Container='FlowSolution#tmpMOLAFlow')

    FlowSolution    = J.getVars2Dict(zone, Container='FlowSolution#Init')
    DataSourceTerms = J.getVars2Dict(zone, Container='FlowSolution#DataSourceTerm')
    # Variables needed in DataSourceTerms: 'radius', 'theta', 'blockage', 'nx', 'nr', 'nt', 'AbscissaFromLE', 'blade2BladeDistance'
    # Optional variables in DataSourceTerms: 'delta0'

    # Coordinates
    cosTheta = np.cos(DataSourceTerms['theta'])
    sinTheta = np.sin(DataSourceTerms['theta'])

    # Flow data
    Density = np.maximum(FlowSolution['Density'], tol)
    Vx, Vy, Vz = FlowSolution['MomentumX']/Density, FlowSolution['MomentumY']/Density, FlowSolution['MomentumZ']/Density
    Wx, Wr, Wt = Vx, Vy*cosTheta+Vz*sinTheta, -Vy*sinTheta + Vz*cosTheta - DataSourceTerms['radius'] * RotationSpeed
    Vmag = (Vx**2 + Vy**2 + Vz**2)**0.5
    Wmag = np.maximum(tol, (Wx**2 + Wr**2 + Wt**2)**0.5)
    Temperature = np.maximum(tol, (FlowSolution['EnergyStagnationDensity']/Density-0.5*Vmag**2.)/FluidProperties['cv'])
    Mrel = Wmag/(FluidProperties['Gamma']*FluidProperties['IdealGasConstant']*Temperature)**0.5
    Mabs = Vmag/(FluidProperties['Gamma']*FluidProperties['IdealGasConstant']*Temperature)**0.5
    funMach = lambda M: (1 + (FluidProperties['Gamma']-1)/2 * M**2) ** (FluidProperties['Gamma']/(FluidProperties['Gamma']-1))
    PressureStagnationRel = FlowSolution['PressureStagnation'] * funMach(Mrel) / funMach(Mabs)

    # Velocity normal and parallel to the skeleton
    # See Cyril Dosnes bibliography synthesis for the local frame of reference
    Wn = np.absolute(Wx*DataSourceTerms['nx'] + Wr*DataSourceTerms['nr'] + Wt*DataSourceTerms['nt']) # Velocity component normal to the blade surface
    Wnx = Wn * DataSourceTerms['nx'] * np.sign(Wx*DataSourceTerms['nx'])
    Wnr = Wn * DataSourceTerms['nr'] * np.sign(Wr*DataSourceTerms['nr'])
    Wnt = Wn * DataSourceTerms['nt'] * np.sign(Wt*DataSourceTerms['nt'])
    Wpx, Wpr, Wpt = Wx-Wnx,   Wr-Wnr,   Wt-Wnt # Velocity component in the plane tangent to the blade surface
    Wp = np.maximum(tol, (Wpx**2+Wpr**2+Wpt**2)**0.5)

    # Deviation of the flow with respect to the blade surface
    # Careful to the sign
    incidence = np.arcsin(Wn/Wmag) 
    incidence*= np.sign(Wx*DataSourceTerms['nx'] + Wr*DataSourceTerms['nr'] + Wt*DataSourceTerms['nt'])

    tmpMOLAFlow = dict(
        theta = DataSourceTerms['theta'],

        Vx = Vx,
        Vy = Vy, 
        Vz = Vz,
        Vmag = Vmag,

        Wx = Wx,
        Wr = Wr,
        Wt = Wt,
        Wmag = Wmag,
        Wn = Wn,
        Wnx = Wnx,
        Wnr = Wnr,
        Wnt = Wnt,
        Wp = Wp,
        Wpx = Wpx,
        Wpr = Wpr,
        Wpt = Wpt,

        # PressureDynamicRel = 0.5*Density*Wmag**2,
        PressureDynamicRel = PressureStagnationRel - FlowSolution['Pressure'],
        Temperature = Temperature,
        PressureStagnationRel = PressureStagnationRel,
        Mrel = Mrel,

        incidence = incidence,

        # Unit vector normal the velocity. Direction of application of the normal force
        unitVectorNormalX = np.cos(incidence) * DataSourceTerms['nx'] - np.sin(incidence)*Wpx/Wp,
        unitVectorNormalR = np.cos(incidence) * DataSourceTerms['nr'] - np.sin(incidence)*Wpr/Wp,
        unitVectorNormalT = np.cos(incidence) * DataSourceTerms['nt'] - np.sin(incidence)*Wpt/Wp,

        # Unit vector parallel to the velocity. Direction of application of the parallel force
        unitVectorParallelX = - Wx / Wmag,
        unitVectorParallelR = - Wr / Wmag,
        unitVectorParallelT = - Wt / Wmag,

    )

    J.set(zone, 'FlowSolution#tmpMOLAFlow', childType='FlowSolution_t', **tmpMOLAFlow)
    tmpMOLAFlowNode = I.getNodeFromName(zone, 'FlowSolution#tmpMOLAFlow')
    I.createChild(tmpMOLAFlowNode, 'GridLocation', 'GridLocation_t', value='CellCenter')

    return tmpMOLAFlow 

def getForceComponents(fn, fp, tmpMOLAFlow):
    '''
    Compute cartesian and cylindrical components of the force with its components in the blade local frame.

    Parameters
    ----------

        fn : numpy.ndarray
            Force component in the direction normal to the chord
        
        fp : numpy.ndarray
            Force component in the direction parallel to the chord (oriented upstream)

        tmpMOLAFlow : dict
            temporary container of flow quantities, as got by :py:func:`getAdditionalFields`

    Returns
    -------
    
        :py:class:`tuple` of :py:class:`numpy.ndarray`

        Force components in x, y, z, r and theta. 
    '''

    # Force in the cylindrical frame of reference
    fx = fn * tmpMOLAFlow['unitVectorNormalX'] + fp * tmpMOLAFlow['unitVectorParallelX']
    fr = fn * tmpMOLAFlow['unitVectorNormalR'] + fp * tmpMOLAFlow['unitVectorParallelR']
    ft = fn * tmpMOLAFlow['unitVectorNormalT'] + fp * tmpMOLAFlow['unitVectorParallelT']

    # Force in the cartesian frame of reference
    fy = -np.sin(tmpMOLAFlow['theta']) * ft + np.cos(tmpMOLAFlow['theta']) * fr
    fz =  np.cos(tmpMOLAFlow['theta']) * ft + np.sin(tmpMOLAFlow['theta']) * fr

    return fx, fy, fz, fr, ft 

def getFieldsAtLeadingEdge(t, abscissa=1e-3,  filename=None, localComm=None):
    return getFieldsAtIsoAbscissaFromLE(t, abscissa, filename, localComm)

def getFieldsAtTrailingEdge(t, abscissa=1-1e-3, filename=None, localComm=None):
    return getFieldsAtIsoAbscissaFromLE(t, abscissa, filename, localComm)

def getFieldsAtIsoAbscissaFromLE(t, abscissa, filename=None, localComm=None):
    '''
    Perfo an iso surface for a constant abscissa from leading edge.

    The extraction is gathered on all ranks.

    Parameters
    ----------
    t : PyTree
        Current PyTree given by the :py:mod:`BodyForceModels` module, restricted to zones belonging to 
        the family involved in the body force modelling.

    abscissa : float
        Value of ``'AbscissaFromLE'`` to perform the iso-surface (should be near one).

        .. danger::

            If this value is to low, the extraction could be not correctly performed.

    filename : :py:class:`str` or :py:obj:`None`
        If not :py:obj:`None`, save the extracted surface in a file. 
        Only rank 0 writes this file, but all ranks have the whole data.
    
    localComm : MPI communicator
        A given sub-communicator to restrein MPI exchanges the iso-surface extracting to ranks involved with this communicator.
        If :py:obj:`None`, take MPI.COMM_WORLD

    Returns
    -------
    PyTree
        2D surface.

    '''
    from MOLA.Postprocess import isoSurfaceAllGather, comm
    if localComm is None:
        localComm = comm
    rank = localComm.Get_rank()

    IsoSurface = isoSurfaceAllGather(t, fieldname='AbscissaFromLE', value=abscissa, container='FlowSolution#DataSourceTerm', localComm=localComm)
    rank = localComm.Get_rank()
    if filename and rank == 0:
        J.save(IsoSurface, filename)

    return IsoSurface

def getInterpolatorsInHeightAndTheta(surface, variables, Container='FlowSolution#tmpMOLAFlowV'):
    '''
    Build 2D interpolators for the required **variables** depending on ``ChannelHeight`` and ``theta``. 
    Starting from a 2D **surface** that could be a revolution of the leading edge (got from 
    :py:func:`getFieldsAtLeadingEdge`) or representing averaged quantities on a row, this function 
    allows to use these quantities on the whole multi-zones, 3D domain of the current row.

    .. note::

        Interpolators are built with ``scipy.interpolate.NearestNDInterpolator``.

    Parameters
    ----------
    surface : PyTree
        Must be a 2D tree.

    variables : list
        List of variables to build interpolators.

    Container : str, optional
        Container which contains **variables**, by default 'FlowSolution#tmpMOLAFlowV'

    Returns
    -------
    dict
        Dictionary of 2D interpolators. Each key corresponds to a variable. To call
        an interpolator for given values of ChannelHeight and theta, use for instance:

        >> interpDict['MomentumX'](h, theta)

    '''

    hlist = []
    thetalist = []
    varDict = dict((var, []) for var in variables)
    for zone in I.getZones(surface):
        FS = I.getNodeFromName1(zone, 'FlowSolution#DataSourceTermV')
        h = I.getValue(I.getNodeFromName1(FS, 'ChannelHeight'))
        theta = I.getValue(I.getNodeFromName1(FS, 'theta'))

        hlist.extend(list(h))
        thetalist.extend(list(theta))

        FlowSolution = J.getVars2Dict(zone, VariablesName=variables, Container=Container)
        for var in variables:
            assert FlowSolution[var][0] is not None, J.FAIL+f'{var} is not found in {Container}'+J.ENDC
            varDict[var].extend(list(FlowSolution[var]))

    interpDict = dict()
    for var in variables:
        interpDict[var] = scipy.interpolate.NearestNDInterpolator(list(zip(hlist, thetalist)), varDict[var])

    return interpDict
