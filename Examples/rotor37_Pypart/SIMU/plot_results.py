#!/usr/bin/python

# Python general packages
import os
import numpy as np
import copy

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Cassiopee packages
import Converter.PyTree   as C
import Converter.Internal as I


# def convertPyTree2Dict(t, container=I.__FlowSolutionCenters__):
#     data = dict()
#     for zone in I.getZones(t):
#         ZoneName = I.getName(zone)
#         VarNames, = C.getVarNames(zone, excludeXYZ=True)
#         VarNames = [var.replace('centers:', '').replace('nodes:', '') for var in VarNames]
#         FlowSol = I.getNodeFromName1(zone, container)
#         data[ZoneName] = dict()
#         loadsSubset = data[ZoneName]
#         if FlowSol:
#             for VarName in VarNames:
#                 Var = I.getNodeFromName1(FlowSol, VarName)
#                 if Var:
#                     loadsSubset[VarName] = Var[1]
#     return data

def convertPyTree2Dict(t, container=I.__FlowSolutionCenters__, average='massflow'):
    data = dict()
    for zone in I.getZones(t):
        ZoneName = I.getName(zone)
        VarNames, = C.getVarNames(zone, excludeXYZ=True)
        VarNames = [var.replace('centers:', '').replace('nodes:', '') for var in VarNames]
        FlowSol = I.getNodeFromName1(zone, container)
        data[average] = dict()
        loadsSubset = data[average]
        if FlowSol:
            for VarName in VarNames:
                Var = I.getNodeFromName1(FlowSol, VarName)
                if Var:
                    loadsSubset[VarName] = Var[1]
    return data

def plotRadialProfile(data, data2=None, variables=None, label=None, label2=None,
    filename='radialProfiles.pdf', container=I.__FlowSolutionCenters__):
    '''
    Plot radial profiles

    Parameters
    ----------

        data : :py:class:`dict` or PyTree or :py:class:`str`
            Data to plot. Should be either:

            * a :py:class:`dict` like:

              >>> data[avgType][var]

            * a PyTree: in this case it is converted into a :py:class:`dict`

            * a :py:class:`str` cooresponding to a file name: in this case the
              file is converted in a PyTree then a :py:class:`dict`

        data2 : idem that **data** or :py:obj:'None'
            If not :py:obj:'None', second data to plot

        variables : :py:class:`list` of :py:class:`str`
            Name of variables to plot

        label : :py:class:`str` or :py:obj:'None'
            label of **data** in plots

        label2 : :py:class:`str` or :py:obj:'None'
            label of **data2** in plots

        filename : py:class:`str`
            Name of the multi-pages PDF file to write

    '''

    ExtractionInfo = None
    ExtractionInfo2 = None

    if isinstance(data, str):
        data = C.convertFile2PyTree(data)
    if data2 and isinstance(data2, str):
        data2 = C.convertFile2PyTree(data2)

    if isinstance(data, list):
        ExtractionInfo = I.getNodeFromNameAndType(data, '.ExtractionInfo', 'UserDefinedData_t')
        data = convertPyTree2Dict(data, container=container)
    if data2 and isinstance(data2, list):
        ExtractionInfo2 = I.getNodeFromNameAndType(data2, '.ExtractionInfo', 'UserDefinedData_t')
        data2 = convertPyTree2Dict(data2, container=container)


    with PdfPages(filename) as pdf:
        # First pages with infomation
        txt = 'RADIAL PROFILES\n\n'
        for info in [ExtractionInfo, ExtractionInfo2]:
            if info:
                ExtractionType = I.getValue(I.getNodeFromName(info, 'type'))
                if ExtractionType == 'IsoSurface':
                    field = I.getValue(I.getNodeFromName(info, 'field'))
                    value = I.getValue(I.getNodeFromName(info, 'value'))
                    txt += 'IsoSurface {} = {}\n'.format(field, value)
                ReferenceRow = I.getNodeFromName(info, 'ReferenceRow')
                tag = I.getNodeFromName(ExtractionInfo, 'tag')
                if ReferenceRow and tag:
                    txt += '{} {}\n'.format(I.getValue(ReferenceRow), I.getValue(tag))
        firstPage = plt.figure()
        firstPage.clf()
        firstPage.text(0.5,0.5,txt, transform=firstPage.transFigure, size=12, ha="center", va="center")
        pdf.savefig()
        plt.close()

        input_variables = copy.deepcopy(variables)

        for avgType in data:
            if not input_variables:
                variables = data[avgType].keys()
            for var in variables:
                if var == 'ChannelHeight' or (var not in data[avgType]):
                    continue
                print('  > plot {}'.format(var))
                plt.figure()
                plt.plot(data[avgType][var], data[avgType]['ChannelHeight']*100., 'b-', label=label)
                if data2 and var in data2[avgType]:
                    plt.plot(data2[avgType][var], data2[avgType]['ChannelHeight']*100., 'r-', label=label2)
                plt.xlabel(var)
                plt.ylabel('Channel Height (%)')
                plt.grid()
                plt.legend()
                plt.title('{} {}'.format(avgType, var))

                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()

def getExtractionInfo(surface):
    '''
    Get information into ``.ExtractionInfo`` of **surface**.

    Parameters
    ----------

        surface : PyTree
            Base corresponding to a surface, with a ``.ExtractionInfo`` node

    Returns
    -------

        info : dict
            dictionary with the template:

            >>> info[nodeName] = nodeValue

    '''
    ExtractionInfo = I.getNodeFromNameAndType(surface, '.ExtractionInfo', 'UserDefinedData_t')
    info = dict((I.getName(node), str(I.getValue(node))) for node in I.getChildren(ExtractionInfo))
    return info

def getBaseFromExtractionInfo(surfaces, **kwargs):
    '''
    Inside a top tree **surfaces**, search for the base with a ``.ExtractionInfo``
    matching the requirements in **kwargs**

    Parameters
    ----------

        surfaces : PyTree
            top tree

        kwargs : unwrapped :py:class:`dict`
            parameters required in the ``.ExtractionInfo`` node of the searched
            base

    Returns
    -------

        surface : PyTree or :py:obj:`None`
            base that matches with **kwargs**
    '''
    for surface in I.getNodesFromType1(surfaces, 'CGNSBase_t'):
        info = getExtractionInfo(surface)
        if all([key in info and info[key] == value for key, value in kwargs.items()]):
            return surface
    return None

def getZoneFromExtractionInfo(surfaces, **kwargs):
    '''
    Inside a top tree **surfaces**, search for the zone with a ``.ExtractionInfo``
    matching the requirements in **kwargs**

    Parameters
    ----------

        surfaces : PyTree
            top tree

        kwargs : unwrapped :py:class:`dict`
            parameters required in the ``.ExtractionInfo`` node of the searched
            zone

    Returns
    -------

        surface : PyTree or :py:obj:`None`
            zone that matches with **kwargs**
    '''
    for surface in I.getZones(surfaces):
        info = getExtractionInfo(surface)
        if all([key in info and info[key] == value for key, value in kwargs.items()]):
            return surface
    return None


if __name__ == '__main__':

    # > USER DATA
    POST_DIR = 'OUTPUT'

    ############################################################################
    # PLOT DATA
    all_post = C.convertFile2PyTree(os.path.join(POST_DIR, 'all_post.cgns'))
    radialProfiles = I.getNodeFromNameAndType(all_post, 'RadialProfiles', 'CGNSBase_t')
    compareRadialProfiles = I.getNodeFromNameAndType(all_post, 'compareRadialProfiles', 'CGNSBase_t')
    os.chdir(POST_DIR)

    avg = 'massflow'
    fanUps = getZoneFromExtractionInfo(radialProfiles,
        ReferenceRow='R37',  tag='InletPlane', average=avg)
    fanDowns = getZoneFromExtractionInfo(radialProfiles,
        ReferenceRow='R37',  tag='OutletPlane', average=avg)

    plotRadialProfile(fanUps, data2=fanDowns,
                label='upstream', label2='downstream',
                filename='radialProfiles_R37.pdf')

    plotRadialProfile(compareRadialProfiles, container=I.__FlowSolutionNodes__,
                filename='radialProfiles_R37_perfo.pdf')
