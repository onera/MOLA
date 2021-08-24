'''
MOLA 2 - Fields module

Operations on Flow Solution fields
'''

import numpy as np
import Converter.Internal as I

def new(t, FieldNames, Container='FlowSolution', LocationIfAbsent='Vertex'):
    '''
    Create (or set to 0) a list of fields at user-provided zones.

    INPUTS

    t - (PyTree, base, zone, list of zones) - CGNS element where new fields are
        to be created or initialized to 0

    FieldNames - (list of strings) - list of the new fields names

    Container - (string) - CGNS FlowSolution container name where the new
        fields are being created.

    LocationIfAbsent - (string) - if the desired container is non-existent, this
        attribute determines the location of the fields to be created.

    OUTPUTS

    fieldsPointers - (list of numpy arrays) - list of pointers of the LAST ZONE
        provided by user, in the same order as FieldNames
    '''
    npOpts = dict(dtype=np.float64, order='F')
    for zone in I.getZones(t):
        GridType, NPts, NCells, eltName, cellDim = I.getZoneDim(zone)
        FlowSol = I.getNodeFromName1(zone, Container)

        if FlowSol is None:
            FlowSol = I.createNode(Container, 'FlowSolution_t', parent=zone)
            Location = LocationIfAbsent
            I.newGridLocation(value=Location, parent=FlowSol)
        else:
            GridLocation = I.getNodeFromName1(FlowSol, 'GridLocation')

            if GridLocation is not None:
                Location = I.getValue(GridLocation)

            else:
                AuxiliarField = FlowSol[2][0][1]
                AuxFieldNPts = len(AuxiliarField.ravel(order='K'))
                if AuxFieldNPts == C.getNPts(zone):
                    Location = 'Vertex'
                elif AuxFieldNPts == C.getNCells(zone):
                    Location = 'CellCenter'
                else:
                    raise TypeError('inexistent GridLocation, and could determine field shape')


        fields = []
        for FieldName in FieldNames:
            if GridType == 'Structured':
                Ni,Nj,Nk = NPts, NCells, eltName
                if Location == 'Vertex':
                    if Nk > 0:
                        field = np.zeros((Ni,Nj,Nk), **npOpts)
                    elif Nj > 0:
                        field = np.zeros((Ni,Nj), **npOpts)
                    else:
                        field = np.zeros((Ni), **npOpts)
                elif Location == 'CellCenter':
                    if Nk > 0:
                        field = np.zeros((Ni-1,Nj-1,Nk-1), **npOpts)
                    elif Nj > 0:
                        field = np.zeros((Ni-1,Nj-1), **npOpts)
                    else:
                        field = np.zeros((Ni-1), **npOpts)
                else:
                    raise TypeError('Location %s not supported'%Location)
            elif GridType == 'Unstructured':
                if Location == 'Vertex':
                    field = np.zeros((NPts), **npOpts)
                elif Location == 'CellCenter':
                    field = np.zeros((NCells), **npOpts)
                else:
                    raise TypeError('Location %s not supported'%Location)

            I._createUniqueChild(FlowSol, FieldName, 'DataArray_t', value=field)
            fields.append( field )

    return fields

def get(t, FieldsNames=[], Container='FlowSolution', OutputObject='list',
        NumpyAsVector=False):
    '''
    Get the the pointers of numpy arrays of all (or some requested)
    fields of a CGNS component.

    INPUTS

    t - (PyTree, Base, zone, list of zones) - Element with zones where numpy
        pointers are requested

    FieldNames - (list of strings) - list of field names to get. If the list
        is empty, then all suitable fields found at the container are returned.

    Container - (string) - Name of the container (FlowSolution_t type) where the
        fields are found

    OutputObject - (string) - Choose the kind of output object returned:

        'list': a single list including all numpy arrays, as found from the
            top-down structure of the user-provided input <t>. Example:

                [numpy.ndarray, numpy.ndarray, numpy.ndarray, ...]

        'dict': a single dictionary whose values are the corresponding numpy
            arrays. ONLY SUITABLE FOR A SINGLE ZONE. Example:

                    OutputDict[<fieldname>] = numpy.ndarray

        'dictWithZoneNames': a dictionary of dictionaries. First key is the
            corresponding zone name (must be unique!). Second key corresponds
            to the field name. Example:

            OutputDict[<ZoneName>][<fieldname>] = numpy.ndarray

    NumpyAsVector - (boolean) - if True, returns a 1D (ravel) view of the numpy
        array

    OUTPUTS

    list or dictionary of numpy arrays. See OutputObject attribute for more
    information
    '''
    zones = I.getZones(t)
    ZonesQty =  len(zones)

    getAllFields = True if not FieldNames else False

    if OutputObject == 'list':
        out = []
        for z in zones:
            FlowSol = I.getNodeFromName1(z, Container)
            if getAllFields:
                for fieldNode in FlowSol[2]:
                    if fieldNode[3] != 'DataArray_t': continue
                    if NumpyAsVector: out.append(fieldNode[1].ravel(order='K'))
                    else: out.append(fieldNode[1])
            else:
                for fieldNode in FlowSol[2]:
                    if fieldNode[0] not in FieldsNames: continue
                    if NumpyAsVector: out.append(fieldNode[1].ravel(order='K'))
                    else: out.append(fieldNode[1])

    if OutputObject == 'dict':
        if ZonesQty > 1: raise AttributeError('More than one zone exist. Cannot use OutputObject="dict"')
        out = {}
        FlowSol = I.getNodeFromName1(zones[0], Container)
        if getAllFields:
            for fieldNode in FlowSol[2]:
                if fieldNode[3] != 'DataArray_t': continue
                fieldname = fieldNode[0]
                if NumpyAsVector:
                    out[fieldname] = fieldNode[1].ravel(order='K'))
                else:
                    out[fieldname] = fieldNode[1]
        else:
            for fieldname in FieldsNames:
                fieldNode = I.getNodeFromName1(FlowSol, fieldname)
                if NumpyAsVector:
                    out[fieldname] = fieldNode[1].ravel(order='K'))
                else:
                    out[fieldname] = fieldNode[1]

    elif OutputObject == 'dictWithZoneNames':
        out = {}
        for z in zones:
            ZoneName = z[0]
            out[ZoneName] = {}
            FlowSol = I.getNodeFromName1(z, Container)
            if getAllFields:
                for fieldNode in FlowSol[2]:
                    if fieldNode[3] != 'DataArray_t': continue
                    fieldname = fieldNode[0]
                    if NumpyAsVector:
                        out[ZoneName][fieldname] = fieldNode[1].ravel(order='K')
                    else:
                        out[ZoneName][fieldname] = fieldNode[1]
            else:
                for fieldname in FieldsNames:
                    fieldNode = I.getNodeFromName1(FlowSol, fieldname)
                    if NumpyAsVector:
                        out[ZoneName][fieldname] = fieldNode[1].ravel(order='K')
                    else:
                        out[ZoneName][fieldname] = fieldNode[1]

    else:
        raise AttributeError('OutputObject %s not recognized'%OutputObject)

    return out
