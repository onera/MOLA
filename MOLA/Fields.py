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
