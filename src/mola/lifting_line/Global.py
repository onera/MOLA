import Converter.Internal as I

# Global constants
# -> Fluid constants
Gamma, Rgp = 1.4, 287.058
Mus, Cs, Ts= 1.711e-5, 110.4, 273.0 # Sutherland const.

FAIL  = '\033[91m'
GREEN = '\033[92m'
WARN  = '\033[93m'
MAGE  = '\033[95m'
CYAN  = '\033[96m'
ENDC  = '\033[0m'

NamesOfChordSpanThickwiseFrameNoTangential = [
    ['ChordwiseX','ChordwiseY','ChordwiseZ'],
    ['SpanwiseX','SpanwiseY','SpanwiseZ'],
    ['ThickwiseX','ThickwiseY','ThickwiseZ'],
    ['PitchRelativeCenterX','PitchRelativeCenterY','PitchRelativeCenterZ'],
    ['PitchAxisX','PitchAxisY','PitchAxisZ'],
    ]

NamesOfChordSpanThickwiseFrame = NamesOfChordSpanThickwiseFrameNoTangential + \
    [['TangentialX', 'TangentialY', 'TangentialZ']]


def getLiftingLines(t):
    return [c for c in I.getZones(t) if checkComponentKind(c, kind='LiftingLine')]

def checkComponentKind(component, kind='LiftingLine'):
    '''
    Function to determine whether a component (CGNS Base or zone) is of kind
    given by attribute **kind**.

    Parameters
    ----------

        Component : node
            Component whose kind verification is requested

        kind : str
            Kind to verify. For example : ``'Propeller'``, ``'LiftingLine'``...

    Returns
    -------

        Result : bool
            :py:obj:`True` if node kind corresponds to the requested one
    '''
    ZoneInfo = I.getNodeFromName1(component,'.Component#Info')
    if ZoneInfo is None: return False
    kindC = I.getNodeFromName1(ZoneInfo,'kind')
    kindC = I.getValue(kindC)

    return kind == kindC

