'''
Main subpackage for LiftingLine related operations

06/05/2022 - L. Bernardos - first creation
'''
from ..Core import np,RED,GREEN,WARN,PINK,CYAN,ENDC,interpolate
from ..Mesh import Curve
from ..Node import Node
from ..Zone import Zone
from ... import __version__

class LiftingLine(Curve):
    """docstring for LiftingLine"""
    def __init__(self, SpanMin, SpanMax, N, SpanwiseDistribution=dict(),
                       GeometricalLaws=dict(), *args, **kwargs):
        super().__init__(*args, **kwargs)

        Span = X = np.linspace(SpanMin, SpanMax, N)
        Y = np.zeros(N, dtype=float)
        Z = np.zeros(N, dtype=float)
        GridCoordinates = Node(Parent=self,
               Name='GridCoordinates', Type='GridCoordinates_t',
               Children=[Node(Name='CoordinateX', Value=X, Type='DataArray_t'),
                         Node(Name='CoordinateY', Value=Y, Type='DataArray_t'),
                         Node(Name='CoordinateZ', Value=Z, Type='DataArray_t')])

        self.setValue(np.array([[N,N-1,0]],dtype=np.int,order='F'))

        s = self.abscissa()

        LLx, LLy, LLz = self.xyz()
        SpecialTreatment = ['Airfoils','Span','s']
        Variables2Invoke = [v for v in GeometricalLaws if v not in SpecialTreatment]
        LLDict = self.newFields(Variables2Invoke+['Span','s'], return_type='dict')
        LLDict['Span'][:] = X
        RelSpan = Span/SpanMax

        InterpLaws = {}
        for GeomParam in LLDict:
            if GeomParam in SpecialTreatment: continue
            InterpLaws[GeomParam+'_law']=GeometricalLaws[GeomParam]['InterpolationLaw']
            try: InterpOptions = GeometricalLaws[GeomParam]['InterpolationOptions']
            except KeyError: InterpOptions = dict()
            LLDict[GeomParam][:] = interpolate(RelSpan,
                                    GeometricalLaws[GeomParam]['RelativeSpan'],
                                    GeometricalLaws[GeomParam][GeomParam],
                                    InterpLaws[GeomParam+'_law'],
                                    **InterpOptions)

        LLx[:] = Span
        if 'Sweep' in LLDict:    LLy[:] = -LLDict['Sweep']
        if 'Dihedral' in LLDict: LLz[:] =  LLDict['Dihedral']

        # Add Airfoils node
        if 'RelativeSpan' in GeometricalLaws['Airfoils']:
            AbscissaPolar = interpolate(GeometricalLaws['Airfoils']['RelativeSpan'],
                                                                     RelSpan, s)
            GeometricalLaws['Airfoils']['Abscissa'] = AbscissaPolar
        elif 'Abscissa' in GeometricalLaws['Airfoils']:
            AbscissaPolar = GeometricalLaws['Airfoils']['Abscissa']
        else:
            raise ValueError("Attribute Polars (dict) must contain 'RelativeSpan' or 'Abscissa' key")


        self.newFields(['AoA', 'Mach', 'Reynolds', 'Cl', 'Cd','Cm'])

        self.setParameters('.Component#Info', kind='LiftingLine',
                                              MOLAversion=__version__,
                                              GeometricalLaws=GeometricalLaws)

        if SpanwiseDistribution: self.discretize( **SpanwiseDistribution )
