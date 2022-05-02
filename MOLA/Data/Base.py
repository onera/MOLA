'''
Implements class **Base**, which inherits from :py:class:`Node`

21/12/2021 - L. Bernardos - first creation
'''

import numpy as np
from .Node import Node
from .Zone import Zone

class Base(Node):
    """docstring for Base"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setType('CGNSBase_t')

        if self.value() is None:
            BaseValue = np.array([3,3],dtype=np.int,order='F')
            for child in self.children():
                if isinstance(child, Zone):
                    BaseValue = np.array([child.dim(),3],dtype=np.int,order='F')
                    break
            self.setValue(BaseValue)

        if self.name() == 'Node': self.setName( 'Base' )

    def save(self,*args,**kwargs):
        from .Tree import Tree
        t = Tree()
        t.addChild( self )
        t.save(*args,**kwargs)

    def dim(self):
        return self.value()[0]

    def zones(self):
        return [c for c in self.children() if isinstance(c, Zone)]

    def setPhysicalDimension(self, PhysicalDimension):
        self.value()[0] = PhysicalDimension
