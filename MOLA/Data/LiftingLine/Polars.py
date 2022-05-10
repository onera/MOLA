'''
Main subpackage for Polars related operations

06/05/2022 - L. Bernardos - first creation
'''

from ..Core import np,RED,GREEN,WARN,PINK,CYAN,ENDC,interpolate
from ..Node import Node
from ..Zone import Zone


class Polars(Zone):
    """docstring for Polars"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
