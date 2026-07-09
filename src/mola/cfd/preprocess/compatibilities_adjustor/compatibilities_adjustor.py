import os
from abc import ABC, abstractmethod
from mola.dependency_injector.retriever import load_source

class CompatibilitiesAdjustor(ABC):

    @abstractmethod
    def apply(self, workflow):
        ...

def _instantiate_compatibilities_adjustor(solver : str) -> CompatibilitiesAdjustor:
    current_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_path,
                             f'compatibilities_adjustor_{solver}.py')
    solverModule = load_source('solverModule', file_path)
    obj : CompatibilitiesAdjustor = getattr(solverModule, 
            f"CompatibilitiesAdjustor{solver.capitalize()}")
    return obj()
