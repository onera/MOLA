from mola.cfd.preprocess.compatibilities_adjustor import CompatibilitiesAdjustor
from mola.logging import mola_logger

class CompatibilitiesAdjustorFast(CompatibilitiesAdjustor):

    def __init__(self):
        self._allowed_turbulence_modeling = [
            'LES',
            'ILES',
            'DNS',
            'Laminar',
            'Euler',
            'ZDES',
            'SA',
        ]

        self._allowed_schemes = [
            'ausm+',
            'Roe',
        ]

    def apply(self, workflow):
        self.adjust_models(workflow)
        self.adjust_numerics(workflow)
        self.adjust_split_and_distribute(workflow)

    def adjust_models(self, workflow):
        requested_model = workflow.Turbulence['Model']

        if requested_model not in self._allowed_turbulence_modeling:
            msg = (f"Turbulence Model {requested_model} not implemented in Fast.\n"
                   f"Allowed models: {str(self._allowed_turbulence_modeling)}\n"
                    "Switching to 'SA'")
            mola_logger.user_warning(msg)
            workflow.Turbulence['Model'] = 'SA'

    
    def adjust_numerics(self, workflow):
        requested_scheme = workflow.Numerics['Scheme']

        if requested_scheme not in self._allowed_schemes:
            msg = (f"Numerical Scheme {requested_scheme} not implemented in Fast.\n"
                   f"Allowed schemes: {str(self._allowed_schemes)}\n"
                    "Switching to 'Roe'")
            mola_logger.user_warning(msg)
            workflow.Numerics['Scheme'] = 'Roe'

    def adjust_split_and_distribute(self, workflow):
        split_opts = workflow.SplittingAndDistribution

        strategy = split_opts['Strategy']
        if strategy != 'AtPreprocess':
            msg = f'fast solver requires splitting in preprocess, switching strategy from "{strategy}" to "AtPreprocess"'
            mola_logger.user_warning(msg)
            split_opts['Strategy'] = 'AtPreprocess'
            split_opts['Splitter'] = 'Cassiopee'

        if split_opts['Distributor'].lower() != 'cassiopee':
            distributor = split_opts['Distributor']
            msg = f'fast solver requires to pre-assign mpi ranks in preprocess, for doing this switching distributor from distributor "{distributor}" to "Cassiopee"'
            mola_logger.user_warning(msg)
            split_opts['Distributor'] = 'Cassiopee'
