from mola.cfd.preprocess.compatibilities_adjustor import CompatibilitiesAdjustor
from mola.logging import mola_logger

class CompatibilitiesAdjustorSonics(CompatibilitiesAdjustor):

    def apply(self, workflow):
        self.adjust_split_and_distribute(workflow)

    def adjust_split_and_distribute(self, workflow):
        split_opts = workflow.SplittingAndDistribution

        strategy = split_opts['Strategy']
        if strategy.lower() != 'atcomputation':
            msg = f'sonics solver requires splitting at computation, switching strategy from "{strategy}" to "AtComputation"'
            mola_logger.user_warning(msg)
            split_opts['Strategy'] = 'AtComputation'

        splitter = split_opts['Splitter']
        if splitter.lower() != 'maia':
            msg = f'sonics solver requires maia splitter, switching from "{splitter}" to "maia"'
            mola_logger.user_warning(msg)
            split_opts['Splitter'] = 'maia'

        distributor = split_opts['Distributor']
        if distributor.lower() != 'maia':
            msg = f'sonics solver requires maia distributor. Switching from {distributor} to maia.'
            mola_logger.user_warning(msg)
            split_opts['Distributor'] = 'maia'
