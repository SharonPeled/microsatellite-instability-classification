from joblib import Parallel, delayed, wrap_non_picklable_objects
from ..utils import set_global_configs
from .Logger import Logger


class ParallelProcessingManager(Logger):
    def __init__(self, num_processes, verbose, log_importance, log_format, random_seed, tile_progress_log_freq):
        self.num_processes = 1 if num_processes is None else num_processes
        self.job_manager = Parallel(n_jobs=self.num_processes)
        self.verbose = verbose
        self.log_importance = log_importance
        self.log_format = log_format
        self.random_seed = random_seed
        self.tile_progress_log_freq = tile_progress_log_freq
        self._log(f"ParallelProcessingManager initialize with {self.num_processes} processes.", log_importance=2)

    def execute_parallel_loop(self, func, param_generator, log_file_generator):
        """
        param: func - function to parallel execute
        param: param_generator - parameter generator for func.
        func must accept parameters as args (no keyword arguments).
        param: log_file_generator - for each execution, set a log file.
        """
        return self.job_manager(delayed(_job)(func, log_file, self.verbose, self.log_format, self.log_importance,
                                              self.random_seed, self.tile_progress_log_freq, *args)
                                for args, log_file in zip(param_generator, log_file_generator))


def _job(job_func, log_file, verbose, log_format, log_importance, random_seed, tile_progress_log_freq, *args):
    set_global_configs(verbose=verbose,
                       log_file=log_file,
                       log_format=log_format,
                       log_importance=log_importance,
                       random_seed=random_seed,
                       tile_progress_log_freq=tile_progress_log_freq)
    return job_func(*args)
