from torch.multiprocessing import Pool, set_start_method
from src.utils import set_global_configs
from src.components.objects.Logger import Logger
import traceback


class ParallelProcessingManager(Logger):
    def __init__(self, num_processes, verbose, log_importance, log_format, random_seed, tile_progress_log_freq):
        self.num_processes = 1 if num_processes is None else num_processes
        set_start_method("spawn")
        self.verbose = verbose
        self.log_importance = log_importance
        self.log_format = log_format
        self.random_seed = random_seed
        self.tile_progress_log_freq = tile_progress_log_freq
        self._log(f"ParallelProcessingManager initialize with {self.num_processes} processes.", log_importance=2)

    def execute_parallel_loop(self, func, param_generator, log_file_args_generator):
        """
        param: func - function to parallel execute
        param: param_generator - parameter generator for func.
        func must accept parameters as args (no keyword arguments).
        param: log_file_generator - for each execution, set a log file.
        """
        if self.num_processes == 1:
            for args, log_file_args in zip(param_generator, log_file_args_generator):
                _job(func, log_file_args, self.verbose, self.log_format, self.log_importance,
                                              self.random_seed, self.tile_progress_log_freq, *args)
        else:
            with Pool(processes=self.num_processes) as pool:
                job_param_generator = ((func, log_file, self.verbose, self.log_format, self.log_importance,
                                                  self.random_seed, self.tile_progress_log_freq, *args)
                                    for args, log_file in zip(param_generator, log_file_args_generator))
                return pool.starmap(_job, job_param_generator, chunksize=1)


def _job(job_func, log_file_args, verbose, log_format, log_importance, random_seed, tile_progress_log_freq, *args):
    try:
        set_global_configs(verbose=verbose,
                           log_file_args=log_file_args,
                           log_format=log_format,
                           log_importance=log_importance,
                           random_seed=random_seed,
                           tile_progress_log_freq=tile_progress_log_freq)
        return job_func(*args)
    except Exception as e:
        Logger.log(f"""Exception {e}""", log_importance=2)
        Logger.log(f"""Traceback {traceback.format_exc()}""", log_importance=2)
