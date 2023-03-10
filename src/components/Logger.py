import logging
import logging.config


class Logger:
    """
    Wrapper class for python logging.
    In order to filter logs from imported modules we deploy the following log levels mapping:
    WARN (log_importance 0) - All logs, similar to INFO
    ERROR (log_importance 1) - Only high log_importance logs, summary logs and not operative logs
    CRITICAL (log_importance 2) - Only logs that is critical, such as errors or potential errors.
    """
    LOG_IMPORTANCE_MAP = {0: logging.WARN,
                          1: logging.ERROR,
                          2: logging.CRITICAL}
    TILE_PROGRESS_LOG_FREQ = 0

    def _log(self, msg, log_importance=0, name=None, **kwargs):
        msg = Logger.add_importance_level_to_msg(msg, log_importance)
        if name:
            logger = logging.getLogger(name)
            logger.log(msg=msg, level=Logger.LOG_IMPORTANCE_MAP[log_importance], **kwargs)
        else:
            logger = logging.getLogger(type(self).__name__)
            logger.log(msg=msg, level=Logger.LOG_IMPORTANCE_MAP[log_importance], **kwargs)
        Logger.flush_logger(logger)

    @staticmethod
    def log(msg, log_importance=0, **kwargs):
        msg = Logger.add_importance_level_to_msg(msg, log_importance)
        logging.log(msg=msg, level=Logger.LOG_IMPORTANCE_MAP[log_importance], **kwargs)
        Logger.flush_logger(logging.getLogger())

    @staticmethod
    def set_default_logger(verbose, log_file_args, log_importance, log_format, tile_progress_log_freq):
        Logger.TILE_PROGRESS_LOG_FREQ = tile_progress_log_freq
        if verbose == 1:
            logging.basicConfig(handlers=[logging.FileHandler(*log_file_args), ],
                                level=Logger.LOG_IMPORTANCE_MAP[log_importance],
                                **log_format)
        elif verbose == 2:
            logging.basicConfig(handlers=[logging.StreamHandler(), ], level=Logger.LOG_IMPORTANCE_MAP[log_importance],
                                **log_format)
        elif verbose == 3:
            logging.basicConfig(handlers=[logging.FileHandler(*log_file_args), logging.StreamHandler()],
                                level=Logger.LOG_IMPORTANCE_MAP[log_importance], **log_format)

    @staticmethod
    def add_importance_level_to_msg(msg, log_importance):
        return f"[{log_importance}] {msg}"

    @staticmethod
    def flush_logger(logger):
        for handle in logger.handlers:
            handle.flush()





