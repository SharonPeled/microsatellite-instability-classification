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
    def set_default_logger(configs):
        if configs.VERBOSE == 1:
            logging.basicConfig(handlers=[logging.FileHandler(configs.LOG_FILE), ],
                                level=Logger.LOG_IMPORTANCE_MAP[configs.LOG_IMPORTANCE],
                                **configs.LOG_FORMAT)
        elif configs.VERBOSE == 2:
            # Create a stream handler to log to the console
            logging.basicConfig(handlers=[logging.StreamHandler(), ], level=Logger.LOG_IMPORTANCE_MAP[configs.LOG_IMPORTANCE],
                                **configs.LOG_FORMAT)
        elif configs.VERBOSE == 3:
            # Create a stream handler to log to the console
            logging.basicConfig(handlers=[logging.FileHandler(configs.LOG_FILE), logging.StreamHandler()],
                                level=Logger.LOG_IMPORTANCE_MAP[configs.LOG_IMPORTANCE], **configs.LOG_FORMAT)

    @staticmethod
    def add_importance_level_to_msg(msg, log_importance):
        return f"[{log_importance}] {msg}"

    @staticmethod
    def flush_logger(logger):
        for handle in logger.handlers:
            handle.flush()





