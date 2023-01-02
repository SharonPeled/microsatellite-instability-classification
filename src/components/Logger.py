import logging
import logging.config


class Logger:
    """
    Wrapper class for python logging.
    In order to filter logs from imported modules we deploy the following log levels mapping:
    WARN (importance 0) - All logs, similar to INFO
    ERROR (importance 1) - Only high importance logs, summary logs and not operative logs
    CRITICAL (importance 2) - Only logs that is critical, such as errors or potential errors.
    """
    LOG_IMPORTANCE_MAP = {0: logging.WARN,
                     1: logging.ERROR,
                     2: logging.CRITICAL}

    def _log(self, msg, importance=0, name=None, **kwargs):
        if name:
            logging.getLogger(name).log(msg=msg, level=Logger.LOG_IMPORTANCE_MAP[importance], **kwargs)
        else:
            logging.getLogger(type(self).__name__).log(msg=msg, level=Logger.LOG_IMPORTANCE_MAP[importance], **kwargs)

    @staticmethod
    def log(msg, importance=0, **kwargs):
        logging.log(msg=msg, level=Logger.LOG_IMPORTANCE_MAP[importance], **kwargs)

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





