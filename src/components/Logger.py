import logging
import logging.config


class Logger:
    """
    Wrapper class for python logging.
    All logs are in WARNING level to filter import modules logs that can be in INFO and DEBUG modes.
    """
    def _log(self, msg, name=None, **kwargs):
        if name:
            logging.getLogger(name).log(msg=msg, level=logging.WARNING, **kwargs)
        else:
            logging.getLogger(type(self).__name__).log(msg=msg, level=logging.WARNING, **kwargs)

    @staticmethod
    def log(msg, **kwargs):
        logging.log(msg=msg, level=logging.WARNING, **kwargs)

    @staticmethod
    def set_default_logger(configs):
        if configs.DEBUG_MODE:
            logging.basicConfig(handlers=[logging.StreamHandler(), ], level=logging.WARNING,
                                **configs.LOG_FORMAT_DEBUG_MODE)
        elif configs.VERBOSE == 0:
            # suppressing all logs
            logging.getLogger().setLevel(logging.NOTSET)
        elif configs.VERBOSE == 1:
            logging.basicConfig(handlers=[logging.FileHandler(configs.LOG_FILE), ], level=logging.WARNING,
                                **configs.LOG_FORMAT)
        elif configs.VERBOSE == 2:
            # Create a stream handler to log to the console
            logging.basicConfig(handlers=[logging.StreamHandler(), ], level=logging.WARNING,
                                **configs.LOG_FORMAT)
        elif configs.VERBOSE == 3:
            # Create a stream handler to log to the console
            logging.basicConfig(handlers=[logging.FileHandler(configs.LOG_FILE), logging.StreamHandler()],
                                level=logging.WARNING, **configs.LOG_FORMAT)





