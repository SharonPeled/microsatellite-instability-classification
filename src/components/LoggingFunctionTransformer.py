from sklearn.preprocessing import FunctionTransformer
import logging
from .Logger import Logger


class LoggingFunctionTransformer(FunctionTransformer, Logger):
    def __init__(self, func, **kwargs):
        super().__init__(func, **kwargs)

    def transform(self, *args, **kwargs):
        self._log(f"Starting - args {args}, kwargs {kwargs}", name=self.func.__name__)
        temp = super().transform(*args, **kwargs)
        self._log(f"Finished - args {args}, kwargs {kwargs}", name=self.func.__name__)
        return temp