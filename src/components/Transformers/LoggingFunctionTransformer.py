from sklearn.preprocessing import FunctionTransformer
from src.components.Objects.Logger import Logger


class LoggingFunctionTransformer(FunctionTransformer, Logger):
    def __init__(self, func, log_importance=0, **kwargs):
        super().__init__(func, **kwargs)
        self.log_importance = log_importance

    def transform(self, *args, **kwargs):
        self._log(f"Starting - args {args}, kwargs {kwargs}", name=self.func.__name__, log_importance=self.log_importance)
        res = super().transform(*args, **kwargs)
        self._log(f"Finished - args {args}, kwargs {kwargs}", name=self.func.__name__, log_importance=self.log_importance)
        return res
