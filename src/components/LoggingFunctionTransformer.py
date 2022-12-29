from sklearn.preprocessing import FunctionTransformer
import logging


class LoggingFunctionTransformer(FunctionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        logging.info(f"Called function {self.func} on object {self} with args {args} and kwargs {kwargs}")
        return super().__call__(*args, **kwargs)