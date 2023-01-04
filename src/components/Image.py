from .Logger import Logger


class Image(Logger):
    def __init__(self, path, **kwargs):
        self.path = path
        self.metadata = kwargs
        self.metadata['path'] = self.path
        self.img = None

    def get(self, key, soft=False):
        if key not in self.metadata.keys():
            try:
                return self.img.get(key)
            except Exception as e:
                if soft:
                    return None
                raise e
        return self.metadata[key]

    def set(self, key, val):
        self.metadata[key] = val

    @classmethod
    def from_img(cls, img_obj, new_img_attr):
        new_img_obj = cls(img_obj.path)
        new_img_obj.__dict__.update({k: v for k, v in img_obj.__dict__.items() if k != "img"})
        new_img_obj.img = new_img_attr
        return new_img_obj

    def __getattr__(self, attr):
        """
        A wrapper function that allows to use all self.img methods (resize, crop, etc.) directly on the img object
        without wrapping each method.
        :param attr: attribute to get/call over self.img
        :return: Image object when attr is callable, self.tile.attr otherwise
        """
        if self.img is not None and callable(getattr(self.img, attr)):
            def wrapper(*args, **kwargs):
                result = getattr(self.img, attr)(*args, **kwargs)
                if isinstance(result, type(self.img)):
                    # create a new Tile object using the from_tile class method
                    return self.from_img(self, result)
                return result
            return wrapper
        return getattr(self.img, attr)

    def __repr__(self):
        return self.__str__()















