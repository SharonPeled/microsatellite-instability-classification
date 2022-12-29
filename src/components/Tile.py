from pyvips import Image
import os
import numpy as np
import pyvips


class Tile:
    def __init__(self, path, slide_uuid = None, otsu_val = None):
        self.path = path
        self.slide_uuid = slide_uuid
        self.tile = None
        self.otsu_val = otsu_val
        self.out_filename = os.path.basename(self.path)

    @classmethod
    def from_tile(cls, tile):
        # create a new Tile object and copy all attributes from the existing tile object
        new_tile = cls(tile.path)
        new_tile.__dict__.update({k: v for k, v in tile.__dict__.items() if k != "tile"})
        new_tile.tile = tile.tile
        return new_tile

    def load(self):
        self.tile = pyvips.Image.new_from_file(self.path).numpy()

    def save(self, processed_tile_dir):
        np.save(os.path.join(processed_tile_dir, self.slide_uuid, self.out_filename), self.tile)

    def set_filename_suffix(self, suffix):
        """
        Append a suffix to the filename of the tile. The suffix is separated from the rest of the filename
        by an underscore character.
        If the tile has already been processed by another filter, the existing suffix will be replaced with
        the new one.
        This allows for tracking which tiles have been processed by which filters, and to maintain order relation
        between filters.
        :param suffix: The suffix to append to the filename.
        :return: None
        """
        filename, file_extension = self.out_filename.split('.')
        attrs = filename.split('_')[:2] # first 2 attrs are row and col
        attrs.append(suffix)
        self.out_filename = '_'.join(attrs) + file_extension

    def recover(self, tile_recovery_suffix):
        filename, file_extension = self.path.split('.')
        if filename.split('_')[-1] == tile_recovery_suffix:
            # already recovered
            return
        new_name = filename + '_' + tile_recovery_suffix + '.' + 'jpg'
        os.rename(self.path, new_name)
        self.path = new_name

    def __getattr__(self, attr):
        """
        A wrapper function that allows to use all self.tile methods (resize, crop, etc.) directly on the Tile object
        without wrapping method.
        :param attr: attribute to get/call over self.tile
        :return: Tile object when attr is callable, self.tile.attr otherwise
        """
        if self.tile is None:
            raise Exception("Tile not loaded.")
        if callable(getattr(self.slide, attr)):
            def wrapper(*args, **kwargs):
                result = getattr(self.slide, attr)(*args, **kwargs)
                # create a new Slide object using the from_slide class method
                return self.from_tile(result)
            return wrapper
        return getattr(self.tile, attr)





