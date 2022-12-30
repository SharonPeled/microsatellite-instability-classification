import os
import numpy as np
import pyvips
from .Image import Image


class Tile(Image):
    def __init__(self, path, slide_uuid=None, **kwargs):
        super().__init__(path, slide_uuid=slide_uuid, **kwargs)
        self.out_filename = os.path.basename(self.path)

    def load(self):
        self.img = pyvips.Image.new_from_file(self.path).numpy()

    def save(self, processed_tiles_dir):
        np.save(os.path.join(processed_tiles_dir, self.slide_uuid, self.out_filename), self.img)

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





