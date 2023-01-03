import os
import numpy as np
import pyvips
from .Image import Image
from PIL import Image as PLI_Image


class Tile(Image):
    def __init__(self, path, slide_uuid=None, **kwargs):
        super().__init__(path, slide_uuid=slide_uuid, **kwargs)
        self.out_filename = os.path.basename(self.path)

    def load(self):
        self.img = pyvips.Image.new_from_file(self.path).numpy()

    def save(self, processed_tiles_dir):
        path = os.path.join(processed_tiles_dir, self.get('slide_uuid'), self.out_filename)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        PLI_Image.fromarray(self.img).save(path)

    def add_filename_suffix(self, suffix):
        """
        Append a suffix to the filename of the tile. The suffix is separated from the rest of the filename
        by an underscore character.
        :param suffix: The suffix to append to the filename.
        :return: None
        """
        filename, file_extension = self.out_filename.split('.')
        filename += '_' + suffix
        self.out_filename = filename + '.' + file_extension

    def get_tile_position(self):
        col, row = self.out_filename[:-4].split('_')[:2]
        return row,col

    @staticmethod
    def recover(path, tile_recovery_suffix):
        filename, file_extension = os.path.basename(path).split('.')
        if tile_recovery_suffix in filename.split('_'):
            # already recovered
            return
        new_name = filename + '_' + tile_recovery_suffix + '.' + file_extension
        os.rename(path, os,path.join(os.path.dirname(path), new_name))

    def __str__(self):
        if self.img is None:
            return f"""<{type(self).__name__} - uuid:{self.get('slide_uuid')} Not loaded.>"""
        return f"""<{type(self).__name__} - {self.out_filename} shape:{self.img.shape}, uuid:{self.get('slide_uuid')}>"""





