import os
import numpy as np
import pyvips
from .Image import Image
from PIL import Image as PLI_Image


class Tile(Image):
    def __init__(self, path=None, img=None, slide_uuid=None, out_filename=None, **kwargs):
        super().__init__(path, img, slide_uuid=slide_uuid, **kwargs)
        if path is None and out_filename is None:
            raise Exception("Tile must have valid path or valid out_filename, they both None.")
        if path is None:
            self.out_filename = out_filename
        else:
            self.out_filename = os.path.basename(self.path)

    def load(self):
        if self.img is None:
            self.img = pyvips.Image.new_from_file(self.path).numpy()
        else:
            self.img = self.img.numpy()[:,:,:-1]

    def save(self, processed_tiles_dir):
        path = os.path.join(processed_tiles_dir, self.get('slide_uuid'), self.out_filename)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        PLI_Image.fromarray(self.img).save(path)
        self.set('processed_tile_dir', os.path.dirname(path))

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

    def __str__(self):
        if not isinstance(self.img, np.ndarray):
            return f"""<{type(self).__name__} - uuid:{self.get('slide_uuid')} Not loaded.>"""
        return f"""<{type(self).__name__} - {self.out_filename} shape:{self.img.shape}, uuid:{self.get('slide_uuid')}>"""





