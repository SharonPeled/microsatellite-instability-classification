from pyvips import Image
import os


class Tile:
    def __init__(self, path, otsu_val):
        self.path = path
        self.tile = None
        self.otsu_val = None
        self.out_filename = os.path.basename(self.path)

    def load(self):
        self.tile = pyvips.Image.new_from_file(self.path).numpy()

    def add_filename_suffix(self, suffix):
        filename, file_extension = self.out_filename.split('.')
        self.out_filename = filename + '_' + suffix + file_extension

    def __getattr__(self, attr):
        if self.tile is None:
            raise Exception("Tile not loaded.")
        return getattr(self.tile, attr)





