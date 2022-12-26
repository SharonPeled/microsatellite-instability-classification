from pyvips import Image
from glob import glob
from pathlib import Path
import os
from Tile import Tile


class Slide:
    def __init__(self, path, tile_dir=None):
        self.path = path
        self.tile_dir = tile_dir
        self.slide = None

    def load(self):
        self.slide = pyvips.Image.new_from_file(self.path)

    def get_UUID(self):
        return Path(self.slide.get('filename')).parent.name

    def set_tile_dir(self, tile_dir):
        self.tile_dir = tile_dir

    def apply_pipeline(self, pipeline):
        # TODO: delete stuff after finished
        self.load()
        self.slide = pipeline[0].transform(self.slide)
        if self.tile_dir:
            otsu_val = self.slide.get("otsu_val")
            for tile_path in glob(os.path.join(self.tile_dir, '*.jpg')):
                tile = Tile(tile_path, otsu_val=otsu_val)
                tile.load()
                tile = pipeline[1].transform(tile)


                # TODO: save the transformed tile back to disk
                tile.write_to_file(tile_path)
        return self

    def __getattr__(self, attr):
        if self.slide is None:
            raise Exception("Slide not loaded.")
        return getattr(self.slide, attr)