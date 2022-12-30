import pandas as pd
from pyvips import Image
from glob import glob
from pathlib import Path
import os
from .Tile import Tile
import pyvips
from .Logger import Logger

class Slide(Logger):
    def __init__(self, path, tile_dir=None):
        """
        :param path: directory name must be uuid of the slide, as in the gdc-client format
        :param tile_dir:
        """
        self.path = path
        self.tile_dir = tile_dir
        self.slide = None
        self.uuid = self._get_UUID()
        self.otsu_val = None

    @classmethod
    def from_slide(cls, slide, new_img):
        # create a new Slide object and copy all attributes from the existing slide object
        new_slide = cls(slide.path, slide.tile_dir)
        new_slide.__dict__.update({k: v for k, v in slide.__dict__.items() if k != "slide"})
        new_slide.slide = new_img
        return new_slide

    def load(self):
        self.slide = pyvips.Image.new_from_file(self.path)

    def _get_UUID(self):
        return Path(self.path).parent.name

    def set_tile_dir(self, tile_dir):
        self.tile_dir = tile_dir

    def apply_pipeline(self, pipeline_list):
        self._log(f'Processing {self.uuid}')
        for resolution, pipeline in pipeline_list:
            if resolution == 'slide':
                pipeline.transform(self)
            elif resolution == 'tile':
                tiles = self.get_tiles(otsu_val=self.slide.otsu_val, slide_uuid=self.uuid)
                for tile in tiles:
                    pipeline.transform(tile)
        self._log(f'Finished processing {self.uuid}')
        return self

    def get_tiles(self, **kwargs):
        if not self.tile_dir:
            raise Exception("""You have to tile the image before applying a pipeline over tiles. 
                                tile_dir is None.""")
        return [Tile(tile_path, **kwargs) for tile_path in glob(os.path.join(self.tile_dir, '*.jpg'))]

    def get_tile_summary_df(self):
        csv_rows = []
        for tile_path in glob(os.path.join(self.tile_dir, '*.jpg')):
            csv_row_dict = {'tile_path': tile_path}
            attrs = os.path.basename(tile_path)[:-4].split('_')
            col, row = attrs[:2] # pyvips save it with col_row format
            csv_row_dict['row'] = row
            csv_row_dict['col'] = col
            filter_suffixes_dict = {f:True for f in attrs[2:]}
            csv_row_dict.update(filter_suffixes_dict)
            csv_rows.append(csv_row_dict)
        df = pd.DataFrame(csv_rows)
        df.fillna(False, inplace=True)
        return df

    def recover_tiles(self):
        pass

    def __getattr__(self, attr):
        """
        A wrapper function that allows to use all self.slide methods (resize, crop, etc.) directly on the Slide object
        without wrapping method.
        :param attr: attribute to get/call over self.slide
        :return: Slide object when attr is callable, self.slide.attr otherwise
        """
        if self.slide is None:
            raise Exception("Slide not loaded.")
        if callable(getattr(self.slide, attr)):
            def wrapper(*args, **kwargs):
                result = getattr(self.slide, attr)(*args, **kwargs)
                if isinstance(result, type(self.slide)):
                    # create a new Slide object using the from_slide class method
                    return self.from_slide(self, result)
                return result
            return wrapper
        return getattr(self.slide, attr)
