import pandas as pd
from pyvips import Image
from glob import glob
from pathlib import Path
import os
from .Tile import Tile


class Slide:
    def __init__(self, path, tile_dir=None):
        self.path = path
        self.tile_dir = tile_dir
        self.slide = None
        self.uuid = self._get_UUID()

    def load(self):
        self.slide = pyvips.Image.new_from_file(self.path)

    def _get_UUID(self):
        return Path(self.slide.get('filename')).parent.name

    def set_tile_dir(self, tile_dir):
        self.tile_dir = tile_dir

    def apply_pipeline(self, pipeline_list):
        for resulotion, pipeline in pipeline_list:
            if resulotion == 'slide':
                self.slide = pipeline.transform(self.slide)
            if resulotion == 'tile':
                tiles = self.get_tiles(otsu_val=self.slide.get("otsu_val"), slide_uuid=self.uuid)
                for tile in tiles:
                    pipeline[1].transform(tile)
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
        if self.slide is None:
            raise Exception("Slide not loaded.")
        self.slide = getattr(self.slide, attr)
        return self
