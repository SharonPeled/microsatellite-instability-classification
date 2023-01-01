import pandas as pd
from glob import glob
from pathlib import Path
import os
from .Tile import Tile
import pyvips
from .Image import Image


class Slide(Image):
    def __init__(self, path, slide_uuid=None, **kwargs):
        """
        :param slide_uuid:
        :param path: if slide_uuid=None then path directory name must be uuid of the slide, as in the gdc-client format
        :param tiles_dir: directory for storing all tiles from all slides
        """
        super().__init__(path, slide_uuid=slide_uuid, **kwargs)
        if not slide_uuid:
            self.set('slide_uuid', self._get_uuid())
        self.tile_dir = None

    def load(self):
        self.img = pyvips.Image.new_from_file(self.path)

    def _get_uuid(self):
        return Path(self.path).parent.name

    def set_tile_dir(self, tiles_dir):
        self.tile_dir = os.path.join(tiles_dir, self.get('slide_uuid'))

    def apply_pipeline(self, pipeline_list):
        self._log(f"""Processing {self}""")
        for resolution, pipeline in pipeline_list:
            if resolution == 'slide':
                self = pipeline.transform(self)
            elif resolution == 'tile':
                tiles = self.get_tiles(otsu_val=self.get('otsu_val'), slide_uuid=self.get('slide_uuid'))
                self._log(f"""Processing {len(tiles)} tiles.""")
                for tile in tiles:
                    pipeline.transform(tile)
                self._log(f"""Finished processing {len(tiles)} tiles.""")
        self._log(f"""Finish processing {self}""")
        return self

    def get_tiles(self, **kwargs): #TODO: str method for tile and slide and slide number
        if not self.tile_dir:
            raise Exception("""You have to tile the image before applying a pipeline over tiles. 
                                tile_dir is None.""")
        return [Tile(tile_path, **kwargs) for tile_path in glob(f"{self.tile_dir}/**/*.jpg", recursive=True)] # all .jpg files

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
        df = pd.DataFrame(csv_rows, index=['row', 'col'])
        df.fillna(False, inplace=True)
        return df

    def __str__(self):
        if self.img is None:
            return f"""<{type(self).__name__} - uuid:{self.get('slide_uuid')} Not loaded.>"""
        shape = (self.img.height, self.img.width, self.img.bands)
        otsu_val = self.get('otsu_val',soft=True)
        return f"""<{type(self).__name__} - shape:{shape}, otsu_val:{otsu_val}, uuid:{self.get('slide_uuid')}>"""
