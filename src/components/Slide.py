import pandas as pd
from glob import glob
from pathlib import Path
import os
from .Tile import Tile
import pyvips
from .Image import Image
from tqdm import tqdm
import json
import traceback
import datetime


class Slide(Image):
    def __init__(self, path, slide_uuid=None, load_metadata=True, **kwargs):
        """
        :param slide_uuid:
        :param path: if slide_uuid=None then path directory name must be uuid of the slide, as in the gdc-client format
        :param tiles_dir: directory for storing all tiles from all slides
        """
        super().__init__(path, slide_uuid=slide_uuid, **kwargs)
        if slide_uuid is None:
            self.set('slide_uuid', self._get_uuid())
        self.set('metadata_path', os.path.join(os.path.dirname(self.path), 'metadata.json'))
        if load_metadata:
            self.load_slide_metadata()

    def load(self):
        self.img = pyvips.Image.new_from_file(self.path)

    def load_slide_metadata(self):
        metadata_path = self.get('metadata_path')
        if not os.path.exists(metadata_path):
            return
        with open(metadata_path, 'r') as file:
            loaded_metadata = json.load(file)
            for k, v in loaded_metadata.items():
                if k not in self.metadata.keys():
                    self.metadata[k] = v

    def save_metadata(self):
        self.set("Saving metadata time", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        with open(self.metadata['metadata_path'], 'w') as file:
            json.dump(self.metadata, file, indent=4)

    def _get_uuid(self):
        return Path(self.path).parent.name

    def set_tile_dir(self, tiles_dir):
        self.set('tile_dir', os.path.join(tiles_dir, self.get('slide_uuid')))

    def apply_pipeline(self, pipeline_list, ind, num_slides):
        if self.get('Done preprocessing', soft=True): # todo add validation to all dataset that slide are finished
            self._log(f"""Slide already processed: {ind+1}/{num_slides}""", importance=1)
            return self
        try:
            self._log(f"""Processing {self}""")
            for i, (resolution, pipeline) in enumerate(pipeline_list):
                if resolution == 'slide':
                    self = pipeline.transform(self)
                elif resolution == 'tile':
                    tiles = self.get_tiles(otsu_val=self.get('otsu_val'), slide_uuid=self.get('slide_uuid'))
                    tile_ind = self.get('tile_ind', soft=True)
                    tile_ind = 0 if tile_ind is None else tile_ind
                    left = len(tiles) - tile_ind
                    self._log(f"""Processed {tile_ind}/{len(tiles)} tiles.""", importance=1)
                    self._log(f"""Processing {len(tiles) - tile_ind} tiles.""", importance=1)
                    with tqdm(tiles, desc=f"""Slide {ind+1}/{num_slides} ({int(((ind+1)/num_slides)*100)}%)""",
                              initial=tile_ind,position=0, leave=True) as tile_tqdm:
                        for tile_ind, tile in enumerate(tiles[tile_ind:], start=tile_ind):
                            pipeline.transform(tile)
                            tile_tqdm.update(1)
                    self._log(f"""Finished processing {len(tiles)} tiles.""", importance=1)
                self.set(f'Finished ({resolution}, {i}).', True)
            self.set('Done preprocessing', True)
            self.save_metadata()
            self._log(f"""Finish processing {self}""", importance=1)
        except Exception as e:
            self._log(f"""Processing interrupt on slide {ind+1}/{num_slides} ({int(((ind+1)/num_slides)*100)}%)""", importance=2)
            if 'tile_ind' in locals():
                self.set('tile_ind', tile_ind)
            self.save_metadata()
            self._log(f"""metadata saved for slide {self.get('slide_uuid')}""", importance=2)
            self._log(f"""Exception {e}""", importance=2)
            self._log(f"""Traceback {traceback.format_exc()}""", importance=2)
        return self

    def get_tiles(self, **kwargs):
        if not self.get('tile_dir', soft=True):
            raise Exception("""You have to tile the image before applying a pipeline over tiles. 
                                tile_dir is None.""")
        tile_dir = self.get('tile_dir')
        return [Tile(tile_path, **kwargs) for tile_path in sorted(glob(f"{tile_dir}/**/*.jpg", recursive=True))] # all .jpg files

    def get_tile_summary_df(self, processed_tiles_dir, filters):
        csv_rows = []
        tile_dir = os.path.join(processed_tiles_dir, self.get('slide_uuid'))
        for tile_path in glob(f"{tile_dir}/**/*.jpg", recursive=True):
            csv_row_dict = {'tile_path': tile_path}
            attrs = os.path.basename(tile_path)[:-4].split('_')
            col, row = attrs[:2] # pyvips save it with col_row format
            csv_row_dict['row'] = row
            csv_row_dict['col'] = col
            filter_suffixes_dict = {f:f in attrs for f in filters}
            csv_row_dict.update(filter_suffixes_dict)
            csv_rows.append(csv_row_dict)
        df = pd.DataFrame(csv_rows)
        df[['row', 'col']] = df[['row', 'col']].astype(int)
        df.set_index(['row', 'col'], inplace=True, drop=False)
        df.fillna(False, inplace=True)
        return df

    def __str__(self):
        if self.img is None:
            return f"""<{type(self).__name__} - uuid:{self.get('slide_uuid')} Not loaded.>"""
        shape = (self.img.height, self.img.width, self.img.bands)
        otsu_val = self.get('otsu_val',soft=True)
        return f"""<{type(self).__name__} - shape:{shape}, otsu_val:{otsu_val}, uuid:{self.get('slide_uuid')}>"""
