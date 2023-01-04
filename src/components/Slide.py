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
from ..utils import generate_summary_df_from_filepaths


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
        self._log(f"""metadata saved for slide {self}.""", importance=2)

    def _get_uuid(self):
        return Path(self.path).parent.name

    def set_tile_dir(self, tiles_dir):
        self.set('tile_dir', os.path.join(tiles_dir, self.get('slide_uuid')))

    def apply_pipeline(self, pipeline_list, ind, num_slides):
        if self.get('Done preprocessing', soft=True):
            self._log(f"""Slide already processed: {ind+1}/{num_slides}""", importance=1)
            return self
        try:
            self._log(f"""Processing {self}""")
            for i, (resolution, pipeline) in enumerate(pipeline_list):
                if resolution == 'slide':
                    self = pipeline.transform(self)
                elif resolution == 'tile':

                    tiles = self.get_tiles(otsu_val=self.get('otsu_val'), slide_uuid=self.get('slide_uuid'))
                    tile_ind, tiles_in_path_out_filename_tuples = self.get_previous_tile_place()

                    left = len(tiles) - tile_ind
                    self._log(f"""Processed {tile_ind}/{len(tiles)} tiles.""", importance=1)
                    self._log(f"""Processing {len(tiles) - tile_ind} tiles.""", importance=1)

                    with tqdm(tiles, desc=f"""Slide {ind+1}/{num_slides} ({int(((ind+1)/num_slides)*100)}%)""",
                              initial=tile_ind,position=0, leave=True) as tile_tqdm:
                        for tile_ind, tile in enumerate(tiles[tile_ind:], start=tile_ind):
                            tile = pipeline.transform(tile)
                            tiles_in_path_out_filename_tuples.append([tile.path, tile.out_filename])
                            tile_tqdm.update(1)
                            if tile_ind == 30:
                                raise Exception('Test')
                    self.save_summary_df(tiles_in_path_out_filename_tuples)

                    self._log(f"""Finished processing {len(tiles)} tiles.""", importance=1)
                self.set(f'Finished ({resolution}, {i}).', True)
            self.set('Done preprocessing', True)
            self.save_metadata()
            self._log(f"""Finish processing {self}""", importance=1)
        except Exception as e:
            self._log(f"""Processing interrupt on slide {ind+1}/{num_slides} ({int(((ind+1)/num_slides)*100)}%)""", importance=2)
            if 'tiles_in_path_out_filename_tuples' in locals():
                self.save_summary_df(tiles_in_path_out_filename_tuples)
            self.save_metadata()
            self._log(f"""Exception {e}""", importance=2)
            self._log(f"""Traceback {traceback.format_exc()}""", importance=2)
        return self

    def get_tiles(self, **kwargs):
        if not self.get('tile_dir', soft=True):
            raise Exception("""You have to tile the image before applying a pipeline over tiles. 
                                tile_dir is None.""")
        tile_dir = self.get('tile_dir')
        return [Tile(tile_path, **kwargs) for tile_path in sorted(glob(f"{tile_dir}/**/*.jpg", recursive=True))] # all .jpg files

    def get_previous_tile_place(self):
        summary_df = self.get_tile_summary_df()
        if summary_df.empty:
            return 0, []
        return len(summary_df), list(zip(summary_df['tile_path'], summary_df['filename']))

    def save_summary_df(self, tiles_in_path_out_filename_tuples):
        out_path = os.path.join(os.path.dirname(self.path), 'tile_summary_df.csv')
        df = generate_summary_df_from_filepaths(tiles_in_path_out_filename_tuples)
        df.to_csv(out_path, index=False)
        self.set('tile_summary_df_path', out_path)
        self.log(f"""Summary df saved for slide {self}.""", importance=2)
        return df

    def get_tile_summary_df(self):
        if not self.get('tile_summary_df_path', soft=True):
            return pd.DataFrame()
        df = pd.read_csv(self.get('tile_summary_df_path'))
        df.set_index(['row', 'col'], inplace=True, drop=False)
        return df
    
    def update_recovery_tile_summary_df(self, tiles_in_path_out_filename_tuples):
        curr_df = generate_summary_df_from_filepaths(tiles_in_path_out_filename_tuples)
        upd_df = self.get_tile_summary_df()
        if upd_df.empty:
            return
        curr_df = curr_df[~curr_df.tile_path.isin(upd_df)] # removing overlapping tiles
        new_df = pd.concat([curr_df, upd_df], axis=0)
        new_df.fillna(False, inplace=True)
        new_df.reset_index(drop=True)
        new_df.to_csv(self.get('tile_summary_df_path'), index=False)
        self.log(f"""Summary df recovery update for slide {self}.""", importance=2)
        return new_df

    def __str__(self):
        if self.img is None:
            return f"""<{type(self).__name__} - uuid:{self.get('slide_uuid')} Not loaded.>"""
        shape = (self.img.height, self.img.width, self.img.bands)
        otsu_val = self.get('otsu_val',soft=True)
        return f"""<{type(self).__name__} - shape:{shape}, otsu_val:{otsu_val}, uuid:{self.get('slide_uuid')}>"""
