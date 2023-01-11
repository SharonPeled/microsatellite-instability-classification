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
from ..utils import generate_summary_df_from_filepaths, get_time
import datetime


class Slide(Image):
    def __init__(self, path, slide_uuid=None, load_metadata=True, **kwargs):
        """
        :param slide_uuid:
        :param path: if slide_uuid=None then path directory name must be uuid of the slide, as in the gdc-client format
        :param tiles_dir: directory for storing all tiles from all slides
        """
        super().__init__(path=path, slide_uuid=slide_uuid, **kwargs)
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
        self._log(f"""metadata saved for slide {self}.""", log_importance=2)

    def _get_uuid(self):
        return Path(self.path).parent.name

    def set_tile_dir(self, tiles_dir):
        self.set('tile_dir', os.path.join(tiles_dir, self.get('slide_uuid')))

    def apply_pipeline(self, pipeline_list, ind, num_slides):
        if self.get('Done preprocessing', soft=True):
            self._log(f"""Slide already processed: {ind+1}/{num_slides}""", log_importance=1)
            return self
        try:
            self._log(f"""Processing {self}""", log_importance=1)
            for i, (resolution, pipeline) in enumerate(pipeline_list):

                if resolution == 'slide':
                    self = pipeline.transform(self)

                elif resolution == 'tile':
                    tile_size = self.get('tile_size')
                    tile_ind, tiles_in_path_out_filename_tuples = self.get_previous_tile_place()
                    tile_coords = self.get_tile_coordinates()
                    left = len(tile_coords) - tile_ind
                    self._log(f"""Processed {tile_ind}/{len(tile_coords)} tiles.""", log_importance=1)
                    self._log(f"""Processing {len(tile_coords) - tile_ind} tiles.""", log_importance=1)

                    with tqdm(tile_coords, initial=tile_ind,position=0, leave=True) as tile_tqdm:
                        for tile_ind, (x,y) in enumerate(tile_coords[tile_ind:], start=tile_ind):
                            tile_img = self.img.crop(x*tile_size, y*tile_size, tile_size, tile_size)
                            tile = Tile(path=f"{x}_{y}.jpg", img=tile_img, otsu_val=self.get('otsu_val'), slide_uuid=self.get('slide_uuid'))
                            tile = pipeline.transform(tile)
                            tiles_in_path_out_filename_tuples.append([tile.path, tile.out_filename])
                            tile_tqdm.update(1)
                            tile_tqdm.set_description(f"""{str(get_time())}  [Slide] ({ind+1}/{num_slides} {int(((ind+1)/num_slides)*100)}%)""",
                                                      refresh=True)

                    self.save_summary_df(tiles_in_path_out_filename_tuples)
                    self._log(f"""Finished processing {len(tile_coords)} tiles.""", log_importance=1)

                self.set(f'Finished ({resolution}, {i}).', True)
            self.set('Done preprocessing', True)
            self.save_metadata()
            self._log(f"""Finish processing {self}""", log_importance=1)
        except Exception as e:
            self._log(f"""Processing interrupt on slide {ind+1}/{num_slides} ({int(((ind+1)/num_slides)*100)}%)""", log_importance=2)
            if 'tiles_in_path_out_filename_tuples' in locals():
                self.save_summary_df(tiles_in_path_out_filename_tuples)
            self.save_metadata()
            self._log(f"""Exception {e}""", log_importance=2)
            self._log(f"""Traceback {traceback.format_exc()}""", log_importance=2)
        return self

    def get_previous_tile_place(self):
        summary_df = self.get_tile_summary_df()
        if summary_df.empty:
            return 0, []
        return len(summary_df), list(zip(summary_df['tile_path'], summary_df['filename']))

    def get_tile_coordinates(self):
        num_x_tiles = self.get('num_x_tiles')
        num_y_tiles = self.get('num_y_tiles')
        return [(x,y) for x in range(num_x_tiles) for y in range(num_y_tiles)]

    def save_summary_df(self, tiles_in_path_out_filename_tuples):
        out_path = os.path.join(os.path.dirname(self.path), 'tile_summary_df.csv')
        df = generate_summary_df_from_filepaths(tiles_in_path_out_filename_tuples)
        df.to_csv(out_path, index=False)
        # adding percentages of each filter of metadata
        tile_value_counts = {k:list(v)
                             for k,v in df.select_dtypes(include='bool').apply(lambda x: [x.mean(), x.sum()]).items()}
        for k,v in tile_value_counts.items():
            self.set(k, v)
        self._log(f"Tiles value counts: {tile_value_counts}", log_importance=1)
        self.set('tile_summary_df_path', out_path)
        self.log(f"""Summary df saved for slide {self}.""", log_importance=2)
        return df

    def get_tile_summary_df(self):
        if not self.get('tile_summary_df_path', soft=True):
            return pd.DataFrame()
        df = pd.read_csv(self.get('tile_summary_df_path'))
        df.set_index(['row', 'col'], inplace=True, drop=False)
        return df
    
    def update_recovery_tile_summary_df(self, tiles_in_path_out_filename_tuples):
        upd_df = generate_summary_df_from_filepaths(tiles_in_path_out_filename_tuples)
        if upd_df.empty:
            return
        curr_df = self.get_tile_summary_df()
        non_overlap_curr_df = curr_df[~curr_df.tile_path.isin(upd_df.tile_path)] # removing overlapping tiles
        new_df = pd.concat([non_overlap_curr_df, upd_df], axis=0)
        new_df.fillna(False, inplace=True)
        new_df.reset_index(drop=True)
        new_df.to_csv(self.get('tile_summary_df_path'), index=False)
        self.log(f"""Summary df recovery update for slide {self}.""", log_importance=2)

    def __str__(self):
        if self.img is None:
            return f"""<{type(self).__name__} - uuid:{self.get('slide_uuid')} Not loaded.>"""
        shape = (self.img.height, self.img.width, self.img.bands)
        otsu_val = self.get('otsu_val',soft=True)
        return f"""<{type(self).__name__} - shape:{shape}, otsu_val:{otsu_val}, uuid:{self.get('slide_uuid')}>"""
