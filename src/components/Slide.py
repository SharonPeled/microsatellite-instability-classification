import pandas as pd
from pathlib import Path
import os
from .Tile import Tile
import pyvips
from .Image import Image
from tqdm import tqdm
import json
import traceback
from ..utils import get_time
import datetime
from collections import defaultdict
from torchvision import transforms
import torchstain


class Slide(Image):
    def __init__(self, path, slide_uuid=None, load_metadata=True, device=None, **kwargs):
        """
        :param slide_uuid:
        :param path: if slide_uuid=None then path directory name must be uuid of the slide, as in the gdc-client format
        :param tiles_dir: directory for storing all tiles from all slides
        """
        super().__init__(path=path, slide_uuid=slide_uuid, device=device, **kwargs)
        if slide_uuid is None:
            self.set('slide_uuid', self._get_uuid())
        self.set('metadata_path', os.path.join(os.path.dirname(self.path), 'metadata.json'))
        if load_metadata:
            self.load_slide_metadata()
        self.img_r = None  # reduced image
        self.img_r_level = None
        self.summary_df = pd.DataFrame()
        self.downsample = None
        self.color_normalizer = None

    def load(self, load_level=None):
        self.img = pyvips.Image.new_from_file(self.path).extract_band(0, n=3) # removing alpha channel
        if isinstance(load_level, int):
            if int(self.img.get('openslide.level-count')) - 1 >= load_level:
                self.img_r = pyvips.Image.new_from_file(self.path, level=load_level).extract_band(0, n=3) # removing alpha channel
                self.img_r_level = int(load_level)
        if isinstance(load_level, list):
            for level in load_level:
                if int(self.img.get('openslide.level-count')) - 1 >= level:
                    self.img_r = pyvips.Image.new_from_file(self.path, level=level).extract_band(0, n=3) # removing alpha channel
                    self.img_r_level = int(level)
                    break
        if load_level is not None and self.img_r is None:
            raise Exception(f"Loading level {load_level} failed.")

    def load_level_to_memory(self):
        self.img_r.write_to_memory()
        height_r, width_r = self.img_r.height, self.img_r.width
        width_ratio, height_ratio = int(self.width / width_r), int(self.height / height_r)
        if int(self.width / width_r) != int(self.height / height_r):
            raise Exception(f"""The lower level of slide is downsampled inconsistently across axis.
            Width ratio: {width_ratio}, Height ratio: {height_ratio}""")
        self.downsample = int(self.width / width_r)
        self.log(f"""Reduced image downsampled: {self.downsample}.""", log_importance=1)

    def unload_level_from_memory(self):
        del self.img_r

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

    def init_tiles_summary_df(self, tissue_attr):
        num_x_tiles = self.get('num_x_tiles')
        num_y_tiles = self.get('num_y_tiles')
        self.summary_df = pd.DataFrame(index=[(x,y) for x in range(num_x_tiles)
                                              for y in range(num_y_tiles)]).rename_axis('row_col', axis=1)
        self.summary_df[tissue_attr] = True
        self.set('tissue_attr', tissue_attr)

    def add_attribute_summary_df(self, tile_indexes, attr_name, val, other_val, is_tissue_filter):
        self.summary_df.loc[tile_indexes, attr_name] = val
        self.summary_df[attr_name] = self.summary_df[attr_name].fillna(other_val)
        if is_tissue_filter:
            tissue_attr = self.get('tissue_attr')
            self.summary_df[tissue_attr] = (self.summary_df[tissue_attr]&(~self.summary_df[attr_name]))

    def _get_uuid(self):
        return Path(self.path).parent.name

    def get_downsample(self):
        return self.downsample

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
                    tiles_inds = self.get_tissue_indexes()
                    self._log(f"""Processing {len(tiles_inds)} tissue tiles.""", log_importance=1)

                    with tqdm(tiles_inds, position=0, leave=True) as tile_tqdm:
                        tile_inds_by_norm_res = defaultdict(lambda: [])
                        for x, y in tiles_inds:
                            tile_img = self.img.crop(y*tile_size, x*tile_size, tile_size, tile_size)
                            tile = Tile(path=f"{x}_{y}.jpg", img=tile_img, slide_uuid=self.get('slide_uuid'),
                                        device=self.device, color_normalizer=self.color_normalizer)
                            tile.add_filename_suffix(self.get('tissue_attr'))
                            tile = pipeline.transform(tile)
                            norm_result = tile.get('norm_result', soft=True)
                            if norm_result is not None:
                                tile_inds_by_norm_res[tile.get('norm_result', soft=True)].append((x, y))
                            tile_tqdm.update(1)
                            tile_tqdm.set_description(f"""{str(get_time())}  [Slide] ({ind+1}/{num_slides} {int(((ind+1)/num_slides)*100)}%)""",
                                                      refresh=True)

                    for (res, attr_name), tiles_inds in tile_inds_by_norm_res:
                        is_tissue_filter = not res  # if res is False - norm fail and it is a filter
                        self.add_attribute_summary_df(tiles_inds, attr_name, True, False, is_tissue_filter=is_tissue_filter)

                    self.save_summary_df()
                    if pipeline.steps[-1][0] == 'save_processed_tile':
                        self.set('processed_tiles_dir', pipeline.steps[-1][1].kw_args['processed_tiles_dir'])
                    self._log(f"""Finished processing {len(tiles_inds)} tiles.""", log_importance=1)

                self.set(f'Finished ({resolution}, {i}).', True)
            self.set('Done preprocessing', True)
            self.save_metadata()
            self._log(f"""Finish processing [Slide] ({ind+1}/{num_slides} - {self}""", log_importance=1)
        except Exception as e:
            self._log(f"""Processing interrupt on slide {ind+1}/{num_slides} ({int(((ind+1)/num_slides)*100)}%)""",
                      log_importance=2)
            self.save_summary_df()
            self.save_metadata()
            self._log(f"""Exception {e}""", log_importance=2)
            self._log(f"""Traceback {traceback.format_exc()}""", log_importance=2)
        return self

    def get_tissue_indexes(self):
        tissue_tiles = self.summary_df[self.summary_df[self.get('tissue_attr')]]
        return tissue_tiles.index.values

    def save_summary_df(self):
        out_path = os.path.join(os.path.dirname(self.path), 'summary_df.csv')
        self.summary_df.to_csv(out_path)
        # adding percentages of each filter of metadata
        tile_value_counts = {k: list(v)
                             for k, v in self.summary_df.select_dtypes(include='bool').apply(lambda x:
                                                                                             [x.mean(), x.sum()]).items()}
        for k,v in tile_value_counts.items():
            self.set(k, v)
        self._log(f"Tiles value counts: {tile_value_counts}", log_importance=1)
        self.set('summary_df_path', out_path)
        self.log(f"""Summary df saved for slide {self}.""", log_importance=2)

    def __str__(self):
        if self.img is None:
            return f"""<{type(self).__name__} - uuid:{self.get('slide_uuid')} Not loaded.>"""
        shape = (self.img.height, self.img.width, self.img.bands)
        otsu_val = self.get('otsu_val',soft=True)
        return f"""<{type(self).__name__} - shape:{shape}, otsu_val:{otsu_val}, uuid:{self.get('slide_uuid')}>"""

    def fit_color_normalizer(self, ref_img_path):
        ref_img = pyvips.Image.new_from_file(ref_img_path)
        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255)
        ])
        torch_normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
        torch_normalizer.fit(T(ref_img.numpy()))
        self.color_normalizer = torch_normalizer
