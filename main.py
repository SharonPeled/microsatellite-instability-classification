import os
from glob import glob
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from src.components.Dataset import SlideDataset
from src.configs import Configs
from src.preprocessing.function_transformers import *

# TODO: adding save CDFs graphs and pixel images
# TODO: adding logs!
if __name__ == '__main__':
    slide_dataset = SlideDataset(Configs.SLIDE_DIR)
    pipeline = [
        Pipeline([('scale_mpp', FunctionTransformer(resize, kw_args={'target_mpp': Configs.TARGET_MPP})),
                  ('center_crop', FunctionTransformer(center_crop, kw_args={'tile_size': Configs.TILE_SIZE})),
                  ('calc_otsu', FunctionTransformer(calc_otsu)),
                  ('tile', FunctionTransformer(tile, kw_args={'tile_dir': Configs.TILE_DIR,
                                                              'tile_size': Configs.TILE_SIZE}))]),
        Pipeline([('filter_otsu', FunctionTransformer(center_crop, kw_args=Configs.OTSU_FILTER)), #TODO: add the sort of filters with fixing random black spots
                  ('filter_black', FunctionTransformer(filter_black, kw_args=Configs.BLACK_FILTER)),
                  ('filter_pen', FunctionTransformer(filter_pen, kw_args=Configs.PEN_FILTER)),
                  ])
    ]
    slide_dataset.apply_pipeline(pipeline)

# img:
#     image_iterator
#     load
#     resize
#     crop
#     tile
#     tile_iterator (iter, pipline):
#         pipeline
#         otsu_filter
#         dark_filter
#         pen_filter
#         color_norm



