from sklearn.pipeline import Pipeline
from src.components.Dataset import SlideDataset
from src.components.Logger import Logger
from src.components.LoggingFunctionTransformer import LoggingFunctionTransformer
from src.preprocessing.function_transformers import *
from src.configs import Configs


# TODO: adding save CDFs graphs and pixel images
# TODO: adding logs!
# TODO: Spread more logs
# TODO: make it with a progress bar and log to file
# TODO: handle when process stops
# TODO: Not to process filtered tiles
if __name__ == '__main__':
    Logger.log('Starting preprocessing ..', importance=1)
    slide_dataset = SlideDataset(Configs.SLIDES_DIR, load_metadata=Configs.LOAD_METADATA)
    pipeline_list = [
        ('slide', Pipeline([
            ('load_slide', LoggingFunctionTransformer(load_slide)),
            ('scale_mpp', LoggingFunctionTransformer(resize, kw_args={'target_mpp': Configs.TARGET_MPP})),
            ('center_crop', LoggingFunctionTransformer(center_crop, kw_args={'tile_size': Configs.TILE_SIZE})),
            ('calc_otsu', LoggingFunctionTransformer(calc_otsu)),
            ('save_tiles', LoggingFunctionTransformer(save_tiles, kw_args={'tiles_dir': Configs.TILES_DIR,
                                                                           'tile_size': Configs.TILE_SIZE}))
        ])),
        ('tile', Pipeline([
            ('load_tile', LoggingFunctionTransformer(load_tile)),
            ('filter_black', LoggingFunctionTransformer(filter_black, kw_args=Configs.BLACK_FILTER)),
            ('filter_pen', LoggingFunctionTransformer(filter_pen, kw_args=Configs.PEN_FILTER)),
            ('filter_otsu', LoggingFunctionTransformer(filter_otsu, kw_args=Configs.OTSU_FILTER)),
            ('macenko_color_norm', LoggingFunctionTransformer(macenko_color_norm,
                                                              kw_args={'ref_img_path':Configs.COLOR_NORM_REF_IMG,
                                                                       'succ_norm_suffix': Configs.COLOR_NORMED_SUFFIX})),
            ('save_processed_tile', LoggingFunctionTransformer(save_processed_tile,
                                                               kw_args={'processed_tiles_dir': Configs.PROCESSED_TILES_DIR}))])),
        ('slide', LoggingFunctionTransformer(recover_missfiltered_tiles, kw_args={'pen_filter': Configs.PEN_FILTER,
                                                                                  'black_filter': Configs.BLACK_FILTER,
                                                                                  'superpixel_size': Configs.SUPERPIXEL_SIZE,
                                                                                  'tile_recovery_suffix': Configs.TILE_RECOVERY_SUFFIX,
                                                                                  'tile_suffixes': Configs.TILE_SUFFIXES,
                                                                                  'processed_tiles_dir': Configs.PROCESSED_TILES_DIR})),
    ]
    slide_dataset.apply_pipeline(pipeline_list)
