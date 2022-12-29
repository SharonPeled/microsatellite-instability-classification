from sklearn.pipeline import Pipeline
from src.components.Dataset import SlideDataset
from src.components.LoggingFunctionTransformer import LoggingFunctionTransformer
from src.preprocessing.function_transformers import *
from src.configs import Configs
import logging


# TODO: adding save CDFs graphs and pixel images
# TODO: adding logs!
# TODO: Override the LoggingFunctionTransformer and add logs
if __name__ == '__main__':
    logging.info('Starting preprocessing ..')
    slide_dataset = SlideDataset(Configs.SLIDE_DIR)
    pipeline_list = [
        ('slide', Pipeline([
            ('load_slide', LoggingFunctionTransformer(load_slide)),
            ('scale_mpp', LoggingFunctionTransformer(resize, kw_args={'target_mpp': Configs.TARGET_MPP})),
            ('center_crop', LoggingFunctionTransformer(center_crop, kw_args={'tile_size': Configs.TILE_SIZE})),
            ('calc_otsu', LoggingFunctionTransformer(calc_otsu)),
            ('save_tiles', LoggingFunctionTransformer(save_tiles, kw_args={'tile_dir': Configs.TILE_DIR,
                                                                    'tile_size': Configs.TILE_SIZE}))])),
        ('tile', Pipeline([
            ('load_tile', LoggingFunctionTransformer(load_tile)),
            ('filter_black', LoggingFunctionTransformer(filter_black, kw_args=Configs.BLACK_FILTER)),
            ('filter_pen', LoggingFunctionTransformer(filter_pen, kw_args=Configs.PEN_FILTER)),
            ('filter_otsu', LoggingFunctionTransformer(center_crop, kw_args=Configs.OTSU_FILTER)),
            ('macenko_color_norm', LoggingFunctionTransformer(macenko_color_norm, kw_args=Configs.OTSU_FILTER)),
            ('save_processed_tile', LoggingFunctionTransformer(save_processed_tile,
                                                        kw_args={'processed_tile_dir': Configs.PROCESSED_TILE_DIR}))])),
        ('slide', LoggingFunctionTransformer(recover_missfiltered_tiles))
    ]
    slide_dataset.apply_pipeline(pipeline_list)


