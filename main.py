from sklearn.pipeline import Pipeline
from src.components.Dataset import SlideDataset
from src.components.Logger import Logger
from src.components.LoggingFunctionTransformer import LoggingFunctionTransformer
from src.preprocessing.function_transformers import *
from src.configs import Configs


# TODO: adding save CDFs graphs and pixel images
# TODO: adding logs!
# TODO: Override the LoggingFunctionTransformer and add logs
if __name__ == '__main__':
    Logger.log('Starting preprocessing ..')
    slide_dataset = SlideDataset(Configs.SLIDES_DIR)
    pipeline_list = [
        ('slide', Pipeline([
            ('load_slide', LoggingFunctionTransformer(load_slide)),
            ('scale_mpp', LoggingFunctionTransformer(resize, kw_args={'target_mpp': Configs.TARGET_MPP})),
            ('center_crop', LoggingFunctionTransformer(center_crop, kw_args={'tile_size': Configs.TILE_SIZE})),
            ('calc_otsu', LoggingFunctionTransformer(calc_otsu)),
            ('save_tiles', LoggingFunctionTransformer(save_tiles, kw_args={'tiles_dir': Configs.TILES_DIR,
                                                                           'tile_size': Configs.TILE_SIZE}))])),
        ('tile', Pipeline([
            ('load_tile', LoggingFunctionTransformer(load_tile)),
            ('filter_black', LoggingFunctionTransformer(filter_black, kw_args=Configs.BLACK_FILTER)),
            ('filter_pen', LoggingFunctionTransformer(filter_pen, kw_args=Configs.PEN_FILTER)),
            ('filter_otsu', LoggingFunctionTransformer(filter_otsu, kw_args=Configs.OTSU_FILTER)),
            ('macenko_color_norm', LoggingFunctionTransformer(macenko_color_norm,
                                                              kw_args={'ref_img_path':Configs.COLOR_NORM_REF_IMG})),
            ('save_processed_tile', LoggingFunctionTransformer(save_processed_tile,
                                                               kw_args={'processed_tiles_dir': Configs.PROCESSED_TILES_DIR}))])),
        # ('slide', LoggingFunctionTransformer(recover_missfiltered_tiles)) # TODO:  review this
    ]
    slide_dataset.apply_pipeline(pipeline_list)
