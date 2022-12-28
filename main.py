from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from src.components.Dataset import SlideDataset
from src.preprocessing.function_transformers import *
from src.configs import Configs


# TODO: adding save CDFs graphs and pixel images
# TODO: adding logs!
if __name__ == '__main__':
    slide_dataset = SlideDataset(Configs.SLIDE_DIR)
    pipeline_list = [
        ('slide', Pipeline([
            ('load_slide', FunctionTransformer(load_slide)),
            ('scale_mpp', FunctionTransformer(resize, kw_args={'target_mpp': Configs.TARGET_MPP})),
            ('center_crop', FunctionTransformer(center_crop, kw_args={'tile_size': Configs.TILE_SIZE})),
            ('calc_otsu', FunctionTransformer(calc_otsu)),
            ('save_tiles', FunctionTransformer(save_tiles, kw_args={'tile_dir': Configs.TILE_DIR,
                                                                    'tile_size': Configs.TILE_SIZE}))])),
        ('tile', Pipeline([
            ('load_tile', FunctionTransformer(load_tile)),
            ('filter_black', FunctionTransformer(filter_black, kw_args=Configs.BLACK_FILTER)),
            ('filter_pen', FunctionTransformer(filter_pen, kw_args=Configs.PEN_FILTER)),
            ('filter_otsu', FunctionTransformer(center_crop, kw_args=Configs.OTSU_FILTER)),
            ('macenko_color_norm', FunctionTransformer(macenko_color_norm, kw_args=Configs.OTSU_FILTER)),
            ('save_processed_tile', FunctionTransformer(save_processed_tile,
                                                        kw_args={'processed_tile_dir': Configs.PROCESSED_TILE_DIR}))])),
        ('slide', FunctionTransformer(recover_missfiltered_tiles))
    ]
    slide_dataset.apply_pipeline(pipeline_list)


