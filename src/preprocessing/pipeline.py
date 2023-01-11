from sklearn.pipeline import Pipeline
from ..components.Dataset import SlideDataset
from ..components.LoggingFunctionTransformer import LoggingFunctionTransformer
from .function_transformers import *
from ..configs import Configs




# TODO: handle the MAG - not resizing, adding mag to metadata

# TODO: otsu approx on small image!!

# TODO: try to multi thread
# TODO: learn about slurm

def execute_preprocessing_pipeline():
    Logger.log('Starting preprocessing ..', log_importance=1)
    slide_dataset = SlideDataset(Configs.SLIDES_DIR, load_metadata=Configs.LOAD_METADATA)
    pipeline_list = [
        ('slide', Pipeline([
            ('load_slide', LoggingFunctionTransformer(load_slide)),
            # ('scale_mpp', LoggingFunctionTransformer(resize, kw_args={'target_mpp': Configs.TARGET_MPP})),
            ('center_crop', LoggingFunctionTransformer(center_crop, kw_args={'tile_size': Configs.TILE_SIZE})),
            ('calc_otsu', LoggingFunctionTransformer(calc_otsu, log_importance=1)),
            # ('save_tiles', LoggingFunctionTransformer(save_tiles, kw_args={'tiles_dir': Configs.TILES_DIR,
            #                                                                'tile_size': Configs.TILE_SIZE}))
        ])),
        ('tile', Pipeline([
            ('load_tile', LoggingFunctionTransformer(load_tile)),
            ('filter_otsu', LoggingFunctionTransformer(filter_otsu, kw_args=Configs.OTSU_FILTER)),
            ('filter_black', LoggingFunctionTransformer(filter_black, kw_args=Configs.BLACK_FILTER)),
            ('filter_pen', LoggingFunctionTransformer(filter_pen, kw_args=Configs.PEN_FILTER)),
            # ('macenko_color_norm', LoggingFunctionTransformer(macenko_color_norm,
            #                                                   kw_args={'ref_img_path': Configs.COLOR_NORM_REF_IMG,
            #                                                            'succ_norm_suffix': Configs.COLOR_NORMED_SUFFIX,
            #                                                            'fail_norm_suffix': Configs.FAIL_COLOR_NORMED_SUFFIX})),
            ('save_processed_tile', LoggingFunctionTransformer(save_processed_tile,
                                                               kw_args={'processed_tiles_dir': Configs.PROCESSED_TILES_DIR,
                                                                        'tissue_suffix': Configs.TISSUE_SUFFIX}))
        ])),
        ('slide', Pipeline([
            ('recover_missfiltered_tiles', LoggingFunctionTransformer(recover_missfiltered_tiles,
                                                                      kw_args={'pen_filter': Configs.PEN_FILTER,
                                                                               'black_filter': Configs.BLACK_FILTER,
                                                                               'superpixel_size': Configs.SUPERPIXEL_SIZE,
                                                                               'tile_suffixes': Configs.TILE_SUFFIXES,
                                                                               'ref_img_path': Configs.COLOR_NORM_REF_IMG,
                                                                               'processed_tiles_dir': Configs.PROCESSED_TILES_DIR})),
             ('generate_slide_color_grid', LoggingFunctionTransformer(generate_slide_color_grid,
                                                                      kw_args={'tile_suffixes': Configs.TILE_SUFFIXES,
                                                                               'suffixes_to_colors_map': Configs.SUFFIXES_TO_COLOR_MAP}))
        ])),
    ]
    slide_dataset.apply_pipeline(pipeline_list)
