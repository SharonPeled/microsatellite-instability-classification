from sklearn.pipeline import Pipeline
from ..components.Dataset import SlideDataset
from ..components.LoggingFunctionTransformer import LoggingFunctionTransformer
from .function_transformers import *
from ..configs import Configs
from ..components.ParallelProcessingManager import ParallelProcessingManager






# TODO: documenting all the tricks

# TODO: fine tuned the thresholds (more strict)

# TODO: learn about slurm


def execute_preprocessing_pipeline(with_tiling, num_processes):
    Logger.log('Starting preprocessing ..', log_importance=1)
    process_manager = ParallelProcessingManager(num_processes=num_processes,
                                                verbose=Configs.VERBOSE,
                                                log_importance=Configs.LOG_IMPORTANCE,
                                                log_format=Configs.LOG_FORMAT,
                                                random_seed=Configs.RANDOM_SEED,
                                                tile_progress_log_freq=Configs.TILE_PROGRESS_LOG_FREQ)
    slide_dataset = SlideDataset(Configs.SLIDES_DIR, load_metadata=Configs.LOAD_METADATA, device=Configs.DEVICE,
                                 slide_log_file=Configs.SLIDE_LOG_FILE)

    pipeline_list = [
        ('slide', Pipeline([
            ('load_slide', LoggingFunctionTransformer(load_slide,
                                                      kw_args={'load_level': Configs.REDUCED_LEVEL_TO_MEMORY})),
            ('scale_mpp', LoggingFunctionTransformer(resize, kw_args={'target_mag_power': Configs.TARGET_MAG_POWER,
                                                                      'mag_attr': Configs.MAG_ATTR})),
            ('load_reduced_image_to_memory', LoggingFunctionTransformer(load_reduced_image_to_memory)),
            ('center_crop_reduced_image', LoggingFunctionTransformer(center_crop_reduced_image,
                                                                     kw_args={'tile_size': Configs.TILE_SIZE,
                                                                              'tissue_attr': Configs.TISSUE_ATTR})),
            ('filter_otsu_reduced_image', LoggingFunctionTransformer(filter_otsu_reduced_image,
                                                                     kw_args=Configs.OTSU_FILTER)),
            ('filter_black_reduced_image', LoggingFunctionTransformer(filter_black_reduced_image, kw_args=Configs.BLACK_FILTER)),
            ('filter_pen_reduced_image', LoggingFunctionTransformer(filter_pen_reduced_image, kw_args=Configs.PEN_FILTER)),
            ('generate_slide_color_grid', LoggingFunctionTransformer(generate_slide_color_grid,
                                                                     kw_args={'attrs_to_colors_map': Configs.ATTRS_TO_COLOR_MAP})),
            ('unload_reduced_image', LoggingFunctionTransformer(unload_reduced_image)),
            ('center_crop', LoggingFunctionTransformer(center_crop)),
            ('fit_color_normalizer', LoggingFunctionTransformer(fit_color_normalizer,
                                                                kw_args={'ref_img_path': Configs.COLOR_NORM_REF_IMG}))
        ]))]

    if with_tiling:
        pipeline_list.append(
            ('tile', Pipeline([
                ('load_tile', LoggingFunctionTransformer(load_tile)),
                ('macenko_color_norm', LoggingFunctionTransformer(macenko_color_norm,
                                                                  kw_args={'succ_norm_attr': Configs.COLOR_NORM_SUCC,
                                                                           'fail_norm_attr': Configs.COLOR_NORM_FAIL})),
                ('save_processed_tile', LoggingFunctionTransformer(save_processed_tile,
                                                                   kw_args={'processed_tiles_dir': Configs.PROCESSED_TILES_DIR,
                                                                            'fail_norm_attr': Configs.COLOR_NORM_FAIL}))
            ]))
        )
    slide_dataset.apply_pipeline(pipeline_list, process_manager)
