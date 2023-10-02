import argparse
# from src.tumor_classification.training import train as train_tumor
# from src.tumor_classification.predict import predict as predict_tumor
# from src.semantic_segmentation.OOD_validation_tumor_TCGA import OOD_validation_tumor_TCGA
# from src.semantic_segmentation.OOD_validation_ss_IRCCS import OOD_validation_ss_IRCCS
# from src.semantic_segmentation.training import train as train_semantic_seg
# from src.semantic_segmentation.predict import predict as predict_semantic_seg
# from src.tumor_distance_estimation.training import train as train_tumor_regression
from src.general_utils import get_time, set_global_configs
# from src.subtype_classification.training_MIL import train as train_subtype_classification_mil
# from src.subtype_classification.training_tile_based import train as train_subtype_classification_tile
# from src.subtype_classification.pretraining_tile_based import train as pretrain_subtype_classification_tile
# from src.variant_classification.training import train as train_variant_classification
# from src.variant_classification.permutaion_test import train as permutation_variant_classification
import signal
import datetime
from src.configs import Configs
import matplotlib
import json
matplotlib.use('agg')


def deploy_config_file(filepath):
    with open(filepath, 'rb') as file:
        config_dict = json.load(file)
    time_str = get_time()
    for key, val in config_dict.items():
        if isinstance(val, list) and '{time}' in val[0]:
            val[0] = config_dict[key][0].format(time=time_str)
        if hasattr(Configs, key):
            setattr(Configs, key, val)
        else:
            raise NotImplementedError(f"Config not recognized: {key}: {val}.")


def write_to_file(s, frame_object=None, **kargs):
    """
    Signal handler - writes to file and don't stop the program
    Process still can be killed using: kill -9 <pid>
    """
    with open("out.txt", "a") as file:
        s = f"{str(datetime.datetime.now())}: {str(s)}\n"
        file.write(s)
        file.flush()
        print(s, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filepath', type=str)
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument("--slide_ids", nargs="+", type=str)
    parser.add_argument('--thumbnails-only', action='store_true')
    parser.add_argument('--suppress-signals', action='store_true')
    parser.add_argument('--bring-thumbnails', type=str)
    parser.add_argument('--bring-tumor-thumbnails', type=str)
    parser.add_argument('--bring-semantic-seg-thumbnails', type=str)
    parser.add_argument('--bring-slide-logs', type=str)
    parser.add_argument('--train-tumor-classifier', action='store_true')
    parser.add_argument('--train-semantic-seg', action='store_true')
    parser.add_argument('--train-tumor-regression', action='store_true')
    parser.add_argument('--train-subtype-classification-tile', action='store_true')
    parser.add_argument('--pretrain-subtype-classification-tile', action='store_true')
    parser.add_argument('--train-subtype-classification-mil', action='store_true')
    parser.add_argument('--train-variant-classification', action='store_true')
    parser.add_argument('--permutation-variant-classification', action='store_true')
    parser.add_argument('--OOD-validation-tumor-TCGA', action='store_true')
    parser.add_argument('--OOD-validation-ss-IRCCS', action='store_true')
    parser.add_argument('--inference-semantic-seg', action='store_true')
    parser.add_argument('--inference-tumor-tiles', action='store_true')
    parser.add_argument('--num-processes', type=int)
    parser.add_argument('--generate-tumor-thumbnails', action='store_true')
    parser.add_argument('--generate-semantic-seg-thumbnails', action='store_true')
    parser.add_argument('--clean-all', action='store_true',
                        help='Deletes all logs, tiles and other artifacts generated by the program.')
    args = parser.parse_args()
    if args.config_filepath:
        deploy_config_file(args.config_filepath)
    set_global_configs(verbose=Configs.VERBOSE,
                       log_file_args=Configs.PROGRAM_LOG_FILE_ARGS,
                       log_importance=Configs.LOG_IMPORTANCE,
                       log_format=Configs.LOG_FORMAT,
                       random_seed=Configs.RANDOM_SEED,
                       tile_progress_log_freq=Configs.TILE_PROGRESS_LOG_FREQ)
    if args.suppress_signals:
        print("suppressing signals!", flush=True)
        catchable_sigs = set(signal.Signals) - {signal.SIGKILL, signal.SIGSTOP}
        for sig in catchable_sigs:
            signal.signal(sig, write_to_file)
            print(f"{sig} - Suppressed.", flush=True)
    # if args.clean_all:
    #     print(f"Are you sure you want to delete all generated artifacts in {Configs.ROOT} ?")
    #     if input().lower() in ['y', 'yes']:
    #         delete_all_artifacts(Configs)
    if args.preprocess:
        from src.preprocessing.pipeline import execute_preprocessing_pipeline
        execute_preprocessing_pipeline(with_tiling=True, num_processes=args.num_processes, slide_ids=args.slide_ids)
    if args.thumbnails_only:
        from src.preprocessing.pipeline import execute_preprocessing_pipeline
        execute_preprocessing_pipeline(with_tiling=False, num_processes=args.num_processes, slide_ids=args.slide_ids)
    # if args.bring_slide_logs:
    #     bring_joined_log_file(Configs.SLIDES_DIR, Configs.PROGRAM_LOG_FILE_ARGS[0], args.bring_slide_logs)
    # if args.train_tumor_classifier:
    #     train_tumor()
    # if args.inference_tumor_tiles:
    #     predict_tumor()
    # if args.train_semantic_seg:
    #     train_semantic_seg()
    # if args.inference_semantic_seg:
    #     predict_semantic_seg()
    # if args.train_tumor_regression:
    #     train_tumor_regression()
    # if args.train_subtype_classification_tile:
    #     Configs.set_task_configs('SC')
    #     train_subtype_classification_tile()
    # if args.pretrain_subtype_classification_tile:
    #     Configs.set_task_configs(['DN', 'SC'])
    #     pretrain_subtype_classification_tile()
    # if args.train_subtype_classification_mil:
    #     Configs.set_task_configs('SC')
    #     train_subtype_classification_mil()
    # if args.train_variant_classification:
    #     Configs.set_task_configs('VC')
    #     train_variant_classification()
    # if args.permutation_variant_classification:
    #     Configs.set_task_configs('VC')
    #     permutation_variant_classification()
    # if args.generate_tumor_thumbnails:
    #     df_pred = load_df_pred(pred_dir=Configs.TUMOR_PREDICT_OUTPUT_PATH,
    #                            class_to_index=Configs.TUMOR_CLASS_TO_IND)
    #     generate_thumbnails_with_tissue_classification(df_pred=df_pred,
    #                                                    slides_dir=Configs.SLIDES_DIR,
    #                                                    class_to_index=Configs.TUMOR_CLASS_TO_IND,
    #                                                    class_to_color=Configs.TUMOR_CLASS_TO_COLOR,
    #                                                    summary_df_filename=Configs.SUMMARY_DF_FILENAME,
    #                                                    summary_df_pred_merged_filename=Configs.TUMOR_SUMMARY_DF_PRED_MERGED_FILENAME,
    #                                                    thumbnail_filename=Configs.TUMOR_THUMBNAIL_FILENAME)
    # if args.generate_semantic_seg_thumbnails:
    #     df_pred = load_df_pred(pred_dir=Configs.SS_PREDICT_OUTPUT_PATH,
    #                            class_to_index=Configs.SS_CLASS_TO_IND)
    #     generate_thumbnails_with_tissue_classification(df_pred=df_pred,
    #                                                    slides_dir=Configs.SLIDES_DIR,
    #                                                    class_to_index=Configs.SS_CLASS_TO_IND,
    #                                                    class_to_color=Configs.SS_CLASS_TO_COLOR,
    #                                                    summary_df_filename=Configs.SUMMARY_DF_FILENAME,
    #                                                    summary_df_pred_merged_filename=Configs.SS_SUMMARY_DF_PRED_MERGED_FILENAME,
    #                                                    thumbnail_filename=Configs.SS_THUMBNAIL_FILENAME)
    #
    # if args.OOD_validation_tumor_TCGA:
    #     OOD_validation_tumor_TCGA()
    # if args.OOD_validation_ss_IRCCS:
    #     OOD_validation_ss_IRCCS()
    # if args.bring_thumbnails:
    #     bring_files(Configs.SLIDES_DIR, Configs.THUMBNAIL_FILENAME, args.bring_thumbnails)
    # if args.bring_tumor_thumbnails:
    #     bring_files(Configs.SLIDES_DIR, Configs.TUMOR_THUMBNAIL_FILENAME, args.bring_tumor_thumbnails)
    # if args.bring_semantic_seg_thumbnails:
    #     bring_files(Configs.SLIDES_DIR, Configs.SS_THUMBNAIL_FILENAME, args.bring_semantic_seg_thumbnails)


if __name__ == "__main__":
    main()


