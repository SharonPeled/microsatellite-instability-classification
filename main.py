import argparse
from src.preprocessing.pipeline import execute_preprocessing_pipeline
from src.tumor_classification.training import train as train_tumor
from src.tumor_classification.predict import predict as predict_tumor
from src.semantic_segmentation.OOD_validation_tumor_TCGA import OOD_validation_tumor_TCGA
from src.semantic_segmentation.OOD_validation_ss_IRCCS import OOD_validation_ss_IRCCS
from src.semantic_segmentation.training import train as train_semantic_seg
from src.semantic_segmentation.predict import predict as predict_semantic_seg
from src.utils import bring_files, bring_joined_log_file, delete_all_artifacts, \
    generate_thumbnails_with_tissue_classification, load_df_pred
import signal
import datetime
from src.configs import Configs
import matplotlib
matplotlib.use('agg')


def write_to_file(s, frame_object=None, **kargs):
    """
    Signal handler - writes to file and don't stop the program
    Process still can be killed using: kill -9 <pid>
    """
    with open("out.txt", "a") as file:
        s = f"{str(datetime.datetime.now())}: {str(s)}\n"
        file.write(s)
        file.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--thumbnails-only', action='store_true')
    parser.add_argument('--suppress-signals', action='store_true')
    parser.add_argument('--bring-thumbnails', type=str)
    parser.add_argument('--bring-tumor-thumbnails', type=str)
    parser.add_argument('--bring-semantic-seg-thumbnails', type=str)
    parser.add_argument('--bring-slide-logs', type=str)
    parser.add_argument('--train-tumor-classifier', action='store_true')
    parser.add_argument('--train-semantic-seg', action='store_true')
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
    if args.suppress_signals:
        catchable_sigs = set(signal.Signals) - {signal.SIGKILL, signal.SIGSTOP}
        for sig in catchable_sigs:
            signal.signal(sig, write_to_file)
    if args.clean_all:
        print(f"Are you sure you want to delete all generated artifacts in {Configs.ROOT} ?")
        if input().lower() in ['y', 'yes']:
            delete_all_artifacts(Configs)
    if args.preprocess:
        execute_preprocessing_pipeline(with_tiling=True, num_processes=args.num_processes)
    if args.thumbnails_only:
        execute_preprocessing_pipeline(with_tiling=False, num_processes=args.num_processes)
    if args.bring_slide_logs:
        bring_joined_log_file(Configs.SLIDES_DIR, Configs.PROGRAM_LOG_FILE_ARGS[0], args.bring_slide_logs)
    if args.train_tumor_classifier:
        train_tumor()
    if args.inference_tumor_tiles:
        predict_tumor()
    if args.train_semantic_seg:
        train_semantic_seg()
    if args.inference_semantic_seg:
        predict_semantic_seg()
    if args.generate_tumor_thumbnails:
        df_pred = load_df_pred(pred_dir=Configs.TUMOR_PREDICT_OUTPUT_PATH,
                               class_to_index=Configs.TUMOR_CLASS_TO_IND)
        generate_thumbnails_with_tissue_classification(df_pred=df_pred,
                                                       slides_dir=Configs.SLIDES_DIR,
                                                       class_to_index=Configs.TUMOR_CLASS_TO_IND,
                                                       class_to_color=Configs.TUMOR_CLASS_TO_COLOR)
    if args.generate_semantic_seg_thumbnails:
        df_pred = load_df_pred(pred_dir=Configs.SS_PREDICT_OUTPUT_PATH,
                               class_to_index=Configs.SS_CLASS_TO_IND)
        generate_thumbnails_with_tissue_classification(df_pred=df_pred,
                                                       slides_dir=Configs.SLIDES_DIR,
                                                       class_to_index=Configs.SS_CLASS_TO_IND,
                                                       class_to_color=Configs.SS_CLASS_TO_COLOR)
    if args.OOD_validation_tumor_TCGA:
        OOD_validation_tumor_TCGA()
    if args.OOD_validation_ss_IRCCS:
        OOD_validation_ss_IRCCS()
    if args.bring_thumbnails:
        bring_files(Configs.SLIDES_DIR, 'thumbnail.png', args.bring_thumbnails)
    if args.bring_tumor_thumbnails:
        bring_files(Configs.SLIDES_DIR, 'tumor_thumbnail.png', args.bring_tumor_thumbnails)
    if args.bring_semantic_seg_thumbnails:
        bring_files(Configs.SLIDES_DIR, 'semantic_seg_thumbnail.png', args.bring_semantic_seg_thumbnails)


if __name__ == "__main__":
    main()


