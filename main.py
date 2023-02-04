import argparse
from src.preprocessing.pipeline import execute_preprocessing_pipeline
from src.tumor_classification.training import train
from src.tumor_classification.predict import predict
from src.utils import bring_files, bring_joined_log_file, delete_all_artifacts, \
    generate_thumbnails_with_tumor_classification, load_df_tumor_pred
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
    parser.add_argument('--bring-slide-logs', type=str)
    parser.add_argument('--train-tumor-classifier', action='store_true')
    parser.add_argument('--inference-tumor-tiles', action='store_true')
    parser.add_argument('--num-processes', type=int)
    parser.add_argument('--generate-tumor-thumbnails', action='store_true')
    parser.add_argument('--clean-all', action='store_true',
                        help='Deletes all logs, tiles and other artifacts generated by the program.')
    args = parser.parse_args()
    if args.suppress_signals:
        catchable_sigs = set(signal.Signals) - {signal.SIGKILL, signal.SIGSTOP}
        for sig in catchable_sigs:
            signal.signal(sig, write_to_file)
    if args.clean_all:
        print("Are you sure you want to delete all generated artifacts?")
        if input().lower() in ['y', 'yes']:
            delete_all_artifacts(Configs)
    if args.preprocess:
        execute_preprocessing_pipeline(with_tiling=True, num_processes=args.num_processes)
    if args.thumbnails_only:
        execute_preprocessing_pipeline(with_tiling=False, num_processes=args.num_processes)
    if args.bring_thumbnails:
        bring_files(Configs.SLIDES_DIR, 'thumbnail.png', args.bring_thumbnails)
    if args.bring_tumor_thumbnails:
        bring_files(Configs.SLIDES_DIR, 'tumor_thumbnail.png', args.bring_tumor_thumbnails)
    if args.bring_slide_logs:
        bring_joined_log_file(Configs.SLIDES_DIR, Configs.PROGRAM_LOG_FILE_ARGS[0], args.bring_slide_logs)
    if args.train_tumor_classifier:
        train()
    if args.inference_tumor_tiles:
        predict()
    if args.generate_tumor_thumbnails:
        df_tumor_pred = load_df_tumor_pred(pred_dir=Configs.TUMOR_PREDICT_OUTPUT_PATH,
                                           class_to_index=Configs.TUMOR_CLASS_TO_IND,
                                           tumor_class_name=Configs.TUMOR_CLASS)
        generate_thumbnails_with_tumor_classification(df_tumor_pred=df_tumor_pred,
                                                      slides_dir=Configs.SLIDES_DIR)


if __name__ == "__main__":
    main()


