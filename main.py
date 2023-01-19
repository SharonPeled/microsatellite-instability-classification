import argparse
from src.preprocessing.pipeline import execute_preprocessing_pipeline
from src.utils import bring_files, bring_joined_log_file
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
    parser.add_argument('--bring-slide-logs', type=str)
    parser.add_argument('--num_processes', type=int)
    args = parser.parse_args()
    if args.suppress_signals:
        catchable_sigs = set(signal.Signals) - {signal.SIGKILL, signal.SIGSTOP}
        for sig in catchable_sigs:
            signal.signal(sig, write_to_file)
    if args.preprocess:
        execute_preprocessing_pipeline(with_tiling=True, num_processes=args.num_processes)
    if args.thumbnails_only:
        execute_preprocessing_pipeline(with_tiling=False, num_processes=args.num_processes)
    if args.bring_thumbnails:
        bring_files(Configs.SLIDES_DIR, '*.png', args.bring_thumbnails)
    if args.bring_slide_logs:
        bring_joined_log_file(Configs.SLIDES_DIR, Configs.SLIDE_LOG_FILE, args.bring_slide_logs)


if __name__ == "__main__":
    main()


