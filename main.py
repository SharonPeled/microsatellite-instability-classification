import argparse
from src.preprocessing.pipeline import execute_preprocessing_pipeline
import signal
import datetime
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--suppress-signals', action='store_true')
    args = parser.parse_args()
    if args.suppress_signals:
        catchable_sigs = set(signal.Signals) - {signal.SIGKILL, signal.SIGSTOP}
        for sig in catchable_sigs:
            signal.signal(sig, write_to_file)
    if args.preprocess:
        execute_preprocessing_pipeline()