import os.path
import os
from torch.utils.data import Dataset
from glob import glob
from .Slide import Slide
from .Logger import Logger
import time


class SlideDataset(Dataset, Logger):
    def __init__(self, slides_dir, slide_log_file, device, load_metadata=True):
        self.device = device
        self.slides_dir = slides_dir
        self.slide_paths = sorted(glob(f"{slides_dir}/**/*.svs", recursive=True))  # all .svs files
        self.slides = None
        self.load_metadata = load_metadata
        self.slide_log_file = slide_log_file
        self._log(f'Created with {len(self.slide_paths)} slides.', log_importance=1)

    def apply_pipeline(self, pipeline, process_manager):
        self._log(f'Applying pipeline on {len(self.slide_paths)} slides.', log_importance=1)
        param_generator = ((path, pipeline, ind) for ind, path in enumerate(self.slide_paths))
        log_file_generator = (os.path.join(os.path.dirname(slide_path), self.slide_log_file)
                              for slide_path in self.slide_paths)
        self.slides = process_manager.execute_parallel_loop(self._apply_on_slide, param_generator, log_file_generator)
        self._log(f'Finished pipeline on {len(self.slide_paths)} slides.', log_importance=1)

    def _apply_on_slide(self, path, pipeline, ind):
        beg = time.time()
        slide = Slide(path, load_metadata=self.load_metadata, device=self.device)
        slide.apply_pipeline(pipeline, ind, len(self.slide_paths))
        self.log(f"[Slide ({ind+1}/{len(self.slide_paths)})] Total Processing time: {int(time.time() - beg)} seconds.",
                 log_importance=2)

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, idx):
        return self.slides[idx]

