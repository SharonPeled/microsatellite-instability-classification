from torch.utils.data import Dataset
import os
from glob import glob
from .Slide import Slide
from .Logger import Logger


class SlideDataset(Dataset, Logger):
    def __init__(self, slides_dir):
        self.slides_dir = slides_dir
        self.slide_paths = glob(f"{slides_dir}/**/*.svs", recursive=True)  # all .svs files
        self.slides = None
        self._log(f'Created with {len(self.slide_paths)} slides.')

    def apply_pipeline(self, pipeline):
        self._log(f'Applying pipeline on {len(self.slide_paths)} slides.')
        self.slides = [Slide(path).apply_pipeline(pipeline) for path in self.slide_paths]
        self._log(f'Finished pipeline on {len(self.slide_paths)} slides.')

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, idx):
        return self.slides[idx]

