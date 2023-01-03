from torch.utils.data import Dataset
import os
from glob import glob
from .Slide import Slide
from .Logger import Logger


class SlideDataset(Dataset, Logger):
    def __init__(self, slides_dir, load_metadata=True):
        self.slides_dir = slides_dir
        self.slide_paths = sorted(glob(f"{slides_dir}/**/*.svs", recursive=True))  # all .svs files
        self.slides = None
        self.load_metadata = load_metadata
        self._log(f'Created with {len(self.slide_paths)} slides.', importance=1)

    def apply_pipeline(self, pipeline):
        self._log(f'Applying pipeline on {len(self.slide_paths)} slides.', importance=1)
        processed_slides = []
        for ind, path in enumerate(self.slide_paths):
            slide = Slide(path, load_metadata=self.load_metadata)
            slide = slide.apply_pipeline(pipeline, ind, len(self.slide_paths))
            processed_slides.append(slide)
        self.slides = processed_slides
        self._log(f'Finished pipeline on {len(self.slide_paths)} slides.', importance=1)

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, idx):
        return self.slides[idx]

