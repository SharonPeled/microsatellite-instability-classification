from torch.utils.data import Dataset
import os
from glob import glob
from .Slide import Slide
import logging


class SlideDataset(Dataset):
    def __init__(self, slide_dir):
        self.slide_dir = slide_dir
        self.slide_paths = [y for x in os.walk(slide_dir) 
                            for y in glob(os.path.join(x[0], '*.svs'))] # all .svs files
        self.slides = None
        logging.info(f'Created with {len(self.slide_paths)} slides.')

    def apply_pipeline(self, pipeline):
        logging.info(f'Applying pipeline on {len(self.slide_paths)} slides.')
        self.slides = [Slide(path).apply_pipeline(pipeline) for path in self.slide_paths]
        logging.info(f'Finished pipeline on {len(self.slide_paths)} slides.')

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, idx):
        return self.slides[idx]

