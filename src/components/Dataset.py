from torch.utils.data import Dataset
import os
from glob import glob
from .Slide import Slide


class SlideDataset(Dataset):
    def __init__(self, slide_dir):
        self.slide_dir = slide_dir
        self.slides = [Slide(y) for x in os.walk(slide_dir)
                       for y in glob(os.path.join(x[0], '*.svs'))] # all .svs files

    def apply_pipeline(self, pipeline):
        self.slides = [slide.apply_pipeline(pipeline) for slide in self.slides]

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, idx):
        return self.slides[idx]

