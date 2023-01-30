from torch.utils.data import DataLoader, Dataset, Subset
import torchstain
from PIL import Image
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchstain")


class TileDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)
    

class MacenkoNormalizerTransform(object):
    def __init__(self, ref_path):
        ref_img = Image.open(ref_path)
        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255)
        ])
        self.torch_normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
        self.torch_normalizer.fit(T(ref_img))

    def __call__(self, x):
        return self.torch_normalizer.normalize(x*255, stains=False)[0].permute(2,0,1)/255.0