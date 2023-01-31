import torchstain
from PIL import Image
from torchvision import transforms


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
        return self.torch_normalizer.normalize(x * 255, stains=False)[0].permute(2, 0, 1) / 255.0
