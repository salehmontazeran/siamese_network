import os
from glob import glob
from typing import Dict
from typing import Union, Any, Sequence, Callable, Tuple, List

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T


class Compose(T.Compose):
    """Custom Compose which processes a list of inputs"""

    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = transforms

    def __call__(self, x: Union[Any, Sequence]):
        if isinstance(x, Sequence):
            for t in self.transforms:
                x = [t(i) for i in x]
        else:
            for t in self.transforms:
                x = t(x)
        return x


class ToTensor(object):
    """Custom ToTensor op which doesn't perform min-max normalization"""

    def __init__(self, permute_dims: bool = True):
        self.permute_dims = permute_dims

    def __call__(self, x: np.ndarray) -> torch.Tensor:

        if x.dtype == "uint16":
            x = x.astype("int32")

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        if x.ndim == 2:
            if self.permute_dims:
                x = x[:, :, None]
            else:
                x = x[None, :, :]

        # Convert HWC->CHW
        if self.permute_dims:
            if x.ndim == 4:
                x = x.permute((0, 3, 1, 2)).contiguous()
            else:
                x = x.permute((2, 0, 1)).contiguous()

        return


class LEVIRCDPlus(torch.utils.data.Dataset):
    """LEVIR-CD+ dataset from 'S2Looking: A Satellite Side-Looking
    Dataset for Building Change Detection', Shen at al. (2021)
    https://arxiv.org/abs/2107.09244
    'LEVIR-CD+ contains more than 985 VHR (0.5m/pixel) bitemporal Google
    Earth images with dimensions of 1024x1024 pixels. These bitemporal images
    are from 20 different regions located in several cities in the state of
    Texas in the USA. The capture times of the image data vary from 2002 to
    2020. Images of different regions were taken at different times. The
    bitemporal images have a time span of 5 years.'
    """

    splits = ["train", "test"]

    def __init__(
        self,
        root: str = "./data",
        split: str = "train",
        transform: Compose = Compose([ToTensor()]),
    ):
        assert split in self.splits
        self.root = root
        self.transform = transform
        self.files = self.load_files(root, split)

    @staticmethod
    def load_files(root: str, split: str):
        files = []
        images = glob(os.path.join(root, split, "A", "*.png"))
        images = sorted([os.path.basename(image) for image in images])
        for image in images:
            image1 = os.path.join(root, split, "A", image)
            image2 = os.path.join(root, split, "B", image)
            mask = os.path.join(root, split, "label", image)
            files.append(dict(image1=image1, image2=image2, mask=mask))
        return files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        """Returns a dict containing x, mask
        x: (2, 13, h, w)
        mask: (1, h, w)
        """
        files = self.files[idx]
        mask = np.array(Image.open(files["mask"]))
        mask = np.clip(mask, 0, 1)
        image1 = np.array(Image.open(files["image1"]))
        image2 = np.array(Image.open(files["image2"]))
        image1, image2, mask = self.transform([image1, image2, mask])
        x = torch.stack([image1, image2], dim=0)
        return dict(x=x, mask=mask)
