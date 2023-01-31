import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class ChangeDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.a_folder = os.path.join(root_dir, "A")
        self.b_folder = os.path.join(root_dir, "B")
        self.label_folder = os.path.join(root_dir, "label")
        self.image_names = os.listdir(self.a_folder)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Load the pre-image
        a_img_path = os.path.join(self.a_folder, self.image_names[idx])
        a_img = Image.open(a_img_path)

        # Load the post-image
        b_img_path = os.path.join(self.b_folder, self.image_names[idx])
        b_img = Image.open(b_img_path)

        # Load the label image
        label_path = os.path.join(self.label_folder, self.image_names[idx])
        label = Image.open(label_path)

        if self.transform:
            a_img = self.transform(a_img)
            b_img = self.transform(b_img)

            label = np.clip(label, 0, 1)
            label_transforms = transforms.Compose([transforms.ToTensor()])
            label = label_transforms(label)

        x = torch.stack([a_img, b_img], dim=0)
        sample = dict(x=x, mask=label)

        return sample


# Example usage
if __name__ == "__main__":
    root_dir = "./data/val"
    data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    dataset = ChangeDetectionDataset(root_dir, data_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    print(dataset[0])
