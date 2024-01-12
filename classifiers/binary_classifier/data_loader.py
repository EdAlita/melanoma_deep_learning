from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '_mask.jpg'))
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Assuming mask is grayscale

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
