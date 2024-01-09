from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm 

class rawData():
    def __init__(self, root='data/', mode='dummy-binary', img_size=(256, 256)):
        """
        Initialize the rawData class.

        Args:
            root (str): Root directory containing image data.
            mode (str): Data mode, e.g., 'train-binary'.
            img_size (tuple): Size of the images (height, width).
        """
        self.mode = mode 
        self.root = os.path.join(root, mode.split('-')[-1], mode.split('-')[0])
        self.size = img_size
        self.shape = (img_size[0], img_size[1])

        self.img_arr, self.lab_arr, self.tag_arr = self.read(root=self.root, size=self.size, shuffle=False)

    def read(self, root=None, size=(128, 128), shuffle=True):
        """
        Read images from the specified root directory.

        Args:
            root (str): Root directory containing image data.
            size (tuple): Size of the images (height, width).
            shuffle (bool): Whether to shuffle the images.

        Returns:
            np.array: Numpy array containing image data.
        """
        if root is None or not os.path.exists(root):
            raise ValueError("Please provide a valid root directory.")

        tags = []
        imgs = []
        labs = []

        for class_folder in os.listdir(root):
            class_path = os.path.join(root, class_folder)

            if os.path.isdir(class_path):
                label = class_folder

                for file_name in tqdm(set(os.listdir(class_path)), desc=str(label)):
                    
                    if self.mode.split('-')[-1] == 'multiclass':
                        keys = np.array(['bcc', 'mel', 'scc'])    
                        mapping = {'bcc': 0,
                                   'mel': 1, 
                                   'scc': 2}
                            
                        for k in keys:
                            if k in file_name:
                                label = k
                                break
                    else:
                        mapping = {'nevus': 0,
                                   'others': 1}
                    
                    if file_name.endswith('.jpg'):
                        image_path = os.path.join(class_path, file_name)

                        # Open and resize image
                        image = cv.imread(image_path)
                        image = cv.resize(image, size)

                        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                        fname, _ = file_name.split('.')

                        imgs.append(image)
                        tags.append(fname)
                        labs.append([mapping[label]])

        img_array = np.array(imgs, dtype=np.uint8)

        lab_array = np.array(labs, dtype=np.uint8)
        tag_array = np.array(tags)
        # if shuffle:
        #     np.random.shuffle(array)

        return img_array, lab_array, tag_array
    
    def preprocess(self, should_save=False):
        """
        Preprocess images and return numpy arrays.

        Args:
            should_save (bool): Whether to save the preprocessed images.

        Returns:
            np.array: Preprocessed image data.
        """
        # Implementation of preprocessing if needed
        return self.images
    
    def plot(self, random=True, size=5):
        """
        Plot random or specified number of images with labels and histograms.

        Args:
            random (bool): Whether to plot random images.
            size (int): Number of images to plot.
        """
        if random:
            indices = np.random.randint(len(self.ordered_images), size=size)
        else:
            indices = np.arange(min(size, len(self.ordered_images)))

        num_rows = size // 5  # Assuming you want 5 columns per row

        # Plot images
        fig, axes = plt.subplots(num_rows, 5, figsize=(15, 3*num_rows))
        axes = axes.ravel()

        for i, index in enumerate(indices):
            image, label, file_name = self.ordered_images[index]
            axes[i].imshow(image)
            axes[i].set_title(f'Label: {label}\nFile Name: {file_name}')
            axes[i].axis('off')

        # Plot histograms
        fig, axes = plt.subplots(num_rows, 5, figsize=(15, 3*num_rows))
        axes = axes.ravel()

        for i, index in enumerate(indices):
            image, label, file_name = self.ordered_images[index]

            # Let's have a look at the H channel.
            ch1, ch2, ch3 = cv.split(image)

            # Calculate the histograms
            hist_ch1 = cv.calcHist([ch1], [0], None, [256], [0, 256])
            hist_ch2 = cv.calcHist([ch2], [0], None, [256], [0, 256])
            hist_ch3 = cv.calcHist([ch3], [0], None, [256], [0, 256])

            # Plot the histograms
            axes[i].plot(hist_ch1, color='pink')
            axes[i].plot(hist_ch2, color='orange')
            axes[i].plot(hist_ch3, color='brown')
            axes[i].set_title(f'Label: {label}\nFile Name: {file_name}')
            axes[i].axis('off')
            
        # Plot channels
        fig, ax = plt.subplots(3, 5, figsize=(15, 3*3))

        for i, index in enumerate(indices):
            image, label, file_name = self.ordered_images[index]

            ch1, ch2, ch3 = cv.split(image)
            
            ax[0,i].imshow(ch1)
            ax[0,i].set_title(f'Label: {label}\nFile Name: {file_name} ch1')
            ax[0,i].axis('off')
            ax[1,i].imshow(ch2)
            ax[1,i].set_title(f'Label: {label}\nFile Name: {file_name} ch2')
            ax[1,i].axis('off')
            ax[2,i].imshow(ch3)
            ax[2,i].set_title(f'Label: {label}\nFile Name: {file_name} ch3')
            ax[2,i].axis('off')
            
        plt.tight_layout()
        plt.show()


class LesionDataset(Dataset):
    def __init__(self, root='data/', mode='dummy-binary', img_size=(256, 256), transform=None, target_transform=None):
        """
        Initialize the LesionDataset class.

        Args:
            root (str): Root directory containing image data.
            mode (str): Data mode, e.g., 'train-binary'.
            img_size (tuple): Size of the images (height, width).
            transform: PyTorch transformations applied to the images.
            target_transform: PyTorch transformations applied to the labels.
        """
        self.raw_data = rawData(root=root, mode=mode, img_size=img_size)
        self.transform = transform
        self.target_transform = target_transform
        self.images = self.raw_data.img_arr
        self.labels = self.raw_data.lab_arr
        self.fnames = self.raw_data.tag_arr

    def __len__(self):
        return len(self.labels)
    
    def __str__(self):
        return str(len(self.raw_data.lab_arr))

    def __getitem__(self, idx):
        image = torch.from_numpy(self.raw_data.img_arr[idx])

        if self.transform:
            image = self.transform(image)

        label = torch.from_numpy(self.raw_data.lab_arr[idx])
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
    
    def concat(self, imgs_label_0, imgs_label_1, gen_imgs_label_0, gen_imgs_label_1, transform=None, target_transform=None):
        """
        Concatenate two sets of images and labels.

        Args:
            imgs_label_0: Images labeled as 0.
            imgs_label_1: Images labeled as 1.
            gen_imgs_label_0: Generated images labeled as 0.
            gen_imgs_label_1: Generated images labeled as 1.
            transform: PyTorch transformations applied to the images.
            target_transform: PyTorch transformations applied to the labels.
        """
        self.img_labels = pd.DataFrame(
            np.concatenate((np.zeros(len(imgs_label_0), dtype=int),
                            np.ones(len(imgs_label_1), dtype=int)))
        )
        self.images = np.concatenate((gen_imgs_label_0, gen_imgs_label_1))
        self.transform = transform
        self.target_transform = target_transform


if __name__ == '__main__':
    dataset = LesionDataset()
    print(dataset)

    print(dataset[0][0].shape)

    # Define basic transformations
    basic_transform = transforms.Compose([
        transforms.ToTensor(),
        # Add more transformations as needed
    ])

    # Create instances of the datasets
    # train_dataset = LesionDataset(mode='train-binary', transform=basic_transform)
    test_dataset = LesionDataset(mode='val-binary', transform=basic_transform)

    # Create DataLoader instances for batching
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)