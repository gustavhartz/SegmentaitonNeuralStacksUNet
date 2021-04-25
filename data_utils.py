import PIL.Image as Image
import os
import glob
import torch
from torch.utils.data import DataLoader
import cv2
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import numpy as np
from AugmentationFuncs.funcs import elastic, add_noise, shear, rotate, zoom


class EM_DATA(torch.utils.data.Dataset):
    """
    Data class to load in and tranform the data given on campusnet
    """

    def __init__(self, train, size, _transform, data_path="EM_ISBI_Challenge", validation=False):
        """
        This assumes the same folder structure as the campusnet data i.e
        EM_ISBI_Challenge
            - test_images
            - train_images
            - train_labels
            - validation_images
            - validation_masks
        :param train: Boolean to select if the test or train data should be loaded
        :param size: Image size to return
        :param data_path: Path to the EM_ISBI_Challenge folder
        :param _transform: A torchvision transform object containing only non random transformations!
        """

        self._size = size
        self._transform = _transform
        self._train = train
        self._validation = validation
        self._root_dir = data_path
        self.data_path = os.path.join(self._root_dir, 'train_images' if self._train else 'test_images')
        self.mask_paths = sorted(glob.glob(os.path.join(self._root_dir, "train_labels", "*.png"))) if self._train else None
        if self._validation:
            self.data_path = os.path.join(self._root_dir, 'validation_images')
            self.mask_paths = sorted(glob.glob(os.path.join(self._root_dir, "validation_labels", "*.png")))
        self.image_paths = sorted(glob.glob(os.path.join(self.data_path, "*.png")))
        self._test_mask = torch.zeros(self._size)

    def __len__(self):
        """
        Returns the total number of samples
        :return: # of images
        """
        return len(self.image_paths)

    def transform(self, image, mask):
        """
        The idea of this function is that you can add transformations. This can be done in multiple ways
        but here it is important that the exact same transformation is done to the training label and image
        :param image:
        :param mask:
        :return: Transformed images and masks
        """

        """ Example on how you could at a transformation you define yourself
        # Random horizontal flipping
        if self._train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
        """
        if self._train and not self._validation:
            if random.random() > 0.1:
                # Choose random augmentation
                func = np.random.choice([elastic, add_noise, shear, rotate, zoom])
                image, mask = func(image, mask)

        # Apply the transformations defined in the input transform parameter. Remember to end it with 'to_tensor'
        image = self._transform(image)
        mask = self._transform(mask) if (self._train or self._validation) else self._test_mask
        return image, mask

    def _load_mask(self, idx):
        """
        Helper function to load in masks
        :param idx: index to return
        :return: mask with the given index
        """
        mask = Image.open(self.mask_paths[idx])
        mask = np.asarray(mask).copy()
        mask[mask>0] = 255
        return mask

    def __getitem__(self, idx):
        """
        This is the entire idea of this function and makes it a python generator function which is what Pytorch
         assume for the dataloader
        :param idx: Image index to return
        :return: Return a X, Y pair of image and mask
        """
        image_path = self.image_paths[idx]

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = self._load_mask(idx) if self._train or self._validation else self._test_mask
        X, y = self.transform(image, mask)
        return X, y