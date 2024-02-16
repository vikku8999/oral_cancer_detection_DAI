import torch

from PIL import Image
from pandas import DataFrame
from typing import Tuple, Dict, List
from torch.utils.data import Dataset, DataLoader

"""
Contains functions for training and testing a PyTorch model.
"""
from torchvision import datasets, models, transforms


# Write a custom dataset class (inherits from torch.utils.data.Dataset)
class ImageFolderCustom(Dataset):

    # 1. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self,
                 targ_dir: str,
                 path_df: DataFrame,
                 transform=None) -> None:

        # Get all image paths, classes
        self.img_df = path_df

        # Set all images to proper path
        self.img_df['path'] = self.check_path(targ_dir)

        self.paths = list(self.img_df['path'])

        # Setup transforms
        self.transform = transform

        self.classes, self.class_to_idx = self.find_classes()

    # 2. check if its already in proper format
    def check_path(self,
                   targ_dir: str) -> DataFrame:
        if str(targ_dir) in self.img_df.iloc[0,0]:
            return self.img_df['path'].astype('string')
        else:
            return str(targ_dir)+ '/' +  self.img_df['path'].astype('string')

    # 3. Make function to load images
    def load_image(self,
                   index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.img_df.iloc[index, 0]
        return Image.open(image_path)

    # 4. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return self.img_df.shape[0]

    # 5. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self,
                    index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name  = self.img_df.iloc[index, 1] # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return data, label (X, y)


    def find_classes(self) -> Tuple[List[str], Dict[str, int]]:

        col = self.img_df.columns
        # 1. Get the class names by scanning the target directory
        classes = sorted(self.img_df[col[1]].unique())

        # 2. Raise an error if class names not found
        if not classes:
            raise FileNotFoundError(f"Couldn't find any classes.")

        # 3. Create a dictionary of index labels (computers prefer numerical rather than string labels)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    