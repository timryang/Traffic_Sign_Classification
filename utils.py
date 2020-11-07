import os
import torch
import torch.utils.data
import sys
import numpy as np
from IPython.core.debugger import set_trace
from skimage.io import imread
from skimage.feature import hog
from skimage.transform import resize
from skimage.color import rgb2gray
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None

def vectorize_and_label(directories, do_RGB=False, do_hog=True, orientations=8, ppc=8, cpb=4, block_norm='L2'):
    image_vect = []
    label_vect = []
    challenge_type = []
    challenge_level = []
    for i_dir in directories:
        for subdir, dirs, files in os.walk(i_dir):
            for filename in files:
                label_vect.append(int(filename[3:5]))
                challenge_type.append(int(filename[6:8]))
                challenge_level.append(int(filename[9:11]))
                file_dir = os.path.join(subdir, filename)
                as_gray = not do_RGB
                image = imread(file_dir,as_gray=as_gray)
                if np.max(image)>1:
                    image = image.astype(float)/255
                image = resize(image, (28,28))
                if image.std() == 0:
                    image = (image-image.mean())
                else:
                    image = (image-image.mean())/image.std()
                if do_hog:
                    hog_feature = hog(image, orientations=orientations, pixels_per_cell=(ppc,ppc), cells_per_block=(cpb,cpb), block_norm=block_norm, multichannel=do_RGB)
                    image_vect.append(hog_feature)
                else:
                    image_vect.append(image.ravel())
                    
    return np.array(image_vect), np.array(label_vect), np.array(challenge_type), np.array(challenge_level)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def standardization(tensor):
    if not _is_tensor_image(tensor):
        raise TypeError('Tensor is not a torch image')

    for t in tensor:
        t.sub_(t.mean()).div_(t.std())
        
    return tensor


def l2normalize(tensor):
    if not _is_tensor_image(tensor):
        raise TypeError('Tensor is not a torch image')

    tensor = tensor.mul(255)
    norm_tensor = tensor/torch.norm(tensor)
    return norm_tensor


def make_dataset (traindir):
    img = []
    for i_dir in traindir:
        for subdir, dirs, files in os.walk(i_dir):
            for fname in files:
                target = int(fname[3:5]) - 1
                path = os.path.join(subdir, fname)
                item = (path, target)
                img.append(item)
    return img


class CURETSRDataset (torch.utils.data.Dataset):
    def __init__(self, traindir, transform=None, target_transform =None,
                loader = pil_loader):
        self.traindir = traindir
        self.imgs = make_dataset (traindir)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)