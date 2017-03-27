import torch
import torch.utils.data as data
import numpy as np

from PIL import Image

class GridworldData(data.Dataset):
    def __init__(self, file, imsize, train=True, transform=None, target_transform=None):
        assert file.endswith('.npz') # Must be .npz format
        self.file = file
        self.imsize = imsize
        self.transform = transform
        self.target_transform = target_transform
        self.train = train # training set or test set
        
        self.images, self.S1, self.S2, self.labels = self._process(file, self.train)

    def __getitem__(self, index):
        img = self.images[index]
        s1 = self.S1[index]
        s2 = self.S2[index]
        label = self.labels[index]
        
        # [Convert to PIL Image format, consistent with other datasets]
        # Not for this dataset, because of stacked binary images{0, 1}, obstacles and goal
        # Uncomment it for image input
        #img = Image.fromarray(img, mode='L')
        
        if self.transform is not None:
            img = self.transform(img)
        else: # Internal default transform: Just to Tensor
            img = torch.from_numpy(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
            
        # Ensure labels in naive float type
        # DataLoader has bug with np.int/float type in default_collate()
        return img, int(s1), int(s2), int(label)
        
    def __len__(self):
        return self.images.shape[0]
        
    def _process(self, file, train):
        """Data format: A list, [train data, test data]
        Each data sample: label, S1, S2, Images, in this order.
        """
        if train:
            idx = 0
        else:
            idx = 1
            
        with np.load(file) as f:
            # We do not convert it to Tensor for convenience
            # Since with slicing/transpose, Tensor becomes non-contiguous
            # And view() cannot handle non-contiguous tensor
            # Then use transforms.ToTensor() when define DataLoader
            data = f.items()[0][1][idx]
        
        labels = data[:, 0]
        S1 = data[:, 1]
        S2 = data[:, 2]
        images = data[:, 3:].reshape([-1, self.imsize, self.imsize, 2])
        images = images.transpose([0, 3, 1, 2]) # Convert from NHWC to NCHW, PyTorch format
        
        # Data type conversion
        S1 = S1.astype(int) # (S1, S2) location are integers
        S2 = S2.astype(int)
        labels = labels.astype(int) # labels are integers
        
        return images, S1, S2, labels
