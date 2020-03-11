from torchvision import transforms, utils, datasets
from torch.utils.data import DataLoader, Dataset, sampler, SubsetRandomSampler, TensorDataset
import torch, h5py
from PIL import Image
import numpy as np

class Data(Dataset):

    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1, 2, 0))
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)

class DataGen:

    def __init__(self, config, logger):
        self.data = dict()
        self.config = config
        self.logger = logger

    def load_data(self, path):
        # loading data
        hf = h5py.File(path, 'r')
        # train, test data with labels being converted to numpy array from HDF5 format
        self.data["x_train"] = np.array(hf.get("X_train"), dtype=np.float32)
        self.data["y_train"] = np.array(hf.get("y_train"), dtype=np.int64)
        self.data["x_test"] = np.array(hf.get("X_test"), dtype=np.float32)
        self.data["y_test"] = np.array(hf.get("y_test"), dtype=np.int64)
        print("Training data: ", self.data["x_train"].shape)
        print("Training labels: ", self.data["y_train"].shape)
        print("Testing data: ", self.data["x_test"].shape)
        print("Testing labels: ", self.data["y_test"].shape)

    def split_data(self):
        valid_size = self.config["DATALOADER"]["VALIDATION_SPLIT"]  # % of data to be used for validation
        num_train = len(self.data["x_train"])  # get number of training samples
        indices = list(range(num_train))  # get indices of training data
        np.random.shuffle(indices)  # shuffle data randomly
        split = int(np.floor(valid_size * num_train))  # split threshold
        train_idx, valid_idx = indices[split:], indices[:split]  # split data
        X_train_1 = self.data["x_train"][train_idx, :, :, :]
        y_train_1 = self.data["y_train"][train_idx]
        X_valid_1 = self.data["x_train"][valid_idx, :, :, :]
        y_valid_1 = self.data["y_train"][valid_idx]

        # convert data to lists
        self.data["x_train"] = list(X_train_1.transpose(0, 3, 1, 2))
        self.data["y_train"] = list(y_train_1)
        self.data["x_valid"] = list(X_valid_1.transpose(0, 3, 1, 2))
        self.data["y_valid"] = list(y_valid_1)
        self.data["x_test"] = list(self.data["x_test"].transpose(0, 3, 1, 2))
        self.data["y_test"] = list(self.data["y_test"])


    def configure_dataloaders(self):

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.data["train_dataset"] = Data(self.data["x_train"], self.data["y_train"], transform=transform_train)
        self.data["valid_dataset"] = Data(self.data["x_valid"], self.data["y_valid"], transform=transform_test)
        self.data["test_dataset"] = Data(self.data["x_test"], self.data["y_test"], transform=transform_test)
        self.data["train_dataloader"] = DataLoader(self.data["train_dataset"], batch_size=self.config["HYPERPARAMETERS"]["BATCH_SIZE"])
        self.data["valid_dataloader"] = DataLoader(self.data["valid_dataset"], batch_size=self.config["HYPERPARAMETERS"]["BATCH_SIZE"])
        self.data["test_dataloader"] = DataLoader(self.data["test_dataset"], batch_size=self.config["HYPERPARAMETERS"]["BATCH_SIZE"])









