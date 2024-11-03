"""pytorchexample: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner, PathologicalPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, RandomHorizontalFlip, ColorJitter

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""
    def __init__(self):
        super(Net, self).__init__()

        # 1st Convolutional Layer: (3 channels input, 32 output channels, kernel size 5, padding 2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3))
        self.pool = nn.MaxPool2d(kernel_size=(2,2))  # Pooling layer with 2x2 window
        # 2nd Convolutional Layer: (32 input channels, 64 output channels, kernel size 5, padding 2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3))

        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 10)  # For 10-class classification (like CIFAR-10)

    def forward(self, x):
        # Apply convolution -> ReLU -> pooling sequence
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = F.relu(self.conv3(x))  # 14x14 -> 7x7

        # Flatten the output for the fully connected layer
        x = x.view(-1, 256)  # Reshape to [batch_size, 256*7*7] for the fully connected layer
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


fds = None  # Cache FederatedDataset
fds_test = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    global fds_test

    alpha=[0.1,0.2,0.1,0.1,0.2]
    for i in range(95):
        alpha.append(1)
    partitioner = DirichletPartitioner(alpha=alpha,
                                        min_partition_size=10, 
                                        partition_by="label", 
                                        num_partitions=num_partitions,
                                        self_balancing=False)

    # partitioner = PathologicalPartitioner(
    #     num_partitions=num_partitions,
    #     partition_by="label",
    #     num_classes_per_partition=2,
    #     class_assignment_mode="deterministic"
    # )

    # partitioner = IidPartitioner(num_partitions=num_partitions)
    
    if fds is None:
        print("Creating fds...")
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )

    if fds_test is None:
        # partitioner = IidPartitioner(num_partitions=num_partitions)
        print("Creating fds_test...")
        fds_test = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"test": partitioner},
        )

    partition_train = fds.load_partition(partition_id)
    partition_test = fds_test.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    # partition_train_test = partition.train_test_split(test_size=0.15, seed=42)

    train_transforms = Compose([RandomHorizontalFlip(), ColorJitter(brightness=0.2, contrast=0.2)])

    pytorch_transforms = Compose(
        [ToTensor(), Resize((24,24)), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def train_transform(batch):
        """Apply transforms to the train set."""
        batch["img"] = [train_transforms(img) for img in batch["img"]]
        return batch

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train = partition_train.with_transform(train_transform)
    partition_train = partition_train.with_transform(apply_transforms)

    partition_test = partition_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(partition_test, batch_size=batch_size)

    return trainloader, testloader

c_local_dict = None  # Cache dict for all clients c_local
def load_c_local(partition_id: int, num_partitions: int):
    global c_local_dict

    if c_local_dict is None:
        print("Creating c_local_dict...")
        c_local_dict = {}
        for i in range(num_partitions):
            c_local_dict[i] = None

    c_local = c_local_dict[partition_id]
    if c_local is None:
        print("No c_local found for : " + str(partition_id))

    return c_local

def set_c_local(partition_id: int, c_local):
    global c_local_dict
    print("Setting " + str(partition_id) + " c_local...")
    c_local_dict[partition_id] = c_local
