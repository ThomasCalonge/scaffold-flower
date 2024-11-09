import numpy as np
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib.figure import Figure

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, RandomHorizontalFlip, ColorJitter

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner, PathologicalPartitioner
from flwr_datasets.visualization import plot_label_distributions

# Simple 2 layer fully connected network model to mimic Scaffold experiment
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 200)  # MNIST images are 28x28 pixels
        self.fc2 = nn.Linear(200, 26)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Simple logistic regression model to mimic Scaffold experiment
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(28 * 28, 26)  # MNIST images are 28x28 pixels

#     def forward(self, x):
#         x = x.view(-1, 28 * 28)
#         x = self.fc1(x)
#         return x
    
def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

# Cache for Federated Dataset (train and test)
fds = None  
def load_data(partition_id: int, num_partitions: int, batch_size: int):
    # Only initialize `FederatedDataset` once
    global fds

    # Use any partitioner you want for your experiment
    # Caution. If you want to access the train and test split from hugging face, you have to declare two different partitioner. Otherwise the train split will erase the test split.

    #partitioner_train = IidPartitioner(num_partitions=num_partitions)
    #partitioner_test = IidPartitioner(num_partitions=num_partitions)
    partitioner_train = PathologicalPartitioner(
        num_partitions=num_partitions,
        partition_by="label",
        num_classes_per_partition=1,
        class_assignment_mode="deterministic"
    )

    partitioner_test = PathologicalPartitioner(
        num_partitions=num_partitions,
        partition_by="label",
        num_classes_per_partition=1,
        class_assignment_mode="deterministic"
    )
    
    #dataset="uoft-cs/cifar10"
    #dataset = "ylecun/mnist"
    dataset = "tanganke/emnist_letters"

    if fds is None:
        fds = FederatedDataset(
            dataset=dataset,
            partitioners={"train": partitioner_train, "test": partitioner_test},
        )
        # Figures of train and test splits. You should always inspect them when using a new dataset.
        fig, ax, df = plot_label_distributions(
            fds.partitioners["train"],
            label_name="label",
            plot_type="bar",
            size_unit="absolute",
            partition_id_axis="x",
            legend=True,
            verbose_labels=True,
            title="Dataset per partition for train",)
        fig.savefig('dataset_train.png', format='png')
        fig, ax, df = plot_label_distributions(
            fds.partitioners["test"],
            label_name="label",
            plot_type="bar",
            size_unit="absolute",
            partition_id_axis="x",
            legend=True,
            verbose_labels=True,
            title="Dataset per partition for test",)
        fig.savefig('dataset_test.png', format='png')

    partition_train = fds.load_partition(partition_id, "train")
    partition_test = fds.load_partition(partition_id, "test")

    # Use this if you want to add noise to the data to generalize better for your model
    # train_transforms = Compose([RandomHorizontalFlip(), ColorJitter(brightness=0.2, contrast=0.2)])
    # def train_transform(batch):
    #     """Apply transforms to the train set."""
    #     batch["image"] = [train_transforms(img) for img in batch["image"]]
    #     return batch
    # partition_train = partition_train.with_transform(train_transform)

    pytorch_transforms = Compose(
        [ToTensor(), Normalize(0.1307, 0.3081)]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    partition_train = partition_train.with_transform(apply_transforms)
    partition_test = partition_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(partition_test, batch_size=batch_size)

    return trainloader, testloader

# Custom function to load c_local variable from a serialize bytes buffer inside a file
# Path should always be in "c_local_folder" and partition_id.txt
def load_c_local(partition_id: int):
    path = "c_local_folder/" + str(partition_id) +".txt"
    if os.path.exists(path):
        with open(path, 'rb') as f:
            c_delta_bytes = f.read()

        array = np.frombuffer(c_delta_bytes, dtype=np.float64)
        return array
    else:
        return None

# Custom function to serialize to bytes and save c_local variable inside a file
def set_c_local(partition_id: int, c_local):
    path = "c_local_folder/" + str(partition_id) +".txt"

    c_local_list = []
    for param in c_local:
        c_local_list += param.flatten().tolist()

    c_local_numpy = np.array(c_local_list, dtype=np.float64)
    c_local_bytes = c_local_numpy.tobytes()

    with open(path, 'wb') as f:
        f.write(c_local_bytes)

