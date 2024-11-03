"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from pytorchexample.task import Net, get_weights, load_data, set_weights, load_c_local, set_c_local
from torch.optim import lr_scheduler

import numpy as np
import copy

from flwr.common.parameter import bytes_to_ndarray

class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, local_epochs, learning_rate, partition_id, c_local):
        self.net = Net()
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.partition_id = partition_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.99)
        self.c_local = c_local


    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_weights(self.net, parameters)
        results = train(self,
            net=self.net,
            trainloader=self.trainloader,
            valloader=self.valloader,
            epochs=self.local_epochs,
            learning_rate=self.lr,
            device=self.device,
            config=config,
            c_local=self.c_local,
            parameters=parameters
        )
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

def train(self, net, trainloader, valloader, epochs, learning_rate, device, config, c_local, parameters):
    """Train the model on the training set."""
    c_global_bytes = config['c_global']
    # Deserialize c_global list from bytes to float32
    c_global = np.frombuffer(c_global_bytes, dtype=np.float64)

    # Cache trainable global parameters
    global_weight = [param.detach().clone() for param in self.net.parameters()]

    if c_local is None:
        print("No cache for c_local")
        c_local = [torch.zeros_like(param) for param in self.net.parameters()]

    net.to(device)  # move model to GPU if available
    net.train()
    # Train the global model with local data
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            self.optimizer.zero_grad()
            self.criterion(net(images.to(device)), labels.to(device)).backward()
            self.optimizer.step()

        # Adds Scaffold computation of c_diff in parameters
        for param, c_l, c_g in zip(self.net.parameters(), c_local, c_global):
            param.grad.data -= learning_rate * (c_g - c_l)

    # Update local control variate
    # Declare Scaffold variables
    y_delta = []
    c_plus = []
    c_delta = []

    # Compute y_delta (difference of model before and after training)
    for param_l, param_g in zip(self.net.parameters(), global_weight):
        y_delta.append(param_l - param_g)

    # Erase net params with y_delta params for weight averaging in FedAvg
    for param, new_w in zip(self.net.parameters(), y_delta):
        param.data = new_w.clone().detach() 

    # Compute c_plus : Options 2
    for c_l, c_g, param_l, param_g in zip(c_local, c_global, self.net.parameters(), global_weight):
        c_plus.append(c_l - c_g + (param_g - param_l)/(epochs * learning_rate))

    # Compute c_plus : Options 1
    # c_plus_net = Net()
    # set_weights(c_plus_net, parameters)

    # criterion = torch.nn.CrossEntropyLoss().to(self.device)
    # optimizer = torch.optim.SGD(c_plus_net.parameters(), lr=learning_rate, momentum=0.9)

    # c_plus_net.to(device)  # move model to GPU if available
    # c_plus_net.train()
    # for batch in trainloader:
    #     images = batch["img"]
    #     labels = batch["label"]
    #     optimizer.zero_grad()
    #     criterion(c_plus_net(images.to(device)), labels.to(device)).backward()
    #     optimizer.step()

    # for x_param in c_plus_net.parameters():
    #     c_plus.append(x_param)

    # Compute c_delta
    for c_p, c_l in zip(c_plus, c_local):
        c_delta.append(c_p - c_l)

    set_c_local(self.partition_id, c_plus)

    # Create a bytes stream for c_delta
    # Flatten list to be compatible with numpy
    c_delta_list = []
    for param in c_delta:
        c_delta_list += param.flatten().tolist()
        
    c_delta_numpy = np.array(c_delta_list, dtype=np.float64)
    # Serialize to bytes
    c_delta_bytes = c_delta_numpy.tobytes()

    self.scheduler.step()
    val_loss, val_acc = test(net, valloader, device)

    results = { 
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "c_delta": c_delta_bytes,
    }
    return results


def test(net, testloader, device):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy

def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Read run_config to fetch hyperparameters relevant to this run
    batch_size = context.run_config["batch-size"]
    trainloader, valloader = load_data(partition_id, num_partitions, batch_size)
    c_local = load_c_local(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    # Return Client instance
    return FlowerClient(trainloader=trainloader, valloader=valloader, local_epochs=local_epochs, learning_rate=learning_rate, partition_id=partition_id, c_local=c_local).to_client()

# Flower ClientApp
app = ClientApp(client_fn)

