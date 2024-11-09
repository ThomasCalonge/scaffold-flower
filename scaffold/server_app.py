from typing import List, Tuple
import torch
import numpy as np
import os
import shutil

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedProx, FedAdagrad
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.server.client_proxy import ClientProxy

from scaffold.task import Net, get_weights, set_weights

class Scaffold(FedAvg):
    def __init__(self, global_model, global_learning_rate, num_clients, **kwargs):
        super().__init__(**kwargs)
        self.c_global = [torch.zeros_like(param) for param in global_model.parameters()]
        self.current_weights = parameters_to_ndarrays(self.initial_parameters)
        self.num_clients = num_clients
        self.global_learning_rate = global_learning_rate

    def configure_fit(
        self, server_round, parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # Get the standard client/config pairs from the FedAvg super-class
        client_config_pairs = super().configure_fit(
            server_round, parameters, client_manager
        )

        # Serialize c_global to be compatible with config FitIns return values
        c_global = []
        for param in self.c_global:
            c_global += param.flatten().tolist()  # Flatten all params
            
        global_c_numpy = np.array(c_global, dtype=np.float64)
        global_c_bytes = global_c_numpy.tobytes()

        # Return client/config pairs with the c_global serialized control variate
        return [
            (
                client,
                FitIns(
                    fit_ins.parameters,
                    {**fit_ins.config, "c_global": global_c_bytes},
                ),
            )
            for client, fit_ins in client_config_pairs
        ]
        
    def aggregate_fit(self, server_round, results, failures):
        # Use FedAvg aggregate_fit function to average y_delta weights
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        if parameters_aggregated is None:
            return None, {}

        fedavg_weights_aggregate = parameters_to_ndarrays(parameters_aggregated)

        # Aggregating the updates of y_delta to current weight cf. Scaffold equation (n°5)
        for current_weight, fed_weight in zip(self.current_weights, fedavg_weights_aggregate):
            current_weight += fed_weight * self.global_learning_rate

        # Initalize c_delta_sum for the weight average
        c_delta_sum = [np.zeros_like(c_global) for c_global in self.c_global]

        for _, fit_res in results:
            # Getting serialized buffer from fit metrics 
            c_delta = np.frombuffer(fit_res.metrics["c_delta"], dtype=np.float64)
            # Sum all c_delta in a single weight vector
            for i in range(len(c_delta_sum)):
                c_delta_sum[i] += np.array(c_delta[i], dtype=np.float64)

        for i in range(len(self.c_global)):
            # Aggregating the updates of c_global cf. Scaffold equation (n°5)
            c_delta_avg = c_delta_sum[i] / self.num_clients
            self.c_global[i] += torch.tensor(c_delta_avg)

        return ndarrays_to_parameters(self.current_weights), metrics_aggregated

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    print("accuracy : " + str(sum(accuracies) / sum(examples)))
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    # Construct c_local folder for all clients and erase cache if already present
    path = "c_local_folder/"
    if not os.path.exists(path):
        os.makedirs(path)
    else: 
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error while deleting {file_path}: {e}")

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Initialize model parameters
    global_model = Net()
    ndarrays = get_weights(global_model)
    parameters = ndarrays_to_parameters(ndarrays)

    # Define the strategy
    strategy = Scaffold(
        global_model=global_model,
        num_clients=context.run_config["num-clients"], # Caution, this config should be always equal to num-supernodes
        global_learning_rate=context.run_config["global-learning-rate"],
        fraction_fit=context.run_config["fraction-fit"],
        fraction_evaluate=context.run_config["fraction-evaluate"],
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)