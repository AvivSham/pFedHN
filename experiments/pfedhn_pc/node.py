from experiments.dataset import gen_random_loaders


class BaseNodesForLocal:
    def __init__(
        self,
        data_name,
        data_path,
        n_nodes,
        base_layer,
        layer_config,
        base_optimizer,
        optimizer_config,
        device,
        batch_size=128,
        classes_per_node=2
    ):

        self.data_name = data_name
        self.data_path = data_path
        self.n_nodes = n_nodes
        self.classes_per_node = classes_per_node
        self.batch_size = batch_size

        self.local_layers = [
            base_layer(**layer_config).to(device) for _ in range(self.n_nodes)
        ]
        self.local_optimizers = [
            base_optimizer(self.local_layers[i].parameters(), **optimizer_config) for i in range(self.n_nodes)
        ]

        self.train_loaders, self.val_loaders, self.test_loaders = gen_random_loaders(
            self.data_name,
            self.data_path,
            self.n_nodes,
            self.batch_size,
            self.classes_per_node
        )

    def __len__(self):
        return self.n_nodes
