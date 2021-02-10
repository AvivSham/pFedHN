from experiments.dataset import gen_random_loaders


class BaseNodes:
    def __init__(
            self,
            data_name,
            data_path,
            n_nodes,
            batch_size=128,
            classes_per_node=2
    ):

        self.data_name = data_name
        self.data_path = data_path
        self.n_nodes = n_nodes
        self.classes_per_node = classes_per_node

        self.batch_size = batch_size

        self.train_loaders, self.val_loaders, self.test_loaders = None, None, None
        self._init_dataloaders()

    def _init_dataloaders(self):
        self.train_loaders, self.val_loaders, self.test_loaders = gen_random_loaders(
            self.data_name,
            self.data_path,
            self.n_nodes,
            self.batch_size,
            self.classes_per_node
        )

    def __len__(self):
        return self.n_nodes
