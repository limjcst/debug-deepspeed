import argparse
from functools import partial
from typing import List
from typing import Tuple

import deepspeed
import torch
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset


class NaiveModule(torch.nn.Module):

    def __init__(self, n_in: int, d_model: int = 1024, n_layer: int = 2):
        super().__init__()

        self._n_in = n_in
        self._d_model = d_model
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self._n_in, self._d_model),
            *(
                torch.nn.TransformerEncoderLayer(
                    d_model=self._d_model, nhead=8, batch_first=True, norm_first=False
                )
                for _ in range(n_layer)
            ),
            torch.nn.Linear(self._d_model, self._n_in),
        )

    def batch_fn(
        self, batch: torch.Tensor, dtype: torch.dtype = torch.float32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = batch.to(dtype=dtype, copy=True)
        return batch, batch

    def loss_fn(
        self, output: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.mse_loss(output, label)

    def to_layers(self) -> List[torch.nn.Module]:
        return self.layers

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.layers(data)


class MyDataset(IterableDataset):

    def __init__(self, n_in: int, batch_size: int = 1):
        super().__init__()

        self._data = [
            torch.rand((batch_size, 8, n_in)), torch.rand((batch_size, 12, n_in))
        ]

    def __getitem__(self, index):
        return NotImplemented

    def __iter__(self):
        for item in self._data:
            yield item


def _train(args: argparse.Namespace):
    _n_in = 32
    model = NaiveModule(n_in=_n_in, n_layer=2)

    deepspeed.init_distributed()
    torch.cuda.set_device(args.local_rank)
    net = deepspeed.PipelineModule(
        model.to_layers(), num_stages=args.num_stages, loss_fn=model.loss_fn,
    )
    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=net.parameters(),
    )
    engine: deepspeed.PipelineEngine = engine
    engine.set_batch_fn(partial(model.batch_fn, dtype=engine.communication_data_type))

    dataset = MyDataset(n_in=_n_in, batch_size=engine.train_micro_batch_size_per_gpu())
    dataloader = DataLoader(dataset, batch_size=None, num_workers=1)
    training_iter = iter(deepspeed.utils.RepeatingLoader(dataloader))
    for step in range(args.steps):
        loss = engine.train_batch(training_iter)


def _main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )
    parser.add_argument("--num_stages", type=int)
    parser.add_argument("--steps", type=int)
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    _train(args)


if __name__ == "__main__":
    _main()
