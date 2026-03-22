from typing import Callable

import torch
import torch.nn as nn
from torch.functional import F

from model_types import EmbedInfo, Embedder, EmbedType, ForwardArgs, Prediction


ARGS_TO_ACTIVATION = {
    "linear":   lambda x: x,
    "tanh":     F.tanh,
    "cube":     lambda x: x,  # NOTE: dy.cube has no direct F equivalent; using identity
    "logistic": torch.sigmoid,
    "relu":     F.relu,
    "elu":      F.elu,
    "selu":     F.selu,
    "softsign": F.softsign,
    "swish":    F.silu,       # equivalent to x * sigmoid(x), same as dy.cmult(h, dy.logistic(h))
}


class GloveEmbedder(nn.Module, Embedder):
    """Static GloVe word embeddings — PyTorch equivalent of DyNet lookup parameters.

    Inherits from nn.Module so that the embedding weights are tracked by PyTorch
    and updated by the optimizer, matching DyNet's trainable lookup parameters.
    """

    def __init__(self, static_embed_file: str):
        nn.Module.__init__(self)
        self.id_to_token = []
        self.token_to_id = {}
        pretrained = []

        for line in open(static_embed_file):
            parts = line.strip().split()
            word = parts[0].lower()
            vector = [float(v) for v in parts[1:]]
            self.token_to_id[word] = len(self.id_to_token)
            self.id_to_token.append(word)
            pretrained.append(vector)

        self.embed_info = EmbedInfo(
            name="glove", embed_type=EmbedType.WORD, dim=len(pretrained[0])
        )
        # freeze=False to match DyNet trainable lookup parameters
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(pretrained, dtype=torch.float64), freeze=False
        )

    def embed(self, words: list) -> torch.Tensor:
        return self.embedding(torch.tensor(words))

    def get_embed_info(self) -> EmbedInfo:
        return self.embed_info


class ForwardModel(nn.Module):
    """
    PyTorch feedforward model — equivalent to DyNetModel.

    Architecture:
      input (77 features [+ 4 * embed_dim if GloVe])
        -> hidden_layers_num fully-connected layers of size hidden_layer_size
        -> linear output (scalar per option)
        -> log_softmax over all options
    """

    def __init__(
        self,
        features_num: int,
        hidden_layers_num: int,
        hidden_layer_size: int,
        dropout: float,
        embedder: Embedder,
        activation_fn: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()

        self.input_size = features_num
        self.embedder = embedder
        self.dropout = dropout

        if embedder is not None:
            self.input_size += 4 * embedder.get_embed_info().dim

        layers = [nn.Linear(self.input_size, hidden_layer_size, bias=True, dtype=torch.float64)]
        for _ in range(1, hidden_layers_num):
            layers.append(
                nn.Linear(hidden_layer_size, hidden_layer_size, bias=True, dtype=torch.float64)
            )
        self.hidden = nn.ModuleList(layers)
        self.activation_fn = activation_fn
        self.final_sum = nn.Linear(hidden_layer_size, 1, bias=True, dtype=torch.float64)

    def forward(self, args: ForwardArgs) -> tuple:
        if len(args.options) == 1:
            return None, 0

        option_features = [torch.tensor(option.features) for option in args.options]
        inputs = torch.stack(option_features)  # shape: (num_options, features_num)

        if self.embedder is not None:
            qvecs = self.embedder.embed(args.query)
            qvec_max = torch.max(qvecs, dim=0).values
            qvec_mean = torch.mean(qvecs, dim=0)

            all_ovecs = [self.embedder.embed(option.text_rep) for option in args.options]
            ovec_maxes = torch.stack([torch.max(v, dim=0).values for v in all_ovecs])
            ovec_means = torch.stack([torch.mean(v, dim=0) for v in all_ovecs])

            inputs = torch.cat(
                [
                    inputs,
                    qvec_max.repeat(len(args.options), 1),
                    qvec_mean.repeat(len(args.options), 1),
                    ovec_maxes,
                    ovec_means,
                ],
                dim=1,
            )

        if self.dropout > 0:
            inputs = F.dropout(inputs, p=self.dropout, training=args.is_train)

        h = inputs
        for layer in self.hidden:
            h = layer(h)
            h = self.activation_fn(h)

        final = torch.sum(h, dim=1).unsqueeze(-1)   # (num_options, 1)
        nll = -F.log_softmax(final, dim=0)

        dense_gold = torch.tensor(
            [1.0 / len(args.gold) if i in args.gold else 0.0 for i in range(len(args.options))],
            dtype=torch.float64,
        )
        loss = torch.dot(dense_gold, nll.squeeze(dim=-1))

        predicted_link = torch.argmax(final, dim=0)
        pred_prob = torch.exp(-nll)[predicted_link].squeeze()
        gold_prob = torch.exp(-nll)[args.gold].squeeze()

        prediction = Prediction(
            pred_idx=predicted_link,
            gold_idx=torch.tensor(args.gold),
            pred_prob=pred_prob,
            gold_prob=gold_prob,
        )
        return prediction, loss

    def get_ids(self, words: list) -> list:
        """Convert token strings to GloVe word IDs. Returns raw words if no embedder."""
        if self.embedder is None:
            return words
        backup = self.embedder.token_to_id.get("<unka>", 0)
        return [self.embedder.token_to_id.get(word, backup) for word in words]
