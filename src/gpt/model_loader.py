"""Load GPT-2 vocabulary, configuration, and trained parameters from disk."""

import json
import os
import re
from typing import NamedTuple

import numpy as np

from utils.tensorflow_checkpoint_reader import TensorflowCheckpointReader


class LayerNormParameters(NamedTuple):
    """Learned layer-normalization tensors (checkpoint keys "g" and "b")."""

    scale: np.ndarray   # (d_model,)
    offset: np.ndarray  # (d_model,)


class LinearParameters(NamedTuple):
    """Learned affine projection tensors (checkpoint keys "w" and "b")."""

    weight: np.ndarray  # (in_features, out_features)
    bias: np.ndarray    # (out_features,)


class TransformerBlockParameters(NamedTuple):
    """Every learned tensor a single Transformer block needs."""

    attention_norm: LayerNormParameters
    qkv_projection: LinearParameters
    attention_output_projection: LinearParameters
    feed_forward_norm: LayerNormParameters
    feed_forward_input_projection: LinearParameters
    feed_forward_output_projection: LinearParameters


class ModelLoader:
    """Load the released OpenAI GPT-2 files needed for inference."""

    SUPPORTED_SIZES = ("124M", "355M", "774M", "1558M")

    def __init__(self, model_size="124M", models_dir="models"):
        if model_size not in self.SUPPORTED_SIZES:
            raise ValueError(
                f"Unsupported model size {model_size!r}. "
                f"Expected one of {self.SUPPORTED_SIZES}."
            )

        model_dir = os.path.join(models_dir, model_size)
        checkpoint_path = TensorflowCheckpointReader.latest_checkpoint(model_dir)
        if not checkpoint_path:
            raise FileNotFoundError(f"Model files not found in '{model_dir}'. Run download_model_files.bat first.")

        self.vocabulary = self._load_vocabulary(model_dir)
        self.merge_rules = self._load_merge_rules(model_dir)

        with open(os.path.join(model_dir, "hparams.json")) as file:
            hyperparameters = json.load(file)

        self.attention_head_count = hyperparameters["n_head"]
        self.context_size = hyperparameters["n_ctx"]
        parameters = self._load_parameters(
            checkpoint_path, hyperparameters["n_layer"]
        )
        self.token_embeddings = parameters["wte"]
        self.position_embeddings = parameters["wpe"]
        self.blocks = [
            self._build_block(raw_block) for raw_block in parameters["blocks"]
        ]
        self.final_layer_norm = self._build_layer_norm(parameters["ln_f"])

    @staticmethod
    def _build_layer_norm(raw_parameters):
        return LayerNormParameters(
            scale=raw_parameters["g"],
            offset=raw_parameters["b"],
        )

    @staticmethod
    def _build_linear(raw_parameters):
        return LinearParameters(
            weight=raw_parameters["w"],
            bias=raw_parameters["b"],
        )

    @classmethod
    def _build_block(cls, raw_block):
        attention = raw_block["attn"]
        feed_forward = raw_block["mlp"]
        return TransformerBlockParameters(
            attention_norm=cls._build_layer_norm(raw_block["ln_1"]),
            qkv_projection=cls._build_linear(attention["c_attn"]),
            attention_output_projection=cls._build_linear(attention["c_proj"]),
            feed_forward_norm=cls._build_layer_norm(raw_block["ln_2"]),
            feed_forward_input_projection=cls._build_linear(feed_forward["c_fc"]),
            feed_forward_output_projection=cls._build_linear(feed_forward["c_proj"]),
        )

    @staticmethod
    def _load_vocabulary(model_dir):
        with open(os.path.join(model_dir, "encoder.json")) as file:
            return json.load(file)

    @staticmethod
    def _load_merge_rules(model_dir):
        with open(
            os.path.join(model_dir, "vocab.bpe"), encoding="utf-8"
        ) as file:
            bpe_data = file.read()
        return [
            tuple(merge.split())
            for merge in bpe_data.splitlines()[1:]
            if merge.strip()
        ]

    @staticmethod
    def _load_parameters(checkpoint_path, layer_count):
        def set_nested(dictionary, keys, value):
            if not keys:
                return value
            key = keys[0]
            dictionary[key] = set_nested(
                dictionary.get(key, {}), keys[1:], value
            )
            return dictionary

        parameters = {"blocks": [{} for _ in range(layer_count)]}

        reader = TensorflowCheckpointReader(checkpoint_path)
        for name, _ in reader.list_variables():
            value = np.squeeze(reader.load_variable(name))
            name = name.removeprefix("model/")

            block_match = re.fullmatch(r"h([0-9]+)/(.*)", name)
            if block_match:
                block_number = int(block_match.group(1))
                parameter_path = block_match.group(2).split("/")
                set_nested(
                    parameters["blocks"][block_number], parameter_path, value
                )
            else:
                set_nested(parameters, name.split("/"), value)

        return parameters
