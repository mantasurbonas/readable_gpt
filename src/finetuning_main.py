"""Fine-tune the loaded GPT-2 weights on a short text fragment, then maybe generate.

This is the training counterpart of main.py. A training step improves the NumPy
arrays that ModelLoader built. When an inference string is passed, the inference
`gpt.GPT` class predicts the next tokens from those very same, now improved, arrays.
Without one, the run stops after training.

The default run is transient. The files in models/124M/ change only when
--save is passed.
"""

import argparse
import os
import sys

import finetuning
import gpt
from utils.tensorflow_checkpoint_reader import TensorflowCheckpointReader
from utils.tensorflow_checkpoint_writer import TensorflowCheckpointWriter

MODEL_SIZE = "124M"
MODELS_DIR = "models"


def main(argv=None):
    arguments = _parse_arguments(argv)
    learn_text = _resolve_source(
        arguments.learn,
        arguments.learn_from_file,
        "--learn",
        "--learn-from-file",
        required=True,
    )
    inference_text = _resolve_source(
        arguments.infere,
        arguments.infere_from_file,
        "--infere",
        "--infere-from-file",
        required=False,
    )

    model_loader = gpt.ModelLoader(model_size=MODEL_SIZE, models_dir=MODELS_DIR)
    tokenizer = gpt.Tokenizer(
        vocabulary=model_loader.vocabulary,
        merge_rules=model_loader.merge_rules,
    )

    try:
        training_pair = finetuning.NextTokenTrainingPair(
            tokenizer.encode(learn_text),
            max_length=arguments.max_length,
        )
    except ValueError as error:
        raise SystemExit(str(error))

    if training_pair.was_trimmed:
        print(
            f"Trimmed the fragment from {training_pair.original_token_count} tokens "
            f"to {arguments.max_length} so the demo stays short."
        )
    print(
        f"Training on {training_pair.position_count} next-token positions "
        f"for {arguments.steps} step(s)."
    )

    fine_tuner = finetuning.FineTuner(model_loader, learning_rate=arguments.learning_rate)
    for step_number in range(1, arguments.steps + 1):
        loss = fine_tuner.train_on(training_pair)
        print(f"step {step_number:3d}  loss {loss:.4f}")

    if arguments.save:
        _save_weights(fine_tuner.adjustable_weights())
    else:
        print(f"Transient run: the files in {_model_dir()}/ were not changed.")

    if inference_text is not None:
        _generate_and_print(model_loader, tokenizer, inference_text, arguments.tokens)


def _model_dir():
    return os.path.join(MODELS_DIR, MODEL_SIZE)


def _save_weights(adjustable_weights):
    checkpoint_path = TensorflowCheckpointReader.latest_checkpoint(_model_dir())
    if not checkpoint_path:
        raise SystemExit(f"Cannot save: no checkpoint found in {_model_dir()}/.")

    TensorflowCheckpointWriter(checkpoint_path).overwrite(adjustable_weights)
    print(f"Saved updated weights to {checkpoint_path}.data shard.")


def _generate_and_print(model_loader, tokenizer, inference_text, token_count):
    token_ids = list(tokenizer.encode(inference_text))
    if not token_ids:
        raise SystemExit(
            "The inference text encodes to no tokens, so there is nothing to continue."
        )

    print(f"Generating {token_count} token(s) with the inference GPT class...")
    inference_gpt = gpt.GPT(
        token_embeddings=model_loader.token_embeddings,
        position_embeddings=model_loader.position_embeddings,
        blocks=model_loader.blocks,
        final_layer_norm=model_loader.final_layer_norm,
        attention_head_count=model_loader.attention_head_count,
        context_size=model_loader.context_size,
    )
    for _ in range(token_count):
        token_ids.append(inference_gpt.predict_next_token(token_ids))
        print(".", end="", flush=True)

    print()
    print(tokenizer.decode(token_ids))


def _resolve_source(literal, file_path, flag, file_flag, required):
    if literal is not None and file_path is not None:
        raise SystemExit(f"Pass only one of {flag} or {file_flag}.")
    if literal is None and file_path is None:
        if required:
            raise SystemExit(f"Pass {flag} or {file_flag}.")
        return None
    if literal is not None:
        if not literal.strip():
            raise SystemExit(f"{flag} is empty.")
        return literal
    return _read_text_file(file_path)


def _read_text_file(path):
    if not os.path.isfile(path):
        raise SystemExit(f"{path!r} is not a file.")
    with open(path, encoding="utf-8-sig") as file:
        text = file.read()
    if not text.strip():
        raise SystemExit(f"{path!r} is empty.")
    return text


def _parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description="Fine-tune GPT-2 124M on a short text fragment."
    )
    parser.add_argument("--learn", help="Literal text to train on.")
    parser.add_argument("--learn-from-file", help="UTF-8 file whose contents are the training text.")
    parser.add_argument("--infere", help="Literal prompt to generate from after training.")
    parser.add_argument("--infere-from-file", help="UTF-8 file whose contents are the generation prompt.")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Overwrite the checkpoint data shard after training.",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--tokens",
        type=int,
        default=20,
        help="How many tokens to generate after training, when an inference source is present.",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--max-length",
        type=int,
        default=32,
        help="Trim the training text to this many tokens.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
