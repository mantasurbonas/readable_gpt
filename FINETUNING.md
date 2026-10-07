# Finetuning GPT-2 in this project

The inference program in `src/main.py` loads the OpenAI 124M weights and guesses the next token. This lesson shows how those same numbers can be improved.

Finetuning does not change the architecture. It changes the values inside the arrays the model already uses: the embedding tables, every attention and feed-forward projection, and every layer-norm scale and offset.

## What a run does

1. Load the checkpoint into NumPy arrays, the same way inference does.
2. Turn the given text into tokens. At each position the model scores every vocabulary word as the next token.
3. Compare those scores with the token that actually comes next. How wrong those scores were is the loss.
4. Walk the error backward through the same layers, in reverse order, and nudge every weight a little (Adam).
5. When `--infere` or `--infere-from-file` is present, build the existing `GPT` class on those same arrays and generate the next tokens from that string. With neither flag, stop after training.

A training step cannot reuse `GPT.predict_next_token`. That method keeps only the last position and throws the intermediates away. The `finetuning` classes repeat the calculation and remember what they need for passing blame back.

```
learn text
    ↓
calculate  →  loss  →  pass blame back  →  adjust weights (in-place)
    ↓
optional infere text → inference GPT.predict_next_token
```

The default run never opens a model file for writing. `--save` overwrites the float values in the checkpoint data shard. The index, vocabulary, and `hparams.json` stay as they are, so `run.bat` can load a saved model with no code changes.

## Two modes

**Transient (default).** Train in memory and exit. The next run starts from the original OpenAI weights again. Generation runs only when an inference flag is passed.

```
finetune.bat --learn="The cat sat on the mat and looked at the moon."
```

**Persisted.** Same training, then replace the previous weight file.

```
finetune.bat --save --learn="The cat sat on the mat and looked at the moon."
```

**Generate after training.** Same training, then continue from a separate prompt. Omit both inference flags to skip this step.

```
finetune.bat --learn="The cat sat on the mat and looked at the moon." --infere="The cat sat"
```

`--save` really does replace the previous weights. There is no automatic backup. `download_model_files.bat` downloads the original OpenAI checkpoint again.

Useful flags:

- `--learn` literal training text
- `--learn-from-file` UTF-8 file to train on. Exactly one of `--learn` or `--learn-from-file` is required
- `--infere` literal generation prompt
- `--infere-from-file` UTF-8 file to generate from. Pass neither inference flag to skip generation, or exactly one of them
- `--steps` how many training passes (default 20)
- `--tokens` how many tokens to generate afterward (default 20). Used only when an inference source is present
- `--learning-rate` Adam step size (default 5e-5)
- `--max-length` trim the training text to this many tokens (default 32). The inference prompt is not trimmed

`--learn` and `--infere` are always literal text. A path is read only when the matching `-from-file` flag is set. Those files are UTF-8, and a leading byte-order mark is ignored.

## Will the files break?

The file format can be saved many times. Training only replaces the floats; every array keeps its original shape. The loader does not wear out.

A crash in the middle of `--save` is the real corruption case. The writer copies the data shard, writes into the copy, and replaces the original only after every tensor fits. If a size check fails, nothing is written.

What does get worse is the model's behavior. A short fragment trained for a long time, or with a large learning rate, teaches the model to recite that fragment and forget ordinary English. That is overfitting, not a broken file. If a saved model starts producing nonsense, run `download_model_files.bat` and start from the official weights again.

## Where the new code lives

The `src/finetuning/` package mirrors `src/gpt/`: the same file names, class names and constructor arguments. Each class also remembers what it calculated, so that it can **pass blame back** to its inputs and its weights.

| File | Role |
|---|---|
| `src/finetuning_main.py` | Command line: load, train, generate, optional save |
| `src/finetuning/trainable_gpt.py` | `TrainableGPT`: scores every next token, passes blame back through the whole model |
| `src/finetuning/transformer_block.py` | `TransformerBlock`, mirrors the inference class |
| `src/finetuning/multi_head_attention.py` | `SingleHeadAttention` and `MultiHeadAttention` |
| `src/finetuning/feed_forward.py` | `FeedForward` |
| `src/finetuning/logits_calculator.py` | `LogitsCalculator`, scores every word at every position |
| `src/finetuning/linear_layer.py`, `layer_normalization.py`, `activation.py` | The small building blocks that the inference code uses as plain functions |
| `src/finetuning/adjustable_weights.py` | `AdjustableWeights`: a weight array plus its blame |
| `src/finetuning/next_token_loss_calculator.py` | `NextTokenLossCalculator`: how wrong was the model about the real next token? |
| `src/finetuning/weight_adjuster.py` | In-place Adam update |
| `src/finetuning/next_token_training_pair.py` | `NextTokenTrainingPair`: text cut into "what the model sees" and "what it should guess" |
| `src/finetuning/require_calculated.py` | `require_calculated`: fails clearly when blame is passed back before calculating |
| `src/finetuning/fine_tuner.py` | `FineTuner`: one training step |
| `src/utils/tensorflow_checkpoint_writer.py` | Atomic overwrite of the `.data` shard |
| `src/finetuning/check_blame_by_nudging.py` | Self-check of every part |

Words used in the code, and their usual names in the field:

| In the code | Usual name |
|---|---|
| `calculate()` | forward pass |
| `pass_blame_back()` | backward pass (backpropagation) |
| blame | gradient |
| loss | cross-entropy loss |
| adjustable weights | parameters |
| weight adjuster | optimizer (Adam) |
| checking by nudging | numerical gradient check |

The files under `src/gpt/` that inference already uses are not part of this lesson. They stay frozen-weight readers. After the weight adjuster has written into the arrays they point at, they generate from the updated model.

To check that every part passes blame back correctly, run from the project root:

```
set PYTHONPATH=src
.venv\Scripts\python.exe -m finetuning.check_blame_by_nudging
```
