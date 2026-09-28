"""Run the complete educational GPT-2 text-generation pipeline."""

import sys

from gpt import GPT, ModelLoader, Tokenizer
from utils.tracer import Tracer

DEFAULT_PROMPT = "Ladies and Gentelmen, hello and welcome to my modest "


if __name__ == "__main__":
    model_loader = ModelLoader(model_size="124M", models_dir="models")

    print(
        f"model loaded! "
        f"context size: {model_loader.context_size}, "
        f"vocabulary size: {len(model_loader.vocabulary)}, "
        f"blocks: {len(model_loader.blocks)}, "
        f"attention heads: {model_loader.attention_head_count}"
    )

    tokenizer = Tokenizer(
        vocabulary=model_loader.vocabulary,
        merge_rules=model_loader.merge_rules
    )

    prompt = DEFAULT_PROMPT
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:]) 

    token_ids = tokenizer.encode(prompt)

    tracer = Tracer()
    tracer.off() # change to on() to trace the algorithm

    gpt = GPT(
        token_embeddings=model_loader.token_embeddings,
        position_embeddings=model_loader.position_embeddings,
        blocks=model_loader.blocks,
        final_layer_norm=model_loader.final_layer_norm,
        attention_head_count=model_loader.attention_head_count,
        context_size=model_loader.context_size,
        tracer=tracer,
    )

    for _ in range(1):
        next_token = gpt.predict_next_token(token_ids)
        token_ids.append(next_token)
        print(".", end="", flush=True)

    print()

    decoded_string = tokenizer.decode(token_ids)

    print(decoded_string)
