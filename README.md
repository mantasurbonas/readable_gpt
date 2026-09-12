# Project summary

A readable / understandable implementation of GPT-2 architecture.  
The code is intentionally verbose and un-compacted, in order to explain all the concepts in human language.

This is *inference* only (no model training). It uses GPT-2 124M weights and 1024 token context from OpenAI. No dependencies except for NumPy and regex.


# Usage

## Installation

```
setup.bat
```

Installs venv and all missing dependencies.

## Download model files

```
download_model_files.bat
```

Downloads the GPT-2 124M weights from OpenAI into `models/124M/`. Uses `curl`, which is built into Windows 10+. Only needs to be run once.

## Running

```
run.bat The cat climbed onto the
```

This will output
```
The cat climbed onto the roof of the building and began to
```

If you omit the prompt, a built-in default is used.

# About GPT architecture

GPT stands for **Generative Pre-trained Transformer**. The simplest way to think about it is:

> **GPT is a very advanced “guess the next word” machine that learned language by reading enormous amounts of text.**

Imagine you type:

> “The cat climbed onto the ___”

GPT might predict: **roof**, **table**, **bed**, etc. It doesn’t just look at the last word. It looks at the whole context and figures out which earlier words matter.

The acronym meanings are:

**Generative** — it generates new text.

**Pre-trained** — it was pre-trained on a huge amount of data beforehand.

**Transformer** — the neural-network architecture it uses (read about it below).

We will first follow a sentence through the entire model. After that overview, we will return to the individual parts and examine how they work.

### 1. Your sentence gets broken into tokens

GPT doesn't directly see words. It sees small chunks called **tokens**.

For example:

```text
"I love programming!"
```

might become something roughly like:

```text
"I" | " love" | " program" | "ming" | "!"
```

Each token is converted into a number.

So internally, GPT sees something more like:

```text
[42, 891, 17234, 392, 8]
```

### 2. Tokens become vectors

GPT turns every token into a long list of numbers called an **embedding**.

You can imagine an embedding as a location on a giant “meaning map.”

For example, the representations of:

```text
dog
cat
wolf
```

might end up relatively close together, while:

```text
dog
democracy
microwave
```

would be much farther apart.

The real vectors can contain thousands of numbers (our model uses 768).

### 3. GPT also needs to know word order

Consider:

> “The dog bit the man.”

versus:

> “The man bit the dog.”

Same words, but very different meaning.

So GPT adds information about **position** so it knows which token came first, second, third, etc.

Now GPT has roughly:

```text
token meaning + token position
```

for every token.

### 4. The vectors pass through Transformer blocks

A GPT model contains a stack of repeated structures called **Transformer blocks**.

The Python code defines the structure of a block. The pre-trained model files provide the learned numbers, called **weights**, that each block uses.

Every block has the same structure, but it has its own weights. This means that the blocks perform the same kinds of steps, while their different learned weights let them detect and transform different patterns.

The first block receives the vectors containing token meaning and position. It transforms those vectors and passes its output to the second block. The second block transforms that output and passes it to the third, and so on:

```text
token meaning + position
            ↓
    Transformer block 1
            ↓
    Transformer block 2
            ↓
    Transformer block 3
            ↓
           ...
            ↓
    final token representations
```

Our GPT-2 124M model has **12 Transformer blocks**.

Inside each block, attention helps tokens gather relevant information from other tokens, and a feed-forward network processes that information. Layer normalization and residual connections help these transformations work reliably.

We will examine all four of those parts later. For now, the important idea is:

> **A Transformer is a stack of blocks that repeatedly improves each token's representation using the context around it.**

As information passes through the blocks, the model can build richer representations of the context.

For example, early blocks might notice:

> “Apple is a word.”

Later blocks might figure out:

> “Apple refers to the technology company here, not the fruit.”

### 5. GPT predicts the next token

After all the Transformer blocks, GPT produces a score for every token it knows. Those scores can be turned into probabilities.

For:

> “The capital of France is”

it might internally produce something like:

```text
Paris      96%
London      1%
France      0.5%
Berlin      0.3%
pizza       0.00001%
...
```

Then it chooses a token, for example:

> **Paris**

Now the input becomes:

> “The capital of France is Paris”

and GPT predicts the **next** token again. Then again, and again. That's how a full response is generated.

It doesn't normally generate the entire paragraph at once, but rather more like:

```text
predict → add token → predict → add token → predict...
```

extremely quickly.

### 6. The whole journey at a glance

The complete process looks like this:

```text
Your message
    ↓
Split into tokens
    ↓
Turn tokens into vectors
    ↓
Add position information
    ↓
┌─────────────────────────────┐
│ Transformer block 1         │
│ Transformer block 2         │
│ Transformer block 3         │
│            ...              │
└─────────────────────────────┘
    ↓
Assign a score to every known token
    ↓
Choose a token
    ↓
Add it to the message and repeat
    ↓
Response
```

That is the entire GPT process at a high level. The following sections take a closer look at what happens inside the blocks and how the final token scores are produced.

### 7. More details

#### 7.1 Inside one Transformer block

Each Transformer block performs the same sequence of operations:

```text
        input
          ↓
          ↓   ──────────────────────┐
          ↓                         │
        layer normalization         │
          ↓                         │
        attention                   │
          ↓                         │
        output + original input <───┘
          ↓
          ↓   ──────────────────────┐
          ↓                         │
        layer normalization         │
          ↓                         │
        feed-forward network        │
          ↓                         │
        output + original input <───┘
          ↓
        result
```

The result becomes the input to the next Transformer block. We will now examine each part separately.

#### 7.2 Attention and attention heads

Attention is the key idea behind Transformers.

Suppose the sentence is:

> “Sarah dropped the glass because **it** was slippery.”

When GPT processes the word **“it”**, it needs to figure out what “it” refers to.

Attention lets the model basically ask:

> “Which earlier words should I pay attention to?”

It might give higher importance to:

```text
glass   ██████████
Sarah   ██
dropped ███
because █
```

The model learns these relationships automatically.

This mechanism is called **self-attention** because the tokens gather information from other tokens in the same piece of text.

GPT doesn't have just one attention system. Each block has multiple **attention heads** working at the same time. Our model has 12 attention heads in every block.

An attention head is like one specialized “relationship detector.” Its job is to look at each token and ask:

> “Which other tokens matter to me right now, and how much?”

One head might learn to notice:

* pronouns
* grammar
* nearby words
* names
* cause and effect
* quotations
* code structure

You can imagine a group of tiny detectives studying the same sentence for different clues.

##### How one attention head works

For every token, the model creates three vectors:

* **Query (Q):** “What am I looking for?”
* **Key (K):** “What kind of information do I contain?”
* **Value (V):** “What information should I give you if you pay attention to me?”

Imagine:

> “Sarah gave Emma her book.”

When processing **“her”**, its **query** might effectively be looking for something like:

> “Which earlier person could this refer to?”

The words “Sarah” and “Emma” each have **keys**. The query is compared with those keys.

Very simplified:

```text
Query from "her"
      ↓
compare with keys

Sarah   → score 0.7
gave    → score 0.1
Emma    → score 0.9
book    → score 0.2
```

Those scores are turned into attention weights, roughly:

```text
Sarah   ██████
gave    █
Emma    █████████
book    ██
```

Then the model takes the **values** from those words and mixes them according to those weights.

So the new representation for `"her"` now contains information gathered from the words that mattered most.

```text
QKᵀ
↓
"How relevant is each token?"

softmax
↓
"Turn those relevance scores into percentages"

× V
↓
"Collect information from those tokens"
```

##### Why have multiple heads?

Because one single attention pattern isn't enough.

Suppose GPT reads:

> “John couldn't lift the box because it was too heavy.”

Different heads could potentially learn different relationships:

```text
Head 1: "it" → "box"
        pronoun relationship

Head 2: "heavy" → "box"
        property relationship

Head 3: "couldn't lift" → "heavy"
        cause/effect relationship

Head 4: nearby words
        grammar/syntax
```

Then their outputs are combined.

So you can picture multi-head attention like this:

```text
                    ┌→ Head 1 → grammar
Sentence → tokens ──┼→ Head 2 → references
                    ├→ Head 3 → relationships
                    ├→ Head 4 → context
                    └→ Head N → ...
                           ↓
                    combine results
                           ↓
                 richer representation
```

Some attention heads can be fairly interpretable, while others do complicated things that aren't easy to describe in human language.

The cleanest mental model is:

> **Each attention head looks at the same sentence from a different learned perspective.**

And **multi-head attention** means GPT can examine many different relationships at the same time.

##### Who creates the attention heads?

The model architecture decides the layout: in our model there are exactly 12 heads per block.

During training, the model learns the weights used by each head in each block. Those weights are saved in the model files.

At runtime, our Python code loads the frozen weights. It then uses the current token representations and those weights to calculate Q, K, and V. These newly calculated values are called **activations**. The weights stay the same from one prediction to the next, but the activations depend on the input text.

#### 7.3 GPT may only look backward

GPT's attention is not allowed to read the future.

When the model is working on a token, it can use that token and everything before it. Tokens to the right are hidden. This rule is called a **causal mask** (“causal” here just means “causes can only come from the past”).

Take:

```text
The capital of France is Paris
```

While the model is at **is**, the allowed context is:

```text
The  capital  of  France  is  [Paris is hidden]
```

It must guess the next token from that prefix alone. That is what makes GPT a next-token machine: it cannot peek at the answer.

#### 7.4 Feed-forward network

After attention, each token passes through a **feed-forward network** (also called an **MLP**).

You can think of attention as:

> “Find the relevant information.”

And the feed-forward network as:

> “Think about / transform that information.”

The feed-forward network is part of the model, and each block has its own learned feed-forward weights. Its actual activations and results are calculated at runtime from the current token representations.

A typical Transformer feed-forward network looks roughly like:

```text
input
  ↓
Linear layer 1
  ↓
activation function
  ↓
Linear layer 2
  ↓
output
```

For GPT-2 124M, each token representation contains `768` numbers. Its feed-forward network temporarily expands that representation to `3072` numbers before reducing it to `768` numbers again.

Unlike attention, the feed-forward network processes each token separately. Information is shared between tokens by attention; the feed-forward network then transforms the information held by each token.

#### 7.5 Layer normalization keeps the numbers well-behaved

As a token representation moves through the model, its numbers can grow, shrink, or become unevenly distributed.

**Layer normalization** rescales those numbers into a more consistent range. This helps each part of the Transformer receive values that are easier to work with.

Layer normalization does not mix information between different tokens. It normalizes each token's vector independently, then applies a learned scale and offset.

As the block map showed, layer normalization happens before both attention and the feed-forward network. There is also one final layer normalization after the last Transformer block.

#### 7.6 Residual connections preserve information

Attention and the feed-forward network transform the token representations, but GPT does not simply replace the old representations with the new ones.

Instead, it adds each part's output back to its input. This shortcut is called a **residual connection**.

There are two residual connections in each block:

```text
attention output + input to attention

feed-forward output + input to feed-forward
```

Residual connections let information pass through a block directly while attention and the feed-forward network add useful changes.

Another way to think about this is that each part describes an **update** to the existing representation instead of having to rebuild the entire representation from nothing.

#### 7.7 Logits: scores before choosing the next token

After the final Transformer block and layer normalization, GPT turns the last token's `768`-number representation into one score for every token in its vocabulary.

These scores are called **logits**. A larger logit means the model considers that token a better choice, but logits are not probabilities: they do not have to be between `0` and `1`, and they do not have to add up to `100%`.

A function called **softmax** can convert the logits into probabilities:

```text
logits → softmax → probabilities
```

Our implementation does not need to calculate those probabilities because it always chooses the token with the largest logit.

The chosen token is appended to the input, and the entire inference process repeats to produce the following token.

#### 7.8 Why is it called “Transformer”?

Before Transformers, many language models processed text more like reading a sentence **one word at a time**.

Transformers introduced multi-head attention, which lets the model examine relationships between many tokens much more efficiently.

The architecture repeatedly **transforms** each token's representation. A token begins with a general meaning and a position. As it passes through the blocks, it gathers context and becomes a richer representation of what that token means in this particular text.

That architecture turned out to scale extremely well.
