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

So GPT turns every token into a long list of numbers called an **embedding**.

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

### 4. The most important part: Attention

This is the key idea behind Transformers.

Suppose the sentence is:

> “Sarah dropped the glass because **it** was slippery.”

When GPT processes the word **“it”**, it needs to figure out what “it” refers to.

Attention lets the model basically ask:

> “Which earlier words should I pay attention to?”

It might give higher importance to:

```text
glass  ██████████
Sarah  ██
dropped ███
because █
```

The model learns these relationships automatically.

This mechanism is called **self-attention**.

And GPT doesn't have just one attention system. It has many **attention heads** working at the same time.

One head might learn to notice:

* pronouns
* grammar
* nearby words
* names
* cause and effect
* quotations
* code structure

You can imagine a bunch of tiny detectives studying the same sentence for different clues.


### 5. Attention happens inside Transformer blocks

A GPT model is basically a huge stack of repeated building blocks.

Very simplified:

```text
Input tokens
     ↓
Embeddings
     ↓
┌────────────────────┐
│ Transformer Block  │
│                    │
│  Self-Attention    │
│        ↓           │
│ Neural Network     │
└────────────────────┘
     ↓
┌────────────────────┐
│ Transformer Block  │
└────────────────────┘
     ↓
┌────────────────────┐
│ Transformer Block  │
└────────────────────┘
     ↓
    ...
     ↓
Next-token prediction
```

Large models can contain **dozens or hundreds of these layers**.

As information passes through them, the model's understanding of the context becomes richer.

For example, early layers might notice:

> “Apple is a word.”

Later layers might figure out:

> “Apple refers to the technology company here, not the fruit.”

Note: our model has 12 blocks, and each block has 12 attention heads.


### 6. There's another neural network inside each block

After attention, each token passes through a **feed-forward network** (also called **MLP**).

You can think of attention as:

> “Find the relevant information.”

And the MLP as:

> “Think about / transform that information.”


### 7. Layer normalization keeps the numbers well-behaved

As a token representation moves through the model, its numbers can grow, shrink, or become unevenly distributed.

**Layer normalization** rescales those numbers into a more consistent range. This helps each part of the Transformer receive values that are easier to work with.

Layer normalization does not mix information between different tokens. It normalizes each token's vector independently, then applies a learned scale and offset.

The layer normalization happens before both attention and the feed-forward.

### 8. Residual connections preserve information

Attention and the feed-forward network transform the token representations, but GPT does not simply replace the old representations with the new ones.

Instead, it adds each part's output back to its input, so each block in fact is:

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
        feed forward                │
          ↓                         │
        output + original input <───┘
          ↓
        result
```

This shortcut is called a **residual connection**. It lets information pass through a block directly while attention and the feed-forward network add useful changes.


### 9. GPT predicts the next token

After all the Transformer layers, GPT produces scores that can be turned into probabilities.

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

#### GPT may only look backward

Attention is not allowed to read the future.

When the model is working on a token, it can use that token and everything before it. Tokens to the right are hidden. This rule is called a **causal mask** (“causal” here just means “causes can only come from the past”).

Take:
```
> The capital of France is Paris
```

While the model is at **is**, the allowed context is:
```text
The  capital  of  France  is  [Paris is hidden]
```

It must guess the next token from that prefix alone. That is what makes GPT a next-token machine: it cannot peek at the answer.

### Why is it called “Transformer”?

Before Transformers, many language models processed text more like reading a sentence **one word at a time**.

Transformers introduced multi-head attention, which lets the model examine relationships between many tokens much more efficiently.

That architecture turned out to scale extremely well.

So the whole process can be pictured like this:

```text
Your message
    ↓
Split into tokens
    ↓
Turn tokens into vectors
    ↓
Add position information
    ↓
Transformer
┌─────────────────────────────┐
│ ( normalization )           │
│ Attention: what matters?    │
│ ( normalization )           │
│ Neural net: process it      │
│ ( normalization )           │
│ Attention: what matters?    │
│ ( normalization )           │
│ Neural net: process it      │
│            ...              │
└─────────────────────────────┘
    ↓
Assign probabilities to ALL known tokens
    ↓
Choose token
    ↓
Repeat
    ↓
Response
```

### More details: Attention Heads

An **attention head** is like one specialized “relationship detector” inside a Transformer.

Its job is to look at each token and ask: "Which other tokens matter to me right now, and how much?"

Suppose the sentence is:

> “The dog that chased the cats was tired.”

When processing **“was”**, one attention head might learn that the important word is **“dog”**, because “dog” is the subject.

Another head might focus on nearby grammar. Another might track pronouns. Another might detect quotation structure, code syntax, or long-distance relationships.

So instead of having one attention mechanism, GPT uses **multiple attention heads in parallel**.

### How one attention head works

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

#### Why have multiple heads?

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

#### Who creates the attention heads?

The model architecture decides the layout: in our model there are exactly 12 heads per block. Model training creates weights and those weights are stored in the model file (frozen).
At runtime, GPT loads those frozen weights, takes the runtime token values, and calculates Q, K, and V (activations).


### Feed-forward

The feed-forward network is also **part of the model**, and its actual activations/results are also computed at runtime.

A typical Transformer feed-forward block looks roughly like:

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

For GPT-2 124M, each token representation contains `768` numbers. Its feed-forward network temporarily expands that representation to `3072` numbers.

### Logits: scores before choosing the next token

After the final Transformer block and layer normalization, GPT turns the last token's `768`-number representation into one score for every token in its vocabulary.

These scores are called **logits**. A larger logit means the model considers that token a better choice, but logits are not probabilities: they do not have to be between `0` and `1`, and they do not have to add up to `100%`.

A function called **softmax** can convert the logits into probabilities:

```text
logits → softmax → probabilities
```

NOTE: Our implementation does not need to calculate those probabilities because it always chooses the token with the largest logit.

The chosen token is appended to the input, and the entire inference process repeats to produce the following token.
