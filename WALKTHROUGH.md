# Walkthrough: predicting one next token

This file follows one short prompt through the GPT-2 124M.

The prompt is:

```text
Cat ate the
```

The model guesses the next token. For this particular LLM, that token is `" food"`.

---

## 1. Text becomes token ids

GPT does not read letters. The tokenizer splits the string into tokens and looks each one up in a vocabulary table.

```text
"Cat ate the"
        ↓
"Cat" | " ate" | " the"
        ↓
[ 21979,  15063,   262 ]
```

Three pieces of text -> three integers. Those integers stay in this order: position 0 is `"Cat"`, position 1 is `" ate"`, position 2 is `" the"`.

---

## 2. Token ids become meaning vectors

The model keeps a table of **50,257 rows × 768 columns**. Each token id selects one row.

```text
21979  →  [ 0.04, -0.07,  0.19,  ... ]     768 numbers   ("Cat")
15063  →  [ 0.05, -0.19,  0.13,  ... ]     768 numbers   (" ate")
262    →  [-0.04,  0.01,  0.04,  ... ]     768 numbers   (" the")
```

These rows are the **token embeddings**. They carry a meaning of each token, with no idea yet of where it sits in the sentence.

---

## 3. Position is added

The model also has a position table: row 0 for “first token”, row 1 for “second token”, and so on, up to 1024.

```text
position 0  →  [-0.02, -0.20,  0.00,  ... ]     768 numbers
position 1  →  [ 0.02, -0.05, -0.09,  ... ]     768 numbers
position 2  →  [ 0.00, -0.08,  0.05,  ... ]     768 numbers
```

Add them element by element:

```text
[ 0.04, -0.07,  0.19,  ... ]   +  [-0.02, -0.20,  0.00,  ... ]
[ 0.05, -0.19,  0.13,  ... ]   +  [ 0.02, -0.05, -0.09,  ... ]
[-0.04,  0.01,  0.04,  ... ]   +  [ 0.00, -0.08,  0.05,  ... ]
        ↓
[ 0.02, -0.27,  0.20,  ... ]     768 numbers   ("Cat"   at place 0)
[ 0.07, -0.24,  0.03,  ... ]     768 numbers   (" ate"  at place 1)
[-0.04, -0.08,  0.10,  ... ]     768 numbers   (" the"  at place 2)
```

Still three vectors, 768 numbers each. Now each row knows **what** it is and **where** it is.

This is the input to the first Transformer block.

---

## 4. Transformer block 1

Every block does the same two jobs:

1. **Attention** — let tokens gather information from earlier tokens
2. **Feed-forward** — let each token think about what it just gathered

Each job is wrapped in layer normalization and a residual connection (add the original input back). The code does this:

```text
        input
          │
          ├──────────────────────────┐
          ↓                           │
        layer normalization           │
          ↓                           │
        attention                     │
          ↓                           │
        output + original input  <────┘
          │
          ├──────────────────────────┐
          ↓                           │
        layer normalization           │
          ↓                           │
        feed-forward                  │
          ↓                           │
        output + original input  <────┘
          ↓
        result
```

The traces below are this first block only. Blocks 2 through 12 do the same work with different learned weights.

### 4.1 Layer normalization (before attention)

Each of the three rows is rescaled on its own. Tokens do not mix here.

```text
[ 0.02, -0.27,  0.20,  ... ]
[ 0.07, -0.24,  0.03,  ... ]
[-0.04, -0.08,  0.10,  ... ]
        ↓  layer norm
[ 0.01, -0.10,  0.02,  ... ]     768 numbers
[ 0.07, -0.17, -0.04,  ... ]     768 numbers
[-0.04, -0.05,  0.02,  ... ]     768 numbers
```

### 4.2 Attention

One matrix multiply turns each 768-number row into **2304** numbers: Query, Key, and Value sitting side by side.

```text
768  →  2304   per token

"Cat"   [ 0.21, -0.35,  0.34,  ... | -1.43,  2.06,  0.99,  ... |  0.03,  0.14, -0.02,  ... ]
" ate"  [ 0.99,  1.28,  0.29,  ... | -1.92,  3.14,  1.59,  ... |  0.25, -0.13,  0.18,  ... ]
" the"  [ 0.64, -0.16,  0.53,  ... | -2.24,  2.63,  1.92,  ... | -0.14, -0.00,  0.12,  ... ]
         └──── 768 Query ────┘   └───── 768 Key ──────┘   └──── 768 Value ────┘
```

Those 768 Query / Key / Value numbers are then split across **12 heads**. Each head only sees **64** of them. Head 0 takes the first 64 of each, so the start of its vectors is the start of the rows above.

#### One head, in slow motion

Take head 0. It holds three short vectors per token:

```text
Queries (64 numbers each)          Keys (64)                       Values (64)
"Cat"   [ 0.21, -0.35,  0.34, ... ]   [-1.43,  2.06,  0.99, ... ]    [  0.03,  0.14, -0.02, ... ]
" ate"  [ 0.99,  1.28,  0.29, ... ]   [-1.92,  3.14,  1.59, ... ]    [  0.25, -0.13,  0.18, ... ]
" the"  [ 0.64, -0.16,  0.53, ... ]   [-2.24,  2.63,  1.92, ... ]    [ -0.14, -0.00,  0.12, ... ]
```

Compare every Query with every Key. That produces a **3 × 3** table of raw scores:

```text
              Key: Cat     ate      the
Query: Cat      0.24    -7.27   -13.24
Query: ate      6.76     6.24   -13.31
Query: the     -6.26    -7.29   -23.95
```

Cell `[row i, column j]` is “how much should token *i* care about token *j*?”

The scores are divided by √64 = 8 so they do not explode. The top-left cell is `0.24 / 8 = 0.03`. Then a **causal mask** hides the future. A token may only look at itself and earlier tokens. Forbidden cells get a huge negative number:

```text
              Key: Cat     ate      the
Query: Cat      0.03    -1e10    -1e10      ← "Cat" may not look at later words
Query: ate      0.85     0.78    -1e10
Query: the     -0.78    -0.91    -2.99
```

Softmax turns each row into mixing weights that add up to 1:

```text
              Cat     ate     the
"Cat"        1.00    0.00    0.00      ← only itself
" ate"       0.52    0.48    0.00      ← "Cat" and itself, nearly even
" the"       0.50    0.44    0.06      ← whole prefix, heaviest on "Cat"
```

Then each token mixes Values with those percentages. For `" the"`:

```text
0.50 × (value of "Cat")  +  0.44 × (value of " ate")  +  0.06 × (value of " the")
        ↓
[ 0.12,  0.02,  0.07,  ... ]     64 numbers
```

`"Cat"` put all of its weight on itself, so its mixed row is its own Value: `[ 0.03, 0.14, -0.02, ... ]`.

After this head, we still have **three** vectors, now 64 numbers each:

```text
"Cat"   [ 0.03,  0.14, -0.02,  ... ]
" ate"  [ 0.14,  0.01,  0.08,  ... ]
" the"  [ 0.12,  0.02,  0.07,  ... ]
```

`" the"` has collected a blend of the prefix: half from `"Cat"`, 0.44 from `" ate"`, and a little from itself. This is one head's view of the sentence. The choice of `" food"` is made later, after multiple such steps.

#### The other 11 heads

Heads 1 through 11 do the same thing with different 64-number slices. Each still outputs 3 × 64.

Glue all 12 heads side by side. Head 0 occupies the first 64 columns, so the start of each glued row matches that head's mixed values above:

```text
12 heads × 64 numbers  =  768

"Cat"   [ 0.03,  0.14, -0.02,  ... | head1 64 | ... | head11 64 ]
" ate"  [ 0.14,  0.01,  0.08,  ... | head1 64 | ... | head11 64 ]
" the"  [ 0.12,  0.02,  0.07,  ... | head1 64 | ... | head11 64 ]
```

One last linear layer remixes those neighborhoods so a finding from one head can spread across the whole row:

```text
        ↓  output projection  (768 → 768)
[ 0.29, -0.78, -0.20,  ... ]     768 numbers   attention output for "Cat"
[ 0.04, -0.66, -0.80,  ... ]     768 numbers   attention output for " ate"
[-0.56,  0.15, -0.30,  ... ]     768 numbers   attention output for " the"
```

### 4.3 Add the original input back

Attention produced an **update** ("nudge" to the new value), not a replacement. Add it to the vectors that entered this block:

```text
[ 0.02, -0.27,  0.20,  ... ]   +  [ 0.29, -0.78, -0.20,  ... ] = [ 0.31, -1.04,  0.00,  ... ]
[ 0.07, -0.24,  0.03,  ... ]   +  [ 0.04, -0.66, -0.80,  ... ] = [ 0.12, -0.90, -0.77,  ... ]
[-0.04, -0.08,  0.10,  ... ]   +  [-0.56,  0.15, -0.30,  ... ] = [-0.59,  0.07, -0.20,  ... ]

```

Still three tokens. The `" the"` row now carries both its original meaning-plus-position and what attention gathered from `"Cat"` and `" ate"`.

### 4.4 Layer normalization (before feed-forward)

Again, each row on its own:

```text
[ 0.31, -1.04,  0.00,  ... ]
[ 0.12, -0.90, -0.77,  ... ]
[-0.59,  0.07, -0.20,  ... ]
        ↓  layer norm
[ 0.08, -0.15,  0.01,  ... ]     768 numbers
[ 0.05, -0.11, -0.11,  ... ]     768 numbers
[-0.03,  0.04, -0.04,  ... ]     768 numbers
```

### 4.5 Feed-forward

The feed-forward network does **not** look at other tokens. Each of the three rows is stretched, bent, and squeezed independently.

```text
768  →  3072     stretch
        ↓
GELU             bend (a smooth activation; without it the two linear layers would collapse)
        ↓
3072 →  768      squeeze
```

For `" the"`:

```text
[-0.03,  0.04, -0.04,  ... ]          768
        ↓  linear
[ 0.26, -2.50, -1.75,  ... ]          3072
        ↓  GELU
[ 0.16, -0.02, -0.07,  ... ]          3072
        ↓  linear
[-0.74,  0.19, -0.50,  ... ]          768
```

The other two tokens get their own 3072-wide detour and come back as 768 as well:

```text
[ 0.33,  0.53,  0.17,  ... ]     768 numbers   feed-forward output for "Cat"
[ 0.16,  0.72, -0.44,  ... ]     768 numbers   feed-forward output for " ate"
[-0.74,  0.19, -0.50,  ... ]     768 numbers   feed-forward output for " the"
```

### 4.6 Add the original (post-attention) input back

```text
[ 0.31, -1.04,  0.00,  ... ]   +  [ 0.33,  0.53,  0.17,  ... ]
[ 0.12, -0.90, -0.77,  ... ]   +  [ 0.16,  0.72, -0.44,  ... ]
[-0.59,  0.07, -0.20,  ... ]   +  [-0.74,  0.19, -0.50,  ... ]
        ↓
[ 0.64, -0.52,  0.17,  ... ]     768 numbers
[ 0.28, -0.18, -1.21,  ... ]     768 numbers
[-1.33,  0.26, -0.70,  ... ]     768 numbers
```

That is the output of Transformer block 1. Three tokens in, three tokens out.

---

## 5. Transformer blocks 2 through 12

The same diagram runs **11 more times**, with different learned weights each time.

The three vectors keep the same shape. Their contents get richer:

```text
after block  1   [ 0.64, -0.52,  0.17,  ... ]
                 [ 0.28, -0.18, -1.21,  ... ]
                 [-1.33,  0.26, -0.70,  ... ]

after block  2   [ 0.21, -1.31,  0.30,  ... ]
                 [ 0.75,  0.21, -1.50,  ... ]
                 [-1.54,  0.97, -0.84,  ... ]

                 ...

after block 12   [-0.65, -0.44, -1.17,  ... ]     768 numbers   "Cat"
                 [-1.74, -3.05, -2.62,  ... ]     768 numbers   " ate"
                 [ 2.30,  3.21,  1.81,  ... ]     768 numbers   " the"
```

Early blocks tend to move local / surface patterns. Later blocks tend to move more abstract ones. By the last block, the third row is no longer just “the word *the* in position 2”. It is a representation of this whole prefix, with the fact that a cat ate (roughly) some object.

---

## 6. Final layer normalization

One more normalization, using a scale and offset that sit outside the blocks:

```text
[-0.65, -0.44, -1.17,  ... ]
[-1.74, -3.05, -2.62,  ... ]
[ 2.30,  3.21,  1.81,  ... ]
        ↓  final layer norm
[-0.12, -0.06, -0.30,  ... ]     768 numbers
[-0.15, -0.23, -0.38,  ... ]     768 numbers
[ 0.20,  0.31,  0.15,  ... ]     768 numbers
```

---

## 7. Only the last token is used to pick the next word

To guess what comes *after* `"Cat ate the"`, the algorithm needs only the last token. The previous tokens helped build it, through attention, but they are not scored against the vocabulary.

```text
[ 0.20,  0.31,  0.15,  ... ]     768 numbers   representation of " the" in this sentence
```

That vector is compared with **every row** of the same embedding table used at the start (50,257 × 768). Each comparison is one number, called a **logit**.

The table shows the five largest logits, the three prompt tokens, and the first and last vocabulary entries. A larger logit is a better fit. These scores are negative, so `-82.70` beats `-82.73`.

```text
token id     token              logit
     0       "!"                -90.91
   262       " the"             -86.07
   976       " same"            -82.77
  2057       " food"            -82.70     ← largest
  2187       " whole"           -82.74
  5935       " egg"             -82.73
  6174       " meat"            -82.91
 15063       " ate"             -89.71
 21979       "Cat"              -94.13
 50256       "<|endoftext|>"    -91.35
```
50,257 scores total.

This algorithm takes the **largest logit**. The nearest rivals are `" egg"`, `" whole"`, `" same"`, and `" meat"`, a few hundredths lower.

```text
largest logit -82.70  →  2057
        ↓
decode 2057  →  " food"
```

---

## 8. The prediction

```text
Cat ate the  +  " food"
        ↓
Cat ate the food
```

If we wanted another token, we would append `2057` to `[21979, 15063, 262]` and run the whole forward pass again on four tokens. This walkthrough stops after one.

---

## Shapes, one more time

```text
"Cat ate the"
        ↓ tokenize
3 token ids                         21979, 15063, 262
        ↓ embed + position
3 × 768
        ↓ 12 × (norm → attention → add → norm → feed-forward → add)
3 × 768
        ↓ final layer norm
3 × 768
        ↓ keep last row
768
        ↓ compare with 50,257 embeddings
50,257 logits
        ↓ argmax
1 token id                          2057
        ↓ decode
" food"
```
