# Abstractive Text Summarization using Custom BERT Encoder-Decoder (From Scratch)

An end-to-end PyTorch implementation of an **Encoder-Decoder Transformer (`Seq2SeqBert`) built from scratch**—without any pretrained weights—for abstractive news summarization on the **EdinburghNLP/XSum** dataset.

---

## Project Overview

This project builds every core component of a BERT-style sequence-to-sequence Transformer from the ground up:
1. **Dataset Exploration & Preprocessing:** Loaded `EdinburghNLP/xsum` and analyzed token/word distributions across articles and one-sentence summaries.
2. **Custom WordPiece Tokenizer:** Trained a domain-specific `BertWordPieceTokenizer` (8,000 vocabulary size) from scratch on 50,000 BBC news articles and summaries.
3. **Transformer Architecture from Scratch:** Implemented token + positional embeddings, scaled dot-product multi-head self-attention, causal masking, encoder-decoder cross-attention, GELU feed-forward layers, residual connections, and layer normalization in PyTorch (`13.66M` trainable parameters).
4. **Training & Validation:** Trained for 5 epochs on a T4 GPU using `AdamW`, gradient clipping, and a linear warmup-and-decay learning rate schedule, checkpointing the lowest-validation-loss weights (`best_model.pt`).
5. **Evaluation & Visualization:** Evaluated qualitative predictions and quantitative **ROUGE** metrics on unseen test data, and plotted training/validation loss and perplexity curves.

---

## Dataset & Preprocessing

* **Dataset:** [`EdinburghNLP/xsum`](https://huggingface.co/datasets/EdinburghNLP/xsum) (BBC news articles paired with single-sentence abstractive summaries)
  * Full splits: `204,045` train / `11,332` validation / `11,334` test
  * Active subsets used: `50,000` train / `2,000` validation / `2,000` test
* **Text Length Statistics (5,000-sample analysis):**
  * **Documents:** Average `375.6` words | 95th percentile `920.0` words | Max `4,281` words
  * **Summaries:** Average `21.2` words | 95th percentile `30.0` words | Max `55` words
* **Tokenization & Sequence Limits:**
  * Custom `BertWordPieceTokenizer` (`vocab_size = 8000`, `min_frequency = 2`, lowercase, special tokens `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`)
  * `SRC_LEN = 256` tokens (source document cap with `[CLS]` and `[SEP]`)
  * `TGT_LEN = 48` tokens (target summary cap with `[CLS]` and `[SEP]`)

---

## Model Architecture (`Seq2SeqBert`)

| Component | Configuration / Details | Parameters |
| :--- | :--- | :---: |
| **Hidden Dimension (`hidden_size`)** | `256` | — |
| **Attention Heads (`num_heads`)** | `4` (`head_dim = 64`) | — |
| **Layers (`num_layers`)** | `4` Encoder layers + `4` Decoder layers | — |
| **Feed-Forward Inner Size (`ffn_size`)** | `1,024` (GELU activation, `dropout = 0.1`) | — |
| **Embeddings (`BertEmbeddings`)** | Token (`8000 × 256`) + Learned Positional (`256 × 256`) | `2,114,048` |
| **Encoder (`BertEncoder`)** | 4 × (`MultiHeadAttention` + `FeedForward` + `LayerNorm`) | `5,273,088` |
| **Decoder (`BertDecoder`)** | 4 × (Causal Self-Attn + Cross-Attn + `FeedForward` + `LayerNorm`) | `6,327,808` |
| **LM Head (`nn.Linear`)** | Projects `256 → 8000` vocabulary logits | `2,056,000` |
| **Total Trainable Parameters** | Initialized from scratch ($\mathcal{N}(0, 0.02)$) | **`13,656,896`** |

---

## Training Setup & Progression

* **Hardware:** NVIDIA Tesla T4 GPU (Google Colab)
* **Optimizer & Loss:** `AdamW` (`lr = 3e-4`, `weight_decay = 0.01`), `CrossEntropyLoss(ignore_index=-100)`, gradient norm clipping (`1.0`)
* **LR Schedule:** Linear warmup for `500` steps followed by linear decay over `7,815` total steps (`5 epochs × 1,563 steps/epoch`, `batch_size = 32`)

### Epoch-by-Epoch Results

| Epoch | Train Loss | Validation Loss | Validation Perplexity ($e^{\text{val\_loss}}$) | Checkpoint |
| :---: | :---: | :---: | :---: | :--- |
| **Before Training** | — | `9.038` | `~8,417.0` | Random baseline ($\ln 8000 \approx 8.99$) |
| **Epoch 1** | `5.691` | `4.543` | `94.0` | Saved `best_model.pt` |
| **Epoch 2** | `4.239` | `4.043` | `57.0` | Saved `best_model.pt` |
| **Epoch 3** | `3.802` | `3.813` | `45.3` | Saved `best_model.pt` |
| **Epoch 4** | `3.532` | `3.709` | `40.8` | Saved `best_model.pt` |
| **Epoch 5** | **`3.357`** | **`3.671`** | **`39.3`** | **Saved `best_model.pt` (Best)** |

---

## Evaluation Results (Test Set)

Using autoregressive greedy decoding from `best_model.pt` on unseen test articles:

| Metric | Score | Interpretation |
| :--- | :---: | :--- |
| **ROUGE-1** | **`0.2260`** | Strong unigram overlap for a compact Transformer trained from scratch without pretraining |
| **ROUGE-2** | **`0.0509`** | Reflects abstractive paraphrasing rather than extractive n-gram copying |
| **ROUGE-L** | **`0.1818`** | Confirms fluent, BBC-style single-sentence summary structure |
| **ROUGE-Lsum** | **`0.1813`** | Consistent sentence-level structural alignment |

### Sample Predictions

* **Example 1**
  * **Reference:** *There is a "chronic" need for more housing for prison leavers in Wales, according to a charity.*
  * **Predicted:** *more than 1, 000 people have been criticised for failing to help people who have been put up in prison for more than a year, the welsh government has said.*
* **Example 2**
  * **Reference:** *A man has appeared in court after firearms, ammunition and cash were seized by police in Edinburgh.*
  * **Predicted:** *a man has been charged with the attempted murder of a man in edinburgh.*

---

## Key Takeaways

1. **Smooth Convergence Without Overfitting:** Both training loss (`5.691 → 3.357`) and validation loss (`4.543 → 3.671`) decreased consistently across all 5 epochs, reducing validation perplexity from `8,417` (random guessing) down to `39.3`.
2. **Syntactic & Domain Grounding:** Despite having no pretrained weights, the custom `Seq2SeqBert` architecture learned to generate grammatical, one-sentence BBC-style leads grounded in the article's core topic and location (e.g., Wales/prison, Edinburgh/court).
3. **Pretraining & Decoding Trade-offs:** Training from scratch on an 8,000-word vocabulary limits factual precision on rare named entities, and greedy (`argmax`) decoding occasionally introduces phrase repetition—both of which can be improved with beam search decoding or pretrained backbones.

---

## Repository Structure & Quickstart

```text
├── Summarization_using_BERT_model_assignment-3.ipynb   # Main Colab/Jupyter notebook
├── tokenizer/
│   └── vocab.txt                                       # Trained 8,000-token WordPiece vocabulary
├── best_model.pt                                       # Saved best checkpoint (Epoch 5)
└── README.md                                           # Project documentation
```

### Installation & Running

```bash
pip install torch transformers datasets evaluate rouge_score tokenizers matplotlib numpy
```

Open `Summarization_using_BERT_model_assignment-3.ipynb` in Jupyter or Google Colab (GPU runtime recommended) and run all cells sequentially.
