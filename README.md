# Customer Support Ticket Intelligence

Predicting which department (`queue`) and how urgent (`priority`) a support ticket is, purely from its text, plus a semantic search feature that surfaces how similar past tickets were actually resolved.

Built from first principles using TF-IDF and classical ML, with an evidence-based comparison against pretrained embeddings to check whether the extra complexity was worth it. It wasn't, for this dataset - details below.

## Dataset

Around 21,000 English support tickets, sourced from Kaggle and combined from two files, deduplicated and cleaned. Two prediction targets: `queue` (10-class department routing, meaningfully imbalanced) and `priority` (3-class urgency, fairly balanced). A chi-square test confirmed queue and priority are not independent.

The dataset's `language` label turned out to be unreliable - roughly 26% of rows tagged as German were actually English text. This was caught by running `langdetect` across the full dataset rather than trusting the label or a sample.

## Exploratory data analysis

Before any modeling, the cleaned dataset (about 21,000 rows, zero duplicate ticket bodies, nulls handled) was explored on its own terms:

- **Ticket length and word frequency** across the corpus, to get a feel for how long tickets typically run and which words dominate before any cleaning-driven assumptions were made.
- **`queue` class distribution**: meaningfully imbalanced, ranging from 287 to 6,231 rows per class, which is why macro-F1 was chosen over accuracy for evaluation later on.
- **Chi-square testing between `queue` and `priority`**, confirming the two are not independent. Certain queues (Service Outages, Technical Support) skew heavily toward high priority, while others (General Inquiry, HR) skew low. This directly shaped a modeling decision: `priority`'s model does not use `queue` as an input feature, since predicting one target from another would be circular for a brand-new ticket.
- **A side check on boilerplate language**: whether words like "dear," "customer," and "support" carried any real signal across queues. Chi-square testing showed these words are statistically associated with certain queues, but the effect is practically flat, an early example of a pattern that reappeared later with the `type` feature's contribution to the final models.

## Text cleaning: a bug worth naming

**Negation bug**: the default English stopword list strips words like `not`, `no`, and contractions such as `isn't` and `won't`, which silently reversed ticket meaning - "software is not running" became "software running" after cleaning. This affected 12.3% of tickets and was fixed by excluding negation terms from the stopword set.

The pattern behind this, and behind the `language` label issue above, is the same: generic NLP tooling makes assumptions that do not always hold for a specific dataset, and the only way to know is to check rather than assume.

## Modeling

Features: TF-IDF (unigrams and bigrams) over cleaned ticket text, combined with a one-hot encoded `type` field, inside a single leakage-safe pipeline - the vectorizer and encoder are fit only on the training split.

Seven models were compared per target using 5-fold cross-validation, scored on macro-F1 rather than accuracy, since accuracy is misleading given `queue`'s roughly 20x class imbalance. Random Forest won clearly for both targets and was tuned further via randomized search.

| Target | Final macro-F1 (holdout) | Beats dummy baseline by |
|---|---|---|
| `queue` | 0.687 | ~15x |
| `priority` | 0.701 | ~3.7x |

Random Forest outperformed the linear models here in part because features (about 8,000) do not vastly outnumber training rows (about 14,700), a friendlier ratio for trees than the textbook sparse-TF-IDF case. It also lines up with what the data looks like: queue-routing signal is often a single, highly discriminative keyword ("billing," "outage"), which a tree captures cleanly in one split, while a linear model has to spread weight across many overlapping, correlated features to represent the same thing.

## Semantic search

Cosine similarity over TF-IDF vectors retrieves the most similar past tickets along with how they were actually resolved, useful for an agent asking how something like this was handled before.

A pretrained embedding approach (spaCy's GloVe-based vectors) was tested against this TF-IDF baseline on queries deliberately written to avoid literal word overlap with the tickets they should match. TF-IDF won clearly. Naive averaging of word vectors has no equivalent to TF-IDF's built-in downweighting of common words, so boilerplate language ("dear," "please," "could") ends up diluting the signal from the few words that actually matter. TF-IDF remains the method used for retrieval.

## Try it

A Streamlit app (app.py) wraps the trained pipeline: enter a subject, description, and ticket type, and it predicts the queue and priority, then surfaces similar past tickets and how they were resolved.

Not currently hosted: the tuned Random Forest models are large (200 unconstrained trees over ~8,000 features), and pushing them past GitHub's file size limit and Streamlit Cloud's free-tier memory ceiling would need either a lighter model configuration or Git LFS. The app runs locally, see Setup below.

## Project structure

```
01_EDA.ipynb                          - Combine, clean, verify, explore
02_Modeling_and_prediction.ipynb      - Features, model comparison, tuning, semantic search
app.py                                - Streamlit app
requirements.txt                      - Dependencies for the app
*.pkl                                 - Fitted models, encoders, and search index, exported from the modeling notebook
```

## Setup

To run the notebooks:

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords punkt wordnet omw-1.4
```

To run the app locally, once the `.pkl` files exist in the same folder:

```bash
streamlit run app.py
```
