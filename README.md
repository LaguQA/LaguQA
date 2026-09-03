# LaguQA

One printed songbook, 107 Indonesian national and regional songs, every melody
transcribed by hand into ABC 2.1 notation. From those transcriptions come 1,200
multiple-choice questions with five options each, and every answer key traces
back to a specific page of the book.

The questions ask what the book actually prints: composer, region of origin,
key, time signature, tempo marking, lyrics, and the melody. Some show a fragment
of number notation and ask which song it belongs to. Others ask how many bars a
song has, or which note is its highest. No audio is used anywhere.

Number notation is how music is taught in most Indonesian schools. No benchmark
tests whether a language model can read it.

Part of an undergraduate thesis in Informatics, Universitas Ahmad Dahlan.

[Baca dalam bahasa Indonesia](README.id.md)

| | |
|---|---|
| Website | [laguqa.github.io](https://laguqa.github.io) |
| Code | [github.com/IRedDragonICY/LaguQA](https://github.com/IRedDragonICY/LaguQA) |
| Demo | [IRedDragonICY/LaguQA-Demo](https://huggingface.co/spaces/IRedDragonICY/LaguQA-Demo) |
| Dataset | [IRedDragonICY/LaguQA](https://huggingface.co/datasets/IRedDragonICY/LaguQA) |
| Dataset mirror | [kaggle.com/datasets/ireddragonicy/laguqa](https://www.kaggle.com/datasets/ireddragonicy/laguqa) |
| Model | [IRedDragonICY/LaguQA-Gemma4-E2B](https://huggingface.co/IRedDragonICY/LaguQA-Gemma4-E2B) |
| Everything on Hugging Face | [LaguQA collection](https://huggingface.co/collections/IRedDragonICY/laguqa-6a9826460786a111107e430e) |

## Results

Thirteen models were scored zero-shot on all 1,200 questions. Not one of them
beat a guesser that has never heard of any of these songs.

| | Accuracy (%) |
|---|---|
| Guesser that follows the answer-key distribution | 32.1 |
| Best untrained model, sahabatai-9b | 29.5 |
| gemma4-e2b, untrained | 24.1 |
| gemma4-e2b after LoRA, three seeds | 52.3 to 61.0 |
| Random guessing | 17.2 |

The floor here is the distribution-following guesser, not random guessing. Of
the time-signature answers, 70.2% are 4/4, and of the key answers, 70.9% are
Do = C. Memorise those two facts, know nothing else, and you already have 32.1%.
Random guessing sits at 17.2%, and using that as the floor would make every
model look like it knows something.

Fine-tuning moves gemma4-e2b into the fifties and sixties. Those three numbers
come from one recipe run three times with different seeds and nothing else
changed: same training file by sha256, same learning rate, same 2,622 steps.
The spread is 8.7 points. That is wider than most of the gaps people report
between one hyperparameter setting and another, so a single run proves very
little.

Per-category scores for every model are in
[`docs/tabel/mc/papan-skor.md`](docs/tabel/mc/papan-skor.md).

## What is in here

| Folder | Contents |
|---|---|
| `src/laguqa/scans/` | preparing the scanned book pages |
| `src/laguqa/notation/` | ABC 2.1 validator and converters |
| `src/laguqa/dataset/` | assembling the song table |
| `src/laguqa/benchmark/` | question generation, scoring, control baselines |
| `src/laguqa/report/` | leaderboards, tables, result charts |
| `scripts/` | the pipeline, numbered in the order it runs |
| `hasil/` | manifest and per-question answers for every run |
| `tests/` | test suite |
| `modal_train.py` | LoRA training on Modal |

Three things live elsewhere. The song data is derived from a copyrighted book
and ships under its own licence through Hugging Face and Kaggle. The adapter
weights are hundreds of megabytes per run and ship through Hugging Face. The
thesis manuscript and the program that typesets it are not research code and
stay out of version control entirely.

## Reproducing

```bash
pip install -e .
```

Notation handling and question generation use nothing outside the standard
library, on purpose, so the benchmark can be rebuilt without a build toolchain.
Training and the extra evaluations need `modal`, `matplotlib`, and `pandas`.

Get the data first:

```bash
hf download IRedDragonICY/LaguQA --repo-type dataset --local-dir data
```

Regenerate the questions from the song table:

```bash
python scripts/10_generate_benchmark.py
```

Score a model on the multiple-choice track:

```bash
python scripts/22_evaluate_mc.py
```

Train a LoRA adapter:

```bash
python modal_train.py
```

Rebuild the leaderboard and charts from whatever is in `hasil/`:

```bash
python scripts/19_leaderboard.py && python scripts/31_charts.py
```

Scripts run in numeric order, from preparing scans to publishing releases. Each
one takes `--help`.

## How answers are scored

The model is never asked to type a letter. For each question, all five option
texts are appended to the prompt one at a time, the mean log-probability of the
option's tokens is computed, and the highest one counts as the answer.

Two other methods were tried first and both misled. Scoring the generated text
punishes models that answer at length: two of the comparison models wrote open
reasoning until they ran out of tokens without ever committing to an option.
Scoring the probability of a single letter token measures which letter a model
likes, which turns out to have little to do with whether it knows the song.

## Licence

The licence for this code has not been decided, so copyright remains with the
author. The data is released separately under CC BY-NC 4.0 because it derives
from a printed book; the copyright notice travels with the dataset files.

## Citation

```bibtex
@misc{hendianto2026laguqa,
  author       = {Hendianto, Mohammad Farid},
  title        = {{LaguQA}: A Benchmark for Indonesian National and Regional
                  Song Understanding in Large Language Models},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/IRedDragonICY/LaguQA}},
  note         = {Undergraduate thesis, Informatics,
                  Universitas Ahmad Dahlan}
}
```
