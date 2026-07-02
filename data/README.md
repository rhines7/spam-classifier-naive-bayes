# SpamAssassin dataset setup

The raw email corpus is **not** included in this repository (~9,350 files).

## Expected layout

After setup, this directory should contain:

```
data/spamassassin/
├── easy_ham/
├── easy_ham_2/
├── hard_ham/
├── spam/
└── spam_2/
```

## Option 1: Symlink from a local copy (recommended for development)

If you already have the corpus in the parent course folder:

**Windows (Command Prompt, run from repo root):**

```cmd
mklink /J data\spamassassin "..\SpamAssassinMessages"
```

**macOS / Linux:**

```bash
ln -s ../SpamAssassinMessages data/spamassassin
```

## Option 2: Download and extract

1. Obtain the SpamAssassin public corpus ([SpamAssassin corpus sources](https://spamassassin.apache.org/old/publiccorpus/)).
2. Extract so the five category folders listed above live under `data/spamassassin/`.

## Verify

From the repo root:

```bash
python -c "from pathlib import Path; p=Path('data/spamassassin'); print('OK' if p.is_dir() and any(p.iterdir()) else 'Missing data')"
```

Then run the pipeline:

```bash
python spam_classifier.py
```
