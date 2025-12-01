#!/usr/bin/env python3
import random
from pathlib import Path

# CHANGE: using your real base directory
BASE = Path("/mimer/NOBACKUP/groups/dionysus/cleland/stanza").resolve()

SVENSKA_PROJEKT = BASE / "svenska-projekt"
DIACHRONIC_DEV = BASE / "ud/UD_Swedish-diachronic/sv_diachronic-ud-dev.conllu"
DIACHRONIC_TRAIN = BASE / "ud/UD_Swedish-diachronic/sv_diachronic-ud-train.conllu"

OUTPUT_TRAIN = BASE / "combined-train.conllu"
OUTPUT_DEV = BASE / "combined-dev.conllu"

# CHANGE: deterministic randomness for reproducibility
random.seed(1337)


def read_conllu(path: Path):
    """Read a CoNLL-U file and return a list of sentences (blocks)."""
    with path.open("r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    # Sentences are separated by blank lines
    return content.split("\n\n")


# --- Collect all sentences ---

all_sentences = []

# CHANGE: include all .conllu inside svenska-projekt/
for conllu_file in SVENSKA_PROJEKT.rglob("*.conllu"):
    print(f"Reading {conllu_file}")
    all_sentences.extend(read_conllu(conllu_file))

# Include diachronic files explicitly
print(f"Reading {DIACHRONIC_DEV}")
all_sentences.extend(read_conllu(DIACHRONIC_DEV))

print(f"Reading {DIACHRONIC_TRAIN}")
all_sentences.extend(read_conllu(DIACHRONIC_TRAIN))

print(f"Total sentences loaded: {len(all_sentences)}")

# --- Shuffle & split ---

random.shuffle(all_sentences)  # CHANGE: shuffle sentences blindly

n = len(all_sentences)
split_point = int(n * 0.90)

train_sents = all_sentences[:split_point]
dev_sents = all_sentences[split_point:]

print(f"Train: {len(train_sents)} sentences")
print(f"Dev:   {len(dev_sents)} sentences")


# --- Write output files ---
def write_conllu(path: Path, sentences):
    with path.open("w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s.strip() + "\n\n")

print(f"Writing {OUTPUT_TRAIN}")
write_conllu(OUTPUT_TRAIN, train_sents)

print(f"Writing {OUTPUT_DEV}")
write_conllu(OUTPUT_DEV, dev_sents)

print("Done.")