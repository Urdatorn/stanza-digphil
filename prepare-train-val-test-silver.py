#!/usr/bin/env python3

'''
Accepted language codes:
sv, diachron, bm, nn, dk, de_lit, is

Example arg:
python prepare-train-val-test.py sv diachron bm nn dk de_lit
'''

from collections import defaultdict
from pathlib import Path
import random
import sys

random.seed(1337)

# ============================================================
# SETTINGS
# ============================================================

USE_GOLD_IN_TRAIN = True

ud_treebank_groups_used = sys.argv[1:]

AT_RISK_DEPRELS = {
    "acl",
    "advcl",
    "ccomp",
    "appos",
    "iobj",
    "parataxis",
    "nummod",
    "flat:name",
}

# ============================================================
# BASE PATHS
# ============================================================
BASE = Path.cwd()

SVENSKA_PROJEKT = BASE / "ud-treebanks-sv"
NORSKA_PROJEKT  = BASE / "ud-treebanks-bm"
NYNORSKA_PROJEKT = BASE / "ud-treebanks-nn"
DANSKA_PROJEKT  = BASE / "ud-treebanks-dk"
TYSKA_PROJEKT   = BASE / "ud-treebanks-de_lit"
ICELANDIC_PROJEKT = BASE / "ud-treebanks-is"

DIGPHIL_MACHINE = Path("/home/urdatorn/git/Diachronic_Treebanks_DigPhil/parsed_data/stanza_results/stanza_results_non_preprocessed_norm")
DIGPHIL_GOLD    = Path("/home/urdatorn/git/Diachronic_Treebanks_DigPhil/parsed_data/validated")

# ============================================================
# Redirect outputs to UD_Swedish-diachronic/
# ============================================================
OUTPUT_DIR = BASE / "ud" / "UD_Swedish-diachronic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_TRAIN = OUTPUT_DIR / "sv_diachronic-ud-train.conllu"
OUTPUT_DEV   = OUTPUT_DIR / "sv_diachronic-ud-dev.conllu"
OUTPUT_TEST  = OUTPUT_DIR / "sv_diachronic-ud-test.conllu"  # Copy of dev (required by Stanza)

OUTPUT_TRAIN_SILVER = OUTPUT_DIR / "sv_diachronic-ud-train-silver.conllu"
OUTPUT_GOLD_AS_VAL  = Path("gold_as_val.conllu")

# ============================================================
# BASIC HELPERS
# ============================================================
def read_conllu(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    return [] if not text else text.split("\n\n")

def extract_sent_id(block: str) -> str | None:
    for line in block.split("\n"):
        if line.startswith("# sent_id"):
            parts = line.split("=", 1)
            if len(parts) == 2:
                return parts[1].strip()
            return line.split("# sent_id", 1)[1].strip()
    return None

def write_conllu(path: Path, sentences):
    with path.open("w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s.strip() + "\n\n")

def load_from_treebank_dir(directory: Path):
    collected = []
    for path in directory.rglob("*.conllu"):
        print(f"Reading: {path}")
        collected.extend(read_conllu(path))
    return collected

def sentence_deprels(block: str) -> set[str]:
    rels = set()
    for line in block.split("\n"):
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) == 10:
            deprel = fields[7]
            if deprel and deprel != "_":
                rels.add(deprel)
    return rels
# ============================================================
# CoNLL-U VALIDATOR
# ============================================================
class CoNLLUValidator:
    def __init__(self):
        self.errors = []

    def validate_sentence(self, sentence_lines, sent_id=None):
        self.errors = []
        
        if not sentence_lines:
            self.errors.append("Empty sentence")
            return False
        
        tokens = []
        roots = []
        token_ids = set()
        
        for line_num, line in enumerate(sentence_lines, 1):
            try:
                fields = line.split('\t')
                if len(fields) != 10:
                    self.errors.append(f"Line {line_num}: Expected 10 fields, got {len(fields)}")
                    continue
                
                token_id, form, lemma, upos, xpos, feats, head, deprel, deps, misc = fields
                
                if '-' in token_id or '.' in token_id:
                    continue
                
                try:
                    token_id_int = int(token_id)
                    head_int = int(head)
                except ValueError:
                    self.errors.append(f"Line {line_num}: Invalid token ID or head")
                    continue
                
                token_ids.add(token_id_int)
                
                if head_int == 0:
                    roots.append(token_id_int)
                
                tokens.append({
                    'id': token_id_int,
                    'form': form,
                    'lemma': lemma,
                    'upos': upos,
                    'head': head_int,
                    'deprel': deprel
                })
            except Exception as e:
                self.errors.append(f"Line {line_num}: Error: {e}")
        
        if len(roots) == 0:
            self.errors.append("No root found")
        elif len(roots) > 1:
            self.errors.append(f"Multiple roots found: {roots}")
        
        for token in tokens:
            if token['head'] != 0 and token['head'] not in token_ids:
                self.errors.append(f"Token {token['id']} has invalid head {token['head']}")
        
        if not self._check_no_cycles(tokens):
            self.errors.append("Dependency cycle detected")
        
        for token in tokens:
            if not token['form'] or token['form'] == '_':
                self.errors.append(f"Token {token['id']}: Missing form")
            if not token['upos'] or token['upos'] == '_':
                self.errors.append(f"Token {token['id']}: Missing UPOS")
            if not token['deprel'] or token['deprel'] == '_':
                self.errors.append(f"Token {token['id']}: Missing deprel")
        
        return len(self.errors) == 0

    def _check_no_cycles(self, tokens):
        heads = {t['id']: t['head'] for t in tokens}
        for start in tokens:
            visited = set()
            current = start['id']
            while current != 0 and current in heads:
                if current in visited:
                    return False
                visited.add(current)
                current = heads[current]
        return True

    def get_errors(self):
        return self.errors


# ============================================================
# CLEANING PIPELINE
# ============================================================
def clean_sentences(sentence_blocks):
    validator = CoNLLUValidator()
    cleaned = []

    for block in sentence_blocks:
        lines = [l for l in block.split("\n") if not l.startswith("#")]
        comments = [l for l in block.split("\n") if l.startswith("#")]
        sent_id = None
        for c in comments:
            if c.startswith("# sent_id"):
                sent_id = c.split("=", 1)[1].strip() if "=" in c else None

        if validator.validate_sentence(lines, sent_id):
            cleaned.append(block)
        else:
            print(f"[REMOVED] sent_id={sent_id}  ERRORS={validator.get_errors()}")

    return cleaned

# ============================================================
# Load only requested treebanks
# ============================================================
train_sentences = []

if "sv" in ud_treebank_groups_used:
    train_sentences.extend(load_from_treebank_dir(SVENSKA_PROJEKT))

if "bm" in ud_treebank_groups_used:
    train_sentences.extend(load_from_treebank_dir(NORSKA_PROJEKT))

if "nn" in ud_treebank_groups_used:
    train_sentences.extend(load_from_treebank_dir(NYNORSKA_PROJEKT))

if "dk" in ud_treebank_groups_used:
    train_sentences.extend(load_from_treebank_dir(DANSKA_PROJEKT))

if "is" in ud_treebank_groups_used:
    train_sentences.extend(load_from_treebank_dir(ICELANDIC_PROJEKT))

if "de_lit" in ud_treebank_groups_used:
    train_sentences.extend(load_from_treebank_dir(TYSKA_PROJEKT))


# ============================================================
# DigPhil machine ALWAYS added to SILVER (minus gold overlap)
# ============================================================
def map_sent_ids_by_file(directory: Path):
    mapping = {}
    for path in directory.glob("*.conllu"):
        blocks = read_conllu(path)
        ids = {extract_sent_id(b) for b in blocks if extract_sent_id(b)}
        mapping[path.name] = ids
    return mapping

gold_ids = map_sent_ids_by_file(DIGPHIL_GOLD)

silver_sentences = []
print("Loading DigPhil MACHINE for SILVER (minus gold overlap)…")
for machine_file in DIGPHIL_MACHINE.glob("*.conllu"):
    blocks = read_conllu(machine_file)
    filename = machine_file.name
    gold_for_this = gold_ids.get(filename, set())

    for block in blocks:
        sid = extract_sent_id(block)
        if sid and sid in gold_for_this:
            continue
        silver_sentences.append(block)


# ============================================================
# DigPhil gold: 10% for val, 90% for test
# Optionally include gold test portion in train
# ============================================================
gold_sentences = []
for gold_file in DIGPHIL_GOLD.glob("*.conllu"):
    print(f"Reading GOLD: {gold_file}")
    gold_sentences.extend(read_conllu(gold_file))

# Gold will be split into val (10%) and test (90%) - nothing goes to train

# STRATIFIED DEV SPLIT (to prevent zeroing of deprels)

random.shuffle(gold_sentences)

TARGET_DEV_SIZE = max(1, int(len(gold_sentences) * 0.10))
MIN_PER_DEPREL = 5   # small but sufficient

dev_sentences = []
used = set()

# index sentences by deprel
by_deprel = defaultdict(list)
for i, sent in enumerate(gold_sentences):
    for d in sentence_deprels(sent):
        if d in AT_RISK_DEPRELS:
            by_deprel[d].append(i)

# ensure coverage
for d in AT_RISK_DEPRELS:
    candidates = by_deprel.get(d, [])
    random.shuffle(candidates)
    for idx in candidates[:MIN_PER_DEPREL]:
        if idx not in used:
            dev_sentences.append(gold_sentences[idx])
            used.add(idx)

# fill remainder randomly
for i, sent in enumerate(gold_sentences):
    if len(dev_sentences) >= TARGET_DEV_SIZE:
        break
    if i not in used:
        dev_sentences.append(sent)
        used.add(i)

# Use remaining gold sentences (90%) as test set
test_sentences = [
    sent for i, sent in enumerate(gold_sentences)
    if i not in used
]
print(f"Using {len(test_sentences)} remaining gold sentences as TEST (90% of gold)")

if USE_GOLD_IN_TRAIN:
    train_sentences.extend(test_sentences)
    print(f"USE_GOLD_IN_TRAIN=True: Added {len(test_sentences)} gold TEST sentences to TRAIN")

# ============================================================
# CLEAN ALL OUTPUTS
# ============================================================
print("Cleaning TRAIN...")
train_sentences = clean_sentences(train_sentences)

print("Cleaning SILVER...")
silver_sentences = clean_sentences(silver_sentences)

print("Cleaning DEV...")
dev_sentences = clean_sentences(dev_sentences)

print("Cleaning TEST...")
test_sentences = clean_sentences(test_sentences)


# ============================================================
# WRITE FINAL OUTPUTS (now inside UD_Swedish-diachronic/)
# ============================================================
print(f"Writing TRAIN → {OUTPUT_TRAIN} ({len(train_sentences)} valid sentences)")
write_conllu(OUTPUT_TRAIN, train_sentences)

print(f"Writing SILVER → {OUTPUT_TRAIN_SILVER} ({len(silver_sentences)} valid sentences)")
write_conllu(OUTPUT_TRAIN_SILVER, silver_sentences)

print(f"Writing DEV → {OUTPUT_DEV} ({len(dev_sentences)} valid sentences)")
write_conllu(OUTPUT_DEV, dev_sentences)

print(f"Writing GOLD_AS_VAL → {OUTPUT_GOLD_AS_VAL} ({len(dev_sentences)} sentences, for logging)")
write_conllu(OUTPUT_GOLD_AS_VAL, dev_sentences)

print(f"Writing TEST → {OUTPUT_TEST} (90% DIGPHIL_GOLD, {len(test_sentences)} valid sentences)")
write_conllu(OUTPUT_TEST, test_sentences)

print("Done.")