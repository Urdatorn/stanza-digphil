#!/usr/bin/env bash
set -euo pipefail

# ================================
# USAGE:
#   ./make_new_model_BERT_SILVER.sh
# ================================

LANGCODES=("sv")

# ========================================
# 0. SET UP LOGGING
# ========================================

mkdir -p logs

timestamp=$(date +"%Y%m%d_%H%M%S")

# Join language codes: sv_nn_da
LANG_JOINED=$(printf "_%s" "${LANGCODES[@]}")
LANG_JOINED="${LANG_JOINED:1}"

# Build log filename
LOGFILE="logs/log_bert-base-swedish-cased_${LANG_JOINED}_${timestamp}.txt"

# Redirect output to tee
exec > >(tee -a "$LOGFILE") 2>&1

echo "=== LOGFILE: $LOGFILE ==="
echo "Language codes: ${LANGCODES[*]}"
echo "Using pretrained model: KBLab/bert-base-swedish-cased"
echo

# ========================================
# 1. PREPARE TRAIN/VAL/TEST SPLITS
# ========================================
echo "Running: python prepare-train-val-test-silver.py ${LANGCODES[*]}"
python prepare-train-val-test-silver.py "${LANGCODES[@]}"

# ========================================
# 2. LOAD CONFIG
# ========================================
echo "Sourcing scripts/config.sh"
source scripts/config.sh

# ========================================
# 3. PREPARE STANZA DATASET
# ========================================
echo "Running stanza dataset preparation…"
python -m stanza.utils.datasets.prepare_depparse_treebank UD_Swedish-diachronic \
    --gold

# ========================================
# 4. TRAIN THE DEPENDENCY PARSER
# ========================================
echo "Running stanza dependency parser training…"
python -m stanza.utils.training.run_depparse UD_Swedish-diachronic \
    --batch_size 32 \
    --dropout 0.33 \
    --use_bert \
    --bert_model KBLab/bert-base-swedish-cased \
    --device xpu:0
#    --silver_file "ud/UD_Swedish-diachronic/sv_diachronic-ud-train-silver.conllu" \

echo "DONE."
echo "Full log saved to: $LOGFILE"

# ========================================
# 5. UPDATE 'latest.txt' SYMLINK
# ========================================
ln -sf "$(basename "$LOGFILE")" logs/latest.txt
echo "Symlink updated: logs/latest.txt → $(basename "$LOGFILE")"

# ========================================
# 5. PLOT LOSS AND LAS
# ========================================
python loss.py