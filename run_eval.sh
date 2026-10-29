#!/usr/bin/env bash
# run_eval.sh
# Predicts with both models (best + current) across all decode-mode / threshold
# configs and writes RESULT.md.
#
# Usage:
#   bash run_eval.sh
#
# Requires:
#   - torchrun (PyTorch ≥1.9)
#   - evaluate.py, compute-necer.sh  (repo root)
#   - data_to_upload/aishell_dataset/test-ne.{json,text}
#   - exp/copyne_aishell_beta1/{best,current}.model
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── paths ─────────────────────────────────────────────────────────────────────
EXP_DIR="exp/copyne_aishell_beta1"
CHAR_DICT="data_to_upload/aishell_vocab/char.vocab"
TEST_NE_DICT="data_to_upload/aishell_vocab/test_ne.vocab"
INPUT="data_to_upload/aishell_dataset/test-ne.json"
# compute-necer.sh derives the json sibling from the text path automatically
GOLD_TEXT="data_to_upload/aishell_dataset/test-ne.text"
CONFIG="conf/copyne.yaml"
CMVN="data_to_upload/aishell1_global_cmvn_mel80"
BATCH=16
BEAM=10
RESULTS_DIR="results"
RESULT_MD="RESULT.md"

mkdir -p "$RESULTS_DIR"

# Ensure gold text reference exists
if [[ ! -f "$GOLD_TEXT" ]]; then
    uv run python scripts/json_to_text.py "$INPUT" "$GOLD_TEXT"
fi

# ── model configs ─────────────────────────────────────────────────────────────
# Each entry: "label:relative_model_file"
# For "best"    → path = EXP_DIR (loads best.model directly)
# For "current" → we create a temp dir with a best.model symlink → current.model
MODELS=("best:best.model" "current:current.model")

# ── decode configs ────────────────────────────────────────────────────────────
# copy_attention gets multiple thresholds; others get a single run (no threshold)
declare -A THRESHOLDS
THRESHOLDS["copy_attention"]="0.7 0.8 0.9 0.95"
THRESHOLDS["attention"]=""
THRESHOLDS["ctc_greedy_search"]=""

DECODE_MODES=("copy_attention" "attention" "ctc_greedy_search")

# ── helpers ───────────────────────────────────────────────────────────────────
extract_metric() {
    # grab the value after "Overall" in a compute-wer output file
    local file="$1"
    grep -E "^Overall" "$file" 2>/dev/null | awk '{print $3}' | head -1
}

# ── write RESULT.md header ────────────────────────────────────────────────────
cat > "$RESULT_MD" <<'HEADER'
# Evaluation Results

> **Test set**: `data_to_upload/aishell_dataset/test-ne` (named-entity subset)
> **Config**: `conf/copyne.yaml` | **Beam size**: 10 | **Batch size**: 16

| Model | Decode Mode | Copy Threshold | CER (%) | NE-CER (%) |
|-------|-------------|----------------|---------|------------|
HEADER

# ── main loop ─────────────────────────────────────────────────────────────────
TMPDIR_BASE="$(mktemp -d "${SCRIPT_DIR}/.tmp_eval_XXXXXX")"
trap 'rm -rf "$TMPDIR_BASE"' EXIT

for MODEL_ENTRY in "${MODELS[@]}"; do
    MODEL_LABEL="${MODEL_ENTRY%%:*}"
    MODEL_FILE="${MODEL_ENTRY##*:}"
    SRC_MODEL="${SCRIPT_DIR}/${EXP_DIR}/${MODEL_FILE}"

    if [[ ! -f "$SRC_MODEL" ]]; then
        echo "[WARN] $SRC_MODEL not found — skipping."
        continue
    fi

    # For evaluate.py, --path must contain a file named best.model.
    # "best" → use EXP_DIR directly.
    # "current" → create a temp dir with best.model → actual current.model.
    if [[ "$MODEL_LABEL" == "best" ]]; then
        MODEL_PATH_ARG="$EXP_DIR"
    else
        TMPDIR_MODEL="${TMPDIR_BASE}/${MODEL_LABEL}"
        mkdir -p "$TMPDIR_MODEL"
        ln -sf "$SRC_MODEL" "${TMPDIR_MODEL}/best.model"
        MODEL_PATH_ARG="$TMPDIR_MODEL"
    fi

    for DECODE_MODE in "${DECODE_MODES[@]}"; do
        RAW_THRESH="${THRESHOLDS[$DECODE_MODE]}"
        # Normalise: at least one pass per mode
        THRESH_ITER="${RAW_THRESH:-none}"

        for THRESH in $THRESH_ITER; do
            if [[ "$THRESH" == "none" ]]; then
                TAG="${MODEL_LABEL}-${DECODE_MODE}"
                THRESH_FLAG=""
                THRESH_DISPLAY="-"
            else
                TAG="${MODEL_LABEL}-${DECODE_MODE}-cp${THRESH}"
                THRESH_FLAG="--copy_threshold ${THRESH}"
                THRESH_DISPLAY="${THRESH}"
            fi

            RES_PREFIX="${RESULTS_DIR}/${TAG}"

            echo "════════════════════════════════════════════════"
            echo " Model : $MODEL_LABEL ($MODEL_FILE)"
            echo " Mode  : $DECODE_MODE"
            echo " Thresh: $THRESH_DISPLAY"
            echo " Out   : ${RES_PREFIX}.pred"
            echo "════════════════════════════════════════════════"

            # ── predict ──────────────────────────────────────────────────────
            # shellcheck disable=SC2086
            uv run torchrun --nnodes=1 --nproc_per_node=1 evaluate.py \
                --char_dict      "$CHAR_DICT" \
                --add_context \
                --att_type       simpleatt \
                --add_copy_loss \
                --config         "$CONFIG" \
                --cmvn           "$CMVN" \
                --input          "$INPUT" \
                --test_ne_dict   "$TEST_NE_DICT" \
                --path           "$MODEL_PATH_ARG" \
                --res            "${RES_PREFIX}.pred" \
                --decode_mode    "$DECODE_MODE" \
                $THRESH_FLAG \
                --batch_size     "$BATCH" \
                --beam_size      "$BEAM"

            ASR_PRED="${RES_PREFIX}.pred.asr"

            # ── compute CER + NE-CER ─────────────────────────────────────────
            bash compute-necer.sh "$GOLD_TEXT" "$ASR_PRED"

            CER_VAL=$(extract_metric "${ASR_PRED}.cer")
            NECER_VAL=$(extract_metric "${ASR_PRED}.ne-cer")

            # ── append row ───────────────────────────────────────────────────
            printf "| %-7s | %-20s | %-14s | %-7s | %-10s |\n" \
                "$MODEL_LABEL" "$DECODE_MODE" "$THRESH_DISPLAY" \
                "$CER_VAL" "$NECER_VAL" \
                >> "$RESULT_MD"

            echo "  → CER=${CER_VAL}  NE-CER=${NECER_VAL}"
            echo ""
        done
    done
done

echo ""
echo "══════════════════════════════════════"
echo " All done. Results → $RESULT_MD"
echo "══════════════════════════════════════"
cat "$RESULT_MD"
