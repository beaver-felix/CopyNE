# CopyNE + Qwen3-ASR-0.6B + U2++ Bidirectional Decoding

Graduation Thesis project: End-to-End Contextual ASR under constrained compute budget.
Combining frozen **Qwen/Qwen3-ASR-0.6B-hf** multimodal backbone, trainable 3-layer Stacked-LSTM Named Entity (NE) Encoder, **CopyNE Dot-Product Attention + Dict-Enhanced Linear Classifier**, and **U2++ Bidirectional Look-Back / Rescoring**.

---

## 1. Kaggle 2x T4 Environment Setup

In Kaggle notebook cell:
```bash
# 1. Clone repository
!git clone https://github.com/beaver-felix/CopyNE.git
%cd CopyNE

# 2. Install uv and dependencies
!pip install uv
!uv pip install --system torch==2.14.1 torchaudio==2.11.1 "transformers>=5.13.0" orjson polars pyyaml
```

---

## 2. Data Preparation

```bash
%cd /kaggle/working/CopyNE/data_to_upload
# Download and unpack AISHELL dataset
!wget -c https://us.openslr.org/resources/33/data_aishell.tgz
!tar -zxvf data_aishell.tgz

# Update wav paths in json manifests
%cd /kaggle/working/CopyNE
!python scripts/change_wav_path.py data_to_upload/aishell_dataset/train_addne.json data_to_upload
!python scripts/change_wav_path.py data_to_upload/aishell_dataset/dev_addne.json data_to_upload
!python scripts/change_wav_path.py data_to_upload/aishell_dataset/test_addne.json data_to_upload
!python scripts/change_wav_path.py data_to_upload/aishell_dataset/dev-ne.json data_to_upload
!python scripts/change_wav_path.py data_to_upload/aishell_dataset/test-ne.json data_to_upload
```

---

## 3. Verify Qwen3-ASR Backbone Hook

```bash
uv run python scripts/verify_qwen3_hook.py
```

---

## 4. Training on Kaggle 2x T4 GPUs

### Row 2: Baseline Original CopyNE (Zipformer CTC/Attn from scratch)
```bash
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 --master_port=29501 train.py \
    --train data_to_upload/aishell_dataset/train_addne.json \
    --dev data_to_upload/aishell_dataset/dev_addne.json \
    --test data_to_upload/aishell_dataset/test_addne.json \
    --add_context \
    --pad_context 2 \
    --att_type simpleatt \
    --add_copy_loss \
    --train_ne_dict data_to_upload/aishell_vocab/train_ne.vocab \
    --dev_ne_dict data_to_upload/aishell_vocab/dev_ne.vocab \
    --path exp/copyne_aishell_beta1/ \
    --batch_size 16 \
    --seed 777 \
    --config conf/copyne.yaml \
    --char_dict data_to_upload/aishell_vocab/char.vocab \
    --cmvn data_to_upload/aishell1_global_cmvn_mel80 \
    --num_workers 2 \
    --max_hours 11.0
```

### Row 3: Proposed Model (Frozen Qwen3 + L2R CopyNE, `alpha_r2l = 0.0`)
```bash
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 train_qwen_copyne.py \
    --config conf/qwen3_copyne_u2pp.yaml \
    --train data_to_upload/aishell_dataset/train_addne.json \
    --dev data_to_upload/aishell_dataset/dev_addne.json \
    --path exp/qwen3_copyne_l2r \
    --alpha_r2l 0.0 \
    --max_hours 11.0
```

### Row 4: Proposed Model (Frozen Qwen3 + Full U2++ R2L Rescoring, `alpha_r2l = 0.3`)
```bash
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 train_qwen_copyne.py \
    --config conf/qwen3_copyne_u2pp.yaml \
    --train data_to_upload/aishell_dataset/train_addne.json \
    --dev data_to_upload/aishell_dataset/dev_addne.json \
    --path exp/qwen3_copyne_u2pp \
    --alpha_r2l 0.3 \
    --max_hours 11.0 \
    --pre_model exp/qwen3_copyne_l2r/checkpoint_best.pt
```

> **Resume across Kaggle 12-hour session limits**:
> Add `--pre_model <output_dir>/checkpoint_session_timeout.pt` (or `current.model` for Row 2 baseline).

---

## 5. Thesis 4-Row Evaluation & Metrics

```bash
# Row 1a: Zero-shot frozen Qwen3-ASR-0.6B
uv run python evaluate_qwen_copyne.py --mode zero_shot --res results/row1a_qwen_zeroshot.pred

# Row 1b: System-prompt hotwords baseline
uv run python evaluate_qwen_copyne.py --mode prompt_hotwords --res results/row1b_qwen_hotwords.pred

# Row 2: Original CopyNE baseline
bash run_eval.sh

# Row 3: Proposed L2R CopyNE
uv run python evaluate_qwen_copyne.py --mode copyne_l2r --path exp/qwen3_copyne_l2r --copy_threshold 0.8 --res results/row3_qwen_copyne_l2r.pred

# Row 4: Proposed Full U2++ (L2R + R2L Rescoring)
uv run python evaluate_qwen_copyne.py --mode copyne_u2pp --path exp/qwen3_copyne_u2pp --copy_threshold 0.7 --res results/row4_qwen_copyne_u2pp.pred

# Compute CER and NE-CER
bash compute-necer.sh data_to_upload/aishell_dataset/test-ne.text results/row4_qwen_copyne_u2pp.pred.asr
```
