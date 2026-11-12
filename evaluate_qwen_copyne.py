#!/usr/bin/env python3
"""
evaluate_qwen_copyne.py: Benchmark Evaluation for Qwen3-ASR-0.6B + CopyNE + U2++ Decoding.
Supports 4 Evaluation Modes:
  1. 'zero_shot': Pretrained Qwen3-ASR without hotwords.
  2. 'prompt_hotwords': In-context entity prompting in Qwen3 chat template.
  3. 'copyne_l2r': Fine-tuned Left-to-Right CopyNE dot-product copy decoding.
  4. 'copyne_u2pp': Fine-tuned Two-Pass L2R + R2L Bidirectional look-back rescoring.

Outputs:
  - <res>.asr: Standard Kaldi/supar format (<utt_id> <hyp_text>) compatible with compute-necer.sh.
  - <res>.metrics.json: Entity Recall (NER), Char Error Rate (CER), and count statistics.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import orjson
import torch
import torch.nn.functional as F
import torchaudio as ta
import yaml

from model.qwen_copyne_u2pp import QwenCopyNEU2PlusPlus
from utils.qwen_copyne_data import load_ne_vocab_orjson

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("evaluate_qwen_copyne")


def compute_char_error_rate(ref: str, hyp: str) -> float:
    """Compute character error rate between reference and hypothesis strings."""
    ref_chars = list(ref.replace(" ", ""))
    hyp_chars = list(hyp.replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    len_r = len(ref_chars)
    len_h = len(hyp_chars)
    dp = [[0] * (len_h + 1) for _ in range(len_r + 1)]

    for i in range(len_r + 1):
        dp[i][0] = i
    for j in range(len_h + 1):
        dp[0][j] = j

    for i in range(1, len_r + 1):
        for j in range(1, len_h + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # deletion
                    dp[i][j - 1] + 1,      # insertion
                    dp[i - 1][j - 1] + 1,  # substitution
                )
    return dp[len_r][len_h] / float(len_r)


def load_audio_tensor(wav_path: str, target_sr: int = 16000) -> Optional[torch.Tensor]:
    """Load and resample waveform to mono 16kHz float tensor."""
    if not os.path.exists(wav_path):
        logger.error(f"Audio file not found: {wav_path}")
        return None
    try:
        waveform, sr = ta.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != target_sr:
            resampler = ta.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        return waveform.squeeze(0)
    except Exception as e:
        logger.error(f"Failed to read waveform from {wav_path}: {e}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-ASR CopyNE + U2++ on Contextual Test Sets",
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=str, default="conf/qwen3_copyne_u2pp.yaml", help="Path to config YAML")
    parser.add_argument("--model_path", "--path", type=str, default=None, help="Path to trained checkpoint (.pt) or directory")
    parser.add_argument("--qwen_model_id", type=str, default=None, help="Hugging Face model ID for Qwen3 backbone")
    parser.add_argument("--test_data", "--test", "--input", type=str, default="data_to_upload/aishell_dataset/test-ne.json", help="Path to test JSONL file")
    parser.add_argument("--test_ne_dict", type=str, default="data_to_upload/aishell_vocab/test_ne.vocab", help="Optional external entity vocabulary file")
    parser.add_argument(
        "--mode",
        type=str,
        default="copyne_u2pp",
        choices=["zero_shot", "prompt_hotwords", "copyne_l2r", "copyne_u2pp"],
        help="Evaluation benchmark mode",
    )
    parser.add_argument("--res", type=str, default="results.asr", help="Output hypothesis file path")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam search size")
    parser.add_argument("--copy_threshold", type=float, default=0.7, help="Probability threshold for entity copy trigger")
    parser.add_argument("--alpha_r2l", type=float, default=0.3, help="R2L backward rescoring weight for U2++ mode")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum generated transcription tokens")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (e.g. cuda:0 or cpu)")
    parser.add_argument("--language", type=str, default="Chinese", help="Target transcription language")
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine Device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running evaluation on device: {device}, Mode: {args.mode}")

    # Load Config
    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    qwen_model_id = args.qwen_model_id or config.get("model", {}).get("qwen_model_id", "Qwen/Qwen3-ASR-0.6B-hf")

    # Load Processor and Qwen3 Backbone
    logger.info(f"Loading processor and backbone from: {qwen_model_id}")
    # Disable broken torchvision in transformers to bypass Kaggle C++ operator mismatch (torchvision::nms missing)
    try:
        import transformers.utils.import_utils as _tr_import_utils
        _tr_import_utils.is_torchvision_available = lambda: False
        _tr_import_utils.is_torchvision_v2_available = lambda: False
    except Exception as _tr_err:
        logger.warning(f"Could not patch transformers torchvision availability flag: {_tr_err}")

    try:
        try:
            from transformers.models.qwen3_asr import (
                Qwen3ASRProcessor as AutoProcessor,
                Qwen3ASRForConditionalGeneration as AutoModelForCausalLM,
            )
        except Exception as _dir_err:
            logger.info(f"Direct Qwen3ASR import fallback to generic AutoProcessor: {_dir_err}")
            from transformers import AutoModelForCausalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(qwen_model_id, fix_markdown=False)
        qwen_backbone = AutoModelForCausalLM.from_pretrained(
            qwen_model_id,
            torch_dtype=torch.float16,
            device_map=None,
        ).to(device)
        qwen_backbone.eval()
        qwen_backbone.requires_grad_(False)
    except Exception as e:
        logger.error(f"Error loading Qwen3 model {qwen_model_id}: {e}")
        raise

    im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None:
        im_end_id = 151645

    # If CopyNE modes, load custom architecture and weights
    copyne_model = None
    if args.mode in ["copyne_l2r", "copyne_u2pp"]:
        logger.info(f"Initializing QwenCopyNEU2PlusPlus architecture...")
        copyne_model = QwenCopyNEU2PlusPlus(
            qwen_model=qwen_backbone,
            d_e=config.get("model", {}).get("d_e", 512),
            d_a=config.get("model", {}).get("d_a", 512),
            emb_proj_dim=config.get("model", {}).get("emb_proj_dim", 256),
            lstm_layers=config.get("model", {}).get("lstm_layers", 3),
            r2l_mode=config.get("model", {}).get("r2l_mode", "cross_attn"),
            r2l_dim=config.get("model", {}).get("d_e", 512),
            r2l_layers=config.get("model", {}).get("r2l_layers", 3),
            alpha_r2l=args.alpha_r2l if args.mode == "copyne_u2pp" else 0.0,
            freeze_base_lm_head=config.get("model", {}).get("freeze_base_lm", True),
        ).to(device)
        copyne_model.eval()

        ckpt_path = args.model_path
        if ckpt_path and os.path.isdir(ckpt_path):
            candidate_best = os.path.join(ckpt_path, "checkpoint_best.pt")
            candidate_last = os.path.join(ckpt_path, "checkpoint_last.pt")
            if os.path.exists(candidate_best):
                ckpt_path = candidate_best
            elif os.path.exists(candidate_last):
                ckpt_path = candidate_last

        if ckpt_path and os.path.exists(ckpt_path):
            logger.info(f"Loading checkpoint weights from: {ckpt_path}")
            try:
                ckpt = torch.load(ckpt_path, map_location=device)
                state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                copyne_model.load_trainable_state_dict(state_dict, strict=False)
                logger.info("Checkpoint weights loaded successfully.")
            except Exception as e:
                logger.error(f"Failed loading checkpoint from {ckpt_path}: {e}")
                raise
        else:
            logger.warning(f"No checkpoint provided via --model_path. Evaluating with initialized modules!")

    # External Entity Dictionary if provided
    global_entities: List[str] = []
    if args.test_ne_dict and os.path.exists(args.test_ne_dict):
        logger.info(f"Loading global entity dictionary from: {args.test_ne_dict}")
        global_entities = load_ne_vocab_orjson(args.test_ne_dict)
        logger.info(f"Loaded {len(global_entities)} external entities.")

    # Read Test JSONL
    if not os.path.exists(args.test_data):
        logger.error(f"Test data file does not exist: {args.test_data}")
        sys.exit(1)

    test_samples = []
    with open(args.test_data, "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    test_samples.append(orjson.loads(line))
                except Exception as e:
                    logger.error(f"JSON parsing error: {e}")

    logger.info(f"Evaluating {len(test_samples)} utterances...")
    out_dir = os.path.dirname(args.res)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    results_lines = []
    total_gold_entities = 0
    total_recalled_entities = 0
    total_cer = 0.0
    evaluated_count = 0
    start_eval_time = time.time()

    for idx, sample in enumerate(test_samples):
        utt_id = sample.get("key", f"utt_{idx:05d}")
        wav_path = sample.get("wav", "")
        if not os.path.isabs(wav_path) and not os.path.exists(wav_path):
            candidate = os.path.join(os.path.dirname(args.test_data), wav_path)
            if os.path.exists(candidate):
                wav_path = candidate

        audio_tensor = load_audio_tensor(wav_path)
        if audio_tensor is None:
            logger.warning(f"Skipping utterance {utt_id} due to audio load failure.")
            continue

        audio_np = audio_tensor.numpy()
        gold_text = sample.get("txt", "").strip()

        # Extract gold entities for recall metric (supports both 'entities' and AISHELL 'ne_lst')
        gold_entities: List[str] = []
        if "entities" in sample and isinstance(sample["entities"], (list, tuple, set)):
            gold_entities.extend(str(e).strip() for e in sample["entities"] if str(e).strip())
        for item in sample.get("ne_lst", []):
            if len(item) == 3:
                _, st, ed = item
                if ed >= st:
                    cand = gold_text[st : ed + 1].strip()
                    if cand:
                        gold_entities.append(cand)

        # Gather context dictionary entities for inference
        utt_entities = list(sample.get("entities", []))
        if not utt_entities and global_entities:
            utt_entities = list(global_entities)
        elif not utt_entities and gold_entities:
            utt_entities = list(dict.fromkeys(gold_entities))

        hyp_text = ""

        # Mode Execution
        if args.mode == "zero_shot":
            conv = [
                {
                    "role": "user",
                    "content": [{"type": "audio", "audio": audio_np, "sampling_rate": 16000}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"language {args.language}<asr_text>"}],
                },
            ]
            batch = processor.apply_chat_template(conv, tokenize=True, return_dict=True, continue_final_message=True)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if "input_features" in batch:
                batch["input_features"] = batch["input_features"].to(torch.float16)

            prompt_len = batch["input_ids"].shape[1]
            with torch.no_grad():
                gen_ids = qwen_backbone.generate(
                    **batch,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=im_end_id,
                    pad_token_id=processor.tokenizer.pad_token_id or im_end_id,
                )
            hyp_ids = gen_ids[0, prompt_len:].tolist()
            hyp_text = processor.tokenizer.decode(hyp_ids, skip_special_tokens=True).strip()

        elif args.mode == "prompt_hotwords":
            hotwords_str = "、".join(utt_entities) if utt_entities else ""
            hotwords_hint = f"上下文专有名词候选词：{hotwords_str}。\n请识别以下语音：" if hotwords_str else ""
            conv = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": hotwords_hint},
                        {"type": "audio", "audio": audio_np, "sampling_rate": 16000},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"language {args.language}<asr_text>"}],
                },
            ]
            batch = processor.apply_chat_template(conv, tokenize=True, return_dict=True, continue_final_message=True)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if "input_features" in batch:
                batch["input_features"] = batch["input_features"].to(torch.float16)

            prompt_len = batch["input_ids"].shape[1]
            with torch.no_grad():
                gen_ids = qwen_backbone.generate(
                    **batch,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=im_end_id,
                    pad_token_id=processor.tokenizer.pad_token_id or im_end_id,
                )
            hyp_ids = gen_ids[0, prompt_len:].tolist()
            hyp_text = processor.tokenizer.decode(hyp_ids, skip_special_tokens=True).strip()

        elif args.mode in ["copyne_l2r", "copyne_u2pp"]:
            assert copyne_model is not None
            conv = [
                {
                    "role": "user",
                    "content": [{"type": "audio", "audio": audio_np, "sampling_rate": 16000}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"language {args.language}<asr_text>"}],
                },
            ]
            batch = processor.apply_chat_template(conv, tokenize=True, return_dict=True, continue_final_message=True)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if "input_features" in batch:
                batch["input_features"] = batch["input_features"].to(torch.float16)

            # Build Entity Tensors (DualStackedLSTMContextEncoder appends null entity at index N)
            ent_subwords = []
            for ent in utt_entities:
                toks = processor.tokenizer(ent, add_special_tokens=False).input_ids
                if toks:
                    ent_subwords.append(toks)

            if len(ent_subwords) > 0:
                max_ent_len = max(len(t) for t in ent_subwords)
                ne_l2r = torch.full((len(ent_subwords), max_ent_len), -1, dtype=torch.long, device=device)
                ne_r2l = torch.full((len(ent_subwords), max_ent_len), -1, dtype=torch.long, device=device)
                lens = []
                for i, toks in enumerate(ent_subwords):
                    ne_l2r[i, : len(toks)] = torch.tensor(toks, dtype=torch.long, device=device)
                    ne_r2l[i, : len(toks)] = torch.tensor(list(reversed(toks)), dtype=torch.long, device=device)
                    lens.append(len(toks))
                ne_lens = torch.tensor(lens, dtype=torch.long, device=device)
            else:
                ne_l2r = torch.empty((0, 1), dtype=torch.long, device=device)
                ne_r2l = torch.empty((0, 1), dtype=torch.long, device=device)
                ne_lens = torch.empty((0,), dtype=torch.long, device=device)

            with torch.no_grad():
                Z_l2r, Z_r2l = copyne_model.ne_encoder(ne_l2r, ne_r2l, ne_lens)
                pred_token_lists = copyne_model.recognize_two_pass(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    input_features=batch["input_features"],
                    input_features_mask=batch.get("input_features_mask"),
                    Z_l2r=Z_l2r,
                    Z_r2l=Z_r2l,
                    ne_tokens_l2r=ne_l2r,
                    ne_lengths=ne_lens,
                    beam_size=args.beam_size,
                    copy_threshold=args.copy_threshold,
                    use_r2l_rescoring=(args.mode == "copyne_u2pp"),
                    max_new_tokens=args.max_new_tokens,
                )

            hyp_ids = pred_token_lists[0]
            hyp_text = processor.tokenizer.decode(hyp_ids, skip_special_tokens=True).strip()

        # Sanitize hypothesis text (remove potential artifacts)
        hyp_text = hyp_text.replace("<asr_text>", "").replace("<|im_end|>", "").strip()
        results_lines.append(f"{utt_id} {hyp_text}\n")
        evaluated_count += 1

        # Entity Recall Accounting (against utterance gold entities)
        for ent in gold_entities:
            total_gold_entities += 1
            if ent in hyp_text:
                total_recalled_entities += 1

        # CER Accounting
        if gold_text:
            cer_val = compute_char_error_rate(gold_text, hyp_text)
            total_cer += cer_val

        if (idx + 1) % 50 == 0 or (idx + 1) == len(test_samples):
            current_recall = (total_recalled_entities / max(1, total_gold_entities)) * 100.0
            avg_cer = (total_cer / max(1, evaluated_count)) * 100.0
            logger.info(
                f"Processed {idx + 1}/{len(test_samples)} | "
                f"Entity Recall: {current_recall:.2f}% | CER: {avg_cer:.2f}%"
            )

    # Write .asr Output File (write both args.res and args.res + ".asr" for compute-necer.sh compatibility)
    with open(args.res, "w", encoding="utf-8") as f_out:
        f_out.writelines(results_lines)
    logger.info(f"Transcription hypotheses written to {args.res}")
    if not args.res.endswith(".asr"):
        asr_out = f"{args.res}.asr"
        with open(asr_out, "w", encoding="utf-8") as f_asr:
            f_asr.writelines(results_lines)
        logger.info(f"Transcription hypotheses also written to {asr_out}")

    # Compute Final Benchmark Metrics
    final_recall = (total_recalled_entities / max(1, total_gold_entities)) * 100.0 if total_gold_entities > 0 else 0.0
    final_cer = (total_cer / max(1, evaluated_count)) * 100.0 if evaluated_count > 0 else 0.0
    total_time = time.time() - start_eval_time

    metrics = {
        "mode": args.mode,
        "evaluated_utterances": evaluated_count,
        "total_gold_entities": total_gold_entities,
        "total_recalled_entities": total_recalled_entities,
        "entity_recall_percent": round(final_recall, 2),
        "mean_cer_percent": round(final_cer, 2),
        "total_seconds": round(total_time, 2),
        "speed_utt_per_sec": round(evaluated_count / max(0.1, total_time), 2),
    }

    metrics_file = f"{args.res}.metrics.json"
    with open(metrics_file, "wb") as f_m:
        f_m.write(orjson.dumps(metrics, option=orjson.OPT_INDENT_2))
    logger.info(f"Evaluation metrics saved to {metrics_file}:")
    logger.info(
        f"==> Mode: {args.mode} | Entity Recall: {final_recall:.2f}% | "
        f"Mean CER: {final_cer:.2f}% | Utterances: {evaluated_count}"
    )


if __name__ == "__main__":
    main()

