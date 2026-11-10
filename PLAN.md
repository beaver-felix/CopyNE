# Exhaustive Execution Plan: Frozen Qwen3-ASR-0.6B + CopyNE + U2++ Bidirectional Decoding (Kaggle 2x T4 Edition)

## 0. Mandatory Rules & Target Hardware Ground Truth (Kaggle 2x NVIDIA T4)

1. **Runner**: Always execute Python with `uv run python ...` or `uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 ...` inside `/shared/CopyNE` (or Kaggle working directory).
2. **Packages**:
   - Use `orjson` (`orjson.loads`, `orjson.dumps`) instead of `json`.
   - Use `polars` or `duckdb` instead of `pandas`.
   - Use `pnpm` instead of `npm` if any JS tooling is touched.
3. **Exception Handling**: Never put `pass` inside `except` blocks. Always log an explicit message or re-raise.
4. **Target Hardware — Kaggle 2x NVIDIA T4 (`2 x 16 GB VRAM`, Turing `sm_75`)**:
   - **VRAM**: 16 GB per GPU (32 GB total). Supports `batch_size = 8` or `16` per GPU (`effective_batch_size = 32` across 2 GPUs with `accum_grad = 1` or `2`).
   - **T4 Precision Rule (`torch.float16`, NOT `bfloat16`)**:
     - NVIDIA T4 is Turing architecture (`sm_75`, compute capability `7.5`). T4 has hardware **FP16 Tensor Cores** (`65 TFLOPS`), NOT native hardware BF16 Tensor Cores (`torch.cuda.is_bf16_supported()` emulates BF16 slowly on `sm_75`).
     - Select precision dynamically via compute capability:
       ```python
       def get_optimal_backbone_dtype() -> torch.dtype:
           if torch.cuda.is_available():
               major, _ = torch.cuda.get_device_capability()
               return torch.bfloat16 if major >= 8 else torch.float16  # T4 is major==7 -> float16
           return torch.float32
       ```
     - Note: `Qwen3ASRAudioEncoderLayer.forward` (`modeling_qwen3_asr.py` lines 236–238) has built-in FP16 dynamic range clamping (`torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)`), making `torch.float16` safe and fast on T4.
     - Keep all trainable CopyNE + U2++ parameters in **FP32** (`torch.float32`) so optimizer updates and dot-product attention scores never underflow/overflow without needing `GradScaler`.
   - **Kaggle Session Time Limit (`--max_hours`)**:
     - Kaggle GPU sessions terminate at 12 hours. Include `--max_hours 11.0` and full checkpoint resumption (`--pre_model exp/.../current.model`) with `dist.barrier()` across both T4 GPUs, matching `train.py` and `asr.py`.

---

## 1. Critical `transformers.models.qwen3_asr` & 2x T4 DDP Traps (Read Before Coding)

1. **Trap 1 — Dynamic Hidden Dimension**:
   - Read `d_model = qwen.config.text_config.hidden_size` and `vocab_size = qwen.config.text_config.vocab_size` (`151936`) dynamically from `qwen.config`. Never hardcode `896`.
2. **Trap 2 — Call `qwen.model(...)`, Not `qwen(...)`**:
   - `Qwen3ASRForConditionalGeneration.forward` runs `self.lm_head(hidden_states)` over all prompt + audio tokens (`(B, S, 151936)`), wasting >2 GB VRAM at `B=8`.
   - Call `qwen.model(...)` under `with torch.no_grad():` to get `outputs.last_hidden_state` of shape `(B, S, d_model)` before `lm_head`.
3. **Trap 3 — `outputs.audio_hidden_states` is 2D Flattened `(sum_T_a, d_model)`, NOT 3D `(B, T_a, d_model)`**:
   - `Qwen3ASREncoder.forward` packs valid post-CNN audio frames across the batch into a 2D tensor of shape `(total_audio_tokens_in_batch, d_model)`.
   - Unpack `outputs.audio_hidden_states` into a 3D padded tensor `H_audio` `(B, max_T_a, d_model)` and boolean padding mask `audio_pad_mask` `(B, max_T_a)` (where `True` = padded position for `nn.MultiheadAttention`):
     ```python
     audio_token_id = qwen.config.audio_token_id  # 151676
     audio_lens = (input_ids == audio_token_id).sum(dim=1).tolist()
     audio_chunks = torch.split(outputs.audio_hidden_states, audio_lens, dim=0)
     H_audio = torch.nn.utils.rnn.pad_sequence(audio_chunks, batch_first=True, padding_value=0.0)
     max_T_a = H_audio.size(1)
     audio_pad_mask = (
         torch.arange(max_T_a, device=H_audio.device).unsqueeze(0)
         >= torch.tensor(audio_lens, device=H_audio.device).unsqueeze(1)
     )
     ```
4. **Trap 4 — `Qwen3ASRProcessor` Uses Left-Padding by Default**:
   - In a batch of size $B$, the token index $s_0^{(b)}$ of `<asr_text>` differs for each sample $b \in \{0, \dots, B-1\}$.
   - Locate $s_0^{(b)}$ per sample using `(input_ids[b] == asr_text_token_id).nonzero(as_tuple=True)[0][-1].item()`, slice the transcription states $d_{0:U_b-1}^{(b)} = H_{\text{last}}[b, s_0^{(b)} : s_0^{(b)} + U_b, :]$ and target tokens $y_{1:U_b}^{(b)} = \text{input\_ids}[b, s_0^{(b)} + 1 : s_0^{(b)} + 1 + U_b]$, and right-pad them across the batch to `(B, U_max, d_model)` and `(B, U_max)`.
5. **Trap 5 — Warm-Started `W_out` Split Linear Parameterization**:
   - Mathematically, $W_{\text{out}} = [W_{\text{base}} \;|\; W_{\text{dict}}] \in \mathbb{R}^{|V| \times (d_{\text{model}} + d_e)}$ where $W_{\text{base}}$ is initialized from `qwen.lm_head.weight` and $W_{\text{dict}}, b_{\text{out}}$ are initialized to `0.0`:
     $$W_{\text{out}} \begin{bmatrix} d_u \\ r_u \end{bmatrix} + b_{\text{out}} = W_{\text{base}} d_u + W_{\text{dict}} r_u + b_{\text{out}}$$
   - Implement as:
     - `base_out_proj`: either frozen `qwen.lm_head` (`freeze_base_lm_head: true`, default, 0 extra memory) OR a trainable `nn.Linear(d_model, vocab_size, bias=False)` initialized with `qwen.lm_head.weight.data.clone().float()` trained at `lr = 1e-5` (100x smaller than `1e-3`).
     - `dict_out_proj = nn.Linear(d_e, vocab_size, bias=True)` initialized to zeros (`nn.init.zeros_`), trained at `lr = 1e-3`.
   - This enables differential learning rates (`1e-5` on base columns vs. `1e-3` on dictionary columns) cleanly via PyTorch optimizer param groups without slicing gradients on a concatenated tensor.
6. **Trap 6 — 2x T4 DDP Deadlock & NCCL Timeout Prevention**:
   - Set `broadcast_buffers=False` in `DistributedDataParallel`.
   - When evaluating on dev set on Rank 0, always place `dist.barrier()` **before** and **after** Rank 0 dev evaluation + checkpoint saving so Rank 1 never races into the next epoch's collective communications and triggers an NCCL timeout (matching `asr.py` lines 1596–1660).

---

## 2. Step-by-Step Implementation Plan

### Step 1: Environment & Hidden-State Hook Verification Script (`scripts/verify_qwen3_hook.py`)

1. Create `scripts/verify_qwen3_hook.py` to verify `Qwen/Qwen3-ASR-0.6B-hf` loading on T4 (`torch.float16`), chat template formatting, `<asr_text>` boundary detection, `outputs.last_hidden_state` slicing, `outputs.audio_hidden_states` 2D-to-3D unpacking, and exact logit equivalence at step 0.
2. Write the exact code below to `scripts/verify_qwen3_hook.py`:

```python
import os
import torch
import torch.nn.functional as F
import torchaudio as ta
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration


def get_optimal_backbone_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def build_training_conversation(audio_array, sr: int, transcript: str, language: str = "Chinese"):
    return [
        {
            "role": "user",
            "content": [{"type": "audio", "audio": audio_array, "sampling_rate": sr}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": f"language {language}<asr_text>{transcript}"}],
        },
    ]


def main():
    model_id = os.environ.get("QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-0.6B-hf")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = get_optimal_backbone_dtype()

    processor = AutoProcessor.from_pretrained(model_id)
    qwen = Qwen3ASRForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device)
    qwen.eval()
    qwen.requires_grad_(False)

    d_model = qwen.config.text_config.hidden_size
    vocab_size = qwen.config.text_config.vocab_size
    audio_token_id = qwen.config.audio_token_id
    asr_text_id = processor.tokenizer.convert_tokens_to_ids("<asr_text>")
    im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    print(
        f"[Check 1] device={device}, dtype={dtype}, num_gpus={torch.cuda.device_count()}, "
        f"d_model={d_model}, vocab_size={vocab_size}, audio_token_id={audio_token_id}, "
        f"asr_text_id={asr_text_id}, im_end_id={im_end_id}"
    )

    wav_path = "BAC009S0916W0486.wav"
    waveform, sr = ta.load(wav_path)
    if sr != 16000:
        waveform = ta.transforms.Resample(sr, 16000)(waveform)
        sr = 16000
    audio_np = waveform[0].numpy()

    transcript = "中原地产首席分析师张大伟说"
    conv = [build_training_conversation(audio_np, sr, transcript, language="Chinese")]
    batch = processor.apply_chat_template(
        conv,
        tokenize=True,
        return_dict=True,
        continue_final_message=False,
    )
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    if "input_features" in batch:
        batch["input_features"] = batch["input_features"].to(dtype)

    with torch.no_grad():
        backbone_out = qwen.model(
            input_ids=batch["input_ids"],
            input_features=batch["input_features"],
            input_features_mask=batch["input_features_mask"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
        )

    H_last = backbone_out.last_hidden_state  # (B, S, d_model)
    audio_flat = backbone_out.audio_hidden_states  # (sum_T_a, d_model)

    # Unpack audio_hidden_states to (B, T_a, d_model)
    audio_lens = (batch["input_ids"] == audio_token_id).sum(dim=1).tolist()
    audio_chunks = torch.split(audio_flat, audio_lens, dim=0)
    H_audio = torch.nn.utils.rnn.pad_sequence(audio_chunks, batch_first=True, padding_value=0.0)

    # Locate <asr_text> boundary s_0
    s0_indices = (batch["input_ids"][0] == asr_text_id).nonzero(as_tuple=True)[0]
    assert len(s0_indices) > 0, "Failed to find <asr_text> token in input_ids"
    s0 = int(s0_indices[-1].item())

    # Target tokens y_1 ... y_U (ending with <|im_end|>)
    target_ids = batch["input_ids"][0, s0 + 1 :]
    U = target_ids.size(0)
    d_u = H_last[0, s0 : s0 + U, :]  # (U, d_model)

    assert d_u.shape == (U, d_model), f"Expected {(U, d_model)}, got {d_u.shape}"
    assert int(target_ids[-1].item()) == im_end_id or int(target_ids[-2].item()) == im_end_id, (
        f"Expected <|im_end|> ({im_end_id}) at end of target_ids, got {target_ids[-3:].tolist()}"
    )

    # Verify zero-init dict_proj reproduces exact Qwen3 lm_head logits
    d_e = 512
    dict_proj = torch.nn.Linear(d_e, vocab_size, bias=True).to(device=device, dtype=torch.float32)
    torch.nn.init.zeros_(dict_proj.weight)
    torch.nn.init.zeros_(dict_proj.bias)
    r_u = torch.randn(U, d_e, device=device, dtype=torch.float32)

    with torch.no_grad():
        base_logits = qwen.lm_head(d_u).float()
        copyne_init_logits = F.linear(d_u, qwen.lm_head.weight).float() + dict_proj(r_u)
        max_diff = (base_logits - copyne_init_logits).abs().max().item()

    assert max_diff == 0.0, f"Logit mismatch at init: {max_diff}"
    print(f"[Check 2] H_last={tuple(H_last.shape)}, H_audio={tuple(H_audio.shape)}, U={U}, max_logit_diff={max_diff}")
    print("[PASS] Step 1 verification succeeded.")


if __name__ == "__main__":
    main()
```

3. Run verification:
   ```bash
   uv run python scripts/verify_qwen3_hook.py
   ```

---

### Step 2: Data Pipeline & BPE Subword Maximum-Matching Copy Target Builder (`utils/qwen_copyne_data.py`)

1. Create `utils/qwen_copyne_data.py` implementing:
   - Fast JSONL & NE dictionary loading with `orjson`.
   - **BPE Space-Prefix Handling**: Tokenizes each entity `ne_str` both without and with a leading space (`" " + ne_str` for ASCII-initial strings) so subword maximum matching catches both utterance-initial and mid-sentence BPE subwords.
   - **L2R & R2L Subword Maximum Matching (`build_bpe_copy_targets`)**:
     - Given transcript token sequence $y = (y_1, \dots, y_U)$ (where $y_U = \text{eos\_id}$) and batch entity list $E_b = [e_0, e_1, \dots, e_{N-1}]$ (sorted by length descending, `null_idx = N`):
     - For matched span $[u, u + L_k)$:
       - **L2R Copy Target** (`att_tgt_l2r`): sets `att_tgt_l2r[u] = k` and `att_tgt_l2r[u + 1 : u + L_k] = null_idx`.
       - **R2L Copy Target** (`att_tgt_r2l`, aligned with `r2l_target_ids = [y_{U-1}, ..., y_1, eos_id]`): sets `att_tgt_r2l[text_len - (u + L_k)] = k` (triggering at the rightmost token of the entity when decoding right-to-left).
2. Write the exact code below to `utils/qwen_copyne_data.py`:

```python
import os
import random
from typing import List, Tuple
import orjson
import torch
import torchaudio as ta
from torch.nn.utils.rnn import pad_sequence


def load_jsonl_orjson(path: str) -> List[dict]:
    records = []
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(orjson.loads(line))
    return records


def load_ne_vocab_orjson(path: str) -> List[str]:
    with open(path, "rb") as f:
        vocab_dict = orjson.loads(f.read())
    itos = {int(v): k for k, v in vocab_dict.items()}
    return [itos[i] for i in range(len(itos))]


def tokenize_entity_variants(tokenizer, ne_str: str) -> Tuple[List[int], List[List[int]]]:
    """Return canonical token IDs and all matching BPE variants (with/without leading space for Latin/ASCII)."""
    canonical_ids = tokenizer.encode(ne_str, add_special_tokens=False)
    variants = [canonical_ids]
    if len(ne_str) > 0 and ne_str[0].isascii() and not ne_str[0].isspace():
        spaced_ids = tokenizer.encode(" " + ne_str, add_special_tokens=False)
        if spaced_ids and spaced_ids != canonical_ids:
            variants.append(spaced_ids)
    return canonical_ids, variants


def build_bpe_copy_targets(
    target_token_ids: List[int],
    entity_variants: List[List[List[int]]],
    null_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build L2R and R2L copy targets and reversed target token IDs.

    Args:
        target_token_ids: list of length U, [y_1, y_2, ..., y_{U-1}, eos_id]
        entity_variants: list of length N; entity_variants[k] is a list of BPE token-ID lists for entity k
        null_idx: int == N (index of null entity e_0)

    Returns:
        att_tgt_l2r: (U,) long tensor with values in [0, N]
        att_tgt_r2l: (U,) long tensor with values in [0, N] aligned with r2l_target_ids
        r2l_target_ids: (U,) long tensor [y_{U-1}, y_{U-2}, ..., y_1, eos_id]
    """
    U = len(target_token_ids)
    text_len = U - 1  # exclude final eos_id from entity span matching
    eos_id = target_token_ids[-1]
    text_ids = target_token_ids[:text_len]

    att_tgt_l2r = [null_idx] * U
    matched_spans: List[Tuple[int, int, int]] = []

    candidates: List[Tuple[int, int, List[int]]] = []
    for ent_idx, vars_for_ent in enumerate(entity_variants):
        for var_ids in vars_for_ent:
            if len(var_ids) > 0:
                candidates.append((len(var_ids), ent_idx, var_ids))
    candidates.sort(key=lambda x: x[0], reverse=True)

    u = 0
    while u < text_len:
        matched = False
        for var_len, ent_idx, var_ids in candidates:
            if u + var_len <= text_len and text_ids[u : u + var_len] == var_ids:
                att_tgt_l2r[u] = ent_idx
                matched_spans.append((u, u + var_len, ent_idx))
                u += var_len
                matched = True
                break
        if not matched:
            u += 1

    r2l_text_ids = list(reversed(text_ids))
    r2l_target_ids = r2l_text_ids + [eos_id]
    att_tgt_r2l = [null_idx] * U

    for st, ed, ent_idx in matched_spans:
        r2l_trigger_pos = text_len - ed
        if 0 <= r2l_trigger_pos < text_len:
            att_tgt_r2l[r2l_trigger_pos] = ent_idx

    return (
        torch.tensor(att_tgt_l2r, dtype=torch.long),
        torch.tensor(att_tgt_r2l, dtype=torch.long),
        torch.tensor(r2l_target_ids, dtype=torch.long),
    )


class QwenCopyNEDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        json_file: str,
        ne_dict_file: str,
        is_training: bool = True,
        beta_distractors: int = 2,
        language: str = "Chinese",
    ):
        super().__init__()
        self.records = load_jsonl_orjson(json_file)
        self.ne_vocab = load_ne_vocab_orjson(ne_dict_file) if (ne_dict_file and os.path.exists(ne_dict_file)) else []
        self.is_training = is_training
        self.beta_distractors = beta_distractors
        self.language = language

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        wav_path = rec["wav"]
        waveform, sr = ta.load(wav_path)
        if sr != 16000:
            waveform = ta.transforms.Resample(sr, 16000)(waveform)
            sr = 16000
        audio_np = waveform[0].numpy()
        sentence = rec["txt"]
        ne_set = {
            sentence[st : ed + 1]
            for _, st, ed in rec.get("ne_lst", [])
            if ed > st and len(sentence[st : ed + 1].strip()) > 0
        }
        return {
            "key": rec["key"],
            "audio_np": audio_np,
            "sr": sr,
            "sentence": sentence,
            "ne_set": ne_set,
            "language": self.language,
        }


class QwenCopyNECollator:
    def __init__(
        self,
        processor,
        ne_vocab: List[str],
        is_training: bool = True,
        is_dev: bool = False,
        beta_distractors: int = 2,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.ne_vocab = ne_vocab
        self.is_training = is_training
        self.is_dev = is_dev
        self.beta_distractors = beta_distractors
        self.asr_text_id = self.tokenizer.convert_tokens_to_ids("<asr_text>")
        self.pad_token_id = self.tokenizer.pad_token_id

    def _sample_batch_entities(self, batch: List[dict]) -> List[str]:
        if not self.is_training and not self.is_dev:
            return list(self.ne_vocab)

        batch_ne_set = set()
        for item in batch:
            if len(item["ne_set"]) > 0:
                batch_ne_set |= item["ne_set"]
            elif self.is_training:
                s = item["sentence"]
                if len(s) >= 2:
                    for _ in range(random.randint(1, 2)):
                        span_len = min(random.randint(2, 4), len(s))
                        st = random.randint(0, len(s) - span_len)
                        cand = s[st : st + span_len].strip()
                        if len(cand) >= 2:
                            batch_ne_set.add(cand)

        if self.is_training and len(self.ne_vocab) > 0:
            target_total = max(len(batch), (1 + self.beta_distractors) * max(1, len(batch_ne_set)))
            attempts = 0
            while len(batch_ne_set) < target_total and attempts < target_total * 4:
                attempts += 1
                cand = random.choice(self.ne_vocab)
                if len(cand) > 1:
                    batch_ne_set.add(cand)

        if len(batch_ne_set) == 0 and len(self.ne_vocab) > 0:
            batch_ne_set.add(self.ne_vocab[0])

        return sorted(batch_ne_set, key=len, reverse=True)

    def __call__(self, batch: List[dict]) -> dict:
        batch_ne_lst = self._sample_batch_entities(batch)
        null_idx = len(batch_ne_lst)

        ne_l2r_tensors: List[torch.Tensor] = []
        ne_r2l_tensors: List[torch.Tensor] = []
        entity_variants: List[List[List[int]]] = []

        for ne_str in batch_ne_lst:
            canon_ids, variants = tokenize_entity_variants(self.tokenizer, ne_str)
            if len(canon_ids) == 0:
                canon_ids = [self.pad_token_id]
            ne_l2r_tensors.append(torch.tensor(canon_ids, dtype=torch.long))
            ne_r2l_tensors.append(torch.tensor(list(reversed(canon_ids)), dtype=torch.long))
            entity_variants.append(variants)

        if len(ne_l2r_tensors) > 0:
            ne_tokens_l2r = pad_sequence(ne_l2r_tensors, batch_first=True, padding_value=-1)
            ne_tokens_r2l = pad_sequence(ne_r2l_tensors, batch_first=True, padding_value=-1)
            ne_lengths = torch.tensor([t.numel() for t in ne_l2r_tensors], dtype=torch.long)
        else:
            ne_tokens_l2r = torch.empty((0, 1), dtype=torch.long)
            ne_tokens_r2l = torch.empty((0, 1), dtype=torch.long)
            ne_lengths = torch.empty((0,), dtype=torch.long)

        conversations = []
        for item in batch:
            prefill = (
                f"language {item['language']}<asr_text>{item['sentence']}"
                if (self.is_training or self.is_dev)
                else f"language {item['language']}<asr_text>"
            )
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [{"type": "audio", "audio": item["audio_np"], "sampling_rate": item["sr"]}],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": prefill}],
                    },
                ]
            )

        proc_out = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            continue_final_message=not (self.is_training or self.is_dev),
        )

        input_ids = proc_out["input_ids"]
        s0_list: List[int] = []
        target_l2r_list: List[torch.Tensor] = []
        target_r2l_list: List[torch.Tensor] = []
        att_tgt_l2r_list: List[torch.Tensor] = []
        att_tgt_r2l_list: List[torch.Tensor] = []
        target_lens: List[int] = []

        for b in range(len(batch)):
            pos = (input_ids[b] == self.asr_text_id).nonzero(as_tuple=True)[0]
            assert len(pos) > 0, f"<asr_text> token missing in sample {batch[b]['key']}"
            s0 = int(pos[-1].item())
            s0_list.append(s0)

            if self.is_training or self.is_dev:
                valid_mask = proc_out["attention_mask"][b, s0 + 1 :].bool()
                tgt_ids = input_ids[b, s0 + 1 :][valid_mask].tolist()
                att_l2r, att_r2l, r2l_ids = build_bpe_copy_targets(tgt_ids, entity_variants, null_idx)
                target_l2r_list.append(torch.tensor(tgt_ids, dtype=torch.long))
                target_r2l_list.append(r2l_ids)
                att_tgt_l2r_list.append(att_l2r)
                att_tgt_r2l_list.append(att_r2l)
                target_lens.append(len(tgt_ids))

        result = {
            "keys": [item["key"] for item in batch],
            "sentences": [item["sentence"] for item in batch],
            "input_ids": proc_out["input_ids"],
            "attention_mask": proc_out["attention_mask"],
            "input_features": proc_out["input_features"],
            "input_features_mask": proc_out["input_features_mask"],
            "s0_indices": torch.tensor(s0_list, dtype=torch.long),
            "ne_lst": batch_ne_lst,
            "ne_tokens_l2r": ne_tokens_l2r,
            "ne_tokens_r2l": ne_tokens_r2l,
            "ne_lengths": ne_lengths,
        }

        if self.is_training or self.is_dev:
            result["target_l2r"] = pad_sequence(target_l2r_list, batch_first=True, padding_value=-100)
            result["target_r2l"] = pad_sequence(target_r2l_list, batch_first=True, padding_value=-100)
            result["att_tgt_l2r"] = pad_sequence(att_tgt_l2r_list, batch_first=True, padding_value=-100)
            result["att_tgt_r2l"] = pad_sequence(att_tgt_r2l_list, batch_first=True, padding_value=-100)
            result["target_lengths"] = torch.tensor(target_lens, dtype=torch.long)

        return result
```

3. Run verification:
   ```bash
   uv run python -c "
   from transformers import AutoTokenizer
   from utils.qwen_copyne_data import tokenize_entity_variants, build_bpe_copy_targets
   tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-ASR-0.6B-hf')
   ne_list = ['中原地产', '张大伟', 'Quilter']
   variants = [tokenize_entity_variants(tok, ne)[1] for ne in ne_list]
   sent_ids = tok.encode('中原地产首席分析师张大伟 meets Quilter', add_special_tokens=False) + [tok.eos_token_id]
   l2r, r2l, _ = build_bpe_copy_targets(sent_ids, variants, null_idx=len(ne_list))
   assert (l2r == 0).sum().item() == 1 and (l2r == 1).sum().item() == 1 and (l2r == 2).sum().item() == 1
   assert (r2l == 0).sum().item() == 1 and (r2l == 1).sum().item() == 1 and (r2l == 2).sum().item() == 1
   print('[PASS] Step 2 BPE L2R & R2L target builder verified.')
   "
   ```

---

### Step 3: Build `QwenCopyNEU2PlusPlus` Architecture (`model/qwen_copyne_u2pp.py`)

1. Create `model/qwen_copyne_u2pp.py` implementing all Path B components with clean DDP separation:
   - **`DualStackedLSTMContextEncoder`**: Frozen `qwen.get_input_embeddings()` $\to$ trainable `nn.Linear(d_model, 256)` $\to$ 3-layer L2R & R2L `nn.LSTM(256, 512, num_layers=3)` + trainable null vectors $z_0^{\text{L2R}}, z_0^{\text{R2L}} \in \mathbb{R}^{512}$.
   - **`CopyNEDotProductAttention`**: Scaled dot-product attention ($d_a = 512$) returning dictionary summary $r_u$ and $\log P_c(e_i \mid d_u)$.
   - **Warm-Started `W_out` (`base_out_proj` + `dict_out_proj`)**:
     - Supports `freeze_base_lm_head=True` (reuses frozen `qwen.lm_head.weight` directly) or `freeze_base_lm_head=False` (initializes `base_out_proj = nn.Linear(d_model, vocab_size, bias=False)` with `qwen.lm_head.weight.data.clone().float()` for `1e-5` differential LR on 16 GB T4).
     - Initializes `dict_out_proj = nn.Linear(d_e, vocab_size, bias=True)` to zeros (`0.0`) and trains at `1e-3`.
   - **`U2PlusPlusR2LDecoder`**:
     - Default on 2x T4 (16 GB/GPU): **Design U2++-A** (`r2l_mode="cross_attn"`), 3-layer causal R2L Transformer Decoder cross-attending to unpacked $H_{\text{audio}} \in \mathbb{R}^{B \times T_a \times d_{\text{model}}}$, plus R2L `CopyNEDotProductAttention` over $z_i^{\text{R2L}}$.
     - Also supports **Design U2++-B** (`r2l_mode="lightweight_rescorer"`).
2. Write the exact code below to `model/qwen_copyne_u2pp.py`:

```python
import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence


class DualStackedLSTMContextEncoder(nn.Module):
    """Trainable 3-layer L2R + R2L Stacked-LSTM NE Encoder over frozen Qwen3 token embeddings."""

    def __init__(
        self,
        frozen_embedding: nn.Embedding,
        d_model: int,
        emb_proj_dim: int = 256,
        d_e: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.frozen_embedding = frozen_embedding
        self.frozen_embedding.requires_grad_(False)
        self.emb_proj = nn.Linear(d_model, emb_proj_dim)
        self.lstm_l2r = nn.LSTM(
            input_size=emb_proj_dim,
            hidden_size=d_e,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.lstm_r2l = nn.LSTM(
            input_size=emb_proj_dim,
            hidden_size=d_e,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.null_l2r = nn.Parameter(torch.zeros(1, d_e))
        self.null_r2l = nn.Parameter(torch.zeros(1, d_e))
        self.d_e = d_e

    def forward(
        self,
        ne_tokens_l2r: torch.Tensor,
        ne_tokens_r2l: torch.Tensor,
        ne_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if ne_tokens_l2r.numel() == 0 or ne_lengths.numel() == 0:
            return self.null_l2r, self.null_r2l

        safe_l2r = ne_tokens_l2r.clamp(min=0)
        safe_r2l = ne_tokens_r2l.clamp(min=0)

        with torch.no_grad():
            raw_emb_l2r = self.frozen_embedding(safe_l2r).float()
            raw_emb_r2l = self.frozen_embedding(safe_r2l).float()

        x_l2r = self.emb_proj(raw_emb_l2r)
        x_r2l = self.emb_proj(raw_emb_r2l)

        lengths_cpu = ne_lengths.detach().cpu().clamp(min=1)
        packed_l2r = pack_padded_sequence(x_l2r, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_r2l = pack_padded_sequence(x_r2l, lengths_cpu, batch_first=True, enforce_sorted=False)

        _, (h_n_l2r, _) = self.lstm_l2r(packed_l2r)
        _, (h_n_r2l, _) = self.lstm_r2l(packed_r2l)

        z_l2r = h_n_l2r[-1]
        z_r2l = h_n_r2l[-1]

        Z_l2r = torch.cat([z_l2r, self.null_l2r], dim=0)
        Z_r2l = torch.cat([z_r2l, self.null_r2l], dim=0)
        return Z_l2r, Z_r2l


class CopyNEDotProductAttention(nn.Module):
    """CopyNE scaled dot-product attention over NE dictionary representations Z."""

    def __init__(self, d_dec: int, d_e: int = 512, d_a: int = 512):
        super().__init__()
        self.W_q = nn.Linear(d_dec, d_a, bias=True)
        self.W_k = nn.Linear(d_e, d_a, bias=True)
        self.scale = 1.0 / math.sqrt(d_a)

    def forward(
        self, d_u: torch.Tensor, Z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.W_q(d_u)
        k = self.W_k(Z)
        scores = torch.matmul(q, k.transpose(0, 1)) * self.scale
        copy_log_probs = F.log_softmax(scores, dim=-1)
        copy_probs = copy_log_probs.exp()
        r_u = torch.matmul(copy_probs, Z)
        return r_u, copy_log_probs, copy_probs


class U2PlusPlusR2LDecoder(nn.Module):
    """
    U2++ Right-to-Left (R2L) Look-Back Decoder & CopyNE Rescorer.
    Default on 2x T4: r2l_mode="cross_attn" (Design U2++-A, 3-layer causal Transformer Decoder over H_audio).
    """

    def __init__(
        self,
        frozen_embedding: nn.Embedding,
        frozen_lm_head_weight: nn.Parameter,
        d_model: int,
        d_e: int = 512,
        d_a: int = 512,
        r2l_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 4,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        r2l_mode: str = "cross_attn",
    ):
        super().__init__()
        self.frozen_embedding = frozen_embedding
        self.frozen_lm_head_weight = frozen_lm_head_weight
        self.r2l_mode = r2l_mode
        self.r2l_dim = r2l_dim

        if r2l_mode == "cross_attn":
            self.tok_proj = nn.Linear(d_model, r2l_dim)
            self.mem_proj = nn.Linear(d_model, r2l_dim)
            self.pos_emb = nn.Embedding(2048, r2l_dim)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=r2l_dim,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
            self.out_norm = nn.LayerNorm(r2l_dim)
        elif r2l_mode == "lightweight_rescorer":
            self.state_proj = nn.Linear(d_model, r2l_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=r2l_dim,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.out_norm = nn.LayerNorm(r2l_dim)
        else:
            raise ValueError(f"Unsupported r2l_mode: {r2l_mode}")

        self.copy_attn_r2l = CopyNEDotProductAttention(d_dec=r2l_dim, d_e=d_e, d_a=d_a)
        self.r2l_to_lm = nn.Linear(r2l_dim + d_e, d_model, bias=True)

    def forward(
        self,
        r2l_target_ids: torch.Tensor,
        target_lengths: torch.Tensor,
        H_audio: torch.Tensor,
        audio_pad_mask: torch.Tensor,
        Z_r2l: torch.Tensor,
        eos_token_id: int,
        d_l2r_padded: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, U = r2l_target_ids.shape
        device = r2l_target_ids.device
        causal_mask = nn.Transformer.generate_square_subsequent_mask(U, device=device)
        tgt_pad_mask = torch.arange(U, device=device).unsqueeze(0) >= target_lengths.unsqueeze(1)

        if self.r2l_mode == "cross_attn":
            bos_col = torch.full((B, 1), eos_token_id, dtype=torch.long, device=device)
            r2l_in_ids = torch.cat([bos_col, r2l_target_ids[:, :-1].clamp(min=0)], dim=1)
            with torch.no_grad():
                tok_emb = self.frozen_embedding(r2l_in_ids).float()
            pos_ids = torch.arange(U, device=device).unsqueeze(0)
            x = self.tok_proj(tok_emb) + self.pos_emb(pos_ids)
            mem = self.mem_proj(H_audio.float())
            d_r2l = self.transformer(
                tgt=x,
                memory=mem,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=audio_pad_mask,
            )
            d_r2l = self.out_norm(d_r2l)
        else:
            assert d_l2r_padded is not None, "d_l2r_padded required for lightweight_rescorer"
            rev_states = []
            for b in range(B):
                ub = int(target_lengths[b].item())
                rev_b = torch.flip(d_l2r_padded[b, :ub, :], dims=[0])
                if ub < U:
                    pad_b = torch.zeros((U - ub, d_l2r_padded.size(-1)), device=device, dtype=d_l2r_padded.dtype)
                    rev_b = torch.cat([rev_b, pad_b], dim=0)
                rev_states.append(rev_b)
            x = self.state_proj(torch.stack(rev_states, dim=0).float())
            d_r2l = self.transformer(
                src=x,
                mask=causal_mask,
                src_key_padding_mask=tgt_pad_mask,
            )
            d_r2l = self.out_norm(d_r2l)

        r_u_r2l, r2l_copy_log_probs, r2l_copy_probs = self.copy_attn_r2l(d_r2l, Z_r2l)
        cat_r2l = torch.cat([d_r2l, r_u_r2l], dim=-1)
        lm_hidden = self.r2l_to_lm(cat_r2l).to(self.frozen_lm_head_weight.dtype)
        r2l_logits = F.linear(lm_hidden, self.frozen_lm_head_weight).float()
        r2l_vocab_log_probs = F.log_softmax(r2l_logits, dim=-1)
        return r2l_vocab_log_probs, r2l_copy_log_probs, r2l_copy_probs


class QwenCopyNEU2PlusPlus(nn.Module):
    """
    Complete Frozen Qwen3-ASR-0.6B + Trainable CopyNE + U2++ Bidirectional Look-Back Model.
    """

    def __init__(
        self,
        qwen_model,
        d_e: int = 512,
        d_a: int = 512,
        emb_proj_dim: int = 256,
        lstm_layers: int = 3,
        r2l_mode: str = "cross_attn",
        r2l_dim: int = 512,
        r2l_layers: int = 3,
        alpha_r2l: float = 0.3,
        freeze_base_lm_head: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.qwen = qwen_model
        self.qwen.eval()
        self.qwen.requires_grad_(False)

        self.d_model = self.qwen.config.text_config.hidden_size
        self.vocab_size = self.qwen.config.text_config.vocab_size
        self.audio_token_id = self.qwen.config.audio_token_id
        self.eos_token_id = 151645  # <|im_end|>
        self.d_e = d_e
        self.alpha_r2l = alpha_r2l
        self.freeze_base_lm_head = freeze_base_lm_head

        frozen_emb = self.qwen.get_input_embeddings()
        frozen_lm_head_weight = self.qwen.lm_head.weight

        # 1. Trainable 3-layer Dual Stacked-LSTM NE Encoder
        self.ne_encoder = DualStackedLSTMContextEncoder(
            frozen_embedding=frozen_emb,
            d_model=self.d_model,
            emb_proj_dim=emb_proj_dim,
            d_e=d_e,
            num_layers=lstm_layers,
            dropout=dropout,
        )

        # 2. Trainable L2R CopyNE Dot-Product Attention
        self.copy_attn_l2r = CopyNEDotProductAttention(
            d_dec=self.d_model,
            d_e=d_e,
            d_a=d_a,
        )

        # 3. Warm-Started Dict-Enhanced Final Linear Layer W_out = [W_base | W_dict] + b_out
        if not freeze_base_lm_head:
            self.base_out_proj = nn.Linear(self.d_model, self.vocab_size, bias=False)
            with torch.no_grad():
                self.base_out_proj.weight.copy_(frozen_lm_head_weight.detach().float())
        else:
            self.base_out_proj = None

        self.dict_out_proj = nn.Linear(d_e, self.vocab_size, bias=True)
        nn.init.zeros_(self.dict_out_proj.weight)
        nn.init.zeros_(self.dict_out_proj.bias)

        # 4. Trainable U2++ Bidirectional R2L Module (only instantiated if alpha_r2l > 0)
        if self.alpha_r2l > 0.0:
            self.r2l_decoder = U2PlusPlusR2LDecoder(
                frozen_embedding=frozen_emb,
                frozen_lm_head_weight=frozen_lm_head_weight,
                d_model=self.d_model,
                d_e=d_e,
                d_a=d_a,
                r2l_dim=r2l_dim,
                num_layers=r2l_layers,
                r2l_mode=r2l_mode,
                dropout=dropout,
            )
        else:
            self.r2l_decoder = None

    def get_optimizer_param_groups(self, lr_modules: float = 1e-3, lr_base_lm: float = 1e-5, weight_decay: float = 0.01):
        """Return differential learning-rate parameter groups (1e-3 for new modules, 1e-5 for warm-started base_out_proj)."""
        fast_params = []
        slow_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("base_out_proj."):
                slow_params.append(param)
            else:
                fast_params.append(param)
        groups = [{"params": fast_params, "lr": lr_modules, "weight_decay": weight_decay}]
        if slow_params:
            groups.append({"params": slow_params, "lr": lr_base_lm, "weight_decay": weight_decay})
        return groups

    def extract_frozen_backbone_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor,
        s0_indices: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self.qwen.eval()
            qwen_dtype = self.qwen.lm_head.weight.dtype
            outputs = self.qwen.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features.to(qwen_dtype),
                input_features_mask=input_features_mask,
                return_dict=True,
            )
            H_last = outputs.last_hidden_state
            audio_flat = outputs.audio_hidden_states

            audio_lens = (input_ids == self.audio_token_id).sum(dim=1).tolist()
            audio_chunks = torch.split(audio_flat, audio_lens, dim=0)
            H_audio = pad_sequence(audio_chunks, batch_first=True, padding_value=0.0)
            max_T_a = H_audio.size(1)
            audio_pad_mask = (
                torch.arange(max_T_a, device=H_audio.device).unsqueeze(0)
                >= torch.tensor(audio_lens, device=H_audio.device).unsqueeze(1)
            )

            d_list = []
            for b in range(input_ids.size(0)):
                s0 = int(s0_indices[b].item())
                ub = int(target_lengths[b].item())
                d_list.append(H_last[b, s0 : s0 + ub, :].float())
            d_l2r_padded = pad_sequence(d_list, batch_first=True, padding_value=0.0)

        return d_l2r_padded, H_audio, audio_pad_mask

    def compute_l2r_distributions(
        self, d_u: torch.Tensor, Z_l2r: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r_u, copy_log_probs, copy_probs = self.copy_attn_l2r(d_u, Z_l2r)
        if self.base_out_proj is not None:
            base_logits = self.base_out_proj(d_u)
        else:
            qwen_dtype = self.qwen.lm_head.weight.dtype
            base_logits = F.linear(d_u.to(qwen_dtype), self.qwen.lm_head.weight).float()
        dict_logits = self.dict_out_proj(r_u)
        vocab_log_probs = F.log_softmax(base_logits + dict_logits, dim=-1)
        return vocab_log_probs, copy_log_probs, copy_probs

    def forward(self, batch: dict) -> Dict[str, torch.Tensor]:
        if "cached_d_l2r" in batch:
            d_l2r = batch["cached_d_l2r"].float()
            H_audio = batch["cached_H_audio"]
            audio_pad_mask = batch["cached_audio_pad_mask"]
        else:
            d_l2r, H_audio, audio_pad_mask = self.extract_frozen_backbone_states(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                input_features=batch["input_features"],
                input_features_mask=batch["input_features_mask"],
                s0_indices=batch["s0_indices"],
                target_lengths=batch["target_lengths"],
            )

        Z_l2r, Z_r2l = self.ne_encoder(
            ne_tokens_l2r=batch["ne_tokens_l2r"],
            ne_tokens_r2l=batch["ne_tokens_r2l"],
            ne_lengths=batch["ne_lengths"],
        )

        l2r_vocab_logp, l2r_copy_logp, _ = self.compute_l2r_distributions(d_l2r, Z_l2r)
        target_l2r = batch["target_l2r"]
        att_tgt_l2r = batch["att_tgt_l2r"]

        loss_l2r_vocab = F.nll_loss(
            l2r_vocab_logp.reshape(-1, self.vocab_size),
            target_l2r.reshape(-1),
            ignore_index=-100,
            reduction="mean",
        )
        loss_l2r_copy = F.nll_loss(
            l2r_copy_logp.reshape(-1, Z_l2r.size(0)),
            att_tgt_l2r.reshape(-1),
            ignore_index=-100,
            reduction="mean",
        )
        loss_l2r = loss_l2r_vocab + loss_l2r_copy

        if self.alpha_r2l > 0.0 and self.r2l_decoder is not None:
            r2l_vocab_logp, r2l_copy_logp, _ = self.r2l_decoder(
                r2l_target_ids=batch["target_r2l"],
                target_lengths=batch["target_lengths"],
                H_audio=H_audio,
                audio_pad_mask=audio_pad_mask,
                Z_r2l=Z_r2l,
                eos_token_id=self.eos_token_id,
                d_l2r_padded=d_l2r,
            )
            loss_r2l_vocab = F.nll_loss(
                r2l_vocab_logp.reshape(-1, self.vocab_size),
                batch["target_r2l"].reshape(-1),
                ignore_index=-100,
                reduction="mean",
            )
            loss_r2l_copy = F.nll_loss(
                r2l_copy_logp.reshape(-1, Z_r2l.size(0)),
                batch["att_tgt_r2l"].reshape(-1),
                ignore_index=-100,
                reduction="mean",
            )
            loss_r2l = loss_r2l_vocab + loss_r2l_copy
            loss_total = (1.0 - self.alpha_r2l) * loss_l2r + self.alpha_r2l * loss_r2l
        else:
            loss_r2l_vocab = torch.zeros_like(loss_l2r)
            loss_r2l_copy = torch.zeros_like(loss_l2r)
            loss_r2l = torch.zeros_like(loss_l2r)
            loss_total = loss_l2r

        return {
            "loss": loss_total,
            "loss_l2r": loss_l2r,
            "loss_l2r_vocab": loss_l2r_vocab,
            "loss_l2r_copy": loss_l2r_copy,
            "loss_r2l": loss_r2l,
            "loss_r2l_vocab": loss_r2l_vocab,
            "loss_r2l_copy": loss_r2l_copy,
        }

    @torch.no_grad()
    def recognize_two_pass(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor,
        Z_l2r: torch.Tensor,
        Z_r2l: torch.Tensor,
        ne_tokens_l2r: torch.Tensor,
        ne_lengths: torch.Tensor,
        beam_size: int = 5,
        copy_threshold: float = 0.7,
        use_r2l_rescoring: bool = True,
        max_new_tokens: int = 128,
    ) -> List[List[int]]:
        self.eval()
        device = input_ids.device
        qwen_dtype = self.qwen.lm_head.weight.dtype
        B = input_ids.size(0)
        N_ent = Z_l2r.size(0) - 1

        base_out = self.qwen.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features.to(qwen_dtype),
            input_features_mask=input_features_mask,
            use_cache=True,
            return_dict=True,
        )
        audio_flat = base_out.audio_hidden_states
        audio_lens = (input_ids == self.audio_token_id).sum(dim=1).tolist()
        audio_chunks = torch.split(audio_flat, audio_lens, dim=0)
        H_audio_all = pad_sequence(audio_chunks, batch_first=True, padding_value=0.0)
        max_T_a = H_audio_all.size(1)
        audio_pad_mask_all = (
            torch.arange(max_T_a, device=device).unsqueeze(0)
            >= torch.tensor(audio_lens, device=device).unsqueeze(1)
        )

        best_predictions: List[List[int]] = []

        for b in range(B):
            valid_pos = attention_mask[b].nonzero(as_tuple=True)[0]
            prefix_ids = input_ids[b, valid_pos].unsqueeze(0)
            audio_b = audio_chunks[b]

            prefix_embeds = self.qwen.get_input_embeddings()(prefix_ids)
            audio_mask_b = (prefix_ids == self.audio_token_id).unsqueeze(-1).expand_as(prefix_embeds)
            prefix_embeds = prefix_embeds.masked_scatter(audio_mask_b, audio_b.to(prefix_embeds.dtype))

            beams: List[Tuple[float, List[int], bool]] = [(0.0, [], False)]

            for _ in range(max_new_tokens):
                if all(fin for _, _, fin in beams):
                    break

                new_candidates: List[Tuple[float, List[int], bool]] = []
                active_indices = [i for i, (_, _, fin) in enumerate(beams) if not fin]
                for i, (sc, seq, fin) in enumerate(beams):
                    if fin:
                        new_candidates.append((sc, seq, True))

                if not active_indices:
                    break

                d_u_list = []
                for idx in active_indices:
                    _, seq, _ = beams[idx]
                    if len(seq) > 0:
                        gen_ids = torch.tensor([seq], dtype=torch.long, device=device)
                        gen_embeds = self.qwen.get_input_embeddings()(gen_ids)
                        full_embeds = torch.cat([prefix_embeds, gen_embeds], dim=1)
                    else:
                        full_embeds = prefix_embeds
                    lm_out = self.qwen.model.language_model(inputs_embeds=full_embeds, use_cache=False, return_dict=True)
                    d_u_list.append(lm_out.last_hidden_state[:, -1:, :].float())

                d_u_batch = torch.cat(d_u_list, dim=0)
                vocab_logp, copy_logp, copy_p = self.compute_l2r_distributions(d_u_batch, Z_l2r)
                vocab_p = vocab_logp.squeeze(1).exp()
                copy_p = copy_p.squeeze(1)

                for k_i, beam_idx in enumerate(active_indices):
                    base_score, seq, _ = beams[beam_idx]
                    p_c = copy_p[k_i].clone()
                    if N_ent > 0 and copy_threshold > 0.0:
                        max_ne_prob = p_c[:-1].max().item()
                        if max_ne_prob <= copy_threshold:
                            p_c[:-1] = 0.0
                            p_c[-1] = 1.0

                    q_vocab = (p_c[-1] * vocab_p[k_i]).clamp(min=1e-12).log()
                    if N_ent > 0 and p_c[:-1].max().item() > 0.0:
                        q_ne = p_c[:-1].clamp(min=1e-12).log()
                        q_unified = torch.cat([q_vocab, q_ne], dim=0)
                    else:
                        q_unified = q_vocab

                    topk_logp, topk_idx = torch.topk(q_unified, k=beam_size)
                    for step_logp, cand_id in zip(topk_logp.tolist(), topk_idx.tolist()):
                        if cand_id < self.vocab_size:
                            next_seq = seq + [cand_id]
                            fin = cand_id in (151643, 151645)
                            new_candidates.append((base_score + step_logp, next_seq, fin))
                        else:
                            ent_idx = cand_id - self.vocab_size
                            ent_len = int(ne_lengths[ent_idx].item())
                            ent_toks = ne_tokens_l2r[ent_idx, :ent_len].tolist()
                            next_seq = seq + ent_toks
                            new_candidates.append((base_score + step_logp, next_seq, False))

                new_candidates.sort(key=lambda x: x[0] / max(1, len(x[1])), reverse=True)
                beams = new_candidates[:beam_size]

            if use_r2l_rescoring and self.alpha_r2l > 0.0 and self.r2l_decoder is not None and len(beams) > 1:
                r2l_seqs = []
                r2l_lens = []
                for _, seq, _ in beams:
                    clean_seq = [t for t in seq if t not in (151643, 151645)]
                    r2l_ids = list(reversed(clean_seq)) + [self.eos_token_id]
                    r2l_seqs.append(torch.tensor(r2l_ids, dtype=torch.long, device=device))
                    r2l_lens.append(len(r2l_ids))

                r2l_pad = pad_sequence(r2l_seqs, batch_first=True, padding_value=-100)
                r2l_len_t = torch.tensor(r2l_lens, dtype=torch.long, device=device)
                H_aud_rep = H_audio_all[b : b + 1].expand(len(beams), -1, -1)
                aud_mask_rep = audio_pad_mask_all[b : b + 1].expand(len(beams), -1)

                r2l_vocab_logp, r2l_copy_logp, _ = self.r2l_decoder(
                    r2l_target_ids=r2l_pad,
                    target_lengths=r2l_len_t,
                    H_audio=H_aud_rep,
                    audio_pad_mask=aud_mask_rep,
                    Z_r2l=Z_r2l,
                    eos_token_id=self.eos_token_id,
                )

                best_combined = -float("inf")
                best_seq: List[int] = beams[0][1]
                for n_i, (l2r_score, seq, _) in enumerate(beams):
                    u_len = r2l_lens[n_i]
                    tgt_n = r2l_seqs[n_i]
                    tok_logp = r2l_vocab_logp[n_i, :u_len, :].gather(-1, tgt_n.unsqueeze(-1)).squeeze(-1).sum().item()
                    combined = (1.0 - self.alpha_r2l) * l2r_score + self.alpha_r2l * tok_logp
                    if combined > best_combined:
                        best_combined = combined
                        best_seq = seq
                best_predictions.append([t for t in best_seq if t not in (151643, 151645)])
            else:
                best_predictions.append([t for t in beams[0][1] if t not in (151643, 151645)])

        return best_predictions
```

---

### Step 4: Kaggle 2x T4 Distributed (DDP) Training Pipeline (`train_qwen_copyne.py` & `conf/qwen3_copyne_u2pp.yaml`)

1. Create `conf/qwen3_copyne_u2pp.yaml` tuned for Kaggle 2x T4 (`16 GB` per GPU):
   ```yaml
   qwen_model_id: "Qwen/Qwen3-ASR-0.6B-hf"
   d_e: 512
   d_a: 512
   emb_proj_dim: 256
   lstm_layers: 3
   r2l_mode: "cross_attn"      # Design U2++-A (3-layer R2L Transformer over H_audio) fits easily in 16GB T4
   r2l_dim: 512
   r2l_layers: 3
   alpha_r2l: 0.3              # 0.0 for Row 3 (L2R-only ablation), 0.3 for Row 4 (Full U2++)
   freeze_base_lm_head: true   # Keep W_out[:, :d_model] frozen to qwen.lm_head.weight (set false to tune at lr_base_lm=1e-5)
   dropout: 0.1
   beta_distractors: 2
   batch_size: 8               # 8 per T4 GPU * 2 GPUs * accum_grad 2 = 32 effective batch size
   accum_grad: 2
   max_epoch: 15
   lr_modules: 0.001           # 1e-3 for 3-LSTM NE encoder, CopyNE attention, dict_out_proj, U2++ R2L module
   lr_base_lm: 0.00001         # 1e-5 if freeze_base_lm_head is false
   warmup_steps: 2000
   grad_clip: 5.0
   log_interval: 50
   ```
2. Create `train_qwen_copyne.py` with full 2x T4 `torchrun` DDP support (`nccl`, `DistributedSampler`, `broadcast_buffers=False`, Rank-0 dev eval surrounded by `dist.barrier()`, `--max_hours` for Kaggle 12h session limits, and `--pre_model` checkpoint resumption):

```python
import argparse
import os
from datetime import datetime, timedelta
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration, get_linear_schedule_with_warmup

from model.qwen_copyne_u2pp import QwenCopyNEU2PlusPlus
from utils.qwen_copyne_data import QwenCopyNECollator, QwenCopyNEDataset
from supar.utils.logging import init_logger, logger


def setup_ddp() -> Tuple[bool, int, int]:
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
        torch.cuda.set_device(local_rank)
        return True, local_rank, world_size
    return False, 0, 1


def get_optimal_backbone_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def get_trainable_state_dict(raw_model: QwenCopyNEU2PlusPlus) -> dict:
    return {
        k: v.cpu()
        for k, v in raw_model.state_dict().items()
        if not k.startswith("qwen.")
    }


def main():
    is_ddp, local_rank, world_size = setup_ddp()

    parser = argparse.ArgumentParser(description="Train Frozen Qwen3-ASR + CopyNE + U2++ on 2x T4")
    parser.add_argument("--config", default="conf/qwen3_copyne_u2pp.yaml")
    parser.add_argument("--path", required=True, help="Experiment output directory")
    parser.add_argument("--pre_model", default="None", help="Checkpoint path to resume across Kaggle sessions")
    parser.add_argument("--train", default="data_to_upload/aishell_dataset/train_addne.json")
    parser.add_argument("--dev", default="data_to_upload/aishell_dataset/dev_addne.json")
    parser.add_argument("--train_ne_dict", default="data_to_upload/aishell_vocab/train_ne.vocab")
    parser.add_argument("--dev_ne_dict", default="data_to_upload/aishell_vocab/dev_ne.vocab")
    parser.add_argument("--alpha_r2l", type=float, default=None, help="Override alpha_r2l (0.0 for L2R-only ablation)")
    parser.add_argument("--r2l_mode", type=str, default=None, choices=["cross_attn", "lightweight_rescorer"])
    parser.add_argument("--max_hours", type=float, default=11.0, help="Stop cleanly before Kaggle 12h timeout")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional cap for smoke testing")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()

    if local_rank == 0:
        os.makedirs(args.path, exist_ok=True)
    if is_ddp:
        dist.barrier()

    init_logger(logger, os.path.join(args.path, f"train_rank{local_rank}.log"))
    torch.manual_seed(args.seed + local_rank)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.alpha_r2l is not None:
        cfg["alpha_r2l"] = args.alpha_r2l
    if args.r2l_mode is not None:
        cfg["r2l_mode"] = args.r2l_mode

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = get_optimal_backbone_dtype()

    processor = AutoProcessor.from_pretrained(cfg["qwen_model_id"])
    qwen = Qwen3ASRForConditionalGeneration.from_pretrained(cfg["qwen_model_id"], torch_dtype=dtype).to(device)

    raw_model = QwenCopyNEU2PlusPlus(
        qwen_model=qwen,
        d_e=cfg["d_e"],
        d_a=cfg["d_a"],
        emb_proj_dim=cfg["emb_proj_dim"],
        lstm_layers=cfg["lstm_layers"],
        r2l_mode=cfg["r2l_mode"],
        r2l_dim=cfg["r2l_dim"],
        r2l_layers=cfg["r2l_layers"],
        alpha_r2l=cfg["alpha_r2l"],
        freeze_base_lm_head=cfg.get("freeze_base_lm_head", True),
        dropout=cfg["dropout"],
    ).to(device)

    train_set = QwenCopyNEDataset(args.train, args.train_ne_dict, is_training=True, beta_distractors=cfg["beta_distractors"])
    train_collator = QwenCopyNECollator(processor, train_set.ne_vocab, is_training=True, is_dev=False, beta_distractors=cfg["beta_distractors"])
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=local_rank, shuffle=True) if is_ddp else None
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=train_collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if local_rank == 0:
        dev_set = QwenCopyNEDataset(args.dev, args.dev_ne_dict, is_training=False, beta_distractors=cfg["beta_distractors"])
        dev_collator = QwenCopyNECollator(processor, dev_set.ne_vocab, is_training=False, is_dev=True, beta_distractors=cfg["beta_distractors"])
        dev_loader = DataLoader(dev_set, batch_size=cfg["batch_size"], shuffle=False, collate_fn=dev_collator, num_workers=args.num_workers)
    else:
        dev_loader = None

    param_groups = raw_model.get_optimizer_param_groups(
        lr_modules=cfg["lr_modules"],
        lr_base_lm=cfg.get("lr_base_lm", 1e-5),
        weight_decay=0.01,
    )
    optimizer = AdamW(param_groups)
    total_steps = max(1, (len(train_loader) * cfg["max_epoch"]) // cfg["accum_grad"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg["warmup_steps"], num_training_steps=total_steps)

    start_epoch = 1
    best_dev_loss = float("inf")

    if args.pre_model != "None" and os.path.exists(args.pre_model):
        ckpt = torch.load(args.pre_model, map_location=device)
        if "model_state_dict" in ckpt:
            raw_model.load_state_dict(ckpt["model_state_dict"], strict=False)
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_dev_loss = ckpt.get("best_dev_loss", float("inf"))
            logger.info(f"Rank:{local_rank} resumed from {args.pre_model} at epoch {start_epoch}")
        else:
            raw_model.load_state_dict(ckpt, strict=False)

    model = (
        DDP(raw_model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
        if is_ddp
        else raw_model
    )

    trainable_params = [p for p in raw_model.parameters() if p.requires_grad]
    accum_grad = cfg["accum_grad"]
    train_start = datetime.now()

    for epoch in range(start_epoch, cfg["max_epoch"] + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        raw_model.qwen.eval()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            batch_gpu = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(batch_gpu)
            loss = out["loss"] / accum_grad
            loss.backward()

            if step % accum_grad == 0 or step == len(train_loader):
                nn.utils.clip_grad_norm_(trainable_params, cfg["grad_clip"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % cfg["log_interval"] == 0 and local_rank == 0:
                logger.info(
                    f"Epoch {epoch} Step {step}/{len(train_loader)} "
                    f"loss={out['loss'].item():.4f} l2r={out['loss_l2r'].item():.4f} "
                    f"copy_l2r={out['loss_l2r_copy'].item():.4f} r2l={out['loss_r2l'].item():.4f}"
                )

            if args.max_steps is not None and step >= args.max_steps:
                break

        # Barrier 1 before Rank 0 dev evaluation (prevents NCCL timeout on 2x T4)
        if is_ddp:
            dist.barrier()

        if local_rank == 0 and dev_loader is not None:
            raw_model.eval()
            dev_loss_sum = 0.0
            dev_steps = 0
            with torch.no_grad():
                for batch in dev_loader:
                    batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    out = raw_model(batch_gpu)
                    dev_loss_sum += out["loss"].item()
                    dev_steps += 1
                    if args.max_steps is not None and dev_steps >= args.max_steps:
                        break

            avg_dev_loss = dev_loss_sum / max(1, dev_steps)
            logger.info(f"Epoch {epoch} avg_dev_loss={avg_dev_loss:.4f}")

            ckpt_payload = {
                "epoch": epoch,
                "model_state_dict": get_trainable_state_dict(raw_model),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dev_loss": min(best_dev_loss, avg_dev_loss),
            }
            torch.save(ckpt_payload, os.path.join(args.path, "current.model"))
            if avg_dev_loss < best_dev_loss:
                best_dev_loss = avg_dev_loss
                torch.save(ckpt_payload, os.path.join(args.path, "best.model"))

        # Barrier 2 after Rank 0 finishes saving
        if is_ddp:
            dist.barrier()

        torch.cuda.empty_cache()

        hours_elapsed = (datetime.now() - train_start).total_seconds() / 3600.0
        if args.max_hours is not None and hours_elapsed >= args.max_hours:
            logger.info(f"Rank:{local_rank} reached max_hours={args.max_hours:.2f}h ({hours_elapsed:.2f}h). Stopping cleanly.")
            break

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

3. Launch commands on Kaggle 2x T4:
   ```bash
   # Row 3: L2R CopyNE only (alpha_r2l = 0.0) across 2x T4 GPUs
   uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 train_qwen_copyne.py \
       --config conf/qwen3_copyne_u2pp.yaml \
       --path exp/qwen3_copyne_l2r \
       --alpha_r2l 0.0 \
       --max_hours 11.0

   # Row 4: Full L2R + U2++ R2L Rescoring (alpha_r2l = 0.3) across 2x T4 GPUs
   uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 train_qwen_copyne.py \
       --config conf/qwen3_copyne_u2pp.yaml \
       --path exp/qwen3_copyne_u2pp \
       --alpha_r2l 0.3 \
       --max_hours 11.0
   ```

---

### Step 5: Evaluation & Thesis 4-Row Ablation Matrix (`evaluate_qwen_copyne.py`)

1. Create `evaluate_qwen_copyne.py` using `get_optimal_backbone_dtype()` (`torch.float16` on T4) and handling both raw state dicts and `{"model_state_dict": ...}` checkpoints saved by `train_qwen_copyne.py`:

```python
import argparse
import os
import yaml
import orjson
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

from model.qwen_copyne_u2pp import QwenCopyNEU2PlusPlus
from utils.qwen_copyne_data import (
    QwenCopyNECollator,
    QwenCopyNEDataset,
    load_jsonl_orjson,
    tokenize_entity_variants,
)


def get_optimal_backbone_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def compute_entity_recall(json_path: str, pred_asr_path: str) -> dict:
    records = {r["key"]: r for r in load_jsonl_orjson(json_path)}
    preds = {}
    with open(pred_asr_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                preds[parts[0]] = parts[1].strip()
            elif len(parts) == 1:
                preds[parts[0]] = ""

    total_entities = 0
    recalled_entities = 0
    for key, rec in records.items():
        if key not in preds:
            continue
        hyp = preds[key]
        sent = rec["txt"]
        for _, st, ed in rec.get("ne_lst", []):
            if ed > st:
                ne_str = sent[st : ed + 1]
                total_entities += 1
                if ne_str in hyp:
                    recalled_entities += 1

    recall_pct = (100.0 * recalled_entities / total_entities) if total_entities > 0 else 0.0
    return {"entity_recall_pct": round(recall_pct, 2), "total_entities": total_entities, "recalled": recalled_entities}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-ASR + CopyNE + U2++ Ablations")
    parser.add_argument("--config", default="conf/qwen3_copyne_u2pp.yaml")
    parser.add_argument("--path", default="", help="Checkpoint dir containing best.model")
    parser.add_argument("--input", default="data_to_upload/aishell_dataset/test-ne.json")
    parser.add_argument("--test_ne_dict", default="data_to_upload/aishell_vocab/test_ne.vocab")
    parser.add_argument("--res", required=True, help="Output prefix (writes <res>.asr)")
    parser.add_argument("--mode", required=True, choices=["zero_shot", "prompt_hotwords", "copyne_l2r", "copyne_u2pp"])
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--copy_threshold", type=float, default=0.7)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = get_optimal_backbone_dtype()

    processor = AutoProcessor.from_pretrained(cfg["qwen_model_id"])
    qwen = Qwen3ASRForConditionalGeneration.from_pretrained(cfg["qwen_model_id"], torch_dtype=dtype).to(device)
    qwen.eval()

    dataset = QwenCopyNEDataset(args.input, args.test_ne_dict, is_training=False)
    collator = QwenCopyNECollator(processor, dataset.ne_vocab, is_training=False, is_dev=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=2)

    asr_out_path = args.res + ".asr"
    os.makedirs(os.path.dirname(os.path.abspath(asr_out_path)), exist_ok=True)

    if args.mode in ("zero_shot", "prompt_hotwords"):
        hotword_str = ", ".join(dataset.ne_vocab[:200]) if args.mode == "prompt_hotwords" else None
        with open(asr_out_path, "w", encoding="utf-8") as fout:
            for i in range(len(dataset)):
                item = dataset[i]
                req = processor.apply_transcription_request(
                    audio=item["audio_np"],
                    language="Chinese",
                    prompt=hotword_str,
                    sampling_rate=item["sr"],
                )
                req = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in req.items()}
                req["input_features"] = req["input_features"].to(dtype)
                with torch.no_grad():
                    gen_ids = qwen.generate(**req, max_new_tokens=128)
                prompt_len = req["input_ids"].size(1)
                hyp = processor.decode(gen_ids[0, prompt_len:], return_format="transcription_only").strip()
                fout.write(f"{item['key']} {hyp}\n")
    else:
        alpha = cfg["alpha_r2l"] if args.mode == "copyne_u2pp" else 0.0
        model = QwenCopyNEU2PlusPlus(
            qwen_model=qwen,
            d_e=cfg["d_e"],
            d_a=cfg["d_a"],
            emb_proj_dim=cfg["emb_proj_dim"],
            lstm_layers=cfg["lstm_layers"],
            r2l_mode=cfg["r2l_mode"],
            r2l_dim=cfg["r2l_dim"],
            r2l_layers=cfg["r2l_layers"],
            alpha_r2l=alpha,
            freeze_base_lm_head=cfg.get("freeze_base_lm_head", True),
            dropout=cfg["dropout"],
        ).to(device)
        ckpt_path = os.path.join(args.path, "best.model")
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        model.eval()

        ne_l2r_t, ne_r2l_t = [], []
        for ne_str in dataset.ne_vocab:
            c_ids, _ = tokenize_entity_variants(processor.tokenizer, ne_str)
            if not c_ids:
                c_ids = [processor.tokenizer.pad_token_id]
            ne_l2r_t.append(torch.tensor(c_ids, dtype=torch.long))
            ne_r2l_t.append(torch.tensor(list(reversed(c_ids)), dtype=torch.long))

        ne_tokens_l2r = pad_sequence(ne_l2r_t, batch_first=True, padding_value=-1).to(device)
        ne_tokens_r2l = pad_sequence(ne_r2l_t, batch_first=True, padding_value=-1).to(device)
        ne_lengths = torch.tensor([t.numel() for t in ne_l2r_t], dtype=torch.long, device=device)

        with torch.no_grad():
            Z_l2r, Z_r2l = model.ne_encoder(ne_tokens_l2r, ne_tokens_r2l, ne_lengths)

        with open(asr_out_path, "w", encoding="utf-8") as fout:
            for batch in loader:
                hyps_ids = model.recognize_two_pass(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    input_features=batch["input_features"].to(device),
                    input_features_mask=batch["input_features_mask"].to(device),
                    Z_l2r=Z_l2r,
                    Z_r2l=Z_r2l,
                    ne_tokens_l2r=ne_tokens_l2r,
                    ne_lengths=ne_lengths,
                    beam_size=args.beam_size,
                    copy_threshold=args.copy_threshold,
                    use_r2l_rescoring=(args.mode == "copyne_u2pp"),
                )
                for key, tok_ids in zip(batch["keys"], hyps_ids):
                    hyp_str = processor.tokenizer.decode(tok_ids, skip_special_tokens=True).replace(" ", "").strip()
                    fout.write(f"{key} {hyp_str}\n")

    stats = compute_entity_recall(args.input, asr_out_path)
    with open(asr_out_path + ".entity_recall.json", "wb") as f:
        f.write(orjson.dumps(stats))
    print(f"[Eval Complete] {asr_out_path} -> {stats}")


if __name__ == "__main__":
    main()
```

2. Run `compute-necer.sh` on each `<res>.asr` output to populate the 4-row thesis table in `RESULT.md`.
