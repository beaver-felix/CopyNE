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

