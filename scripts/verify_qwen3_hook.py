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

