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

    def get_trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        """Extract only trainable modules state dict, skipping frozen Qwen backbone to save memory."""
        trainable_keys = {name for name, p in self.named_parameters() if p.requires_grad}
        trainable_buffers = {name for name, b in self.named_buffers() if not name.startswith("qwen.")}
        allowed = trainable_keys | trainable_buffers
        return {k: v.cpu() for k, v in self.state_dict().items() if k in allowed or not k.startswith("qwen.")}

    def load_trainable_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = False):
        """Load trainable state dict with flexible DDP 'module.' prefix handling."""
        clean_sd = {}
        for k, v in state_dict.items():
            k_clean = k[7:] if k.startswith("module.") else k
            clean_sd[k_clean] = v
        return self.load_state_dict(clean_sd, strict=strict)

