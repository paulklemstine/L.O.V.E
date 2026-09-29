#!/usr/bin/env python3
"""Encode prompts longer than CLIP's 77-token window.

CLIP's text encoders have a hard 77-token limit (a learned position-embedding
table, `max_position_embeddings: 77` in the checkpoint config). diffusers
silently truncates past that point, so any tail of the prompt is discarded
without an error -- the image still renders, just missing whatever was cut.

Each encoder can only ever see 75 content tokens per forward pass, but the
UNet's cross-attention accepts a sequence of any length. So we split the prompt
into 75-token chunks, encode each independently, and concatenate the resulting
`last_hidden_state` sequences. Every token in the prompt then reaches the UNet.

Both SDXL text encoders must agree on the chunk count (their outputs are
concatenated on the feature dim), and classifier-free guidance requires the
negative embedding to have the same sequence length as the positive one -- so
the shorter side is padded out with EOS-filled chunks rather than truncated.
"""
import torch


def _chunk_ids(tokenizer, text, chunk=75):
    ids = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
    return [ids[i:i + chunk] for i in range(0, len(ids), chunk)] or [[]]


def num_chunks(tokenizer, text, chunk=75):
    return max(1, -(-len(tokenizer(text, truncation=False,
                                   add_special_tokens=False)["input_ids"]) // chunk))


def _pooled(out):
    """text_encoder_2 returns CLIPTextModelOutput (text_embeds); the first
    encoder is a plain CLIPTextModel (pooler_output). Only encoder 2's pooled
    vector is used for SDXL, but handle both shapes defensively."""
    val = getattr(out, "text_embeds", None)
    if val is None:
        val = getattr(out, "pooler_output", None)
    return val


def _encoder_device(text_encoder, fallback):
    """Where the encoder's weights will be while it runs.

    Under model_cpu_offload the parameters are parked in CPU RAM, so
    `next(params).device` reports cpu even though accelerate's pre_forward
    hook moves them to the GPU for the duration of the call. The hook knows the
    real execution device, so ask it first.
    """
    hook = getattr(text_encoder, "_hf_hook", None)
    dev = getattr(hook, "execution_device", None)
    if dev is not None:
        return torch.device(dev)
    try:
        return next(text_encoder.parameters()).device
    except StopIteration:
        return torch.device(fallback or "cpu")


def _run_encoder(text_encoder, input_ids):
    """Run the encoder on CPU, bypassing accelerate's offload hook.

    model_cpu_offload replaces `forward` with a wrapper that moves the module
    onto the GPU first. Calling the encoder directly would therefore defeat
    the offload and leave ~2GB resident, which is more than a 6GB card has
    spare once the UNet loads. `_old_forward` is the pre-hook implementation.
    """
    old_forward = getattr(text_encoder, "_old_forward", None)
    if old_forward is not None and input_ids.device.type == "cpu":
        return old_forward(input_ids)
    return text_encoder(input_ids)


def _encode_chunks(chunks, tokenizer, text_encoder, chunk=75, want_pooled=True,
                   device=None):
    """Encode pre-split chunks; returns (sequence [1, N*77, hidden], pooled).

    Runs on CPU by design. Under model_cpu_offload the encoders are already
    parked in CPU RAM, and diffusers only gets them onto the GPU transiently
    inside a single pipeline call. Encoding out here and leaving ~2GB resident
    is what pushes a 6GB card over when the UNet loads, so the CPU path is
    deliberate: it costs a couple of seconds against a multi-minute render.
    """
    dev = torch.device("cpu")
    seqs, pooled = [], None
    for ids in chunks:
        # CLIP pads on the right with EOS (not the pad token) and always keeps
        # the sequence at exactly chunk+2 so the position embeddings line up.
        seq = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
        seq = seq + [tokenizer.eos_token_id] * (chunk + 2 - len(seq))
        out = _run_encoder(text_encoder, torch.tensor([seq], device=dev))
        seqs.append(out.last_hidden_state)
        if pooled is None and want_pooled:
            pooled = _pooled(out)
        del out
    return torch.cat(seqs, dim=1), pooled


def encode_long_prompt(prompt, negative, pipe, device, chunk=75):
    """Build SDXL conditioning for a prompt of any length.

    Returns (prompt_embeds, negative_prompt_embeds, pooled, negative_pooled),
    ready to pass straight to pipe(prompt_embeds=...).

    Every prompt goes through the CPU encoder path, including short ones. An
    earlier version short-circuited single-window prompts to `pipe.encode_prompt`
    for speed, but that runs the encoders through accelerate's offload hook, and
    the hook only returns them to CPU in `maybe_free_model_hooks()` at the end of
    a full `pipe(...)` call. Called standalone it left text_encoder_2 resident
    (1325MB measured), and the UNet's 3744MB no longer fit alongside it on a
    6GB card -- so any prompt under the 77-token window died with a CUDA OOM
    while longer ones rendered fine. The uniform path costs a couple of seconds
    and cannot hit that.
    """
    tok1, tok2 = pipe.tokenizer, pipe.tokenizer_2
    te1, te2 = pipe.text_encoder, pipe.text_encoder_2

    n = num_chunks(tok1, prompt, chunk)
    neg_n = num_chunks(tok1, negative, chunk) if negative else 1

    # Pad the negative up to the positive side's chunk count so CFG can
    # concatenate them; both encoders use the same target count.
    total = max(n, neg_n)
    pos1 = _chunk_ids(tok1, prompt, chunk)
    pos2 = _chunk_ids(tok2, prompt, chunk)
    neg1 = _chunk_ids(tok1, negative, chunk) if negative else [[]]
    neg2 = _chunk_ids(tok2, negative, chunk) if negative else [[]]
    # Tokenizers can disagree on chunk boundaries; the feature-dim concat in
    # SDXL requires identical sequence lengths, so normalise on `total`.
    for group in (pos1, pos2, neg1, neg2):
        while len(group) < total:
            group.append([])

    with torch.no_grad():
        s1, _ = _encode_chunks(pos1, tok1, te1, chunk, want_pooled=False)
        s2, pooled = _encode_chunks(pos2, tok2, te2, chunk, want_pooled=True)
        ns1, _ = _encode_chunks(neg1, tok1, te1, chunk, want_pooled=False)
        # encoder 2's pooled vector drives SDXL's unconditional branch, so the
        # negative needs its own. Reusing the positive one conditioned the
        # negative branch on the prompt it is meant to steer away from.
        ns2, npooled = _encode_chunks(neg2, tok2, te2, chunk, want_pooled=True)

    dtype = pipe.unet.dtype
    pe = torch.cat([s1, s2], dim=-1).to(device, dtype)
    npe = torch.cat([ns1, ns2], dim=-1).to(device, dtype)
    pp = pooled.to(device, dtype)
    npp = npooled.to(device, dtype) if npooled is not None else pp
    return pe, npe, pp, npp
