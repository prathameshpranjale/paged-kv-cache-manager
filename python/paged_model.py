"""Run a Qwen2- or Llama-architecture model with its K/V stored in fixed-size blocks, addressed through
block tables owned by the C++ kvpage::KVManager (via the kvpage_py binding).

Only the attention K/V storage differs from stock HF code: every layer's K/V lives in
K[layer][physical_block, offset, kv_head, head_dim], and attention gathers the context through
the request's block table. Everything else (norms, projections, MLP, RoPE) reuses HF modules.
"""
import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb  # identical for Qwen2 and Llama

import kvpage_py


class PagedEngine:
    def __init__(self, model, num_blocks, block_size):
        self.m = model.eval()
        c = model.config
        self.L = c.num_hidden_layers
        self.nh = c.num_attention_heads
        self.kvh = c.num_key_value_heads
        self.hd = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        self.B = block_size
        self.kv = kvpage_py.KVManager(num_blocks, block_size)
        self.dtype = next(model.parameters()).dtype
        self.K = torch.zeros(self.L, num_blocks, block_size, self.kvh, self.hd, dtype=self.dtype)
        self.V = torch.zeros_like(self.K)
        self.prefill_tokens_computed = 0
        self.cow_data_copies = 0

    # ---- model forward over `tokens` placed at positions start..start+n-1 of request `rid` ----
    @torch.no_grad()
    def _forward(self, rid, tokens, start):
        B, n = self.B, len(tokens)
        table = self.kv.block_table(rid)
        total = start + n
        assert self.kv.num_tokens(rid) >= total, "manager must already account for these tokens"
        pos = torch.arange(start, total)
        wblk = torch.tensor([table[p // B] for p in range(start, total)])
        woff = pos % B
        ctx_blocks = torch.tensor(table[: (total + B - 1) // B])
        mask = torch.arange(total)[None, :] > pos[:, None]  # [n, total]: True = not allowed

        mdl = self.m.model
        h = mdl.embed_tokens(torch.tensor(tokens))[None]  # [1, n, H]
        cos, sin = mdl.rotary_emb(h, pos[None])
        for li, layer in enumerate(mdl.layers):
            a = layer.self_attn
            x = layer.input_layernorm(h)
            q = a.q_proj(x).view(1, n, self.nh, self.hd).transpose(1, 2)
            k = a.k_proj(x).view(1, n, self.kvh, self.hd).transpose(1, 2)
            v = a.v_proj(x).view(1, n, self.kvh, self.hd).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            # write the new K/V into their blocks (never into a block shared with another request)
            self.K[li, wblk, woff] = k[0].transpose(0, 1)
            self.V[li, wblk, woff] = v[0].transpose(0, 1)
            # gather the whole context through the block table
            kc = self.K[li][ctx_blocks].reshape(-1, self.kvh, self.hd)[:total].permute(1, 0, 2)
            vc = self.V[li][ctx_blocks].reshape(-1, self.kvh, self.hd)[:total].permute(1, 0, 2)
            rep = self.nh // self.kvh
            kc, vc = kc.repeat_interleave(rep, 0), vc.repeat_interleave(rep, 0)
            s = torch.matmul(q[0], kc.transpose(-1, -2)) * (self.hd ** -0.5)  # [nh, n, total]
            s = s.masked_fill(mask[None], float("-inf")).softmax(-1)
            o = torch.matmul(s, vc).transpose(0, 1).reshape(1, n, self.nh * self.hd)
            h = h + a.o_proj(o)
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        return self.m.lm_head(mdl.norm(h[:, -1]))[0]  # logits of the last position

    # ---- request lifecycle ----
    def prefill(self, rid, seq, parent=None, shared_tokens=0):
        """Admit `rid` with context `seq`; returns logits for the next token, or None if no memory."""
        if parent is None:
            ok = self.kv.admit(rid, len(seq))
            start = 0
        else:
            start = min(shared_tokens, self.kv.num_tokens(parent), len(seq)) // self.B * self.B
            ok = self.kv.admit_with_prefix(rid, parent, shared_tokens, len(seq))
        if not ok:
            return None
        self.prefill_tokens_computed += len(seq) - start  # shared full blocks are not recomputed
        return self._forward(rid, seq[start:], start)

    def fork(self, rid, parent):
        self.kv.fork(rid, parent)

    def decode(self, rid, last_token):
        """Append `last_token` to the cache; returns next-token logits, or None if out of blocks."""
        before_tokens = self.kv.num_tokens(rid)
        old_last = self.kv.block_table(rid)[-1] if before_tokens else None
        if not self.kv.append_token(rid):
            return None
        new_last = self.kv.block_table(rid)[-1]
        if before_tokens % self.B != 0 and new_last != old_last:  # copy-on-write: copy the data too
            self.K[:, new_last] = self.K[:, old_last]
            self.V[:, new_last] = self.V[:, old_last]
            self.cow_data_copies += 1
        return self._forward(rid, [last_token], before_tokens)

    def free(self, rid):
        self.kv.free(rid)

    def kv_bytes_per_token(self):
        return 2 * self.L * self.kvh * self.hd * self.K.element_size()


def _decode_batch(self, rids, last_tokens):
    """One decode step for several requests in a single batched forward pass (lengths may differ)."""
    B = self.B
    before = [self.kv.num_tokens(r) for r in rids]
    for r, n in zip(rids, before):
        old_last = self.kv.block_table(r)[-1] if n else None
        if not self.kv.append_token(r):
            raise RuntimeError("out of blocks in decode_batch")
        new_last = self.kv.block_table(r)[-1]
        if n % B != 0 and new_last != old_last:  # copy-on-write: copy the data before anyone writes
            self.K[:, new_last] = self.K[:, old_last]
            self.V[:, new_last] = self.V[:, old_last]
            self.cow_data_copies += 1
    return self._forward_decode_batch(rids, last_tokens, before)


@torch.no_grad()
def _forward_decode_batch(self, rids, tokens, positions):
    B, b = self.B, len(rids)
    tables = [self.kv.block_table(r) for r in rids]
    pos = torch.tensor(positions)
    max_blocks = max((p + 1 + B - 1) // B for p in positions)
    ctx = torch.zeros(b, max_blocks, dtype=torch.long)  # padded with block 0; padding is masked out
    for i, (t, p) in enumerate(zip(tables, positions)):
        nb = (p + 1 + B - 1) // B
        ctx[i, :nb] = torch.tensor(t[:nb])
    wblk = torch.tensor([t[p // B] for t, p in zip(tables, positions)])
    woff = pos % B
    mask = torch.arange(max_blocks * B)[None, :] > pos[:, None]  # [b, T]: True = not allowed
    mdl = self.m.model
    h = mdl.embed_tokens(torch.tensor(tokens))[:, None]  # [b, 1, H]
    cos, sin = mdl.rotary_emb(h, pos[:, None])
    rep = self.nh // self.kvh
    for li, layer in enumerate(mdl.layers):
        a = layer.self_attn
        x = layer.input_layernorm(h)
        q = a.q_proj(x).view(b, 1, self.nh, self.hd).transpose(1, 2)
        k = a.k_proj(x).view(b, 1, self.kvh, self.hd).transpose(1, 2)
        v = a.v_proj(x).view(b, 1, self.kvh, self.hd).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        self.K[li, wblk, woff] = k[:, :, 0]
        self.V[li, wblk, woff] = v[:, :, 0]
        kc = self.K[li][ctx].reshape(b, max_blocks * B, self.kvh, self.hd).permute(0, 2, 1, 3).repeat_interleave(rep, 1)
        vc = self.V[li][ctx].reshape(b, max_blocks * B, self.kvh, self.hd).permute(0, 2, 1, 3).repeat_interleave(rep, 1)
        s = torch.matmul(q, kc.transpose(-1, -2)) * (self.hd ** -0.5)  # [b, nh, 1, T]
        s = s.masked_fill(mask[:, None, None, :], float("-inf")).softmax(-1)
        o = torch.matmul(s, vc).transpose(1, 2).reshape(b, 1, self.nh * self.hd)
        h = h + a.o_proj(o)
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    return self.m.lm_head(mdl.norm(h[:, 0]))  # [b, vocab]


PagedEngine.decode_batch = _decode_batch
PagedEngine._forward_decode_batch = _forward_decode_batch
