"""Negative control: break copy-on-write (allocate the new block but do NOT copy the K/V data).
model_check's COW test must FAIL, proving the harness can detect a real bug."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build-py"))
sys.path.insert(0, os.path.dirname(__file__))
import torch
import model_check as mc
from paged_model import PagedEngine
from transformers import AutoModelForCausalLM, AutoTokenizer

class NoCopyEngine(PagedEngine):
    def decode(self, rid, last_token):
        n = self.kv.num_tokens(rid)
        if not self.kv.append_token(rid):
            return None
        return self._forward(rid, [last_token], n)   # forgot to copy the old block's data

mc.PagedEngine = NoCopyEngine
name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
base = tok("Paged attention splits the key value cache of a language model into fixed size blocks. " * 6)["input_ids"]
mc.test_cow(model, base)
failed = [r for r in mc.rows if not r[1]]
print("NEGATIVE CONTROL:", "harness detected the bug (good)" if failed else "NOT detected (bad)")
sys.exit(0 if failed else 1)
