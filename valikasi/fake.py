"""Miniature random Qwen2 with a fake char-level tokenizer, for code-path tests only.
Scores from it mean nothing."""
import torch

from .pipeline import Model


class FakeTok:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        s = "".join(f"<{m['role']}>{m['content']}\n" for m in messages)
        return s + ("<assistant>" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 256 for c in text]

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        ids = torch.tensor([self.encode(text)])
        return _Enc(input_ids=ids, attention_mask=torch.ones_like(ids))

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(32 + int(i) % 95) for i in ids)


class _Enc(dict):
    def __init__(self, **kw):
        super().__init__(**kw)

    def to(self, device):
        return _Enc(**{k: v.to(device) for k, v in self.items()})


def tiny_model(seed=0):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=256, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=4096)
    return Model(FakeTok(), Qwen2ForCausalLM(cfg), "cpu")
