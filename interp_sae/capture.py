import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def load_model(name="distilgpt2"):
    model = GPT2LMHeadModel.from_pretrained(name)
    tokenizer = GPT2TokenizerFast.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def _blocks(model):
    return model.transformer.h


def capture_activations(model, inputs, points=("resid", "mlp")):
    blocks = _blocks(model)
    store = {point: {} for point in points}
    handles = []

    def make_resid_hook(idx):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            store["resid"][idx] = hidden.detach()
        return hook

    def make_mlp_hook(idx):
        def hook(module, args, output):
            store["mlp"][idx] = output.detach()
        return hook

    for idx, block in enumerate(blocks):
        if "resid" in points:
            handles.append(block.register_forward_hook(make_resid_hook(idx)))
        if "mlp" in points:
            handles.append(block.mlp.register_forward_hook(make_mlp_hook(idx)))

    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for handle in handles:
            handle.remove()

    result = {}
    for point in points:
        result[point] = torch.stack([store[point][i] for i in range(len(blocks))])
    return result


def capture_from_text(model, tokenizer, prompts, points=("resid", "mlp")):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    return capture_activations(model, inputs, points)
