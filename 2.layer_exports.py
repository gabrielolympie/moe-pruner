import gc
import _pickle as pickle

from memory_utils import load_module_weights_and_freeze_optimized, load_weight_cached, load_model_config, create_empty_model, create_empty_layer
from torch_utils import destruct_module_optimized, create_empty_layer_fp8, memory_cleanup

from tqdm.auto import tqdm
import torch
import os

weight_map, config = load_model_config("deepseek_v3")
weight_file = weight_map['model.embed_tokens.weight']

model_name = "DeepSeek-V3"

def memory_cleanup():
    """Perform thorough memory cleanup"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


if __name__ == "__main__":  # Corrected the main guard
    os.makedirs("layers", exist_ok=True)
    model = create_empty_model(config)

    ## Embed
    model.model.embed_tokens = load_module_weights_and_freeze_optimized(
        model.model.embed_tokens,
        f"model.embed_tokens",
        weight_map,
        "deepseek_v3",
        max_workers=32,
        fp8_format="e4m3",
    )

    torch.save(model.model.embed_tokens.state_dict(), 'layers/embed_tokens.pt')

    ## End norm
    model.model.norm = load_module_weights_and_freeze_optimized(
        model.model.norm,
        f"model.norm",
        weight_map,
        "deepseek_v3",
        max_workers=32,
        fp8_format="e4m3",
    )

    torch.save(model.model.norm.state_dict(), 'layers/norm.pt')


    ## Lm head
    model.lm_head = load_module_weights_and_freeze_optimized(
        model.lm_head,
        f"lm_head",
        weight_map,
        "deepseek_v3",
        max_workers=32,
        fp8_format="e4m3",
    )

    torch.save(model.lm_head.state_dict(), 'layers/lm_head.pt')


    destruct_module_optimized(model)
    memory_cleanup()
    ## Layers
    for i in tqdm(range(62)):

        layer = create_empty_layer(config, layer_idx=i)
        layer = load_module_weights_and_freeze_optimized(
            layer,
            f"model.layers.{i}",
            weight_map,
            "deepseek_v3",
            max_workers=16,
            fp8_format="e4m3",
        )
        memory_cleanup()

        torch.save(layer.state_dict(), f'./layers/layer_{i}.pt')
        destruct_module_optimized(layer)
        del layer # Explicitly delete the layer after destruction
        memory_cleanup() # cleanup after deleting layer