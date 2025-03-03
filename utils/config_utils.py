from dataclasses import dataclass
import os


@dataclass
class GenerationParams:
    n_batch: int = 256
    batch_size: int = 8
    max_length: int = 512

@dataclass
class DistillationParams:
    n_epochs: int = 1
    target_routed_expert: int = 4
    target_active_expert: int = 2
    eval_batches:int =16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 8e-4
    end_factor: float = 0.1
    max_workers: int = 16
    calibrate_merge: bool = False
    skip_first_tokens:int = 64
    pruning_method:str ="topk", # topk , act_cl, state_cl
    dora_rank:int =16,

@dataclass
class PathConfig:
    model_name: str
    intermediate_states: str = "data/intermediate_states"
    expert_states: str = "data/expert_states"
    expert_activations: str =  "data/expert_states"
    distillation_logs: str = "distillation_logs"
    moe_states: str="moe_states"