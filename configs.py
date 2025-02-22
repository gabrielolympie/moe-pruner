from dataclasses import dataclass
import os

@dataclass
class GenerationParams:
    n_batch: int = 4
    batch_size: int = 8
    max_length: int = 512

@dataclass
class DistillationParams:
    n_epochs: int = 1
    n_batch: int = 128
    n_train_batch : int = 116
    batch_size: int = 16
    max_length: int = 512
    gradient_accumulation_steps: int = 1
    calibration_batches: int = 16
    learning_rate: float = 8e-4
    end_factor: float = 0.1
    temperature: float = 1.0
    lora_type: str = "dora"
    lora_rank: int = 16
    lora_alpha: int = 16
    max_workers: int = 16
    fp8_format: str = "e4m3"
    distiller_device: str = "cuda:1"

@dataclass
class PathConfig:
    model_name: str = "deepseek_v3"
    base_dir: str = "data/distillation_runs"
    checkpoint_dir: str = "layers"
    intermediate_dir: str = "data/intermediate_states"
    midlayer_dir : str = "data/midlayer_states"
    exp_states: str = "data/exp_states"
    log_dir: str = "distillation_logs"
    expert_activation_dir: str = "data/expert_activation"

    def __post_init__(self):
        for dir_name in [
            self.base_dir,
            self.log_dir,
            self.intermediate_dir,
            self.exp_states,
            self.expert_activation_dir,
        ]:
            os.makedirs(dir_name, exist_ok=True)

    def get_layer_path(self, layer_idx: int) -> str:
        return os.path.join(self.checkpoint_dir, f"layer_{layer_idx}.ckpt")

    def get_intermediate_path(self, layer_idx: int, batch_idx: int) -> str:
        os.makedirs(
            os.path.join(self.intermediate_dir, f"layer_{layer_idx}"), exist_ok=True
        )
        return os.path.join(
            self.intermediate_dir, f"layer_{layer_idx}", f"batch{batch_idx}.pt"
        )
        
    def get_midlayer_path(self, layer_idx: int, batch_idx: int) -> str:
        os.makedirs(
            os.path.join(self.midlayer_dir, f"layer_{layer_idx}"), exist_ok=True
        )
        return os.path.join(
            self.midlayer_dir, f"layer_{layer_idx}", f"batch{batch_idx}.pt"
        )

    def get_exp_path(self, layer_idx: int, batch_idx: int) -> str:
        os.makedirs(os.path.join(self.exp_states, f"layer_{layer_idx}"), exist_ok=True)
        return os.path.join(self.exp_states, f"layer_{layer_idx}", f"batch{batch_idx}.pt")
    
    def get_expert_activation_path(self, layer_idx: int) -> str:
        os.makedirs(self.expert_activation_dir, exist_ok=True)
        return os.path.join(self.expert_activation_dir, f"layer_{layer_idx}.pickle")

    def get_distillation_path(self, n_experts: int, n_active: int) -> str:
        return os.path.join(
            self.base_dir, f"{self.model_name}_{n_experts}@{n_active}"
        )

