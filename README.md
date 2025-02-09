# moe-pruner: DeepSeek-v3 Pruning for the GPU Poor

This repository provides a methodology for pruning DeepSeek-v3, a Mixture-of-Experts (MoE) model, to make it more accessible for users with limited GPU resources ("the GPU poor").  This is achieved by significantly reducing the number of experts in the MoE layers.

**Warning:** This is a highly experimental technique.  Results are not guaranteed, and significant improvements are planned for future versions.  Use at your own risk.

## CONTRIBUTE
Due to hardware limitation, i was forced to make a lot of trade off. If you'd like to participate in upcoming versions with a higher compute budget, you can donate here: https://gofund.me/1516dccd

You can also propose improvement on the git repo.

## Project Goal

The primary goal is to drastically reduce the computational and memory requirements of DeepSeek-v3 while retaining a reasonable level of performance.  This is accomplished through a multi-stage distillation and pruning process.

## Methodology

The pruning pipeline consists of the following steps:

1.  **Calibration Data Download (`0. CalibrationDownload.ipynb`):** Downloads the necessary calibration dataset used for the distillation process.  This dataset is used to guide the smaller, pruned model to mimic the behavior of the larger, original model.

2.  **Model Download and Patching (`0. ModelDownload.ipynb`):** Downloads the DeepSeek-v3 model and applies patches to the model's code. These patches are crucial for enabling efficient training and GPU utilization during the subsequent pruning steps.  This likely modifies the model's forward pass to allow for layer-by-layer processing.

3.  **Layer-wise Distillation (`1. LayerDistillation.ipynb`):** This is the core of the pruning process.  The model is loaded one layer at a time.  For each MoE layer, a smaller, "pruned" version is created with significantly fewer experts.  Knowledge distillation is then used to train the pruned layer, using the original layer as a teacher.  The pruned layer learns to mimic the output of the original layer, effectively compressing the knowledge contained within the larger set of experts.  *Note:* The current implementation uses full distillation.  LoRA (Low-Rank Adaptation) was considered but may not be effective; if testing shows LoRA is insufficient, the code will revert to full fine-tuning.

4.  **Expert Aggregation (`2. UnHealedAggregation.ipynb`):**  The individually distilled and pruned expert layers are consolidated into a single, self-contained PyTorch module. This step prepares the model for the subsequent healing and quantization stages.  The term "UnHealed" suggests that the gating mechanism (which selects which expert to use) is not yet optimized in this aggregated model.

5.  **Post-training (Healing) (`3. Posttraining.ipynb`):**  Fine-tunes *only* the distilled experts and the gating mechanism.  This "healing" step is crucial to recover performance lost during the aggressive pruning.  The gating mechanism learns to effectively route inputs to the reduced set of experts, and the experts themselves are further refined to improve overall model accuracy.

6.  **AWQ Quantization (`4. AWQQuantisation.ipynb`):** Converts the model to the AWQ (Activation-aware Weight Quantization) format.  AWQ is a quantization technique that reduces the model's memory footprint and computational cost, making it suitable for inference with tools like vLLM.  This step is essential for running the pruned model on less powerful hardware.

## Target Sizes of This Method:

This method targets the following model sizes, scaling down from a base model:

| Base Model Size      | Scaled Model Size      | Notes      |
| :------------------- | :--------------------- | :---------- |
| 256@8 (Base)          | DeepSeek-V3-671B@37B    | Full       |
| 22@6                 | DeepSeek-V3-Lite-72B@31B | Large      |
| 16@4                 | DeepSeek-V3-Lite-57B@26B | Medium     |
| 8@2                  | DeepSeek-V3-Lite-36B@21B | Small      |
| 4@1                  | DeepSeek-V3-Lite-26B@19B | Nano       |

## Model Links
### Unhealed
v0.1 4@1: https://huggingface.co/AlphaGaO/DeepSeek-V3-4a1-unhealed-v0.1

## Iterations:
v0.1:
- Distillation : Full expert distillation, 4096 samples with seq_length 515.
- Post training : Lora tuning (rank / alpha = 16), 65536 samples, seq_length 64 (hardware constraint).
- Attention layers and shared experts are untouched.

v0.2: (incoming)
- Distillation : Full gate distillation, lora expert distillation, 16384 samples with seq_length 515.

## Improvements and Future Work (v0.2+)

The initial experiments have revealed several areas for improvement:

*   **Reconstruction Loss Gradient:**  A key observation is that the reconstruction loss (the difference between the original and pruned layer outputs) increases significantly with layer depth.  Later layers (e.g., layer 40) have much higher loss than earlier layers (e.g., layer 10).  This suggests that MoEs are particularly important for deeper layers, and a more effective pruning strategy might vary the number of experts based on layer depth.

*   **Calibration Dataset Size:** The current pipeline uses a relatively small calibration dataset (4096 samples) due to resource constraints.  Increasing the size of this dataset is expected to improve distillation quality.

Planned experiments for v0.2 and beyond include:

*   **Scaling Calibration Dataset:**  Significantly increase the size of the calibration dataset to improve the quality of the distilled, pruned model.

*   **Adaptive Expert Count:**  Implement a variable number of experts per layer.  Earlier layers might have fewer experts, while deeper layers (where MoEs seem more critical) retain more.

*   **Iterative Expert Pruning:**  Instead of removing all experts at once, prune them gradually in an iterative process. This might allow for better adaptation and less performance loss.

*   **Expert Fusion:** Explore fusing experts that exhibit high activation correlation or output similarity using techniques like SLERP (Spherical Linear Interpolation).  This could further reduce the number of experts without significantly impacting performance.

*   **Shared Expert Training:**  Investigate sharing weights among multiple distilled MoE layers.  This could lead to more efficient distillation and potentially optimize the order of experts. This idea aims at training expert that could be reusable in several layers.

## Hardware Requirements

The pipeline was developed and optimized for the following configuration:

*   **Storage:** 2TB SSD with at least 512MB/s read/write speed.
*   **RAM:** 128GB DDR4 3600MHz
*   **CPU:** AMD Ryzen 9 3950X (16 cores, 32 threads)
*   **GPU:** 2 x NVIDIA RTX 3090 (aggregated 48GB GDDR6X VRAM)

You may be able to run parts of the pipeline with less powerful hardware, particularly if you adjust parameters (e.g., reduce batch sizes, use a smaller calibration dataset).  However, the full pipeline, especially the distillation stage, is computationally demanding.

## Software Requirements

*   CUDA: 12.1
*   PyTorch: 2.6 (or later compatible versions)
*   transformers library
*   peft library (Parameter-Efficient Fine-Tuning)

**Installation (Example using Conda):**

```bash
conda create -n moe-pruner python=3.10
conda activate moe-pruner
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers peft
```

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd moe-pruner
    ```

2.  **Run the notebooks in order:**
    *   `0. CalibrationDownload.ipynb`
    *   `0. ModelDownload.ipynb`
    *   `1. LayerDistillation.ipynb`
    *   `2. UnHealedAggregation.ipynb`
    *   `3. Posttraining.ipynb`
    *   `4. AWQQuantisation.ipynb`

    Make sure to adjust paths and parameters within the notebooks as needed for your environment. Each notebook should be run sequentially.

## Disclaimer

This project is experimental and under active development.  The provided code and methodology are subject to change.  There is no guarantee of performance or stability.  Use at your own discretion.

## Contributing

Contributions are welcome!  If you have suggestions for improvements, bug fixes, or new features, please open an issue or submit a pull request.