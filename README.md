# moe-pruner: Democratizing DeepSeek-v3 with Expert Fusion

This repository introduces a novel, experimental methodology for fusing DeepSeek-v3, a powerful Mixture-of-Experts (MoE) model.  Our goal is to dramatically reduce its computational and memory footprint, making it accessible to the GPU poors.  We achieve this by aggressively reducing the number of experts within the MoE layers, combined with knowledge distillation techniques.

**Important:** This project is highly experimental and under active development. Results are not guaranteed, and significant improvements are planned.  Use at your own risk and be aware that the code and methodology are subject to change.

## Project Goal

Our primary objective is to make DeepSeek-v3 usable on consumer-grade hardware.  We aim to drastically lower the computational and memory demands while preserving a reasonable level of performance. This is achieved through a carefully designed, multi-stage process involving expert pruning, model consolidation, and knowledge distillation.

## Models
Naming convention is Num Fused experts / Num weights. I did not add the number of active experts because this can change during inference depending on the activated origin experts.
Uploading both final and unhealed models, for finetuning, i recommend using the unhealed models.

### DeepSeek-Coder-v2-lite
2 Fused Experts / 2B Params : https://huggingface.co/AlphaGaO/Deepseek-Coder-V2-Lite-Instruct-Fused-2E-2B-preview
4 Fused Experts / 2.5B Params : https://huggingface.co/AlphaGaO/Deepseek-Coder-V2-Lite-Instruct-Fused-4E-2_5B-preview
8 Fused Experts / 3B Params : https://huggingface.co/AlphaGaO/Deepseek-Coder-V2-Lite-Instruct-Fused-8E-3B-preview
16 Fused Experts / 5B Params : https://huggingface.co/AlphaGaO/Deepseek-Coder-V2-Lite-Instruct-Fused-16E-5B-preview

### DeepSeek-Coder-v2-lite unhealed
2 Fused Experts / 2B Params : https://huggingface.co/AlphaGaO/Deepseek-Coder-V2-Lite-Instruct-Fused-2E-2B-preview-Unhealed
4 Fused Experts / 2.5B Params : https://huggingface.co/AlphaGaO/Deepseek-Coder-V2-Lite-Instruct-Fused-4E-2_5B-preview-Unhealed
8 Fused Experts / 3B Params : https://huggingface.co/AlphaGaO/Deepseek-Coder-V2-Lite-Instruct-Fused-8E-3B-preview-Unhealed
16 Fused Experts / 5B Params : https://huggingface.co/AlphaGaO/Deepseek-Coder-V2-Lite-Instruct-Fused-16E-5B-preview-Unhealed

## Methodology and Ablation Study

### Core Methodology:

1.  **Calibration Dataset Creation:**  A small, high-quality dataset is constructed for calibration. This dataset is crucial for guiding the pruning process.
2.  **Expert Fusing:**  Various fusion strategies are employed to identify and remove less critical experts (see "Fusion Approaches" below for details).
3.  **Model Consolidation:** The pruned model, with a significantly reduced number of experts, is consolidated.
4.  **Post-Training (Knowledge Recovery):**  The consolidated model is further trained on an instruction dataset (UltraChat, with some LIMO) to recover performance lost during pruning.  No DPO or RLHF is performed, resulting in a "raw" output style.

### Fusion Approaches

A key challenge was identifying an effective pruning method.  We explored (and continue to explore) several approaches, including:

*   **Top-k Pruning:**  The simplest approach – selecting the *k* most frequently activated experts per layer and discarding the rest.
*   **Activation Clustering (Act\_cl):** Clustering experts based on their co-activation patterns.  Experts that fire together are considered redundant.
*   **State Clustering (State\_cl):** Clustering experts based on the similarity of their output states.  Experts producing similar outputs are considered redundant.
*   **Progressive Merge:**  Iteratively halving the number of experts, with "healing" (fine-tuning) steps in between to mitigate performance loss.
*   **Multiplexage:**  Fusing experts and introducing a bias term to the hidden states, dependent on the original gate activation vector. This allows the model to retain some of the routing information from the original MoE.
*   **Fused LoRA:**  Combining expert weights and adding a LoRA (Low-Rank Adaptation) term that incorporates a latent projection of the gate vector.
*   **Mixture of LoRA:**  Decomposing experts into a core expert and a mixture of LoRA experts. The original gate information is used to manage the activation patterns of the LoRA experts.

**Expert Fusion Techniques:**

To combine the weights of experts during merging, we utilize:

*   **Multi-SLERP:** Spherical Linear Interpolation (SLERP) of the mean expert weights.
*   **SCE (Similarity-based Coefficient Estimation):**  A technique developed by [FuseAI](https://github.com/fanqiwan/FuseAI).
*   **Naive Averaging:** Simple averaging of expert weights.

### Notes and Observations:

The most promising approach currently is a hybrid of "Multiplexage" and "Mixture of LoRA." This method exhibits interesting scaling properties along three key axes:

*   **Expert Scaling:**  Preliminary experiments suggest a *linear* relationship between the number of final experts and the fused model's performance.  This could allow for performance prediction based on model size.
*   **Data Scaling:**  Distillation losses during pruning were far from saturation, suggesting that the process follows standard LLM training scaling laws. Further data could lead to significant improvements.
*   **Rank Scaling:**  The rank of the "Mixture of LoRA" allows for efficient control over the number of parameters dedicated to knowledge recovery. This scaling appears to follow a polynomial law.
* **Dataset choice**: Initially, post-training was done with the dolphin-r1 dataset, but this resulted in an undesirable writing style. Therefore, the final models were trained with ultrachat dataset, with additional data from LIMO.

## Contributions & Included Resources

Due to hardware constraints, several custom implementations and hacks were necessary. This repository includes resources for:

*   **AWQ Single-Layer Loading:** Load individual layers into GPU memory to prevent Out-of-Memory (OOM) errors.
*   **AWQ Dequantization and DoRA Merge:**  Utilities for lower memory usage via 4-bit quantization.
*   **VRAM Management Utilities:**  Tools for managing PyTorch VRAM and loading specific model modules (addressing issues with standard `accelerate` implementations).
*   **DoRA Layer Implementation:**  Adapted for BNB and HQQ (inspired by [Answer.ai's fsdp qlora](https://github.com/AnswerDotAI/fsdp_qlora)).
*   **Expert Analysis Utilities:**  Tools for analyzing expert activations and similarity, implemented with Numba for performance.
*   **Legacy FP8 Linear Layer:**  A pure PyTorch implementation of DeepSeek's FP8 kernel, compatible with Ampere GPU inference (training compatibility is uncertain).
*   **Progressive Expert Merging:**  A novel method that leverages SCE merging to iteratively combine experts based on output similarity, preserving knowledge effectively.
*   **Expert Multiplexing:**  An experimental technique for merging experts while retaining original gate information for appropriate input routing. This appears to be the most stable method, with consistent scaling behavior.
*   **Fused Expert Layers:**  Includes both a training version (with helpers for merging existing layers) and an inference version (for loading and running the fused models).  Also includes patched DeepSeek modules for easier handling.

## Disclaimer

This project is highly experimental and actively under development. The provided code and methodology are subject to change.  Performance and stability are not guaranteed.  Use at your own discretion.

## Acknowledgements

We gratefully acknowledge the contributions of the following projects:

*   **CognitiveComputations:** Creators of the high-quality dolphin-r1 dataset.
*   **FuseAI:** Developers of the FuseChat project and the SCE merging technique.
*   **DeepSeek:** For their open-source contributions and excellent technical work.
*   **AnswerAI:** For their work on fsdp qlora.
*   **Hugging Face and PyTorch:** For providing the essential ecosystem and tools that facilitate rapid iteration.
*   **OpenAI:**  For inadvertently highlighting the importance of open-source AI through their contrasting approach.

## Contributing

Contributions are highly encouraged!  If you have suggestions, bug fixes, or new features, please open an issue or submit a pull request.

Areas where contributions are particularly welcome:

*   **Quantization:**  The current implementation of fused experts presents challenges for integration with existing quantization engines. Expertise in this area is greatly appreciated.
*   **Inference:**  Adaptations for inference frameworks like vLLM, Aphrodite, exllama v2, and llama.cpp would significantly improve the usability of this project.
*   **Further Ablation Studies:**  More extensive testing and refinement of the pruning and fusion techniques.
*  **Theoretical grounding** : Some part of the method are still mostly empirical, formalizing them would help stability.

## Call to Action / Support

This project was developed under significant hardware limitations, requiring numerous trade-offs.  If you would like to support the development of future versions with increased computational resources, you can donate here: [https://gofund.me/1516dccd](https://gofund.me/1516dccd).  You are also welcome to propose improvements directly on the GitHub repository.