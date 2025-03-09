# moe-pruner: DeepSeek-v3 Pruning for the GPU Poor

This repository provides a methodology for pruning DeepSeek-v3, a Mixture-of-Experts (MoE) model, to make it more accessible for users with limited GPU resources ("the GPU poor").  This is achieved by significantly reducing the number of experts in the MoE layers.

**Warning:** This is a highly experimental technique.  Results are not guaranteed, and significant improvements are planned for future versions.  Use at your own risk.

## CONTRIBUTE
Due to hardware limitation, i was forced to make a lot of trade off. If you'd like to participate in upcoming versions with a higher compute budget, you can donate here: https://gofund.me/1516dccd

You can also propose improvement on the git repo.

## Project Goal

The primary goal is to drastically reduce the computational and memory requirements of DeepSeek-v3 while retaining a reasonable level of performance.  This is accomplished through a multi-stage distillation and pruning process.

## Methodology and abloation study.
### Core methodology:
- 1. Create a small but high quality calibration dataset
- 2. Use the dataset to prune experts, with several variant (see ablation below)
- 3. Consolidate the model with pruned experts
- 4. Post train on an instruct dataset to recover

### Pruning approaches
The most challenging work was to find an appropriate pruning method, i tested several approaches (some ablation details can be found in the EXPERIMENTATION_README)
- Topk pruning: simply take the k most activated experts on a given layer and prune every other
- Act_cl : cluster experts based on their co activation
- State_cl : cluster experts based on the similarity of their outputs
- Progressive_merge : halve progressively the number of experts, with some healing in between
- Multiplexage : fuse the experts and add a bias term to the hiddenstates that is dependant on the gate activation vector
- Fused lora : fuse experts and add a lora term that takes into account a latent projection of the gate vector
- Mixture of lora : decompose the experts into a core expert and a mixture of lora experts that uses the original gate information to handle activations patterns

To fuse the experts, we relied on three methodology:
- multi slerp : spherical interpolation of mean expert
- sce : a technique developped by https://github.com/fanqiwan/FuseAI
- naive : a simple weight averaging

### Notes:
The most powerfull approach seems to be a mix of the multiplexage and mixture of lora approach.
This technique is the final one used to train the current batch of fused models, and it exhibit interesting properties in term of scaling laws, according to three axis:
- Expert scaling : the end performance of the fused model seems to scale **linearly** in my experiments with the number of final experts, hence with more extensive testing it might be possible to predict in advance the final performance based on number of weights.
- Data scaling : the distillation losses during the pruning where far from being saturated, and seems to follow a similar pattern as usual llm training, following the habitual scaling laws.
- Rank scaling : through the rank of the mixture of Lora, it is possible to adapt efficiently the number of parameters allocated to healing, hence enabling to scale the technique on more compute. This scaling seems to use a regular polynomial scaling law as well.
- Post training on dolphin r1 gave a weird vibe to the models writing, hence is switched to ultrachat for this part, with a bit of the LIMO dataset in addition. I did not perform any DPO or RLHF to the model, so the outputs will be kinda raw.


## Contributions
Due to hardware limitation, this repo required a few hacks to work properly. You'll find ressources for:
- AWQ single layer loading, load a single layer in the gpu memory to avoid OOM
- AWQ dequant and AWQ Dora merge, enabling lower memory usage thanks to 4bit quantization
- Utilities to manage torch vram and load targeted modules inside a model (for some reasons accelerate implementation of the same stuff was not working)
- Dora layer implementation, with adaptations for bnb and hqq (inspired by the Answer.ai repo fsdp qlora)
- Utilities to analyse experts activations and experts similarity, implemented with numba for blazing fast execution
- A legacy fp8_linear layer, which is a pure pytorch implementation of the Deepseek fp8 kernel, compatible with Ampere gpu inference (not sure about training).
- Progressive expert merging, a new method that is using  sce merging to progressively merge the experts based on the similarity of their output, achieving a good knowledge preservation of the experts.
- Expert Multiplexing : an experimental way to merge several expert into a single one, preserve the original gate of the model, and route the inputs in the merged model adequately. So far seems the most stable method, with coherent scaling.
- Fused expert layers, with both a training version with helpers to merge existing layer, and an inference version to just load and run, some patched deepseek modules as well to ease handling.


## Disclaimer
This project is experimental and under active development.  The provided code and methodology are subject to change.  There is no guarantee of performance or stability.  Use at your own discretion.

## Acknowledgements
Special thanks to several projects that inspired and helped the construction of this:
- [CognitiveComputations](https://huggingface.co/cognitivecomputations) for their work on the very high quality dolphin-r1 dataset
- [FuseChat](https://github.com/fanqiwan/FuseAI/tree/main/FuseChat) for their work on knowledge fusion
- [DeepSeek](https://github.com/deepseek-ai) for their openess and awesome technical work
- [AnswerAI](https://github.com/AnswerDotAI/fsdp_qlora) for their work on fsdp qlora, even if i did not use it in the end
- [HuggingFace](https://huggingface.co/) and [PyTorch](https://pytorch.org/) for creating the ecosystem and tools that enabled to iterate very quickly.
- [OpenAi](https://openai.com/) for teaching me the value of open source AI by giving the example of what you shouldn't do.

## Contributing

Contributions are welcome!  If you have suggestions for improvements, bug fixes, or new features, please open an issue or submit a pull request.