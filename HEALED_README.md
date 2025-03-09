---
library_name: transformers
---
<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

## DeepSeek-Coder-V2-lite-instruct-Fused-preview:  Compressed and Efficient Code Generation

This README introduces a series of fused models derived from DeepSeek-Coder-V2-lite-instruct.  These models represent a significant reduction in size while retaining strong performance, demonstrating a novel approach to model compression.

**Motivation (Why Fuse an "Old" Model?)**

This project serves as a proof-of-concept for a new model pruning and fusion technique.  The core methodology is detailed in the accompanying GitHub repository: [https://github.com/gabrielolympie/moe-pruner](https://github.com/gabrielolympie/moe-pruner).  The long-term goal is to provide a method for pruning large Mixture-of-Experts (MoE) models, creating smaller, locally runnable models that maintain a significant portion of the original model's capabilities.

**Preview Models:  Exploring Compression Levels**

The DeepSeek-Coder-V2-lite-instruct model, which utilizes 64 experts, forms the foundation for these preview models.  We offer four variations, each with a different level of compression:

*   **16 Fused Experts (~6B parameters):**  1/4 size reduction.
*   **8 Fused Experts (~4B parameters):**  1/8 size reduction.
*   **4 Fused Experts (~3B parameters):**  1/16 size reduction.
*   **2 Fused Experts (~2B parameters):**  1/32 size reduction.

Despite their significantly reduced size, these models demonstrate surprisingly strong performance, exceeding expectations for their parameter counts.  Further, more comprehensive testing is planned.

**Technical Details and Scaling Properties**

The fusion technique, refined through multiple iterations, combines *expert multiplexing* with a *mixture of LoRA decomposition*.  Preliminary results indicate promising scaling properties along three key dimensions:

1.  **Expert Scaling:**  The performance of the fused model appears to scale *linearly* with the number of fused experts. This suggests the potential for predicting final performance based on the number of retained weights.

2.  **Data Scaling:**  Distillation losses during the pruning process were far from saturation.  The observed trends align with typical LLM training scaling laws, indicating further improvements are possible with more training data.

3.  **Rank Scaling:**  The rank of the LoRA mixture allows for efficient adjustment of the number of parameters used during the "healing" process. This enables scaling the technique to larger compute budgets, exhibiting polynomial scaling behavior similar to standard LoRA approaches.

These models are designated as "previews" because the distillation losses were not saturated during training.  Future iterations, potentially utilizing additional calibration data, may further enhance performance.

## Call to Action: Contribute and Support

Due to hardware constraints, several trade-offs were necessary.  If you are interested in supporting future development with increased compute resources, donations are welcome: [https://gofund.me/1516dccd](https://gofund.me/1516dccd)

**We actively encourage community contributions!**  We particularly welcome expertise in the following areas:

*   **Quantization:** The current implementation of fused experts presents challenges for existing quantization engines.  Assistance in developing compatible quantization strategies is highly desired.
*   **Inference:**  Integration with popular inference frameworks like vLLM, AphroditeEngine, ExLlamaV2, and llama.cpp would significantly improve usability.

If you have suggestions for improvements, bug fixes, or new features, please open an issue or submit a pull request on the GitHub repository. Your contributions are valuable!

#
#
#
#


## Original Model Card

---
license: other
license_name: deepseek-license
license_link: LICENSE
---
<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/logo.svg?raw=true" width="60%" alt="DeepSeek-V2" />
</div>
<hr>
<div align="center" style="line-height: 1;">
  <a href="https://www.deepseek.com/" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/badge.svg?raw=true" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://chat.deepseek.com/" target="_blank" style="margin: 2px;">
    <img alt="Chat" src="https://img.shields.io/badge/🤖%20Chat-DeepSeek%20V2-536af5?color=536af5&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/deepseek-ai" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DeepSeek%20AI-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://discord.gg/Tc7c45Zzu5" target="_blank" style="margin: 2px;">
    <img alt="Discord" src="https://img.shields.io/badge/Discord-DeepSeek%20AI-7289da?logo=discord&logoColor=white&color=7289da" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/qr.jpeg?raw=true" target="_blank" style="margin: 2px;">
    <img alt="Wechat" src="https://img.shields.io/badge/WeChat-DeepSeek%20AI-brightgreen?logo=wechat&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://twitter.com/deepseek_ai" target="_blank" style="margin: 2px;">
    <img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-deepseek_ai-white?logo=x&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/LICENSE-CODE" style="margin: 2px;">
    <img alt="Code License" src="https://img.shields.io/badge/Code_License-MIT-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/LICENSE-MODEL" style="margin: 2px;">
    <img alt="Model License" src="https://img.shields.io/badge/Model_License-Model_Agreement-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>
<p align="center">
  <a href="#4-api-platform">API Platform</a> |
  <a href="#5-how-to-run-locally">How to Use</a> |
  <a href="#6-license">License</a> |
</p>


<p align="center">
  <a href="https://github.com/deepseek-ai/DeepSeek-Coder-V2/blob/main/paper.pdf"><b>Paper Link</b>👁️</a>
</p>

AWQ quantized version of DeepSeek-Coder-V2-Lite-Instruct model.

---

# DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence

## 1. Introduction
We present DeepSeek-Coder-V2,  an open-source Mixture-of-Experts (MoE) code language model that achieves performance comparable to GPT4-Turbo in code-specific tasks. Specifically, DeepSeek-Coder-V2 is further pre-trained from DeepSeek-Coder-V2-Base  with 6 trillion tokens sourced from a high-quality and multi-source corpus. Through this continued pre-training, DeepSeek-Coder-V2 substantially enhances the coding and mathematical reasoning capabilities of DeepSeek-Coder-V2-Base, while maintaining comparable performance in general language tasks. Compared to  DeepSeek-Coder, DeepSeek-Coder-V2 demonstrates significant advancements in various aspects of code-related tasks, as well as reasoning and general capabilities.  Additionally, DeepSeek-Coder-V2 expands its support for programming languages from 86 to 338, while extending the context length from 16K to 128K.

<p align="center">
  <img width="100%" src="https://github.com/deepseek-ai/DeepSeek-Coder-V2/blob/main/figures/performance.png?raw=true">
</p>

In standard benchmark evaluations, DeepSeek-Coder-V2 achieves superior performance compared to closed-source models such as GPT4-Turbo, Claude 3 Opus, and Gemini 1.5 Pro in coding and math benchmarks.  The list of supported programming languages can be found in the paper.

## 2. Model Downloads

We release the DeepSeek-Coder-V2 with 16B and 236B parameters based on the [DeepSeekMoE](https://arxiv.org/pdf/2401.06066) framework, which has actived parameters of only 2.4B and 21B , including base and instruct models, to the public. 

<div align="center">

|            **Model**            | **#Total Params** | **#Active Params** | **Context Length** |                         **Download**                         |
| :-----------------------------: | :---------------: | :----------------: | :----------------: | :----------------------------------------------------------: |
|   DeepSeek-Coder-V2-Lite-Base   |        16B        |        2.4B        |        128k        | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Base) |
| DeepSeek-Coder-V2-Lite-Instruct |        16B        |        2.4B        |        128k        | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct) |
|     DeepSeek-Coder-V2-Base      |       236B        |        21B         |        128k        | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Base) |
|   DeepSeek-Coder-V2-Instruct    |       236B        |        21B         |        128k        | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct) |

</div>


## 3. Chat Website

You can chat with the DeepSeek-Coder-V2 on DeepSeek's official website: [coder.deepseek.com](https://coder.deepseek.com/sign_in)

## 4. API Platform
We also provide OpenAI-Compatible API at DeepSeek Platform: [platform.deepseek.com](https://platform.deepseek.com/). Sign up for over millions of free tokens. And you can also pay-as-you-go at an unbeatable price.

<p align="center">
  <img width="40%" src="https://github.com/deepseek-ai/DeepSeek-Coder-V2/blob/main/figures/model_price.jpg?raw=true">
</p>


## 5. How to run locally
**Here, we provide some examples of how to use DeepSeek-Coder-V2-Lite model. If you want to utilize DeepSeek-Coder-V2 in BF16 format for inference, 80GB*8 GPUs are required.**

### Inference with Huggingface's Transformers
You can directly employ [Huggingface's Transformers](https://github.com/huggingface/transformers) for model inference.

#### Code Completion
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Base", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
input_text = "#write a quick sort algorithm"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

#### Code Insertion
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Base", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
input_text = """<｜fim▁begin｜>def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = []
    right = []
<｜fim▁hole｜>
        if arr[i] < pivot:
            left.append(arr[i])
        else:
            right.append(arr[i])
    return quick_sort(left) + [pivot] + quick_sort(right)<｜fim▁end｜>"""
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True)[len(input_text):])
```

#### Chat Completion

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
messages=[
    { 'role': 'user', 'content': "write a quick sort algorithm in python."}
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
# tokenizer.eos_token_id is the id of <|EOT|> token
outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))
```



The complete chat template can be found within `tokenizer_config.json` located in the huggingface model repository.

An example of chat template is as belows:

```bash
<｜begin▁of▁sentence｜>User: {user_message_1}

Assistant: {assistant_message_1}<｜end▁of▁sentence｜>User: {user_message_2}

Assistant:
```

You can also add an optional system message:

```bash
<｜begin▁of▁sentence｜>{system_message}

User: {user_message_1}

Assistant: {assistant_message_1}<｜end▁of▁sentence｜>User: {user_message_2}

Assistant:
```

### Inference with vLLM (recommended)
To utilize [vLLM](https://github.com/vllm-project/vllm) for model inference, please merge this Pull Request into your vLLM codebase: https://github.com/vllm-project/vllm/pull/4650.

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

max_model_len, tp_size = 8192, 1
model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model=model_name, tensor_parallel_size=tp_size, max_model_len=max_model_len, trust_remote_code=True, enforce_eager=True)
sampling_params = SamplingParams(temperature=0.3, max_tokens=256, stop_token_ids=[tokenizer.eos_token_id])

messages_list = [
    [{"role": "user", "content": "Who are you?"}],
    [{"role": "user", "content": "write a quick sort algorithm in python."}],
    [{"role": "user", "content": "Write a piece of quicksort code in C++."}],
]

prompt_token_ids = [tokenizer.apply_chat_template(messages, add_generation_prompt=True) for messages in messages_list]

outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)

generated_text = [output.outputs[0].text for output in outputs]
print(generated_text)
```



## 6. License

This code repository is licensed under [the MIT License](https://github.com/deepseek-ai/DeepSeek-Coder-V2/blob/main/LICENSE-CODE). The use of DeepSeek-Coder-V2 Base/Instruct models is subject to [the Model License](https://github.com/deepseek-ai/DeepSeek-Coder-V2/blob/main/LICENSE-MODEL). DeepSeek-Coder-V2 series (including Base and Instruct) supports commercial use.


## 7. Contact
If you have any questions, please raise an issue or contact us at [service@deepseek.com](service@deepseek.com).
