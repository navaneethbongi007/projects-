# projects-
---

# 🦙 Fine‑Tuning Llama‑2‑7B on Google Colab

## 📘 Overview
This notebook demonstrates how to fine‑tune the **Llama‑2‑7B language model** on a single Google Colab GPU using **PEFT (Parameter‑Efficient Fine‑Tuning)** and **QLoRA (Quantized Low‑Rank Adaptation)**.  
The goal is to adapt the model into a **chatbot for mental health counseling conversations**, leveraging Hugging Face’s ecosystem.

---

## ⚙️ Setup
The notebook installs and configures the following libraries:
- `transformers`, `trl`, `accelerate`  
- `peft` (from Hugging Face GitHub)  
- `datasets` for loading training data  
- `bitsandbytes` for 4‑bit quantization  
- `einops` (required for Falcon models)  
- `wandb` for experiment tracking  

---

## 📂 Dataset
We use the Hugging Face dataset:  
**`Amod/mental_health_counseling_conversations`**  
- ~3,500 rows  
- Features:  
  - `Context` → user’s input (mental health scenario)  
  - `Response` → counselor’s reply  

Example:
```json
{
  "Context": "I'm going through some things with my feelings...",
  "Response": "If everyone thinks you're worthless, then maybe you need to find new people..."
}
```

---

## 🧠 Model
- Base model: `TinyPixel/Llama-2-7B-bf16-sharded`  
- Quantization: 4‑bit (`nf4`) with `bitsandbytes`  
- Tokenizer: AutoTokenizer with EOS padding  
- Fine‑tuning: LoRA configuration (`alpha=16`, `dropout=0.1`, `r=64`)  

---

## 🏋️ Training
- Trainer: Hugging Face `SFTTrainer`  
- Batch size: 4  
- Gradient accumulation: 4  
- Optimizer: `paged_adamw_32bit`  
- Learning rate: `2e-4`  
- Max steps: 100  
- Sequence length: 3000  

⚠️ Note: Training may hit **CUDA OutOfMemoryError** on Colab T4 GPUs. Adjust batch size, sequence length, or use gradient checkpointing to mitigate.

---

## 📦 Saving & Sharing
- Model saved locally in `outputs/`  
- LoRA config reloaded via `get_peft_model`  
- Model pushed to Hugging Face Hub:  
  `https://huggingface.co/ashishpatel26/Llama2_Finetuned_Articles_Constitution_3300_Instruction_Set` [(huggingface.co in Bing)](https://www.bing.com/search?q="https%3A%2F%2Fhuggingface.co%2Fashishpatel26%2FLlama2_Finetuned_Articles_Constitution_3300_Instruction_Set")

---

## ▶️ Inference
Example usage:
```python
text = dataset['Context'][0]
inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 📊 Results
- The model generates chatbot‑style responses aligned with counseling data.  
- Sample output shows empathetic but sometimes repetitive replies.  
- Further fine‑tuning or prompt engineering can improve quality.

---

## 📝 Notes
- Colab T4 GPU has **15GB VRAM**; large models may exceed memory.  
- Use smaller batch sizes or gradient checkpointing to avoid OOM errors.  
- Hugging Face Hub integration allows sharing and collaboration.  
