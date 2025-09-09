

# GPT2Twin: Dual LoRA Training for Generation and Truth Evaluation

## 🔹 Overview

**GPT2Twin** is a training framework that uses two LoRA-adapted GPT-2 models working together:

* **Generator LoRA**: learns to generate responses from prompts using standard causal language modeling (next-token prediction).
* **Discriminator LoRA**: learns to classify or regress how close a generated response is to the target ground truth.

Together, they form a **twin system**: one produces, the other evaluates.
This creates a self-contained loop where GPT-2 both **learns language** and **learns the truth of its own outputs**.

---

## 🔹 Motivation

Traditional training setups use:

* A single GPT-2 model fine-tuned on prompt→response pairs, or
* Reinforcement learning with external reward models (e.g., sentence transformers).

**GPT2Twin** removes the external dependency. Both generation and evaluation live inside the **same GPT-2 architecture with lightweight LoRA adapters**.

This makes the system:

* **Parameter-efficient**: only LoRA weights are trained.
* **Symmetric**: both generator and discriminator share the same backbone.
* **Flexible**: the discriminator can output binary labels (`correct / incorrect`) or continuous scores (`0..1` similarity).

---

## 🔹 How It Works

### 1. Generator LoRA

* Based on `GPT2LMHeadModel + LoRA`.
* Trained with **causal LM loss**:

  ```
  L_gen = CrossEntropy(next_token_prediction)
  ```

### 2. Discriminator LoRA

* Based on `GPT2ForSequenceClassification + LoRA` (or regression head).
* Input: `"GEN: <generated> || TAR: <target>"`.
* Output: probability or score of similarity.
* Trained with **supervised classification/regression loss**:

  ```
  L_disc = CrossEntropy(label)   # or MSE for regression
  ```

### 3. Coupled Training (optional)

* The generator can also receive a **reward loss** from the discriminator:

  ```
  L_gen = LM_loss + λ * (-log P_disc(close))
  ```
* This makes it similar to **RLHF**, but the reward model is another GPT-2 LoRA.

---

## 🔹 Example Training Loop

```python
stats = twin.step(
    prompt="The capital of France is",
    target_text="Paris",
    target_label=1  # 1 = close/correct
)
print(stats)
```

Output:

```json
{
  "loss_gen": 2.13,
  "loss_disc": 0.45,
  "gen_text": "The capital of France is Paris",
  "prob_close": 0.92
}
```

---

## 🔹 Applications

* **Language learning models** that self-evaluate.
* **RLHF-like pipelines** without external reward models.
* **Adversarial self-training**: generator produces, discriminator enforces truth.
* **Research experiments** in parameter-efficient multi-agent learning.

---

## 🔹 Future Work

* Extend discriminator to multi-class scoring (grading scale instead of binary).
* Apply to larger backbones (GPT-NeoX, LLaMA) with LoRA adapters.
* Explore curriculum learning: discriminator guides generator in increasingly difficult tasks.
* Investigate stability compared to GANs and PPO-based RLHF.

---

## 🔹 Summary

**GPT2Twin** is a simple but powerful idea:
👉 *Two LoRA GPT-2s, one generating, one judging.*

It provides a lightweight way to align a generator with truth, using only GPT-2 backbones and LoRA adapters — no external models required.
