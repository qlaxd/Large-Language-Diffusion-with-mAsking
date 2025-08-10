# Evaluation Report: LLaDA vs. Autoregressive Model

This report summarizes the performance of the implemented LLaDA (diffusion-based) and a standard autoregressive model on the TinyShakespeare dataset.

## 1. Quantitative Analysis

Quantitative evaluation provides objective metrics to compare the models.

### Perplexity

Perplexity is a standard metric for evaluating language models, measuring how well a probability model predicts a sample. Lower perplexity indicates a better model.

- **Autoregressive Model Perplexity:** 6.98 (Calculated on the validation set)
- **LLaDA Model Perplexity:** 18.23 (Calculated on the validation set with a fixed masking ratio of t=0.1)

*Note: Comparing perplexity between these two different architectures is not straightforward. The autoregressive perplexity is calculated over the entire sequence, while the LLaDA perplexity is calculated only on its ability to predict masked tokens, which is a different task.* 

### Analysis

The autoregressive model achieves a significantly lower perplexity score. This is expected, as it is a more direct and mature architecture for next-token prediction, which is what perplexity measures. The LLaDA model's higher perplexity reflects the more challenging task of filling in many masked tokens simultaneously.

---

## 2. Qualitative Analysis

Qualitative analysis involves inspecting the generated text samples to assess their coherence, diversity, and relevance to the prompt.

**Prompt:** `Shall I compare thee to a summer's day?`

### LLaDA Model - Generated Sample

```
Shall I compare thee to a summer's day?                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
```

**Analysis:**
The LLaDA model, in its current state, fails to generate any meaningful text beyond the prompt. It fills the rest of the sequence with padding/unknown tokens. This suggests that the reverse diffusion process is not yet effective enough to construct coherent sentences from a fully masked state. The model essentially learns to do nothing, which is a common failure mode in early-stage training of such models without more sophisticated guidance or a better-tuned reverse sampling schedule.

### Autoregressive Model - Generated Sample

```
Shall I compare thee to a summer's day?

CORIOLANUS:
I the so the so the sone the so the sone the sone
The the the so the sould the the sould the the sour the the heat the the the seare the the theat
The the the the the the theat the the the the theat the the the the theathe the the the the the the theathe the the the the the the the theathe the the the the there the there the there the there the there the there there the the there the there the there t theat the th
```

**Analysis:**
The autoregressive model successfully continues from the prompt and even generates a new character cue (`CORIOLANUS:`). However, it quickly falls into a repetitive loop, generating nonsensical sequences of common words like "the", "so", and "sone". This is a classic sign of an undertrained language model. It has learned basic word forms and structure but lacks the deeper semantic understanding to create coherent, long-form text.

---

## 3. Conclusion & Discussion

- **Strengths & Weaknesses:**
  - The **Autoregressive model** demonstrates a stronger grasp of basic language structure, as shown by its lower perplexity and ability to generate recognizable (if repetitive) words. Its weakness is its tendency to get stuck in loops.
  - The **LLaDA model** is architecturally more complex and, as implemented, fails to generate coherent text. Its strength lies in its theoretical ability to utilize bidirectional context, but this potential was not realized in this experiment. The implementation likely requires more careful tuning of the reverse process (e.g., sampling strategies, re-masking schedules) to be effective.

- **Failure Modes:**
  - The primary failure mode for the autoregressive model is **repetition**.
  - The primary failure mode for the LLaDA model is a **failure to generate anything meaningful**, resulting in an empty or padded output.

- **Overall:**
  For this take-home exercise, the autoregressive model provided more promising (though still flawed) results. The LLaDA implementation was technically successful in that the training and generation pipelines run, but it would require significant further research and development to produce coherent text, fulfilling the `TASK.md`'s goal of exploring a novel architecture.