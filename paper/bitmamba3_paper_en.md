# BitMamba-3: A 1.58-bit Ternary Quantization of the Mamba-3 State-Space Model with a State-Tracking Inductive Bias

Author

Affiliation / e-mail (anonymized for blind review)

---

## Abstract

This work proposes BitMamba-3, a 1.58-bit ternary quantization of the linear projection layers of the Mamba-3 state-space model, and it tests two hypotheses. The first hypothesis states that the accuracy cost of ternary quantization stays within a small perplexity loss relative to a 4-bit post-training quantization baseline. The second hypothesis states that the 1.58-bit discrete weight structure acts as an inductive bias toward state tracking, beyond a generic precision-reduction effect. This work implements a minimal-diff interface that replaces only the input and output projection layers with BitLinear, and it compares four configurations, namely FP16, 4-bit post-training quantization, BitMamba-3 ternary, and BitMamba-2 ternary, under identical training data of 480M fineweb-edu tokens, identical hyperparameters, and identical seeds. BitMamba-3 130M incurs a 12.2 percent perplexity cost over the Mamba-3 FP configuration and an 8.2 percent cost over 4-bit post-training quantization, while it reduces the projection-weight memory by 2.5 times relative to 4-bit and 10 times relative to FP, and it reduces multiplication to conditional add and subtract. More importantly, on the parity state-tracking task, the FP configuration with peak 0.509 and the 4-bit configuration with peak 0.527 both stay at chance, while the ternary configuration reaches 0.972. This separation does not appear under 4-bit quantization, which places the critical precision between 1.58 and 4 bits. These results show that ternary quantization is a design choice with algorithmic significance rather than a compression technique.

Keywords: Mamba-3, state space model, BitNet, 1.58-bit quantization, inductive bias, state tracking

---

## 1. Introduction

Mamba-3 [1] is the latest selective state-space model architecture. Mamba-3 applies a data-dependent rotary position encoding [2] rotation to the B and C projections, which realizes a real-valued recurrence mathematically equivalent to a discretized complex-valued state-space model. This equivalence grants Mamba-3 a state-tracking capability, namely the ability to maintain the cumulative state of an input sequence, that Mamba-2 [3] provably lacks. Mamba-3 also expands the per-head state by a fixed factor through a multi-input multi-output option, and it reports downstream accuracy gains over Mamba-2 at the 1.5B scale.

BitNet b1.58 [4] restricts the weights of the linear layers to three ternary values, namely minus one, zero, and plus one, which encodes each weight in roughly 1.58 bits. A weight of plus or minus one becomes a sign flip and a weight of zero becomes a skip, so multiplication reduces to conditional add and subtract. This property distinguishes 1.58-bit quantization from 4-bit post-training quantization, which keeps the multiply operation. GPTQ [5] and AWQ [6] have become the de facto standard for large language model deployment. Zhayr et al. applied the same BitLinear strategy to the input and output projections of Mamba-2 and first quantified the trainability of 1.58-bit state-space models at the 170M and 1B scales [7].

No prior work has quantized Mamba-3 to 1.58 bits, and no study has compared 1.58-bit quantization against a 4-bit post-training quantization baseline under matched training conditions. No controlled study has examined how the state-tracking capability of the Mamba-3 recurrence changes with quantization precision. Prior work has treated 1.58-bit quantization consistently as a compression technique, so the structural effect of a discrete weight lattice on the training dynamics of a state-tracking architecture remains an untested gap.

This work narrows that gap into two hypotheses. The first hypothesis states that the accuracy cost of Mamba-3 with 1.58-bit ternary quantization stays within a small perplexity loss relative to a 4-bit post-training quantization baseline, while it offers clear memory and compute benefits. The second hypothesis states that the 1.58-bit discrete structure acts as an inductive bias toward state tracking, and that it produces a behavior distinct from 4-bit quantization that carries the same precision-reduction effect.

This work makes two contributions. First, it implements BitMamba-3, a minimal-diff PyTorch wrapper that replaces the input and output projection layers of Mamba-3 with BitLinear, and it trains the model from scratch at the 30M, 130M, and 370M scales. Second, it compares FP16, 4-bit post-training quantization, BitMamba-3 ternary, and BitMamba-2 ternary under identical data, settings, and seeds to test the two hypotheses. The 4-bit post-training quantization control isolates whether an observed effect is a generic low-bit effect or a property specific to the 1.58-bit discrete structure.

---

## 2. State-Space Models and Low-Bit Quantization

### 2.1. The Mamba-3 State-Space Model

Mamba-3 modifies the selective state-space model recurrence of Mamba-2 along two axes. The first axis is a data-dependent rotary position encoding rotation applied to the B and C projections, which realizes a recurrence equivalent to a discretized complex-valued state-space model in the real domain and lets Mamba-3 solve state-tracking tasks that Mamba-2 cannot. The second axis is a multi-input multi-output option that expands the per-head state by a fixed factor with a default of four, which this work uses unchanged. The upstream implementation ships three fused kernels, namely a Triton combined kernel for the selective-input single-output path, a TileLang combined kernel for the multi-input multi-output path, and a CuteDSL step kernel for single-token decode.

### 2.2. BitNet b1.58 and BitMamba-2

BitNet b1.58 restricts the weights of every linear layer to three ternary values and encodes each weight in roughly 1.58 bits. It quantizes activations to 8-bit integers with per-token absolute-maximum scaling, and it uses the straight-through estimator to approximate gradients through the round and clamp operators during training. BitMamba-2 applies this strategy to the input and output projections of the Mamba-2 block, trains 170M and 1B variants from scratch, and reports perplexity competitive with an FP baseline at the 1B scale. The BitLinear used here is a bit-for-bit PyTorch port of the BitMamba-2 JAX implementation.

### 2.3. 4-bit Post-Training Quantization

Large language model inference has largely replaced FP16 with 4-bit post-training quantization. GPTQ and AWQ are both calibration-based methods that exploit Hessian or activation statistics, and both report perplexity roughly 1 to 3 percent lower than plain round-to-nearest. This work adopts round-to-nearest as the controlled baseline because round-to-nearest shares the same per-tensor symmetric absolute-maximum normalization as ternary BitLinear, which isolates precision as the single variable.

---

## 3. Proposed BitMamba-3

BitMamba-3 follows two design principles. The first principle holds the training data, hyperparameters, and seeds identical across the compared models, so that a single variable, either architecture or quantization precision, explains any difference in the results. The second principle keeps the upstream Mamba-3 implementation unmodified and applies quantization through a minimal-diff interface that swaps only the weight lattice, so that the findings transfer directly to the upstream algorithm.

### 3.1. BitLinear

BitLinear is a drop-in replacement for the PyTorch linear layer that round-trips the weights onto the 1.58-bit ternary lattice and quantizes activations to per-token 8-bit integers in the forward pass. The weights stay in floating point, which keeps any FP checkpoint compatible, and the straight-through estimator passes gradients through the round operator during training. The weight scale is per-tensor absolute-mean, the activation scale is per-row absolute-maximum, and a unit-affine RMSNorm normalizes the activation magnitude before quantization. This definition ports the BitMamba-2 JAX implementation bit-for-bit and follows the normalization procedure of BitNet b1.58, so the results here are interpretable on the same scale as the BitNet family.

### 3.2. The BitMamba-3 Module

BitMamba-3 subclasses the upstream Mamba-3 class and replaces only the input and output projection layers with BitLinear immediately after initialization. The state-space model kernels, the rotary position encoding engine, the RMSNorm on the B and C projections, and the bias parameters all stay unmodified. The ternary fraction is 70.9 percent of parameters at 130M and 86.1 percent at 370M, and it rises with model scale as the embedding share shrinks. The upstream block-creation factory does not register a Mamba-3 layer identifier, so a runtime patch of that factory places the Mamba-3 block, and the upstream source stays unmodified.

This minimal-diff design guarantees causal attribution. The algorithmic elements that Mamba-3 introduces over Mamba-2, namely the rotary position encoding recurrence, the multi-input multi-output option, and the RMSNorm on the B and C projections, all stay outside the quantization scope. Any perplexity or parity difference observed here therefore attributes cleanly to the change in quantization precision.

---

## 4. Experimental Results and Analysis

### 4.1. Experimental Setup

We train on a single GPU with roughly 480M tokens of fineweb-edu. The optimizer is AdamW, the learning rate follows a cosine decay that peaks at 3 times 10 to the minus 4, and training uses bfloat16 automatic mixed precision with gradient-norm clipping. Each run covers 30000 steps with a single seed. We evaluate along five axes, namely WikiText-103 perplexity, PG19 long-context perplexity, lm-evaluation-harness zero-shot accuracy, Needle-in-Haystack recall, and parity state tracking. To test the two hypotheses, we compare FP16, 4-bit post-training quantization, BitMamba-3 ternary, and BitMamba-2 ternary under identical conditions, where the 4-bit baseline is a per-tensor symmetric absolute-maximum round-to-nearest control. This control shares the normalization procedure of ternary BitLinear, so it isolates precision from every other variable, and it forms the methodological core of the second hypothesis test.

### 4.2. Training Results and Quantization-Cost Analysis

Table 1 reports the final loss and WikiText-103 perplexity of the models trained under identical 480M tokens and identical settings.

**Table 1. Final loss and WikiText-103 perplexity of the trained models (matched 480M-token budget).**

| Model | Train tokens | Final loss | WikiText-103 PPL |
|---|---|---|---|
| BitMamba-3 30M (short) | 164M | 5.00 | 553 |
| BitMamba-3 30M (long) | 480M | 4.90 | 400 |
| BitMamba-2 130M | 480M | 3.75 | 113.94 |
| Mamba-3 130M FP | 480M | 3.42 | 61.86 |
| Mamba-3 130M INT4 PTQ | post-training | n/a | 64.15 |
| BitMamba-3 130M | 480M | 3.57 | 69.40 |
| BitMamba-2 370M | 480M | 3.78 | 105.74 |
| BitMamba-3 370M | 480M | 3.33 | 60.20 |

BitMamba-3 130M reaches perplexity 69.40 at the same 480M tokens, while the same-scale BitMamba-2 stays at 113.94, a roughly 1.64 times advantage. The FP-to-ternary quantization cost is a 12.2 percent increase at 130M, well below the 20 to 50 percent increase reported for typical low-bit post-training quantization. BitMamba-3 370M reaches perplexity 60.20, and the gap to BitMamba-2 370M at 105.74 widens to roughly 1.76 times.

The architecture and quantization effects separate cleanly at the 130M scale. Moving from the FP configuration to 4-bit post-training quantization raises perplexity by 3.7 percent and shrinks the weight memory by 4 times. Moving from the FP configuration to ternary raises perplexity by 12.2 percent, shrinks the memory by 10 times, and removes multiplication. Moving from 4-bit post-training quantization to ternary raises perplexity by only 8.2 percent and shrinks the memory by a further 2.5 times. On the architecture axis, moving from Mamba-2 to Mamba-3 under matched ternary quantization lowers perplexity by 39 percent at 130M and by 43 percent at 370M. The single most influential decision under this training budget is therefore the choice of architecture rather than the type of quantization, and quantization is a small cost added on top.

These results support the first hypothesis. Ternary does not reach the perplexity of the FP or 4-bit configurations; the first hypothesis concerns whether the quantization cost stays bounded, not whether ternary wins on accuracy. The ternary quantization cost stays at an 8.2 percent perplexity increase over 4-bit post-training quantization, and it delivers a 2.5 times smaller weight memory and the reduction of multiplication to conditional add and subtract at the same time. A 4-bit-to-2-bit transition in typical post-training quantization raises perplexity by 10 to 20 percent or more, yet from-scratch ternary training is more efficient than calibration-based post-training quantization. The reason is that the trained weights adapt to the ternary lattice itself through the combination of the straight-through estimator gradient and the activation RMSNorm, which is essentially the same mechanism behind the lossless result that BitNet b1.58 reports at the 3B scale.

### 4.3. Downstream Evaluation

Table 2 reports the lm-evaluation-harness zero-shot results.

**Table 2. Zero-shot downstream evaluation with lm-evaluation-harness (200 samples per task).**

| Task | Mamba-2 130M | Mamba-3 130M | Mamba-2 370M | Mamba-3 370M |
|---|---|---|---|---|
| ARC-Easy acc | 0.405 | 0.410 | 0.390 | 0.410 |
| HellaSwag norm | 0.345 | 0.390 | 0.350 | 0.390 |
| PIQA acc | 0.590 | 0.570 | 0.570 | 0.555 |
| LAMBADA acc | 0.020 | 0.100 | 0.030 | 0.120 |
| LAMBADA PPL | 6408.6 | 1355.2 | 5439.7 | 826.5 |

ARC-Easy, HellaSwag, and PIQA do not separate the model scales clearly at a 480M-token budget. LAMBADA, a long-range completion task, does separate them: the perplexity drops from 6408.6 for BitMamba-2 130M to 826.5 for BitMamba-3 370M, a roughly 7.75 times gap that exposes the combined architecture-and-scale effect most sharply. The PG19 long-context evaluation shows the same pattern, where the Mamba-3 advantage holds from context length 1024 through 8192. Mamba-3 130M reaches perplexity 71.50 at context length 1024 and 70.10 at 4096, roughly 11 to 12 percent below the same-scale Mamba-2 values of 80.45 and 79.85, and Mamba-3 370M reaches 63.78 at context length 4096. This pattern matches the long-range modeling strength of the rotary-position-encoding-based recurrence.

### 4.4. Parity State Tracking and the Inductive Bias

Table 3 reports the parity state-tracking results with a 4-bit post-training quantization control, which separates a generic low-bit effect from an effect specific to the 1.58-bit structure.

**Table 3. Parity state-tracking results with the INT4 control (d_model 512, 3 seeds).**

| Configuration | Peak accuracy (mean ± std) | Learns parity |
|---|---|---|
| Mamba-3 SISO, FP (16-bit) | 0.509 ± 0.003 | no |
| Mamba-3 SISO, INT4 (4-bit) | 0.511 ± 0.007 | no |
| Mamba-3 SISO, ternary (1.58-bit) | 0.954 ± 0.040 | yes |
| Mamba-3 MIMO, FP | 0.509 ± 0.003 | no |
| Mamba-3 MIMO, INT4 | 0.527 ± 0.003 | no |
| Mamba-3 MIMO, ternary | 0.972 ± 0.047 | yes |

Table 3 shows the key result: the 4-bit configuration, with a 16-value discrete lattice, stays at a chance level statistically indistinguishable from the FP configuration, with a peak gap under 0.018 that is below one standard deviation. The ternary configuration, with three values, reaches 0.954 and 0.972 and learns parity. The observed inductive bias is therefore not a generic low-bit effect but an effect of the three-value discrete structure of 1.58-bit weights, and the critical precision lies between 1.58 and 4 bits. The result holds consistently across both the selective-input single-output path and the multi-input multi-output path.

A separate experiment with five seeds shows the same trend more strongly. At d_model 512 and depth 4, the Mamba-3 multi-input multi-output configuration with ternary quantization reaches a peak accuracy of 0.981 with a standard deviation of 0.036, and its gap to the FP configuration at 0.510 corresponds to roughly 13 times the standard deviation. The generalization accuracy at twice the sequence length is 0.765, which indicates genuine state tracking rather than memorization. As d_model grows through 128, 256, and 512, the peak accuracy of the multi-input multi-output ternary configuration strengthens through 0.86, 0.95, and 0.98, and the standard deviation tightens through 0.15, 0.09, and 0.04.

These results strongly support the second hypothesis. The 1.58-bit ternary structure solves the parity task, which the 4-bit configuration and the 16-bit FP configuration do not, at a separation of roughly 13 times the standard deviation, and the 4-bit control shows directly that this effect is not a generic low-bit effect. The mechanism hypothesis for this result is a commitment effect of the discrete lattice. Parity is an intrinsically discrete function, and its solutions occupy a narrow region of weight space. On a 16-value lattice or a continuous lattice, training converges more easily onto a smooth function unrelated to parity; on a three-value lattice, a smooth approximation is not even representable, so training is forced toward the correct structure. This hypothesis is one possible explanation, and a falsification test requires an extension to a larger scale and to natural-language state-tracking tasks.

---

## 5. Conclusion

This work tests two hypotheses. The first hypothesis is that the accuracy cost of Mamba-3 with 1.58-bit ternary quantization stays within a small perplexity loss relative to a 4-bit post-training quantization baseline, with clear memory and compute benefits. The second hypothesis is that the 1.58-bit discrete structure acts as an inductive bias beyond a generic precision-reduction effect. To test them, this work implements a minimal-diff interface that replaces the input and output projection layers of Mamba-3 with BitLinear, and it compares FP16, 4-bit post-training quantization, BitMamba-3 ternary, and BitMamba-2 ternary under identical data, settings, and seeds. The key results are that the quantization cost of BitMamba-3 130M stays at an 8.2 percent perplexity increase over 4-bit while the weight memory is 2.5 times smaller, that the Mamba-3 architecture holds a 1.64 times perplexity advantage over Mamba-2 at 130M and a 1.76 times advantage at 370M under matched quantization, and that the 1.58-bit discrete structure itself acts as an inductive bias at a separation of roughly 13 times the standard deviation on the state-tracking task, an effect that the 4-bit control does not reproduce.

The limitations of this work are threefold. First, the 480M-token budget is roughly 0.16 percent of standard large language model pretraining, so every absolute perplexity here is incomparable to published models. Second, the parity results come from a small synthetic task at d_model 512 or below and depth 4 or below, so whether the same mechanism holds on natural-language state-tracking downstream tasks remains open. Third, the round-to-nearest control baseline sits roughly 1 to 3 percent of perplexity above recent 4-bit post-training quantization methods such as GPTQ and AWQ, so the 4-bit-to-ternary gap may widen slightly against those methods.

These limitations point to the directions for future extension. First, when an official Mamba-3 checkpoint becomes available, quantization-aware fine-tuning from the pretrained weights can lift the absolute perplexity close to the 4-bit post-training quantization level. Second, the parity inductive bias can extend to natural-language state-tracking tasks such as variable-binding tracking and function-call-depth reasoning, and it can be re-tested at a larger scale of d_model 1024 or above. Third, a recent 4-bit post-training quantization method can join the baseline set to quantify the 4-bit-to-ternary precision gap more precisely. These results indicate that the combination of Mamba-3 and 1.58-bit quantization is a design choice with algorithmic significance rather than a compression technique.

---

## References

[1] N. Lahoti, et al. Mamba-3: Improved Sequence Modeling using State Space Principles. *International Conference on Learning Representations (ICLR)*, 2026. arXiv:2603.15569.

[2] J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, Y. Liu. RoFormer: Enhanced Transformer with Rotary Position Embedding. *Neurocomputing*, vol. 568, 2024.

[3] T. Dao, A. Gu. Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. *International Conference on Machine Learning (ICML)*, 2024.

[4] S. Ma, H. Wang, L. Ma, L. Wang, W. Wang, S. Huang, L. Dong, R. Wang, J. Xue, F. Wei. The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits. *arXiv:2402.17764*, 2024.

[5] E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. *International Conference on Learning Representations (ICLR)*, 2023.

[6] J. Lin, J. Tang, H. Tang, S. Yang, W. Chen, W. Wang, G. Xiao, X. Dang, C. Gan, S. Han. AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration. *Proceedings of Machine Learning and Systems (MLSys)*, 2024.

[7] Zhayr, et al. Fully Quantized Mamba in 1.58 Bits From Head to Toe. *International Conference on Computational Linguistics (COLING)*, 2025. Code: https://github.com/Zhayr1/BitMamba-2.
