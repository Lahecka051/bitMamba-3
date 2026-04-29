# 용어 사전 — 지표 방향성과 의미

본 문서는 BitMamba-3 논문에 등장하는 모든 지표·용어를 정리하고 **각 지표가 낮을수록 좋은지 / 높을수록 좋은지**를 명시한다.

> ⚠️ 가장 흔한 혼동: **PPL은 낮을수록 좋다**. "PPL이 낮아졌다 = 성능이 좋아졌다"

---

## 1. 평가 지표 (Evaluation Metrics)

### 1.1 Perplexity (PPL) — **낮을수록 좋음 ↓**

- **정의**: `PPL = exp(평균 cross-entropy loss)`
- **직관**: 모델이 다음 토큰을 예측할 때의 평균 "헷갈림" 정도. 매 단어마다 몇 개의 후보 중 하나를 고르는 만큼 헷갈리는지를 표현
- **방향**: **낮을수록 좋음**
- **범위**:
  - 완벽한 모델 → PPL = 1 (정답 1개만 확신)
  - 무작위 추론 → PPL ≈ vocab 크기 (예: 50,000)
  - 잘 학습된 LLM → PPL 10~30
  - 본 논문의 130M 학습량 부족 모델 → PPL 60~120
- **사용 데이터셋**:
  - **WikiText-103**: 위키피디아 발췌, 단문 위주
  - **LAMBADA**: 마지막 단어 맞추기 (long-range completion)
  - **PG19**: Project Gutenberg 1919년 이전 장편 소설 (long-context)

### 1.2 Cross-entropy Loss — **낮을수록 좋음 ↓**

- **정의**: 정답 토큰의 음의 로그 확률 평균. `loss = -log P(정답)`
- **방향**: **낮을수록 좋음** (PPL과 같은 방향, `PPL = exp(loss)`)
- **범위**:
  - 완벽 → 0
  - 무작위 → log(vocab) ≈ 10.8 (vocab=50K 기준)
  - 학습 초기 → 10~11
  - 학습 후 → 3~5

### 1.3 Accuracy (acc) — **높을수록 좋음 ↑**

- **정의**: 정답률. 0과 1 사이 값
- **방향**: **높을수록 좋음**
- **사용처**:
  - **ARC-Easy / ARC-Challenge**: 초등 / 고등 과학 객관식 (4지선다, 무작위 = 0.25)
  - **HellaSwag**: 문장 마무리 추론 (4지선다, 무작위 = 0.25)
  - **PIQA**: 물리 상식 추론 (2지선다, 무작위 = 0.50)
  - **LAMBADA acc**: 마지막 단어 정확히 맞춘 비율
  - **Parity acc**: bit 시퀀스의 누적 XOR 정답률 (2지선다, 무작위 = 0.50)

### 1.4 acc_norm (normalized accuracy) — **높을수록 좋음 ↑**

- **정의**: 정답 후보 길이 차이를 정규화한 정확도. 짧은 정답이 부당하게 높은 확률을 받는 효과 보정
- **사용처**: HellaSwag, ARC, PIQA에서 일반적
- **방향**: **높을수록 좋음**
- **본 논문에서 acc보다 의미 있음**: 정확한 비교 시 acc_norm 사용 권장

### 1.5 log probability (log-prob, log_prob) — **0에 가까울수록 좋음 ↑**

- **정의**: 정답 토큰들의 로그 확률 합 또는 평균. 항상 음수 (`log P(x) < 0`)
- **방향**: **0에 가까울수록(덜 음수일수록) 좋음**
- **사용처**: Needle-in-haystack에서 magic number 회상 점수
- **예시 해석**:
  - log_prob = −3 → 모델이 약 5%(`exp(-3)`) 확률로 정답
  - log_prob = −10 → 모델이 약 0.005% 확률로 정답 (거의 못 맞춤)
- **본 논문 결과**: BitMamba-3 130M의 L=2K depth=100% = **−4.67** (좋음), depth=0% = −9.70 (덜 좋음)

### 1.6 Mean Log-Prob — **0에 가까울수록 좋음 ↑**

- log-prob의 시도 평균. 동일한 방향성

### 1.7 Needle-in-Haystack 결과 해석

- **정의**: 긴 문맥(haystack) 안에 숨긴 한 문장(needle)의 정보를 모델이 회상하는 정도를 mean log-prob로 측정
- **컨텍스트 L** (L=512, 2048, 4096): 길이가 길수록 어려움
- **깊이 (depth)** (0%, 50%, 100%):
  - **0%**: needle이 문맥 시작 부분 (질문에서 가장 먼 = 가장 잊기 쉬움)
  - **50%**: 중간
  - **100%**: 문맥 끝 (질문에 가까움 = 가장 회상 쉬움)
- **본 논문 결과 패턴**: 모든 모델에서 depth=100% > depth=50% > depth=0% (예상)

---

## 2. Parity Task (상태 추적)

### 2.1 Peak parity accuracy — **1에 가까울수록 좋음 ↑**

- **정의**: 학습 중 도달한 최고 parity 정확도. 5개 시드의 평균±표준편차로 보고
- **방향**: **높을수록 좋음**
- **무작위 수준**: 0.50 (2지선다)
- **본 논문의 핵심 결과**:
  - Mamba-3 + ternary @ d=512: **0.981 ± 0.036** (5/5 시드 학습 성공)
  - Mamba-3 + INT4 @ d=512: 0.527 ± 0.003 (random)
  - Mamba-3 + FP @ d=512: 0.510 ± 0.003 (random)

### 2.2 Final parity accuracy — **높을수록 좋음 ↑**

- 학습 종료 시점의 정확도. peak보다 낮을 수 있음 (불안정)

### 2.3 2× seqlen 일반화 정확도 — **높을수록 좋음 ↑**

- 학습 시퀀스 길이의 2배 길이에서 측정한 정확도
- **본 논문에서 중요**: 진정한 state-tracking인지 단순 외움인지 구분

---

## 3. 양자화 / 정밀도 용어

### 3.1 비트 수 (bits per parameter) — **낮을수록 메모리 적음, 정확도 손실 가능**

- **FP16 (16비트 부동소수점)**: 표준 학습 정밀도
- **INT8 (8비트 정수)**: 활성값 양자화에 흔히 사용
- **INT4 (4비트 정수)**: 가중치 양자화 표준 (GPTQ, AWQ 등)
- **Ternary (1.58비트)**: {−1, 0, +1} 세 값. log₂(3) ≈ 1.58 비트로 표현
- **방향성 분리**:
  - 비트 ↓ → 메모리 ↓ (좋음), 일반적으로 정확도 ↓ (나쁨)
  - 단, 본 논문에서 ternary는 inductive bias 효과로 일부 task에서 정확도 ↑

### 3.2 PTQ (Post-Training Quantization)

- **정의**: 학습 후 가중치를 양자화. 추가 학습 없음
- **장점**: 빠름, 보정 데이터만 필요
- **단점**: 일반적으로 학습-인식 양자화보다 PPL 손실 큼
- 본 논문의 INT4 baseline은 PTQ

### 3.3 QAT (Quantization-Aware Training)

- **정의**: 학습 중 양자화 효과를 시뮬레이션하면서 학습
- 본 논문의 BitMamba-3는 from-scratch QAT의 일종

### 3.4 RTN (Round-To-Nearest)

- **정의**: 가장 단순한 양자화 — 가장 가까운 격자점으로 반올림
- 본 논문 INT4 baseline에 사용

### 3.5 STE (Straight-Through Estimator)

- **정의**: 라운드/클리핑 같은 비미분 연산의 기울기를 항등 함수로 근사하여 통과시키는 학습 기법
- 본 논문 BitLinear에 사용

### 3.6 absmax / absmean

- **absmax**: 텐서의 절댓값 최댓값. 활성값 스케일링에 사용 (per-row absmax)
- **absmean**: 텐서의 절댓값 평균. ternary 가중치 스케일링에 사용 (per-tensor absmean)

---

## 4. 모델 아키텍처 용어

### 4.1 Mamba-2 / Mamba-3 / SSM

- **SSM (State Space Model)**: Transformer 대안 시퀀스 모델. O(L) 메모리로 긴 시퀀스 처리
- **Mamba-2**: 선택적 SSM. RoPE 없음. parity 못 풀음
- **Mamba-3**: Mamba-2 + RoPE 회귀. 이산화된 복소-값 SSM과 등가. parity 풀 수 있음

### 4.2 SISO / MIMO

- **SISO (Single-Input Single-Output)**: 표준 SSM 회귀. 헤드 당 상태 1개
- **MIMO (Multi-Input Multi-Output)**: 헤드 당 상태를 mimo_rank 배수로 확장
- **mimo_rank**: 상태 확장 배수 (Mamba-3 기본값 = 4)

### 4.3 d_model / d_state / d_inner / headdim

- **d_model**: hidden state 차원 (모델 너비)
- **d_state**: SSM 상태 차원 (Mamba-3 기본값 128, 본 논문 64로 축소)
- **d_inner**: SSM 내부 확장 차원 (보통 d_model × 2)
- **headdim**: 어텐션-style 헤드 당 차원

### 4.4 chunk_size

- **정의**: SSM chunk-scan 알고리즘의 청크 크기
- 메모리/속도 trade-off에 영향. 본 논문은 Blackwell shmem 한계로 8 사용

### 4.5 RoPE (Rotary Positional Embedding)

- **정의**: 회전 행렬을 통한 위치 인코딩
- Mamba-3에서는 B/C 투영에 적용되어 복소-값 SSM과 등가 회귀 구현

### 4.6 RMSNorm

- **정의**: Root Mean Square Normalization. LayerNorm의 단순화 버전
- **공식**: `RMSNorm(x) = x / sqrt(mean(x²) + ε)`
- 본 논문 BitLinear의 활성 정규화 단계에서 사용

### 4.7 BitLinear

- **정의**: nn.Linear의 ternary 가중치 + INT8 활성 대체
- BitNet b1.58의 핵심 모듈
- 본 논문에서 BitMamba-3의 in_proj/out_proj에 적용

### 4.8 in_proj / out_proj

- **in_proj**: SSM 블록 입력 투영. 가중치가 가장 큰 layer 중 하나
- **out_proj**: SSM 블록 출력 투영
- 본 논문에서 양자화 대상

---

## 5. 학습 관련 용어

### 5.1 학습 토큰 (training tokens) — **많을수록 일반적으로 좋음 ↑**

- **정의**: 학습에 사용된 총 토큰 수
- 본 논문: **480M tokens** (4억 8천만)
- 비교: published Mamba-2 130M = 300B tokens (3000억) — 본 논문의 625배

### 5.2 Step (학습 스텝)

- **정의**: 하나의 mini-batch에 대해 forward + backward + optimizer step
- 본 논문: 30K steps × (effective batch 16 × seqlen 2048) = 약 1B token-pass

### 5.3 Effective Batch Size

- `batch_size × grad_accum × num_gpus`
- 본 논문 130M/370M: 4 × 4 = 16 (130M), 2 × 8 = 16 (370M)

### 5.4 Cosine LR / Warmup

- **Warmup**: 학습 초기 LR을 0 → peak로 선형 증가 (본 논문 2000 step)
- **Cosine decay**: peak LR에서 min_lr로 cosine 곡선으로 감소

### 5.5 AdamW

- 표준 옵티마이저. Adam + weight decay 분리
- 본 논문: β₁=0.9, β₂=0.95, weight_decay=0.1

### 5.6 Gradient Checkpointing (grad_ckpt)

- **정의**: 메모리 절약을 위해 forward 중간값을 버리고 backward 시 재계산
- 메모리 ↓, 시간 ↑ trade-off

### 5.7 Throughput (tokens/sec)

- **방향**: **높을수록 좋음 ↑**
- 본 논문 RTX 5090: 30M = 195K tok/s, 130M = 44K tok/s, 370M = 25K tok/s

### 5.8 Wall-clock time

- **방향**: 짧을수록 좋음 ↓
- 본 논문 130M training: ~3시간

---

## 6. 통계 용어

### 6.1 σ (시그마, 표준편차) — **작을수록 결과 안정 ↓**

- **정의**: 측정값의 분산을 나타내는 표준편차
- **방향**: 작을수록 결과가 시드 간 일관됨 (좋음)
- 본 논문 표기: `0.981 ± 0.036` 형식 (mean ± std)

### 6.2 σ-separation (시그마 분리)

- **정의**: 두 분포 평균의 차이를 표준편차로 나눈 값
- **공식**: `(μ_A - μ_B) / σ`
- **임계 해석**:
  - 1σ ≈ 32% 우연 가능성 → 통계적 유의성 없음
  - 2σ ≈ 5% 우연 가능성 → 일반적 임계점
  - 3σ ≈ 0.3% 우연 가능성 → 학계 통상 임계
  - 5σ ≈ 0.00006% 우연 가능성 → 강한 유의성
  - **13σ ≈ 사실상 0** → 결정적 (본 논문 ternary vs FP parity 결과)

### 6.3 Effect Size

- **정의**: 차이의 절대 크기. 통계적 유의성과 별개
- 본 논문 ternary - FP parity: 0.471 (상대적으로 매우 큼)

### 6.4 p-value

- **정의**: 귀무가설이 옳다는 가정 하에 관측 결과가 나올 확률
- **방향**: 작을수록 유의성 강함
- 본 논문 13σ → p ≪ 10⁻¹⁰

### 6.5 시드 (seed)

- 무작위성 통제용 정수. 동일 시드 → 동일 결과 재현
- 본 논문 parity 핵심 실험: 5개 시드 (0, 1, 2, 3, 4)

---

## 7. 빠른 방향 참조표

| 지표 카테고리 | 좋은 방향 |
|---|---|
| **PPL (Perplexity)** | **낮을수록 좋음 ↓** |
| **Cross-entropy Loss** | **낮을수록 좋음 ↓** |
| **Accuracy** (acc, acc_norm) | 높을수록 좋음 ↑ |
| **Log probability** (log-prob) | **0에 가까울수록 좋음** (덜 음수일수록) ↑ |
| **Parity accuracy** | 높을수록 좋음 ↑ (1에 근접) |
| **표준편차 σ** | 작을수록 좋음 (안정적) ↓ |
| **σ-separation** | **클수록 좋음** (차이가 강함) ↑ |
| **Effect size** | 클수록 좋음 ↑ |
| **p-value** | 작을수록 좋음 ↓ |
| **메모리 사용량** | 작을수록 좋음 ↓ |
| **Throughput (tok/s)** | 높을수록 좋음 ↑ |
| **Wall-clock time** | 짧을수록 좋음 ↓ |
| **비트 수** | trade-off — 낮으면 메모리·HW 좋음, 정확도 ↓ 가능 |

---

## 8. 본 논문의 핵심 결과 재해석 (방향 명시)

### 핵심 발견 1: 양자화 비용은 작다
- Mamba-3 FP PPL **61.86** vs BitMamba-3 ternary PPL **69.40**
- **+12.2% PPL 증가** = "조금 나빠짐" (PPL은 낮을수록 좋으므로)
- 일반적인 PTQ는 +20~50% — 본 논문은 +12%로 **상대적으로 적은 손실**

### 핵심 발견 2: Mamba-3는 Mamba-2보다 우수
- BitMamba-2 130M PPL **113.94** → BitMamba-3 130M PPL **69.40**
- **−39% PPL 감소** = "1.64배 더 좋음" (PPL이 1.64배 작으므로)

### 핵심 발견 3: Ternary는 inductive bias로 작용
- Mamba-3 FP parity peak **0.510** ± 0.003 (random 수준)
- Mamba-3 INT4 parity peak **0.527** ± 0.003 (random 수준)
- Mamba-3 ternary parity peak **0.972** ± 0.047 (학습 성공)
- **+0.45 accuracy 증가** = "결정적으로 우수" (acc는 높을수록 좋음)
- **σ-separation 13σ** = "통계적으로 결정적" (clean separation)

→ 모든 핵심 발견이 일관되게 **BitMamba-3가 BitMamba-2보다 우수, ternary가 FP·INT4보다 일부 task에서 우수**임을 보임.
