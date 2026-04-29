# 2025-2026 양자화 기법 종합 서베이 (92편)

> LLM, CV, Diffusion, MoE, KV Cache, SSM 전 영역 양자화 논문 목록
> 2025.01 ~ 2026.03 기준

---

## 1. LLM 가중치 양자화 (39편)

### ICLR 2025 (7편)
| 논문 | 핵심 |
|------|------|
| CBQ | 블록 간 상관관계 극저비트 양자화 |
| SpinQuant | 학습된 회전 행렬 이상치 완화 |
| LeanQuant | 손실 오차 그리드 405B PTQ |
| STBLLM | N:M 희소성 + 이진화 |
| SVDQuant | 저랭크 이상치 흡수 4bit Diffusion |
| Effective Interplay | 희소성+양자화 적용 순서 이론 (Spotlight) |
| COAT | FP8 옵티마이저+활성화 압축 |

### ICLR 2026 (15편)
| 논문 | 핵심 |
|------|------|
| PT2-LLM | 삼진(ternary) PTQ |
| CodeQuant | MoE 이상치 클러스터링 |
| Tequila | 데드존 없는 삼진 양자화 |
| SliderQuant | 슬라이더 기반 PTQ 보정 |
| ParoQuant | 쌍별 회전 추론형 LLM |
| AnyBCQ | 이진 코드 다중 정밀도 |
| UniQL | 양자화+저랭크 통합 엣지 |
| SERQ | 중요도 기반 저랭크 오류 보상 |
| LogART | 로그 스케일 PTQ 한계 확장 |
| Bridging FP4 Gap | FP4 이론-실제 격차 해결 |
| BOF4 | 블록 최적 부동소수점 4bit |
| Geometry of GPTQ | GPTQ = Babai 알고리즘 증명 |
| STaMP | 시퀀스 변환 혼합 정밀도 |
| Training Dynamics PTQ | 사전학습 동역학 → 양자화 강건성 |
| Ultra-Low-Bit Reasoning QAT | 추론형 극저비트 QAT |

### NeurIPS/ICML/ACL 2025 (12편)
ParetoQ, DartQuant, Quantization Error Propagation, LittleBit, HBLLM, Q-Palette (NeurIPS) / any4, BlockDialect, GPTAQ, SKIM, FP4 Training (ICML) / Low-Bit Favors Undertrained (ACL)

### arXiv 2025 (5편)
RSQ, Binary W+A PTQ, BASE-Q, DBellQuant, CCQ

---

## 2. KV Cache 양자화 (15편)

| 논문 | 학회 | 핵심 |
|------|------|------|
| TurboQuant | ICLR 2026 | 온라인 VQ, 2-3bit, 보정 불필요 |
| Channel-Aware Mixed-Precision | ICLR 2026 | 채널별 혼합 정밀도 장문맥 |
| Q&C | ICLR 2026 | 양자화+캐시 통합 |
| AQUA-KV | ICLR 2026 | 적응적 2-2.5bit 무손실 |
| KBVQ-MoE | ICLR 2026 | KLT+SVD+VQ MoE 압축 |
| CommVQ | ICML 2025 | 가환 VQ KV 압축 |
| SageAttention2 | ICML 2025 | 이상치 평활화 INT4 어텐션 |
| SageAttention3 | NeurIPS 2025 | FP4 마이크로스케일 어텐션 |
| XQuant | arXiv | 1.4bit 미만 크로스레이어 |
| PM-KVQ | OpenReview | 장문 CoT 혼합 정밀도 |
| Expected Attention | arXiv | 미래 쿼리 추정 KV 압축 |
| VQKV | arXiv | 82.8% 압축 98.6% 성능 |
| RotateKV | arXiv | 이상치 적응 회전 2bit |
| GPU-Accel INT8 KV | arXiv | CUDA 커널 INT8 KV |
| Oaken | ISCA 2025 | 온/오프라인 하이브리드 |

---

## 3. Diffusion 모델 양자화 (8편)

ViDiT-Q, DGQ (ICLR 2025) / Quant-dLLM, PTQ4ARVG, Gradient-Aligned (ICLR 2026) / VETA-DiT (NeurIPS 2025) / Quant Meets dLLMs (arXiv)

---

## 4. 비전/멀티모달 양자화 (5편)

AutoQVLA, Shift-and-Sum, Inlier-Centric (ICLR 2026) / OuroMamba (ICCV 2025) / Point4Bit (NeurIPS 2025)

---

## 5. MoE 양자화 (2편)

Efficient MoE Quant, BBQ (ICLR 2026)

---

## 6. QAT 및 양자화 미세조정 (4편)

QeRL, On-the-Fly LoRA, QWHA, DPQuant (ICLR 2026)

---

## 7. 메모리 효율 어텐션/추론 (6편)

FlashAttention-3, FlashInfer, SpargeAttention (2025) / Mamba-3, PLENA (2026)

---

## 8. SSM 양자화 (1편)

SSDi8: Mamba-2 SSD INT8 (ICLR 2026)

---

## 9. 서베이 (4편)

Low-bit LLM Survey, Low-bit DNN Survey, Comprehensive LLM Quant Eval, Systematic Characterization

---

## 연구 연결점 (PPQ 프로젝트 기준)

| 우선순위 | 논문 | 연결점 |
|:--------:|------|--------|
| 1 | SSDi8 | Mamba SSM INT8 직접 참고 |
| 2 | Effective Interplay | Sparsity+Quant 순서 이론 |
| 3 | OuroMamba | Vision Mamba 양자화 |
| 4 | MBQ | VLM 모달리티별 민감도 |
| 5 | COAT | FP8 훈련 메모리 최적화 |
| 6 | SVDQuant | 이상치 처리 저랭크+양자화 |

---

*2026년 3월 기준. TurboQuant, RotorQuant 등 KV 양자화 최신 동향 추가 반영.*
