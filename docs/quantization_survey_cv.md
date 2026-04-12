# 2025-2026 CV/VLM 적용 가능 양자화 논문 서베이

> Computer Vision 모델 또는 Vision-Language Model에 직접 적용 가능한 양자화 논문 정리
> 2026년 3월 기준, 4개 카테고리 24편

---

## A. VLM 전용 양자화

| # | 논문 | 학회 | 핵심 | 적용 모델 |
|---|------|------|------|----------|
| 1 | MBQ: Modality-Balanced Quantization | CVPR 2025 | 모달리티별 민감도 보정, W3A16 최대 11.6% 향상 | LLaVA, InternVL2, Qwen2-VL |
| 2 | Q-VLM: Post-training Quantization for VLMs | NeurIPS 2024 | 4bit PTQ, 2.78x 메모리 압축 | LLaVA 7B/13B |
| 3 | AutoQVLA | ICLR 2026 | 채널별 중요도 비균등 양자화 | VLA 모델 |

## B. Vision Transformer / CV 모델 전용

| # | 논문 | 학회 | 핵심 | 적용 모델 |
|---|------|------|------|----------|
| 4 | FIMA-Q: Fisher Information Matrix PTQ | CVPR 2025 | Fisher 정보 행렬 기반 W4A4 ViT PTQ | DeiT, Swin, ViT |
| 5 | Inlier-Centric PTQ for Object Detection | ICLR 2026 | 인라이어 중심 검출 모델 PTQ | DETR, RT-DETR |
| 6 | Point4Bit: 4-bit Point Cloud Detection | NeurIPS 2025 | 3D 포인트 클라우드 4bit PTQ | PointPillars, CenterPoint |
| 7 | OuroMamba: Data-Free Vision Mamba Quant | ICCV 2025 | Vision Mamba 데이터 없는 양자화 | Vim, VMamba |
| 8 | Scheduling Weight Transitions for QAT | ICCV 2025 | QAT 수렴 안정성 향상 | ResNet, ViT |
| 9 | Task-Specific Zero-shot QAT for Detection | ICCV 2025 | 합성 데이터 기반 검출 QAT | YOLO, DETR |
| 10 | PTQ for Video Matting | ICLR 2026 | 프레임 간 일관성 유지 PTQ | RVM 등 |
| 11 | QSCA: Monocular Depth Estimation Quant | NeurIPS 2025 | 자기 보상 보조 모듈 깊이 추정 양자화 | Depth Anything, MiDaS |
| 12 | QBasicVSR: Video SR Quantization | NeurIPS 2025 | 시간축 적응형 비디오 SR 양자화 | BasicVSR++ |
| 13 | PMQ-VE: Progressive Video Enhancement | NeurIPS 2025 | 점진적 다중 프레임 양자화 | 비디오 디노이징 |

## C. Diffusion/생성 모델

| # | 논문 | 학회 | 핵심 | 적용 모델 |
|---|------|------|------|----------|
| 14 | SVDQuant: 4-Bit Diffusion | ICLR 2025 | SVD 저랭크 이상치 흡수, 3.6x 메모리 절감 | FLUX.1, SDXL |
| 15 | ViDiT-Q: Diffusion Transformer Quant | ICLR 2025 | 타임스텝별 동적 양자화 | DiT, PixArt, Open-Sora |
| 16 | DGQ: Distribution-Aware Group Quant | ICLR 2025 | 분포 인식 그룹 양자화 | Stable Diffusion |
| 17 | VETA-DiT: 4-bit DiT | NeurIPS 2025 | 분산 균등화 + 시간 적응형 | DiT, Latte |
| 18 | Quant-dLLM: Diffusion LLM PTQ | ICLR 2026 | 확산 LLM 최초 PTQ 평가 | LLaDA-8B, Dream-7B |
| 19 | Gradient-Aligned Calibration | ICLR 2026 | 그래디언트 정렬 보정 | Stable Diffusion, DiT |
| 20 | Shift-and-Sum for Visual AR | ICLR 2026 | 곱셈 없는 시프트-합 양자화 | VAR, LlamaGen |
| 21 | PTQ4ARVG: AR Visual Generation PTQ | ICLR 2026 | 토큰 누적 오류 이중 보정 | VQGAN, LlamaGen |

## D. 범용이지만 CV/VLM 적용 가능

| # | 논문 | 학회 | 핵심 | CV/VLM 적용 근거 |
|---|------|------|------|-----------------|
| 22 | Effective Interplay (Sparsity+Quant) | ICLR 2025 Spotlight | 2:4 sparsity + INT4/INT8 적용 순서 이론 | ViT, 검출 모델 Tensor Core 활용 |
| 23 | SSDi8: Mamba-2 INT8 | ICLR 2026 | SSD 모델 8bit 양자화 | Vision Mamba SSM 레이어 |
| 24 | COAT: FP8 Training | ICLR 2025 | 옵티마이저+활성화 FP8 압축 | ViT/VLM 대규모 훈련 |

---

## 최신 추가 (2026 Q1)

| 논문 | 학회 | 핵심 |
|------|------|------|
| TurboQuant: Online VQ for KV Cache | ICLR 2026 | 데이터 비의존 온라인 벡터 양자화, 2-3bit KV 캐시 |
| RotorQuant: Rotation-based KV Quant | 2026 | 회전 기반 KV 캐시 양자화 최적화 |
| AQUA-KV: Adaptive KV Quantization | ICLR 2026 | 레이어 간 KV 의존성 활용, 2-2.5bit 무손실 |
| XQuant: Cross-Layer KV Compression | 2025 | 1.4bit 미만 KV 캐시 양자화 |

---

*2026년 3월 기준 작성*
