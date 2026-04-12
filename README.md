# Computer Vision Lab

> Mamba-3 복소수 상태 공간 모델의 위상 보존 양자화 실험 공간
> 2026년 4월 1일 ~ 진행 중 (Phase 1-2 실패, 방향 재검토 중)

## Research Goal

**Phase-Preserving Quantization (PPQ)**: Mamba-3의 복소수 상태 전이에서 양자화 시 발생하는 위상 오류의 재귀적 누적 문제를 해결하는 극좌표 기반 양자화 프레임워크 개발

## Status: Phase 1-2 Pilot Failed

극좌표 양자화가 직교좌표 대비 유의미한 차이를 보이지 못함. 방향 재검토 중.

## Experiment Log (34 commits)

| Phase | Experiment | Result |
|:------|:-----------|:-------|
| Phase 0 | Mamba-3 SSM+ViT hybrid 구현 | OK |
| Phase 0 | TAQ pilot: 3 가설 검증 | Verified |
| Phase 1 | AGX Orin inference benchmark | Baseline 확보 |
| Phase 1 | RTX 5090 quantization benchmark (FP32~FP4) | Baseline 확보 |
| Phase 1 | TRT native benchmark (FP32/FP16/INT8) | OK |
| Phase 1 | Quantization collapse boundary analysis | 5 experiments |
| Phase 1 | Multi-backbone collapse comparison (ViT-S/DINOv2/ConvNeXt) | Collapse 확인 |
| Phase 1 | TRT all-precision quality eval (mAP/P/R/F1) | 정밀도별 비교 |
| Phase 1 | Real ONNX Q/DQ quantization (INT8/FP8) | OK |
| Phase 1 | NVIDIA Model Optimizer (modelopt 0.42) | 실행 |
| Phase 1 | INT8 fair comparison: 3 calibration x 4 models | 비교 완료 |
| Phase 1 | INT4 AWQ Q/DQ TRT: 4 models | 실행 |
| Phase 1 | Low-bit AWQ+TAQ (INT4/INT3/INT2) | Collapse 심각 |
| Phase 1 | ViT-S GRU benchmark: fair comparison | 비교 완료 |
| Phase 2 | Mixed-precision polar quantization for Mamba-3 SSM | **실패** |

## Key Findings (Failed)

1. **Quantization Collapse**: INT4 이하에서 모든 백본(ViT-S, DINOv2, ConvNeXt)에서 정확도 붕괴 발생
2. **Polar vs Cartesian**: 극좌표 양자화가 직교좌표 대비 유의미한 차이 없음 (Phase 1 Go/No-Go 기준 미충족)
3. **Mixed Precision**: Phase 2 혼합 정밀도 극좌표 양자화에서도 개선 미미

## Hardware

- RTX 5090 32GB (Blackwell)
- Jetson AGX Orin 64GB
- TensorRT 10.x, CUDA 12.x, PyTorch 2.6+

## Files

- `ppq/` - Phase-Preserving Quantization 실험 코드
  - `phase1_pilot.py` / `phase1_pilot_v2.py` - Phase 1 파일럿
  - `phase2_experiments.py` / `phase2_mixed_precision.py` - Phase 2 혼합 정밀도
  - `train_mamba3_*.py` - Mamba-3 모델 학습
  - `results/` - 실험 결과 JSON
- `.gitignore` - 모델 체크포인트, ONNX 제외

## Related Research

- [연구 계획서 (PPQ Framework)](https://github.com/Lahecka051/ComputerVision) - 20주 계획, Phase 1에서 중단
- [국방품질연구회 KCI 논문](https://github.com/Lahecka051/Defense_Quality_Research_Council) - ViT 기반 소형 객체 탐지

## Tech Stack

`Python` `PyTorch` `TensorRT` `ONNX` `NVIDIA ModelOpt` `OpenCV` `CUDA`
