# TAQ Pilot: Temporal-Adaptive Quantization for Video SSM

> SSM state delta 기반 시간적 적응 양자화 검증 실험
> 2026년 4월 — **구현 완료, 실용성 부재로 중단**

## Status: Concluded (Not Viable)

구현 자체는 성공했으나, 다음 이유로 실용성이 없다고 결론:
- 기존 연구(PTQ4VM, OuroMamba)에서 유사한 접근이 이미 존재
- TAQ의 정확도 이득이 균일 양자화 대비 미미 (< 0.5%)
- 추가 연산 오버헤드(state delta 계산) 대비 이점 부족

## Experiment Pipeline

```
run_all.sh (전체 무인 자동화)
  ├── step0_setup.sh    # conda env 생성 (taq-vidssm)
  ├── step1_data.py     # UCF-101 다운로드 (fallback: 합성 데이터)
  ├── step2_model.py    # VideoMamba 또는 ViT-S+Mamba/GRU 구성
  ├── step3_correlation # 가설1: SSM state delta <-> 시각 변화 상관
  ├── step4_sensitivity # 가설2: 공간 vs 시간 양자화 민감도
  ├── step5_taq_proto   # 가설3: TAQ vs 균일 양자화 비교
  └── step6_report      # 종합 리포트 자동 생성
```

## Hypotheses & Results

| 가설 | 기준 | 결과 | 판정 |
|:-----|:-----|:-----|:-----|
| H1: State delta와 시각 변화 상관 | rho > 0.5 | 상관 존재하나 약함 | Weak |
| H2: 공간/시간 양자화 민감도 차이 | delta > 2% | 차이 있으나 모델 의존적 | Partial |
| H3: TAQ가 균일 양자화 대비 우위 | gain > 0.5% | 미미한 차이 | Failed |

## Conclusion

- 시간적 적응 양자화(TAQ) 개념은 이론적으로 타당하나 실험적 이점이 불충분
- 기존 Vision Mamba PTQ 기법(PTQ4VM, OuroMamba)이 더 효과적
- 이후 PPQ (Phase-Preserving Quantization for Mamba-3) 연구로 방향 전환

## Hardware
- RTX 5090 32GB (Blackwell, SM 9.0)
- Conda env: `taq-vidssm` (Python 3.11, PyTorch 2.9+cu128)

## Tech Stack
`Python` `PyTorch` `mamba-ssm` `timm` `TensorRT` `CUDA`
