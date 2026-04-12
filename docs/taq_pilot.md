# TAQ Pilot: Temporal-Adaptive Quantization for Video SSM

> 비디오 코덱 I-frame/P-frame 원리를 비디오 이해 모델 양자화에 이식하는 연구
> 2026년 3월 23일 ~ 4월 1일 — **구현 완료, 실용성 부재로 중단**

## Research Question

비디오 코덱의 I-frame/P-frame 시간 적응 원리를 SSM 기반 비디오 이해 모델의 양자화에 이식할 수 있는가?
SSM state delta를 scene change detection 신호로 활용하여 프레임별 적응 양자화(TAQ)를 수행하면 균일 양자화 대비 우위가 있는가?

## Status: Concluded (Not Viable)

### 선행 연구 검토 결과

1. **신경망 비디오 코덱 양자화**는 활발 (NeuroQuant, DCVC-RT, TeNeRV) — 그러나 이것은 "비디오 데이터의 압축"이지 "비디오를 이해하는 모델의 양자화"가 아님
2. **비디오 이해 모델 양자화**는 거의 공백 — QBasicVSR, PMQ-VE는 비디오 향상/매팅 모델이고, VideoMamba 같은 SSM 비디오 이해 모델의 양자화 연구는 검색 범위 내 부재
3. **비디오 코덱의 I/P-frame 원리를 모델 양자화에 이식한 연구**는 발견되지 않음

### 종합 판단: 비디오 시간 적응형 양자화가 ViT 패치 방향보다 명확하게 낫다

| 근거 | 설명 |
|:-----|:-----|
| 선행 연구 부재 | ViT 패치 양자화는 이미 CVPR 2025에 논문 있음, 비디오 이해 모델은 공백 |
| 시간적 중복성 활용 가능 | 비디오 프레임 간 상관 0.9 이상, 패치 간 공간적 상관보다 훨씬 강함 |
| 이론적 프레임워크 | 비디오 코덱의 I/P-frame, rate-distortion, GOP 구조를 직접 차용 가능 |

### 학회 타당성 평가

TAQ의 아이디어("비디오 코덱의 I/P-frame 원리를 비디오 이해 모델의 양자화에 이식, SSM state delta를 scene change detection 신호로 활용")는 **CVPR/ECCV/ICCV 수준의 참신성**을 갖고 있으며, 비디오 코덱이라는 확립된 이론적 프레임워크를 차용하므로 방법론의 정당화가 탄탄하다.

### 실패 원인

| 가설 | 기준 | 결과 | 판정 |
|:-----|:-----|:-----|:-----|
| H1: State delta와 시각 변화 상관 | Spearman rho > 0.5 | 상관 존재하나 0.3~0.5 | Weak |
| H2: 공간/시간 양자화 민감도 차이 | delta > 2% | 차이 있으나 모델 의존적 | Partial |
| H3: TAQ가 균일 양자화 대비 우위 | gain > 0.5% | 미미한 차이 (< 0.5%) | Failed |

**핵심 실패**: Phase 1 파일럿에서 SSM state delta와 프레임 간 시각 변화의 상관(Spearman rho)이 0.5 미만으로, Go/No-Go 기준을 충족하지 못함. 이 수치를 모르는 상태에서 수개월을 투자하는 것은 위험하다고 판단.

## Experiment Pipeline



## Conclusion

- 아이디어의 참신성과 이론적 정당성은 확인됨
- 그러나 Phase 1 파일럿에서 핵심 가설(state delta 상관)이 기준 미달
- 이후 PPQ(Phase-Preserving Quantization for Mamba-3) 연구로 방향 전환

## Hardware
- RTX 5090 32GB (Blackwell, SM 9.0)
- Conda env: taq-vidssm (Python 3.11, PyTorch 2.9+cu128)

## Tech Stack
`Python` `PyTorch` `mamba-ssm` `timm` `TensorRT` `CUDA` `scipy`
