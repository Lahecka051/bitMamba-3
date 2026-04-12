# Computer Vision SOTA 전 영역 현황 (2026-03-21)

> 85개 모델, 22개 카테고리 정리

---

## 카테고리별 주요 SOTA

| 카테고리 | 모델 수 | 2025-2026 핵심 모델 |
|:---------|-------:|:------------------|
| Detection (Closed) | 9 | RF-DETR 2XL (60.1 mAP), YOLO26, D-FINE |
| Detection (Open-Vocab) | 3 | Grounding DINO 1.6 Pro, YOLOE |
| Segmentation | 3 | SAM 3 (컨셉 세그멘테이션) |
| Vision Encoder | 5 | DINOv3 (7B), Perception Encoder (2B) |
| VLM Proprietary | 3 | GPT-5.4, Gemini 3.1 Pro, Claude Opus 4.6 |
| VLM Open | 5 | Qwen3.5-VL (397B-A17B) |
| Image Generation | 5 | FLUX.2 (32B) |
| Video Generation | 11 | Sora 2, Veo 3.1, Kling 3.0 |
| Video Understanding | 3 | InternVideo2.5, V-JEPA 2.1 |
| 3D Vision | 3 | 3D Gaussian Splatting, FastGS |
| Backbone | 2 | MambaVision (SSM+Attention) |
| Edge/Efficient | 3 | MobileNetV4, RepViT |
| Pose/Human | 3 | Sapiens (2B, +7.6 mAP) |
| Depth/Geometry | 4 | Depth Anything 3 (ICLR 2026 Oral) |
| Image Restoration | 5 | SUPIR, RestoreVAR |
| Tracking | 3 | SAM 3 Tracker, ByteTrack |
| Document AI/OCR | 4 | DeepSeek-OCR, Florence-2 |
| World Models | 3 | NVIDIA Cosmos |
| Medical Vision | 3 | BiomedCLIP, MedSAM |
| Autonomous Driving | 3 | UniAD (CVPR 2023 Best Paper) |

---

## Detection SOTA 비교 (COCO val2017 mAP@50-95)

| 모델 | mAP | Params | T4 속도 | 비고 |
|:-----|----:|-------:|--------:|:-----|
| RF-DETR-2XL | **60.1** | 126.9M | 17.2ms | ICLR 2026 |
| D-FINE-X (O365) | 59.3 | 62M | 11.5ms | ICLR 2025 Spotlight |
| LW-DETR-X (O365) | 58.3 | 118.0M | 13.0ms | |
| YOLO26x | 57.5 | 55.7M | 11.8ms | E2E, NMS-free |
| Co-DETR (O365+COCO) | 66.0 | ~218M | 비실시간 | 비실시간 최고 |

## Edge Detection 비교

| 모델 | mAP | Params | T4 속도 |
|:-----|----:|-------:|--------:|
| RF-DETR-N | 48.4 | 30.5M | 2.3ms |
| YOLO26n | 40.9 | 2.4M | 1.7ms |
| D-FINE-N | 42.7 | 3.8M | 2.1ms |

---

*2026년 3월 21일 기준 작성*
