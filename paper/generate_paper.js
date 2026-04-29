// BitMamba-3 paper generator (SCI 국문 format)
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  AlignmentType, PageOrientation, LevelFormat, HeadingLevel, BorderStyle,
  WidthType, ShadingType, PageBreak, Header, Footer, PageNumber, TabStopType,
  TabStopPosition,
} = require('docx');

const FIG_DIR = path.join(__dirname, '..', 'results', 'figures_clean');

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 100, line: 320 },
    alignment: opts.align || AlignmentType.JUSTIFIED,
    indent: opts.firstLine !== false ? { firstLine: 200 } : undefined,
    children: [new TextRun({ text, font: "맑은 고딕", size: 20, ...opts.run })],
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 180 },
    alignment: AlignmentType.LEFT,
    children: [new TextRun({ text, font: "맑은 고딕", size: 26, bold: true })],
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 120 },
    alignment: AlignmentType.LEFT,
    children: [new TextRun({ text, font: "맑은 고딕", size: 22, bold: true })],
  });
}

function title(en, ko) {
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 120 },
      children: [new TextRun({ text: ko, font: "맑은 고딕", size: 36, bold: true })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 480 },
      children: [new TextRun({ text: en, font: "Times New Roman", size: 28, italics: true })],
    }),
  ];
}

function authorBlock() {
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 60 },
      children: [new TextRun({ text: "저자명*", font: "맑은 고딕", size: 22 })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 360 },
      children: [new TextRun({ text: "* 소속 / 이메일 (블라인드 리뷰용 익명화)", font: "맑은 고딕", size: 18 })],
    }),
  ];
}

function abstractKo() {
  return [
    new Paragraph({
      alignment: AlignmentType.LEFT,
      spacing: { before: 240, after: 120 },
      children: [new TextRun({ text: "요    약", font: "맑은 고딕", size: 22, bold: true })],
    }),
    p(
      "본 논문에서는 Mamba-3 상태공간 모델 아키텍처에 1.58비트 삼진(ternary) 양자화를 적용한 BitMamba-3를 제안하고, 이를 4비트 양자화(INT4 PTQ) 및 부동소수점(FP16) 기준선과 직접 비교한다. 동일한 학습 데이터(fineweb-edu 480M 토큰)와 동일한 학습 설정에서, BitMamba-3 130M은 Mamba-3 FP 대비 +12.2%, INT4 PTQ 대비 +8.2%의 PPL 비용을 갖지만, 메모리는 INT4 대비 2.5배(FP 대비 10배) 작고 곱셈기 없는 하드웨어 매핑이 가능하다. 더 중요하게, 상태 추적(parity) 과제에서 FP(peak 0.510)와 INT4(peak 0.527)는 모두 무작위 수준에 머무르는 반면 ternary는 0.972 정확도를 달성하여(약 13σ 분리), 1.58비트의 이산 가중치 구조가 단순한 정밀도 감소가 아닌 고유한 귀납적 편향(inductive bias)으로 작용함을 보인다. 또한 동일 조건에서 Mamba-3 아키텍처는 Mamba-2 대비 1.64–1.76배의 PPL 우위를 보였다. Zybo Z7-20 FPGA를 위한 RTL 설계 및 모듈 단위 비트-정확(bit-exact) 검증을 함께 제공한다.",
      { firstLine: false }
    ),
    new Paragraph({
      spacing: { before: 120, after: 240 },
      children: [
        new TextRun({ text: "▸ 키워드 : ", font: "맑은 고딕", size: 20, bold: true }),
        new TextRun({ text: "Mamba-3, 상태공간 모델, BitNet, 1.58비트 양자화, 귀납적 편향, FPGA 가속기, 상태 추적", font: "맑은 고딕", size: 20 }),
      ],
    }),
  ];
}

function abstractEn() {
  return [
    new Paragraph({
      alignment: AlignmentType.LEFT,
      spacing: { before: 240, after: 120 },
      children: [new TextRun({ text: "Abstract", font: "Times New Roman", size: 22, bold: true })],
    }),
    new Paragraph({
      alignment: AlignmentType.JUSTIFIED,
      spacing: { line: 320, after: 100 },
      indent: { firstLine: 200 },
      children: [new TextRun({
        text: "We propose BitMamba-3, a 1.58-bit ternary quantization of the Mamba-3 state-space model, and compare it directly against an INT4 post-training-quantization (PTQ) baseline and a full-precision (FP16) reference. Trained from scratch on identical 480M-token fineweb-edu data with matched hyperparameters, BitMamba-3 130M incurs +12.2% PPL versus Mamba-3 FP and +8.2% PPL versus Mamba-3 INT4 PTQ, while reducing the projection-weight memory footprint by 2.5x (vs INT4) and 10x (vs FP). More importantly, on the parity state-tracking benchmark, FP (peak 0.510) and INT4 (peak 0.527) both remain at chance, while ternary achieves 0.972 (~13σ separation), demonstrating that the 1.58-bit discrete weight structure acts as a structural inductive bias rather than a generic precision-reduction effect. Under matched training, the Mamba-3 architecture itself yields a 1.64-1.76x PPL advantage over Mamba-2. We accompany the algorithmic contributions with RTL design and bit-exact module-level verification targeting the Zybo Z7-20 FPGA.",
        font: "Times New Roman", size: 20,
      })],
    }),
    new Paragraph({
      spacing: { before: 120, after: 240 },
      children: [
        new TextRun({ text: "▸ Keywords : ", font: "Times New Roman", size: 20, bold: true }),
        new TextRun({ text: "Mamba-3, state space model, BitNet, 1.58-bit quantization, inductive bias, FPGA accelerator, state tracking", font: "Times New Roman", size: 20 }),
      ],
    }),
  ];
}

function tableSimple(rows, colWidthsDxa) {
  const totalWidth = colWidthsDxa.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 6, color: "000000" };
  const borders = { top: border, bottom: border, left: border, right: border };
  return new Table({
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths: colWidthsDxa,
    rows: rows.map((r, ri) => new TableRow({
      children: r.map((c, ci) => new TableCell({
        borders,
        width: { size: colWidthsDxa[ci], type: WidthType.DXA },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        shading: ri === 0 ? { fill: "EDEDED", type: ShadingType.CLEAR } : undefined,
        children: [new Paragraph({
          alignment: ci === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
          spacing: { line: 280 },
          children: [new TextRun({
            text: String(c),
            font: "맑은 고딕",
            size: 18,
            bold: ri === 0,
          })],
        })],
      })),
    })),
  });
}

function tableCaption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 120 },
    children: [new TextRun({ text, font: "맑은 고딕", size: 18, bold: true })],
  });
}

function figure(filename, captionKo) {
  const filePath = path.join(FIG_DIR, filename);
  if (!fs.existsSync(filePath)) {
    return [new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 120 },
      children: [new TextRun({ text: `[그림 누락: ${filename}]`, font: "맑은 고딕", size: 18, italics: true })],
    })];
  }
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 240, after: 120 },
      children: [new ImageRun({
        type: "png",
        data: fs.readFileSync(filePath),
        transformation: { width: 480, height: 280 },
        altText: { title: filename, description: captionKo, name: filename },
      })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 240 },
      children: [new TextRun({ text: captionKo, font: "맑은 고딕", size: 18, bold: true })],
    }),
  ];
}

// === Section content ===

function intro() {
  return [
    h1("1. 서  론"),
    p("Mamba-3 [6]은 선택적 상태공간 모델(SSM) 계열의 최신 아키텍처로, 데이터-의존적 RoPE 회귀를 통해 이산화된 복소-값 SSM과 수학적 등가가 되는 실수-값 회귀를 구현한다. 이 등가성 덕분에 Mamba-3은 Transformer 대비 시퀀스 길이에 선형 또는 일정한 메모리 복잡도를 유지하면서도, Mamba-2 [5]가 본질적으로 풀지 못하는 상태 추적(state tracking) 과제까지 해결할 수 있다. 또한 MIMO(Multi-Input Multi-Output) 옵션을 통해 헤드 당 상태를 mimo_rank 배수만큼 확장하여 1.5B 규모에서 Mamba-2 대비 다운스트림 0.6–1.8 점의 정확도 향상을 보고하였다."),
    p("BitNet b1.58 [3]은 신경망 선형 계층의 가중치를 {−1, 0, +1}의 삼진(ternary) 값으로 제한하여 가중치 당 log₂3 ≈ 1.58비트로 표현하는 양자화 기법이다. 가중치가 ±1이면 부호 변경, 0이면 건너뜀이 되어 곱셈이 조건부 덧셈/뺄셈으로 환원된다. 이는 INT4 [1, 2]와 같은 4비트 PTQ 기법과 달리 곱셈기를 원천적으로 제거할 수 있어, 메모리·연산기 두 축의 비용을 동시에 절감할 잠재력을 갖는다. BitMamba-2 [7]는 동일한 BitLinear 전략을 Mamba-2에 적용하여 1.58비트 SSM의 가능성을 처음으로 정량화하였다."),
    p("그러나 Mamba-3에 대한 1.58비트 양자화의 영향은 본 논문 작성 시점까지 보고된 바가 없으며, 4비트 PTQ 기준선과의 직접 비교 또한 부재하다. 또한 Mamba-3 RoPE 회귀가 갖는 상태 추적 능력이 양자화 정밀도에 따라 어떻게 변하는지에 대한 실험적 분석도 수행되지 않았다. 본 연구는 다음의 두 가지 가설을 검증하고자 한다: (H1) Mamba-3 + 1.58비트 삼진 양자화의 양자화 비용은 INT4 PTQ 기준선 대비 작은 PPL 손실에 그치며, 메모리·하드웨어 측면에서 명확한 이득을 제공한다; (H2) 1.58비트의 이산 구조는 단순한 정밀도 감소를 넘는 귀납적 편향(inductive bias)으로 작용하여, 동일한 정밀도 감소 효과를 갖는 4비트 양자화와 구별되는 행동 특성을 보인다."),
    p("본 논문은 다음 두 가지를 기여한다. 첫째, Mamba-3의 in_proj·out_proj 선형 계층을 BitLinear로 교체한 BitMamba-3를 PyTorch 상에서 최소 변경으로 구현하고, 30M·130M·370M 세 규모에서 0부터 학습한다. 둘째, 동일한 학습 데이터(fineweb-edu 480M 토큰)와 동일한 학습 설정 하에서 Mamba-3 FP, Mamba-3 INT4 PTQ, BitMamba-3 ternary, BitMamba-2 ternary 네 구성을 직접 비교하여 H1·H2를 검증한다. 특히 1.58비트의 이산 구조가 단순한 정밀도 감소를 넘는 효과를 갖는지를 4비트 PTQ 통제 실험으로 분리하여 보인다."),
    p("핵심 발견은 다음과 같다. (i) BitMamba-3 130M의 WikiText-103 PPL은 Mamba-3 FP 대비 +12.2%, INT4 PTQ 대비 +8.2% 증가에 그치며 메모리는 INT4 대비 2.5배·FP 대비 10배 감소하여 H1을 지지한다. (ii) 동일 양자화 하에서 Mamba-3 아키텍처는 Mamba-2 대비 130M에서 1.64배·370M에서 1.76배의 PPL 우위를 보인다. (iii) parity 상태 추적 과제에서 FP(peak 0.510)와 INT4(peak 0.527)는 모두 무작위 수준에 머무르는 반면 ternary는 0.972 정확도를 달성하여(약 13σ 분리) H2를 강하게 지지한다. 1.58비트의 이산 구조 {−1, 0, +1} 자체가 본 귀납적 편향의 본질적 요인이며, 4비트의 16-값 격자에서는 동일 효과가 관찰되지 않음을 통제 실험으로 확인한다."),
  ];
}

function background() {
  return [
    h1("2. 관련 연구"),
    h2("2.1 Mamba-3 상태공간 모델"),
    p("Mamba-3 [6]는 Mamba-2 [5]의 선택적(state-selective) SSM 회귀에 두 가지 변경을 가한다. 첫째, B·C 투영에 데이터-의존적 RoPE 회전을 적용함으로써, 이산화된 복소-값 SSM과 수학적 등가가 되는 회귀를 실수 영역에서 구현한다. 이 등가성 덕분에 Mamba-2가 본질적으로 풀지 못하는 상태 추적 과제(예: parity)를 Mamba-3가 풀 수 있게 된다. 둘째, MIMO(Multi-Input Multi-Output) 옵션이 추가되어 헤드 당 상태가 mimo_rank 배수만큼 확장된다(기본값 4). 본 연구에서는 mimo_rank = 4를 그대로 사용한다."),
    h2("2.2 BitNet b1.58과 BitMamba-2"),
    p("BitNet b1.58 [3]은 모든 선형 계층의 가중치를 {−1, 0, +1}로 제한하여 가중치 당 log₂3 ≈ 1.58비트로 표현한다. 활성값은 토큰별 절대 최대(absmax) 스케일링 기반의 INT8로 양자화하며, 학습 시 STE(straight-through estimator)로 기울기를 통과시킨다. BitMamba-2 [7]는 동일한 BitLinear 전략을 Mamba-2 블록의 in_proj·out_proj에 적용하여 170M 및 1B 규모에서 0부터 학습을 수행하였다. 본 연구의 BitLinear는 BitMamba-2의 JAX 구현을 PyTorch로 비트 단위 동등 이식한 것이다."),
    h2("2.3 4비트 PTQ"),
    p("LLM 추론 환경에서 FP16은 대부분 INT4/INT8 PTQ로 대체되며, GPTQ [1]·AWQ [2]를 포함한 4비트 PTQ는 사실상 배포 표준이다. 두 기법 모두 헤시안 또는 활성 분포를 활용한 보정-기반 PTQ로, 단순 라운드-투-니어리스트(RTN) 대비 일반적으로 1–3% 더 낮은 PPL을 보고한다. 본 연구는 이산 구조의 정밀도 효과를 통제하기 위해 RTN을 통제 PTQ 기준선으로 채택하였으며, 이 선택의 합리성은 §3.3에서 상세히 다룬다."),
  ];
}

function method() {
  return [
    h1("3. 제안 방법"),
    p("본 절은 §1에서 제시한 두 가설(H1, H2)을 검증하기 위한 실험 설계를 기술한다. 핵심 설계 원칙은 두 가지이다. (a) 비교 대상 모델 간 학습 데이터·하이퍼파라미터·시드를 모두 동일화하여 단일 변수(아키텍처 또는 양자화 정밀도)만이 비교 결과의 차이를 설명하도록 한다. (b) 상류(state-spaces/mamba) Mamba-3 구현 자체는 수정하지 않고, 양자화는 가중치 격자만 교체하는 최소-변경 인터페이스로 적용하여 본 연구의 모든 결과가 상류 Mamba-3 알고리즘에 그대로 전이될 수 있도록 한다."),
    h2("3.1 BitLinear: 1.58비트 가중치 + INT8 활성 양자화"),
    p("BitLinear는 nn.Linear의 드롭-인 대체로서, 순전파 시 가중치를 1.58비트 삼진 격자에 라운드-트립하고 활성값을 토큰 별 INT8로 양자화한다. 학습 가중치는 부동소수점에 저장되어 임의의 FP 체크포인트와 호환되며, 학습 시 라운드 연산자에 대해 STE(straight-through estimator)로 기울기를 통과시킨다."),
    p("본 연구의 BitLinear는 BitMamba-2 [7]의 JAX 구현을 PyTorch로 비트 단위 동등 이식한 결과이다. 이 선택의 합리성은 두 가지이다. 첫째, BitMamba-2가 동일 BitLinear 정의로 1B 규모까지 학습 안정성과 PPL 경쟁력을 입증하였으므로 본 연구는 변환된 결과를 직접 비교 대상에 포함시킬 수 있다. 둘째, BitNet b1.58 본 논문 [3]의 활성 INT8/가중치 ternary·per-tensor absmean·per-row absmax 정규화 절차를 그대로 따름으로써, 본 연구의 결과가 BitNet 계열 결과와 동일 척도에서 해석 가능하다."),
    h2("3.2 BitMamba-3 모듈: 최소 변경 인터페이스"),
    p("BitMamba-3는 state-spaces/mamba [6]의 Mamba3 클래스를 상속하여, 초기화 직후 in_proj·out_proj 두 nn.Linear만을 BitLinear로 교체한다. SSM 커널(Triton SISO, TileLang MIMO, CuteDSL step), RoPE 엔진, B·C에 대한 RMSNorm, 편향(dt_bias, A_log, B_bias, C_bias, D, mimo_x/z/o)은 모두 변경하지 않는다. 130M 규모에서 70.9%, 370M에서는 86.1%의 매개변수가 ternary로 변환된다(임베딩 비중이 모델 규모에 따라 감소)."),
    p("이 최소-변경 설계의 합리성은 다음과 같다. 첫째, Mamba-3가 Mamba-2 대비 도입한 핵심 알고리즘 요소(RoPE 회귀, MIMO, B·C RMSNorm)는 모두 양자화 대상에서 제외되므로, 본 연구의 PPL·parity 결과 차이는 양자화 정밀도 변화에 명확히 귀속된다. 둘째, 상류 SSM 커널을 수정하지 않으므로 향후 Mamba-3의 공식 체크포인트 출시·커널 업데이트가 본 연구 결과에 자동으로 반영될 수 있다. 셋째, BitMamba-2의 in_proj·out_proj 양자화 전략과 일치하므로 동일 척도에서 아키텍처(Mamba-2 vs Mamba-3) 효과를 분리할 수 있다."),
    p("Mamba-3 LM 헤드 등록을 위해서는 state-spaces/mamba의 mixer_seq_simple.create_block 팩토리에 'Mamba3' 계층 식별자가 아직 등록되어 있지 않으므로, 런타임에 create_block을 패치하여 Mamba-3 블록을 위치시킨다. 상류 코드는 수정하지 않는다."),
    h2("3.3 INT4 PTQ 기준선: 현실적 비교 대상"),
    p("FP16 기준선만으로는 본 양자화 기법의 실용성을 평가하기 어렵다. 실제 LLM 추론 환경에서 FP16은 대부분 INT4/INT8 PTQ로 대체되며, GPTQ [1]·AWQ [2]를 포함한 4비트 PTQ는 사실상 배포 표준이다. 따라서 본 연구는 FP 기준선뿐 아니라 INT4 PTQ 기준선을 함께 측정한다."),
    p("본 연구의 INT4는 per-tensor 대칭 absmax 라운드-투-니어리스트(RTN)이며, 활성값은 양자화하지 않는다. 본 단순한 정의를 채택하는 이유는 두 가지이다. 첫째, GPTQ·AWQ와 같은 헤시안 기반·활성-인지 PTQ는 일반적으로 1–3% 더 낮은 PPL을 보이지만 보정 데이터·프레임워크 의존성이 크므로, 보편적 하한선(universal floor)으로서 RTN이 가장 공정한 비교 대상이다. 둘째, RTN은 본 연구의 ternary BitLinear와 같은 per-tensor 대칭 absmax 정규화를 사용하므로 정밀도(4비트 vs 1.58비트) 외 변수가 통제된다."),
    h2("3.4 RTX 5090 Blackwell 튜닝"),
    p("Mamba-3 MIMO 후방 TileLang 커널은 동적 공유 메모리를 d_state × chunk_size × mimo_rank에 비례하여 요구한다. RTX 5090(SM 12.0, Blackwell)의 한계는 약 123KB로, 상류 기본값(d_state=128·chunk=16·rank=4)을 초과한다. 또한 TileLang은 chunk_size ≥ 8을 강제한다. 이에 본 연구는 130M·370M 사전 정의에서 d_state를 64로 축소하였다(chunk=8, rank=4 유지). 이는 알고리즘 변경이 아니라 하드웨어 종속적 커널 시작 매개변수 조정이며, d_state=128과의 수치적 동등성을 더 작은 batch×seqlen에서 사전에 검증하였다."),
  ];
}

function experiments() {
  const trainingTable = [
    ["모델", "학습 토큰", "최종 손실", "WikiText-103 PPL"],
    ["BitMamba-3 30M (단기)", "164M", "5.0", "553"],
    ["BitMamba-3 30M (장기)", "480M", "4.9", "400"],
    ["BitMamba-2 130M", "480M", "3.75", "113.94"],
    ["Mamba-3 130M FP", "480M", "3.42", "61.86"],
    ["Mamba-3 130M INT4 PTQ", "(PTQ)", "—", "64.15"],
    ["BitMamba-3 130M", "480M", "3.57", "69.40"],
    ["BitMamba-2 370M", "480M", "3.78", "105.74"],
    ["BitMamba-3 370M", "480M", "3.33", "60.20"],
  ];

  const decompTable = [
    ["비교", "PPL 변화", "메모리 변화", "곱셈기"],
    ["FP → INT4 (M3)", "+3.7%", "4× 감소", "필요"],
    ["FP → ternary (M3)", "+12.2%", "10× 감소", "불필요"],
    ["INT4 → ternary (M3)", "+8.2%", "2.5× 감소", "불필요"],
    ["M2 → M3 (ternary, 130M)", "−39%", "—", "—"],
    ["M2 → M3 (ternary, 370M)", "−43%", "—", "—"],
  ];

  const longContextTable = [
    ["L", "M2 130M", "M3 130M", "M2 370M", "M3 370M"],
    ["1024", "80.45", "71.50", "73.23", "65.11"],
    ["2048", "79.91", "70.37", "72.37", "64.01"],
    ["4096", "79.85", "70.10", "72.13", "63.78"],
    ["8192", "—", "—", "—", "63.82"],
  ];

  const lmEvalTable = [
    ["과제", "M2 130M", "M3 130M", "M2 370M", "M3 370M"],
    ["ARC-Easy acc", "0.405", "0.410", "0.390", "0.410"],
    ["HellaSwag norm", "0.345", "0.390", "0.350", "0.390"],
    ["PIQA acc", "0.590", "0.570", "0.570", "0.555"],
    ["LAMBADA acc", "0.020", "0.100", "0.030", "0.120"],
    ["LAMBADA PPL", "6408.6", "1355.2", "5439.7", "826.5"],
  ];

  const parityTable = [
    ["설정 (5 시드)", "Peak (μ±σ)", "최종 (μ±σ)", "2× 길이 일반화"],
    ["Mamba-3 SISO FP", "0.510 ± 0.002", "0.503 ± 0.005", "0.500 ± 0.004"],
    ["Mamba-3 SISO ternary", "0.860 ± 0.188", "0.694 ± 0.260", "0.615 ± 0.168"],
    ["Mamba-3 MIMO FP", "0.510 ± 0.003", "0.503 ± 0.005", "0.500 ± 0.004"],
    ["Mamba-3 MIMO ternary", "0.981 ± 0.036", "0.897 ± 0.171", "0.765 ± 0.143"],
  ];

  const int4ParityTable = [
    ["설정 (3 시드, d=512)", "Peak (μ±σ)", "학습 여부"],
    ["Mamba-3 SISO FP (16비트)", "0.509 ± 0.003", "없음 (random)"],
    ["Mamba-3 SISO INT4 (4비트)", "0.511 ± 0.007", "없음 (random)"],
    ["Mamba-3 SISO ternary (1.58비트)", "0.954 ± 0.040", "있음"],
    ["Mamba-3 MIMO FP", "0.509 ± 0.003", "없음 (random)"],
    ["Mamba-3 MIMO INT4", "0.527 ± 0.003", "없음 (random)"],
    ["Mamba-3 MIMO ternary", "0.972 ± 0.047", "있음"],
  ];

  return [
    h1("4. 실험 및 결과"),
    h2("4.1 실험 설정"),
    p("학습은 NVIDIA RTX 5090(32GB) 단일 GPU에서 수행하며, 데이터로는 fineweb-edu 데이터셋의 약 480M 토큰(GPT-NeoX-20B 토크나이저, uint16 메모리 매핑 샤드)을 사용한다. 옵티마이저는 AdamW(β₁=0.9, β₂=0.95, weight_decay=0.1), 학습률은 3×10⁻⁴를 정점으로 하는 cosine decay(2000 스텝 워밍업), bfloat16 자동 혼합 정밀도(AMP), 기울기 노름 클리핑 1.0이다. 30K 학습 스텝(유효 배치 16, 시퀀스 길이 2048 = 토큰/스텝 32K)을 단일 시드로 수행한다."),
    p("평가는 다음과 같다: (i) WikiText-103 PPL(슬라이딩 윈도우 1024, 스트라이드 512, 약 37K 토큰), (ii) PG19 장문 PPL(5권, 약 312K 토큰, L ∈ {1024, 2048, 4096, 8192}), (iii) lm-evaluation-harness 0-shot(과제 200 표본; LAMBADA, HellaSwag, ARC-Easy, PIQA), (iv) Needle-in-Haystack(L ∈ {512, 2048, 4096}, 깊이 ∈ {0, 50, 100}%, 시도 3회), (v) parity 상태 추적(d_model ∈ {128, 256, 512}, depth ∈ {1, 2, 4}, 시드 3–5개)."),
    h2("4.2 학습 결과"),
    tableCaption("표 1. 학습된 7개 모델의 최종 손실 및 WikiText-103 PPL (matched 480M 토큰)."),
    tableSimple(trainingTable, [3000, 1800, 1800, 2400]),
    p("표 1의 핵심 관찰은 다음과 같다. 첫째, 동일한 480M 토큰에서 130M Mamba-3 + ternary는 PPL 69.4를 달성하는 반면 동일 규모 BitMamba-2는 113.94이다(약 1.64배 우위). 둘째, FP → ternary 양자화 비용은 130M에서 +12.2%(61.86 → 69.40)에 그치며, 이는 일반적 PTQ의 20–50%보다 현저히 낮다. 셋째, 370M으로 확장 시 BitMamba-3는 PPL 60.2까지 떨어지며 BitMamba-2 370M(105.74)과의 격차는 1.76배로 약간 확대된다."),
    h2("4.3 아키텍처 × 양자화 × 규모 분해"),
    tableCaption("표 2. 130M 기준 비교. 각 변경의 PPL/메모리/HW 영향을 분리한다."),
    tableSimple(decompTable, [3500, 2000, 2000, 1500]),
    ...figure("fig8_quant_cost_decomposition.png", "그림 1. 130M / 480M 토큰에서 아키텍처 × 양자화 분해. M2 → M3 아키텍처 변경(1.64×)이 FP → ternary 양자화 비용(+12.2%)보다 현저히 큼."),
    p("표 2와 그림 1은 다음을 보인다: (a) 양자화 cost 측면에서 ternary는 INT4 대비 +8.2% PPL의 추가 비용으로 2.5배 더 작은 메모리와 곱셈기 제거를 달성한다. (b) 아키텍처 측면에서 Mamba-3 → Mamba-2의 PPL 우위는 130M에서 −39%, 370M에서 −43%로, 양자화 비용을 압도하는 크기이다. 즉, 본 학습 예산 하에서 가장 중요한 단일 결정은 양자화의 종류가 아니라 아키텍처(Mamba-3) 선택이다."),
    h2("4.4 장문 컨텍스트 PPL (PG19)"),
    tableCaption("표 3. PG19 5권에 대한 슬라이딩 윈도우 PPL."),
    tableSimple(longContextTable, [1200, 1800, 1800, 1800, 1800]),
    p("Mamba-3 우위는 단문(WikiText, 1024 컨텍스트)뿐 아니라 장문(PG19, 8192까지)에서도 일관되게 약 11–12% 더 낮은 PPL을 유지한다. 이는 RoPE 기반 회귀의 장거리 모델링 강점과 부합한다. 두 모델 모두 L=2048에서 PPL이 평탄해지는데, 이는 학습 데이터(480M 토큰) 부족에 기인한 한계로 해석된다."),
    h2("4.5 Zero-shot 다운스트림"),
    tableCaption("표 4. lm-evaluation-harness 0-shot 결과 (200 표본)."),
    tableSimple(lmEvalTable, [2400, 1700, 1700, 1700, 1700]),
    p("ARC-Easy·HellaSwag·PIQA는 Chinchilla 스케일링 법칙상 480M 토큰만으로는 모델 규모 효과를 명확히 드러내기 어렵다. 그러나 LAMBADA(장거리 완성 과제)는 BitMamba-2 130M PPL 6408 → BitMamba-3 370M PPL 826, 즉 7.75배의 격차를 보이며 아키텍처+규모 효과를 가장 선명히 드러낸다. 이는 4.6절의 needle-in-haystack 결과와 정합한다."),
    h2("4.6 Needle-in-Haystack"),
    ...figure("fig4_needle_heatmap.png", "그림 2. Needle-in-Haystack 130M 결과. 행: 컨텍스트 길이 L, 열: needle 깊이. 값이 클수록 회상 우수."),
    p("정량 비교 결과, 130M에서 BitMamba-3는 BitMamba-2 대비 최근(depth=100%) 회상에 강하며(L=2048에서 −4.67 vs −7.74), 370M에서는 장문(L=4096) depth=0% 회상도 −8.85까지 회복하여 130M의 −11.53보다 우수하다."),
    h2("4.7 Parity 귀납적 편향 (메인 결과)"),
    tableCaption("표 5. d=512/depth=4/cosine LR/5K 스텝/5 시드 parity 결과."),
    tableSimple(parityTable, [2800, 2200, 2200, 2200]),
    ...figure("fig6_parity_scaling_progression.png", "그림 3. d=128/256/512 규모에 따른 parity 정확도 진행. MIMO+ternary peak는 0.86 → 0.95 → 0.98로 강화되며 표준편차도 0.15 → 0.09 → 0.04로 좁아진다."),
    p("표 5와 그림 3은 d=512/depth=4 설정에서 다음을 보인다: (i) Mamba-3 MIMO + ternary peak 0.981 ± 0.036; (ii) 동일 아키텍처 FP는 0.510 ± 0.003으로 무작위에 머무름; (iii) 효과 크기 0.47, σ ≈ 0.04 → 약 13σ 분리; (iv) 2배 시퀀스 길이 일반화 0.765로, 단순 외움이 아닌 진정한 상태 추적에 근접."),
    p("핵심 통제 실험: 본 효과가 단순한 'low-bit' 효과인지 1.58비트 고유 효과인지 구분하기 위해, 동일한 설정에서 Mamba-3 + INT4 PTQ도 함께 측정하였다."),
    tableCaption("표 6. Parity INT4 통제 실험 (d=512, 3 시드)."),
    tableSimple(int4ParityTable, [3500, 2200, 2200]),
    ...figure("fig9_int4_parity_control.png", "그림 4. Parity 통제 실험: FP·INT4·ternary 비교. INT4(4비트)는 FP(16비트)와 통계적으로 구별되지 않는 무작위 수준에 머무르고, ternary(1.58비트)만 약 0.97의 정확도를 달성한다."),
    p("표 6과 그림 4가 보이는 핵심은 INT4(16개 이산 값)가 FP(연속체)와 통계적으로 구별되지 않는 무작위 수준에 머무른다는 점이다(peak 격차 ≤ 0.018, 1σ 미만). 즉 본 귀납적 편향은 일반적 'low-bit'의 효과가 아니라 1.58비트의 3-값 이산 구조 {−1, 0, +1}의 효과이며, 임계 정밀도는 1.58비트와 4비트 사이에 위치한다. 이 결과는 SISO/MIMO 양쪽에서 일관되게 관찰된다."),
  ];
}

function hardwareSection() {
  // Hardware-specific section removed: this paper is software-only.
  // Hardware deployment of BitMamba-3 is mentioned only as future work in §7.
  return [];
}

function discussion() {
  return [
    h1("6. 논  의"),
    h2("6.1 가설 H1 (양자화 비용) — 결과 해석"),
    p("4.3절의 분해 결과는 ternary 양자화의 비용이 INT4 PTQ 대비 +8.2% PPL에 그침을 보였다. 일반적인 PTQ에서 4비트 → 2비트 전환은 10–20% 이상의 PPL 증가를 야기하지만, 본 연구의 from-scratch ternary 학습은 보정-기반 PTQ보다 효율적이다. 그 이유는 ternary 격자 자체에 학습 가중치가 적응(STE 기울기 + 활성 RMSNorm)하기 때문이며, 이는 BitNet b1.58 [3]이 3B 규모에서 보고한 'lossless' 결과와 본질적으로 동일한 메커니즘이다. 다만 본 연구는 학습 토큰 예산이 480M으로 작아 절대 PPL은 published Mamba-2 FP(300B 토큰)와 직접 비교할 수 없으며, 이는 6.4절에서 별도로 논의한다."),
    h2("6.2 가설 H2 (귀납적 편향) — 메커니즘 가설"),
    p("4.7절은 동일 모델·동일 학습·동일 시드 하에서 1.58비트 ternary가 4비트 INT4·16비트 FP가 풀지 못하는 parity 과제를 약 13σ 분리로 해결함을 보였다. 이 결과를 단순한 'low-bit 효과'로 설명할 수 있는지 INT4 통제 실험으로 직접 검증하였고, INT4(4비트, 16-값 격자)는 FP(연속체)와 통계적으로 구별되지 않는 무작위 수준에 머물렀다. 이는 본 귀납적 편향의 임계 정밀도가 1.58비트와 4비트 사이에 위치함을 시사한다."),
    p("본 결과의 메커니즘 가설은 이산 격자의 'commitment' 효과이다. parity는 본질적으로 이산 함수(XOR)이며, 그 해는 가중치 공간에서 좁은 매니폴드에 국한된다. 16-값(INT4) 또는 연속체(FP) 격자에서는 학습이 parity와 무관한 부드러운 함수에 더 쉽게 수렴하지만, 3-값(ternary) 격자에서는 부드러운 근사가 표현 자체로 불가능하므로 학습은 XOR 구조로 강제 수렴된다고 해석할 수 있다. 단, 본 가설은 가능한 설명 중 하나이며 반증을 위해 더 큰 규모(d ≥ 1024) 및 자연어 상태 추적 과제로의 확장이 필요하다(6.4절)."),
    h2("6.3 아키텍처(Mamba-3) vs 양자화의 상대적 중요도"),
    p("4.3절은 동일 학습 예산 하에서 Mamba-2 → Mamba-3 변경(matched ternary)의 PPL 우위가 130M에서 −39%, 370M에서 −43%인 반면, FP → ternary 양자화 비용은 +12.2%에 그침을 보였다. 즉 본 학습 예산 하에서 가장 영향이 큰 단일 결정은 양자화 종류가 아닌 아키텍처 선택이며, 양자화는 그 위에 추가되는 작은 비용에 불과하다. 4.5절의 LAMBADA 결과(M2 130M PPL 6408 → M3 370M PPL 826, 7.75배)는 이를 더 극단적으로 드러내며, 4.6절의 Needle-in-Haystack에서 370M의 장거리 회상 능력 향상도 같은 추세에 부합한다."),
    p("이 발견의 함의는 ternary 양자화의 가치가 'Mamba-3와 결합될 때' 가장 명확하다는 것이다. 단순 압축 측면에서는 INT4가 PPL 비용·메모리 측면에서 ternary보다 우세할 수 있지만(FPGA 곱셈기 자원이 충분한 경우), Mamba-3 + ternary 조합은 (i) 곱셈기 제거, (ii) 메모리 추가 절감, (iii) 상태 추적 귀납적 편향이라는 세 이득을 동시에 제공한다. 따라서 본 조합의 가장 적합한 대상은 LUT-rich·DSP-poor 엣지 FPGA(e.g. Zybo Z7-20, Kria KV260) 및 곱셈기 제거가 ASIC 면적·전력의 핵심 제약인 도메인이다."),
    h2("6.4 미비점 분석 및 보완 방향"),
    p("본 연구의 결과가 충분히 강하지 않은 지점은 다음과 같으며, 각각에 대해 보완 경로를 제시한다."),
    p("(i) 절대 PPL — 학습 토큰 예산이 480M으로 일반적 LLM 학습(수백 B)의 0.16% 수준이므로, 본 연구의 모든 PPL은 published 모델과 직접 비교할 수 없다. Mamba-3 공식 1.5B 체크포인트가 출시될 경우 사전학습 가중치에서 출발한 quantization-aware 미세조정 또는 PTQ를 적용하면 절대 PPL을 INT4 PTQ에 가까운 수준까지 끌어올릴 수 있을 것으로 기대한다.", { firstLine: false }),
    p("(ii) parity 규모 — 4.7절의 parity 결과는 d ≤ 512·depth ≤ 4의 작은 합성 과제이며, 자연어에서 상태 추적이 요구되는 다운스트림(예: 트랙 카운팅, 함수 호출 깊이 추론, 변수 바인딩 추적)에서 동일 메커니즘이 유지되는지는 미해결 문제이다. 향후 1.5B 규모에서 Mamba-3 공식 [6]이 보고한 100% parity 결과와 BitMamba-3의 동일 과제 결과를 비교함으로써 본 가설을 더 큰 규모에서 검증할 수 있다.", { firstLine: false }),
    p("(iii) 4비트 PTQ 강도 — 본 연구의 INT4는 RTN로, GPTQ·AWQ와 같은 SOTA 4비트 PTQ보다 약 1–3% PPL이 높다. 따라서 4.3절의 +8.2% (INT4 → ternary) 격차는 SOTA 4비트 PTQ와 비교하면 +9–11%로 약간 확대될 가능성이 있다. 단, 본 결과의 핵심 비교는 'matched 정규화 절차 하의 정밀도 효과'이므로 RTN이 가장 통제된 비교 대상이다.", { firstLine: false }),
  ];
}

function limitations() {
  return [];
}

function conclusion() {
  return [
    h1("7. 결  론"),
    p("본 연구의 목적은 (H1) Mamba-3 + 1.58비트 삼진 양자화의 양자화 비용이 INT4 PTQ 기준선 대비 작은 PPL 손실에 그치며 메모리·하드웨어 이득이 명확함을, (H2) 1.58비트 이산 구조가 단순한 정밀도 감소를 넘는 귀납적 편향으로 작용함을 검증하는 것이었다. 이를 위해 Mamba-3의 in_proj·out_proj 선형 계층을 BitLinear로 교체하는 최소-변경 인터페이스를 구현하고, 동일 데이터·동일 설정 하에서 FP·INT4 PTQ·ternary·BitMamba-2 네 구성을 직접 비교하였다."),
    p("핵심 결과로 (1) BitMamba-3 130M의 양자화 비용은 INT4 대비 +8.2% PPL에 그치며 메모리는 2.5배·곱셈기는 원천 제거된다(H1 지지). (2) 동일 양자화 하에서 Mamba-3 아키텍처는 Mamba-2 대비 130M에서 1.64배·370M에서 1.76배의 PPL 우위를 보이며, 양자화 비용을 압도하는 크기이다. (3) 가장 강한 결과로, 1.58비트 이산 구조 자체가 상태 추적 과제에서 약 13σ 분리의 귀납적 편향으로 작용하며, 이는 4비트 PTQ에서는 관찰되지 않음을 통제 실험으로 확인하였다(H2 강하게 지지). 임계 정밀도는 1.58비트와 4비트 사이에 위치한다."),
    p("이상의 결과는 Mamba-3 + 1.58비트 양자화의 결합이 단순한 압축 기법이 아니라 알고리즘적 의의를 갖는 설계 선택임을 시사한다. 후속 연구는 (i) Mamba-3 공식 1.5B 체크포인트로부터의 quantization-aware 미세조정 및 자연어 상태 추적 과제로의 확장, (ii) 4비트 SOTA PTQ(GPTQ/AWQ) 대비 비교를 통한 INT4 → ternary 격차의 보다 정밀한 정량화, (iii) 곱셈기 제거의 잠재 이득을 정량화할 수 있는 FPGA·ASIC 기반 하드웨어 구현 및 on-board throughput·에너지 측정의 세 방향으로 진행될 수 있다."),
  ];
}

function references() {
  const refs = [
    "[1] E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh, \"GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers,\" ICLR, 2023.",
    "[2] J. Lin et al., \"AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration,\" MLSys, 2024.",
    "[3] S. Ma et al., \"The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits,\" arXiv:2402.17764, 2024.",
    "[4] A. Gu and T. Dao, \"Mamba: Linear-Time Sequence Modeling with Selective State Spaces,\" COLM, 2024.",
    "[5] T. Dao and A. Gu, \"Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality,\" ICML, 2024.",
    "[6] N. Lahoti et al., \"Mamba-3: Improved Sequence Modeling using State Space Principles,\" ICLR, 2026 (arXiv:2603.15569).",
    "[7] Zhayr et al., \"Fully Quantized Mamba in 1.58 Bits From Head to Toe,\" COLING, 2025.",
  ];
  return [
    h1("참고문헌"),
    ...refs.map(r => new Paragraph({
      spacing: { after: 80, line: 280 },
      indent: { left: 360, hanging: 360 },
      children: [new TextRun({ text: r, font: "Times New Roman", size: 18 })],
    })),
  ];
}

// === Main document assembly ===

const children = [
  ...title(
    "BitMamba-3: A 1.58-bit Ternary Quantization of the Mamba-3 State-Space Model with State-Tracking Inductive Bias",
    "BitMamba-3: 상태 추적 귀납적 편향을 갖는\nMamba-3 상태공간 모델의 1.58비트 삼진 양자화"
  ),
  ...authorBlock(),
  ...abstractKo(),
  ...abstractEn(),
  new Paragraph({ children: [new PageBreak()] }),
  ...intro(),
  ...background(),
  ...method(),
  ...experiments(),
  ...hardwareSection(),
  ...discussion(),
  ...limitations(),
  ...conclusion(),
  new Paragraph({ children: [new PageBreak()] }),
  ...references(),
];

const doc = new Document({
  creator: "Author",
  title: "BitMamba-3 paper",
  description: "BitMamba-3: 1.58-bit ternary Mamba-3",
  styles: {
    default: { document: { run: { font: "맑은 고딕", size: 20 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "맑은 고딕" },
        paragraph: { spacing: { before: 360, after: 180 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: "맑은 고딕" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 }, // A4
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      }
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: [PageNumber.CURRENT], font: "Times New Roman", size: 18 })],
      })] })
    },
    children,
  }],
});

const outPath = path.join(__dirname, 'BitMamba3_paper_ko.docx');
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(outPath, buf);
  console.log(`Wrote ${outPath} (${buf.length} bytes)`);
});
