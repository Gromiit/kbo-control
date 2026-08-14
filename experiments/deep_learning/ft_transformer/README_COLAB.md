# FT-Transformer V0 — Colab 실행 안내

> ## 권장: `FTTransformer_V0_colab.ipynb`
>
> **Colab 에서 이 노트북 하나를 열고 위에서 아래로 실행하면 V0 평가까지 끝납니다.**
> Drive 마운트 → 저장소 → GPU 확인 → 의존성 → 데이터 → 학습 → 게이트 → 결과 저장이
> 셀 단위로 들어 있고, 각 단계가 실패하면 그 자리에서 멈춥니다.
>
> ```
> experiments/deep_learning/ft_transformer/FTTransformer_V0_colab.ipynb
> ```
>
> 사전 준비는 §2 (Mac 에서 `prep_data.py` 실행 후 `ftt_data.tgz` 를 Drive 에 업로드)
> 하나뿐입니다. 노트북이 §1 의 archive contract — sha256, payload 집합 일치,
> schema 내용 — 을 전부 검사한 뒤에야 학습으로 넘어갑니다.
>
> 아래 shell runner (`setup_colab.sh` / `run_v0_colab.sh`) 는 **advanced users only**
> 로 남겨 둡니다 — 노트북과 같은 일을 하지만 셀 단위 확인이 없습니다.

이 문서만 보고 실행할 수 있게 썼습니다. Mac 에서는 데이터 준비만 하고,
학습은 전부 Colab GPU 에서 합니다.


---

## 0. 이 실험이 묻는 것

v9 를 대체하려는 것이 아닙니다. **v9 와 다른 feature interaction 을 학습하는
독립 모델이 가능한가**만 봅니다.

17차 MLP 변형 6종이 이 데이터에서 다음 직선 위에 놓였습니다.

```
BSS = 2052 × corr(v9) − 1051        R² = 0.998
```

정확도와 탈상관이 1:1 로 교환됩니다. 게이트는 `corr < 0.85` 에서 `BSS ≥ 750`
을 요구하는데 직선은 그 지점에서 **693.2** 를 예측하므로, 통과하려면
**직선에서 약 +57 BSS 벗어나야** 합니다. V0 는 attention 이 그 밖에
착지하는지만 봅니다.

---

## 1. Archive contract

Mac 과 Colab 사이를 건너가는 것은 tarball 하나뿐이므로, **그 안에 무엇이
있는지가 계약**입니다. 양쪽이 똑같이 검사합니다.

```
contract  ftt-v0/1

data/ftt_2024_tr.parquet   1,221,585 행   seasons 2019-2023   p_v9 492,997 (40%)
data/ftt_2024_va.parquet     253,507 행   season  2024        p_v9 253,507 (100%)
data/schema.json           contract · feature 목록 · 행수 · parquet sha256
```

**정확히 이 셋이어야 하며 그 외에는 없어야 합니다.** 금지: `test.csv`,
`work/`, `submit`, `.pkl`, `.cbm`, `.zip`, `checkpoint`, `shard`.

> ### 크기로는 버전을 구분할 수 없습니다
>
> `schema.json` 이 없던 구버전 아카이브도 **같은 260.3 MB** 이고 금지항목
> 검사도 통과합니다. 실제로 그 아카이브가 Drive 에 올라가 Colab 에서
> `schema.json` 을 읽는 셀에서야 터졌습니다. 그래서 이제 payload 를
> **집합 일치**로 검사하고, `schema.json` 에 `contract` 버전을 박아 둡니다.
> 구버전 아카이브는 압축을 풀기 전에 거부됩니다.
>
> **노트북은 없는 `schema.json` 을 추정해서 만들지 않습니다.** 잘못된
> 아카이브로 판정하고 중단합니다 — 40~90분 학습 뒤에 알게 되는 것보다 낫습니다.

---

## 2. 준비 (Mac, 1회)

`prep_data.py` 는 **인자를 받지 않습니다.**

```bash
cd "~/Desktop/LG Aimers 9"
export PYTHON=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
"$PYTHON" experiments/deep_learning/ft_transformer/prep_data.py
```

한 번 실행하면 생성 → 재-open → payload 검사 → schema 검사 → 금지항목 검사 →
sha256 출력까지 끝납니다. 마지막에 이런 블록이 나옵니다.

```
  아카이브  .../ft_transformer/ftt_data.tgz
  크기      260,261,226 bytes (260.3 MB)
  sha256    d89b5522451e95e0d3bdfd9d7cc1653cbf4d410fbe62d69e6dad1145753a660a
  payload   ['data/ftt_2024_tr.parquet', 'data/ftt_2024_va.parquet', 'data/schema.json']
  schema.json sha256  69d73d20d622a3241beb097ca43f2bae419d9b7a03c8ac551f16099db11c3ca2
```

**이 sha256 을 적어 두십시오.** 노트북 셀 5 가 같은 값을 출력해야 합니다.

아카이브는 `ftt_data.tgz.tmp` 로 먼저 만들고 **검증을 통과한 뒤에만** 교체됩니다.
중간에 실패하면 임시 파일만 지워지고 기존 아카이브는 그대로 남습니다.

### 기존 archive 교체 절차

크기가 같아 Drive 에서 눈으로 구분할 수 없으므로 **덮어쓰지 말고 지운 뒤
올리십시오.**

1. Drive 에서 `MyDrive/kbo/ftt_data.tgz` 를 **삭제** (휴지통도 비우기 — 동명
   파일이 두 개 남으면 Colab 이 어느 쪽을 보는지 알 수 없습니다)
2. 새 `ftt_data.tgz` 를 `MyDrive/kbo/` 에 업로드하고 동기화 완료까지 대기
3. Colab 런타임 **재시작** 후 노트북을 **셀 1 부터** 다시 실행
4. 셀 5 의 `sha256` 이 위 값과 같은지, `payload` 가 세 파일인지 확인
5. 이미 풀린 구버전이 남아 있으면 먼저 지우기

```python
!rm -rf /content/kbo/experiments/deep_learning/ft_transformer/data
```

---

## 3. Colab 설정

**런타임 > 런타임 유형 변경 > T4 GPU** 로 바꾸고 런타임을 다시 시작합니다.
GPU 가 아니면 `setup_colab.sh` 가 즉시 멈춥니다.

---

## 4. 실행

### 권장 — Notebook

Colab 에서 `FTTransformer_V0_colab.ipynb` 를 열고 **위에서 아래로** 실행하십시오.
GitHub 에서 바로 열 수 있습니다.

```
File > Open notebook > GitHub > Gromiit/kbo-control
  experiments/deep_learning/ft_transformer/FTTransformer_V0_colab.ipynb
```

셀 구성:

| 셀 | 내용 | 실패 시 |
|---|---|---|
| 1 | Drive 마운트 | — |
| 2 | 저장소 clone / `pull --ff-only` | assert 로 파일 확인 |
| 3 | GPU 확인 | **RuntimeError** — MPS/CPU 학습 차단 |
| 4 | 의존성 | import 실패 시 중단 |
| 5 | 데이터 + **계약 검증** | sha256 · payload 집합 일치 · schema · 행수 · 누수 |
| 6 | 모델 설정 확인 (`--help`) | — |
| 7 | 학습 seed 42 (A/B/C) | 40~90 분 |
| 8 | 게이트 평가 | `work/` 미사용 |
| 9 | 판정 (BSS·corr·gap·blend) | — |
| 10 | `results/` 저장 | 금지 산출물 assert |

### advanced users only — shell runner

노트북과 같은 일을 셀 확인 없이 실행합니다.

```python
# (1) 저장소 + Drive
!git clone https://github.com/Gromiit/kbo-control.git /content/kbo 2>/dev/null || (cd /content/kbo && git pull --ff-only)
from google.colab import drive; drive.mount('/content/drive')
%cd /content/kbo
```

```bash
# (2) 환경 확인 — GPU 없으면 여기서 멈춤
!bash experiments/deep_learning/ft_transformer/setup_colab.sh
```

```bash
# (3) 압축 해제 + 학습 + 평가 + 결과 위치 출력
!bash experiments/deep_learning/ft_transformer/run_v0_colab.sh
```

환경변수로 바꿀 수 있습니다.

```bash
!FTT_ARCHIVE=/content/drive/MyDrive/kbo/ftt_data.tgz SEED=42 MODELS=A,B,C \
  bash experiments/deep_learning/ft_transformer/run_v0_colab.sh
```

---

## 5. 예상 시간 (T4 기준)

| 단계 | 시간 |
|---|---|
| 압축 해제 | 1~2 분 |
| Model A (numeric 101, 1.22M 행, 15 epoch) | 15~30 분 |
| Model B (+ 범주형 4) | 15~30 분 |
| Model C (492,997 행, 40%) | 6~12 분 |
| 게이트 평가 | 1 분 |
| **합계** | **40~90 분** |

`patience 3` 으로 조기 종료하므로 15 epoch 을 다 쓰지 않을 수 있습니다.

---

## 6. 모델 세 갈래

| | 입력 | 출력 | 학습 행 |
|---|---|---|---|
| **A** | numeric 101 | `p = sigmoid(f(x))` | 1,221,585 |
| **B** | numeric 101 + 범주형 4 임베딩 | `p = sigmoid(f(x))` | 1,221,585 |
| **C** | B 와 같은 몸통 | `p = sigmoid(logit(p_v9) + f(x))` | **492,997 (40%)** |

**C 의 행이 적은 것은 버그가 아닙니다.** C 는 `p_v9` 가 있는 행만 쓸 수 있는데
v9 OOF 가 2022~2024 에만 존재합니다. A/B 와 직접 비교할 때 반드시 감안하십시오.

C 의 목적은 성능이 아니라 **"v9 의 로짓을 받은 모델이 v9 를 복제하는가,
아니면 그 위에 resolution 을 더하는가"** 입니다.

타깃은 셋 다 `control_success` 입니다. **residual 타깃은 쓰지 않습니다** —
17차가 그 경로로 2024 dBSS **−177.0 / −218.0**, corr 0.94~0.996 을 냈습니다.

---

## 7. 결과 확인

```
experiments/deep_learning/ft_transformer/
  oof/ftt_A_s42.parquet      row_id, p
  oof/ftt_B_s42.parquet
  oof/ftt_C_s42.parquet
  oof/manifest_s42.json      설정 · 파라미터 수 · 행수 · best BSS
  v0_results_s42.csv         게이트 결과
  results/metrics.json       판정 · 임계값 · schema 요약
  results/gate_report.md     모델별 표와 판정
```

게이트 출력에서 볼 것:

```
model X
  BSS / Resolution / Reliability
  calibration bias (p − y)
  corr(v9)  |  A7 / A9 / BCAT
  frontier 예측 BSS  ->  이탈 ±NN      <-- 여기가 핵심
  4요소 stacking 상한   3요소 대비 ±NN
  고정가중 blend  w=0.05 … 0.30
  GATE  1) BSS>=750  2) corr<=0.95  3) blend>=+30
```

**`이탈` 값이 판정의 핵심입니다.** BSS 와 corr 을 따로 보면 왜 떨어졌는지
놓칩니다. +57 이상이어야 게이트 영역에 들어갑니다.

### 게이트는 Colab 에서 그대로 돕니다

`v0_gate.py` 는 v9 구성요소(`p_A7` / `p_A9` / `p_Bcat`)를 준비된
`data/ftt_2024_va.parquet` 에서 읽습니다. **`work/` 에 접근하지 않으므로** Colab 에서
바로 실행됩니다. `work/research/oof_2024.parquet` 은 Mac 에서 아카이브 없이 돌릴 때의
fallback 으로만 쓰입니다.

결과를 Drive 로 남기려면:

```python
!cp -r experiments/deep_learning/ft_transformer/results /content/drive/MyDrive/kbo/ftt_results
!cp -r experiments/deep_learning/ft_transformer/oof     /content/drive/MyDrive/kbo/ftt_oof
```

---

## 8. 실패 시 체크

| 증상 | 원인과 조치 |
|---|---|
| `torch.cuda.is_available() False` | 런타임 유형이 GPU 가 아님. 변경 후 **런타임 재시작** |
| `아카이브가 없습니다` | Drive 미마운트 또는 경로 오타. `drive.mount` 후 `!ls /content/drive/MyDrive/kbo/` 확인 |
| `CUDA 가 필요합니다` | `train_ftt.py` 의 의도된 거부. MPS/CPU 로는 돌리지 않습니다 |
| `CUDA out of memory` | `--batch 256` 으로 낮추십시오. 기본 512, 토큰 102~106 |
| `train season leaked into 2024` | 아카이브가 잘못된 fold. Mac 에서 `prep_data.py` 재실행 |
| `row_id 정렬 실패` (게이트) | OOF 와 `oof_2024.parquet` 의 row_id 불일치. 같은 seed·같은 아카이브인지 확인 |
| 세션 끊김 | 학습 재개 기능이 없습니다. OOF 를 Drive 로 복사해 두고 seed 단위로 다시 실행하십시오 |
| `MISSING src/deep_learning_state/metrics.py` | `git pull` 이 안 된 상태. `%cd /content/kbo` 후 `!git pull` |
| **`payload 가 계약과 다릅니다 … 누락 ['data/schema.json']`** | **구버전 아카이브입니다.** 크기가 같아 구분되지 않습니다. §2 의 교체 절차대로 Drive 파일을 지우고 새로 만든 것을 올리십시오 |
| `contract … != ftt-v0/1` | 아카이브와 코드 버전 불일치. `git pull` 후 Mac 에서 `prep_data.py` 재실행 |
| `sha256 불일치` (parquet) | Drive 업로드가 중간에 끊겼습니다. 지우고 다시 올린 뒤 런타임 재시작 |
| 셀 5 sha256 이 §2 출력과 다름 | Drive 에 구버전이 남아 있습니다. 동명 파일 중복 여부 확인 |
| 노트북 셀 10 에서 `NameError: TH` | 셀 9 를 건너뛰었습니다. 위에서부터 순서대로 실행하십시오 |
| `v9 구성요소를 찾을 수 없습니다` | 데이터 셀(5)을 실행하지 않았거나 아카이브가 구버전. Mac 에서 `prep_data.py` 재실행 |

---

## 9. 하지 않는 것

- `work/submit_v9.zip` 을 열거나 수정하지 않습니다
- `work/` 아래에 아무것도 만들지 않습니다
- `models.py` / `train.py` / `dataset.py` 등 기존 pipeline 을 수정하지 않습니다
- `test.csv` 를 열지 않습니다 — 학습은 `fold_2024_tr`, 예측은 `fold_2024_va`
- 제출 파일을 만들지 않습니다

---

## 10. V0 이후

게이트 3개를 모두 통과하면 **그때** seed 3개로 확장합니다. 하나라도 떨어지면
추가 실험 없이 종료하고 `work/submit_v9.zip` 을 유지합니다.

```bash
# 통과했을 때만
!SEED=43 bash experiments/deep_learning/ft_transformer/run_v0_colab.sh
!SEED=44 bash experiments/deep_learning/ft_transformer/run_v0_colab.sh
```
