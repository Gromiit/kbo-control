# KBO 제구 성공률 예측 — Mac + Colab 분리 구조

MacBook Pro M4(16GB)에서 **코드·전처리·감사·스모크 테스트**를 하고,
Colab NVIDIA GPU에서 **full training**을 돌리는 구조입니다.

기존 v9 모델과 제출본은 이 구조가 건드리지 않습니다. `work/` 전체가
`.gitignore` 대상이고, 학습 코드는 `work/`를 **읽기만** 합니다.

---

## 먼저 알아둘 것 — 이 sequence 모델은 그 자체로 제출용이 아닙니다

`deep_learning_state`는 **투수의 시즌 내 상태(state)** 를 GRU로 모델링합니다.
한 행의 입력은 그 투수가 **같은 시즌에 앞서 던진 L개 투구**입니다.

이게 왜 의미 있냐면 — 16차 연구에서 투수의 시즌 내 잔차 구조가
**oracle로 Resolution +45**의 가치가 있는데 단일 행 feature로는 복원되지
않는다는 걸 확인했기 때문입니다. 시퀀스 모델은 그 구조를 직접 재는 도구입니다.

**그런데 2025 test는 5행짜리 셔플된 독립 행이고 라벨이 없습니다.**
대회 규정 §5는 test 행 간 집계를 금지합니다. 즉 추론 시점에
"이 투수의 직전 16구" 창을 만들 수 없습니다.

그래서 이 모델의 용도는 둘 중 하나입니다.

1. **측정 도구** — 시즌 내 상태가 실제로 얼마나 있는지 정량화
2. **teacher** — 단일 행 student로 증류(15차 `t9_student.py`가 Trackman에서
   같은 걸 시도해 +1.6/+5.3을 얻었습니다)

full training을 돌리기 전에 이 제약을 알고 계셔야 합니다.

---

## 디렉터리

```
LG Aimers 9/
├── src/deep_learning_state/     학습 패키지 (git 추적)
│   ├── paths.py                 경로 해석 (환경변수로 Mac/Colab 전환)
│   ├── device.py                CUDA > MPS > CPU
│   ├── config.py                YAML + CLI 오버라이드
│   ├── envinfo.py               git hash / torch / GPU / RAM 로깅
│   ├── metrics.py               BSS · Resolution · Reliability · LogLoss · AUC
│   ├── prepare_data.py          [Mac] walk-forward fold 생성
│   ├── make_sequences.py        [Mac] 시퀀스 shard 생성
│   ├── audit.py                 [Mac] leakage 감사 6종
│   ├── dataset.py               shard mmap → CPU → batch → GPU
│   ├── models.py                GRUState / StaticMLP
│   └── train.py                 진입점
├── configs/                     gru_smoke · gru_full · mlp_baseline
├── scripts/                     mac_prepare · mac_smoke · setup_colab · train_colab
├── notebooks/colab_train.ipynb  얇은 래퍼 (학습 로직 없음)
├── experiments/deep_learning_state/
│   ├── results.csv              ← git 추적
│   └── checkpoints/             ← git 제외
├── data/                        ← git 전체 제외
│   ├── folds/                   featurize된 walk-forward fold
│   └── sequences/{train,valid}/S{season}/shard_*.npz
└── work/                        ← 기존 v9 워크스페이스, 읽기 전용·git 제외
```

`work/research/deep_learning/`(17차 MLP 연구)는 **그대로 남아 있습니다.**
`prepare_data.py`가 그 안의 `data.py`를 import해서 재사용합니다 — 복사가
아니라 import라서 v9의 feature 생성과 절대 어긋날 수 없습니다.

---

## Mac workflow

```bash
cd "~/Desktop/LG Aimers 9"

# 1. 최신 코드
git pull

# 2. 전처리 + 시퀀스 + leakage 감사  (L=16, 전체 행)
bash scripts/mac_prepare.sh 16 2023,2024 0

# 3. 스모크 테스트 (100k행 · 1 epoch · resume 확인까지)
bash scripts/mac_smoke.sh

# 4. 커밋
git add -A && git commit -m "..." && git push
```

개별 실행:

```bash
python -m src.deep_learning_state.prepare_data
python -m src.deep_learning_state.make_sequences --seasons 2024 --sequence-length 16
python -m src.deep_learning_state.audit --seasons 2024 --sequence-length 16
python -m src.deep_learning_state.train --config configs/gru_smoke.yaml --device mps
```

---

## Colab workflow

`notebooks/colab_train.ipynb`를 열고 위에서부터 실행합니다. 셀은 전부
`scripts/*.sh` 또는 `python -m ...` 호출뿐이고 학습 로직은 없습니다.

```
1. git clone / pull
2. bash scripts/setup_colab.sh      의존성 + CUDA 확인
3. Drive 마운트 → data/sequences 심볼릭 링크
4. SMOKE_ONLY=1 bash scripts/train_colab.sh
5. CONFIG=configs/gru_full.yaml bash scripts/train_colab.sh
6. 결과를 Drive로 복사
```

직접 명령:

```bash
python -m src.deep_learning_state.train --config configs/gru_full.yaml --device cuda
```

---

## 데이터 이동

shard는 git에 넣지 않습니다. Mac에서 만들어 Drive로 옮깁니다.

```bash
# Mac
cd "~/Desktop/LG Aimers 9"
tar czf /tmp/kbo_seq_S2024_L32.tgz -C data sequences/manifest_S2024_L32.json \
    sequences/train/S2024 sequences/valid/S2024
# → Google Drive의 MyDrive/kbo/ 에 업로드
```

```python
# Colab
!mkdir -p /content/kbo/data && tar xzf /content/drive/MyDrive/kbo/kbo_seq_S2024_L32.tgz -C /content/kbo/data
```

또는 Drive에 `sequences/`를 그대로 두고 심볼릭 링크(노트북 3번 셀).

**용량 기준** (fold 2024, static 168 + L채널 8, float16):
L=16 약 1.0GB, L=32 약 1.6GB.

---

## GPU 자동 선택

`--device`를 생략하면 **CUDA > MPS > CPU** 순으로 자동 선택합니다.
명시했는데 없으면 조용히 CPU로 떨어지지 않고 즉시 종료합니다.

backend별 차이는 `device.py`에만 있습니다. `models.py`와 `train.py`의
forward 경로에는 `if cuda:`가 한 줄도 없습니다.

| | CUDA | MPS | CPU |
|---|---|---|---|
| autocast | bf16/fp16 | **무시** (torch 버전별 불안정) | 무시 |
| pin_memory | O | X | X |
| num_workers | 설정값 | macOS는 ≤2로 제한 | ≤2 |

`mixed_precision: true`를 MPS에서 켜면 조용히 NaN을 만드는 대신
"무시함" 로그를 남기고 fp32로 갑니다.

---

## Leakage 규칙

검증 시즌 S에 대해:

- carry / split / role / low-rank 테이블과 level 로지스틱 회귀를 **시즌 < S 에서만** 적합
- 시즌 S 행은 그 얼어붙은 테이블을 통과 (2025 test 행과 동일 경로)
- 시퀀스 창은 **같은 시즌·같은 투수의 앞선 투구만**, 시즌 경계를 넘지 않음
- scaler는 train split에서만 적합해 얼림

`audit.py`가 6가지를 실제로 검증합니다 — 시즌 격리 / row_id 중복 /
창 재계산 비트 일치 / 패딩·length / scaler 출처 / v9 OOF 정렬.
실패하면 non-zero로 종료해서 `mac_prepare.sh`가 업로드 전에 멈춥니다.

---

## checkpoint / resume

```bash
python -m src.deep_learning_state.train --config configs/gru_full.yaml \
  --resume experiments/deep_learning_state/checkpoints/gru_full_fold2024_last.pt
```

`_last.pt`는 매 epoch, `_best.pt`는 검증 BSS 갱신 시 저장합니다.
model / optimizer / scheduler / GradScaler / epoch / config / git hash가
전부 들어갑니다.

---

## 결과

`experiments/deep_learning_state/results.csv`에 한 줄씩 append:

`run, git_commit, config, fold, seed, model, sequence_length, hidden_size,
epochs, batch_size, device, gpu, dataset, train_rows, valid_rows,
total_seconds, epoch, BSS, Resolution, Reliability, LogLoss, AUC, pred_std,
corr_with_v9, dBSS, dResolution, dReliability, paired, paired_se`

epoch별 궤적은 `{run}_trace.json`.

**판정은 BSS가 아니라 dResolution으로 합니다.**
`dResolution <= 0` FAIL / `< 10` WEAK / `< 20` INTERESTING / `>= 20` STRONG.
17차에서 단일 seed의 +8.8이 3-seed 평균에서 +2.8로 사라졌으므로,
**seed 1개로 판정하지 마세요.**

---

## 병렬 실행 정책

하지 않습니다. GPU 하나에 모델 하나, seed 순차:

```bash
SEEDS='42 43 44' bash scripts/train_colab.sh
```

Mac에서는 `num_workers`가 2로 강제 제한됩니다(16GB 공유 메모리 + MPS).
