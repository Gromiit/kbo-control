# v9 residual correction — PLAN

작성 2026-08-13 · **계획만, 학습 없음** · 기준 커밋 `883bd36`

---

## 0. 결론 먼저 — 세 모델 모두 이미 측정되었고 전부 실패했습니다

| 제안 모델 | 대응하는 기존 실험 | 2024 결과 |
|---|---|---|
| **residual MLP** | 17차 MODEL 2 (`step5.log`) | **dBSS −177.0 / −218.0** |
| **GRU baseline** | Phase A (이 세션) | dRes +136.6이나 **§5·§6 배포 금지** |
| **Transformer encoder** | `883bd36`에서 종결 | 합법 입력 없음 |

특히 **residual MLP는 제안하신 것과 문자 그대로 같은 실험**입니다 — 17차
`run_step5.py`가 "residual learning on top of v9's OOF prediction, in both the
logit and probability modes"를 수행했습니다.

```
=== MODEL 2: v9 residual learning ===
  2023  train [2022] (197,977행)        mode=logit   dRes  +11.8  dBSS   +1.1  corr 0.9961
  2023  train [2022]                    mode=prob    dRes  +11.2  dBSS   +7.8  corr 0.9969
  2024  train [2022,2023] (443,892행)   mode=logit   dRes  -63.1  dBSS -177.0  corr 0.9499
  2024  train [2022,2023]               mode=prob    dRes  -70.2  dBSS -218.0  corr 0.9385
```

**2024에서 −177 / −218입니다.** 그리고 v9와의 상관이 0.94~0.996 — 모델이
v9를 재생산할 뿐입니다.

앙상블로 섞어도 마찬가지입니다.

```
-- resid_logit,  2024
  w=0.05  dBSS +0.6 (paired +0.4±1.0)
  w=0.10  dBSS +0.1 (paired -0.2±1.9)
```

---

## 1. 요청하신 4개 검증

### 1.1 입력 데이터 합법성

주최측 §6 허용 목록: `train.csv`, 평가 환경 `test.csv`, **2019~2024**
`trackman_history.csv`, 규칙상 허용 외부 데이터.

**합법입니다.** 다만 v9 residual을 타깃으로 쓰려면 학습 행에 `p_v9`가 있어야
하고, 여기서 결정적 제약이 생깁니다(§1.4).

### 1.2 sequence 생성 가능 여부

| 종류 | 가능 여부 |
|---|---|
| **시즌 내 (2025) sequence** | **불가.** 창을 만들려면 다른 test 행을 읽어야 하고 §5가 "행 순서 기반 rolling/expanding"을 명시 금지. 2025 투구 로그는 §6이 금지(2025 Trackman)하고 허용 목록은 2019~2024뿐 |
| **historical (시즌 간) sequence** | 생성은 가능하나 **G0에서 FAIL** (same-season ceiling에서도 3-seed plus −6.7 ± 7.2 / basic +7.6 ± 10.7, 유의성 없음) |

즉 **합법이면서 미탐색인 sequence가 없습니다.** GRU/Transformer의 입력이
존재하지 않습니다.

### 1.3 historical 정보만 사용 가능한가

가능합니다. 그리고 그것이 G0가 측정한 대상입니다 — Trackman 2019~2023 투수
프로필 78개 feature로 same-season ceiling에서 **dRes(B−A) −6.7 ± 7.2**.
"주변분포가 담을 수 있는 최대치"가 0이었으므로, 그 위의 sequence encoder는
설계 문서(`historical_pitcher_embedding.md`)의 게이트에 따라 구현하지
않았습니다.

### 1.4 baseline(v9) 대비 residual OOF 평가 — **여기서 막힙니다**

**`p_v9` OOF는 2022·2023·2024 세 시즌에만 존재합니다** (`work/research/oof_*.parquet`).
따라서 v9 residual을 타깃으로 하는 모델의 학습 데이터는:

```
fold 2023   train 2022 만            197,977 행
fold 2024   train 2022 + 2023        443,892 행
```

일반 타깃(`control_success`)이면 fold 2024에서 **1,221,585행**을 쓸 수 있습니다.
**residual 타깃은 그 36%만 씁니다.**

17차가 정확히 이 제약 아래에서 돌았고(`train [2022, 2023] (443,892 rows)`),
결과가 −177/−218이었습니다. 데이터가 3분의 1로 줄고 타깃 분산은 훨씬 작아진
조건에서 신경망이 학습할 것이 남지 않습니다.

이 제약은 앞선 pitcher×batter 실험에서도 동일하게 나타났습니다 — `resid` 타깃의
페어 커버리지가 51.2% → **44.8%**로 떨어졌습니다.

---

## 2. 왜 residual 학습이 구조적으로 불리한가

세 가지가 겹칩니다.

1. **학습 데이터 64% 손실** (§1.4)
2. **타깃 분산 축소**: v9는 이미 BSS 917.2이므로 잔차는 거의 잡음입니다.
   error analysis가 확인했듯 Reliability는 42.9로 BSS의 4.7%뿐이고, 남은 손실은
   Resolution 부족 — 즉 **잔차에 학습할 구조가 별로 없습니다.**
3. **v9 재생산**: 17차 결과의 corr 0.9385~0.9961이 이를 그대로 보여줍니다.
   `init` 대신 `p_v9`를 offset base로 쓰면 모델은 v9를 복사하는 방향으로
   수렴합니다.

대조적으로 17차 MODEL 1(잔차가 아니라 `init` 위에서 라벨 직접 학습)은 전체
1.22M행을 쓰고 corr 0.890으로 더 독립적이었습니다. **그럼에도 3-seed 평균
블렌드는 2024에서 dBSS +0.4였습니다** (`ablation.log` 2절).

> 단일 seed로는 +6.2까지 나옵니다(`step5.log`, w=0.10). 3-seed 평균에서
> +0.4로 사라집니다. 이것이 17차가 "단일 seed 판정 금지"를 남긴 이유이고,
> 이 프로젝트에서 반복 확인된 패턴입니다.

---

## 3. 그럼에도 수행한다면 — 최소 설계

새로 얻을 것이 있는 유일한 축은 **17차가 안 한 조합**입니다. 두 개뿐입니다.

| | 17차가 한 것 | 미탐색 |
|---|---|---|
| residual MLP | seed 1개, logit/prob 2 모드 | **3-seed 평균** |
| MODEL 1 (init 기반) | 3-seed 있음 (ablation) | — |

즉 **"residual MLP를 3 seed로 돌리면 −177이 개선되는가"** 가 유일한 미측정
질문입니다. 그리고 단일 seed에서 −177이면 3-seed 평균이 양수가 될 여지는
없습니다(seed 평균은 분산을 줄일 뿐 편향을 고치지 못합니다).

**따라서 설계상 남은 실험이 없습니다.** GRU/Transformer는 입력이 없고,
residual MLP는 측정 완료이며, 3-seed는 부호를 바꿀 수 없습니다.

---

## 4. 게이트 — 형식상 기록

수행한다면 적용할 기준입니다. 실행은 권하지 않습니다.

```
PASS   3-seed 평균 dBSS > 0  AND  dRes >= +10  AND  paired >= 2σ   (2024, 사전고정 w)
FAIL   그 외
중단   2023 fold 에서 최적 설정이 음수면 2024 를 열지 않음
```

판정은 **2024**로 합니다. 2023은 BCAT 붕괴로 v9 기준선이 403.1까지 내려가
있어(A9 단독 843.8보다 낮음) 블렌드 이득이 과대평가됩니다 — `step5.log`의
2023 dBSS +257.6이 정확히 그 산물입니다.

가중치는 **2023에서 고르고 2024는 한 번만** 엽니다. 17차는 스윕만 보고했고
사전 고정 w=0.10 기준으로는 +0.4였습니다.

---

## 5. 예상

| 결과 | 확률 |
|---|---|
| residual MLP 3-seed 2024 dBSS < 0 | **~85%** |
| 0 ~ +10 | ~13% |
| PASS (≥ +10, 2σ) | **~2%** |

단일 seed −177에서 3-seed로 부호가 뒤집히려면 seed 분산이 편향보다 커야 하는데,
`ablation.log` 1절의 MODEL 1 seed별 BSS 산포(2024: 778.1 / 691.5 / 676.7)를 보면
그 정도는 아닙니다.

---

## 6. 권고

**수행하지 않기를 권합니다.**

1. **residual MLP는 17차 MODEL 2와 동일한 실험**이고 2024에서 −177/−218입니다.
2. **GRU/Transformer는 합법 입력이 없습니다** — 시즌 내 sequence는 §5·§6 금지,
   historical은 G0 FAIL.
3. **v9 residual 타깃은 학습 데이터를 64% 잃습니다** (`p_v9` OOF가 2022~2024뿐).
4. error analysis가 보였듯 **v9의 남은 손실은 Reliability 4.7%가 아니라
   Resolution 95%** 이므로, 잔차 교정이라는 프레이밍 자체가 큰 몫을 겨냥하지
   않습니다.

이 방향을 종료하면 딥러닝 계열은 세 갈래(Phase A sequence, embedding MLP,
residual correction) 모두 기록과 함께 닫힙니다.

**제출본은 `work/submit_v9.zip`으로 유지되며, 이 계획의 어떤 단계도 그것을
수정하지 않습니다.**

---

## 7. 제약 준수

| 제약 | 대응 |
|---|---|
| v9 제출파일 수정 금지 | 열지 않음 |
| test.csv 접근 금지 | 계획·게이트 어디에도 경로 없음. fold/OOF parquet만 |
| 학습 전 PLAN만 | 코드 없음. 승인 전 학습 없음 |
| 기존 모델 수정 금지 | 신규 파일만 예정 |
| 산출물 위치 | `experiments/deep_learning_v2/` 아래만 |
| temporal split | 2023 선택 → 2024 최종 1회, 판정 fold 2024 |

---

## 부록 — 참조한 기존 산출물

| 경로 | 내용 |
|---|---|
| `work/research/deep_learning/step5.log` | MODEL 2 v9 residual learning 결과 |
| `work/research/deep_learning/ablation.log` | MODEL 1 3-seed, embedding ablation A–F |
| `work/research/deep_learning/model_residual.py` | 17차 ResidualMLP 구현 |
| `work/research/oof_{2022,2023,2024}.parquet` | p_v9 OOF (세 시즌뿐) |
| `experiments/error_analysis/REPORT.md` | Reliability 42.9 = BSS의 4.7% |
| `experiments/ensemble/xgb/E0_REPORT.md` | 같은 입력 공간에서 다양성 확보 불가 |
| `research/deep_learning_next/historical_pitcher_embedding.md` | G0 설계·FAIL |
