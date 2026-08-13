# 딥러닝 representation 탐색 — PLAN

작성 2026-08-13 · **계획만, 코드 없음** · 기준 커밋 `43f476b`

---

## 0. 먼저 — 후보 1은 17차가 이미 측정했고, 후보 2는 Phase A가 이미 종결했습니다

계획을 쓰기 전에 `work/research/deep_learning/ablation.log`를 확인했습니다.
**ablation A–F가 제안하신 "pitcher_id embedding + batter_id embedding MLP"와
정확히 같은 실험**이며, 2023·2024 양 fold에서 측정되어 있습니다.

### 2024 fold 결과 (목표 시즌)

| variant | BSS | dResolution | **corr(v9)** | **blend @w=0.1** |
|---|---|---|---|---|
| A no-emb (MLP만) | **777.3** | −155.0 | 0.891 | **+6.4** |
| B context-emb | 621.0 | −291.4 | **0.815** | **−4.9** |
| C +pitcher | 619.1 | −297.1 | **0.815** | **−2.8** |
| D +batter | 634.1 | −283.5 | **0.823** | **−1.2** |
| E pitcher only | 690.5 | −207.7 | **0.848** | **−3.3** |
| **F +pitcher +batter** | **620.2** | **−297.3** | **0.812** | **−4.7** |

### 이 표가 게이트에 대해 말하는 것

주신 게이트는 `PASS: corr <= 0.85 AND blend improvement >= +30`입니다.

**embedding 변형들은 correlation 조건을 이미 통과합니다** (0.812 ~ 0.848).
그런데 **blend는 전부 음수**입니다 (−4.9 ~ −1.2). +30이 아니라 0에도 못
미칩니다.

원인이 표에 그대로 드러납니다. **embedding을 넣으면 단독 성능이 무너집니다** —
no-emb 777.3 → +pitcher+batter **620.2**로 157점을 잃습니다. 즉

> **embedding은 정확도를 파괴하는 대가로 탈상관을 삽니다.**

게이트의 두 조건이 서로 당기는 지점이 여기입니다. correlation 0.85 이하는
"기존과 다르다"는 뜻이지 "쓸모 있게 다르다"는 뜻이 아닙니다. 17차가 그 구분을
이미 측정했고, embedding은 전자만 만족합니다.

### 가중치를 정직하게 고르면

`ablation.log` 2절이 앙상블 스윕을 기록합니다.

```
2023 에서 고른 w=0.50  ->  2024 에서 dBSS -35.1
2024 에서 고른 w=0.05  ->  2023 에서 dBSS +36.9      (평가 시즌 선택 = 편향)
사전 고정 w=0.10       ->  2024 dBSS +0.4 / dRes +2.8
```

2023에서는 DL 블렌드가 크게 이득으로 보이지만(w=0.5에서 dBSS +276.9), 그건
**BCAT가 2023에서 붕괴해 v9 기준선이 403.1로 낮기 때문**입니다(Stage 0 참조).
2024에서는 사전 고정 가중으로 **+0.4**입니다.

### 후보 2 (Transformer encoder)

Phase A가 이미 종결했습니다.

- **시즌 내 sequence**: GRU teacher가 2024 dRes **+136.6**을 냈으나 §5·§6에서
  **배포 금지**입니다. Transformer로 바꿔도 입력이 같으므로 같은 위반입니다.
- **historical (시즌 간) sequence**: G0가 same-season ceiling에서도 FAIL
  (3-seed dRes plus −6.7 ± 7.2, basic +7.6 ± 10.7).

즉 **합법이면서 미탐색인 sequence 표현이 남아 있지 않습니다.**

---

## 1. +30이 도달 가능한 수치인가

측정된 상한들:

```
3요소 stacking in-sample 상한 (2024)        +19.4
  + XGBoost 추가 시                          +21.5   (개선 +2.0)
  + XGB 양 arm                               +21.5
DL(no-emb) 사전고정 w=0.1                    +0.4
DL(embedding) 사전고정 w=0.1                 -4.9 ~ -1.2
```

**`+30`은 기존 3요소를 2024에서 보고 맞춘 완벽한 stacker(+19.4)보다도 높습니다.**
XGBoost라는 다른 라이브러리가 그 상한을 +2.0 밀어 올렸고, 17차 DL은 밀어 올리지
못했습니다.

`+30`을 달성하려면 새 모델이 **기존 세 구성요소가 함께 놓친 정보**를 상당량
가져와야 합니다. 이 프로젝트가 일곱 갈래로 확인한 바에 따르면 그런 정보의
소재지는 **시즌 내 축적**이고, 그것은 규정상 접근 불가입니다.

---

## 2. 그럼에도 수행한다면 — 최소 확인 설계

주신 게이트를 그대로 쓰되, **가장 값싼 순서**로 배치합니다.

### D0 — 무학습 확인 (1시간)

17차가 남긴 것은 지표 trace뿐이고 **예측 파일은 없습니다**. 따라서 4요소
stacking 상한을 재계산하려면 최소한의 재학습이 필요합니다. D0는 그 최소치입니다.

```
모델    MLP + pitcher/batter embedding  (17차 variant F 재현)
        emb dim 8, hidden (256,128,64), init 을 offset 으로
fold    2024 만  (2023 은 BCAT 붕괴로 앙상블 판단 불가 — Stage 0 승인 규율)
seed    3개
학습    fold_2024_tr (2019-2023) -> fold_2024_va (2024)
```

측정:

| 항목 | 기준 |
|---|---|
| 단독 BSS | 참조: A7 808.6 / A9 829.4 / BCAT 827.3 / v9 917.2 |
| corr(로짓) vs A7·A9·BCAT | **PASS ≤ 0.85 · FAIL ≥ 0.95** |
| 4요소 stacking 상한 (2024 in-sample) | 3요소 +19.4 대비 **개선 ≥ +30** PASS · **< +10** FAIL |

**중간 구간(0.85 < corr < 0.95, 또는 개선 10~30)은 FAIL로 처리합니다.** 주신
기준에 명시된 PASS 조건이 AND이므로, 둘 다 만족하지 못하면 진행하지 않습니다.

### D1 — Transformer (D0 통과 시에만)

**D0를 통과하더라도 후보 2는 조건부입니다.** 합법 sequence 소스가 없으므로,
D1은 "sequence feature가 존재하면"이라는 전제 자체가 성립하지 않습니다.
D0 통과 시 재논의합니다.

---

## 3. 산출물

```
experiments/deep_learning/
  PLAN.md              이 문서
  # D0 승인 시에만
  build_dl_oof.py · d0_gate.py · oof/*.parquet · D0_REPORT.md
```

`experiments/deep_learning/` 밖에는 아무것도 만들지 않습니다.

> **디렉터리 이름 주의**: `work/research/deep_learning/`(17차)과
> `src/deep_learning_state/`(Phase A)가 이미 있습니다. 새 디렉터리는
> `experiments/deep_learning/`으로 세 번째입니다. 혼동을 피하려면
> `experiments/dl_repr/` 같은 이름이 나을 수 있습니다 — 지시대로 두었으나
> 변경을 원하시면 말씀해 주십시오.

---

## 4. 예상

| 결과 | 확률 |
|---|---|
| corr ≥ 0.95 → FAIL | ~15% |
| corr 0.85~0.95 → FAIL | ~25% |
| corr ≤ 0.85 이나 blend 개선 < +10 → FAIL | **~58%** |
| **PASS (corr ≤0.85 AND 개선 ≥+30)** | **~2%** |

가장 가능성 높은 경로는 세 번째입니다 — 17차 variant F가 정확히 그것이었습니다
(corr 0.812로 통과, blend −4.7로 탈락).

---

## 5. 권고

**수행하지 않기를 권합니다.**

1. **후보 1은 17차 ablation A–F가 이미 측정했습니다.** 2024 fold에서 embedding
   변형 전부가 blend 음수이고, 단독 BSS도 no-emb 대비 157점 낮습니다.
2. **후보 2는 합법 입력이 없습니다.** 시즌 내 sequence는 §5·§6 위반이고,
   historical sequence는 G0 FAIL입니다.
3. **`+30` 기준은 측정된 어떤 상한보다도 높습니다.** 3요소 완벽 stacker가
   +19.4이고 XGBoost가 +2.0을 더했습니다.

굳이 확인하고 싶으시다면 **D0만** 수행하십시오. 반나절이면 17차 결과의 재현
여부가 나오고, 재현되면 이 방향은 확정적으로 닫힙니다.

어느 경우에도 **제출 파일을 만들지 않고 `work/submit_v9.zip`은 동결 유지**합니다.

---

## 6. 제약 준수

| 제약 | 대응 |
|---|---|
| submit_v9.zip 수정 금지 | 열지 않음 |
| protected files 수정 금지 | 읽기 전용 |
| 기존 모델 수정 금지 | 신규 파일만. `dataset.feature_cols`·`metrics`만 import |
| 제출 파일 생성 금지 | D0는 OOF 예측만 저장. 번들 미생성 |
| `experiments/deep_learning/` 아래만 | 산출물 전부 그 아래 |
| test 데이터 접근 금지 | fold parquet만 사용 |
| 판정 fold | 2024 (2023은 BCAT 붕괴로 앙상블 판단 불가) |
