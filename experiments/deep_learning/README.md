# 딥러닝 · 앙상블 연구 인덱스

최종 갱신 2026-08-13 · 기준 커밋 `883bd36`

**제출본은 `work/submit_v9.zip`으로 동결되어 있습니다.** 아래 어떤 연구도 그것을
수정하지 않았고, 앞으로도 수정하지 않습니다.

---

## 1. 종료된 연구

일곱 갈래가 각각 게이트에서 종료됐습니다. **어느 것도 배포 가능한 이득을 내지
못했습니다.**

### 1.1 Sequence model (Phase A · C)

| | |
|---|---|
| 무엇 | GRU/시퀀스로 투수의 시즌 내 상태를 모델링 |
| 결과 | 2024 dRes **+136.6**, 2023 **+542.3** (3 seed 안정) — 신호는 크고 실재함 |
| **종료 이유** | **규정상 배포 불가.** 입력이 "같은 시즌 직전 32구"인데 2025 창을 만들려면 다른 test 행을 읽어야 하고 §5가 "행 순서 기반 rolling/expanding"을 명시 금지. 2025 투구 로그도 §6이 금지 |
| 증류 시도 | temporal 복원 38.4%(3 seed, R² 0.379로 안정)이나 복원분의 **46%가 v9와 중복** → v9+student 블렌드 **dBSS +0.1 (0.01σ)** |
| 기록 | `src/deep_learning_state/`, 커밋 `0795095` |

index 분해가 원인을 특정합니다 — teacher의 이득은 **시즌 내 100구 이상 축적된
행**에만 있고(+211/+136), 100구 미만에서는 음수(−732~−67)입니다.

### 1.2 Historical pitcher embedding (G0)

| | |
|---|---|
| 무엇 | Trackman 2019–2023 투수 프로필(78 feature)을 legal한 대안 경로로 |
| 결과 | same-season **ceiling에서도** 3-seed FAIL — plus dRes −6.7 ± 7.2, basic +7.6 ± 10.7 |
| **종료 이유** | 주변분포가 담을 수 있는 최대치가 0. basic seed42의 +16.3은 **seed variance의 false positive**(seed44는 −4.4). 더 학습시키면 오히려 악화(−10.3, 2.2σ) |
| 기록 | `src/deep_learning_state/{pitcher_profile,g0_ceiling}.py`, 커밋 `f6e1d5b` |

### 1.3 Embedding MLP

| variant (2024) | BSS | corr(v9) | blend@0.1 |
|---|---|---|---|
| A no-emb | **777.3** | 0.891 | +6.4 |
| E pitcher only | 690.5 | 0.848 | −3.3 |
| F +pitcher +batter | 620.2 | **0.812** | −4.7 |

**종료 이유**: embedding은 **정확도를 파괴하는 대가로 탈상관을 삽니다**.
no-emb 대비 157 BSS 손실. 그리고 BSS와 corr이 **R² 0.998의 선형 관계**
(`BSS ≈ 2052·corr − 1051`)라 "정확하면서 독립적인" 영역이 비어 있습니다.

기록: `work/research/deep_learning/ablation.log`, 커밋 `883bd36`

### 1.4 Residual MLP

```
2024  train [2022,2023] (443,892행)  logit  dRes -63.1  dBSS -177.0  corr 0.9499
2024                                 prob   dRes -70.2  dBSS -218.0  corr 0.9385
```

**종료 이유** 셋:

1. **v9 복제** — corr 0.94~0.996
2. **학습 데이터 64% 손실** — `p_v9` OOF가 2022~2024에만 존재해 fold 2024의
   학습 행이 1,221,585 → 443,892
3. **타깃에 구조가 없음** — v9의 Reliability는 42.9로 BSS의 4.7%뿐. 남은
   손실의 95%는 Resolution

기록: `work/research/deep_learning/step5.log`, `experiments/deep_learning_v2/PLAN.md`

### 1.5 XGBoost ensemble (E0)

| arm | corr(v9) | 단독 BSS | stacking 상한 개선 |
|---|---|---|---|
| base | **0.9064** | 796.1 | +0.3 |
| noinit | **0.8955** | 774.6 | +2.0 |

**종료 이유**: 같은 feature 공간에서는 알고리즘을 바꿔도 다양성이 생기지
않습니다. `noinit`(init을 떼어 다양성 최대화)조차 0.8955로 **기존 최중복
구성요소 A9(0.8960)와 동률**이며, 목표였던 BCAT 수준(0.8187) 근처에도 못
갔습니다. stacking 계수를 보면 XGB는 A7에서 가중치를 빼앗을 뿐(0.422→0.279)
새 정보를 더하지 않습니다.

기록: `experiments/ensemble/xgb/E0_REPORT.md`, 커밋 `a82daa9`

### 1.6 BCAT investigation (Stage 0)

| | 2023 | 2024 |
|---|---|---|
| 상위 10% gap | **−0.2045** | **+0.0057** |
| 절대 임계 `p≥0.6172` 행 | 10.0% | 0.3% |
| 붕괴 구간 투수들의 gap | −0.2045 | **−0.0022** |

**종료 이유**: **fold-specific 사건**입니다. 원인은 CatBoost의 ID feature
(`b_with_ids=True`)와 저표본 엔티티 과신 — 붕괴 구간의 `asof_pitcher_n`
중앙값이 1,311로 정상 구간(3,351)의 39%이고, 같은 행에서 A7/A9는 정상입니다.
2024에서 재현되지 않으며, 2025 모델은 2019–2024 전체로 학습되어 표본이 더
많습니다.

방어선은 전부 순손실입니다:

```
              2023 dBSS   2024 dBSS
clip <= 0.65     +98.5       -0.0
cap 0.08        +267.3       -8.1
shrink x0.7     +176.1      -37.0
```

기록: `experiments/bcat_next/stage0/`, 커밋 `43f476b`

### 1.7 Calibration · feature search · error analysis

| 연구 | 결론 |
|---|---|
| Calibration audit | 분위 기반 분해에서 **단조 변환은 Resolution을 정확히 보존**(정리). 상한은 현재 Reliability이고 전이 안 됨 — 2023→2024 6/6 음수, 전부 8σ 이상. gap 상관 **−0.60** |
| Feature survey | test.csv 48개 중 미사용 2개(`run_top_before`/`run_bot_before`)뿐이며 동어반복. 조건부 공간은 16차가 46개 조건 전수 탐색 |
| pitcher × batter | 46개 중 유일하게 두 시즌 양수였던 후보. 게이트 FAIL (M2 B−A dRes −13.8, dBSS −9.5, 0.90σ) |
| Error analysis | 남은 손실의 **95%가 Resolution**. 19개 분할변수 기여합이 14.5~38.6에 몰려 어떤 축도 두드러지지 않음. 유일한 초과 신호는 **투수 개인**(\|z\|≥3 11명 vs 기대 0.7명)이나 그 경로는 위 다섯 갈래로 이미 막힘 |

기록: `research/calibration_analysis/`, `research/deep_learning_next/`,
`experiments/error_analysis/`

---

## 2. 공통 구조 — 왜 전부 실패했는가

일곱 갈래가 독립적으로 같은 지점을 가리킵니다.

> **정보는 시즌 내에 존재하고, 시즌 경계를 넘지 못합니다.**

조건부 잠재력과 배포 가능성의 괴리가 일관됩니다.

```
pitcher x batter   잠재력 1295  ->  배포  +2.3
pitcher            잠재력   80  ->  배포  -58
GRU teacher        2024 +136.6  ->  student 블렌드 +0.1
calibration        상한  +42.9  ->  전이  -105.8
```

그리고 **입력 공간에 여백이 없습니다** — 원시 컬럼 미사용 2개(동어반복),
조건부 공간 46개 전수 탐색, Trackman 프로필 G0 FAIL, 시퀀스는 규정 금지.

E0(XGBoost)가 이를 알고리즘 축에서 재확인했고, embedding ablation이
**R² 0.998의 BSS–corr trade-off**로 정량화했습니다.

---

## 3. 현재 선택 — FT-Transformer V0

**목적은 v9를 대체하는 것이 아니라, v9와 다른 feature interaction을 학습하는
독립 모델을 확보할 수 있는지 확인하는 것입니다.**

셋 중 FT-Transformer만 검증합니다.

| 후보 | 판단 |
|---|---|
| **FT-Transformer** | **V0 대상.** feature tokenizer + attention으로 MLP 계열과 귀납 편향이 실제로 다름 |
| TabTransformer | 제외 권고 — 범주형이 4개뿐이고, "범주형을 잘 다루는 자리"는 **BCAT가 이미 차지**(corr 0.8187) |
| DeepGBM | 제외 권고 — **설계 목적이 GBDT 증류**라 높은 상관이 의도된 결과 |

### V0가 묻는 것

§1.3의 frontier(`BSS ≈ 2052·corr − 1051`, R² 0.998)를 **벗어나는가**.

```
corr = 0.85 에서 frontier 기대 BSS   693.2
게이트 요구                          750
필요한 이탈                          +57 BSS
```

V0에서 frontier 위에 그대로 얹히면 **V1(stacking 상한)을 돌리지 않습니다** —
그 선 위의 어떤 점도 게이트를 통과하지 못하기 때문입니다.

상세 설계: `experiments/deep_learning_v3/PLAN.md`

---

## 4. 원칙

| 원칙 | 내용 |
|---|---|
| **submit_v9 유지** | `work/submit_v9.zip` (sha256 `e30a19e2…`) 동결. 어떤 실험도 수정하지 않음 |
| **기존 pipeline 변경 금지** | `models.py` `train.py` `dataset.py` `metrics.py` `config.py` `paths.py` `device.py` `audit.py` `make_sequences.py` 및 `configs/` `scripts/` 무수정 |
| **protected files 무변경** | `work/model_v9.pkl`, `work/submit_v9.zip`, `work/submit/{script.py,requirements.txt,model/model.pkl}`, `cat_0.cbm`, `cat_1.cbm` |
| **test.csv 접근 금지** | 연구 코드에 경로가 등장하지 않음. fold/OOF parquet만 사용 |
| **연구용 코드만 추가** | 신규 파일은 `experiments/` 또는 `research/` 아래에만 |
| **temporal split 유지** | 하이퍼파라미터는 2023에서 선택, 2024는 최종 1회. 판정 fold는 **2024** |
| **3 seed 이상** | 단일 seed 판정 금지. 17차 +8.8→+2.8, G0 +16.3→−4.4 전례 |
| **제출 파일 생성 금지** | V2를 통과하더라도 새 번들 제작은 별도 논의 |

### 판정 fold를 2024로 고정하는 이유

2023 fold는 **BCAT가 붕괴**해 있습니다 (BSS 0.0, Rel 1777.6). 그 결과 v9 블렌드가
403.1로 A9 단독(843.8)보다 낮고, 어떤 신규 모델이든 블렌드 이득이 크게
과대평가됩니다 — 17차의 2023 dBSS +257.6이 정확히 그 산물입니다.

---

## 5. 디렉터리 지도

| 경로 | 내용 |
|---|---|
| `experiments/deep_learning/` | **이 인덱스**, embedding MLP 종결 기록, FT-Transformer 작업 위치 |
| `experiments/deep_learning_v2/` | residual correction 종결 기록 |
| `experiments/deep_learning_v3/` | FT-Transformer 상세 설계 |
| `experiments/deep_learning_state/` | Phase A 산출물 (results.csv, trace) |
| `experiments/ensemble/` | 앙상블 · XGBoost E0 |
| `experiments/bcat_next/` | BCAT Stage 0 |
| `experiments/error_analysis/` | v9 오류 진단 |
| `research/calibration_analysis/` | calibration 감사 |
| `research/deep_learning_next/` | feature 조사, historical embedding, pitcher×batter |
| `research/final_submission_audit.md` | 제출본 재현성·규정 준수 감사 |
| `src/deep_learning_state/` | Phase A~C 코드 (변경 금지 대상) |
