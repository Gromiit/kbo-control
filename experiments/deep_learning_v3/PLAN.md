# Tabular Transformer 계열 탐색 — PLAN

작성 2026-08-13 · **계획만, 코드 없음** · 기준 커밋 `883bd36`

---

## 0. 핵심 발견 — 게이트가 관측된 frontier 바깥에 있습니다

계획을 쓰기 전에 17차 ablation A–F(2024 fold)를 게이트 기준으로 다시 재었습니다.

| variant | BSS | corr(v9) | blend@0.1 | BSS≥750 | corr<0.85 | blend≥+30 | 통과 |
|---|---|---|---|---|---|---|---|
| A no-emb | **777.3** | 0.891 | +6.4 | **O** | X | X | 1/3 |
| B context-emb | 621.0 | **0.815** | −4.9 | X | **O** | X | 1/3 |
| C +pitcher | 619.1 | **0.815** | −2.8 | X | **O** | X | 1/3 |
| D +batter | 634.1 | **0.823** | −1.2 | X | **O** | X | 1/3 |
| E pitcher only | 690.5 | **0.848** | −3.3 | X | **O** | X | 1/3 |
| F +pit+bat | 620.2 | **0.812** | −4.7 | X | **O** | X | 1/3 |

**여섯 변형 모두 정확히 1/3만 통과합니다.** BSS를 통과한 것은 corr에서 떨어지고,
corr을 통과한 것은 BSS에서 떨어집니다.

우연이 아닙니다. BSS와 corr이 거의 완전한 선형 관계입니다.

```
BSS ≈ 2052 × corr − 1051        R² = 0.998

  corr = 0.85 에서 기대 BSS  693.2      (게이트 요구 750)
  BSS  = 750 이 되는 corr    0.878      (게이트 요구 < 0.85)
```

**게이트 영역 `BSS ≥ 750 AND corr < 0.85`는 이 frontier 위에서 비어 있습니다.**
필요한 지점은 관측선에서 약 57 BSS 위쪽, 또는 corr 0.028 왼쪽입니다.

### 이것이 의미하는 것

같은 101개 feature 위에서 정확도와 탈상관은 **교환 관계**입니다. embedding을
넣으면 v9와 달라지지만(0.891 → 0.812) 그 대가로 BSS가 157점 무너집니다.
E0(XGBoost)가 내린 결론 — *병목은 알고리즘이 아니라 feature 공간* — 과 같은
현상이며, 이번에는 **R² 0.998로 정량화**되었습니다.

**후보 세 개가 이 선을 벗어날 수 있는가가 이 실험의 실질 질문입니다.**

---

## 1. 후보별 사전 평가

### 1.1 FT-Transformer

feature tokenizer로 수치·범주형을 토큰화하고 Transformer로 attention.
**셋 중 frontier를 벗어날 가능성이 가장 높습니다** — MLP 계열과 귀납 편향이
실제로 다릅니다(feature 간 상호작용을 attention으로 명시 모델링).

다만 표 형식 데이터에서 FT-Transformer가 GBDT를 이기지 못한다는 것은 널리
보고된 바이고, 여기서 필요한 건 이기는 게 아니라 **다르게 틀리는 것**입니다.
그 방향의 근거는 약합니다.

### 1.2 TabTransformer

범주형에만 attention을 걸고 수치형은 그대로 통과시킵니다. **v9의 b_feats는
범주형이 4개(`pitcher_team_id`, `batter_team_id`, `count_state`,
`base_state_c`)뿐**이고, 나머지는 전부 수치형입니다. 즉 TabTransformer의
차별화 지점이 이 데이터에서는 4개 컬럼에만 작동합니다.

그리고 **범주형을 잘 다루는 모델의 자리는 이미 BCAT가 차지하고 있습니다** —
BCAT는 CatBoost로 같은 4개 범주형 + ID 2개를 쓰며, A7/A9와의 상관이
**0.8187**로 세 구성요소 중 가장 독립적입니다. TabTransformer가 성공한다면
BCAT 근처에 착지할 텐데, 그러면 **BCAT와 높은 상관**을 갖게 되어 blend
기여가 제한됩니다.

### 1.3 DeepGBM 계열

**설계 목적이 GBDT 구조를 신경망에 증류하는 것**입니다. 즉 구조적으로 v9의
LightGBM/CatBoost를 모사하도록 만들어졌고, **높은 상관이 설계 의도**입니다.

게이트가 `corr < 0.85`를 요구하는데 DeepGBM은 그 반대를 지향합니다.
**세 후보 중 가장 기대가 낮습니다.**

---

## 2. 요청 조건 검증

| 조건 | 상태 |
|---|---|
| `train.csv`만 사용 | 가능. `data/folds/fold_2024_{tr,va}.parquet`가 train.csv에서 파생 |
| test.csv 접근 금지 | 계획·게이트 어디에도 경로 없음 |
| 2025 feature 금지 | fold 구조가 시즌 < S 로 이미 강제 |
| submit_v9.zip 수정 금지 | 열지 않음 |
| OOF 기반 평가 | fold_2024_tr(2019–2023, 1,221,585행) → fold_2024_va(2024, 253,507행) |
| Brier 기준 | `metrics.decomp` 재사용 (BSS = Res − Rel) |
| v9 상관 계산 | 로짓 공간, `oof_2024.parquet`의 p_A7/p_A9/p_Bcat 및 p_v9 |
| 단독보다 blend 우선 | 게이트 3항 중 blend가 결정 |

**중요 — 타깃 선택**: `control_success`를 직접 학습하고 `init`을 offset base로
씁니다(v9 A계열과 동일). **v9 잔차를 타깃으로 하면 안 됩니다** — `p_v9` OOF가
2022~2024에만 존재해 학습 행이 1,221,585 → 443,892로 64% 줄고, 17차 MODEL 2가
그 조건에서 2024 dBSS **−177.0 / −218.0**을 냈습니다.

---

## 3. `blend dBSS >= +30`은 도달 가능한가

측정된 모든 상한입니다.

```
3요소 로짓 stacking, 2024 in-sample 상한          +19.4
  + XGBoost 추가                                   +21.5   (개선 +2.0)
  + XGB 양 arm (base+noinit)                       +21.5
17차 MLP 3-seed, 사전고정 w=0.10                    +0.4
17차 MLP 단일 seed, 최적 w (평가시즌 선택)           +8.0
17차 embedding 변형                                -4.9 ~ -1.2
17차 residual MLP                                -177.0 / -218.0
```

**`+30`은 기존 3요소를 2024에서 보고 맞춘 완벽한 stacker(+19.4)보다 높습니다.**
그 상한을 지금까지 가장 크게 밀어 올린 것이 XGBoost의 **+2.0**입니다.

`+30`을 달성하려면 새 모델이 **기존 셋이 함께 놓친 정보**를 상당량 가져와야
하는데, error analysis가 답한 바 그 정보의 소재지는 **투수 개인 상태**이고
다섯 갈래로 접근이 막혔습니다.

---

## 4. 그럼에도 수행한다면 — 최소 설계

frontier 이탈 여부만 봅니다. **FT-Transformer 하나만** 돌립니다.

TabTransformer는 §1.2의 이유로(범주형 4개, BCAT와 자리 겹침), DeepGBM은
§1.3의 이유로(높은 상관이 설계 목표) 제외를 권합니다. FT-Transformer가
frontier를 못 벗어나면 나머지 둘은 볼 필요가 없습니다.

```
모델    FT-Transformer (feature tokenizer + 3층 Transformer, d_token 64, heads 8)
입력    v9 numeric_feats 101 + 범주형 4 (b_cat_features)
타깃    control_success,  init 을 offset base 로 (p = sigmoid(init + f(x)))
학습    fold_2024_tr (2019-2023, 1,221,585행)
평가    fold_2024_va (2024, 253,507행)
seed    3개 (42/43/44), 로짓 공간 평균
장비    Colab GPU (Mac MPS 는 attention 학습에 부적합)
```

**측정 순서** — 값싼 것부터, 미달이면 중단합니다.

| 단계 | 내용 | 통과 조건 |
|---|---|---|
| **V0** | 단독 BSS와 corr(v9)를 frontier 선(`BSS ≈ 2052·corr − 1051`)에 대조 | **잔차 ≥ +57 BSS** (즉 게이트 영역 진입) |
| **V1** | 4요소 로짓 stacking 2024 in-sample 상한 | 3요소 +19.4 대비 **+30 이상** |
| **V2** | 사전고정 w (2023에서 선택) 로 2024 blend, 3-seed | **dBSS ≥ +30 AND paired ≥ 2σ** |

V0에서 frontier 위에 그대로 얹히면 **V1을 돌리지 않습니다** — §0이 보였듯
frontier 위의 어떤 점도 게이트를 통과하지 못합니다.

**판정 fold는 2024**입니다. 2023은 BCAT 붕괴로 v9 기준선이 403.1까지 내려가
있어(A9 단독 843.8보다 낮음) blend 이득이 크게 과대평가됩니다 — 17차의 2023
dBSS +257.6이 그 산물입니다.

---

## 5. 예상

| 결과 | 확률 |
|---|---|
| V0 탈락 (frontier 위에 착지) | **~70%** |
| V1 탈락 (stacking 상한 개선 < +30) | ~22% |
| V2 탈락 (전이 실패 / 3-seed 소멸) | ~6% |
| **PASS** | **~2%** |

근거:

1. **frontier의 R²가 0.998**입니다. 6개 변형이 거의 완벽한 직선 위에 있습니다.
   구조가 다른 모델이 이 선을 57 BSS 벗어난다는 가정은 강한 주장입니다.
2. **XGBoost E0가 같은 질문에 답했습니다** — 다른 라이브러리로 알고리즘을 바꿔도
   corr 0.8955~0.9064, stacking 개선 +0.3~+2.0. 병목은 feature 공간입니다.
3. **`+30`은 3요소 in-sample 상한(+19.4)보다 높습니다.**
4. 표 형식 데이터에서 Transformer 계열이 GBDT 대비 갖는 이점은 주로 정확도가
   아니라 표현력인데, 여기서 필요한 건 정확도를 유지한 채 탈상관하는 것입니다 —
   §0의 trade-off가 정확히 그것을 막습니다.

---

## 6. 권고

**V0만 수행하십시오.** Colab GPU로 반나절이면 frontier 이탈 여부가 확정됩니다.

V0를 통과하면 그때 V1/V2를 설계합니다. 탈락하면 **딥러닝 계열 네 갈래(Phase A
sequence, embedding MLP, residual correction, tabular transformer)가 모두
닫히고**, 남는 결론은 `work/submit_v9.zip` 유지입니다.

**TabTransformer와 DeepGBM은 V0 통과 전까지 구현하지 않기를 권합니다.** 둘 다
FT-Transformer보다 사전 확률이 낮고(§1.2, §1.3), V0가 답하는 질문이 셋 모두에
공통이기 때문입니다.

---

## 7. 산출물

```
experiments/deep_learning_v3/
  PLAN.md              이 문서
  # V0 승인 시에만
  build_ftt_oof.py · v0_frontier.py · oof/*.parquet · V0_REPORT.md
```

`experiments/deep_learning_v3/` 밖에는 아무것도 만들지 않습니다. 제출 파일은
생성하지 않으며, V2를 통과하더라도 새 번들 제작은 별도 논의 대상입니다.

---

## 부록 — 참조

| 경로 | 내용 |
|---|---|
| `work/research/deep_learning/ablation.log` | A–F 변형, frontier의 근거 |
| `work/research/deep_learning/step5.log` | residual MLP −177/−218 |
| `experiments/ensemble/xgb/E0_REPORT.md` | 같은 feature 공간에서 다양성 불가 |
| `experiments/ensemble/PLAN.md` | 3요소 stacking 상한 +19.4 |
| `experiments/error_analysis/REPORT.md` | 남은 손실의 95%가 Resolution |
| `work/research/oof_2024.parquet` | p_A7 / p_A9 / p_Bcat / p_v9 |
