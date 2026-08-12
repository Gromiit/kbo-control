# Historical Pitcher Embedding — 설계 문서

작성 2026-08-13 · 상태 **설계만, 코드 없음** · 선행: Phase A~C-2 (commit `0795095`)

---

## 0. 먼저 읽을 것 — 이건 새 가설이 아닙니다

이 방향은 **15차 Trackman 연구가 이미 측정하고 기각한 것**과 데이터 소스·정보 경로가
같습니다. 다른 건 요약 함수뿐입니다(수작업 집계 → 학습된 인코더).

`work/research/trackman/t6_lawful.log` — 같은 `trackman_history.csv`에서 투수 단위
프로필 60개 피처를 만들어 v9에 붙인 결과:

| | 2023 | 2024 |
|---|---|---|
| LEVEL (평균) | +0.1 ± 2.8 | −3.0 ± 2.6 |
| SPREAD (반복성) | +2.4 ± 2.4 | +3.7 ± 2.3 |
| TREND (변화) | −2.8 ± 2.5 | +6.4 ± 2.0 |
| MIX (구종조합) | −0.6 ± 2.2 | +1.2 ± 1.7 |
| **ALL trackman (60개)** | **−2.6 ± 3.6** | **−0.3 ± 3.0** |

두 시즌 모두 **null**이고, 부호가 시즌 간에 뒤집힙니다(LEVEL, TREND). 프로젝트
채택 기준(`dRes ≥ 20` STRONG, `< 10` WEAK)에 한참 못 미칩니다.

그리고 Phase C-2에서 방금 확인한 것 — teacher의 이득은 **시즌 내 축적**에서만 나옵니다
(2024 valid, 시즌 내 투구 index별 dRes):

| index | n | dRes |
|---|---|---|
| 1–4 | 1,563 | −732.1 |
| 5–15 | 4,240 | −145.4 |
| 16–31 | 5,859 | −109.1 |
| 32–99 | 22,221 | −67.4 |
| 100–499 | 91,421 | **+211.0** |
| 500+ | 127,812 | **+135.9** |

시즌 내 이력 100구 미만에서는 teacher가 v9보다 **나쁩니다.** historical embedding은
정의상 시즌 경계 이전만 보므로, 이득이 나오는 구간을 원리적으로 재현하지 못합니다.

**따라서 이 설계의 목적은 "성공시키기"가 아니라 "싸게, 빨리, 확정적으로 판정하기"입니다.**
아래 §8의 게이트가 이 문서의 핵심이며, 나머지는 게이트를 통과했을 때만 의미가 있습니다.

---

## 1. 규정 적합성 — 합법입니다 (조건부)

`~/Desktop/open/data_description.md` §5·§6 원문 기준.

**허용 근거**

> 제공된 `train.csv`, 평가 환경의 `test.csv`, **2019~2024년 `trackman_history.csv`**,
> 그리고 대회 규칙상 허용되는 외부 데이터만 사용할 수 있습니다. (§6)

> 참가자는 이 파일을 이용해 과거 투구 특성, 구종 특성, **투수 단위 요약값** 등 추가
> 피처를 만들 수 있습니다. (§3)

제안 구조는 정확히 "투수 단위 요약값"입니다. 각 test 행은 자기 `pitcher_id`로 **동결된
룩업 테이블**을 조회할 뿐이므로 §5의 행 독립 원칙을 지킵니다.

**반드시 지켜야 하는 경계**

| 항목 | 규정 |
|---|---|
| `test.csv` 다른 행 사용 | **금지** — §5 "행 순서 기반 rolling 또는 expanding feature" |
| 2025 Trackman | **금지** — §6 명시. 시즌 내 갱신 불가 |
| 현재 투구의 Trackman 측정값 | **금지** — §6. 15차 오라클(+342~+383)이 이것이며 사용 불가 |
| 평가 시점 이후 정보 | **금지** — §3 |

**설계상 강제해야 할 것**: 검증 시즌 S에 대해 임베딩 테이블은 **시즌 < S** 자료로만
계산·적합. 2025 추론 시에는 2019~2024 전체로 계산한 테이블 하나를 동결해 `model.pkl`에
넣습니다. 기존 carry/split/role/lowrank 테이블과 동일한 취급입니다.

---

## 2. 현재 feature pipeline 분석

### 2.1 v9 구조

```
raw row ──► features.py: featurize(carry, split, role, lowrank 테이블)
                │  테이블은 전부 시즌 < S 에서만 적합 후 동결
                ▼
          numeric feats 101개 ──► level logreg ──► init (log-odds)
                │                                    │
                ├── A7 (LGB, feats 93, groups 4) ────┤ init_score
                ├── A9 (LGB, feats 101, groups 4) ───┤
                └── BCAT (CatBoost, b_feats 68, cat 4, cat_0/1.cbm)
                                   │
                       p = 0.25·A7 + 0.30·A9 + 0.45·BCAT   (shrink 1.1, shift −0.04)
```

numeric feats 101개 접두사 분포:

```
asof 19 · std 16 · edge 13 · p 7 · count 4 · lr 4 · log 3 · carry 3
pb 3 · runner 3 · pitcher 2 · batter 2 · balls 1 · strikes 1
```

`asof_*`(19개)와 `carry_*`(3개)가 이미 투수 이력 요약을 담당합니다. **임베딩은 이들과
직접 경쟁합니다** — 새 정보를 주지 못하면 중복입니다. C-2에서 student가 복원한 정보의
46%가 v9와 겹쳤던 것과 같은 구도입니다.

### 2.2 삽입 지점

제출 모델이 CatBoost 기반 유지이므로 **BCAT 계열의 `b_feats`(68개)에 임베딩 d개를 추가**
하는 것이 최소 침습입니다. A7/A9(LGB)에도 붙일 수 있으나 실험 축이 늘어나므로 게이트
통과 전에는 BCAT만 건드립니다.

`model.pkl`에 `pitcher_embedding` 테이블(shape `[n_pitcher, d]` + `pitcher_id` 인덱스)을
추가하고 `script.py`가 룩업하면 배포 가능합니다. **단 `script.py`와 `model.pkl`은 현재
보호 대상이므로, 채택이 확정되기 전에는 절대 수정하지 않고 새 번들로만 실험합니다.**

---

## 3. historical sequence 생성

### 3.1 소스와 조인

```
trackman_history.csv   1,793,079 행 · 30 컬럼 · 2019~2024
        │  pitcher_trackman_id
        ▼
work/research/trackman/pmap.parquet   641명 · share ≥ 0.978 (637명 =1.000)
        │  pitcher_id
        ▼
train/test 의 pitcher_id
```

`pmap`은 15차 산출물이며 **재사용합니다**(재구현 금지 — 규칙 2).

### 3.2 시퀀스 정의

투수 P, 대상 시즌 S에 대해:

> **시즌 < S 에서 P가 던진 마지막 N개 투구**, 시간순, 최신이 뒤.

- `N = 512` 기본 (투수당 시즌 평균 ~2,800구이므로 직전 시즌 상당 부분을 덮음)
- 좌측 패딩 + `length` 동반 — **GRU 버그 재발 방지**: 인코더는 마지막 스텝을 읽어야 함
  (`models.py` GRUState의 수정 사유 참조)
- 시즌 경계를 넘되 **S 이상은 절대 포함 금지**

### 3.3 채널 (투구당)

| # | 채널 | 비고 |
|---|---|---|
| 0–7 | `rel_speed` `spin_rate` `induced_vert_break` `horz_break` `extension` `rel_height` `rel_side` `zone_speed` | 결측률 0.14~0.44% (15차 확인) |
| 8–11 | `pitch_type_group` one-hot (fastball/breaking/offspeed/other) | |
| 12–14 | `balls_before/3` `strikes_before/2` `outs_before/2` | 상황 |
| 15 | `pitch_of_pa` clip/6 | |
| 16 | `season_gap = S − season` 정규화 | 최신성 |

**결정적 한계 — outcome 채널이 없습니다.** `trackman_history.csv`에는
`control_success`가 없습니다. Phase A GRU teacher의 **가장 강한 채널이 outcome**이었고
(채널 0), 그것이 +136.6의 원천이었습니다. 여기서는 "어떻게 던지는가"만 인코딩할 수
있고 "얼마나 잘 던졌는가"는 담기지 않습니다.

> 보완안: `train.csv`(2019~2024, 허용 데이터)에서 같은 투수의 과거 `control_success`
> 시퀀스를 별도 채널로 붙이는 것은 합법입니다. 다만 이는 `asof_pitcher_success_rate`
> 계열 19개가 이미 하는 일이라 중복 가능성이 높습니다. **게이트 통과 후에만 검토.**

### 3.4 커버리지 — 설계에 반드시 반영

```
2024 valid 행 중 pmap 매핑 투수      99.2%
2024 valid 고유 투수 391명 중 매핑    352명
2024 에 처음 등장한 투수              81명  →  해당 행 19.9%  (과거 이력 0)
```

15차의 "trackman 프로필 커버리지 0.807"과 일치합니다(1 − 0.199).

**약 20%의 행은 임베딩이 존재하지 않습니다.** 2025 신인/신규 등록 투수도 동일합니다.
따라서:

- `has_history` 이진 플래그를 **반드시 함께** 투입
- 이력 없는 행은 임베딩 0 벡터 (학습된 "unknown" 벡터 금지 — 17차에서 entity
  embedding의 unknown 처리가 temporal overfit을 키웠음)
- 평가는 전체 / `has_history=1` 부분집합 **양쪽 모두** 보고

---

## 4. pitch-level encoder 후보

세 후보 모두 `[N, C=17] → [d]` 를 만들고, **(pitcher, season) 당 한 번만** 계산되어
룩업 테이블이 됩니다. 행마다 인코딩하지 않으므로 추론 비용은 테이블 조회 O(1)입니다.

| # | 인코더 | 파라미터 | 역할 |
|---|---|---|---|
| **E0** | **mean + std pooling → Linear(2C → d)** | ~1k | **대조군.** 15차 LEVEL+SPREAD 계열의 학습 버전. 이걸 못 이기면 시퀀스 구조는 무가치 |
| E1 | GRU(hidden d, 1층) 최종 상태 | ~10k | 순서 의존 구조 |
| E2 | Transformer encoder 2층 4헤드 + CLS | ~50k | 장거리·집합 구조 (순서보다 분포에 강함) |

**E0가 대조군인 것이 이 설계의 핵심입니다.** 15차는 mean/sd/trend/mix를 수작업으로
만들어 null을 얻었습니다. E0는 그것의 학습 버전이므로 ≈ null이 기대값입니다. E1/E2가
E0를 유의하게 못 넘으면 "시퀀스 인코더"라는 프레이밍 자체가 기각됩니다.

**학습 목표**: 임베딩은 지도 신호가 필요합니다. 두 안 중 **(a)** 를 기본으로 합니다.

- **(a) 종단 학습**: 인코더 + 작은 head가 `init`을 offset해 `control_success`를 예측.
  Phase A와 동일 구조(`p = sigmoid(init + f)`), 학습은 시즌 < S 행으로만. 학습 후
  인코더를 동결하고 (pitcher, S) 임베딩을 뽑아 CatBoost에 투입.
- (b) 자기지도(투구 재구성/대조학습) 후 동결 — 라벨 누수 위험이 원천 차단되나 신호가
  과제와 무관해질 위험. (a)가 실패하면 (b)는 볼 필요 없음.

---

## 5. embedding dimension 32 / 64 / 128

**경고**: 유효 엔티티가 적습니다. 검증 시즌 고유 투수 **352명**, pmap 전체 641명입니다.
d=128이면 투수당 128차원을 352개 엔티티로 적합하는 셈이라 17차 entity embedding이
빠진 함정(temporal overfit 증가)과 동일 구조입니다.

| d | 판단 |
|---|---|
| **32** | **기본값.** 17차 "이 데이터는 capacity를 벌한다" 결론과 정합 |
| 64 | 32가 유의미하면 확인 |
| 128 | 32/64가 모두 실패하면 **돌리지 않음** (실패 원인이 capacity 부족일 가능성 낮음) |

d 비교는 **게이트 G2 통과 후에만** 수행합니다. 실패한 방향에서 d를 늘리는 것은 비용
낭비입니다.

---

## 6. fold-2024 temporal validation 설계

기존 `data/folds/` 구조와 leakage 규율을 그대로 승계합니다.

```
fold 2024 (주 검증)
  인코더 학습        : 시즌 2019–2023 행
  임베딩 테이블 계산  : 시즌 2019–2023 Trackman 만
  CatBoost 학습      : 2019–2023 (v9 b_feats 68 + emb d + has_history)
  검증              : 2024 (253,507행)
  기준선            : p_v9  (BSS 917.2 / Res 960.3 / Rel 42.9)

fold 2023 (전이 확인, 필수)
  인코더/임베딩/학습  : 2019–2022 만
  검증              : 2023 (245,525행)
  기준선            : p_v9  (BSS 403.1 / Res 558.7 / Rel 155.0)
```

- **seed 3개 이상** (42/43/44), 평균 ± 표준편차 보고 — 17차 +8.8 → +2.8 전례
- 판정 지표 우선순위: **dResolution → dBSS → paired bootstrap σ → seed 안정성**
- **dRes와 dBSS가 어긋나면 dBSS를 믿습니다.** C-2에서 dRes +4.5 / dBSS +0.1이 나왔고,
  대회 점수는 BSS입니다
- 전체 / `has_history=1` 부분집합 양쪽 보고
- 검증 셋으로 하이퍼파라미터·블렌드 가중치를 고르지 않습니다(2023에서 골라 2024에 적용)

기존 `audit.py`가 검사하는 항목은 그대로 유지하고, 임베딩 전용으로 2개 추가:

1. 임베딩 테이블의 소스 시즌이 전부 < S 인가 (bit 단위)
2. 검증 시즌 행의 임베딩이 학습 시 본 테이블과 동일한가 (동결 확인)

---

## 7. 예상 improvement — 증거 기반 추정

낙관적 추정을 적지 않습니다. 근거는 셋입니다.

1. **15차 동일 소스 60피처**: 2023 −2.6 ± 3.6, 2024 −0.3 ± 3.0 → null
2. **15차 surrogate 역전 현상**: 정보량이 큰 물리량일수록 예측 불가
   (movement +349.6인데 `induced_vert_break` R² 0.09 · `horz_break` R² 0.36;
   예측 잘 되는 `rel_side` R² 0.89는 정보량 release +88.7)
3. **C-2 index 분해**: 이득은 시즌 내 100구 이상 축적 구간에만 존재

| 시나리오 | 2024 dRes (3-seed 평균) | 주관 확률 |
|---|---|---|
| null (15차 재현) | −5 ~ +5 | **~65%** |
| WEAK | +5 ~ +15 | ~25% |
| INTERESTING | +15 ~ +25 | ~8% |
| STRONG (채택 가능) | ≥ +25 그리고 2023도 양수 | **~2%** |

**dBSS 기준으로는 더 비관적입니다.** C-2에서 dRes +4.5가 dBSS +0.1로 소멸했습니다.
Resolution 이득이 Reliability 손실로 상쇄되는 패턴이 반복될 가능성이 높습니다.

**유일하게 근거 있는 희망**: SPREAD(반복성) 계열이 두 시즌 모두 부호가 양수인 **유일한**
가족이었습니다(+2.4, +3.7). 투수의 구질 **분포 형태**에 신호가 있을 수 있고, mean/sd
두 모먼트로는 못 담는 부분을 E2(Transformer, 집합 구조에 강함)가 잡을 여지가 있습니다.
이것이 이 실험의 실질적 가설이며, 그렇다면 **E2 > E0** 가 반드시 관측되어야 합니다.

---

## 8. 게이트 — 이 문서의 핵심

각 게이트에서 조건 미달이면 **즉시 중단하고 기록만 남깁니다.** 다음 단계로 넘어가지
않습니다.

### G0 — 상한 확인 (반나절, 코드 최소)

C-2의 CEILING과 같은 발상. **2024 행 자체로** 인코더·CatBoost를 적합(투수 단위
GroupKFold)해 임베딩이 표현 가능한 최대 이득을 잽니다. temporal 주장도 배포 주장도
아닙니다.

> **통과 조건: CEILING dRes ≥ +20**
> C-2 실측에서 CEILING 56.1% → TEMPORAL 40.5%, 블렌드 이득은 +10.2 → +1.9로
> 약 5배 감쇠했습니다. CEILING이 +20 미만이면 temporal은 사실상 0입니다.

미달 시: **중단.** 15차 결론 재확인으로 기록.

### G1 — 인코더가 pooling을 이기는가 (1일)

E0 / E1 / E2를 fold 2024 temporal, d=32, seed 1개로 비교.

> **통과 조건: max(E1, E2) − E0 ≥ +10 dRes**

미달 시: **중단.** "시퀀스 구조는 mean/sd 이상을 주지 않는다"로 결론. 15차의
LEVEL+SPREAD가 이미 상한이었다는 뜻.

### G2 — 3-seed temporal, 두 시즌 (2일)

승자 인코더로 fold 2024·2023 × seed 42/43/44.

> **통과 조건: 2024 dRes 평균 ≥ +20 그리고 2023 dRes 평균 > 0 그리고 dBSS > 0**

미달 시: **중단.** 여기까지 오면 d 비교(§5)와 A계열 확장은 하지 않습니다.

### G3 — 배포 타당성

- 임베딩 테이블이 `model.pkl`에 들어가는 크기 (641 × d × 4B, d=32이면 82 KB — 무시 가능)
- `has_history=0` 행(~20%)에서 성능 저하가 없는지
- 2025 신규 투수 비율 추정과 그 구간 성능
- **`script.py` 수정은 이 시점에 처음 검토**하며, 원본은 보존하고 새 번들을 만듭니다

---

## 9. 비용 추정

| 단계 | 작업 | 시간 |
|---|---|---|
| 준비 | trackman 조인·시퀀스 shard 생성 (Mac) | 2~3 h |
| G0 | CEILING | 3~4 h |
| G1 | 인코더 3종 비교 | 1 d |
| G2 | 3-seed × 2 fold | 1~2 d (Colab GPU) |
| G3 | 배포 타당성 | 0.5 d |

G0에서 멈출 확률이 가장 높습니다(~65%). **총 반나절로 이 방향의 결론이 납니다.**

---

## 10. 권고

**G0만 수행하십시오.** 반나절 비용으로 확정적인 답이 나오고, 통과하면 그때 G1 이후를
설계하면 됩니다.

전체를 다 돌리는 것은 권하지 않습니다. 15차가 같은 소스로 null을 얻었고, C-2가 이득의
소재지가 시즌 내 축적임을 보였으며, 두 근거가 서로 독립적으로 같은 방향을 가리킵니다.

이 방향이 기각되면 남는 정직한 결론은 **"v9가 최종 제출본"** 이며, 이는 이미
commit `0795095`에 기록되어 있습니다.

---

## 부록 — 재사용할 기존 산출물 (재구현 금지)

| 경로 | 내용 |
|---|---|
| `work/research/trackman/pmap.parquet` | pitcher_id ↔ pitcher_trackman_id, 641명 |
| `work/research/trackman/t6_lawful.py` | 합법 프로필 피처 60개 생성 로직 |
| `work/research/deep_learning/data.py` | walk-forward fold + `numeric_feats()` |
| `data/folds/fold_{S}_{tr,va}.parquet` | featurize 완료 fold (2022/2023/2024) |
| `src/deep_learning_state/metrics.py` | BSS·Resolution·paired bootstrap |
| `src/deep_learning_state/audit.py` | leakage 감사 |
