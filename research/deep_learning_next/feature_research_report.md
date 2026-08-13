# Feature Research Report — 미사용 합법 feature 후보 조사

작성 2026-08-13 · 코드 없음 · 기준 커밋 `f6e1d5b`

---

## 0. 결론 먼저

**요청하신 5개 영역 중 4개는 이미 측정되었고 전부 기각되었습니다.** 16차
`work/research/resolution_discovery/`가 46개 조건을 잠재력(d1)과 배포 가능성(d2)
양쪽으로 스윕했습니다. 남은 1개(pitch_type matchup)만 미탐색이며, 그것도
구조적 제약이 큽니다.

원시 컬럼 수준의 여백은 **사실상 없습니다.** test.csv 48개 중 v9가 안 쓰는 것은
`run_top_before` / `run_bot_before` 둘뿐이고, v9는 이미 `run_total_before` +
`score_diff_home` + `score_diff_pitcher_team`로 같은 정보를 담고 있습니다.

따라서 이 리포트의 실질 내용은 "새 후보 나열"이 아니라 **"어디가 이미 막혔고,
왜 막혔으며, 진짜 남은 틈이 무엇인가"** 입니다.

---

## 1. 배포 제약 — 모든 후보가 통과해야 하는 관문

2025 test 행 하나가 볼 수 있는 것은 정확히 둘뿐입니다.

1. **그 행의 48개 컬럼** (주최측이 계산한 `asof_*` 19개 포함, §5에서 명시 허용)
2. **2019~2024로 만든 동결 룩업 테이블**, 행 안의 엔티티로 조회
   (`pitcher_id`, `batter_id`, 팀, 손, 카운트, 이닝 …)

`test.csv`의 다른 행은 §5·§6에서 금지, 2025 Trackman은 §6에서 금지입니다.

> **이 관문이 가장 많은 후보를 죽입니다.** 예: 16차가 측정한 `rest days`(net 24.8)와
> `prev outing pitches`(net 23.2)는 투수의 **2025 시즌 내 등판 이력**이 있어야
> 계산되는데, test.csv에 `game_date`가 없고(월/요일만) 2025 로그도 없습니다.
> 잠재력은 있으나 **계산 자체가 불가능**합니다.

---

## 2. v9가 이미 쓰고 있는 것

### 2.1 원시 컬럼 — 커버리지 사실상 완전

| | |
|---|---|
| test.csv 컬럼 | 48개 |
| v9 BCAT `b_feats` | 68개 (원시 40 + 파생 28) |
| v9 numeric feats | 101개 |
| **직접 미사용 원시** | `run_top_before`, `run_bot_before` **2개뿐** |

`top_bottom`→`top_T`, `game_type`→`game_F`, `base_state`→`base_state_c`,
`pitcher_id`/`batter_id`→`*_id_num` 으로 전부 인코딩되어 들어갑니다.

### 2.2 조건부 통계 — v9의 SPLIT/ROLE/LOWRANK

v9는 **13개 조건부 split 테이블**을 이미 씁니다 (`edge_*`):

| tag | 엔티티 | 조건 |
|---|---|---|
| `sk` `bl` `ct` | pitcher | 스트라이크 / 볼 / 카운트 |
| `inn` `out` `ro` | pitcher | 이닝 / 아웃 / 주자 |
| `bh` | pitcher | 타자 손 |
| `cth` `ctr` `hro` | pitcher | 카운트×손, 카운트×주자, 손×주자 |
| `bct` `bcth` | **batter** | 카운트, 카운트×손 |
| `pph` | **batter** | 투수 손 |

여기에 ROLE 5종(`p_inning` `p_start` `p_pa_rate` `p_risp` `b_inning` + 상호작용),
LOWRANK rank-2 (`lr_edge` `lr_a0` `lr_a1` `lr_logn`)가 더해집니다.

**즉 "count 상황별 conditional statistic"(요청 4번)은 이미 v9의 핵심 구성요소입니다.**

---

## 3. 선행 측정 결과 — 요청 5개 영역 대조

`d1_potential`은 **잠재력 상한**(같은 시즌 조건부), `d2_forward`는 **배포 가능성**
(이전 시즌 적합 → 다음 시즌 평가). 판정은 d2로 합니다.

### 3.1 요청 1 — pitcher × batter interaction

| | 2023 | 2024 |
|---|---|---|
| d1 잠재력 (net) | 2135.3 | **1295.4 (전체 1위)** |
| **d2 배포 dBSS** | **+0.47 ± 0.95** | **+2.29 ± 1.12** |

**46개 조건 중 두 시즌 모두 양수인 유일한 항목입니다.** 그러나 크기가 +2 수준이고
2.0σ에 못 미칩니다. 잠재력 1295 → 배포 +2.3, **99.8%가 소실**됩니다.

원인은 셀 희소성입니다. d1 2023 기준 `pitcher × batter` K=25,585 셀, **셀당 중앙값
7개 투구**, thin 비율 0.28. 대부분의 매치업이 처음 보는 조합입니다.

**v9 미사용 · 유일한 양수 · 그러나 +2 수준**

### 3.2 요청 2 — pitcher × pitch_type matchup

**d1/d2/d6 어디에도 없습니다. 유일한 미탐색 영역입니다.**

구조적 사실:
- `trackman_history.csv`: 구종 있음(`pitch_type_group`), **결과(control_success) 없음**
- `train.csv`: 결과 있음, **구종 없음** (`asof_pitcher_*_rate` 집계만 존재)
- 15차가 두 파일을 정렬: `work/research/trackman/pitch_match.parquet`
  **809,456행 (train 전체의 54.9%)**

정렬 subset에서는 (구종, 결과) 쌍이 존재하므로 **투수별 구종별 제구 성공률**을
만들 수 있습니다. 2019~2024만 쓰므로 합법입니다.

**그러나 배포 시 결정적 제약**: §6이 **"현재 투구의 실제 구종"** 사용을 금지합니다.
따라서 추론 시 "이 투구가 슬라이더니까 슬라이더 성공률을 쓴다"가 불가능하고,
쓸 수 있는 것은 투수의 **역사적 구종 믹스로 가중평균한 기대값**뿐입니다:

```
E[control | pitcher] = Σ_type  mix_type(pitcher) × rate_type(pitcher)
```

`mix_type`은 test.csv의 `asof_pitcher_{fastball,breaking,offspeed}_rate`로 이미
주어집니다. 그런데 **v9는 그 세 컬럼을 이미 feature로 씁니다.** 즉 새로 더해지는
것은 "구종별 성공률의 투수별 편차"뿐이고, 이는 `asof_pitcher_success_rate`(전체
성공률)와 강하게 겹칩니다.

**v9 중복도 높음 · 미탐색 · 커버리지 54.9%**

### 3.3 요청 3 — batter × pitch_type vulnerability

위와 동일한 정렬 subset에서 계산 가능하나, **타자 쪽 신호가 원래 약합니다.**

| d1 2024 net | |
|---|---|
| `02 batter` | 11.4 |
| `17 batter n` | 8.5 |
| `37 B x Phand` | 47.3 |

| d2 배포 dBSS | 2023 | 2024 |
|---|---|---|
| `batter` | −57.9 | −48.6 |
| `B x count` | −21.8 | −13.8 |
| `B x Phand` | −39.9 | −31.3 |
| `batter n` | −80.0 | −81.1 |

**타자 조건은 전부 큰 음수입니다.** d3_persist도 타자 잔차의 시즌 간 상관이
−0.21 ~ +0.07로 지속성이 없음을 보입니다. 예측 대상이 "투수의 제구 성공"이라
타자 정체성이 본질적으로 약한 것이 자연스럽습니다.

**v9 일부 사용(`bct` `bcth` `pph`) · 배포 테스트 전부 음수**

### 3.4 요청 4 — count 상황별 conditional statistic

**v9가 이미 사용 중**입니다 (`ct` `cth` `ctr` `bct` `bcth`).

추가 조합의 d2 결과:

| cond | 2023 | 2024 |
|---|---|---|
| `P x count` | −0.47 ± 4.08 | −3.96 ± 5.01 |
| `PxB hand x count` | −5.92 ± 6.09 | −5.54 ± 5.94 |
| `situation x count` | −5.54 ± 5.74 | −15.72 ± 5.39 |
| `PxB hand x cnt x out` | −19.02 ± 6.28 | −9.26 ± 5.78 |
| `B x count` | −21.82 ± 3.84 | −13.75 ± 4.78 |

**전부 음수.** 조합을 더 잘게 쪼갤수록 나빠지는 단조 패턴입니다.

### 3.5 요청 5 — time decay historical statistic

부분 탐색됨. `d7_recover`가 λ 축소 그리드를, `d5_deploy`가 λ×K 격자를 돌렸습니다.

| d7_recover (2023→2024) | dBSS | dRes |
|---|---|---|
| λ=0.25 | **+9.27 ± 4.56 (2.0σ)** | −5.47 |
| λ=0.50 | −8.70 ± 9.12 | −17.10 |
| λ=1.00 | −126.34 ± 18.22 | −55.14 |

λ=0.25에서 dBSS는 양수(2.0σ)지만 **dRes는 음수**입니다. 프로젝트 판정 기준이
dResolution이므로 통과가 아닙니다. `d5_deploy`는 방향에 따라 부호가 뒤집힙니다
(2023→2024 λ=0.4에서 −64.3, 2024→2023 λ=0.2에서 +58.9).

**미탐색으로 남은 변형**: v9의 carry 테이블은 **누적 평균**이며 시즌 간 지수감쇠
가중이 없습니다. `d3_persist`가 보여주듯 투수 성공률의 시즌 간 상관은
2022→2023 0.20, 2023→2024 0.67로 **매우 불안정**해서, 최근 시즌에 더 큰 가중을
주는 것이 이론적으로 타당합니다. 다만 d7이 이미 인접 영역에서 부분 기각했습니다.

---

## 4. 미탐색 영역 — d2에 없는 조건

`d1`에 잠재력이 기록되었으나 `d2` 배포 테스트를 거치지 않은 항목:

| cond | d1 2024 net | 배포 가능성 | 비고 |
|---|---|---|---|
| `43 P x month` | **293.9 (2위)** | 계산 가능 (`game_month` 존재) | **시즌 내 폼 궤적** — C-2가 전이 안 됨을 보인 바로 그것 |
| `42 P x game_type` | 95.3 | 가능 | d2에 있음: −90.0 / −49.9 (기각) |
| `44 PxB hand x count x outs` | 30.1 | 가능 | d2에 있음: −19.0 / −9.3 (기각) |
| `28 P fastball rate` | 8.5 | v9 이미 사용 | |
| `30 team (B)` | 19.8 | d2: −2.0 / −6.0 (기각) | |

**`P x month`가 유일하게 큰 잠재력 + 미탐색입니다.** 그러나 월은 시즌 내 시간
인덱스이고, C-2가 "이득의 소재지는 시즌 내 축적, 100구 미만 구간은 음수"임을
보였습니다. 2025 test 행에 `game_month`는 있지만 그 투수의 2025년 월별 폼은
알 수 없고, 2019~2024 월별 테이블은 **연도별 폼이 다르므로 전이 가능성이 낮습니다.**

---

## 5. Top 20 후보 — 예상 impact 순

`impact` = 2024 fold 기준 예상 dResolution(3-seed 평균). 근거는 각 행에 표기.
**leakage** L = 위험, 空 = 안전. **v9중복** ●=높음 ◐=중간 ○=낮음.

| # | 후보 | 소스 | impact | L | v9중복 | 근거 |
|---|---|---|---|---|---|---|
| 1 | pitcher×batter 매치업 (강한 축소, K≥2000) | train | **+2 ~ +5** | | ○ | d2 유일 양수 +0.47/+2.29 |
| 2 | 구종믹스 가중 투수 제구율 (정렬 subset) | tm+train | **0 ~ +4** | | ● | 미탐색, mix는 v9 기사용 |
| 3 | 시즌 지수감쇠 carry (λ≈0.25) | train | **0 ~ +3** | | ● | d7 λ=0.25 dBSS +9.3 / dRes −5.5 |
| 4 | pitcher×batter 손조합 잔차 | train | 0 ~ +2 | | ◐ | d1 15.4 / d8 λ=0.25 +4.5 |
| 5 | 투수 등판 간격 **역사적 평균** | tm | 0 ~ +2 | | ○ | d6 net 24.8/0.3, 시즌 불일치 |
| 6 | 구종별 편차 (fastball−offspeed 제구율차) | tm+train | 0 ~ +2 | | ◐ | 미탐색 |
| 7 | `run_top_before`/`run_bot_before` 분리 | train | 0 ~ +1 | | ● | v9는 total만 사용 |
| 8 | 타자 구종 취약성 | tm+train | −2 ~ +1 | | ◐ | 타자축 d2 전부 음수 |
| 9 | 투수 arsenal 다양성 지수 (엔트로피) | tm | −2 ~ +1 | | ● | G0에서 mix 4개 null |
| 10 | count×foul 조건부 | tm | −3 ~ +1 | **L** | ○ | d6 net 0.66/19.4, **foul은 현재 PA 진행 정보** |
| 11 | `P x month` | train | −5 ~ +3 | **L** | ○ | d1 293.9지만 시즌 내 궤적 |
| 12 | 투수 rank-3 lowrank 확장 | train | −3 ~ +2 | | ● | v9 rank-2 기사용 |
| 13 | 팀 배터리(포수) 대리 | — | — | | | **포수 컬럼 없음, 계산 불가** |
| 14 | `P x game_type` | train | −50 | | ○ | d2 −90.0 / −49.9 |
| 15 | `situation × count` | train | −16 | | ● | d2 −5.5 / −15.7 |
| 16 | `P x inning` | train | −18 | | ● | d2 −17.0 / −18.5 |
| 17 | `B x count` | train | −14 | | ● | d2 −21.8 / −13.8 |
| 18 | `P x Bhand` | train | −24 | | ● | d2 −27.2 / −24.1 |
| 19 | `B x Phand` | train | −31 | | ◐ | d2 −39.9 / −31.3 |
| 20 | `batter n` 조건부 | train | −81 | | ● | d2 −80.0 / −81.1 |

**#13은 계산 불가**입니다 — 포수 식별자가 train/test/trackman 어디에도 없습니다.
목록에 남긴 이유는 "왜 없는지"를 기록하기 위해서입니다.

### leakage 표시 상세

| 후보 | 위험 |
|---|---|
| #10 count×foul | `foul` 카운트는 **현재 타석의 진행 상황**입니다. 현재 투구 이전 파울만 세면 합법이나, trackman 정렬 없이는 train.csv에서 계산 불가하고 정렬 subset은 54.9%뿐 |
| #11 P×month | 월별 테이블을 **검증 시즌 포함**으로 만들면 즉시 누수. 시즌<S 로만 적합해야 하며, 그러면 잠재력 293.9는 거의 사라짐 |
| 전 후보 공통 | 조건부 테이블은 **반드시 시즌<S 에서만 적합**. v9 `build_split_tables(future_seasons=(S,))` 규율을 그대로 따라야 함 |

---

## 6. 권고

**새 feature 탐색을 권하지 않습니다.**

Top 20 중 예상 impact가 양수인 것은 6개이고, 최댓값이 **+5**입니다. 프로젝트
채택 기준은 `dRes ≥ 20`(STRONG), `< 10`(WEAK)입니다. **1위 후보조차 WEAK
구간에 들어가지 못합니다.**

이유는 구조적입니다.

1. **원시 컬럼 여백이 없습니다.** 48개 중 미사용 2개, 그마저 동어반복입니다.
2. **조건부 통계 공간은 46개 조건으로 전수 탐색되었고**, 배포 테스트에서 두 시즌
   모두 양수인 것은 `pitcher × batter` 하나(+2 수준)뿐입니다.
3. **잠재력과 배포 가능성의 괴리가 일관됩니다** — `pitcher × batter` 1295 → +2.3,
   `pitcher` 80 → −58. 조건부 잠재력은 시즌 내 정보이고, 시즌을 넘으면 사라집니다.
   C-2가 index 분해로 같은 결론(이득은 시즌 내 100구 이상 축적에만 존재)에
   도달했고, G0가 historical profile에서 또 같은 결론에 도달했습니다.

**굳이 하나만 시도한다면 #1 (pitcher×batter, 강한 축소)** 입니다. d2에서 유일하게
두 시즌 양수이고 v9 미사용이며, 셀 희소성이 원인이므로 축소 강도(K)를 크게
올리면 +2.3보다 나아질 여지가 있습니다. 비용은 반나절, 게이트는
**3-seed dRes ≥ +10 AND dBSS > 0**를 권합니다. 그 아래면 채택 이득이 없습니다.

**그 외에는 `work/submit_v9.zip`이 최종 제출본이라는 기존 결론이 유효합니다.**

---

## 부록 — 참조한 선행 산출물

| 경로 | 내용 |
|---|---|
| `work/research/resolution_discovery/d1_potential.csv` | 46개 조건 잠재력 (시즌별) |
| `work/research/resolution_discovery/d2_forward.csv` | 16개 조건 배포 temporal 테스트 |
| `work/research/resolution_discovery/d3_persist.csv` | 엔티티 잔차 시즌 간 지속성 |
| `work/research/resolution_discovery/d5_deploy.csv` | λ×K 배포 격자 |
| `work/research/resolution_discovery/d6_trackman.csv` | trackman 파생 조건 (foul/rest/pitch_of_pa) |
| `work/research/resolution_discovery/d7_recover.csv` | 투수 잔차 λ 복원 |
| `work/research/resolution_discovery/d8_bhand.csv` | 타자 손 조건 |
| `work/research/trackman/pitch_match.parquet` | train↔trackman 정렬 809,456행 (54.9%) |
| `work/research/trackman/t6_lawful.log` | trackman 프로필 60피처 null |
| `experiments/deep_learning_state/g0_ceiling_S2024*.csv` | historical profile G0 FAIL |
