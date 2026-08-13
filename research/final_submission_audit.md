# 최종 제출본 감사 — `work/submit_v9.zip`

작성 2026-08-13 · 원본 무수정 · 모든 실행은 scratch 사본에서만

---

## 1. 결론

**`work/submit_v9.zip`이 최종 제출본입니다.**

```
sha256  e30a19e24b73b3ba22547b6fc615e30ebde1820f7b7558784a447c75ac8575b1
크기    25,965,157 B      mtime  Aug 9 02:01:50 2026
백테스트 (model.pkl 자체 기록)   y2024 BSS 917.2   y2023 403.1   est_lb 1066.7
```

이 세션에서 다섯 갈래 연구를 수행했고 **전부 배포 가능한 이득을 내지 못했습니다**(§5).
제출본을 바꿀 근거가 없습니다.

---

## 2. 왜 이것인가 — `work/submit/`이 아닌 이유

두 곳에 제출 번들처럼 보이는 것이 있습니다. **다릅니다.**

| | `work/submit_v9.zip` | `work/submit/` |
|---|---|---|
| `model.pkl` | 71,851,533 B · **version 9** | 34,728,872 B · **version 6** |
| `weights` | `{A7 .25, A9 .30, BCAT .45}` | **`None`** |
| `b_cat_files` | `['cat_0.cbm','cat_1.cbm']` | **`None`** |
| `cat_*.cbm` | 포함 | **없음** |
| 실행 | 정상 | `pkg['weights'].get(...)` 에서 **NoneType 오류** |

`work/submit/`은 v6 시절 작업 디렉터리 잔재입니다. zip 내부 `model/model.pkl`은
`work/model_v9.pkl`과 해시가 동일(`30cfd2f3c3b55d88`)해 v9 본체가 맞습니다.

> **제출 시 `work/submit/`을 압축하지 말고 `work/submit_v9.zip`을 그대로 올리십시오.**

---

## 3. 재현 실행 결과

원본은 열지 않았습니다. 사본(`copy.zip`, 해시 `e30a19e24b73b3ba` 원본과 일치)을
scratch에 만들어 거기서만 실행했습니다.

### 3.1 번들 구성

```
requirements.txt      33 B      lightgbm==4.7.0 / catboost==1.2.10
script.py         37,885 B
model/model.pkl 71,851,533 B
model/cat_0.cbm   771,940 B
model/cat_1.cbm   836,448 B
                 ---------
                73,497,839 B    5 files      CRC 무결성 OK
```

### 3.2 격리 환경

`--system-site-packages` 없이 새 venv를 만들어 `requirements.txt`만 설치했습니다.
**catboost가 pandas·numpy·scipy를 전이 의존성으로 끌어오므로 번들은 선언된
의존성만으로 자립합니다.**

```
catboost 1.2.10 · lightgbm 4.7.0 · numpy 2.5.2 · pandas 3.0.5 · scipy 1.18.0
```

### 3.3 실행

```
=== KBO control-success inference ===
package v9 | shrink=1.10
test: data/test.csv (5, 48)
  A7 mean=0.4839
  A9 mean=0.4765
  BCAT mean=0.4493
weights [0.25 0.3 0.45]
predictions: mean=0.4661 std=0.0543 min=0.3965 max=0.5339
wrote output/submission.csv (5, 2)
```

exit 0, 약 11초. `weights [0.25 0.3 0.45]`가 `model.pkl`의 값과 일치하고 A7·A9·BCAT
세 계열과 CatBoost 두 파일이 모두 로드됐습니다.

### 3.4 재현성 — bit 동일

서로 다른 시점에 **별도로 만든 두 개의 격리 venv**에서 실행한 결과:

```
이번 실행 (신규 venv)  e1089924b7e5e5cb0a776442f55d49f0...
이전 실행 (별도 venv)  e1089924b7e5e5cb0a776442f55d49f0...
-> bit 동일
```

---

## 4. submission.csv 검증

```
row_id,control_success
TEST_000001,0.4121514585151711
TEST_000017,0.3964682115177446
TEST_000213,0.47387586382484426
TEST_005332,0.5143482217492542
TEST_035185,0.5338819544031642
```

| 항목 | 결과 |
|---|---|
| 생성 성공 | PASS |
| 컬럼 == `sample_submission.csv` | PASS (`row_id`, `control_success`) |
| row 수 == test.csv | PASS (5 = 5) |
| row_id 1:1 매칭 | PASS |
| row_id 순서 동일 | PASS |
| row_id 중복 없음 | PASS (0건) |
| prediction NaN 없음 | PASS (0건) |
| prediction (0,1) 범위 | PASS `[0.396468, 0.533882]` |
| dtype | PASS float64 |

mean 0.4661 · std 0.0543.

> 참고: `test.csv`는 **형식 확인용 5건 샘플**입니다. 실제 평가 데이터는 비공개이며
> 평가 서버에서 교체됩니다. row_id가 `TEST_000001`~`TEST_035185`로 띄엄띄엄인 것으로
> 보아 실제 평가셋은 최소 35,185행입니다.

---

## 5. 규정 준수

주최측 `data_description.md` §5·§6 원문 기준.

### 5.1 §5 각 행 독립 예측 — 실증

> 평가 서버에서 실제 `test.csv` 전체가 주어지더라도, 참가자는 `test.csv`의 다른
> 행을 이용해 현재 행의 피처를 만들 수 없습니다.

같은 번들을 두 방식으로 실행해 예측값을 대조했습니다.

```
5행 한번에  vs  각 행을 1행짜리 test.csv 로 단독 실행 5회
  최대 절대차 0.000e+00     bit 동일 True
```

**다른 행의 존재·개수·순서가 예측에 전혀 영향을 주지 않습니다.** 코드를 읽은
추론이 아니라 실행 결과입니다.

정적 분석으로도 확인했습니다 — `main()`에서 도달 가능한 함수는 전부 동결 테이블
merge와 자기 행 산술뿐이고, 집계 함수(`build_carry_tables`, `build_split_tables`,
`build_role_tables`, `build_lowrank_tables`, `_lr_fit`, `_lr_matrix`,
`b_compute_league_rate_asof`)는 **전부 도달 불가**입니다. 단일 파일에 학습·추론
코드가 함께 있어 남은 빌드 전용 코드이며 실행되지 않습니다.

### 5.2 §6 사용 금지 정보 — 구조적으로 불가능

`script.py`의 파일 I/O가 정확히 4개뿐입니다.

```
768  open + pickle.load   →  model/model.pkl
773  read_csv             →  test.csv
840  load_model           →  model/cat_0.cbm, cat_1.cbm
858  to_csv               →  output/submission.csv
```

| 금지 항목 | 판정 |
|---|---|
| **2025년 Trackman 데이터** | **trackman 파일을 열지 않음** |
| **현재 투구의 Trackman 측정값** | 동일 — 접근 자체가 없음 |
| 현재 투구의 위치·판정·결과·구종 | test.csv 48컬럼에 해당 컬럼 없음 |
| 현재 투구 이후 확정 정보 | 입력이 주최측 공식 48컬럼뿐 |
| test 내부 행 기반 피처 | §5.1 실증으로 배제 |

15차 오라클(+342~+383)의 정체가 "현재 투구 물리량"이었는데, 제출본은 trackman을
아예 읽지 않으므로 그 경로가 원천 차단됩니다.

### 5.3 허용 데이터 범위

`train.csv`에서 만든 동결 룩업 테이블(carry / split / role / lowrank)과 평가
환경의 `test.csv`만 사용합니다. §6의 허용 목록 안입니다.

§5가 명시 허용한 `asof_*` 컬럼 19개도 사용합니다 — "각 행의 투구 직전 시점까지의
과거 기록만으로 계산된 공식 입력 피처".

---

## 6. 탐색 종료 근거

다섯 갈래를 수행했고 전부 배포 이득 없음으로 종료됐습니다.

### Phase A — GRU sequence teacher
시즌 내 투수 상태는 **실재하고 큽니다**: 2024 dRes **+136.6**, 2023 **+542.3**
(3 seed 안정). 16차가 +45로 추정한 것보다 큽니다.

**그러나 배포 불가**입니다. 입력이 "같은 시즌 직전 32구"인데, 2025 창을 만들려면
다른 test 행을 읽어야 하고(§5 금지) 2025 투구 로그는 어디에도 없습니다
(§6이 2025 Trackman 금지, 허용 목록은 2019~2024).

### Distillation (Phase C)
teacher를 단일 행 student로 증류. temporal 복원율 **38.4%**(3 seed, R² 0.379로
매우 안정)로 신호는 전이됩니다. 그러나 복원한 것의 **46%가 v9와 겹쳐**
v9+student 블렌드가 **dBSS +0.1 (0.01σ)** 로 소멸했습니다.

index 분해가 원인을 특정합니다 — teacher의 이득은 **시즌 내 100구 이상 축적된
행**에만 있고(+211/+136), 100구 미만 구간에서는 오히려 음수(−732~−67)입니다.

### Historical pitcher embedding (G0)
Trackman 2019–2023 투수 프로필. same-season ceiling에서조차 3-seed FAIL
(plus −6.7 ± 7.2, basic +7.6 ± 10.7, 전부 유의성 없음). basic seed42의 +16.3은
seed variance로 인한 false positive였고, 더 학습시키면 **악화**됩니다(−10.3, 2.2σ).

### pitcher × batter interaction
46개 조건 중 유일하게 두 시즌 부호가 양수였던 마지막 후보. 하이퍼파라미터를
2023에서 고정하고 2024를 한 번 연 결과 **GATE FAIL** (M2 B−A: dRes −13.8,
dBSS −9.5, 0.90σ). 페어 커버리지 44.8%, 셀당 중앙값 2구가 한계였습니다.

### Calibration audit
`decomp`가 분위 구간으로 나누므로 **단조 변환은 Resolution을 정확히 보존**합니다
(temperature·Platt의 dRes가 모든 실행에서 정확히 +0.0). 따라서 상한은 현재
Reliability이고, 그것은 **전이되지 않습니다** — 2023→2024 temperature −105.8,
Platt −100.4, isotonic −141.1 (6/6 음수, 전부 8σ 이상). 두 시즌 gap 상관이
**−0.60**, 부호 일치가 20구간 중 4구간뿐입니다.

### 공통 구조
다섯 결과가 한 지점을 가리킵니다. **정보는 시즌 내에 존재하고, 시즌 경계를 넘지
못합니다.** 조건부 잠재력과 배포 가능성의 괴리가 일관됩니다
(`pitcher × batter` 1295 → +2.3, `pitcher` 80 → −58).

### 추가 탐색을 중단하는 이유
- 원시 컬럼 여백 없음 — test.csv 48개 중 미사용은 `run_top_before`,
  `run_bot_before` 2개뿐이고 `run_total_before` + 점수차 2개와 동어반복
- 조건부 통계 공간 전수 탐색 완료 — 16차 46개 조건
- feature 조사 Top 20의 예상 최댓값이 **+5**, WEAK 기준 +10에 미달
- calibration 방향은 정리(theorem)로 닫힘

새 정보원 없이 조건 변수를 더 조합하는 것은 기대값이 음수입니다.

---

## 7. 무결성

감사 전 기록한 baseline과 감사 후 대조 — **7/7 OK**, mtime도 불변.

```
30cfd2f3c3b55d88  work/model_v9.pkl                                OK
e30a19e24b73b3ba  work/submit_v9.zip                               OK   mtime Aug 9 02:01:50
b5f77dc3a3e4a437  work/submit/script.py                            OK
8ea5d84c13686918  work/submit/model/model.pkl                      OK
3db9007cc31a94d6  work/submit/requirements.txt                     OK
89b2500cb3748266  규은누님_제공파일 복사본/.../cat_0.cbm             OK
aab4b417e9da1701  규은누님_제공파일 복사본/.../cat_1.cbm             OK
```

모든 압축 해제·환경 구축·실행은 scratch 사본에서만 이루어졌고, 저장소에는
`data/`·`work/` 파일이 하나도 추적되지 않습니다.

---

## 8. 제출 절차

```
work/submit_v9.zip  을 그대로 업로드
```

재압축·재생성하지 마십시오. 현재 zip이 CRC 무결성과 실행 재현성을 모두 통과한
바로 그 파일입니다.
