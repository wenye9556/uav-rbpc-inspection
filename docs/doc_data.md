# 当前真实数据、派生统计与证据边界

本文件只描述当前数据口径。数据生成/修订的版本历史不再作为主流程保留；已废弃数据路径和旧 predictor 不得用于 formal 结果。

## 1. 当前正式数据集合

### 1.1 风机

```text
data\turbines_Rodsand_II_clean.csv
```

当前 Step13 运行已成功加载 Rodsand II 风机 90 台。论文主 certified instance 使用 10-turbine 子实例；后续规模实验可改变 `n_turbines`，且该值始终属于有限模型身份。

### 1.2 历史天气

```text
weather\weather_Rodsand_II.csv
weather\waves_Rodsand_II.csv
```

当前运行观测：

- wind rows = 2,928；
- wave rows = 2,928；
- nominal weather 来自真实历史 ERA5/CMEMS 时间序列；
- formal WeatherAmbiguity 不是由相邻 reanalysis 差分临时估计，而是读取经验证的 train-only weather moments。

### 1.3 AIS 轨迹

正式目标：

```text
tracks\track_219018788.csv
MMSI = 219018788
```

已验证：

- 879,937 行；
- 起始日期 2025-03-01；
- 总时长约 122 天；
- 当前 formal 任务窗固定 `track_start_min=410.233333`。

第二条真实轨迹：

```text
tracks\track_219028973.csv
MMSI = 219028973
```

用于数据生成/对照，但 formal Xi ambiguity 不允许跨 MMSI pooling。

## 2. Xi：真实 AIS 船位预测误差

### 2.1 当前 predictor contract

```text
predictor = cv_noleak
predictor_contract = cv_noleak_backward_window_epoch_seconds
timestamp_epoch_contract = utc_datetime64_ns_to_epoch_seconds
horizons = [5,10,15,20,25,30]
sample_overlap_policy = nonoverlap
moments_source = train
```

Formal Step13 在提供 `--xi-train-samples` 时，从实际 train rows 按 concrete MMSI 重建 Xi ambiguity，而不是把 pooled `xi_moments_caseB.csv` 当正式主对象。

### 2.2 Frozen split

Formal train：

```text
tracks\xi_samples_caseB.csv
```

Weather coherent 重建后的联合 validation/test：

```text
tracks\weather_v17_1_stage\xi_weather_samples_validation_caseB.csv
tracks\weather_v17_1_stage\xi_weather_samples_test_caseB.csv
```

目标 MMSI=219018788 过滤后的已验证数量：

| split | n |
|---|---:|
| train | 48,761 |
| validation | 16,101 |
| test | 16,064 |

24 个 ambiguity cells = 6 horizons × 4 vessel states。

Formal 状态：

```text
xi_mmsi = 219018788
cross_pool = False
```

### 2.3 Cell counts

顺序为 `low / DP / straight / turn`。

| h | train | validation | test metadata |
|---:|---|---|---|
| 5 | 1834 / 13735 / 2824 / 1792 | 702 / 4726 / 847 / 447 | 508 / 4871 / 902 / 430 |
| 10 | 937 / 6734 / 1401 / 919 | 351 / 2322 / 422 / 215 | 265 / 2368 / 445 / 221 |
| 15 | 631 / 4450 / 926 / 589 | 238 / 1498 / 287 / 141 | 190 / 1538 / 301 / 134 |
| 20 | 472 / 3279 / 692 / 462 | 177 / 1114 / 213 / 104 | 135 / 1124 / 217 / 123 |
| 25 | 356 / 2599 / 583 / 343 | 152 / 867 / 156 / 91 | 98 / 894 / 182 / 88 |
| 30 | 333 / 2103 / 454 / 313 | 116 / 704 / 146 / 65 | 102 / 707 / 151 / 70 |

当前 24 个 train cells 全部 `sample_rule=raw_state` 且 `n=n_raw_state`；没有触发 `merged_low_speed_pair`。不能根据 validation/test 样本量事后改变 state definition。

### 2.4 Temporal disjointness + purge

目标 MMSI 的时间边界：

```text
train_end        2025-05-12 14:36:34 UTC
validation_start 2025-05-12 15:11:34 UTC
gap              35 min

validation_end   2025-06-06 05:57:42 UTC
test_start       2025-06-06 06:32:42 UTC
gap              35 min
```

最大 horizon = 30 min，因此正式：

```text
--holdout-purge-min 30
```

已通过 Step20 temporal disjointness + purge gate。旧的 `purge=60` 失败不表示 split 错误。

### 2.5 Frozen Xi hash

```text
tracks\xi_samples_caseB.csv
SHA256 = 4ad41474d4ffa06682e77f00d142dad6c9a4281b9cfa3d2f9baa71b7dcc6ffb6
```

Weather moments 必须记录并匹配该实际 train source SHA。

## 3. WeatherAmbiguity：coherent no-leak residual

### 3.1 Current formal contracts

```text
predictor = weather_speed_primary_coherent_noleak
predictor_contract = weather_backward_linear_speed_primary_coherent_epoch_seconds_v2
truth_contract = era5-reanalysis-wind+cmems-historical-wave-hourly-linear-truth-v1
data_contract = real-history-noleak-weather-residuals-global-weather-nonoverlap-v3-coherent-wind
synthetic_weather = False
operational_forecast_archive = False
```

正式含义是“causal no-leak predictor 对真实历史 truth proxy 的 residual”，不声称拥有 ECMWF operational forecast archive。

### 3.2 为什么旧 predictor 被替换

旧 predictor 独立预测 scalar speed 与 wind vector，导致：

```text
hypot(pred_e,pred_n) != pred_speed
```

旧 train residual N=10,356 时：

- mean absolute incoherence = 0.01996198 m/s；
- median = 0.00438；
- p95 = 0.08786；
- p99 = 0.22387；
- max = 1.285268；
- negative scalar-speed predictions = 4；
- min scalar speed = -0.13542849 m/s。

这不是 ULP 噪声，属于结构性统计中心不一致。

### 3.3 Train-only model choice

候选：

- CURRENT；
- VECTOR_PRIMARY；
- SPEED_PRIMARY_CLIPPED。

Train-only 对比：

| model | vector_rmse | component_rmse | speed_rmse | speed_mae | speed_bias | negative_speed | coherence_max |
|---|---:|---:|---:|---:|---:|---:|---:|
| CURRENT | 0.202144 | 0.142937 | 0.137713 | 0.087234 | -0.000046 | 4 | 1.285268 |
| VECTOR_PRIMARY | 0.202144 | 0.142937 | 0.138367 | 0.087318 | 0.019916 | 0 | 0 |
| SPEED_PRIMARY_CLIPPED | 0.196415 | 0.138886 | 0.137558 | 0.087207 | -0.000019 | 0 | ~3.55e-15 |

最终选择 `SPEED_PRIMARY_CLIPPED`，且选择只使用 train residual，不使用 validation/test outcome。

### 3.4 Current predictor construction

1. scalar wind speed：causal backward-linear extrapolation；
2. `pred_speed=max(0,pred_speed_raw)`；
3. raw `(e,n)` extrapolation 只提供方向；
4. vector 缩放到 `pred_speed`；
5. `pred_speed≈0` 时 vector=[0,0]；
6. speed>0 且 raw vector norm≈0 → fail-closed；
7. predicted Hs<0 → fail-closed；
8. 强制 scalar/vector magnitude coherence。

## 4. Weather-only regeneration

使用：

```text
step7_compute_xi.py --weather-only-existing-splits
```

目的：

- 保留 frozen Xi split；
- 只重建 weather-derived products；
- 不读取 final-test performance；
- 前后验证 frozen Xi hashes unchanged。

当前正式目录：

```text
tracks\weather_v17_1_stage\
```

文件：

```text
weather_residual_samples_caseB.csv
weather_moments_caseB.csv
xi_weather_samples_validation_caseB.csv
xi_weather_samples_test_caseB.csv
_weather_regeneration_manifest.json
```

审计：

| split | input | output | dropped |
|---|---:|---:|---|
| train | 86,701 | 86,656 | no_history=45 |
| validation | 28,528 | 28,528 | 0 |
| test | 28,261 | 28,224 | no_truth=37 |

其余异常计数为 0。

Train weather residual：

```text
rows = 10,356
coherence_max = 5.329070518200751e-15
negative_wind_speed = 0
min_wind_speed = 0.0
negative_Hs = 0
min_Hs = 0.03224285
```

`min_wind_speed=0` 是非负支持投影的合法结果，不需要人为设置正下限。

## 5. Step20 formal preflight status

已通过：

```text
formal-publication-preflight-v3-coherent-weather-power-audit-metadata-only-test
PRECHECK = PASS
```

关键数据/provenance gate：

- track / Xi train / stage validation / stage test / weather moments 存在；
- concrete MMSI=219018788；
- train/validation/test MMSI exact；
- three split hashes distinct；
- `cross_pool=False`；
- purge covers max horizon；
- all train horizons available in validation/test；
- weather contract PASS；
- weather moments bound to actual Xi train SHA PASS；
- exact risk budget PASS；
- no legacy E1-final-test consumption；
- `E2_final_test_state=not yet consumed`。

Preflight 对 final-test 只允许 metadata/schema/state 检查，不允许提前消费 outcome。

## 6. Validation statistical-power boundary

当前 formal selection 使用 simultaneous UCB，而不是经验 failure rate 点估计。

Zero-failure 下的样本量门槛：

| sorties | n required |
|---:|---:|
| 1 | 740 |
| 2 | 911 |
| 3 | 1,011 |
| 4 | 1,082 |
| 5 | 1,138 |
| 6 | 1,183 |
| 7 | 1,221 |
| 8 | 1,254 |

达到门槛的 validation cells：

```text
m=1: 6/24
m=2: 4/24
m=3: 4/24
m=4: 4/24
m=5: 3/24
m=6: 3/24
m=7: 3/24
m=8: 3/24
```

因此 optimization exact certificate 与 validation statistical certificate 必须分开报告。不得通过修改 confidence/epsilon/state pooling/route set 来制造 validation PASS。

## 7. 数据与正式优化证书的边界

数据 provenance 通过，只是 formal physical/global certificate 的必要条件之一。最终 `global_certificate_available=True` 还要求：

- 当前 finite model contract；
- route-universe provenance；
- route semantic invariance；
- exact pricing/complete materialized universe；
- branch completeness；
- strict resource audit；
- binary64 numeric contract；
- proof-to-code contract；
- source/instance/model/algorithm hashes。

反之，BPC exact certificate 也不证明：

- IID independence；
- terminal acquisition reliability；
- continuous-time real-world global optimality；
- mission-start nonanticipativity；
- final-test statistical reliability。

## 8. 数据冻结纪律

除非发现可复现的具体数据 bug，否则：

1. 不重切 frozen Xi train/validation/test；
2. 不改变 formal MMSI；
3. 不跨船 pooling；
4. 不根据 validation/test 合并 states；
5. 不重新选择 weather predictor；
6. 不提前查看 final-test outcome；
7. 不用旧根目录 weather-derived 文件替代 `weather_v17_1_stage`；
8. 不把 mechanism synthetic fallback 标成 formal evidence。
