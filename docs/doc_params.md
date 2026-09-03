# 参数口径

本文件只记录当前 Python 实际参数、默认值、正式约束和实验调度参数。旧版本迁移参数与已删除 CLI 不再保留。

## 1. Current formal defaults

### 1.1 Mission discretization

- 作业窗口：`360 min`；
- 单风机巡检：`5 min`；
- 默认 `max_stops=4`；
- 默认 `dtau_min=5`；
- 默认 `deck_delta_min=2.5`；
- 默认 `landing_clear_min=1.0`；
- quick-inspection capacity = 1；
- swap stations = 1；
- 当前 Rodsand II Xi horizons：`5,10,15,20,25,30 min`。

这些都属于当前 finite instance/parameter contract 的组成部分；改变它们会改变模型身份。

### 1.2 Formal semantic defaults

```text
soc_correction = geo2d
soc_risk_allocation = optimized
time_recourse = wait_and_speed
battery_reuse_mode = exact_soc
deck_mode = interval
recovery_predictor = cv_noleak
pool_h = pareto
weather_drcc = on
weather_alignment = timestamp
solver_mode = exact-branch-price-cut
pricing_mode = r-bpc
```

Formal 不允许 synthetic fallback。

## 2. UAV profiles

当前 `step9_model.UAV_PROFILES`：

| key | label | B_k Wh | hover W | W_max m/s | landing wind m/s | swap min | launch min | source |
|---|---|---:|---:|---:|---:|---:|---:|---|
| S | DJI Matrice 30T | 263.2 | 439 | 12.0 | 12.0 | 4.0 | 2.5 | mixed OEM/scenario |
| M | Autel Alpha | 474.0 | 748 | 12.0 | 10.7 | 4.0 | 3.0 | mixed OEM/scenario |
| L | DJI Matrice 400 | 977.0 | 1106 | 12.0 | 12.0 | 5.0 | 3.0 | mixed OEM/scenario |
| M350 | DJI M350 RTK legacy baseline | 526.4 | legacy default | 12.0 | 12.0 | 4.0 | 2.5 | legacy baseline |

E1 正式 UAV 轴是 `S,M,L`。`M350` 保留用于回归/历史算法 benchmark，不应与当前 S/M/L 选型混称。

UAV 参数中仍有 scenario 假设；没有可靠 OEM 数据时不得臆造 battery mass/acquisition cost。

## 3. 风险预算

当前 `Params`：

| failure event | parameter | budget |
|---|---|---:|
| energy | `eps_E` | 0.0125 |
| time | `eps_T` | 0.0125 |
| wave gate | `eps_cap` | 0.0050 |
| wind gate | `eps_gate` | 0.0050 |
| airspeed | `eps_air` | 0.0050 |
| docking reserve | `eps_dock` | 0.0025 |
| stern escort | `eps_escort` | 0.0025 |
| active total | — | 0.0450 |
| mission upper budget | `mission_failure_budget` | 0.0500 |

`eps_cap` 是历史变量名，当前正式含义为 wave-height gate tail risk，不是 acquisition error。

正式模型允许 conservative unallocated risk budget，但不允许 active components 超过 mission budget。不得为了结果更好事后修改 epsilon。

## 4. Terminal recovery boundary

Recovery horizon `h` 是 route decision。名义 recovery point 由同一 `h` 的 ship prediction 给出，位置不确定性由 `xi_h` 描述。

当前有限模型没有独立：

```text
acquisition_radius_m
eps_acq
terminal sensor covariance
```

因此不能通过填 0 或 synthetic sample 声称 terminal acquisition reliability 已建模。

## 5. Exact algorithm parameters

论文正式 API：

```text
--solver-mode exact-branch-price-cut
--pricing-mode r-bpc
```

Step13 的 paper-facing default 也是 `r-bpc`；Step12 库级旧默认保留用于兼容，但论文/正式复现实验应显式传入 `r-bpc`。该 alias 映射到当前已验证的 resource-aware exact BPC 实现。

Formal exact pricing 也不接受 beam width、neighbor count、label cap、candidate cap 等会截断完整路线空间的参数。

R-BPC 的 generated-column exact primal recovery 默认由 `r-bpc` 模式启用；`--archive-primal-recovery-time-limit-s` 只限制每次 restricted recovery 的局部预算。该预算不会把 restricted archive 的 UB 转换为 full-space bound。

`exact-mip` 是 argparse 可选值，但当前实现缺少等价 black-box physics/DRCC encoding，formal 求解会 fail-closed。

### 5.1 Gap targets

Step13 defaults：

```text
coverage_gap_target_abs = 0
energy_gap_target_rel = 0.0
energy_gap_target_abs_wh = 1e-6
```

Formal publication knee 常显式设置：

```text
coverage_gap_target_abs = 0
energy_gap_target_rel = 0
energy_gap_target_abs_wh = 0
```

Gap target 是 anytime stopping condition，不是 exact certificate 的替代条件。只要 global incumbent/bound 仍有正缺口，不能把结果叫 lexicographic optimal。

### 5.2 Deadline

`time_limit_s` 控制统一 cooperative wall-clock deadline。Timeout 不改变模型，只使结果可能保持 unresolved/anytime bounds。

## 6. Strict numeric contracts

当前正式数值语义：

```text
physical_numeric_contract = strict-binary64-feasibility-no-positive-tolerance-v2
```

路线/资源可行不使用正 tolerance 扩域：

- airspeed；
- fixed-touchdown time；
- mission window；
- deck interval；
- UAV ready time；
- battery SOC。

Battery SOC demand 将输入 binary64 转为 `Fraction.from_float` 精确累计，再严格比较 usable capacity。

Formal Xi covariance：

- `sigma_ee/sigma_en/sigma_nn` 视为 binary64-as-real；
- 要求 exact symmetric 2×2 PSD；
- 不使用 scale-relative PSD tolerance；
- horizon 必须精确命中 5-minute grid subset；
- 不允许 nearest/round snapping。

Weather covariance 使用其独立正式合同；正常线性代数 ULP 对称误差不要求逐 bit 对称，但 horizon mapping 必须 exact。

## 7. Current formal data parameters

```text
track_mmsi = 219018788
track_start_min = 410.233333
holdout_purge_min = 30

xi_train = tracks\xi_samples_caseB.csv
validation = tracks\weather_v17_1_stage\xi_weather_samples_validation_caseB.csv
final_test = tracks\weather_v17_1_stage\xi_weather_samples_test_caseB.csv
weather_moments = tracks\weather_v17_1_stage\weather_moments_caseB.csv
```

Formal weather contracts：

```text
weather_speed_primary_coherent_noleak
weather_backward_linear_speed_primary_coherent_epoch_seconds_v2
real-history-noleak-weather-residuals-global-weather-nonoverlap-v3-coherent-wind
```

## 8. E1 parameters

Current Step13 defaults：

```text
e1_uavs = S,M,L
fleet_ks = 1,2,3,4,5,6,7,8
e1_batteries = 0,1,2,3,4,5,6,7,8
e1_b_auto = on
e1_b_cap = 8
e1_sat_patience = 3
knee_frac = 0.95
knee_order = BK
selection_metric = safe_per_inventory_kWh
e1_frontier_time_limit_s = 120
e1_certify_time_limit_s = inherit time_limit_s if omitted
formal_warmstart_seconds = 60
```

论文主 certified instance 使用 10 turbines；资源前沿实验可按研究设计改变 `n_turbines`。常用 `fleet_ks=1,2,3` 和基础 `e1_batteries=0,1,2,3,4`，再由 `e1_b_auto=on` 延伸到结构 cap。

Formal battery extension 还受：

```text
B <= min(e1_b_cap, |I|)
```

结构约束。因为每条 selected route 非空且 turbine packing 限制 route 数不超过 `|I|`，`B>|I|` 不再扩张可行集合。

### 8.1 E1 staged clocks

`--e1-frontier-time-limit-s`：

- 普通 grid 的 coverage-only discovery；
- 默认 120 s；
- 只影响何时返回 incumbent/bound。

`--e1-certify-time-limit-s`：

- blocking predecessor target；
- final knee full lex；
- 默认继承 `--time-limit-s`。

`--e1-b-auto off`：

- 固定 B grid；
- 适合 mechanism/specified grid；
- formal 若 grid 不足以证明 plateau，不会通过重复 tail retries“创造”证据。

`--e1-sat-patience`：

- 控制 experiment scheduler 的 plateau tail definition；
- 不改变单个 `(K,B)` 的数学可行域。

## 9. Certified complete route universe parameters

Step13：

```text
formal_route_universe = auto
formal_route_universe_max_turbines = 8
formal_route_universe_max_stops = 4
formal_route_universe_time_limit_s = 7200
```

`auto` 仅在 small-n 条件满足时 materialize 完整 physical route universe。`force` 要求强制尝试；`off` 返回 implicit exact pricing。

完整 universe 构造时限只是构建预算；未完成则 `complete=False`，不能把部分 materialization 当正式 exact route space。

## 10. E2 parameters

Current defaults：

```text
e2_quantiles = 0.2,0.5,0.8
e2_criteria = recourse_compatible
e2_discovery_time_limit_s = 120
e2_certify_time_limit_s = inherit time_limit_s if omitted
hs_quantile = 0.5
```

`wait_and_speed + recourse_compatible` 启用：

```text
nominal
gaussian
cantelli
vp
```

七方法 benchmark 必须改：

```text
time_recourse = wait_only
e2_criteria = all
```

q<qmax 属于 discovery/diagnostic；qmax 才承担 full exact certification 和 final freeze 资格。

## 11. Step14 A1/A2 parameters

Current defaults：

```text
A1 n_list = 6,8,10
A2 a2_n = 6,8
A2 a2_dtau = 15,10,5
study_mode = formal
uav = auto
k = None
batteries = None
```

Formal Step14 额外要求：

- explicit local turbines/wind/wave；
- Xi train；
- coherent weather moments；
- track or MMSI；
- track-start；
- same Step13 formal semantic flags。

直接手工 `uav/K/B` 可运行可追溯算法 benchmark，但不能绕过 structured E1 validation freeze 去授权 final test。

## 12. Resume and source-tree parameters

Formal identity绑定：

- `source_tree_sha256`；
- parameter SHA；
- instance SHA；
- model SHA；
- algorithm SHA；
- result/proof/scheduler contracts；
- `resume_input_sha256`。

`.md` 被 source-tree hash 覆盖，所以修改文档也会使旧 checkpoint 不再 compatible。这是当前代码的显式 provenance 设计。

## 13. Public/internal boundary

以下不是正式用户参数：

- `implicit_test_columns`；
- `allow_resource_only_columns`；
- private synthetic route fixtures；
- future-row coefficient range registry。

Public exact physical BPC 收到 synthetic route injection 必须 fail-closed。Selftest 的人工 finite route oracle 只能验证 algorithmic certificate，不能产生 physical-model certificate。
