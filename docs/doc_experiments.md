# 实验设计与运行手册

本文件中的命令按当前 `step13_experiment_model.py` / `step14_experiment_algorithm.py` argparse 和 formal fail-closed 条件整理。旧 CLI、旧数据路径和版本迁移命令已删除。

## 1. 证据等级

### 1.1 Formal

Formal 结果必须使用真实本地输入和当前 provenance：

- `--study-mode formal`
- `--validation-mode real_validation`
- `--final-weather-mode real`
- concrete MMSI/track；
- explicit `--track-start-min`；
- Xi train/validation/final-test files；
- coherent weather moments；
- `cv_noleak`；
- `pool_h=pareto`；
- `weather_drcc=on`；
- `soc_correction=geo2d`；
- `battery_reuse_mode=exact_soc`；
- `deck_mode=interval`；
- exact BPC / exact implicit DFS。

Formal optimization certificate 与 validation/final-test statistical evidence 是两条独立证据链。

### 1.2 Mechanism

`--study-mode mechanism --allow-synth` 可用于接口、机制和压力测试。Synthetic 结果不能形成真实平台 reliability claim。

## 2. 当前 Rodsand II formal 数据常量

```text
TURBINES = data\turbines_Rodsand_II_clean.csv
WIND = weather\weather_Rodsand_II.csv
WAVE = weather\waves_Rodsand_II.csv

TRACK = tracks\track_219018788.csv
MMSI = 219018788
TRACK_START_MIN = 410.233333

XI_TRAIN = tracks\xi_samples_caseB.csv
VALIDATION = tracks\weather_v17_1_stage\xi_weather_samples_validation_caseB.csv
FINAL_TEST = tracks\weather_v17_1_stage\xi_weather_samples_test_caseB.csv
WEATHER_MOMENTS = tracks\weather_v17_1_stage\weather_moments_caseB.csv

HOLDOUT_PURGE_MIN = 30
```

`FINAL_TEST` 文件在 formal preflight/协议中必须存在，但其 outcome 只能在最终 freeze 后一次性消费。

## 2.1 论文主算法与核心消融

论文主方法统一记为 **R-BPC**，命令行使用 `--pricing-mode r-bpc`。正文核心消融只保留两个问题：

| configuration | resource-aware timing enrichment | exact generated-column primal recovery |
|---|---:|---:|
| R-BPC | on | on |
| w/o Timing | off | on |
| w/o Recovery | on | off |

Battery half-cap、deck-guided ordering、warm start、cache 等保留为模型 strengthening / implementation details，不作为正文主消融，以避免把论文贡献拆成过多模块。若需要额外运行时间敏感性，可放入补充材料。

当前已归档的 10-turbine、M、`K=2,B=7` 主实例已经完成完整词典序证明：`coverage=8/8`，`energy_incumbent_Wh=1956.6403352966677`，`energy_lower_bound_Wh=1956.6403352966672`，`energy_optimal=True`，`lexicographic_optimal=True`，`global_certificate_available=True`。正式 runtime 约 `20094.27 s`（5.58 h）。对应归档 source-tree SHA256 为 `4ae49deb449cc5603f63b157126052f79a2fd3f5b70a6c868ae2aaf5148aaf37`。论文主表应把 coverage 与 conditional energy 两层证书分列报告，同时把 statistical validation 单独报告，不能由优化证书替代。

## 3. 当前可执行命令

### 3.1 Mechanism full suite

该命令依次贯通 E1 frontier → targeted knee closure → E1 selection → E2 → A1 → A2，仅用于机制验证：

```powershell
python step13_experiment_model.py --exp full_suite --study-mode mechanism --validation-mode synthetic_stress --allow-synth --n-turbines 8 --e1-uavs S,M,L --fleet-ks 1,2,3 --e1-batteries 0,1,2,3,4 --e1-b-auto off --stops-cap 4 --algorithm-max-stops 4 --solver-mode exact-branch-price-cut --pricing-mode r-bpc --time-limit-s 1800 --coverage-gap-target-abs 0 --energy-gap-target-rel 0 --energy-gap-target-abs-wh 1e-6 --resume off
```

注意：fixed-grid `--e1-b-auto off` 允许机制压力测试，但 formal E1 若要证明资源平台通常应使用 `--e1-b-auto on`。

### 3.2 当前 Rodsand II formal full-suite 入口

这条命令与当前 Python 匹配，但只有 E1 形成 **validation-approved、资源最小且 global-lex certified** 的 structured freeze 后，才会进入 formal E2。E2 完整矩阵与 qmax validation-safe candidate 闭合后，**该命令会在 E2 内一次性消费 final test**，随后才启动 A1→A2。若 E1/E2 gate 不成立则 fail-closed，不消费 final test。不要把这条命令当普通 smoke。

```powershell
python step13_experiment_model.py --exp full_suite --study-mode formal --validation-mode real_validation --final-weather-mode real --n-turbines 8 --farm Rodsand_II --e1-uavs S,M,L --fleet-ks 1,2,3 --e1-batteries 0,1,2,3,4 --e1-b-auto on --e1-b-cap 8 --e1-sat-patience 3 --max-stops 4 --stops-cap 4 --algorithm-max-stops 4 --window-min 360 --dtau-min 5 --deck-mode interval --battery-reuse-mode exact_soc --soc-correction geo2d --soc-risk-allocation optimized --time-recourse wait_and_speed --xi-train-samples "tracks\xi_samples_caseB.csv" --validation-samples "tracks\weather_v17_1_stage\xi_weather_samples_validation_caseB.csv" --final-test-samples "tracks\weather_v17_1_stage\xi_weather_samples_test_caseB.csv" --holdout-purge-min 30 --recovery-predictor cv_noleak --pool-h pareto --weather-drcc on --weather-alignment timestamp --turbines-csv "data\turbines_Rodsand_II_clean.csv" --wind-csv "weather\weather_Rodsand_II.csv" --wave-csv "weather\waves_Rodsand_II.csv" --weather-moments-csv "tracks\weather_v17_1_stage\weather_moments_caseB.csv" --track-csv "tracks\track_219018788.csv" --track-mmsi 219018788 --track-start-min 410.233333 --formal-route-universe auto --formal-route-universe-max-turbines 8 --formal-route-universe-max-stops 4 --formal-route-universe-time-limit-s 7200 --formal-warmstart-seconds 60 --e1-frontier-time-limit-s 120 --e1-certify-time-limit-s 7200 --e2-discovery-time-limit-s 120 --e2-certify-time-limit-s 1800 --solver-mode exact-branch-price-cut --pricing-mode r-bpc --time-limit-s 7200 --coverage-gap-target-abs 0 --energy-gap-target-rel 0 --energy-gap-target-abs-wh 0 --resume off
```

由于 `.md` 也进入 `source_tree_sha256`，文档修改后第一次 formal publication run 必须视作新 source tree，用 `--resume off`。旧 source-tree checkpoint 不得直接续入。

### 3.3 仅运行 formal E1 frontier

```powershell
python step13_experiment_model.py --exp E1_frontier --study-mode formal --validation-mode real_validation --final-weather-mode real --n-turbines 8 --farm Rodsand_II --e1-uavs S,M,L --fleet-ks 1,2,3 --e1-batteries 0,1,2,3,4 --e1-b-auto on --e1-b-cap 8 --e1-sat-patience 3 --max-stops 4 --stops-cap 4 --window-min 360 --dtau-min 5 --deck-mode interval --battery-reuse-mode exact_soc --soc-correction geo2d --soc-risk-allocation optimized --time-recourse wait_and_speed --xi-train-samples "tracks\xi_samples_caseB.csv" --validation-samples "tracks\weather_v17_1_stage\xi_weather_samples_validation_caseB.csv" --final-test-samples "tracks\weather_v17_1_stage\xi_weather_samples_test_caseB.csv" --holdout-purge-min 30 --recovery-predictor cv_noleak --pool-h pareto --weather-drcc on --weather-alignment timestamp --turbines-csv "data\turbines_Rodsand_II_clean.csv" --wind-csv "weather\weather_Rodsand_II.csv" --wave-csv "weather\waves_Rodsand_II.csv" --weather-moments-csv "tracks\weather_v17_1_stage\weather_moments_caseB.csv" --track-csv "tracks\track_219018788.csv" --track-mmsi 219018788 --track-start-min 410.233333 --formal-route-universe auto --formal-route-universe-max-turbines 8 --formal-route-universe-max-stops 4 --formal-route-universe-time-limit-s 7200 --formal-warmstart-seconds 60 --e1-frontier-time-limit-s 120 --e1-certify-time-limit-s 7200 --solver-mode exact-branch-price-cut --pricing-mode r-bpc --time-limit-s 7200 --coverage-gap-target-abs 0 --energy-gap-target-rel 0 --energy-gap-target-abs-wh 0 --resume off
```

可用 `--e1-uavs S`、`M` 或 `L` 单独跑一个机型并归档结果。首次当前-source-tree run 用 `--resume off`；只有同 source tree、同 binary64 instance 的中断续跑才改 `--resume on`。

### 3.4 Formal E1 targeted knee refinement

当 selection 为 `uncertified_resource_knee` 且 hard cover cap/plateau 已证明时，**不要扩大 B 轴**，运行：

```powershell
python step13_experiment_model.py --exp E1_knee_refine --study-mode formal --validation-mode real_validation --final-weather-mode real --n-turbines 8 --farm Rodsand_II --e1-uavs S,M,L --fleet-ks 1,2,3 --e1-batteries 0,1,2,3,4 --e1-b-auto on --e1-b-cap 8 --e1-sat-patience 3 --max-stops 4 --stops-cap 4 --window-min 360 --dtau-min 5 --deck-mode interval --battery-reuse-mode exact_soc --soc-correction geo2d --soc-risk-allocation optimized --time-recourse wait_and_speed --xi-train-samples "tracks\xi_samples_caseB.csv" --validation-samples "tracks\weather_v17_1_stage\xi_weather_samples_validation_caseB.csv" --final-test-samples "tracks\weather_v17_1_stage\xi_weather_samples_test_caseB.csv" --holdout-purge-min 30 --recovery-predictor cv_noleak --pool-h pareto --weather-drcc on --weather-alignment timestamp --turbines-csv "data\turbines_Rodsand_II_clean.csv" --wind-csv "weather\weather_Rodsand_II.csv" --wave-csv "weather\waves_Rodsand_II.csv" --weather-moments-csv "tracks\weather_v17_1_stage\weather_moments_caseB.csv" --track-csv "tracks\track_219018788.csv" --track-mmsi 219018788 --track-start-min 410.233333 --formal-route-universe auto --formal-route-universe-max-turbines 8 --formal-route-universe-max-stops 4 --formal-route-universe-time-limit-s 7200 --formal-warmstart-seconds 60 --e1-frontier-time-limit-s 120 --e1-certify-time-limit-s 7200 --solver-mode exact-branch-price-cut --pricing-mode r-bpc --time-limit-s 7200 --coverage-gap-target-abs 0 --energy-gap-target-rel 0 --energy-gap-target-abs-wh 0 --resume on
```

`--resume on` 仅适用于完全相同 source tree/instance 的已有 E1 frontier。新 source tree 必须先 fresh E1。

### 3.5 固定资源点完整词典序证明（论文主实例）

当论文主实例的 `(UAV,K,B)` 已经由研究设计显式固定，而不是等待 E1 resource-knee controller 自动选择时，使用 `E1_lex_certify` 直接执行同一 R-BPC 的完整：

```text
Stage-1 coverage closure
→ fix certified optimal coverage
→ Stage-2 energy minimization
→ full lexicographic certificate
```

例如当前 10-turbine、M、K=2、B=7 主实例：

```powershell
python step13_experiment_model.py --exp E1_lex_certify --study-mode formal --validation-mode real_validation --final-weather-mode real --n-turbines 10 --farm Rodsand_II --uav M --k 2 --batteries 7 --max-stops 4 --stops-cap 4 --algorithm-max-stops 4 --window-min 360 --dtau-min 5 --deck-mode interval --battery-reuse-mode exact_soc --soc-correction geo2d --soc-risk-allocation optimized --time-recourse wait_and_speed --xi-train-samples "tracks\xi_samples_caseB.csv" --validation-samples "tracks\weather_v17_1_stage\xi_weather_samples_validation_caseB.csv" --final-test-samples "tracks\weather_v17_1_stage\xi_weather_samples_test_caseB.csv" --holdout-purge-min 30 --recovery-predictor cv_noleak --pool-h pareto --weather-drcc on --weather-alignment timestamp --turbines-csv "data\turbines_Rodsand_II_clean.csv" --wind-csv "weather\weather_Rodsand_II.csv" --wave-csv "weather\waves_Rodsand_II.csv" --weather-moments-csv "tracks\weather_v17_1_stage\weather_moments_caseB.csv" --track-csv "tracks\track_219018788.csv" --track-mmsi 219018788 --track-start-min 410.233333 --formal-route-universe off --formal-warmstart-seconds 60 --e1-certify-time-limit-s 36000 --solver-mode exact-branch-price-cut --pricing-mode r-bpc --time-limit-s 36000 --archive-primal-recovery on --archive-primal-recovery-time-limit-s 2 --coverage-gap-target-abs 0 --energy-gap-target-rel 0 --energy-gap-target-abs-wh 0 --resume off
```

该入口不声称资源 minimality，也不替代 `E1_knee_refine`；它只对显式固定资源点完成完整词典序优化证明。输出写入 `results/model_experiments/E1_lex_certify/`。

当前归档主结果已用该入口闭合到 `(C*,E*)=(8,1956.640335296667 Wh)`。`pricing_complete=False` 与该证书不矛盾：Stage-2 使用 `rmp-lagrangian-plus-pricing-bound` 得到足够强的 omitted-column 全局下界，使 `energy_lower_bound_Wh` 与 incumbent 在 machine precision 下重合；因此无需穷举完全部 pricing state 也可安全闭合能量最优性。

### 3.6 仅生成 E1 selection

该入口只读 CSV：

```powershell
python step13_experiment_model.py --exp E1_select --e1-csv "results\model_experiments\E1_frontier\E1_frontier.csv" --knee-frac 0.95 --knee-order BK --selection-metric safe_per_inventory_kWh
```

Formal auto-freeze 不是只看 `selection_status` 字符串，而会重新检查资源最小性、coverage/global certificates、result/proof/scheduler contracts 和 `knee_plan_holds`。

### 3.7 Formal E2 主比较：wait_and_speed

只有 E1 存在 validation-approved structured freeze 时才允许自动选择：

```powershell
python step13_experiment_model.py --exp E2_robust --study-mode formal --validation-mode real_validation --final-weather-mode real --n-turbines 8 --farm Rodsand_II --e2-quantiles 0.2,0.5,0.8 --e2-criteria recourse_compatible --e2-discovery-time-limit-s 120 --e2-certify-time-limit-s 1800 --time-recourse wait_and_speed --xi-train-samples "tracks\xi_samples_caseB.csv" --validation-samples "tracks\weather_v17_1_stage\xi_weather_samples_validation_caseB.csv" --final-test-samples "tracks\weather_v17_1_stage\xi_weather_samples_test_caseB.csv" --holdout-purge-min 30 --recovery-predictor cv_noleak --pool-h pareto --weather-drcc on --weather-alignment timestamp --soc-correction geo2d --soc-risk-allocation optimized --battery-reuse-mode exact_soc --deck-mode interval --turbines-csv "data\turbines_Rodsand_II_clean.csv" --wind-csv "weather\weather_Rodsand_II.csv" --wave-csv "weather\waves_Rodsand_II.csv" --weather-moments-csv "tracks\weather_v17_1_stage\weather_moments_caseB.csv" --track-csv "tracks\track_219018788.csv" --track-mmsi 219018788 --track-start-min 410.233333 --solver-mode exact-branch-price-cut --pricing-mode r-bpc --time-limit-s 1800 --coverage-gap-target-abs 0 --energy-gap-target-rel 0 --energy-gap-target-abs-wh 0 --resume on
```

`recourse_compatible` 在 `wait_and_speed` 下只比较：

- nominal；
- gaussian；
- cantelli；
- vp。

当前 S/M 已归档结果的 validation 均未通过，因此它们不会被当前 auto-freeze 选入 formal E2。这是协议 gate，不是 E2 bug。

### 3.8 七方法公平 benchmark

若需要：

```text
nominal / gaussian / cantelli / vp / SAA / box / budget
```

必须统一为：

```powershell
--time-recourse wait_only --e2-criteria all
```

其余 formal provenance 参数与 3.6 相同。该 benchmark 与主 `wait_and_speed` 表分开解释。

## 4. E1 staged exact certification

固定 UAV 档位，令 `C*(K,B)` 为 Stage-1 exact coverage。增加 UAV/电池只增加可用实体且允许不用新增实体，因此 coverage 对 K/B 单调。

当前 scheduler：

1. `E1_frontier` 普通网格点：短 `coverage-only`；
2. 保存 `coverage_incumbent <= C* <= coverage_upper_bound`；
3. 用 hard cover cap / monotone envelope 证明 plateau；
4. 计算 `T=ceil(knee_frac*P)`；
5. 对 BK/KB immediate predecessor 做 `coverage-target` decision；
6. target YES 必须有 exact physical/resource witness；
7. target NO 必须闭合完整正式路线/资源证书；
8. resource minimality 证明后，仅 knee 跑完整 lexicographic solve；
9. frozen lex plan 再做 validation。

`safe_served` 只属于 validation 诊断，不参与资源单调证明。

## 5. Small-n complete physical route universe

当前 formal E1 默认：

```text
--formal-route-universe auto
--formal-route-universe-max-turbines 8
--formal-route-universe-max-stops 4
```

满足阈值时一次性穷举正式 `(launch, ordered sequence, horizon)` 并调用同一 physical/DRCC oracle。只有 manifest：

```text
complete = True
reason = complete
```

并通过 context/column hash 和 invariant 审计时，才可视为完整有限物理列集。

`stops>=2` 的 universe column 只证明 multi-stop 物理列存在；最终 plan 的 `max_stops_observed>=2` 才证明最优/target witness 实际使用 multi-stop。

## 6. A1/A2 算法实验

### 6.1 A1 — solution quality / certificate

比较：

```text
research_greedy
research_restricted_pool
exact_branch_price_cut
```

默认规模：

```text
--n-list 6,8,10
```

研究基线可以有 restricted-pool optimum，但必须保持 `global_certificate_available=False`。只有正式 exact path 可发布完整 finite physical model certificate。

### 6.2 A2 — runtime / scalability

比较 restricted-pool baseline 与 exact BPC，默认：

```text
--a2-n 6,8
--a2-dtau 15,10,5
```

只有两种方法达到相同 lexicographic solution quality 时，runtime ratio 才能解释成同质量速度比较。

### 6.3 Formal A1/A2 的安全入口

当前最安全的 publication 路径是 `step13 --exp full_suite`：Step13 先验证 structured E1 freeze，再把同一 MMSI、track、Xi train、weather moments、track-start、UAV/K/B 和 model flags传给 Step14。

直接调用 Step14 时，formal 至少必须显式提供：

- turbines / wind / wave；
- Xi train；
- weather moments；
- track / MMSI；
- track start；
- same `cv_noleak / pareto / weather_drcc=on / geo2d / interval`。

如果显式手工给 `--uav/--k/--batteries`，只能作为可追溯算法 benchmark；不能绕过 E1 validation freeze 去授权 one-shot final test。

示例（手工配置，**非自动 E1 publication freeze**）：

```powershell
python step14_experiment_algorithm.py --exp A1_accuracy --study-mode formal --n-list 6,8,10 --farm Rodsand_II --uav M --k 2 --batteries 7 --max-stops 4 --window-min 360 --dtau-min 5 --deck-mode interval --turbines-csv "data\turbines_Rodsand_II_clean.csv" --wind-csv "weather\weather_Rodsand_II.csv" --wave-csv "weather\waves_Rodsand_II.csv" --xi-train-samples "tracks\xi_samples_caseB.csv" --weather-moments-csv "tracks\weather_v17_1_stage\weather_moments_caseB.csv" --track-csv "tracks\track_219018788.csv" --track-mmsi 219018788 --track-start-min 410.233333 --recovery-predictor cv_noleak --pool-h pareto --weather-drcc on --weather-alignment timestamp --soc-correction geo2d --soc-risk-allocation optimized --time-recourse wait_and_speed --solver-mode exact-branch-price-cut --pricing-mode r-bpc --time-limit-s 1800 --coverage-gap-target-abs 0 --energy-gap-target-rel 0 --energy-gap-target-abs-wh 1e-6 --resume off
```

A2 将 `--exp A1_accuracy --n-list 6,8,10` 替换为：

```text
--exp A2_speed --a2-n 6,8 --a2-dtau 15,10,5
```

## 7. Joint replay

独立 replay 入口用于检查 split/purge/replay 管线：

```powershell
python step15_replay.py --samples ALL_SPLITS_SAMPLE_CSV --n-turbines 14 --max-stops 4 --train-frac 0.60 --validation-frac 0.20 --purge-min 60 --min-cell-n 30
```

这里的示例是独立 replay 工具，不是当前 frozen formal split 的推荐重切命令。当前 formal frozen split 继续使用已验证 purge=30，不应因本示例重新生成数据。

## 8. Diagnostics

### Step18

`step18_diagnose_multistop.py`：multi-stop root-cause，诊断-only，不签 formal certificate。

### Step19

`step19_diagnose_formal_bottlenecks.py`：energy/time/weather/Xi/risk/resource 瓶颈与 sensitivity，非 baseline 变体不得升级正式证书。

### Step20

`step20_preflight_final.py`：zero-optimization metadata/provenance preflight，不消费 final-test outcome。

## 9. 结果目录

```text
results\model_experiments\E1_frontier\
results\model_experiments\E2_robust_comparison\
results\algorithm_experiments\A1_accuracy\
results\algorithm_experiments\A2_speed\
```

正式归档至少保存：

- run manifest；
- source-tree SHA；
- input hashes；
- parameter/instance/model/algorithm contracts；
- result/proof/scheduler contracts；
- complete-route manifest（small-n E1）；
- target certificates；
- selection；
- validation route detail；
- final freeze sidecar；
- one-shot final-test state。

## 10. Resume 规则

`--resume on` 不是“继续同名实验”这么简单。只有以下一致时才可复用：

- source-tree SHA；
- current result/proof/scheduler contracts；
- binary64 input fingerprint；
- model/parameter/instance/algorithm hashes；
- route/resource numeric semantics；
- UAV/K/B/track/weather/Xi configuration。

文档改动也会改变 source-tree SHA，因此本次文档清理后，新的 source tree 不得直接 resume 文档清理前的 S/M/L checkpoint。

## 11. Final-test 纪律

Formal publication 顺序：

1. frozen train；
2. E1 optimization；
3. validation selection；
4. E2 完整预声明矩阵；
5. freeze E2 final method/plan；
6. consume joint final test once；
7. A1/A2 作为算法实验可随后运行，但不参与 final-test candidate selection。

如果没有 validation-approved E1/E2 candidate，则流程停止，final test 继续保持未消费。不得因为 optimization certificate 已 PASS 就绕过统计 selection gate。

## 12. v3.1 加速版补充实验（新增 2026-08-24）

### 12.1 版本基线

- `git tag baseline-reference`：加速前求解器（对应 5.54 h 主实例归档）。
- `git tag paper-v3.1`：加速版（P1-P6，见 doc_algorithm §12 与 doc_proof
  addendum）。主实例 36 min，C\*/E\* 逐位一致。
- 加速版各实例结果归档于 `results/experiments/E1_*/`（12 组：版本对照 ×4、
  规模轴 n12/n15×2、机队轴 K1×3/K3、无热启动对照）。

### 12.2 论文实验矩阵（A/B 组）

| 实验 | 入口 | 说明 |
|---|---|---|
| A-1 核心消融 w/o Timing | `step13 --exp E1_lex_certify --pricing-mode exact-layered-batch-primal-battery-halfcap-formal`（n=10/M/K=2/B=7） | 历史长模式名即"无 timing enrichment"的 formal 变体 |
| A-1 核心消融 w/o Recovery | 同主实例 + `--archive-primal-recovery off` | 关闭生成列原始恢复 |
| A-2 解质量 | `step14 --exp A1_accuracy --n-list 6,8,10 --pricing-mode r-bpc` | greedy / restricted-pool / exact 三方（step14 已接受 r-bpc 别名） |
| A-3 速度×Δτ | `step14 --exp A2_speed --a2-n 6,8 --a2-dtau 15,10,5 --pricing-mode r-bpc` | 同质量速度比 + 离散化敏感性 |
| B-1 资源前沿 | `step13 --exp E1_frontier` + `E1_knee_refine`（n=8 全网格 / n=10 战略点） | 覆盖最小性 sandwich 证明 |
| B-2 机型轴 | `step13 --exp E1_lex_certify --uav S / M / L`（n=10/K=2/B=7） | 证书级机型分离分析 |
| B-3 电池轴 | `step13 --exp E1_lex_certify --batteries 4,5,6`（n=10/M/K=2） | 边际电池价值曲线 |
| B-6..B-10 计划结构 | 认证解后处理（`results/experiments/E1_*` 的 detail/manifest 字段） | 从认证解免费挖掘：视界分布/停靠率/发射-船态耦合/Wh每台/甘特 |

已知发现（待论文化）：认证方案的发射时刻全部落于动力定位窗口；
n≥12 时 Stage-2 单调用闭合；C\*≡8 对 K∈{2,3}、n∈{10,12,15} 恒定
（电池/时间窗绑定）。
