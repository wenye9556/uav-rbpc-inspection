# 端到端流程

本文件只保留当前可执行流程。旧 checkpoint 迁移史、已修复 runner/encoding 问题和临时调试步骤不再作为正式流程。

## 1. 总原则

当前项目的证据链必须按顺序闭合：

```text
数据/合同预检
→ formal E1 coverage frontier
→ targeted resource-knee certification
→ knee-only full lexicographic optimization
→ validation selection
→ formal E2 validation matrix
→ E2 final method/plan freeze
→ one-shot final test
→ A1 / A2 algorithm experiments
→ paper figures / reporting
```

任何一步未满足其 formal gate，下一步必须停止；不能用手工 CSV 修改、旧 checkpoint、synthetic fallback 或 point estimate 绕过。

## 2. Source-tree 与 resume

`step9_model.source_tree_sha256()` 会哈希当前 `.py/.md/.json/.yaml/.toml/.ini/.cfg` 等源/配置文件。

因此：

- 文档修改也会形成新的 source tree；
- 新 source tree 第一次 publication run 用 `--resume off`；
- `--resume on` 只用于完全相同 source tree + binary64 instance 的中断续跑；
- 已归档旧 source-tree 结果继续作为独立证据保存，不与新 source-tree checkpoint 混排。

## 3. Step7：数据统计与 weather-only 重建

### 3.1 Frozen Xi

正式 train/validation/test 已经完成 disjoint + purge 验证，不重新切分。

Formal Xi：

```text
predictor = cv_noleak
predictor_contract = cv_noleak_backward_window_epoch_seconds
timestamp contract = utc_datetime64_ns_to_epoch_seconds
MMSI = 219018788
cross_pool = False
horizons = [5,10,15,20,25,30]
```

### 3.2 Weather-only coherent regeneration

旧 weather scalar/vector center incoherence 已通过 train-only predictor comparison 修复。

使用：

```text
step7_compute_xi.py --weather-only-existing-splits
```

只重建：

- weather residual；
- weather moments；
- Xi+weather validation；
- Xi+weather test metadata/product。

不改变 frozen Xi split，不读取 final-test performance。

当前 formal weather stage：

```text
tracks\weather_v17_1_stage\
```

## 4. Step20：zero-optimization preflight

在 publication optimization 前运行：

```powershell
python step20_preflight_final.py `
  --track-csv "tracks\track_219018788.csv" `
  --track-mmsi 219018788 `
  --xi-train-samples "tracks\xi_samples_caseB.csv" `
  --validation-samples "tracks\weather_v17_1_stage\xi_weather_samples_validation_caseB.csv" `
  --final-test-samples "tracks\weather_v17_1_stage\xi_weather_samples_test_caseB.csv" `
  --weather-moments-csv "tracks\weather_v17_1_stage\weather_moments_caseB.csv" `
  --holdout-purge-min 30 `
  --soc-correction geo2d `
  --soc-risk-allocation optimized `
  --time-recourse wait_and_speed `
  --recovery-predictor cv_noleak
```

当前已验证状态：

```text
PRECHECK = PASS
E2_final_test_state = not yet consumed
```

Step20 只做 metadata/schema/provenance/risk-power audit，不授权查看 final-test outcome。

## 5. Formal Step13 base instance

Formal 入口必须显式给：

- turbines/wind/wave；
- track CSV 或 concrete MMSI；
- `track_start_min`；
- Xi train；
- validation；
- final-test file；
- coherent weather moments。

固定口径：

```text
study_mode = formal
validation_mode = real_validation
final_weather_mode = real
recovery_predictor = cv_noleak
pool_h = pareto
weather_drcc = on
soc_correction = geo2d
soc_risk_allocation = optimized
time_recourse = wait_and_speed
battery_reuse_mode = exact_soc
deck_mode = interval
solver = exact-branch-price-cut
pricing = r-bpc
```

R-BPC 的正式求解主循环为：

```text
Restricted Master
    -> heuristic-first exact pricing
    -> exact resource audit / valid cuts
    -> generated-column exact primal recovery
    -> branch / full-space pricing closure
```

resource-aware timing enrichment 只作为合法列发现加速器，不承担 full-space 证书责任。

Step13 会再次检查：

- single concrete MMSI；
- Xi predictor/epoch contract；
- no cross-vessel pool；
- weather moments ↔ actual Xi train SHA；
- formal launch construction；
- selected launch-window MMSI 与 bound MMSI；
- source/input/model/proof fingerprints。

## 6. E1：资源前沿

### 6.1 Frontier discovery

对每个 UAV 档位建立：

```text
UAV × K × B
```

small-n formal baseline 先构建一次 certified complete physical route universe：

```text
n_turbines <= 8
max_stops <= 4
formal-route-universe = auto
```

若 universe 没有自然完整结束或 invariant/hash 失败，`complete=False`，不得把部分列池当 exact finite route space。

普通 K/B 网格只做短 `coverage-only` discovery：

```text
coverage_incumbent <= C*(K,B) <= coverage_upper_bound
```

### 6.2 Plateau

coverage 对 K/B 单调。Scheduler 先用：

- hard coverable cap；
- monotone envelope；
- lower/upper-bound sandwich；

证明 plateau。

`safe_served` 不进入优化资源单调证明。

### 6.3 Targeted predecessor closure

计算：

```text
T = ceil(knee_frac × plateau_coverage)
```

BK 顺序下先闭合 B predecessor，再闭合 K predecessor。

当 `T=|I|` 且完整 route universe 可用：

1. `[THM-GBR]` 先计算 universe-level exact battery-energy necessary relaxation；
2. 若 relaxation 已证明 `B_relax*>B`，直接 target NO；
3. 否则进入 full-cover partition binary master；
4. integer pattern 调 exact UAV/battery/SOC/deck resource audit；
5. `INFEASIBLE_PROVEN` 才允许加入 exact resource cut；
6. timeout/UNKNOWN 保持 unresolved。

不要因为 hard cap 已证明仍扩大 B 轴；下一步应是 `--exp E1_knee_refine`。

### 6.4 Knee-only full lex

资源 minimality 闭合后，只对 knee：

1. Stage-1 coverage 全局闭合；
2. 固定最优 coverage；
3. Stage-2 energy 全局闭合；
4. 要求 `global_certificate_available=True`。

`coverage_global_certificate_available=True` 单独不等于 full lex certificate。

### 6.5 Fixed-point full lex certification

若论文主实例的资源点 `(UAV,K,B)` 已显式固定，可使用 `--exp E1_lex_certify` 跳过 resource-knee selection controller，直接在同一 R-BPC 正式路径上执行：

1. Stage-1 coverage 全局闭合；
2. 固定该最优 coverage；
3. Stage-2 energy 全局闭合；
4. 仅当 `energy_optimal=True`、`lexicographic_optimal=True` 且 `global_certificate_available=True` 时，声明完整词典序证书。

该入口不证明所给 `(K,B)` 是资源最小 knee；它只证明该固定资源点内部的词典序最优性。

### 6.6 Validation

对 frozen exact knee plan 用 validation replay。

可能出现：

```text
optimization certificate PASS
validation statistical certificate FAIL
```

这不是矛盾。若 simultaneous UCB 超过 allocation gate，`knee_plan_holds=False`，正式 auto selection 保持未选择。

## 7. E1 selection

`E1_select` 会重新检查结构化字段，而不是信任一个字符串：

- resource threshold point valid；
- knee resource minimality certified；
- coverage certificate；
- global lex certificate；
- current result/proof/scheduler contracts；
- source resume fingerprint；
- `knee_plan_holds=True`。

没有同时通过 optimization + validation 的候选时，不得自动进入 publication E2/A。

## 8. E2：风险方法比较

### 8.1 主表

当前 `wait_and_speed` 只比较：

```text
nominal
gaussian
cantelli
vp
```

因为这四种方法共享当前已认证 return-speed recourse。

### 8.2 Seven-method benchmark

SAA/box/budget 若与上述方法同表比较，必须统一：

```text
time_recourse = wait_only
e2_criteria = all
```

不能把 unsupported recourse rejection 当作方法劣势。

### 8.3 q schedule

- `q < q_max`：短 discovery，允许 anytime results；
- `q = q_max`：长 exact lex certification；
- q_max 未有 `global_certificate_available=True` → `run_status=unresolved`；
- unresolved q_max 在 compatible resume 中重试。

Final-test freeze 只从 q_max + validation-safe + global-certified candidates 选择。

## 9. Step14：A1/A2 formal parity

正式 A1/A2 必须与 Step13 使用同一个 finite instance：

```text
same MMSI
same track SHA
same Xi train SHA
same weather moments SHA
same weather predictor contract
formal launch = True
no cross-vessel Xi pool
```

Step14 不允许：

- formal + allow-synth；
- formal 缺 Xi train；
- formal 缺 track start；
- formal 缺 coherent weather moments；
- formal 使用非 `cv_noleak / pareto / geo2d / interval` 基准口径。

### A1

比较：

```text
research_greedy
research_restricted_pool
exact_branch_price_cut
```

重点：解质量、coverage/energy gap、certificate scope。

### A2

比较 restricted-pool baseline 与 exact BPC：

```text
n × dtau
```

只有同 lex solution quality 时才解释 runtime ratio。

Publication 最安全的 A1/A2 入口仍是 Step13 `--exp full_suite`，因为 Step13 在启动 Step14 子进程前先检查 structured E1 freeze。

## 10. E2 freeze 与 one-shot final test

Formal test 严格 once-only：

1. E1 structured freeze 必须 verified；
2. E2 matrix 必须完整；
3. q_max candidate 必须 global-certified；
4. candidate 必须 validation-safe；
5. route/order/horizon/UAV/battery/turnaround `plan_fingerprint` 固定；
6. train/validation/test byte hashes 必须保持不变；
7. 才允许在 E2 内消费 joint final test 一次；
8. A1/A2 不参与 final-test candidate selection，`full_suite` 在 E2 final-test 步骤之后才启动 Step14；
9. 看到 test 后不得再改模型/参数/selection 并重新看同一个 test。

如果没有候选满足以上条件，流程停止，test 保持未消费。

## 11. Diagnostics

### Step15

联合 replay 管线检查。不是新的 publication optimizer。

### Step18

Multi-stop root-cause diagnostic。只用于判断：

- multi-stop 是否物理存在；
- 是否被 exact plan 实际采用；
- singleton 是否仅是 anytime/pricing discovery artifact。

不签 formal certificate。

### Step19

瓶颈与 sensitivity。任何变更风险预算、weather/geo2d、horizon、recourse 等 counterfactual 都是独立模型/诊断，不得替代 baseline。

## 12. 结果审计

Formal E1 至少检查：

```text
complete route-universe manifest
E1_frontier.csv
E1_knee_target_certificates.csv
E1_selection.csv
E1_validation_route_detail.csv
E1_validation_specificity_audit.csv
run manifest
```

最终 exact knee 要求：

```text
coverage_optimal=True
energy_optimal=True
lexicographic_optimal=True
global_certificate_available=True
```

并核对：

```text
result_contract
result_certificate_contract
formal_proof_contract
proof_contract_sha256
formal_experiment_scheduler_contract
source_tree_sha256
resume_input_sha256
parameter/instance/model/algorithm SHA
```

## 13. 当前已知状态

固定资源论文主实例 `n=10, UAV=M, K=2, B=7` 已通过 `E1_lex_certify` 完成完整词典序优化闭合：

- `coverage_incumbent=coverage_upper_bound=8`；
- `energy_incumbent_Wh=1956.6403352966677`；
- `energy_lower_bound_Wh=1956.6403352966672`；
- `energy_optimal=True`、`lexicographic_optimal=True`；
- `global_certificate_available=True`、`global_route_space_certificate=True`；
- `termination_reason=energy-bound-closed`；
- formal runtime 约 `20094.27 s`（5.58 h）。

该归档运行 source-tree SHA256 为 `4ae49deb449cc5603f63b157126052f79a2fd3f5b70a6c868ae2aaf5148aaf37`。其 exact optimization certificate 已完成，但该新最优计划的 statistical validation 结果仍需单独归档；final test 仍不得因为优化证书自动消费。

更早的 S/M resource-knee 与 validation 归档继续保留作历史 provenance。由于 `.md` 进入 `source_tree_sha256`，本次结果同步后的 ZIP 是新的 source tree；旧归档不能直接 resume，最终 publication reproduction 必须 `--resume off` fresh run。

## 14. 禁止的捷径

- 不为 validation PASS 改 confidence/epsilon；
- 不按 validation/test 合并 states；
- 不把 research restricted-pool optimal 当 global exact；
- 不把 timeout 当 infeasible；
- 不把 old root weather files 混入 coherent stage；
- 不重切 frozen Xi；
- 不提前查看 final-test outcome；
- 不在 source-tree SHA 改变后复用旧 formal checkpoint。
