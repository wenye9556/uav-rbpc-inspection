# 正式求解算法

本文件只描述当前 exact solver。旧版本迁移史已删除；`THM-CU`、`THM-FCT`、`THM-GBR` 等保留为当前算法组成部分，而不是“历史功能”。

## 0. 当前算法对象

正式求解器记为：

> **R-BPC — Resource-Aware Branch-and-Price-and-Cut.**

论文层只区分 Restricted Master、Exact Pricing、Exact Resource Feasibility and Primal Recovery 三个一级组件。公开正式入口：

```python
step12_branch_price.solve_fleet_anytime(
    ...,
    solver_mode="exact-branch-price-cut",
    pricing_mode="r-bpc")
```

`r-bpc` 只是 canonical alias；旧长模式仍保留用于历史复现和回归，但不作为论文算法名称。

当前 exactness 只针对：

```text
finite-route-model-strict-physical-v8-discrete-recovery-target-xi-only-coherent-weather
```

也就是当前输入、binary64 数值、起飞网格、recovery horizons、max stops、物理/DRCC、UAV/battery/deck resource semantics 一起定义的有限离散模型。

## 0.1 当前证书合同

```text
RESULT_CERTIFICATE_CONTRACT
= finite-binary64-physical-route-universe-certificate-v12-source-bound-global-battery-relaxation

FORMAL_PROOF_CONTRACT
= exact-bpc-proof-code-concordance-v9-source-bound-global-battery-relaxation
```

Proof contract 绑定 Step9–Step12 proof-critical 源码字节。Source-tree provenance 还包含 MD；旧 source-tree checkpoint 不得在修改后静默恢复成当前正证书。

## 0.2 Small-n complete physical route universe — THM-CU

对于 formal E1 且：

```text
n <= formal_route_universe_max_turbines
max_stops <= formal_route_universe_max_stops
```

`build_certified_route_universe` 穷举：

\[
R_{\rm phys}
=\{(o,\pi,h): o\in O,\ 1\le|\pi|\le S,\ \pi\text{ elementary ordered},
h\in H,\ \Phi(o,\pi,h)=1\}.
\]

每个候选调用与 implicit exact pricing 相同的 physical/DRCC evaluator。只有：

- enumeration 自然结束；
- context hash 一致；
- column semantics hash 一致；
- 所有 invariants 成立；

才输出 `complete=True`。

完整 materialization 只是 exact representation acceleration，不改变 route definition。若 incomplete/timeout/error，则不能把部分列集当完整 finite physical route universe；可以回到原 implicit exact BPC。

## 0.3 Full-cover target closure — THM-FCT

当 targeted predecessor 的 coverage target 满足：

\[
T=|I|,
\]

且 complete physical route universe 已认证时，full cover + per-turbine set-packing 推出 exact partition：

\[
\sum_{r:i\in S_r}x_r=1,\qquad \forall i\in I.
\]

算法可直接在完整列集上建立 binary full-cover master，而无需重复最大覆盖搜索。

Integer full-cover pattern 仍必须经过 unchanged exact entity-resource audit。只有 `INFEASIBLE_PROVEN` 的 pattern 才允许加入有效 resource cut；`UNKNOWN_TIMEOUT` 不能 cut、不能签 NO。

## 0.4 Universe-level global battery relaxation — THM-GBR

对 full-cover complete-universe target，先构造必要松弛：

- 保留 exact-cover route partition；
- 保留每条 route exact binary64-as-rational SOC energy；
- 只检查 selected routes 是否可放入 `B` 个容量为 `B_use` 的 battery bins；
- 删除 UAV identity/binding、timing、deck、inspection/swap station 等其它真实资源限制。

设该松弛覆盖全部 turbines 所需最少 battery 数为 `B_relax*`。因为任何真实可行 assignment 删除约束后仍是松弛可行解：

\[
B_{\rm relax}^\star>B
\Longrightarrow
\text{full-cover target 在真实资源模型中不可行}.
\]

反方向不成立。`B_relax*<=B` 只说明 energy-only necessary relaxation 可行，不能升级成真实 target YES。

当前 `n<=12` 时用 finite bitmask DP 精确计算；`UNKNOWN_TIMEOUT/SKIPPED_SIZE` 不剪枝、不签 NO。

## 0.5 Persistent exact resource closure

THM-GBR 未关闭 target 时，full-cover binary master 使用：

- active concurrency necessary rows；
- fastest-turnaround necessary rows；
- exact battery-bin-packing cores；
- persistent proven resource-cut ledger；
- unchanged exact UAV/battery/SOC/turnaround DFS；
- independent exact full-cover verification。

只有已经由 exact relaxation 或 exact resource DFS 证明的 cut 可以持久化。Timeout/UNKNOWN 从不写成 false proof。


## 1. 正式入口

正式论文实验统一调用：

```python
step12_branch_price.solve_fleet_anytime(...,
    solver_mode="exact-branch-price-cut",
    pricing_mode="r-bpc")
```

算法标识为 `branch-price-and-cut-with-logic-benders`。历史完整列池、软覆盖、集合划分和 Ryan–Foster 实现保留为研究回归接口，不进入正式全局证书路径。

## 2. 路线空间表示：small-n 完整物化 / large-n 隐式 exact

当 `formal-route-universe=auto` 且满足 small-n 条件时，可调用专用 `build_certified_route_universe` 完整物化正式物理列；该构造器不是 research-baseline 的受限枚举接口。超过阈值或显式 `off` 时，初始列仍仅含少量重验证种子，多停靠路线由节点 implicit exact pricing 按需生成。

公开 exact API 在建立 deadline 前集中校验 `kappa_mode/chance_mode/deck_mode/battery_reuse_mode/pool_h_mode/solver_mode/pricing_mode`、RMP 后端及时间、资源数量和 Gap 参数的有限性/整数性。未知模式不得通过默认 `else` 映射到合法模型；正式 BPC 只接受 `battery_reuse_mode="exact_soc"` 与 `pool_h_mode="pareto"`（后者仅作为正式协议标识，实际定价遍历全部离散 horizon）。正式 RMP 只实现 `scipy-highs-rmp`，其他后端请求 fail-closed。非法风险口径不得继承进程残留 `RM.kappa`。

外部种子中的路线对象、能耗、可行标志均不可信。`seed_cols` 必须在共享 deadline 下至多物化一次；计数、验证和求解复用同一序列，禁止先 `list(...)` 计数后再次迭代导致静默丢列。Python 无法用普通 deadline 检查中断任意阻塞的生成器 `next()`，因此有限 deadline 下默认只消费已经物化的 `list`/`tuple`；一次性或自定义迭代器必须由调用方预先物化，或显式声明 `seed_iterator_nonblocking=True`。未声明的迭代器会被 fail-closed 跳过，且 `seed_columns_revalidated=False`；这只放弃暖启动，不改变完整隐式路线空间的精确定价。`seed_validation` 记录 `consumed_count`、`materialized_count`、`validated_count`、`materialization_complete`、`validation_complete` 等证据。种子随后必须用当前风机对象、当前起飞选项、当前离散回收时长和 `route_feasible_at_h` 重新计算，并在重算期间显式绑定本次求解的 `kappa_mode`；调用前模块中残留的 Cantelli、VP 或高斯口径不得影响种子。`route_order`/`ordered_tids` 保留访问顺序，排序后的覆盖键不能覆盖它。

## 3. RMP

RMP 是风机互斥 set-packing LP：每台风机最多出现一次。它还包含连续时间甲板事件容量、必要 UAV 活动容量、池化 SOC 必要条件、全局资源**精确整数模式排除割**以及节点分支等式。第一阶段目标是纯 `-coverage`；第二阶段固定已证明最优覆盖并最小化计划能耗。

SciPy/HiGHS 返回值必须通过独立 KKT 验证：原始可行性、对偶符号、上下界 marginal、站立性、互补松弛和强对偶。不能只信任 `success=True`。

## 4. 精确按需定价

当前物理与 DRCC 判定是整体函数，源码没有提供可证明安全的可加资源递推，因此 R-BPC 不把 pricing 冒充为具有未证明 dominance 的 RCSP 标号器。正式 pricing 在有限 elementary route space 上采用 heuristic-first、layered exact traversal：

1. 按 stop depth 分层遍历当前节点允许的离散起飞选项；
2. 每层隐式扩展所有有序无重复 turbine sequences，并遍历全部离散 recovery horizons；
3. 每个完整候选调用同一 `route_feasible_at_h` whole-route physical/DRCC evaluator；
4. 使用 binary64 outward interval 计算 reduced cost，只有 `rc_U<0` 的列才是正式 improving column；
5. discovery 阶段可在找到小批严格改善列后提前回 Master 并重算 dual，但这种 early return 只影响搜索顺序；
6. 当需要节点闭合/全局证书时，剩余合法空间必须继续 exact completion，直到所有遗漏列都被检查或由严格有效的 omitted-column bound 排除。

只有 canonical route signature 与全部正式列语义（计划能耗、SOC 能耗、资源区间等）binary64-exact 相同的重复列才可做身份去重；同 signature 但正式语义不同视为模型不一致并 fail-closed。正式证书路径不使用 beam、邻居截断、标签上限、候选数上限或 `max_sequence_evals` 之类会截断完整路线空间的规则。

`pricing_mode="exact-mip"` 当前 fail-closed 返回 `solver_error`，因为源码黑箱物理/DRCC 尚无经证明等价的 MILP/MISOCP 编码。


## 4.1 数值模糊定价的有限进展

每条遗漏路线得到向外舍入区间 `[rc_L, rc_U]`。**正式证书定价的数学阈值统一为 0**：`rc_U < 0` 才证明该列严格改善；`rc_L >= 0` 才证明该列严格非负。`PRICING_EPS` 只可用于启发式排序、展示或非证书诊断，不能定义完整列空间的定价闭合。若 `rc_L < 0 <= rc_U`，该列的符号在 binary64 证书精度下不可判定。算法不再根据 `best_reduced_value` 点估计执行无条件 `continue`，而是加入有限批新的模糊列后重解 RMP。任何合法列加入 RMP 都只扩充 RMP、不会切除完整模型可行解；由于每次加入的签名此前不在 archive 且有限离散路线空间有限，该过程在无 deadline 情形也有组合上的有限进展。严格节点界仍只使用 `rc_L`，中性列本身不提升证书。`pricing_search_complete` 表示本轮隐式空间扫描完成，而 `pricing_closed`（兼容字段 `pricing_complete`）仅在严格约化成本下界不小于 0 时为真。

## 5. Phase-I/Farkas 定价

当当前 RMP 因分支等式缺列而不可行时，建立弹性 Phase-I。对 Phase-I 对偶执行同一精确定价器。普通目标定价的 `PRICING_EPS` 不能直接用于 Farkas 不可行剪枝：若弹性 RMP 的验证对偶下界为 `L_P1`、所有遗漏路线约化成本的严格下界为 `δ`、节点路线 LP 总质量上界为 `M_n=|I\F_n|`，则只使用 `L_P1 + M_n·min(0,δ)` 作为完整隐式 Phase-I 目标下界。仅当该下界严格大于 `ART_TOL` 才证明完整节点不可行。Farkas 阶段会保留幅度小于普通 `PRICING_EPS` 但仍真实为负的列；若完整空间人工目标下界不能证明为正，则节点保留并 fail-closed，不能按不可行剪枝。

## 6. 分支策略

按顺序使用：

1. 服务量 `s_i=0/1`；
2. 有向弧流 `z_ij=0/1`；
3. 唯一路线签名 `x_r=0/1`。

forbidden 条件直接进入定价允许路线过滤；required 条件不作为“所有列必须包含”的过滤器，而作为聚合 RMP 等式进入主问题，其 equality dual 完整进入 pricing reduced cost。每一对分支互斥且完备。

## 7. 实体资源 Logic-Based Benders

每个整数 RMP 候选调用 `step11_algorithm_route_drcc.audit_resource_assignment`。它在“输入 binary64 值解释为精确有限模型数据”的语义下处理 UAV 身份、电池身份、SOC 累计、任务衔接、快速检查、换电、甲板和工位容量，返回 `FEASIBLE / INFEASIBLE_PROVEN / UNKNOWN_TIMEOUT`。正式资源可行域不使用正的工程 tolerance 放宽约束：活动/甲板区间严格采用 `[a,b)`，UAV 后继任务严格要求 `start>=ready`，每块电池的 `E_soc_required_Wh` 用 `Fraction.from_float` 精确累计并严格比较 `<=B_use`。RMP 的 active/deck 必要容量行使用未 round 的真实离散事件时刻和相同的半开 membership，因此主问题必要行与最终实体资源 oracle 针对同一个时间模型。

当前快检/换电事件由 UAV 链中的直接前序任务决定，删除任务会改变前序、服务模式及事件时刻，所以资源可行性不保证向下封闭。对 `INFEASIBLE_PROVEN` 的选择集合 `S`，正式算法加入

\[
\sum_{r\in S}x_r-\sum_{r\notin S}x_r\le |S|-1.
\]

对二元变量，只有“恰好选择 `S` 且不选择任何其他路线”会违反该式。所有后续生成的新路线自动属于 `r\notin S`，其行系数为 `-1`，因此可行超集不会被删除。`UNKNOWN_TIMEOUT` 不加割、不剪枝、不更新 incumbent。

### 7.1 Resource-aware timing enrichment

当当前 incumbent 仍有未覆盖 turbine 且 discovery 轨迹满足触发条件时，R-BPC 可对这些 turbine 搜索 archive 中尚不存在的 singleton exact timing variants \((\tau,h)\)。候选先用与 exact resource oracle 一致的固定半开 deck interval 做兼容性预筛，再调用原 whole-route physical/DRCC evaluator；只有完整物理可行且 canonical signature 新颖的 route 才能进入 archive/RMP。

该 enrichment 是 primal/discovery 加速器而不是定价证书：没有找到 variant 不能证明 omitted route 不存在；其失败不能降低 UB、不能 prune、不能证明 infeasibility 或 pricing closure。被发现的列一旦通过原物理 evaluator，则与其它合法 column 一样可被后续正式 RMP、branching 与 resource audit 使用。

### 7.2 Exact generated-column primal recovery

令当前已生成且通过正式路线语义检查的列集合为 \(\mathcal R_G\subseteq\mathcal R\)。R-BPC 在 root 初始化及成功加列后，可在短预算内对 \(\mathcal R_G\) 求 restricted exact coverage problem，以恢复当前 archive 内更强的整数 witness。候选 witness 必须再次通过 unchanged exact entity-resource audit，只有 `FEASIBLE` 才允许更新正式 incumbent/LB。

因为 \(\mathcal R_G\subseteq\mathcal R\)，一个经 exact audit 接受的 restricted witness 也是 full-space 合法可行解，所以其 coverage 可以安全提高 LB。反之，restricted archive 的 optimum/UB 只对 \(\mathcal R_G\) 有效，绝不进入 full-space UB、node pruning、pricing closure、infeasibility 或 optimality certificate。

## 8. 节点界与全局界

定价完整时使用闭合 RMP 对偶界。定价因全局时间终止时，使用遗漏列约化成本的严格下界修正 RMP 拉格朗日界。若连严格定价 bound 都没有，则覆盖退回节点允许风机数，能耗退回由 `E_plan>=0` 保证的零下界。

开放节点覆盖上界取最大值，能耗下界取最小值。任何受限列池 MIP Gap 都不参与 `global_discrete_physical_model` Gap。`pricing_bound_available` 只说明界是否直接来自定价目标界；即使该字段为假，严格的节点允许风机数/非负能耗回退仍可令 `implicit_route_space_bound_valid=True`，两者不得混同。

## 9. Anytime 终止

一个全局 deadline 覆盖初始列、RMP、Phase-I、精确定价、分支树、资源 DFS 和第二阶段；`_candidate_from_physics`/`route_feasible_at_h` 接收相同绝对 deadline，并在回收 horizon 与物理内部循环合作式检查。任意外部阻塞黑盒无法由普通 Python deadline 硬抢占，因此合同明确为 cooperative：最多允许一次不合作黑盒调用级的时间偏差，不能宣称硬实时。返回状态包括：

- `lexicographic_optimal`；
- `coverage_optimal_energy_gap_target_reached`；
- `coverage_optimal_energy_time_limit`；
- `gap_target_reached`；
- `time_limit_feasible`；
- `solver_error`。

状态不再使用默认 `else` 冒充 Gap 达标。`gap_target_reached` 只有在阶段停止原因明确为 `coverage-gap-target-reached` 且数值 Gap 确实满足用户停止目标时返回；能耗 Gap 同理。**Gap 目标只是 anytime 停止条件，不是 exact optimality 证明条件**：只要严格全局界与 incumbent 之间仍有正缺口，`energy_optimal`/`lexicographic_optimal` 必须保持假。整数 RMP 只提供一个 incumbent；若完整空间严格节点界仍允许改善，则不得因 RMP 整数而 fathom，必要时使用路线变量 `x_r=0/1` 兜底分支消除 LP 数值最优性歧义。Phase-I 无效、Farkas 异常、RMP 非正常停止或定价评估异常均返回 `solver_error`，除非共享墙钟 deadline 已明确触发。

正式集合打包模型自然允许零路线计划。空方案始终是覆盖 0、计划能耗 0 的资源可行 incumbent；当全部非空路线均不可行时，它可被证明为词典序最优。时间中止时也不得把空方案称为“无可行解”。结果显式设置 `empty_plan_allowed=True`。

## 10. 研究基线隔离

`solver_mode="research-baseline"` 可解给定列池或在小实验中建立预生成列池，但始终返回：

```python
bound_scope = "validated_route_pool"
global_certificate_available = False
global_route_space_certificate = False
implicit_route_space_certified = False
```

它不得与正式隐式全路线模型的界或 Gap 混用。完整路线子集枚举只允许在 `selftest.py` 的极小 oracle 中出现。


## 11. 连续速度功率包络

`wait_and_speed` 不再用有限网格最大值近似连续速度区间。对 legacy 立方功率，最大值精确位于区间端点。对 Zeng 功率模型，可将导数写成 $V$ 与一个单调不减括号项的乘积，因此导数至多从负变正一次，功率曲线为“先降后升”的单谷函数，任意闭区间最大值严格位于两个端点之一。代码直接取两个端点功率最大值并加入向外浮点舍入保护。若功率参数非有限或违反非负物理口径，正式路径 fail-closed。该实现既不依赖速度网格，也不会引入原先组件分离式保守松弛。


### κ 参数可重入性与机器级定价下界（第四轮终审修复）

正式 exact BPC 把已校验 `kappa_mode` 转成不可变 `RiskPolicy(mode, one_sided, two_sided)`；SOC、联合天气 SOC、geo2d 双侧盒、固定接地可调空速与路线总判定都显式接收同一对象。正式证书链不读取/改写 `RM.kappa`。缺省 `RiskPolicy` 的旧调用只作为研究/兼容接口。

约化成本证书采用 binary64 向外区间算术：每个 `dual*coefficient` 乘积由相邻 representable float 包络，每次区间加法再次向 `±∞` `nextafter`。因此即使 `10^12` 量级对偶相消得到 `10^-3` 量级约化成本，下端点仍不高于把输入 binary64 视作精确实数时的真实约化成本。完整定价返回最小 lower endpoint；universal pricing bound 也使用相同向下舍入。若发生非有限输入/溢出，`pricing_bound_available=False`，节点只可使用平凡安全界。


数值证书补充：RMP 拉格朗日下界与 `L_RMP + M·δ` / Phase-I 组合采用 binary64 向下有向舍入；任一非有限或溢出情形均 fail-closed。
## 12. 直接模块执行与历史代码隔离

文件前部旧 B&P/Big-M 演示保留为研究参考，但其旧 `__main__` 块被永久关闭。`python step12_branch_price.py` 只打印正式库入口与 `step13_experiment_model.py` CLI 指引；`step11_algorithm_route_drcc.py` 同理。正式证书不能通过“直接运行旧脚本演示”产生。

## 严格实体资源层

资源 DFS 的“exact”指当前 finite binary64 model 上的严格整数可行性，而不是带工程 tolerance 的近似判定。设任务 `j` 的活动区间为 `[s_j,e_j)`，则同一 UAV 上前后任务必须满足
\[
s_{j_2}\ge e_{j_1}+q
\]
（快检复用）或
\[
s_{j_2}\ge e_{j_1}+w
\]
（换电），没有 `-10^{-7}` 的时间放宽。连续容量检查只在真实非空区间开始事件处检查 `a <= t < b`。

若电池 `b` 已累计 SOC 需求为 `U_b`，候选任务需求为 `e_j`，则可行条件严格为
\[
U_b+e_j\le B_{
m use}.
\]
代码将输入 binary64 能量转成 `Fraction` 后累计，因此不会因普通浮点求和或 `+10^{-6}` 松弛把严格超容方案判为可行。

RMP 中甲板/UAV 必要容量行使用与资源 DFS 相同的未 round 事件时刻。即使这些必要行遗漏某种复杂转换冲突，最终整数 incumbent 仍必须通过上述严格 DFS；只有 `INFEASIBLE_PROVEN` 才能生成 exact-pattern 资源割。



## Strict-physical v2 的身份与续跑边界

当前 formal solver 将 `finite-route-model-strict-physical-v8-discrete-recovery-target-xi-only-coherent-weather`、`strict-binary64-feasibility-no-positive-tolerance-v2` 和 `binary64-exact-weather-route-identity-v2` 纳入模型/结果 provenance。天气路线指纹使用 deterministic binary64 exact 状态序列化，不使用 `round(..., n)` 或进程相关的 `id()`。实验续跑除逻辑键外还绑定实际输入的 `resume_input_sha256`；一 ULP 的签名差异即视为不同运行。历史正证书若缺少 model/parameter/instance/algorithm 哈希，或 strict contracts 不是当前版本，必须 fail-closed，不能作为当前结果跳过求解。正式完整列空间 materialization 与 implicit pricing 共享严格任务窗和显式 `RiskPolicy`；materialization 的 launch option/horizon identity 保持 binary64 精确，不使用 decimal round 或固定能耗 dominance tolerance。formal Xi loader 不再容许尺度相对 PSD tolerance：由于 Xi CSV 以 `sigma_ee/sigma_en/sigma_nn` 唯一定义对称 2×2 矩阵，它按 binary64-as-real 精确 PSD 条件校验；`h_min` 逐 binary64 精确命中统计网格，禁止 `round/int/nearest` 吸附。生成端 `step7` 保证写出的 covariance 在 binary64-as-real 下 PSD，并不再把 covariance round 到 0.1。天气 covariance 是独立的数据结构，仍允许内部线性代数产生的正常 ULP 级非对称，不要求逐 bit 对称；weather uncertainty horizon 继续要求 exact mapping。formal 统计数据契约还要求 `purge_min>=max_horizon` 且 nonoverlap 采用严格 `t0>=previous_t1`，不使用正 tolerance。正式求解入口会再次验证实际 XiCell 的 key/state、有限性、对称性与 binary64-as-real PSD，因此绕过 CSV loader 也不能把非法 ambiguity set 带入全局证书；层级收缩与 horizon 插值产生的内部矩则先 canonicalize 到 deterministic binary64 PSD。


### WeatherAmbiguity 输入
正式 BPC 不从 `weather_df` 临时估计天气误差。`--weather-moments-csv` 经 `weather_ambiguity_from_moments_csv` fail-closed 加载后成为模型输入，并进入模型哈希、定价可行性、DRCC 和 resume fingerprint。mechanism 模式仍可显式使用 `weather_ambiguity_from_series` 的 adjacent-reanalysis proxy，但其 `formal_eligible=False`。


## 13. 当前有限离散模型下“精确解”的判定

本项目所称精确解，是**当前输入和离散化共同定义的有限模型**的词典序最优解，而不是连续时间真实系统的全局最优。当前 `exact-branch-price-cut + r-bpc` 路径具备获得该精确解的完备证书机制：精确定价无截断穷尽有限有序路线前缀与离散回收时长；节点使用完整 Phase-I/Farkas 与严格遗漏列界；服务/弧/路线二分支构成完备划分；整数候选必须通过实体 UAV/电池/甲板的严格资源审计；只有 `INFEASIBLE_PROVEN` 才授权精确模式割。

正式结果必须同时检查：

- `coverage_optimal=True`：第一层最大覆盖已经对完整隐式路线空间证明最优；
- `energy_optimal=True`：在固定最优覆盖下，第二层计划能耗已经证明最优；
- `lexicographic_optimal=True`：两层词典序最优同时成立；
- `global_certificate_available=True`：当前参数、实例、模型和算法合同均允许发布完整有限模型证书。

普通 MILP/RMP 的 `optimal`、受限列池 Gap、`pricing_search_complete=True`、或用户设定的正 Gap 目标均不能替代上述条件。`time_limit_s` 是 anytime 截止时间：若在分支树/定价闭合前到期，算法仍返回资源可行 incumbent 与安全全局界，但不得称为精确最优。理论上有限路线空间和完备分支意味着在无外部中断、所有黑箱调用最终返回且资源允许的前提下搜索可穷尽；实际规模的主要瓶颈是当前 implicit DFS 最坏为排列数量级，因此“可证精确”不等于“任意大规模都能在给定时间内证完”。

## 14. Branch-Price-and-Cut、Column Generation 与“精确”的关系

当前正式算法不是“先枚举完整路线池再解 MILP”，也不是“Branch-and-Price 与 Column Generation 二选一”。其节点内部本身就是标准列生成循环：

\[
\text{RMP}\rightarrow\text{dual}\rightarrow\text{pricing}\rightarrow\text{new columns}\rightarrow\text{RMP}.
\]

Branch-and-Price 是在 Branch-and-Bound 的每个节点重复上述列生成；再加入由严格实体资源 Oracle 授权的全局有效 exact-pattern cuts，得到 Branch-Price-and-Cut。因此更准确的正式名称是：

> **Exact Branch-Price-and-Cut with implicit route column generation and exact pricing.**

这里三类操作必须区分：

1. **按需列生成**：避免在求解开始前把指数规模的完整路线集合物化成 master 变量；
2. **启发式/多列加速**：如果未来加入 route pool、heuristic pricing、multi-column generation 或 dual stabilization，它们只能更快地寻找合法改善列，不能独立证明“没有遗漏列”；
3. **精确定价认证**：节点只有在 exact pricing 完整覆盖当前分支允许的隐式路线空间，或取得对所有遗漏列都有效的严格 reduced-cost lower bound 后，才能形成完整列空间 LP bound 并参与安全 fathoming。

R-BPC 的 exact pricing 属于第 3 类。它允许 heuristic-first discovery 与批量回 Master，但在需要证书时仍对剩余路线空间完成严格闭合；最坏情况下仍进行完备的组合搜索。因此“精确”不等于“无枚举”；准确含义是：任何可能改善最优值的离散路线都必须被检查、被隐式覆盖，或被严格有效的界证明可以排除。

### 14.1 当前精确性与未来加速的兼容原则

未来可安全加入：

- route-pool warm start；
- heuristic pricing；
- multi-column generation；
- dual stabilization；
- 经证明安全的 RCSP/ESPPRC dominance。

但必须保持：

\[
\boxed{\text{heuristic pricing 负责找列，exact pricing 负责证明没有遗漏列。}}
\]

若某启发式没有找到改善列，只能说明“该启发式未找到”，不能把节点标记为 `pricing_closed=True`。如果未来用 RCSP/ESPPRC 替代当前 DFS，则每条 dominance、资源递推和 lower bound 都必须对当前 moving-recovery、Xi/Weather DRCC、能量及资源列系数证明等价或安全；否则只能作为找列加速器，不能取代最终认证 Oracle。

## 2026-08 证书控制流补强

整数 RMP 的“数值上接近 0/1”只用于发现候选。`np.rint(x)` 后，算法用 `Fraction.from_float` 对当前 RMP 的每条 inequality/equality 按 binary64-as-real 精确复核；复核失败时不调用资源 oracle、不更新 incumbent，而采用完备的路线变量 `x_r=0/1` 数值兜底分支，若没有合法分支则 fail-closed 保持节点开放。

当 exact pricing 搜索被统一 deadline 中断时，`pricing_closed=False`。但若中断前已有对所有未搜索列有效的严格 reduced-cost 下界 `delta`，则仍可使用 `L+M_n min(0,delta)`，其中 `M_n=|I\F_n|` 得到完整隐式空间节点界；只有该界已经不能改善当前资源可行 incumbent 时才 fathom。否则节点重新入队并终止本次搜索，不把 timeout 当作“无改善列”。Multi-column generation 与共享 route archive 已启用；batch size 只限制每轮加入 RMP 的列数，不限制 exact pricing 扫描空间。


## 15. Route-universe provenance 与 Full-Space Correction

正式 exact BPC 的实际 route universe 必须由 `_candidate_from_physics -> route_feasible_at_h` 定义。公开入口不接受 synthetic `implicit_test_columns`；私有算法 fixture 可以穷举人工有限列集，但只能签发 algorithmic certificate，不能签发 physical-model certificate。最终发布关系为：

\[
G_{physical}=G_{algorithmic}\land P_{route\ universe}\land P_{semantic}\land P_{row\ ranges}\land P_{model/numeric}\land P_{proof\ contract}.
\]

节点 `n` 的 forbidden-service 集合记为 `F_n`。因每条正式路线非空且 node-compatible 列不访问 `F_n`，packing 给出

\[
\sum_r x_r\le M_n:=|I\setminus F_n|.
\]

`L_n` 不是简单复制 solver 的 dual objective。实现先把 inequality marginal 投影到 `y<=0`，再对当前 restricted route variables 的合法证书 box 完整求 Lagrangian infimum；普通 route 的 `[0,1]` cap 由非空路线和 packing 冗余保证。因而即使 binary64 multiplier 不是数学实数意义下精确 stationarity，`L_n` 仍由弱 Lagrangian duality 保证是 lower bound。solver KKT/strong-duality 检查是额外 sanity gate。

若所有遗漏列 reduced cost 有统一严格下界 `delta_n`，则普通目标与 Elastic Phase-I 共用同一修正结构：

\[
z^*_{full,n}\ge L_n+M_n\min(0,\delta_n),
\qquad
\Phi^*_{full,n}\ge L_{P1,n}+M_n\min(0,\delta_{P1,n}).
\]

future-column reduced-cost lower bound 只能使用显式登记的 master-row coefficient range；未知 row 不能默认贡献非负。required service/arc/route 是聚合等式，例如 required arc `(i,j)` 是 `sum_r e_{ijr} x_r=1`，所以不含该弧的其他路线仍然 admissible；其影响通过 equality dual 进入 reduced cost。forbidden branch 才是 route-admissibility filter。

## 16. Proof-to-Code 执行合同

当前 formal proof contract 为 `exact-bpc-proof-code-concordance-v9-source-bound-global-battery-relaxation`。它只升级算法/结果证书语义，不改变 `MODEL_SEMANTICS_CONTRACT`。代码把以下 ID 与函数锚点写入 `FORMAL_PROOF_CODE_ANCHORS`，并将其 SHA 纳入算法 fingerprint：

| ID | 运行时职责 | 核心函数 |
|---|---|---|
| `THM-RU` | formal route universe provenance | public/private route-universe gate + `_exact_pricing_search` |
| `LEM-CS` | route identity 的 formal semantics 不可变 | `_column_semantics_fp`, `_add_columns` |
| `THM-LRC` | RMP Lagrangian LB、future-row range、遗漏列 correction | `_lagrangian_dual_lower_bound`, `_universal_pricing_lower_bound`, `_safe_node_bound_from_pricing` |
| `COR-P1` | 完整空间 Phase-I infeasibility | `_solve_elastic_phase_one`, `_phase_one_infeasibility_proven` |
| `LEM-PAT` | exact-pattern resource cut/current+future coefficient | `_row_coefficient`, resource audit |
| `THM-BR` | forbidden filter + required equality + 完备兜底分支 | `_column_allowed_at_node`, `_master_rows`, branching functions |
| `THM-NUM` | outward RC/bounds + exact-rational binary master/energy | LP validator、RC interval、integer gate、energy exact sum |
| `THM-LEX` | 两阶段树闭合与最终 physical guard | `_solve_branch_price_stage`, `_physical_certificate_guard` |
| `THM-TGT` | fixed-target YES/NO 决策与证书边界 | `_target_infeasibility_algorithmic_proven`, target decision assembly |
| `THM-CU` | complete materialized route universe 的完备性与 provenance | `build_certified_route_universe`, `_validate_certified_route_universe` |
| `THM-FCT` | full-cover partition、resource closure 与 independent exact master verifier | `_solve_complete_universe_fullcover_target`, `_exact_fullcover_master_feasibility` |
| `THM-GBR` | universe-level exact battery-energy necessary relaxation | `_exact_global_fullcover_battery_relaxation`, `_solve_complete_universe_fullcover_target` |

实际执行顺序为：

```text
validate finite binary64 model/contracts
        ↓
[THM-RU] establish formal route-universe provenance
        ↓
[LEM-CS] every inserted route preserves immutable column semantics
        ↓
RMP solve + independent KKT sanity validation
        ↓
[THM-LRC] outward Lagrangian LB + exact/universal pricing
        ↓
[COR-P1] when RMP infeasible, elastic full-space feasibility proof
        ↓
[THM-BR] branch or [LEM-PAT] resource cut, both future-column compatible
        ↓
[THM-NUM] exact binary candidate/resource/energy gates
        ↓
coverage tree safe closure
        ↓
fixed C* energy tree safe closure
        ↓
[THM-LEX] explicit physical certificate conjunction
```

最终 `_physical_certificate_guard` 对应 `doc_proof.md` 的 `[THM-LEX]`，明确要求 algorithmic tree certificate、physical route provenance、route semantic invariant、future-row range contract、binary64/model contract 与 proof-contract fingerprint 同时成立。任何一项失败只允许保留 incumbent/安全 bounds，不能将 `global_certificate_available` 提升为真。




---

## Exact-pricing acceleration telemetry and certified prefix pruning

The canonical R-BPC CLI remains `--solver-mode exact-branch-price-cut --pricing-mode r-bpc`; historical long pricing modes remain available for regression/reproduction. The canonical implementation now exposes per-pricing-call, RMP, and exact-resource-audit telemetry needed to attribute certificate time without changing solver decisions.

In certification pricing only, R-BPC may discard a prefix subtree when its outward-safe reduced-cost lower bound is nonnegative. Stage-2 energy pricing additionally couples mandatory inspection/climb service energy with the fixed-coverage equality dual by possible final route cardinality. This is a formal lower-bound strengthening, not a heuristic score: the route space, physical evaluator, DRCC, exact SOC audit, and strict improving-column test `rc_UB < 0` are unchanged. Discovery/timing enrichment remains discovery-only and cannot use this mechanism to claim pricing closure.

The public result now includes `pricing_call_records`, `rmp_records`, and `resource_audit_records` together with aggregate timing counters. RMP records explicitly report the current SciPy/HiGHS rebuild behavior (`persistent_model=False`, `basis_reuse=False`) so persistent-model work can be evaluated from measured RMP share rather than assumed to be a bottleneck.

For the paper fixed-point entry `E1_lex_certify`, the same telemetry is persisted to `results/model_experiments/E1_lex_certify/E1_lex_certify_telemetry.json`; the compact certificate CSV also retains the aggregate timing/cache/prune counters.

## 12. 加速架构（v3.1：不改变数学输出的实现层加速）

v3.1 对求解器做了五项实现层加速。它们全部满足"逐位一致"或"集合等价"，
证书语义零变化；证明与验证记录见 `docs/doc_proof.md`
（THM-001..007）与 `results/experiments/`。

1. **P1 resource_pattern compat 删除（THM-002）**：该行两分支的 `coeff_lo`
   恒为 -1.0，而不等式下界只消费 `coeff_lo`——指纹兼容性扫描对任何下界数值
   无影响，直接删除。
2. **P2 per-launch 指纹记忆化（THM-003 引理 3.1）**：`_ship_column_fp` /
   `_wx_fp` 为内容纯函数，单次定价调用内按 launch 记忆化，值逐位不变。
3. **P3 编译型前缀下界（THM-001 定理 1）**：对偶固定后行项预编译一次，
   按原行序累加（含零项）——与原实现逐位相同；未预期行类自动回退原函数。
4. **P5 精确缓存特征定价（THM-006）**：超集分支节点自然完成一次完整枚举后，
   后续定价调用直接在物理缓存快照上取候选列，经未改动的 `consider()` 做
   节点过滤/rc/改进判定——considered 列集合、bound、闭合判定与 DFS 逐调用
   一致（定价树只走一次）。selftest 直调传普通 dict 时该路径自动关闭。
5. **P6 RMP 矩阵构建不变量提升（THM-007）**：每列的签名/tid 集合在行循环
   外计算一次，行内系数表达式逐字复制 `_row_coefficient`——矩阵逐位相同
   （修复 no-good cut 累积后每次 LP 重解重复计算路由签名的瓶颈）。

主实例实测（Rodsand II n=10/M/K=2/B=7/S=4）：19926.7 s → 2179.0 s（9.1×），
C\*、E\* 与全部证书字段逐位一致；n=15 实例 3.0 h 完成树穷尽证明。
