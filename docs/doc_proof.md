# 精确算法证明摘要

本文件只保留当前仍承担证书责任的定理、引理与 proof-to-code obligations。旧 contract 字符串、版本迁移说明和已被取代的证书身份已删除。

当前：

```text
FORMAL_PROOF_CONTRACT
= exact-bpc-proof-code-concordance-v9-source-bound-global-battery-relaxation

FORMAL_PROOF_CONTRACT_SHA256
= baf6e6d0e6d4fa9e513d21cd32ad2c4e5f349ea564c079d95ff4c56d0e9fc766

RESULT_CERTIFICATE_CONTRACT
= finite-binary64-physical-route-universe-certificate-v12-source-bound-global-battery-relaxation
```

`global_certificate_available=True` 必须由 algorithmic closure、physical route-universe provenance、semantic invariance、future-row range、binary64/model contract 与 proof contract 的合取共同授权。


## 0. 当前 proof contract：源码绑定与证书作用域

当前 formal proof contract 绑定 Step9–Step12 的**实际源码字节 SHA-256**，而不仅是字符串合同；algorithm hash 同时携带该 code digest。任何 proof-critical 源码变化都会改变证书身份。resume context 进一步绑定当前 source-tree SHA，旧源码状态下的 `anytime-bounds-only` 也不得作为完成的正证书跳过。

最优性措辞同时收紧：Step11 研究 restricted pool 只能给出 `restricted_pool_lexicographic_optimal`，其 legacy/global `lexicographic_optimal` 必须为 false；只有 complete route universe + complete pricing + exact certificate 闭合的正式路径才能声称 finite-model global lexicographic optimality。统计可靠性证据与优化 proof 分离：final-test Hoeffding–Azuma UCB 不证明 BPC exactness，BPC certificate 也不证明 IID independence 或 mission-start nonanticipativity。


## 0.1 THM-GBR — Global Battery Relaxation

**THM-GBR.** 设 `R` 是当前 certified complete physical route universe，full-cover target 为 `T=|I|`。每条 route `r` 服务非空集合 `S_r`，并有 exact binary64-as-rational SOC demand `e_r`；每块实体 battery 的 usable capacity 为 `C=B_use`。构造松弛：保留 exact-cover partition `sum_{r:i in S_r} x_r = 1` 与 selected routes 向 `B` 个容量 `C` 的 identical bins 分配的累计能量约束，删除 UAV identity/binding、时序、甲板、快检/换电站及其余资源限制。令 `B_relax*` 为该松弛覆盖 full mask 所需最少 bins。则任何原资源模型的 full-cover feasible assignment 都诱导一个松弛可行 packing，所以

\[
B_{\mathrm{relax}}^\star > B \quad\Longrightarrow\quad
\text{原 full-cover target INFEASIBLE}.
\]

证明的 load-bearing move 是**可行映射方向**，不是“能量松弛可行即真实可行”。真实 assignment 中每条 selected route 已绑定某个实体 battery，且该 battery 的 exact SOC 累计不超过 `C`；删除其它约束后仍满足同一 energy packing，因此原可行集投影包含于松弛可行集。反命题不成立。

实现用两个有限 bitmask DP 完全枚举这个松弛：`best_energy[M]` 是由 pairwise-disjoint route masks 恰好覆盖 `M` 的最小 exact rational SOC energy；`one_pack[M]` 当且仅当该值 `<=C`；第二个 DP 用包含 first-uncovered-bit 的 bundle 消除排列对称，求 full mask 最少 bundle 数。所有能量和容量来自 `Fraction.from_float`，没有 `epsilon`、decimal rounding 或 floating solver infeasibility。`UNKNOWN_TIMEOUT` / `SKIPPED_SIZE` 均不得推出 NO。

正文若引用 THM-GBR，只需保留：松弛定义、真实可行解到松弛解的映射、`B_relax*>B => NO` 三步 proof checkpoint；DP 递推和边界验证可留在附录/本文件。当前机器可读 proof anchors 位于 `FORMAL_PROOF_CODE_ANCHORS["THM-GBR"]`。


## 0.2 Persistent exact resource closure corollaries

THM-FCT 的 full-cover partition 与 full-pattern strong cut 证明不变。当前实现保留两个 proof-critical corollaries：其一，每个 active sortie 必占用一个实体 battery，所以任何时刻 active concurrency `<= min(K,B)`；任一 UAV 在 route clear 后至少还需 `min(t_quick,t_swap)`（B=1 时为 `t_quick`）才能再次 launch，因此相应 half-open fastest-turn intervals 的 capacity-K clique rows 是必要条件。其二，把所有 temporal/UAV/binding/station constraints 删除后得到 B-bin battery-energy relaxation；若 route subset Q 在 exact binary64-rational energies 下仍无法装入 B 个容量 B_use 的 bins，则任何包含 Q 的真实 resource assignment 都不可行，所以 `sum_{r in Q} x_r <= |Q|-1` 是全局有效 cut。删除元素直到不能再删只影响 cut 强度，不影响有效性。

Persistent ledger 不把“上次 UNRESOLVED”本身当作证明，只序列化已经由上述 exact relaxation 或 exact resource DFS 证明的 cuts；context 绑定 immutable column semantics 与全部 resource 参数。`UNKNOWN_TIMEOUT` 从 battery relaxation、resource audit 或 independent exact-cover verifier 任一路径出现，都不得生成 cut/NO。


## 0.3 THM-FCT — complete-universe full-cover target resource closure

固定一个已通过 `THM-CU` 的 finite binary64 instance，令完整正式物理 route set 为
\(\mathcal R\)，master turbine set 为 \(I\)，target \(T=|I|\)。

### 1. Full-cover equality

原 packing 满足

\[
\sum_{r\in\mathcal R}a_{ir}x_r\le1,\quad i\in I,
\]

且 target row 为

\[
\sum_{r\in\mathcal R}|S_r|x_r=|I|.
\]

把第一组不等式对 \(i\) 求和，左侧正好等于第二式左侧，因此所有 packing slack 的和为0；
每个 slack 非负，故逐项均为0：

\[
\sum_r a_{ir}x_r=1,\quad\forall i.
\]

因此 binary master 显式使用这些 equality 与原 target feasible integer set 等价。

### 2. Full-cover resource strong cut

设 binary master 给出 pattern \(S=\{r:x_r=1\}\)，且 unchanged exact resource audit 返回
`INFEASIBLE_PROVEN`。因为 \(S\) 已逐风机恰好覆盖一次，任意另一个 full-cover set-packing
整数解若包含全部 \(S\)，就不能再加入任何非空 route，否则至少一个 turbine packing row
违反 equality。故在 full-cover target 域上

\[
\sum_{r\in S}x_r\le |S|-1
\]

只排除这个已证明资源不可行的整数 full-cover pattern，同时比全变量 Hamming 表示具有更强
LP 松弛。一般 partial-target resource feasibility 不保证向下封闭，因此该割不得外推。

### 3. NO certificate

`THM-CU` 保证 binary master 含全部合法 route variables；每个新增 strong cut 都由
`INFEASIBLE_PROVEN` full-cover pattern 导出。若有限 binary master 最终被 backend 证明
infeasible，则不存在未枚举 route 或未处理 full-cover integer pattern，故 target NO 成立。
若 MILP status 未证明 infeasible、resource audit 为 `UNKNOWN_TIMEOUT`、universe 不完整或
哈希/语义不一致，则不得签 NO。


## 0.4 THM-CU — 完整物理路线宇宙的 exact acceleration

固定一个 formal finite instance。令 \(\mathcal R\) 为原 implicit exact pricing 定义的所有
物理/DRCC 可行路线。`build_certified_route_universe` 按相同 launch option、有序 elementary
sequence 和 recovery horizon 域穷尽，并对每个候选调用相同的正式物理 evaluator。

若构建自然结束、所有候选域均已遍历、仅使用已证明的必要条件 prefix prune，且
`_route_archive_semantics_invariant`、context SHA 和 columns semantics SHA 均通过，则返回
`complete=True` 且其列集 \(\widehat{\mathcal R}\) 满足

\[
\widehat{\mathcal R}=\mathcal R.
\]

这就是 `THM-CU` 的核心等价性。timeout、evaluator error、reach proof 不完整或语义 invariant
失败均不能推出该等式；实现分别回退到全部 turbines 或返回 `complete=False`。

因此在一个 branch node \(n\) 上，应用 branch filters 后的 RMP 已包含所有
\(\mathcal R_n\) 列：
- 若 RMP infeasible，则 \(\mathcal R_n\) 上不存在 master-feasible 点，可直接 fathom；
- 若 RMP optimal，则不存在 omitted columns，pricing lower bound 为 \(+\infty\)，
  reduced-cost closure 自动成立；
- 若整数选择的 exact resource audit 为 `INFEASIBLE_PROVEN`，仍只加入 exact-pattern
  Hamming cut，然后在同一完整列集继续 branch-and-cut；
- `UNKNOWN_TIMEOUT` 仍不能产生 cut 或 certificate。

故 `THM-TGT` 的 NO 方向在 materialized 模式可由“完整列集 branch-and-cut 树闭合”签发，
无需每节点 Phase-I/Farkas omitted-column search；implicit fallback 的原 Phase-I/Farkas
证明保持不变。`THM-CU` 不证明 Stage-2 optimality，最终 lex certificate 仍需 `THM-LEX`。


## 1. 风机互斥集合打包

每个路线内部风机唯一，主问题逐风机约束 `sum a_ir x_r <= 1`。因此任何整数解中不存在跨路线重复巡检，唯一覆盖数等于 `sum |S_r| x_r`。

## 2. 精确整数模式资源割

当前资源 DFS 将快检或换电事件放在 UAV 直接前序任务的清场时刻之后。删除任务会改变 UAV 前序、电池转换模式和服务事件时刻，因此资源可行性**不保证向下封闭**；不能由 `S` 不可行推出其所有超集不可行。经典子集割

$$
\sum_{r\in S}x_r\le |S|-1
$$

在本模型中不具备一般有效性。

设 `S` 是一个经完整 DFS 证明不可行的整数路线选择模式。正式主问题加入

$$
\boxed{
\sum_{r\in S}x_r-\sum_{r\notin S}x_r\le |S|-1
}
$$

其中第二个求和覆盖完整隐式路线空间；后续生成的任何非 `S` 路线都自动取得系数 `-1`。对任意二元向量，记 `a` 为 `S` 中被选择的路线数，`b` 为 `S` 外被选择的路线数，则左端为 `a-b`。由于 `a\le |S|`，只有 `a=|S|` 且 `b=0` 时左端大于 `|S|-1`。因此该式只排除恰好等于 `S` 的已证不可行模式，所有真子集和真超集均保留，是完整隐式整数模型的全局有效逻辑割。

`UNKNOWN_TIMEOUT` 没有不可行证明，不能添加任何资源模式割。

## 3. 精确定价完整性

对每个允许起飞选项，搜索完整扩展全部有序无重复前缀直到 `max_stops`，并对每个非空序列评估全部离散回收时长。没有启发式删标或候选截断。由有限树的归纳可知：搜索正常结束时，每个节点允许的隐式路线状态均被评估一次，因此最小约化成本精确。该证明依赖穷尽有限排列树，不依赖 RCSP dominance；当前实现没有非平凡状态合并，最坏复杂度为排列数量级。

`tau_reach` 只有在其证明对象完整且排除条件自洽时才被使用；否则定价自动恢复为全部风机集合，所以它不能删除未经证明的可行路线。


## 3.1 RMP 对偶与第一阶段约化值

代码统一使用最小化 RMP。记上界行 $A^{\le}x\le b$ 的 SciPy/HiGHS marginal 为 $u\le0$，等式行 $A^=x=d$ 的 marginal 为 $v$（自由号）。定义 $\pi=-u\ge0$。第一阶段列目标为 $c_r=-|S_r|$，代码中的 reduced cost 是

$$
\bar c_r^{C}=-|S_r|-u^\top A_r^{\le}-v^\top A_r^=
=-|S_r|+\pi^\top A_r^{\le}-v^\top A_r^=.
$$

对应的 reduced profit 为

$$
\bar p_r=-\bar c_r^{C}
=|S_r|-\pi^\top A_r^{\le}+v^\top A_r^=.
$$

代码不再用 `best_reduced_value` 点估计签发改善性或闭合证书。对每条遗漏列计算严格向外区间 $[\bar c_L,\bar c_U]$，**正式证书路径的阈值是精确的 0，而不是 $-\varepsilon_{price}$**：只有 $\bar c_U<0$ 才证明存在严格负约化成本改善列；完整搜索后只有所有遗漏列的严格下界满足 $\bar c_L\ge0$ 才可证明定价闭合。若 $\bar c_L<0\le\bar c_U$，则列可作为中性增补加入 RMP 以取得有限进展，但不能据点估计宣称改善。`PRICING_EPS` 不参与 formal closure，只保留为启发式/显示尺度。

## 3.2 第二阶段约化成本

固定覆盖等式的自由对偶包含在 $v$ 中。第二阶段列目标为 $c_r=E_r^{plan}$，故

$$
\bar c_r^{E}=E_r^{plan}+\pi^\top A_r^{\le}-v^\top A_r^=.
$$

其中 $A_r^{\le}$ 包括风机集合打包、甲板/活动容量、池化 SOC 必要条件与精确资源模式行，$A_r^=$ 包括固定覆盖及节点要求服务、要求弧和要求路线行。资源模式行的列系数为 $+1$ 或 $-1$，因此不能沿用“所有上界行系数非负”的简化。若该行 HiGHS marginal 为 $u_g\le0$，则任意遗漏列的该项约化成本贡献为 $-u_ga_{gr}$，其中 $a_{gr}\in\{-1,+1\}$，其严格通用下界为 $u_g$。代码的 `_universal_pricing_lower_bound` 对每条资源模式行加入该项，保证定价中止时的 $\delta$ 仍对未来列有效。当严格区间上端满足 $\bar c_U^E<0$ 时该列被证明为改善列；完整搜索的严格下端点证明最小约化成本不小于 0 时才闭合。区间跨越 0 的列只作中性 RMP 增补，不作为改善性证据。


### 3.3 数值模糊与终止性

正式证书阈值统一为 $\theta=0$。对遗漏列若 $\bar c_L<0\le\bar c_U$，binary64 向外区间无法证明其符号。旧逻辑若同时看到点估计 $\widehat{\bar c}<0$ 而又没有可证明改善列，会在完全相同的 RMP 上无限 `continue`。当前实现把一批此类**新签名、物理合法**列加入 RMP。增加列只扩充受限主问题到更接近完整主问题，不会切除任何可行解或伪造下界；下一轮这些签名被视为已存在，因此每次模糊处理至少消耗一个有限路线状态。故在有限离散路线空间上，该数值分支具有有限组合进展。节点/全局证书仍只使用严格下端点 $\bar c_L$。

## 4. Phase-I/Farkas

弹性 Phase-I 对任意当前列池可行。设验证后的弹性 RMP 对偶下界为 `L_P1`，遗漏路线最小约化成本有严格下界 `δ`。由每条路线至少覆盖一台风机及 set-packing，在节点 `n`，forbidden-service 集为 `F_n`；node-compatible 路线不访问 `F_n`，故 LP 松弛满足 `sum_r x_r <= |I\setminus F_n|=:M_n`，完整隐式 Phase-I 目标满足 `Phi_full >= L_P1 + M_n·min(0,δ)`。只有该完整空间下界严格大于人工容差 `ART_TOL` 时才可证明节点不可行。`best_rc >= -PRICING_EPS` 只是普通优化停止容差，绝不能单独作为 Farkas 不可行证明；例如 `L_P1=4e-7, M=2, δ=-3e-7` 时安全下界为 `-2e-7`，节点必须保留。若发现真实负约化成本列则加入；定价中断但 universal bound 已足以使上述完整下界严格为正时仍可证明不可行，否则一律未知/fail-closed。

## 5. 分支完备性

服务、弧和具体路线变量在整数解中均为二元。对任一二元聚合量的 `0/1` 分支互斥且并集为父节点整数解。禁止分支进入定价过滤，要求分支进入 RMP 等式和定价约化成本，因此后续列生成不会破坏分支语义。

## 6. 定价超时安全界

对代码的最小化 RMP，令经过 KKT/强对偶独立验证的当前对偶下界为 $L$，定价器给出所有遗漏列最小约化成本的严格下界 $\delta$。集合打包和每列至少服务一台风机给出

$$
\sum_r x_r\le\sum_r|S_r|x_r\le |I|,
$$

在节点 `n` 可取更紧的 $M_n=|I\setminus F_n|$（根节点退化为 $|I|$）。拉格朗日弱对偶给出完整隐式列空间下界

$$
z^*\ge L+M_n\min(0,\delta).
$$

覆盖阶段 $z=-C$，代码使用

$$
\overline C^{node}=\min\left(|I_{allowed}|,
\left\lfloor-\left[L+M_n\min(0,\delta)\right]+\varepsilon_{int}\right\rfloor\right).
$$

能耗阶段直接使用

$$
LB_E^{node}=\max\left(0,L+M\min(0,\delta)\right),
$$

其中零下界由列合同强制的 $E_r^{plan}\ge0$ 支撑。若定价没有有效 $\delta$，代码绝不把 RMP 值冒充完整模型界，而分别退回节点允许风机数和零能耗下界。

## 7. 全局分支树 Gap

开放节点覆盖上界取最大值，固定覆盖能耗下界取最小值：

$$
UB_C^{global}=\max_{n\in\mathcal O}UB_C^n,\qquad
LB_E^{global}=\min_{n\in\mathcal O}LB_E^n.
$$

对资源审计通过的 incumbent $(C^{inc},E^{inc})$，代码报告

$$
Gap_C^{abs}=\lfloor UB_C^{global}+\varepsilon_{int}\rfloor-C^{inc},
\qquad
Gap_C^{pct}=100\frac{Gap_C^{abs}}{\max(1,\lfloor UB_C^{global}+\varepsilon_{int}\rfloor)}.
$$

只有覆盖最优已证明，才定义完整词典序能耗 Gap：

$$
Gap_E^{abs}=E^{inc}-LB_E^{global},\qquad
Gap_E^{pct}=100\frac{Gap_E^{abs}}{\max(|E^{inc}|,\varepsilon_E)}.
$$

覆盖未闭合时能耗全局 Gap 必须为 `None`。节点被剪枝只可能因为完整 Phase-I/Farkas 不可行证明、严格节点界不优于 incumbent、已审计整数解/有效精确资源模式割，或完备二分支。资源未知节点保留在开放队列。因此 Gap 闭合时证书覆盖整个有限隐式路线空间。

## 8. 统一时间限制

所有过程共享单一 `time.monotonic()` deadline。物理路线判定也接收该绝对 deadline 并在内部循环合作式检查。Python 无法硬抢占任意不合作的第三方阻塞函数，因此墙钟合同是 cooperative 而非 hard real-time：最坏可能超出一次黑盒调用时长；一旦该调用返回，后续 horizon/节点不会重置预算继续大规模计算。时间终止只停止进一步搜索，不会改变已经审计通过的 incumbent，也不会使已构造的弱对偶界失效。未完成节点携带安全替代界重新进入开放队列，所以返回 Gap 仍有效。

## 9. 证书范围

证书只针对输入定义的有限离散模型，不覆盖连续时间、未枚举的决策网格外状态、未建模故障或真实平台绝对安全。统计可靠性声明仍需独立真实联合留出数据。


## 10. 空方案

集合打包约束、资源容量上界和非负能耗均允许 `x=0`。空方案不占用甲板、UAV、电池、快检或换电资源，因此资源审计必须返回 `FEASIBLE`。它提供覆盖下界 0 和能耗 0 的合法 incumbent；若精确定价与分支树证明不存在非空改善方案，则覆盖最优值为 0，固定覆盖 0 的第二阶段能耗最优值也为 0。

## 11. 重复物理签名

路线签名相同意味着覆盖、弧、时间区间与所有主问题系数相同。若重复副本的计划能耗与 SOC 需求均不大于已存副本，则它弱支配旧副本，可安全原位替换；反之若两个能量向量不可比，则说明签名遗漏了影响主问题的物理状态，算法必须 fail-closed，不能静默丢弃任一列。


## 12. 连续速度功率包络证明

令

$$
x=\frac{V^2}{2v_0^2},\qquad
f(V)=\sqrt{\sqrt{1+x^2}-x}=\exp\!\left(-\frac12\operatorname{asinh}x\right).
$$

对 Zeng 功率函数求导，可得

$$
P'(V)=s_pVg(V),
$$

其中

$$
g(V)=\frac{6P_0}{U_{tip}^2}+3c_dV-
\frac{P_if(V)}{2v_0^2\sqrt{1+x^2}}.
$$

在物理参数非负时，前两项单调不减。$f(V)$ 与 $(1+x^2)^{-1/2}$ 都是正的单调不增函数，因此其乘积也单调不增，负号后的第三项单调不减。故 $g(V)$ 单调不减，$P'(V)$ 至多发生一次由负到正的符号变化。于是 $P(V)$ 在非负速度域上先降后升或单调，任意闭区间最大值位于端点。代码计算

$$
\overline P=\max\{P(v_{lo}),P(v_{hi})\}
$$

并增加向外浮点舍入保护。`P_zeng` 在 $V<0.01$ 时取常数 $P(0.01)$，不会破坏端点结论。因此有限网格低估连续最大值的证书缺口已经消除，而且没有引入额外保守组件组合。

## 13. 公共状态分类

公开状态由已验证停止条件映射，不使用默认成功兜底。覆盖 `gap_target_reached` 必须同时满足：阶段原因精确为 `coverage-gap-target-reached`，且报告的严格覆盖 Gap 不超过用户目标。能耗目标同理。混合原因 `farkas-phase-time-limit-or-invalid` 在共享 deadline 未实际耗尽时按异常处理，不能作为普通时间中止或 Gap 达标。


### κ 参数闭包与约化成本浮点证书（第四轮终审修复）

正式 exact BPC 的一次求解固定一个不可变 `RiskPolicy`。所有单侧项使用 `policy.one_sided(ε)`，所有 geo2d 双侧项使用 `policy.two_sided(ε)`；因此有限模型可行域只由请求的合法 `kappa_mode` 决定，与进程残留 `RM.kappa` 无关。该性质由四合法模式×四残留全局模式的不变性测试覆盖。

对定价数值证明，把传入的 binary64 目标、行系数和对偶解释为精确实数。IEEE round-to-nearest 的乘积 `fl(ab)` 的精确乘积必位于其两个相邻 representable 数之间；区间加法同理通过向外 `nextafter` 包络。逐项传播得到 `[rc_L,rc_U]`，故 `rc_L <= rc_exact <= rc_U`。完整穷尽定价的遗漏列下界取所有 `rc_L` 的最小值，而不是固定减 `1e-9`。测试包含约 `3.27e12` 对偶尺度的灾难性相消反例：旧固定保护高于精确有理约化成本，新区间严格包络该值。Phase-I 和常规节点界只消费这个 lower endpoint 或同样向下舍入的 universal bound。


数值证书补充：RMP 拉格朗日下界与 `L_RMP + M·δ` / Phase-I 组合采用 binary64 向下有向舍入；任一非有限或溢出情形均 fail-closed。


### 13.1 Gap 停止与 exact optimality 分离

对第二阶段最小化问题，令资源可行 incumbent 的精确 binary64 实数和为 $E^{inc}$，对外报告使用向上包络 $E^{inc,U}$；开放节点严格全局下界为 $LB_E^{global}$. 用户给定的 `energy_gap_target_abs_Wh` 与 `energy_gap_target_rel` 只决定何时停止 anytime 搜索：

$$
E^{inc,U}-LB_E^{global}\le \epsilon_E^{abs}
$$

或相对 Gap 达标时，可以返回 `energy-gap-target-reached`，但只要严格界仍允许正改进，就不得设置 `energy_optimal=True`. Exact optimality 只能来自完整的安全 fathoming：树已穷尽，或所有开放节点的严格下界均不小于当前 incumbent 的精确目标。由于 binary64 上下包络可能有一个 ULP 的表示宽度，exact optimal 已证明时对外显示的安全 `energy_gap_abs_Wh` 仍可能出现极小正数；该数是数值包络宽度，不是未搜索的优化 Gap。

### 13.2 整数 RMP 不等于完整列空间整数最优

节点 RMP 的整数解只能作为 primal incumbent。若遗漏列约化成本严格下界为 $\delta<0$，则即便 RMP 解为整数，也不能直接 fathom。完整空间仍满足

$$
L_{full}\ge L_{RMP}+|I|\min(0,\delta),
$$

只有该严格节点界已不能改善 incumbent 时才可剪枝。若 HiGHS 在数值容差内返回一个整数 RMP 解，而严格对偶/定价界仍允许更优值，当前实现使用完备的路线变量兜底分支 $x_r=0/1$，其两个子节点互斥且并集等于父节点整数可行域；有限列空间下该分支保证不会仅因 LP 数值容差而错误签发最优证书。

### 13.3 路线身份与种子网格

正式路线签名和缓存指纹使用 binary64 精确指纹（`float.hex()`），包括参与路线身份的天气状态；不再对起飞/回收/天气浮点状态做小数位四舍五入。外部种子的 `tau/h` 必须与当前有限离散网格的 binary64 值精确一致；off-grid 值即使仅相差一个 ULP 也不得以 tolerance 接受。种子一旦命中当前网格，仍必须用当前实例重新计算完整物理合同。相同物理签名的列仅按 binary64 精确的 $(E^{plan},E^{soc})$ 偏序做支配替换，不以固定 $10^{-9}$ 容差吞掉真实的小能耗改善。

## 实体资源 oracle 与主问题事件行的一致性证明

对半开区间集合，若容量在某时刻被超过，则存在至少一个参与区间的开始时刻 `t` 也处于同一超容量交集中。因此只需在全部真实非空区间的开始事件处检查 `a_r <= t < b_r` 即可得到单区间资源的精确容量行；这里不得对 `t/a/b` round，也不得用正 tolerance 改写 membership。

最终实体资源 DFS 不依赖这些必要行来证明可行，而是重新执行严格分配。UAV 周转比较为 `start >= ready`；电池 SOC 需求把每个 binary64 能量 `e_j` 映射为 `Fraction.from_float(e_j)`，所以比较 `sum_j e_j <= B_use` 是在“输入 binary64 解释为精确实数”的语义下完成。因而不存在 `start >= ready-eps_t` 或 `used <= B_use+eps_E` 造成的整数可行域扩张。

对称剪枝状态也必须精确：时间用 binary64 精确指纹，电池累计直接使用 exact Fraction；只有真正等价的资源状态才能合并。这样 `INFEASIBLE_PROVEN` 才是严格资源模型上的证明，并可安全授权 exact-pattern cut。


模型身份证书同样使用精确 finite-data 语义。`parameter_contract_sha256` 绑定参数/离散化规则，`instance_contract_sha256` 绑定 turbine、launch、weather、Xi 与资源实例数据，浮点量按 binary64 精确表示规范化；`model_contract_sha256=H(parameter_contract_sha256,instance_contract_sha256)`。因此如果任一会改变路线空间、资源可行域或目标的数据发生变化，模型身份必须变化；续跑或外部结果若模型身份不一致，不得继承正证书。



### Weather provenance 边界
正式天气 DRCC 证书只覆盖 `weather_moments_caseB.csv` 所定义的 residual finite model。该文件由 train-only 的真实历史 no-leak residual 生成；Weather moments 的残差源从天气时间轴自身按 horizon 做全局 nonoverlap 抽样，避免不同船舶或 AIS 频率造成伪重复；validation/test weather errors 不进入 moments。预测器在 t0 不访问 t0 之后的 ERA5/CMEMS，因此不存在 future-weather leakage。此证书不等价于“ECMWF operational forecast archive 误差证书”；如果未来换成官方 forecast archive，必须使用新的 predictor/truth contract 和模型哈希。


## 14. 离散模型精确最优结论

在当前 finite binary64 model 中，若正式求解返回 `coverage_optimal=True`、`energy_optimal=True`、`lexicographic_optimal=True` 且 `global_certificate_available=True`，则证书覆盖当前离散起飞网格、当前 horizon 支持、有序 elementary 路线、全部允许资源模式及当前物理/DRCC 输入定义的完整隐式路线空间，因此该解是该有限离散模型的词典序全局最优解。若任一字段为假，尤其是时间上限、定价未知、开放节点或资源 `UNKNOWN_TIMEOUT` 尚存，则只能报告 incumbent 与严格 Gap，不得把近似解称为精确解。

该结论不扩展到未离散的连续起飞/回收时刻、`max_stops` 之外的路线、未选入实例的风机、未建模终端传感器误差或现实平台所有故障模式。

## 15. 外部审计的五项核心证明义务

本节把算法最容易出现“连接部位”错误的五个问题从检查清单提升为明确证明义务。任何一项失效，都必须令 `global_certificate_available=False`，而不能只降低性能评级。

### 15.1 证明义务 A：节点级 `L + M_n·min(0,δ)` 是否是完整列空间 lower bound

考虑任一最小化节点 LP。令当前已生成列为 \(G\)，遗漏但 node-compatible 的列为 \(U\)。代码并不把 HiGHS 返回的 binary64 行乘子直接当作“精确对偶可行解”。对 inequality marginal 先取

\[
y=\min(\widehat y,0)\le0,
\]

而 equality multiplier \(v\) 自由。对当前 restricted variables 的合法 box \(B_G\)，定义代码实际计算的 Lagrangian 下界

\[
L_G= b^Ty+d^Tv+\inf_{x_G\in B_G}
\sum_{r\in G}(c_r-y^TA_r-v^TH_r)x_r.
\]

这里不要求 \(y,v\) 满足精确 stationarity；只要 \(y\le0\)，对任意 primal-feasible 完整解都有弱拉格朗日不等式。HiGHS 的 KKT/stationarity/strong-duality 检查是独立的 sanity gate，而不是该 lower bound 成立所必需的数学前提。

对所有遗漏列，若有统一严格约化成本下界

\[
\bar c_r=c_r-y^TA_r-v^TH_r\ge\delta,\qquad r\in U,
\]

则对任意完整列空间可行解

\[
c^Tx\ge L_G+\sum_{r\in U}\bar c_r x_r
\ge L_G+\delta\sum_{r\in U}x_r.
\]

每条正式路线至少覆盖一台风机，同时 packing 满足

\[
\sum_r |S_r|x_r\le |I|,
\qquad |S_r|\ge1,
\]

对节点 forbidden-service 集合 `F_n`，所有 node-compatible 路线只覆盖 `I\setminus F_n`，故在 LP 松弛上有更紧的

\[
\sum_r x_r\le |I\setminus F_n|=:M_n.
\]

根节点 `F_n=\varnothing` 时退化为旧的 `M=|I|`。

当 \(\delta\ge0\) 时直接得到 \(c^Tx\ge L_G\)；当 \(\delta<0\) 时，由 \(\sum_{r\in U}x_r\le M_n\) 得

\[
\boxed{z^*_{full}\ge L_G+M_n\min(0,\delta).}
\]

代码在普通 RMP 的 Lagrangian infimum 中把每个 route variable 的证书 box 收紧为 \([0,1]\)，虽然 HiGHS 主问题本身使用 \([0,+\infty)\)。这是冗余但严格合法的：每条正式路线非空，任选 \(i\in S_r\)，对应 packing 行给出 \(x_r\le\sum_{q:i\in S_q}x_q\le1\)。因此该 cap 不删除任何 master-feasible 点，只让 Lagrangian infimum 更紧。

这正是当前 `_lagrangian_dual_lower_bound`、`_universal_pricing_lower_bound` 与 `_safe_node_bound_from_pricing` 共同实现的 `[THM-LRC]`。

**该数学证明依赖的不可破坏条件：**

- 每条 master route 必须至少服务一台风机；
- packing 行必须覆盖每条路线所服务的全部风机；
- Lagrangian inequality multiplier 必须满足 \(y\le0\)，restricted-variable box 必须包含所有当前 master-feasible restricted coordinates；
- `M_n` 必须是不小于当前节点完整 LP 中 `sum x_r` 的严格上界；当前实现取 `|I\setminus F_n|`；
- \(\delta\) 必须对**所有遗漏且当前分支允许的列**有效。

正式程序还额外要求 `_validate_linprog_result` 通过 primal feasibility、dual sign、stationarity、互补松弛与 strong-duality consistency；这是故障检测/solver sanity gate，失败时不允许使用该乘子做 pricing certificate，但它不是弱 Lagrangian lower bound 本身成立的逻辑前提。

如果未来允许“零风机路线”、遗漏某些 packing 系数，或 pricing lower bound 只覆盖部分候选空间，则该 correction 不能继续使用。

### 15.2 证明义务 B：Phase-I correction 是否足以证明完整空间 infeasibility

Elastic Phase-I 最小化人工变量总量 \(\Phi\)。若原节点在完整路线空间可行，则必存在人工变量全部为 0 的解，因此

\[
\Phi^*_{full}=0.
\]

令受限 Phase-I 的验证对偶下界为 \(L_{P1}\)，所有遗漏普通路线在该 Phase-I 对偶下的约化成本满足 \(\bar c_r\ge\delta\)。普通路线仍满足上一节的节点总质量界 \(\sum_r x_r\le M_n=|I\setminus F_n|\)，于是同样有

\[
\boxed{
\Phi^*_{full}
\ge
L_{P1}+M_n\min(0,\delta)
}.
\]

所以只有当

\[
L_{P1}+M_n\min(0,\delta)>\texttt{ART\_TOL}>0
\]

时，才能推出 \(\Phi^*_{full}>0\)，进而证明原节点在完整路线空间不可行。`ART_TOL` 在这里使证明更保守：下界虽为正但不超过 `ART_TOL` 时节点不会被剪掉；它不能把一个真实可行节点变成“已证明不可行”。

**可证伪条件：** 如果 Phase-I 人工变量的构造不再满足“原问题可行 iff 存在人工总量 0 的解”，或者遗漏列 lower bound 并未覆盖完整当前节点路线空间，则该 infeasibility certificate 失效。

### 15.3 证明义务 C：exact-pattern cut 对整数集、LP 松弛和未来列是否全局有效

设 \(S\) 是实体资源 DFS 已返回 `INFEASIBLE_PROVEN` 的**精确路线签名集合**。加入

\[
\boxed{
\sum_{r\in S}x_r-
\sum_{r\notin S}x_r
\le |S|-1.
}
\]

对任意二元向量，记

\[
a=\sum_{r\in S}x_r,
\qquad
b=\sum_{r\notin S}x_r.
\]

若 \(x\neq1_S\)：

- 若至少有一条 \(S\) 中路线未选，则 \(a\le|S|-1\)，从而 \(a-b\le|S|-1\)；
- 若 \(S\) 全部被选但还选择了其他路线，则 \(a=|S|, b\ge1\)，仍有 \(a-b\le|S|-1\)。

只有 \(a=|S|,b=0\)，即恰好模式 \(1_S\)，违反该式。由于该模式已被严格资源 Oracle 证明不属于真实整数可行集，所以该不等式对所有真实整数可行解有效；线性不等式对所有整数可行点有效即意味着它也对这些点的凸包有效，因此可以合法加入 LP RMP 并切除部分分数点。

当前 `_row_coefficient` 对 `resource_pattern` 使用：

- 签名属于 \(S\)：系数 `+1`；
- 其他任何当前或未来路线：系数 `-1`。

所以后续 Column Generation 新产生的列不会逃逸该 cut。与此同时，resource-pattern 行出现负列系数，约化成本通用下界不能再假设所有 inequality coefficients 非负；当前 `_universal_pricing_lower_bound` 对该行显式使用

\[
\min_{a\in\{-1,+1\}}(-u_ga)=u_g,
\qquad u_g\le0.
\]

**可证伪条件：** 若未来列的 pattern 系数没有由签名动态计算、资源 Oracle 的 `INFEASIBLE_PROVEN` 不是完整证明，或 route signature 不能唯一代表该 master 列的资源/主问题身份，则该 cut 不能授权全局证书。

### 15.4 证明义务 D：service / arc / route branching 是否与 pricing 完全兼容

对整数 master 解，风机服务量

\[
s_i=\sum_{r:i\in S_r}x_r
\]

由 packing 满足 \(s_i\in\{0,1\}\)。因此 `s_i=0` 与 `s_i=1` 构成互斥且完备的整数划分。

对有向弧 \((i,j)\)，所有包含该弧的路线都覆盖风机 \(i\)，故 packing 进一步给出

\[
0\le z_{ij}=\sum_{r:(i,j)\in r}x_r\le1,
\]

整数解中同样为二元，因此 `z_ij=0/1` 分支完备。具体路线变量 `x_r=0/1` 则提供最终的变量级完备兜底。

实现必须同时满足两种传播：

1. `forbidden_turbines / forbidden_arcs / forbidden_routes` 直接过滤 pricing 的允许路线空间；
2. `required_turbines / required_arcs / required_routes` 作为 RMP equality rows，并把对应 equality dual 完整进入遗漏列 reduced cost。

这样每个子节点的 pricing 实际求解的是该子节点自己的完整隐式列空间，而不是父节点空间。

**可证伪条件：** 任何 required/forbidden 条件若只进入 RMP 而未进入 pricing 的过滤/约化成本，或只进入 pricing 而未进入 master 约束，都可能使节点 LP bound 与节点整数可行域不一致，必须禁止签发全局证书。

### 15.5 证明义务 E：binary64 interval certificate 是否足以支撑 `global_certificate_available=True`

正式数值语义把每个输入 binary64 payload 解释为对应的精确有限实数。对有限的 binary64 \(a,b\)，IEEE round-to-nearest 计算 \(p=fl(ab)\) 后，精确实数乘积 \(ab\) 必位于 `nextafter(p,-∞)` 与 `nextafter(p,+∞)` 所围成的区间内；加法同理。当前 reduced-cost 线性表达式逐乘积、逐加法向外传播，因此得到

\[
\boxed{
\bar c_L\le \bar c_{exact}\le \bar c_U.
}
\]

于是：

- `rc_U < 0` 才能证明严格改善；
- `rc_L >= 0` 才能证明严格非改善；
- `rc_L < 0 <= rc_U` 只能判为数值符号未知，不能关闭 pricing。

对没有完成全扫描的情形，`_universal_pricing_lower_bound` 还必须是对任意可能遗漏列都成立的代数下界。当前结构中：

- coverage 基础成本 \(-|S_r|\ge-\texttt{max\_stops}\)；
- energy 与 Farkas 普通路线基础成本均不小于 0；
- packing/deck/active/pooled-energy 的列系数非负，而 inequality dual \(u\le0\)，所以其约化成本贡献 \(-ua\ge0\)，忽略它们仍是安全 lower bound；
- exact-pattern 行单独按 \(a\in\{-1,+1\}\) 取最坏贡献；
- coverage equality 的系数位于 \([1,\texttt{max\_stops}]\)，按两个端点取最小值；
- required service/arc/route equality 系数位于 \(\{0,1\}\)，取 `min(0,-dual)`。

节点 lower bound、Phase-I correction 与最终能量 incumbent 还分别使用向下有向舍入和 `Fraction.from_float` 精确有限数累加。任何非有限输入、溢出、无法建立 outward interval 或无法验证 RMP dual 的情况都必须 fail-closed。

**可证伪条件：** 如果未来新增一种 master row 而没有在显式 future-column coefficient-range registry 中注册并证明其取值范围；未知 row 必须直接使 universal bound unavailable/fail-closed；或者某个证书路径重新使用固定 `1e-9` 代替 outward bound；或者 `global_certificate_available=True` 可以在 `coverage_optimal/energy_optimal/lexicographic_optimal` 未同时成立时出现，则 binary64 精确性链被破坏。

### 15.6 五项证明义务与最终证书的逻辑关系

因此当前全局证书不是由单个 solver 状态产生，而是以下合取：

\[
\boxed{
\begin{aligned}
&\text{完整节点路线空间定价/安全遗漏列界}\\
\land{}&\text{完整 Phase-I/Farkas 逻辑}\\
\land{}&\text{完备且与 pricing 一致的 branching}\\
\land{}&\text{只由 INFEASIBLE\_PROVEN 授权的有效资源 cuts}\\
\land{}&\text{binary64 向外数值证书与严格资源审计}\\
\land{}&\text{覆盖阶段全局闭合}\\
\land{}&\text{固定 }C^*\text{ 后能耗阶段全局闭合}
\end{aligned}
}
\]

只有上述链条同时成立，才允许：

```text
coverage_optimal=True
energy_optimal=True
lexicographic_optimal=True
global_certificate_available=True
```

这也是外部 AI/审稿人最推荐的攻击顺序：优先寻找某个“完整空间”量实际上只覆盖 restricted pool、某个未来列没有继承既有 cut/branch row、某个 UNKNOWN 被误转为 infeasible，或某个浮点点估计绕过 outward certificate。任何一个具体反例都比对“Branch-Price-and-Cut”名称本身的争论更有价值。

## 2026-08 补充：近整数主问题与 pricing 中断的证明责任

LP solver 的可行性 tolerance 不能把 `rint(x)` 变成整数可行性证书。设 `y=rint(x) in {0,1}^R'`；正式路径把每个已编码的 `A_ij,b_i` binary64 值映射为 `Fraction.from_float`，精确检查 `A^<= y<=b` 与 `A^=y=d`。只有该检查通过，`y` 才可交给实体资源 oracle 并成为 incumbent 候选。检查失败不会剪枝，只触发完备 `x_r=0/1` 分支或 fail-closed，因此只能减少 false-feasible/false-certificate 风险，不会删除任何真实整数解。

Exact pricing 中断也不等价于闭合。若当前 validated RMP dual lower bound 为 `L`，而中断 oracle 仍给出对所有未完成列成立的统一下界 `delta`，由非空路线、forbidden-service route filtering 与 turbine packing 得到 `sum_r x_r<=M_n=|I\F_n|`，故完整节点仍有 `L+M_n min(0,delta)` 下界。若该界已被 incumbent 支配，fathom 仍是严格安全的；否则节点必须保持 open。该结论同样要求 universal `delta` 已包含 exact-pattern 行的 `+1/-1`、required equality 与所有当前 master row 的系数范围。


## 16. 证书作用域补强：Route Universe 与 Semantic Invariance

### 16.1 `[THM-RU]` Route-Universe Provenance Theorem

令 `R_Phi` 为完整正式物理/DRCC Oracle 在有限离散 launch、elementary sequence 与 recovery horizon 上定义的路线集合，`R_used` 为实际 pricing 搜索集合。BPC 树安全闭合只能推出 `LexOpt(R_used)`。要声明当前 physical model 的全局最优，必须另外证明 `R_used=R_Phi`。因此 formal public path 禁止 `implicit_test_columns`；synthetic fixture 即使 `algorithmic_global_certificate=True`，也必须满足 `route_universe_provenance_certified=False` 与 `physical_model_global_certificate=False`。

### 16.2 `[LEM-CS]` Canonical Route Semantic Invariance Lemma

设 `sigma(r)` 为 exact route signature，`theta(r)` 收集该 master variable 的 objective、master coefficients 与 resource-audit 相关属性。正式求解要求

\[
\sigma(r)=\sigma(r')\Rightarrow\theta(r)=\theta(r')
\]

按 binary64-exact identity 成立。否则一个既有 exact-pattern cut 可能是对旧 `E_soc` 证明不可行，却继续切掉同 signature、较低 `E_soc` 的新 representation；同理 Stage-2 objective 或 pooled-energy coefficient 也会被悄然改写。实现因此只允许 exact-semantic duplicate 去重，任何同 signature 异语义列 fail-closed。

### 16.3 `[THM-LRC]` Full-Space Lagrangian Reduced-Cost Correction Theorem

对节点 `n`，代码把 HiGHS inequality marginal 投影为 `y=min(y_hat,0)<=0`，equality multiplier `v` 保持自由。令当前列集合为 `G_n`，其证书 box 为 `B_n`；普通 route variable 的 `[0,1]` cap 由非空路线 + packing 冗余推出。定义

\[
L_n=b^Ty+d^Tv+\inf_{x_G\in B_n}\sum_{r\in G_n}
(c_r-y^TA_r-v^TH_r)x_r.
\]

这个式子对任意满足符号条件的 \(y,v\) 都是 lower bound，不要求 binary64 multiplier 在数学实数意义下精确 stationarity。`_validate_linprog_result` 仍额外检查 primal feasibility、dual sign、stationarity、complementary slackness 与 strong-duality consistency，用来拒绝伪造/严重错误 solver 输出；但 `[THM-LRC]` 的核心证书责任由完整 box infimum + outward rounding 承担。

定义遗漏列 `rc_r=c_r-y^T A_r-v^T H_r`。若所有遗漏 node-compatible 列有 `rc_r>=delta_n`，且由非空路线与 packing 得到 `sum_r x_r<=M_n=|I\setminus F_n|`，则

\[
\boxed{z^*_{full,n}\ge L_n+M_n\min(0,\delta_n)}.
\]

Phase-I 仅把普通路线目标改为 0 并加入 equality artificials，所以完全同样得到

\[
\boxed{\Phi^*_{full,n}\ge L_{P1,n}+M_n\min(0,\delta_{P1,n})}.
\]

只有后一式严格大于 `ART_TOL` 才可证明完整节点 infeasible。`ART_TOL>0` 只使剪枝更保守。

### 16.4 `[THM-LRC]` Future-row Range Registry

对 future column 的 universal reduced-cost lower bound，所有 master row 必须给出已证明的 coefficient range。当前 inequality rows：packing/deck/active 为 `[0,1]`，pooled-energy 为 `[0,+inf)`，exact-pattern 为 `[-1,1]`；equality rows：coverage 为 `[1,max_stops]`，required service/arc/route 为 `[0,1]`。实现对未知 row 直接 fail-closed，禁止把未来 signed cut 静默当作非负系数行。

### 16.5 `[THM-LEX]` 最终物理证书合取

最终正式物理证书不是 `solver optimal` 的别名。代码中的 `_physical_certificate_guard` 与本文公式逐项同构：

\[
\boxed{
G_{physical}=
G_{algorithmic}\land
P_{route\ universe}\land
P_{semantic\ invariance}\land
P_{row\ ranges}\land
P_{binary64/model}\land
P_{proof\ contract}\land
[mode=exact\text{-}branch\text{-}price\text{-}cut]
}
\]

其中 `P_proof contract` 要求运行结果携带的 `FORMAL_PROOF_CONTRACT`、obligation ID 集合和 proof-contract SHA 与当前代码一致；这不是新的数学约束，而是防止“文档定理已经变化、旧算法哈希仍被当作同一证书语义”的 provenance 门。两阶段 branch tree、Phase-I、pricing、三状态资源审计、exact-pattern cuts 与 exact integer master gate 均按本文件前述条件安全闭合后，`G_algorithmic` 才可能为真。

## 17. Proof-to-Code Concordance（正式定理—代码锚点）

当前证书语义合同为 `exact-bpc-proof-code-concordance-v9-source-bound-global-battery-relaxation`。`step12_branch_price.py` 的 `FORMAL_PROOF_OBLIGATIONS` 与 `FORMAL_PROOF_CODE_ANCHORS` 是机器可读的对应表，并进入 `algorithm_contract_sha256` / `proof_contract_sha256`。模型哈希没有因此改变：本节只绑定“为什么可以发证书”的算法语义。

| ID | 数学命题 | 主要代码锚点 | fail-closed 条件 / 最终责任 |
|---|---|---|---|
| `THM-RU` | 实际 formal pricing universe 等于 \(R_\Phi\) | `solve_fleet_anytime`, `_solve_fleet_anytime_impl`, `_exact_pricing_search` | synthetic public injection 被拒；`route_universe_provenance_certified` |
| `LEM-CS` | canonical signature 映射到不可变 formal semantics | `_column_semantics_fp`, `_add_columns`, `_route_archive_semantics_invariant` | 同 signature 异语义 `RuntimeError`；最终 archive 再审计 |
| `THM-LRC` | restricted box Lagrangian LB + omitted-column correction | `_lagrangian_dual_lower_bound`, `_future_column_coefficient_range`, `_universal_pricing_lower_bound`, `_safe_node_bound_from_pricing` | 非有限/溢出/未知 row → bound unavailable；否则 \(L_n+M_n\min(0,\delta_n)\) |
| `COR-P1` | Elastic Phase-I 完整空间 infeasibility | `_solve_elastic_phase_one`, `_phase_one_full_space_lower_bound`, `_phase_one_infeasibility_proven` | 仅 full-space LB `>ART_TOL` 可剪枝 |
| `LEM-PAT` | exact-pattern cut 是 Hamming-distance no-good | `_row_coefficient`, `_audit_integer_selection` | 仅 `INFEASIBLE_PROVEN` 产 cut；future column 动态 `+1/-1` |
| `THM-BR` | service/arc/route 二分完备且 pricing-compatible | `_column_allowed_at_node`, `_master_rows`, `_branch_on_fractional_solution`, `_branch_on_integral_numeric_ambiguity` | forbidden 过滤；required 作为 equality；无完备分支则节点保持 unresolved |
| `THM-NUM` | fixed-binary64 outward/exact-rational 数值链 | `_validate_linprog_result`, `_column_reduced_cost_interval`, `_exact_binary_master_feasible`, `_energy_of_selection_exact` | straddle-zero/验证失败/非有限值均不升级证书 |
| `THM-LEX` | coverage 先全局闭合，再 fixed-coverage energy 闭合 | `_solve_branch_price_stage`, `_physical_certificate_guard`, `_solve_fleet_anytime_impl` | 任一合取为假则 `physical_model_global_certificate=False` |
| `THM-TGT` | fixed-target YES/NO 不可把 UNKNOWN 升级 | `_target_infeasibility_algorithmic_proven`, `_solve_fleet_anytime_impl` | open node / incomplete pricing-resource-branching 任一存在均不得签 NO |
| `THM-CU` | formal route universe 完整且 provenance/hash 绑定 | `build_certified_route_universe`, `_validate_certified_route_universe`, `_solve_branch_price_stage` | 枚举 timeout/error 或 context/column SHA 不一致则 `complete=False` |
| `THM-FCT` | `T=|I|` 推出 exact partition；resource-infeasible full pattern cut 有效 | `_solve_complete_universe_fullcover_target`, `_exact_fullcover_master_feasibility`, `_fullcover_target_master_rows` | floating MILP infeasible 不能单独闭点；UNKNOWN 不切不闭 |
| `THM-GBR` | 删除时序/UAV/binding/station 后的 exact battery-energy relaxation 是必要松弛 | `_exact_global_fullcover_battery_relaxation`, `_solve_complete_universe_fullcover_target` | 仅 `B_relax*>B` / relaxation 无 exact cover 可签 NO；`FEASIBLE_RELAXATION` 不可签 YES |

### 17.1 `[COR-P1]` 与 `[LEM-PAT]` 的独立说明

`COR-P1` 是 `THM-LRC` 在 Elastic Phase-I 上的直接推论，但单独列 ID，因为它承担 **false infeasible** 的剪枝责任。`LEM-PAT` 的线性式可重写为

\[
\sum_{r\in S}(1-x_r)+\sum_{r\notin S}x_r\ge1,
\]

即 binary route vector 与已证明不可行 pattern 的 Hamming distance 至少为 1。`_row_coefficient` 是当前列和 future columns 共用的唯一系数规则，因此列生成不会改变该 cut 的数学语义。

### 17.2 `[THM-BR]` 的 required/forbidden 非对称性

required service/arc/route **不是** route-domain filter，而是 master aggregate equality；对应 dual 进入 reduced cost。forbidden service/arc/route 才通过 `_column_allowed_at_node` 删除不兼容路线。特别地，required arc \((A,B)\) 下，一条独立路线 \(C\) 仍可被选择，只要另一条选中路线满足该 required-arc equality。

### 17.3 `[THM-NUM]` 的边界

该定理是 fixed finite binary64 program-defined model 的证书：linear reduced-cost/bound 表达式使用 outward enclosure，整数 master/SOC/energy 的关键有限和使用 exact rational binary64 identity。它不声称所有 nonlinear physical/libm 中间量已经被任意精度 real-arithmetic interval formalization。



## 18. E1 资源前沿的严格单调 Sandwich 定理

### 18.1 资源嵌入引理

固定 UAV profile、物理/DRCC route universe、任务窗和其它模型参数。令 `X(K,B)` 为拥有 K 个实体 UAV、B 个实体电池时的真实整数可行 route-selection/resource-assignment 集。若 `K1<=K2` 且 `B1<=B2`，任意 `x in X(K1,B1)` 的原 UAV/battery identity、SOC history、turnaround/inspection/swap/deck schedule 可在大资源实例中原样保留，新增实体保持未使用，因此

\[
\boxed{X(K_1,B_1)\subseteq X(K_2,B_2)}.
\]

Stage-1 目标只最大化覆盖，故

\[
\boxed{C^*(K_1,B_1)\le C^*(K_2,B_2)}.
\]

该命题不依赖 solver、tolerance 或 validation。它依赖“额外资源可闲置”和“route physics 不随 K/B 改写”两个正式合同。

### 18.2 Anytime coverage interval

Coverage-only exact BPC 的 incumbent 来自完整物理/实体资源可行整数解，因此

\[L_{K,B}:=C^{inc}_{K,B}\le C^*(K,B).\]

其 `coverage_upper_bound` 来自每个开放节点的严格 full-space upper bound 聚合；即使 deadline 先到，也有

\[C^*(K,B)\le U_{K,B}:=UB^C_{K,B}.\]

因此每个网格点都提供安全区间 `[L,U]`；`coverage_optimal=True` 等价于该区间严格闭合，而 sandwich 推理本身不要求参与的每个中间格都单独闭树。


### 18.2a 二维资源单调 envelope

由资源嵌入定理，
\[
L^\uparrow(K,B)=\max_{K'\le K,B'\le B}L(K',B'),\qquad
U^\downarrow(K,B)=\min_{K'\ge K,B'\ge B}U(K',B')
\]
满足
\[
\boxed{L^\uparrow(K,B)\le C^*(K,B)\le U^\downarrow(K,B)}.
\]
因此它是严格安全的 bound strengthening，不改变可行域，也不把未闭合格变成 exact。

### 18.2b 无信息长认证停止是安全的

若同一 cell 长认证前后严格区间均为 `[L,U]`，停止再次运行不会改变任何可行解、bound 或 certificate；控制器只保留 `unresolved`。故 no-gain blacklist 是纯调度剪枝，而不是 BPC fathoming 规则。

### 18.3 平台 Sandwich 定理

固定 `Kmax`，取 `B0<B1`。若

\[
L_{K_{max},B_0}=U_{K_{max},B_1}=P,
\]

则任意 `B0<=B<=B1` 有

\[
P=L_{B_0}\le C^*(B_0)\le C^*(B)\le C^*(B_1)\le U_{B_1}=P,
\]

故

\[\boxed{C^*(K_{max},B)=P\quad\forall B\in[B_0,B_1]}.\]

这严格证明了该尾段的零边际平台。另一充分条件是尾点 `L` 已达到由 turbine 可覆盖数给出的硬上限。`safe_served` 是 validation 后统计量，可能随优化选择变化而非单调，不能替代本定理。

### 18.4 BK/KB knee 最小性

设已证明平台值 P，阈值 `T=ceil(frac*P)`。对 BK 顺序，若在 Kmax 上最小候选 B* 满足 `L(Kmax,B*)>=T`，且其直接前驱（若存在）满足 `U(Kmax,B^-)<T`，则由 K 单调性可知所有 `B<B*`、任意扫描 K 都不能达到 T。固定 B* 后，若最小候选 K* 满足 `L(K*,B*)>=T` 且前驱 `U(K^-,B*)<T`，则 `(K*,B*)` 是扫描网格内 BK 字典序最小的真实阈值可达资源点。KB 同理。

注意这是**覆盖/资源 knee 的证明**，不是能耗最优证明。代码随后必须在 `(K*,B*)` 上恢复 `solve_scope="lexicographic"`，完整证明 fixed-coverage Stage-2 能耗，再对该 exact chosen plan 做 validation。只有这一步的 `global_certificate_available=True` 且 validation gate 通过，formal E1 才能把该 knee 标记为 `eligible`。


## 19. E2 最严天气分位冻结命题

E2 的最终候选集合按实验协议预先定义为 `q=q_max` 且通过 validation gate 的 criterion。令 `G_m` 表示方法 m 在 q_max 上的完整 physical lexicographic certificate。正式冻结集合为

\[
\mathcal C_{freeze}=\{m:\;q=q_{max},\;G_m=1,\;H_m=1,\;n_{missing,m}=0,\;C_m>0\},
\]

其中 `H_m` 是 validation 风险门。最终的 safe coverage / coverage / energy 排序只在该集合内进行。于是任意 `q<q_max` 行的 timeout、Gap 或未闭合能耗都不能改变 `C_freeze`，也不能进入 final-test 选择。缩短低 q 的 wall-clock 仅降低诊断曲线的证据等级；不影响 q_max 已完整闭合方案的优化证书。

反之，若某 q_max 方法在长时限后仍 `global_certificate_available=False`，它不得以 `run_status=ok` 进入完成矩阵；若因此没有正式候选，formal E2 必须停止而不能消费 final test。这条门禁专门防止“time-limit incumbent + validation pass”被误解释为精确方法最优方案。

## 20. Auxiliary proof obligations

### Single-vessel provenance

Let `m*` be the unique MMSI selected by the formal track. Formal Xi moments and
all empirical replay samples are functions only of rows with MMSI `m*`.
Consequently no certificate can inherit a statistic from a different vessel.
If a required `(h,c)` cell for `m*` is absent, the formal data oracle is
undefined/fail-closed rather than replaced by a pooled distribution.

### Safe prefix-service pruning

For any elementary prefix pi and any extension r of pi, the formal route time
and energy decompose into the inspection/climb service component plus
nonnegative flight/return/dock components. Hence
`T(r) >= Tsvc(pi)` and `E(r) >= Esvc(pi)` in the declared binary64 program.
If `Tsvc(pi)>60 max(H)` or `Esvc(pi)>B_use`, no extension can be feasible.
Skipping horizon h when `Tsvc(pi)>60h` is likewise exact. The pruning therefore
does not remove a feasible route from the implicit universe and is compatible
with THM-RU/THM-LRC.

### Battery structural cap

Every formal route services at least one turbine, and turbine packing gives
`sum_r |S_r| x_r <= |I|`. Since `|S_r|>=1`, `sum_r x_r<=|I|`. With `B=|I|`
each selected route can be assigned its own battery, so increasing B beyond
`|I|` cannot enlarge the physical feasible set. This is a finite-model
redundancy proof, not an observed-saturation heuristic.

### One-time final-test protocol

No statement derived from the independent final test may influence E1/E2
selection. Formal E1 therefore has no final-test branch. The test is evaluated
only after the E2/A candidate is frozen. A legacy E1 final-test artifact is a
protocol conflict and must be rejected before a formal final run.

## THM-TGT — fixed-coverage target decision and knee closure

Let \(T\in\mathbb Z_{>0}\). Define
\[
  \mathcal F_T(K,B)=
  \left\{x\in\mathcal F(K,B):
    \sum_r |S_r|x_r=T\right\}.
\]

**YES direction.** If the exact solver materializes an integer
\(x\in\mathcal F_T\), rechecks all master rows and obtains a strict
`ResourceAuditStatus.FEASIBLE` assignment, then
\(\mathcal F_T\ne\varnothing\). No route-universe exhaustion is required for
this existential statement, but the witness still carries the formal
physical/provenance/numeric contracts.

**NO direction.** If no witness exists in the incumbent archive, that fact
alone has no proof value. `TARGET_INFEASIBLE` is allowed only when the target
BPC stage has zero open nodes, complete integer branching, complete strict
resource audits, certified future-column pricing bounds, and complete
Phase-I/Farkas pricing at every infeasible restricted master. The usual branch
partition, future-column coefficient ranges, outward binary64 reduced-cost
intervals and Phase-I lower-bound theorem then imply
\(\mathcal F_T=\varnothing\).

Because coverage is integer,
\[
  \mathcal F_T=\varnothing \quad\Longrightarrow\quad
  C^*(K,B)\le T-1
\]
whenever the current rigorous incumbent is already below \(T\). Combining this
with resource monotonicity proves immediate BK/KB predecessor failure and hence
resource-minimality of the first target-feasible grid point.

`THM-TGT` is currently bound in
`exact-bpc-proof-code-concordance-v9-source-bound-global-battery-relaxation`. Target decision
certificates are deliberately separate from `THM-LEX`: they may tighten the
coverage interval, but never claim Stage-2 energy optimality or
`global_certificate_available`.

## 21. R-BPC strengthening / acceleration 的证书边界

### 21.1 Battery half-cap validity

定义

\[
\mathcal R_H=\{r:2E_r^{\mathrm{soc}}>B_{\mathrm{use}}\}.
\]

若同一实体电池执行任意 \(r_1,r_2\in\mathcal R_H\)，则

\[
E_{r_1}^{\mathrm{soc}}+E_{r_2}^{\mathrm{soc}}>B_{\mathrm{use}},
\]

与单块电池 usable SOC 容量矛盾。因此每块电池至多承载一条 \(\mathcal R_H\) route，\(B\) 块实体电池推出

\[
\sum_{r\in\mathcal R_H}x_r\le B.
\]

该 row 是原 exact battery/SOC 可行域的有效必要不等式，可以进入正式 RMP、dual 与 reduced-cost certificate；exact resource audit 仍负责完整电池 identity、SOC、turnaround 可行性。

### 21.2 Resource-aware timing enrichment safety

每个 enrichment route 都由与正式 pricing 相同的 whole-route physical/DRCC evaluator 接受，因此

\[
\mathcal R_{\mathrm{enrich}}\subseteq\mathcal R.
\]

故已找到的 enrichment route 是合法 column，可进入 RMP 并参与寻找 primal feasible solution。但 enrichment 的搜索是有界且触发式的，未找到新列不能推出 \(\mathcal R\setminus\mathcal R_G=\varnothing\)。因此 enrichment 的 negative result 不得用于 pricing closure、UB、pruning、infeasibility 或 optimality certificate。

### 21.3 Exact generated-column primal recovery safety

设当前 generated archive 为 \(\mathcal R_G\subseteq\mathcal R\)。restricted exact solve 返回 witness \(x_G\)，并由 unchanged exact entity-resource audit 重新确认

\[
x_G\in\mathcal F(\mathcal R_G).
\]

由于列与约束语义均来自原 full model，

\[
\mathcal F(\mathcal R_G)\subseteq\mathcal F(\mathcal R),
\]

因此其目标值是 full problem 的合法 primal lower bound：

\[
C^*\ge C(x_G).
\]

所以 recovery 可以安全更新 incumbent/LB。另一方面，restricted optimum \(C_G^*\) 只说明当前 archive 内没有更好组合，不能推出 \(C^*\le C_G^*\)。代码因此禁止把 restricted archive 的 UB、cuts、infeasibility 或 optimality status 导入 full-space proof state。


---

## R-BPC certified prefix-pruning acceleration (exactness-preserving)

This implementation may prune a pricing prefix only in **certification** search, and only when the existing outward-rounded prefix reduced-cost lower bound is mathematically nonnegative. Discovery/timing-enrichment searches do not inherit closure semantics from this mechanism.

For a prefix `p`, let `Ext(p)` be the finite set of legal route columns that complete that prefix under the unchanged R-BPC route space and branch state. The pruning oracle constructs `LB_rc(p)` from the objective lower bound and certified future-row coefficient intervals. Every floating-point interval operation is outward protected by the same binary64 helpers used by the formal pricing proof. The required invariant is

`LB_rc(p) <= min_{r in Ext(p)} rc(r)`.

Therefore `LB_rc(p) >= 0` implies every completion has mathematical reduced cost at least zero. Since a formal improving column still requires `rc_UB < 0`, no improving completion can exist and the subtree can be discarded without changing pricing closure, node bounds, or the global certificate.

### Energy-stage service/cardinality joint bound

For Stage 2 (`min E` at fixed optimal coverage), the implementation strengthens only the objective/coverage part of the prefix bound. For each already visited turbine it reuses the exact inspection-plus-climb energy expression appearing as a nonnegative summand in the unchanged route-energy model. All omitted route-energy terms (travel, takeoff/return, docking and other nonnegative physical contributions) are ignored, so the resulting service-energy quantity is a lower bound on every completion's route energy.

For each unused, branch-allowed turbine the same single-turbine mandatory service-energy floor is computed. If `m` additional stops are selected, the sum of the `m` smallest such floors is a relaxation of the minimum mandatory service energy of any legal `m`-stop extension. The implementation enumerates every feasible future cardinality `m = 0,...,max_stops-|p|` and combines this lower bound with the exact fixed-coverage equality contribution `-lambda_C (|p|+m)`, using outward binary64 interval arithmetic. Taking the minimum over all `m` yields a valid lower bound on `E_route - lambda_C |S_route|` for every completion. Branch and physical restrictions deliberately omitted from this calculation only enlarge the relaxed completion set and therefore cannot invalidate the lower bound.

This joint treatment removes the unsafe/overly-loose conceptual combination “maximum future coverage reward with zero route energy” while preserving the same physical model, DRCC model, SOC audit, route space, improving-column rule (`rc_UB < 0`), and certificate semantics.

### Firewall and regression contract

* Certified prefix pruning is forcibly disabled for discovery searches.
* Bound-construction exceptions fail open: the subtree is enumerated rather than pruned.
* A pruned prefix contributes its certified lower bound to the aggregate omitted-column bound before the subtree is discarded.
* No heuristic score, timing-enrichment failure, restricted-archive optimum, or approximate resource test can trigger this prune.
* `selftest.py --suite pricing_shadow` includes an exhaustive small-prefix regression that compares the complete set of strict-negative route signatures with and without certified prefix pruning across multiple fixed-coverage dual values.

## R-BPC 实现层加速定理（v3.1 addendum）

以下两定理与既有证书链条的关系：二者均为**实现等价性**陈述，不新增任何
剪枝权限、不改变任何数学问题的定义。完整证明见本文件 THM-006/THM-007 的证明节。

**THM-006（exact cached-feature pricing）** 设生产定价调用在某超集分支节点
（forbidden_turbines 与 forbidden_arcs 均为空）上自然完成（complete=True，
无 deadline、无 discovery/neutral 早退）。则该次遍历已覆盖任意后代节点可达
的全部 (launch, ordered elementary prefix, horizon) 键（引理 6.1 键空间
单调；引理 6.2 缓存覆盖）。在此完备性标志置位后，"遍历缓存非 None 列快照
并逐列经未改动的 consider() 定价"与完整 DFS 的 considered 列集合相等
（定理 6），故 best_rc / reduced_value_bound / 改进列集合 / 闭合判定逐调用
一致，证书决策不变（推论 6.1）。实现条件（fail-safe 全清单）：快照过滤逐条
复制 DFS 键过滤；缓存增长或域指纹变化时重建；任何异常键/列跳过；普通 dict
（无状态槽）自动禁用该路径。

**THM-007（RMP 矩阵构建不变量提升）** `_build_restricted_master` 将每列的
路由签名 / tid 集合 / 弧集在行循环外计算一次，行内系数表达式为
`_row_coefficient` 对应分支的逐字复制，未覆盖行类别回退原函数。由于
`_row_coefficient` 的每个分支是（列内容, 行描述符）的确定性纯函数，提升后的
`A_ub / A_eq / objective` 与原实现**逐位相同**（定理 7）。

回归义务：`selftest --suite pricing_shadow` 与 `--suite exact_bpc` 在
v3.1 字节上全 PASS（55 项）；主实例 C\*/E\* 与加速前归档逐位一致；
对抗 oracle（Fraction 精确算术）3,690 对零反例。
