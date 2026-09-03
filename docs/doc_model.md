# 正式数学模型

本文件定义当前有限离散数学模型本身。算法调度、证明加速器和版本迁移史不属于模型定义；THM-CU/FCT/GBR 只改变求解/证明方式，不改变本文件可行域。

当前模型合同：

```text
finite-route-model-strict-physical-v8-discrete-recovery-target-xi-only-coherent-weather
```

模型作用域：

- concrete single-vessel AIS track；
- fixed retrospective mission window；
- discrete launch times；
- discrete recovery horizons；
- elementary ordered turbine routes；
- Xi position uncertainty；
- coherent historical weather residual uncertainty；
- exact UAV/battery/SOC/deck/inspection/swap resource semantics；
- Stage-1 coverage → Stage-2 fixed-coverage energy lexicographic objective。

当前结果明确：

```text
operational_nonanticipativity_certified = False
```

因此不能把 static launch-asof finite master 宣称为严格 mission-start multistage nonanticipative policy。


## 1. 规划范围与基本集合

规划窗口长度为

$$
T = 360\ \mathrm{min}.
$$

定义：

- $I$：风机集合，索引为 $i$；
- $K$：实体 UAV 集合，索引为 $k$；
- $B$：实体飞行电池组集合，索引为 $b$；
- $\mathcal T$：确定性离舰时刻网格；
- $\mathcal H$：允许进入优化的回收时长集合；
- $R$：通过全部物理、统计和风险门禁的候选任务列集合。

正式统计锚点为

$$
\mathcal H^{\mathrm{stat}}
=
\{5,10,15,\ldots,60\}\ \mathrm{min}.
$$

决策时长 $h$ 只能位于统计支持闭区间内。命中锚点时必须存在对应状态统计格；区间内插值时，上下两个锚点必须具有同一分类状态的统计格。禁止区间外外推、最近时长吸附和跨状态借格。

## 2. 候选任务列

一条候选任务列写为

$$
r=(\tau_r,\pi_r,h_r),
$$

其中：

- $\tau_r\in\mathcal T$：离舰时刻，单位为分钟；
- $\pi_r=(i_1,i_2,\ldots,i_{m_r})$：有序风机访问序列；
- $1\le m_r\le s_{\max}$，其中 $s_{\max}=\texttt{max\_stops}$；
- $h_r\in\mathcal H$：从离舰到接地的任务时长，单位为分钟；
- $I_r\subseteq I$：该列完成巡检的不同风机集合。

覆盖参数定义为

$$
a_{ir}=
\begin{cases}
1, & i\in I_r,\\
0, & i\notin I_r.
\end{cases}
$$

每台风机只需由一架 UAV 完成一次固定时长拍摄：

$$
t^{\mathrm{insp}}=300\ \mathrm{s}.
$$

列必须满足六小时接地边界：

$$
\tau_r+h_r\le T.
$$

接地后的着陆清场、快速检查或换电可以延伸到 $T$ 之后，但其完成时间不能支持 $T$ 之后的新离舰任务。

## 3. 船舶预测与回收状态

对离舰时刻 $\tau_r$ 和候选任务时长 $h_r$，预测回收位置为

$$
\widehat{\boldsymbol p}^{\mathrm{rec}}_r
=
\widehat{\boldsymbol p}^{\mathrm{ship}}(\tau_r+h_r\mid\mathcal F_{\tau_r}),
$$

其中 $\mathcal F_{\tau_r}$ 表示离舰时刻之前可用的信息。

实现回收位置写为

$$
\boldsymbol p^{\mathrm{rec}}_r
=
\widehat{\boldsymbol p}^{\mathrm{rec}}_r
+
\boldsymbol\xi_{h_r,c_r},
$$

其中：

- $c_r=c(\tau_r)$ 是外层船位误差统计所使用的离舰状态分类；
- $\boldsymbol\xi_{h_r,c_r}$ 只描述外层船位预测误差；
- 回收阶段的分类状态 $\widehat c^{\mathrm{rec}}_{r,h_r}$ 必须由显式、无未来泄漏的逐时长预测器提供。

正式模式禁止：

$$
\text{起飞状态隐式代理},\quad
\text{最近状态吸附},\quad
\text{分类状态插值},\quad
\text{真实未来状态回填}.
$$

若缺少 $\widehat c^{\mathrm{rec}}_{r,h_r}$，该列直接拒绝。若

$$
\widehat c^{\mathrm{rec}}_{r,h_r}\in\mathcal C^{\mathrm{forbid}},
\qquad
\mathcal C^{\mathrm{forbid}}=\{\text{转弯}\},
$$

则禁止进入船尾伴飞和回收阶段。

## 4. 飞行、伴飞与着舰能耗

### 4.1 完整计划能耗

每条可行列的第二层成本为

$$
E_r^{\mathrm{plan}}
=
E_r^{\mathrm{flight}}
+
E_r^{\mathrm{escort}}
+
E_r^{\mathrm{dock}}.
$$

其中：

- $E_r^{\mathrm{flight}}$：起飞爬升、风机间巡航、固定五分钟巡检、返程飞行；
- $E_r^{\mathrm{escort}}$：到达母船附近后保持船尾相对位置的伴飞等待能量；
- $E_r^{\mathrm{dock}}$：目标对准、甲板相位等待、最终下降、接地和一次复飞储备。

正式计算中不再单独增加最终下降项：

$$
E_r^{\mathrm{land}}=0,
\qquad
T_r^{\mathrm{land}}=0.
$$

最终下降已唯一包含在 $E_r^{\mathrm{dock}}$ 与 $t_r^{\mathrm{dock}}$ 中。

### 4.2 航段风三角

对任一航段，设航迹方向单位向量为 $\widehat{\boldsymbol e}$，巡航高度风向量为 $\boldsymbol w$。分解为

$$
w_{\parallel}=\boldsymbol w^{\mathsf T}\widehat{\boldsymbol e},
\qquad
w_{\perp}
=
\left\|
\boldsymbol w-w_{\parallel}\widehat{\boldsymbol e}
\right\|_2.
$$

保持航迹的可达地速采用

$$
v^{g}
=
w_{\parallel}
+
\sqrt{v_{\mathrm{air}}^2-w_{\perp}^2}.
$$

每条航段必须满足物理空速包络：

$$
v_{\mathrm{air}}^{\min}
\le
v_{\mathrm{air}}^{\mathrm{req}}
\le
v_{\mathrm{air}}^{\max}.
$$

### 4.3 船尾伴飞

正式伴飞模式为船尾相对位置保持。UAV 的地速等于船速，因此名义所需空速向量为

$$
\boldsymbol v^{\mathrm{air}}_{r,\mathrm{escort}}
=
\boldsymbol v^{\mathrm{ship}}_r
-
\boldsymbol w_r.
$$

相应所需空速为

$$
v^{\mathrm{req}}_{r,\mathrm{escort}}
=
\left\|
\boldsymbol v^{\mathrm{ship}}_r-
\boldsymbol w_r
\right\|_2.
$$

固定 $h_r$ 后，令：

- $T_r^{\mathrm{fixed}}$：起飞、出程、台间、巡检等不含返程与对接的固定飞行时间；
- $T_r^{\mathrm{return}}$：名义返程时间；
- $t_r^{\mathrm{dock,wait}}$：用于计算最坏伴飞等待的对接时间下界；
- $P_r^{\mathrm{escort}}$：伴飞功率。

则计划伴飞时间为

$$
T_r^{\mathrm{escort}}
=
\max\left\{
0,
60h_r
-T_r^{\mathrm{fixed}}
-T_r^{\mathrm{return}}
-t_r^{\mathrm{dock,wait}}
\right\}.
$$

伴飞能量为

$$
E_r^{\mathrm{escort}}
=
\frac{P_r^{\mathrm{escort}}T_r^{\mathrm{escort}}}{3600}.
$$

该实现不再使用正式固定 $13\ \mathrm{m/s}$ 盘旋；`legacy_loiter` 仅保留为历史消融模式。

### 4.4 两条仿射能量分支

返程距离变化会同时改变返程飞行能量和伴飞等待能量。令 $d$ 为返程距离，$P_r^{\mathrm{return}}$ 为返程功率，$v_r^{\mathrm{return}}$ 为返程地速，并令

$$
C_r
=
60h_r
-T_r^{\mathrm{fixed}}
-t_r^{\mathrm{dock,wait}}.
$$

无伴飞等待分支为

$$
E_{r,0}(d)
=
E_r^{\mathrm{fixed}}
+
\frac{P_r^{\mathrm{return}}}{3600v_r^{\mathrm{return}}}d.
$$

伴飞等待激活分支为

$$
E_{r,1}(d)
=
E_r^{\mathrm{fixed}}
+
\frac{P_r^{\mathrm{return}}}{3600v_r^{\mathrm{return}}}d
+
\frac{P_r^{\mathrm{escort}}}{3600}
\left(
C_r-\frac{d}{v_r^{\mathrm{return}}}
\right).
$$

不含对接储备的名义计划能量取两分支最大值：

$$
E_r^0(d)
=
\max\{E_{r,0}(d),E_{r,1}(d)\}.
$$

因此两条分支对返程距离的斜率分别为

$$
c_{r,0}^{E}
=
\frac{P_r^{\mathrm{return}}}{3600v_r^{\mathrm{return}}},
$$

$$
c_{r,1}^{E}
=
\frac{P_r^{\mathrm{return}}-P_r^{\mathrm{escort}}}
{3600v_r^{\mathrm{return}}}.
$$

完整计划能量最终为

$$
E_r^{\mathrm{plan}}
=E_r^0+E_r^{\mathrm{dock}}.
$$

## 5. 误差模糊集与统计支持

对统计格 $(h,c)$，外层船位预测误差矩为

$$
\boldsymbol\mu_{h,c}
=
\mathbb E[\boldsymbol\xi_{h,c}],
\qquad
\boldsymbol\Sigma_{h,c}
=
\operatorname{Cov}(\boldsymbol\xi_{h,c}).
$$

矩信息模糊集可写为

$$
\mathcal P_{h,c}
=
\left\{
\mathbb P:
\mathbb E_{\mathbb P}[\boldsymbol\xi]
=\boldsymbol\mu_{h,c},\ 
\operatorname{Cov}_{\mathbb P}(\boldsymbol\xi)
=\boldsymbol\Sigma_{h,c}
\right\}.
$$

若同时启用风预报误差，联合不确定向量写为

$$
\boldsymbol u
=
\begin{bmatrix}
\boldsymbol\xi\\
\Delta\boldsymbol w
\end{bmatrix},
\qquad
\boldsymbol\mu_u
=
\begin{bmatrix}
\boldsymbol\mu_{h,c}\\
\boldsymbol b_w
\end{bmatrix},
$$

并按当前代码假设采用分块协方差：

$$
\boldsymbol\Sigma_u
=
\begin{bmatrix}
\boldsymbol\Sigma_{h,c} & \boldsymbol 0\\
\boldsymbol 0 & \boldsymbol\Sigma_w
\end{bmatrix}.
$$

该分块形式表示优化模型未利用船位误差与风误差之间的跨源相关性。

## 6. 分布鲁棒机会约束

### 6.1 一般仿射矩DRCC

对仿射随机消耗

$$
Z=\bar z+\boldsymbol a^{\mathsf T}\boldsymbol u,
$$

要求

$$
\inf_{\mathbb P\in\mathcal P}
\mathbb P\{Z\le \bar b\}
\ge 1-\epsilon.
$$

代码中的一阶矩充分条件为

$$
\bar b-\bar z
-\boldsymbol a^{\mathsf T}\boldsymbol\mu_u
-\kappa(\epsilon)
\sqrt{
\boldsymbol a^{\mathsf T}
\boldsymbol\Sigma_u
\boldsymbol a
}
\ge 0.
$$

可选择的单侧系数包括：

$$
\kappa_{\mathrm{Cantelli}}(\epsilon)
=
\sqrt{\frac{1-\epsilon}{\epsilon}},
$$

$$
\kappa_{\mathrm{VP}}(\epsilon)
=
\sqrt{\frac{4}{9\epsilon}-1},
$$

其中 VP 系数仅在声明的单峰投影假设及适用域内使用；超出适用域时回退 Cantelli。高斯分位数模式仅作为分布假设对照，不属于分布无关证书。

`solve_fleet_anytime()` 的实际 $\kappa$ 模式必须显式传入、记录并绑定到模型合同。正式 exact BPC 将其解析为不可变 `RiskPolicy` 并贯穿全部风险相关物理函数；外部种子重验也使用同一对象。正式证书路径不得依赖、修改或“修改后恢复”模块全局 `RM.kappa`。


### 6.1.1 连续空速功率上界

在 `wait_and_speed` 回收策略中，返程/伴飞能量需要连续空速区间 $[v_{lo},v_{hi}]$ 上的严格功率最大值。对 Zeng 模型写成

$$
P(V)=s_p\left[P_0\left(1+\frac{3V^2}{U_{tip}^2}\right)+P_i f(V)+c_dV^3+P_{elec}\right],
$$

其中

$$
x=\frac{V^2}{2v_0^2},\qquad
f(V)=\sqrt{\sqrt{1+x^2}-x}=\exp\!\left(-\frac12\operatorname{asinh}x\right).
$$

其导数可写为

$$
P'(V)=s_pV\left[
\frac{6P_0}{U_{tip}^2}+3c_dV-
\frac{P_if(V)}{2v_0^2\sqrt{1+x^2}}
\right].
$$

括号内前两项单调不减，而 $f(V)/\sqrt{1+x^2}$ 为正且单调不增，因此整个括号单调不减。于是 $P'(V)$ 在 $V>0$ 上至多从负变正一次，$P(V)$ 是单谷函数；任意闭区间最大值严格位于端点：

$$
\max_{V\in[v_{lo},v_{hi}]}P(V)=\max\{P(v_{lo}),P(v_{hi})\}.
$$

实现按 `P_zeng` 的 $V\ge0.01$ 截断口径处理低速端，并作向外浮点舍入。该结果不依赖有限速度网格，也不改变原功率模型。非物理或非有限系数直接拒绝。

### 6.2 能量DRCC

两条能量分支分别分配

$$
\epsilon_{E,q}=\frac{\epsilon_E}{2},
\qquad q\in\{0,1\}.
$$

对分支 $q$，定义名义预算

$$
b_{E,rq}
=
B^{\mathrm{use}}
-E_{r,q}^{0}
-E_r^{\mathrm{dock}}.
$$

在一阶模式下，分支余量为

$$
m_{E,rq}
=
b_{E,rq}
-\boldsymbol a_{E,rq}^{\mathsf T}\boldsymbol\mu_u
-\kappa\!\left(\frac{\epsilon_E}{2}\right)
\sqrt{
\boldsymbol a_{E,rq}^{\mathsf T}
\boldsymbol\Sigma_u
\boldsymbol a_{E,rq}
}.
$$

路线能量余量取最严格分支：

$$
m_{E,r}
=
\min_{q\in\{0,1\}}m_{E,rq}.
$$

因此

$$
\mathbb P
\left(
E_r^{\mathrm{actual}}>B^{\mathrm{use}}
\right)
\le
\sum_{q=0}^{1}\frac{\epsilon_E}{2}
=
\epsilon_E.
$$

默认 `soc_correction="geo2d"` 时，船位误差部分不使用返程距离的一阶切平面，而使用二维精确几何恒等式：

$$
d(\boldsymbol\xi)
=
\sqrt{
\left(d_0+\boldsymbol g^{\mathsf T}\boldsymbol\xi\right)^2
+
\left(\boldsymbol g_{\perp}^{\mathsf T}\boldsymbol\xi\right)^2
}.
$$

代码再对沿返程方向和垂直方向分配双侧风险预算，构造高概率距离上界 $D^{\mathrm{geo2d}}$。当同时启用风误差时，船位误差使用二维几何界，风误差继续使用一阶矩界，并在两类误差源之间进行内部 Bonferroni 分配。该内部拆分属于能量事件 $\epsilon_E$ 的组成部分，不得在任务级风险预算中重复计数。

### 6.3 时间DRCC

名义时间预算为

$$
b_{T,r}
=
60h_r
-T_r^{\mathrm{flight},0}
-t_r^{\mathrm{dock}}.
$$

时间余量为

$$
m_{T,r}
=
b_{T,r}
-\boldsymbol a_{T,r}^{\mathsf T}\boldsymbol\mu_u
-\kappa(\epsilon_T)
\sqrt{
\boldsymbol a_{T,r}^{\mathsf T}
\boldsymbol\Sigma_u
\boldsymbol a_{T,r}
}.
$$

要求

$$
m_{T,r}\ge 0,
\qquad
b_{T,r}\ge 0.
$$

### 6.4 离散回收目标与不确定性边界

当前有限模型不再引入独立的近端 `acquisition` 随机变量。回收时长 $h$ 是路线决策的一部分，计划回收点由

$$
\boldsymbol p^{\mathrm{rec}}_{\mathrm{plan}}(h)=\widehat{\boldsymbol p}_{\mathrm{ship}}(t_0+h)
$$

唯一确定；实现回收点由同一 horizon 的真实 AIS/CV 船位预测误差给出：

$$
\boldsymbol p^{\mathrm{rec}}_{\mathrm{real}}(h)=\boldsymbol p^{\mathrm{rec}}_{\mathrm{plan}}(h)+\boldsymbol\xi_h.
$$

因此，`h` 的离散选择、预测船位和 $\boldsymbol\xi_h$ 共同定义当前模型中的回收位置。`acq_error_e_m/acq_error_n_m`、视觉/RTK 末端传感器误差、捕获机构误差等明确位于当前有限模型范围之外；正式模式既不要求这些数据，也不允许通过填零或合成误差把它们伪装成实测通道。对应的证明边界仅覆盖“移动母船预测位置 + $\boldsymbol\xi_h$ + 回收状态/天气/甲板/伴飞/资源”模型，不覆盖真实硬件的传感器级终端捕获可靠性。

### 6.5 天气门、航段空速、对接储备与伴飞

当启用天气不确定性时，路线还必须分别通过：

1. 浪高门；
2. 着舰风门；
3. 航段空速包络；
4. 对接储备风下侧；
5. 船尾伴飞空速包络。

对航段空速，代码利用范数关于风向量的 $1$-Lipschitz 性，构造风误差半径 $r_{\mathrm{air}}$，要求

$$
\max_{\ell\in\mathcal L_r}
 v_{r\ell}^{\mathrm{req}}
+r_{\mathrm{air}}
\le
v_{\mathrm{air}}^{\max}.
$$

对船尾伴飞，要求

$$
v_{r,\mathrm{escort}}^{\mathrm{req}}
+r_{\mathrm{escort}}
\le
v_{\mathrm{air}}^{\max}.
$$

回收点可能触发不同的局部天气场选择。正式路线判定对该路线所有可能的候选回收天气场进行最坏情况聚合：

$$
L_r
=
\prod_{c\in\mathcal W_r^{\mathrm{cand}}}L_{rc},
$$

$$
t_r^{\mathrm{dock}}
=
\max_{c\in\mathcal W_r^{\mathrm{cand}}}t_{rc}^{\mathrm{dock}},
$$

$$
E_r^{\mathrm{dock}}
=
\max_{c\in\mathcal W_r^{\mathrm{cand}}}E_{rc}^{\mathrm{dock}},
$$

并使用最短候选对接时长计算最长可能的伴飞等待：

$$
t_r^{\mathrm{dock,wait}}
=
\min_{c\in\mathcal W_r^{\mathrm{cand}}}t_{rc}^{\mathrm{dock}}.
$$

只有所有候选天气场均开门，即 $L_r=1$，路线才可行。

## 7. 任务级95%风险预算

每个架次的联合失败事件至少包括

$$
F_r
=
F_{E,r}
\cup F_{T,r}
\cup F_{\mathrm{wave},r}
\cup F_{\mathrm{wind},r}
\cup F_{\mathrm{air},r}
\cup F_{\mathrm{dock},r}
\cup F_{\mathrm{escort},r}.
$$

当前活跃事件集合为

$$
\mathcal E_{\rm active}=\{E,T,\mathrm{wave},\mathrm{wind},\mathrm{air},\mathrm{dock},\mathrm{escort}\},
$$

不包含模型外的视觉/RTK acquisition 事件。Bonferroni预算要求

$$
\epsilon_E
+
\epsilon_T
+
\epsilon_{\mathrm{wave}}
+
\epsilon_{\mathrm{wind}}
+
\epsilon_{\mathrm{air}}
+
\epsilon_{\mathrm{dock}}
+
\epsilon_{\mathrm{escort}}
\le 0.05.
$$

当前默认天气开启配置为

$$
\begin{aligned}
\epsilon_E &= 0.0125, &
\epsilon_T &= 0.0125,\\
\epsilon_{\mathrm{wave}} &= 0.0050, &
\epsilon_{\mathrm{wind}} &= 0.0050,\\
\epsilon_{\mathrm{air}} &= 0.0050, &
\epsilon_{\mathrm{dock}} &= 0.0025,\\
\epsilon_{\mathrm{escort}} &= 0.0025.
\end{aligned}
$$

因此七个活跃事件的默认预算总和为 $0.045$，严格不超过任务失败预算上限 $0.05$。剩余 $0.005$ 是未分配的保守余量，不对应任何模型外传感器误差。

因此

$$
\sum_{j\in\mathcal E_{\rm active}}\epsilon_j\le 0.05.
$$

由并集界：

$$
\sup_{\mathbb P\in\mathcal P}
\mathbb P(F_r)
\le 0.05,
$$

从而模型目标为

$$
\inf_{\mathbb P\in\mathcal P}
\mathbb P(\text{架次 }r\text{ 整体成功})
\ge 0.95.
$$

能量内部的两分支预算 $\epsilon_E/2+\epsilon_E/2$ 已包含在单个能量事件 $\epsilon_E$ 内，不得再次加入上述活跃事件总和。默认 weather-on 配置的七个活跃事件预算和为 $0.045\le0.05$；剩余 $0.005$ 仅为未分配的保守余量，并不代表任何未建模传感器事件的风险预算。

## 8. 路线可行性定义

候选列 $r$ 属于正式可行列集合 $R$，当且仅当同时满足：

$$
\begin{aligned}
&\tau_r+h_r\le T,\\
&m_{E,r}\ge 0,\\
&m_{T,r}\ge 0,\\
&b_{T,r}\ge 0,\\
&L_r=1,\\
&\text{航段空速门通过},\\
&\text{目标获取门通过},\\
&\text{船尾伴飞门通过},\\
&\widehat c^{\mathrm{rec}}_{r,h_r}\notin\mathcal C^{\mathrm{forbid}},\\
&\text{任务风险预算合同有效}.
\end{aligned}
$$

## 9. UAV、电池SOC与周转资源

### 9.1 决策变量

主问题列选择变量：

$$
x_r\in\{0,1\},
$$

正式主问题不引入独立风机完成变量。覆盖由集合打包服务量

$$
s_i(x)=\sum_{r\in R}a_{ir}x_r\in\{0,1\}
$$

直接给出；这避免仅写成 $y_i\le\sum_r a_{ir}x_r$ 所造成的松弛语义歧义。

资源子问题为每条被选列分配：

$$
u_{rk}\in\{0,1\},
\qquad
z_{rb}\in\{0,1\},
$$

分别表示路线 $r$ 是否由 UAV $k$ 执行，以及是否使用实体电池组 $b$。

对每条被选路线：

$$
\sum_{k\in K}u_{rk}=x_r,
\qquad
\sum_{b\in B}z_{rb}=x_r.
$$

### 9.2 完整SOC需求

正式业务定义要求实体电池占用能量不低于完整计划能量：

$$
E_r^{\mathrm{soc}}
=
E_r^{\mathrm{plan}}
+U_{E,r},
\qquad
U_{E,r}\ge 0.
$$

由能量余量得到的原始鲁棒需求为

$$
E_{r}^{\mathrm{soc,raw}}
=
B^{\mathrm{use}}-m_{E,r}.
$$

因此满足正式业务定义的实现应采用

$$
E_r^{\mathrm{soc}}
=
\max\left\{
E_r^{\mathrm{plan}},
E_r^{\mathrm{soc,raw}}
\right\},
$$

$$
U_{E,r}
=
\max\left\{
E_r^{\mathrm{soc,raw}}-E_r^{\mathrm{plan}},
0
\right\}.
$$

> **实现状态：已闭合。** `step10_model_routing.py::route_feasible_at_h()` 的固定速度和可调速度分支均执行上述最大值；`step11.solve_resource_master()` 还会拒绝任何 `E_soc_required_Wh < E_plan_Wh` 的外部列。因此有利的带符号均值不能把实体电池占用降到名义计划能量以下。

单组 M350 两块 TB65 的额定能量为

$$
B^{\mathrm{rated}}=526.4\ \mathrm{Wh}.
$$

结束SOC下限为 $20\%$，因此可调度能量为

$$
B^{\mathrm{use}}
=(1-0.20)B^{\mathrm{rated}}
=421.12\ \mathrm{Wh}.
$$

对每个实体电池组 $b$：

$$
\sum_{r\in R}E_r^{\mathrm{soc}}z_{rb}
\le
B_b^{\mathrm{use}}.
$$

若按时间顺序将分配给电池 $b$ 的任务记为 $r_{b,1},r_{b,2},\ldots$，则SOC递推为

$$
\operatorname{SOC}_{b,0}=1,
$$

$$
\operatorname{SOC}_{b,m+1}
=
\operatorname{SOC}_{b,m}
-
\frac{E_{r_{b,m}}^{\mathrm{soc}}}{B_b^{\mathrm{rated}}},
$$

并要求

$$
\operatorname{SOC}_{b,m}\ge 0.20.
$$

不同实体电池组的剩余SOC不得合并。电池首次使用后在规划窗口内固定绑定一架 UAV，不允许无时间、无成本跨 UAV 转移。

### 9.3 甲板与任务区间

定义：

$$
t_r^{\mathrm{launch,start}}
=
\max\{0,\tau_r-t^{\mathrm{launch}}\},
$$

$$
t_r^{\mathrm{rec}}=\tau_r+h_r,
$$

$$
t_r^{\mathrm{clear,end}}
=t_r^{\mathrm{rec}}+t^{\mathrm{clear}}.
$$

任务的起降甲板占用为两个半开区间：

$$
\mathcal D_r
=
\left[
 t_r^{\mathrm{launch,start}},\tau_r
\right)
\cup
\left[
 t_r^{\mathrm{rec}},t_r^{\mathrm{clear,end}}
\right).
$$

单一起降甲板要求任意同时刻 $t$：

$$
\sum_{r:t\in\mathcal D_r}x_r\le 1.
$$

UAV任务占用区间至少覆盖

$$
\left[
 t_r^{\mathrm{launch,start}},
 t_r^{\mathrm{clear,end}}
\right).
$$

同一 UAV 的任务链还必须满足相邻任务的周转就绪时间。

### 9.4 快速检查与换电

设同一 UAV 的相邻任务为 $r\rightarrow s$。

若继续使用同一电池组，则执行快速检查：

$$
t_s^{\mathrm{launch,start}}
\ge
t_r^{\mathrm{clear,end}}+t^{\mathrm{quick}}.
$$

快速检查工位占用区间为

$$
\mathcal Q_{rs}
=
\left[
 t_r^{\mathrm{clear,end}},
 t_r^{\mathrm{clear,end}}+t^{\mathrm{quick}}
\right).
$$

若更换电池，则执行非着陆区换电：

$$
t_s^{\mathrm{launch,start}}
\ge
 t_r^{\mathrm{clear,end}}+t^{\mathrm{swap}}.
$$

换电工位占用区间为

$$
\mathcal S_{rs}
=
\left[
 t_r^{\mathrm{clear,end}},
 t_r^{\mathrm{clear,end}}+t^{\mathrm{swap}}
\right).
$$

任意时刻的工位容量分别满足

$$
\sum_{(r,s):t\in\mathcal Q_{rs}}q_{rs}
\le C^{\mathrm{quick}},
$$

$$
\sum_{(r,s):t\in\mathcal S_{rs}}s_{rs}
\le C^{\mathrm{swap}}.
$$

快速检查和换电均不继续占用着陆甲板。当前默认容量均为可配置的正整数，默认值为 $1$，不是无限并行。

### 9.5 Formal battery half-cap strengthening

令每条路线的严格 SOC 需求为 \(E_r^{\mathrm{soc}}\)，单块电池 usable capacity 为 \(B_{\mathrm{use}}\)，实体电池数为 \(B\)。定义

\[
\mathcal R_H=\{r\in\mathcal R:2E_r^{\mathrm{soc}}>B_{\mathrm{use}}\}.
\]

任意两条 \(\mathcal R_H\) 中的路线若由同一块电池执行，则其 SOC 需求之和严格超过 \(B_{\mathrm{use}}\)。因此每块电池至多承载一条该类路线，得到正式有效不等式

\[
\boxed{\sum_{r\in\mathcal R_H}x_r\le B.}
\]

该 row 是完整实体电池/SOC 可行域的必要条件，只用于加强 RMP 松弛和对应定价对偶；它不替代逐块电池的 exact SOC/turnaround resource audit，也不改变整数可行域。

## 10. 词典序优化模型

每条路线内部有序风机序列 $S_r$ 必须无重复，且主问题满足风机互斥集合打包：

$$
\sum_{r\in R}a_{ir}x_r\le 1,\qquad i\in I.
$$

因此整数可行解的唯一覆盖数满足

$$
C(x)=\sum_{i\in I}\sum_{r\in R}a_{ir}x_r
=\sum_{r\in R}\left(\sum_{i\in I}a_{ir}\right)x_r
=\sum_{r\in R}|S_r|x_r.
$$

第一阶段是纯覆盖目标：

$$
C^*=\max\left\{\sum_{r\in R}|S_r|x_r:x\in\mathcal F\right\},
$$

代码以最小化形式求解

$$
\min\; -\sum_{r\in R}|S_r|x_r.
$$

第一阶段目标不包含架次数、计划能耗或任何人工 tie-break。只有 $C^*$ 被证明后，第二阶段加入固定覆盖等式

$$
\sum_{r\in R}|S_r|x_r=C^*
$$

并求

$$
E^*=\min\left\{\sum_{r\in R}E_r^{\mathrm{plan}}x_r:x\in\mathcal F,\ C(x)=C^*\right\}.
$$

正式模型不要求所有风机必须覆盖，不以最少架次为正式目标，也不存在正式第三层目标。第二层使用完整计划能量 $E_r^{\mathrm{plan}}$，而不是历史名义能量口径。

## 11. 独立统计验证

优化阶段使用 train 数据估计矩，validation 用于模型与风险参数选择。方案冻结后，test 只用于独立审计。

设最终选中架次数为

$$
R_s=|R^{\mathrm{sel}}|.
$$

对每条架次 $r$，使用其独立回放样本计算失败次数 $f_r$ 和样本量 $n_r$。单侧二项上置信界使用同时置信水平

$$
1-\frac{0.05}{R_s}.
$$

记所得失败率上界为 $U_r$。这里 0.05 是 simultaneous confidence construction 的 family-wise error level，不是当前 selection risk gate。正式 weather-on baseline 的 selection 判据使用预声明内部 allocation budget $\epsilon_{\rm alloc}\approx0.045$：

$$
\max_{r\in R^{\mathrm{sel}}}U_r\le \epsilon_{\rm alloc}.
$$

任务 requirement $0.05$ 另行报告，不能替换该更严格 selection gate。gate-critical $U_r$ 必须以未舍入 binary64 值比较。

若多个架次复用同一批历史样本，不得把“架次 $\times$ 样本”行直接合并为独立伯努利样本。合并点估计或合并置信区间只能作为诊断，不构成正式证书。

缺少真实天气误差、真实回收状态、独立性标记、冻结方案或正式test样本时，`formal_reliability_claim_eligible` 必须为假。传感器级终端获取误差不在当前有限模型范围内。

### 11.1 真实回收状态门

优化只能使用起飞时可获得的预测状态 $\widehat c_{\tau,h}$。validation/test必须携带真实回收时刻标签

$$
c^{\mathrm{actual}}_{s,h}.
$$

该标签由完整轨迹在回收时刻计算，只用于事后审计，禁止反馈给路线生成。若

$$
c^{\mathrm{actual}}_{s,h}\in C^{\mathrm{forbid}},
$$

则 `recovery_state_gate` 失败并计入任务联合失败；当前禁止集合至少包含“转弯”。缺少该列、使用计划状态替代或使用合成状态时，只能形成诊断结果，不得形成正式可靠性声明。

## 12. 正式优化模型与证书范围

正式路线空间是隐式有限集合，不在求解开始前完整物化。每条路线保留离散起飞时刻、有序无重复风机序列、离散回收时长、船舶预测、原飞行/能耗/风场/DRCC诊断、计划能耗与 SOC 安全需求。

风机允许不服务，因此正式约束是风机互斥集合打包：

$$
\sum_{r\in\mathcal R}a_{ir}x_r\le1,\qquad i\in\mathcal I.
$$

不是 set partitioning。路线内部必须满足 $|S_r|=|\operatorname{set}(S_r)|$，访问顺序属于路线身份。

第一阶段纯最大化 $\sum_r|S_r|x_r$；只有覆盖最优被证明后，第二阶段固定覆盖并最小化 $\sum_rE_r^{plan}x_r$。实体 UAV、电池、SOC、快检、换电和甲板资源由三态精确审计函数 $\Phi$ 判定。

## 13. 隐式 Branch-Price-and-Cut

正式入口维护节点 RMP，并用隐式穷举式 elementary-sequence DFS 按需产生路线。当前黑箱物理/DRCC没有经证明的可加状态，因此证书路径不做非平凡 dominance 或状态合并；每次定价遍历全部允许起飞选项、有序无重复前缀和离散回收时长。RMP 缺列不可行时使用弹性 Phase-I/Farkas 等价定价。

完备分支依次使用服务量、弧流和具体路线变量的 0/1 分支；每个分支条件进入定价或节点等式。资源 DFS 的前序依赖使可行性不保证向下封闭，因此资源不可行整数集合只产生精确模式排除割

$$
\sum_{r\in S}x_r-\sum_{r\notin S}x_r\le |S|-1,
$$

而不再使用会删除全部超集的 $\sum_{r\in S}x_r\le |S|-1$。资源审计超时未知不加割。

## 14. Anytime 界与 Gap

覆盖 incumbent 必须通过精确资源审计。每个开放节点保留对完整隐式路线空间有效的覆盖上界；定价超时时用严格约化成本 bound 修正，缺少 bound 时退回节点允许风机数。全局覆盖上界为开放节点上界最大值。

覆盖绝对和相对 Gap 为

$$
Gap_C^{abs}=\overline C-C^{inc},\qquad
Gap_C^{pct}=100\frac{\overline C-C^{inc}}{\max(1,\overline C)}.
$$

覆盖未闭合时，`energy_gap_abs_Wh` 和 `energy_gap_pct` 为 `None`，原因是 `coverage optimum not proven`。覆盖已经闭合、但统一 deadline 在第二阶段启动前耗尽时，两者同样为 `None`，原因明确记录为 `energy stage not completed before deadline`。覆盖闭合且第二阶段运行后，固定覆盖分支树的开放节点能耗下界最小值形成全局能耗下界。

受限列池结果只允许 `bound_scope="validated_route_pool"`；正式物理隐式路线证书使用 `bound_scope="global_discrete_physical_model"`；synthetic finite-route fixture 只能使用 `synthetic_finite_route_fixture`，不能继承物理证书范围。`route_space_complete=False` 表示路线未被完整物化，不表示定价界无效。

## 15. 单一墙钟截止时间

最外层只创建一次单调时钟 deadline。初始列、RMP、Phase-I/Farkas、精确定价、分支树、资源 DFS、精确模式割循环和第二阶段均读取同一剩余时间。时间终止返回资源可行 incumbent、安全全局界、Gap、界来源和未完成原因。

## 16. 不声明的内容

任何全局最优或 Gap 仅针对当前有限离散输入模型，不覆盖连续真实世界、网格外决策、未建模控制故障、真实平台绝对安全或所有未来海况。真实业务实验缺少用户显式本地数据时必须 fail-closed。


## 17. 空方案与当前定价实现边界

模型没有“至少巡检一台风机”的约束，因此 `x=0` 是合法资源可行方案，覆盖为 0、计划能耗为 0。正式求解器必须把它作为初始 incumbent；只有业务方明确要求非空任务时，才应在模型中显式加入覆盖下界，而不能在结果后处理中删除空方案。

当前 exact pricing 不预先建立完整路线池，但通过 DFS 穷尽每个离散起飞时刻下的全部有序无重复风机前缀和全部回收时长。因物理/DRCC 是整体黑箱函数，当前没有经证明安全的非平凡资源 dominance 或状态合并；故它是精确的隐式枚举式定价，而不是高效非枚举 RCSP 标号算法。

## 17.1 严格资源可行域的数值定义

本模型把离散化后保存的 binary64 时间端点视为 finite model 数据。半开区间 $[a,b)$ 的重叠定义严格为

\[
\boxed{\max(a_1,a_2)<\min(b_1,b_2)}.
\]

因此哪怕交集长度仅为 $5\times10^{-10}$ min，只要在 binary64 数据上严格为正，就属于真实冲突；不能用 `1e-9` containment tolerance 消去。

对同一 UAV 的连续任务，若前任务清场结束为 $c$，快速检查时长为 $q$、换电时长为 $w$，则下一任务开始分别要求

\[
s\ge c+q
\]

或

\[
s\ge c+w.
\]

不存在 `start >= ready - 1e-7` 一类正松弛。电池约束对每块实体电池 $b$ 为

\[
\boxed{
\sum_{r:\,r\text{ assigned to }b}E_r^{soc}\le B_{\mathrm{use}}
}.
\]

其中每个输入 binary64 `E_soc_required_Wh` 都先用 `Fraction.from_float` 转成其精确有理数，再进行累计与比较；固定的 `1e-7 min` 或 `1e-6 Wh` 不属于正式数学模型。

正式 RMP 中的 active/deck 必要容量行与最终实体资源 oracle 使用同一时间语义：事件时刻不做十进制 `round`，membership 严格按 $a\le t<b$ 计算。RMP 行只是必要条件，最终 `FEASIBLE` 仍必须由实体 UAV/电池资源审计给出；`INFEASIBLE_PROVEN` 才允许生成 exact-pattern resource cut。

## 18. 有限模型身份与资源数值合同

正式证书不仅绑定参数模板，还必须绑定本次被优化的有限实例。定义参数合同哈希

\[
H_P=H(\mathrm{Params},T_{\min},S_{\max},\Gamma,K,B,\text{deck/quick/swap/resource settings},\ldots),
\]

实例合同哈希

\[
H_I=H(\text{turbines},\text{launch options},\text{weather},\Xi,\text{effective resource data},\ldots),
\]

最终模型身份

\[
\boxed{H_M=H(H_P,H_I)}.
\]

代码字段对应为：

- `parameter_contract_sha256 = H_P`；
- `instance_contract_sha256 = H_I`；
- `model_contract_sha256 = H_M`；
- `algorithm_contract_sha256` 另外记录定价模式、批大小和用户 Gap 停止目标。

浮点实例量按 binary64 精确表示（等价于 `float.hex()`）进入规范序列化，因此当 `x` 与 `nextafter(x,\pm\infty)` 代表不同有限状态时，实例身份必须不同。任何会改变 turbine、launch、weather、Xi、资源配置、路线空间或可行域的数据变化，都必须改变当前有限模型 identity。

正式资源数值合同把存储的 binary64 时间端点和能量值解释为有限模型中的精确数据：半开区间冲突使用严格 `<`，UAV 周转使用严格 `start>=ready`，电池 SOC 使用 exact rational accumulation 后比较 `<=B_use`。因此 `INT_TOL`、`ENERGY_TOL_WH` 或 event-time 十进制 round 不属于本节正式模型定义；若历史研究代码使用这些近似，它求解的是不同的近似模型，不能继承本模型的 `global_certificate_available`。


## 严格 binary64 物理边界补充

本节所有正式物理不等式均按“输入和中间正式状态的 binary64 值视作有限模型数据”的语义执行。特别地，`v_req <= v_air_max`、`v_req >= v_air_min`、固定 touchdown 的 `time_core <= h` 与任务窗 `tau+h <= T_min` 均没有正 tolerance 扩域；`nextafter(limit,+inf)` 只要在当前有限模型中严格越界就必须判不可行。天气/Xi 风险参数 `eps` 不设置人工下限；任务级 Bonferroni 分配不允许超出 `mission_failure_budget`，并保持原预算闭合合同。上述规则属于当前模型语义合同，而不是仅用于测试的数值偏好。Xi 与天气 covariance 的输入合同区分处理。正式 Xi CSV 只存一个 `sigma_en`，因此其 2×2 协方差在 schema 上天然对称；loader 将三个 binary64 条目视作精确实数，要求 `sigma_ee>=0`、`sigma_nn>=0` 且 `sigma_ee*sigma_nn-sigma_en^2>=0` 精确成立，不使用尺度相对特征值 tolerance。`step7` 在输出前完成 binary64-safe PSD 收尾，并保留 covariance 的 binary64 精度，避免 0.1 展示性 rounding 制造非法矩。正式 Xi `h_min` 必须精确命中 5,10,...,60 分钟统计网格，禁止 round/int/nearest 吸附；purge 必须严格满足 `purge_min>=max_horizon`，nonoverlap 样本只在下一起点 `t0>=previous_t1` 时保留。天气 covariance 由独立矩阵计算产生，可保留正常 ULP 级对称误差而不要求逐 bit 对称；天气残差矩仍按 exact binary64 horizon 取值，缺格时该路线缺少正式不确定性支撑，不能借最近邻 cell。直接内存构造的 XiAmbiguity 也必须满足同一 covariance 数学合同；层级收缩和同状态 horizon 插值属于模型内部派生运算，派生矩在进入后续 DRCC 前做 deterministic binary64-safe PSD canonicalization，而不是放宽原始 CSV 的合法性。


## 真实历史天气预测误差模型

天气名义场仍由真实历史 ERA5/CMEMS 时间序列提供。天气不确定性不再在 formal 路径中由相邻 reanalysis 差分近似，而由独立的数据预处理阶段估计：对每个 launch time `t0` 和 horizon `h`，`weather_speed_primary_coherent_noleak` 仅使用 `t<=t0` 的最近两条小时记录作后向线性外推，得到风矢量、风速和 Hs 的预测；`t0+h` 的历史 ERA5/CMEMS 线性插值只作为 realized truth。残差定义为 `truth - forecast`。二维风矢量残差估计均值与完整 2×2 协方差，风速大小和 Hs 残差分别估计偏置与标准差。

Formal WeatherAmbiguity 必须：horizon 精确匹配 Xi 的实际支持子集；只由 train residual 估计；train weather residual 必须从共享天气时间轴自身按 horizon 全局 nonoverlap 抽样，而不能由多船 Xi 行重复加权；purge 继承 step7 的 formal 切分合同；2×2 wind covariance 在 binary64-as-real 下 PSD；并绑定 predictor/timestamp/truth/data contracts 与源文件 SHA。该 formal 语义代表“自定义 no-leak predictor 对真实历史 truth proxy 的误差”，不声称使用 ECMWF operational forecast archive。

## 19. 算法证明与模型之间的审计接口

本文件定义有限离散模型本身；`doc_algorithm.md` 定义求解过程；`doc_proof.md` 第 15 节集中证明五个最关键的连接部位：完整空间 reduced-cost correction、Phase-I/Farkas、exact-pattern resource cut、branch/pricing 一致性以及 binary64 interval certificate。模型层任何新增主问题行、路线类型、零覆盖路线、资源随机变量或分支变量，都必须重新检查这些证明义务，不能仅修改模型后继续沿用旧 `global_certificate_available` 语义。

## 20. 当前模型语义核对

本轮证书强化**不改变本文件数学模型**：路线仍非空 elementary sequence，起飞与 recovery horizon 仍为既有有限离散集合，目标仍为 coverage 后 fixed-coverage energy 的严格两阶段词典序，回收位置仍仅由离散 horizon + ship prediction + `xi_h` 给出，terminal acquisition error 仍在模型外。新增的 rounded-RMP exact 行复核和 pricing-interruption full-space bound 仅属于求解证书控制流。


## 21. Canonical route identity 与证书宇宙

正式路线变量仍定义为 `r=(o,pi,h)`，没有改变路线、物理或目标语义。为使 exact-pattern cuts、Stage-2 objective、pooled-energy 行与实体资源 Oracle 始终指向同一数学变量，canonical route signature 在一次正式求解中必须对应唯一且不可变的 formal column semantics：`E_plan_Wh`、`E_soc_required_Wh`、有序访问、route arcs 与 resource intervals 必须 binary64-exact 一致。同 signature 但不同语义不是 dominance 机会，而是模型/实现不一致，正式求解 fail-closed。

同时区分实际搜索宇宙与模型宇宙。记 `R_used` 为求解器实际定价的路线集、`R_Phi={r:Phi(r)=1}` 为正式物理/DRCC 路线集。正式物理全局证书额外要求 `R_used=R_Phi` 的 provenance contract。人工 `implicit_test_columns` 只定义 synthetic `R_used`，因此即使 Branch-Price-and-Cut 对该有限集合算法上完全闭合，也不能推出正式 physical model 的全局最优。

## 22. Vessel identity、risk budget 与 final-test separation

The baseline finite model retains the existing mission-risk event set and
top-level risk allocation. The current protocol does **not** loosen epsilon, replace VP with a
Gaussian assumption, remove `geo2d`, or extend the formal recovery horizon.
Those alternatives are separate sensitivity/model variants.

The current data/protocol contract requires that the Xi ambiguity object used
by a formal physical certificate is conditioned on one concrete vessel MMSI;
cross-vessel pooled statistics are outside the formal model. The same vessel is
used for validation and final-test replay. The final test is not part of E1
selection: E1 freezes a validation-approved exact configuration, E2 freezes
the final method/plan after its complete validation matrix, and only then may the independent test be consumed. A1/A2 are algorithm experiments and do not participate in final-test candidate selection.

Heuristic multi-stop seeds and deterministic physical caching change only the
search order/cost. Every formal route remains an element of the same finite
route universe `R_Phi`, and exact pricing must still establish omitted-column
closure before a global certificate is available.

## 23. Coverage-target decision 与 battery structural cap

The coverage-target scheduler does **not** alter the finite physical model. Xi remains concrete-vessel,
Weather/geo2d/wait-and-speed/VP settings remain unchanged, and the 30-minute
baseline horizon remains unchanged. `coverage-target` is a decision query on
the same feasible set:
\[
\mathcal F_T(K,B)=
\{x\in\mathcal F(K,B):\sum_r |S_r|x_r=T\}.
\]
Consequently a target NO certificate may refine a coverage bound, but cannot be
used as an energy-optimal or lexicographic-global certificate.

The battery structural cap follows from nonempty routes and turbine packing:
\[
\sum_r x_r\le \sum_r |S_r|x_r\le |I|.
\]
Thus \(B\ge |I|\) permits a distinct battery group for every selected route;
larger \(B\) cannot enlarge the integer route-selection feasible set under the
current battery-group semantics.
