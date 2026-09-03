# 相关工作与研究空白

本文件用于论文定位，不按开发版本编年。当前算法贡献围绕论文算法 R-BPC 的实际语义表述：

1. moving-vessel recovery 的 finite physical route model；
2. single-vessel no-leak Xi + coherent weather residual DRCC；
3. exact entity UAV/battery/SOC/deck resource semantics；
4. R-BPC：resource-aware exact Branch-and-Price-and-Cut with implicit route column generation；
5. small-n certified complete physical route universe；
6. full-cover target exact resource closure；
7. universe-level global battery necessary relaxation；
8. staged resource-frontier certification；
9. optimization certificate 与 validation/final-test statistical evidence 分离；
10. source-bound proof/provenance 作为 exactness 的组成部分。

不应把贡献写成“新的电池物理模型”“现实连续系统全局最优”或“已证明所有平台 95% safe”。

### Manuscript 迁移边界

历史 Manuscript 只能作为建模/算法思想来源，不能把旧数字和当前 formal claim 自动合并。当前代码与旧稿至少存在以下重要区别：

- 当前 battery 是实体组 + exact SOC 累计，不是“每 sortie 消耗一个 battery count”；
- 当前处理 UAV binding、turnaround、deck、inspection/swap；
- 当前 risk protocol 区分 internal allocation gate、mission requirement 与 simultaneous UCB；
- 当前 formal Xi 是 concrete single vessel；
- 当前 weather predictor 是 coherent no-leak；
- 当前 small-n exact claim 依赖 certified complete finite physical route universe；
- current proof contract 绑定实际源码与 provenance。

论文中的任何实验优势必须来自当前合同下的 fresh/archived formal result，而不是旧稿表格。


## 0. 当前研究定位

本项目研究船载无人机在海上风电场中的多架次巡检：母船沿预测航迹持续运动，无人机从母船起飞，巡检若干风机后在未来时刻返回母船。模型同时处理：

- AIS/轨迹预测得到的未来回收点及按 `(h,c(τ))` 分格的船位预测误差矩；
- 风、浪、航段空速、对接储备和船尾伴飞风险；回收目标由离散 horizon 与母船预测位置共同决定，传感器级近端目标获取误差不在当前有限模型内；
- 离舰时刻、访问顺序、计划回收时长、UAV、电池 SOC、甲板、快检和换电资源；
- 两层词典序目标：先最大化六小时内可靠完成的不同风机数，再最小化完整计划能耗；
- 正式自定义 Branch-Price-and-Cut、Phase-I/Farkas fallback 与实体资源 Logic-Based Benders。实体资源 oracle 对 finite binary64 model 使用严格半开时间区间、严格 UAV 周转和逐块电池 exact SOC 累计；small-n 可选择经完整性认证的 materialized physical route universe 作为 exact acceleration；论文主 certified instance 可使用 `formal-route-universe=off` 并由隐式 full-space exact pricing closure 完成 coverage 证书；历史 Ryan–Foster 与受限 heuristic pool 仅用于算法回归。

基于当前文献集合，更稳妥的研究空白表述是：**尚未发现有研究同时整合移动未来回收点、预测误差矩、决策索引的模糊集、海上回收作业门/资源风险、实体电池 SOC 和有限模型证书。** 该结论仅对当前文献集合成立，仍需数据库检索复核。

## 1. 候选贡献与文献对照

### 1.1 未来移动回收点与无泄漏预测误差

本项目的回收点由起飞时可获得的信息预测，真实回收点偏差由外层误差 `ξ` 描述；回收时转弯状态由逐时长预测状态控制。与确定性移动平台、固定岸基或巡检期间静止平台相比，差异在于回收位置和允许回收状态都具有预测属性。

当前文献集合中的相近方向包括：移动船舶/USV与UAV协同路由、岸到船配送、AIS轨迹预测和移动甲板控制。现有对照材料可支撑“这些要素分别存在”，但不能仅凭本文件断言没有其他工作将它们组合。

### 1.2 回收时长索引的矩信息模糊集

候选列中的回收时长 `h` 决定读取哪个 `(h,c(τ))` 误差统计格，因此优化决策会选择不同的均值和协方差。准确表述应为“决策索引的模糊集选择”或“decision-dependent ambiguity-cell selection”。这与决策直接改变分布参数的更一般 DD-DRO 模型相关，但不应在没有形式化映射证明时声称两者完全等价。

Luo & Mehrotra、Basciftci et al. 和 Yu & Shen 可作为决策依赖 DRO 的方法学背景；AIS预测文献可支撑预测误差随预测时长变化。它们是否覆盖本项目的具体 UAV 回收结构，需要逐篇证据表确认。

### 1.3 矩信息 DRCC 与独立统计验证

项目支持 Cantelli、VP 等风险界，并通过事件分解和 Bonferroni 预算形成任务级充分条件。不能笼统写成“所有联合风险被一个 SOC 精确重构”；不同事件采用不同标量界、锥界或工程门，伴飞能量还采用两条仿射分支的并集界。

E2的正确研究问题是：在相同 train/validation/test、路线空间和最终资源模型下，比较 DRCC、SAA、确定性及其他基线的覆盖—能耗—可靠性权衡。由于当前 ZIP 不内置正式平台数据、正式 E2 CSV 或冻结后的独立 test，接口在无数据时fail-closed；本文档不陈述“只有DRCC通过”“SAA过拟合”或“DRCC最高安全覆盖”等结果。

### 1.4 海上回收作业门与资源过程进入路由层

项目把浪高门、风速门、航段空速、船尾伴飞和 dock 储备纳入列可行性；回收位置不确定性由离散回收 horizon 对应的真实 AIS 校准 $\xi_h$ 描述。weather-on 默认由七个活跃事件分配风险预算，总和为 0.045，不超过 0.05 的任务失败预算上限；未分配的 0.005 是保守余量，不对应任何未建模传感器事件。

海上着舰控制文献主要提供控制层、相对导航和甲板运动背景；海上 O&M 文献提供天气窗口与作业门背景。是否已有同类工作把全部终端事件纳入路由层，应通过可复核检索确认。

## 2. 与代表性工作的限定性对照

符号：✓ 明确包含；△ 相近但语义不同；✗ 在当前摘要/提取记录中未见；“待核”表示必须回到原文确认。

| 文献方向 | 移动回收平台 | 预测船位误差 | 起飞/回收时长决策 | 风浪/终端门 | DRO/机会约束 | 实体电池SOC与周转 | 当前求解方法 |
|---|---|---|---|---|---|---|---|
| Ismail et al. 2025，USV+UAV风机巡检 | △ 平台 | ✗ | 待核 | ✗/待核 | ✗ | 待核 | 路由与调度方法，需查原文 |
| He et al. 2025，UAV-USV鲁棒协同 | ✓ | ✗，旅行时间预算不确定 | 待核 | △ | 预算鲁棒 | 待核 | set partitioning + B&P&C |
| Li et al. 2025，vessel–UAV协同配送 | ✓ | ✗，轨迹确定性 | △同步决策 | △ | ✗ | 待核 | MISOCP + ALNS |
| Wang et al. 2026a，岸到船无人机路由 | △ 客户船移动、岸基起点 | ✗ | △连续会合 | ✗/待核 | ✗ | 待核 | MISOCP + branch-and-price |
| 移动甲板着舰控制文献 | ✓ 控制层 | △姿态/运动预测 | ✗路由决策 | ✓ | 通常非路由DRCC | ✗ | MPC/视觉伺服/运动补偿 |
| 决策依赖DRO方法学 | ✗场景无关 | 形式化不确定参数 | ✓一般决策依赖 | ✗ | ✓ | ✗ | 锥/MILP/多阶段方法 |
| **本项目有限模型** | **✓** | **✓，按 `(h,c(τ))`** | **✓** | **✓，七个活跃事件，默认和0.045≤0.05** | **✓，矩信息DRCC** | **✓** | **自定义BPC+Phase-I/Farkas+资源Benders；定价为隐式全排列DFS** |

该表只用于说明模型维度，不构成“全球首次”证明。

## 3. 文献主题分类

当前参考文献表共43条，按以下主题组织：

1. 海上风电 UAV 巡检；
2. 船载/移动平台 UAV 回收与控制；
3. UAV-TSP/VRP、移动平台和岸到船配送；
4. 鲁棒、随机、分布鲁棒和决策依赖优化；
5. AIS 船舶轨迹预测与误差建模；
6. 海上 O&M 调度与风浪环境；
7. 列生成、branch-and-price、RCSPP和完整枚举；
8. 词典序/层级目标路径优化。

正式综述必须给每篇文献保存：PDF文件名、页码证据、提取人/工具、提取日期、题名和DOI核验状态。仅有参考文献列表不足以证明“逐篇精读”。

## 4. 可使用的理论支撑与限制

| 论文论点 | 可作为背景的文献 | 使用限制 |
|---|---|---|
| 海上风机可使用 UAV 巡检 | Ismail et al.; Fontenla-Carrera et al.; Kabbabe Poleo et al. | 不能由此推导移动母船和95%可靠性 |
| AIS可用于船舶轨迹预测 | Murray & Perera; Gao et al.; Li et al.; Liu et al. | 必须另行证明当前预测器无泄漏及误差统计有效 |
| 矩信息机会约束/DRCC重构 | Calafiore & El Ghaoui; Zhao et al. | 需逐条匹配假设，不能把所有事件统称为同一精确SOC |
| 决策依赖模糊集 | Luo & Mehrotra; Basciftci et al.; Yu & Shen | 当前模型是按决策选择统计格，需谨慎说明映射关系 |
| 海上风浪作业门 | de Matos Sá et al.; Si et al.; Tian et al.; Procházka et al.; Xu et al. | 控制层或O&M阈值不等于路由层联合可靠性证明 |
| 列生成/B&P | He et al.; Wang et al. 2026a; Irawan et al. | 只支撑算法背景；当前证书来自自定义分支定价树、隐式全排列精确定价、Phase-I/Farkas 与实体资源审计 |
| 词典序路径目标 | Bent & Van Hentenryck; Shi et al. | 它们的“车辆数优先”仅是历史结构类比，不支撑当前覆盖优先目标 |

投稿前建议将研究空白分为两类：

- **当前集合支持的相对空白：** 在已收集文献中未见同时覆盖全部核心维度；
- **仍需检索验证的强空白：** “首次应用”“无任何先例”“唯一方法”等绝对主张，当前证据不足，不应使用。

## 5. 参考文献（APA 格式，按作者姓氏字母序）

> **书目信息状态：** 以下条目继承自历史工作底稿，本轮文档审计只对部分 2025–2026 题名和 DOI 做了抽查，没有对43条逐项重新验证。投稿前必须以出版社页面、Crossref或原始PDF逐条复核；`[待核verify]` 表示已知仍需确认。★ 仅表示历史底稿中的后补条目，不表示本轮新检索。

Basciftci, B., Ahmed, S., & Shen, S. (2021). Distributionally robust facility location problem under decision-dependent stochastic demand. *European Journal of Operational Research, 292*(2), 548–561. https://doi.org/10.1016/j.ejor.2020.11.002 ★

★ Bent, R., & Van Hentenryck, P. (2004). A two-stage hybrid local search for the vehicle routing problem with time windows. *Transportation Science, 38*(4), 515–530. https://doi.org/10.1287/trsc.1030.0049

Bruni, M. E., Khodaparasti, S., & Perboli, G. (2023). The drone latency location routing problem under uncertainty. *Transportation Research Part C: Emerging Technologies, 156*, 104322. https://doi.org/10.1016/j.trc.2023.104322

Calafiore, G. C., & El Ghaoui, L. (2006). On distributionally robust chance-constrained linear programs. *Journal of Optimization Theory and Applications, 130*(1), 1–22. https://doi.org/10.1007/s10957-006-9084-x

Cho, G., Choi, J., Bae, G., & Oh, H. (2022). Autonomous ship deck landing of a quadrotor UAV using feed-forward image-based visual servoing. *Aerospace Science and Technology, 130*, 107869. https://doi.org/10.1016/j.ast.2022.107869

Dalgic, Y., Lazakis, I., & Turan, O. (2015). Advanced logistics planning for offshore wind farm operation and maintenance activities. *Ocean Engineering, 101*, 211–226. https://doi.org/10.1016/j.oceaneng.2015.04.040

de Matos Sá, M., Correia da Fonseca, F. X., Amaral, L., & Castro, R. (2024). Optimising O&M scheduling in offshore wind farms considering weather forecast uncertainty and wake losses. *Ocean Engineering, 301*, 117518. https://doi.org/10.1016/j.oceaneng.2024.117518

Dukkanci, O., Kara, B. Y., & Bektaş, T. (2021). Minimizing energy and cost in range-limited drone deliveries with speed optimization. *Transportation Research Part C: Emerging Technologies, 125*, 102985. https://doi.org/10.1016/j.trc.2021.102985

Fontenla-Carrera, G., Aldao Pensado, E., Veiga-López, F., & González-Jorge, H. (2025). Efficient offshore wind farm inspections using a support vessel and UAVs. *Ocean Engineering, 332*, 121416. https://doi.org/10.1016/j.oceaneng.2025.121416

Gao, D.-w., Zhu, Y.-s., Zhang, J.-f., He, Y.-k., Yan, K., & Yan, B.-r. (2021). A novel MP-LSTM method for ship trajectory prediction based on AIS data. *Ocean Engineering, 228*, 108956. https://doi.org/10.1016/j.oceaneng.2021.108956

Ha, Q. M., Deville, Y., Pham, Q. D., & Hà, M. H. (2018). On the min-cost traveling salesman problem with drone. *Transportation Research Part C: Emerging Technologies, 86*, 597–621. https://doi.org/10.1016/j.trc.2017.11.015

He, Q., Liu, W., Liu, T.-L., & Tian, Q. (2025). Robust coordinated path planning for unmanned aerial vehicles and unmanned surface vehicles in maritime monitoring with travel time uncertainty. *Transportation Research Part B: Methodological, 199*, 103284. https://doi.org/10.1016/j.trb.2025.103284

Irawan, C. A., Ouelhadj, D., Jones, D., Stålhane, M., & Sperstad, I. B. (2017). Optimisation of maintenance routing and scheduling for offshore wind farms. *European Journal of Operational Research, 256*(1), 76–89. https://doi.org/10.1016/j.ejor.2016.05.059

Irawan, C. A., Starita, S., Chan, H. K., Eskandarpour, M., & Reihaneh, M. (2023). Routing in offshore wind farms: A multi-period location and maintenance problem with joint use of a service operation vessel and a safe transfer boat. *European Journal of Operational Research, 309*(1), 460–481. https://doi.org/10.1016/j.ejor.2022.07.051

Ismail, A. H., Song, X., Ouelhadj, D., Al-Behadili, M., & Fraess-Ehrfeld, A. (2025). Unmanned surface vessel routing and unmanned aerial vehicle swarm scheduling for offshore wind turbine blade inspection. *Expert Systems With Applications, 284*, 127534. https://doi.org/10.1016/j.eswa.2025.127534

Kabbabe Poleo, K., Crowther, W. J., & Barnes, M. (2021). Estimating the impact of drone-based inspection on the levelised cost of electricity for offshore wind farms. *Results in Engineering, 9*, 100201. https://doi.org/10.1016/j.rineng.2021.100201

Ksciuk, J., Kuhlemann, S., Tierney, K., & Koberstein, A. (2023). Uncertainty in maritime ship routing and scheduling: A literature review. *European Journal of Operational Research, 308*(2), 499–524. https://doi.org/10.1016/j.ejor.2022.08.006

Lazakis, I., & Khan, S. (2021). An optimization framework for daily route planning and scheduling of maintenance vessel activities in offshore wind farms. *Ocean Engineering, 225*, 108752. https://doi.org/10.1016/j.oceaneng.2021.108752

Li, H., Jiao, H., & Yang, Z. (2023a). AIS data-driven ship trajectory prediction modelling and analysis based on machine learning and deep learning methods. *Transportation Research Part E: Logistics and Transportation Review, 175*, 103152. https://doi.org/10.1016/j.tre.2023.103152

Li, H., Jiao, H., & Yang, Z. (2023b). Ship trajectory prediction based on machine learning and deep learning: A systematic review and methods analysis. *Engineering Applications of Artificial Intelligence, 126*, 107062. https://doi.org/10.1016/j.engappai.2023.107062

Li, J., Zhang, G., Jiang, C., & Zhang, W. (2023c). A survey of maritime unmanned search system: Theory, applications and future directions. *Ocean Engineering, 285*, 115359. https://doi.org/10.1016/j.oceaneng.2023.115359

Li, Y., Wang, S., Sun, H., & Zhou, S. (2025). Collaborative vessel–unmanned aerial vehicle routing for time-window-constrained offshore parcel delivery. *Transportation Research Part C: Emerging Technologies, 178*, 105189. https://doi.org/10.1016/j.trc.2025.105189

Liu, R. W., Hu, K., Liang, M., Li, Y., Liu, X., & Yang, D. (2023). QSD-LSTM: Vessel trajectory prediction using long short-term memory with quaternion ship domain. *Applied Ocean Research, 136*, 103592. https://doi.org/10.1016/j.apor.2023.103592

★ Luo, F., & Mehrotra, S. (2020). Distributionally robust optimization with decision dependent ambiguity sets. *Optimization Letters, 14*(8), 2565–2594. https://doi.org/10.1007/s11590-020-01574-3

Ma, Y., Liu, Y., Bai, X., Guo, Y., Yang, Z., Wang, L., Tao, T., & Zhang, L. (2024). DivideMerge: A multi-vessel optimization approach for cooperative operation and maintenance scheduling in offshore wind farm. *Renewable Energy, 229*, 120758. https://doi.org/10.1016/j.renene.2024.120758

Meng, S., Li, D., Liu, J., & Chen, Y. (2024). The multi-visit drone-assisted routing problem with soft time windows and stochastic truck travel times. *Transportation Research Part B: Methodological, 190*, 103004. https://doi.org/10.1016/j.trb.2024.103004

Murray, B., & Perera, L. P. (2020). A dual linear autoencoder approach for vessel trajectory prediction using historical AIS data. *Ocean Engineering, 209*, 107478. https://doi.org/10.1016/j.oceaneng.2020.107478

Murray, B., & Perera, L. P. (2021). An AIS-based deep learning framework for regional ship behavior prediction. *Reliability Engineering & System Safety, 215*, 107819. https://doi.org/10.1016/j.ress.2021.107819

Murray, C. C., & Chu, A. G. (2015). The flying sidekick traveling salesman problem: Optimization of drone-assisted parcel delivery. *Transportation Research Part C: Emerging Technologies, 54*, 86–109. https://doi.org/10.1016/j.trc.2015.03.005

Murray, C. C., & Raj, R. (2020). The multiple flying sidekicks traveling salesman problem: Parcel delivery with multiple drones. *Transportation Research Part C: Emerging Technologies, 110*, 368–398. https://doi.org/10.1016/j.trc.2019.11.003

Procházka, O., Novák, F., Báča, T., Gupta, P. M., Pěnička, R., & Saska, M. (2024). Model predictive control-based trajectory generation for agile landing of unmanned aerial vehicle on a moving boat. *Ocean Engineering, 313*, 119164. https://doi.org/10.1016/j.oceaneng.2024.119164

★ Shi, Y., Zhou, Y., Boudouh, T., & Grunder, O. (2020). A lexicographic-based two-stage algorithm for vehicle routing problem with simultaneous pickup–delivery and time window. *Engineering Applications of Artificial Intelligence, 95*, 103901. https://doi.org/10.1016/j.engappai.2020.103901

Si, G., Xia, T., Wang, D., Gebraeel, N., Pan, E., & Xi, L. (2025). Maintenance scheduling and vessel routing for offshore wind farms with multiple ports considering day-ahead wind-wave predictions. *Applied Energy, 379*, 124915. https://doi.org/10.1016/j.apenergy.2024.124915

Sperstad, I. B., Stålhane, M., Dinwoodie, I., Endrerud, O.-E. V., Martin, R., & Warner, E. (2017). Testing the robustness of optimal access vessel fleet selection for operation and maintenance of offshore wind farms. *Ocean Engineering, 145*, 334–343. https://doi.org/10.1016/j.oceaneng.2017.09.009

Tian, E., Li, Y., Liao, Y., & Cao, J. (2024). UAV-USV docking control system based on motion compensation deck and attitude prediction. *Ocean Engineering, 307*, 118223. https://doi.org/10.1016/j.oceaneng.2024.118223

Wang, M., Chen, S., & Meng, Q. (2026a). Drone routing problem for shore-to-ship delivery services considering non-linear energy consumption. *Transportation Research Part B: Methodological, 206*, 103410. https://doi.org/10.1016/j.trb.2026.103410

Wang, S., Li, Y., & Xing, H. (2023). A novel method for ship trajectory prediction in complex scenarios based on spatio-temporal features extraction of AIS data. *Ocean Engineering, 281*, 114846. https://doi.org/10.1016/j.oceaneng.2023.114846

Wang, Y., Yang, M., Hu, Q., & Xu, M. (2026b). Routing and scheduling problem for mothership and drones in shore-to-ship delivery. *Transportation Research Part E: Logistics and Transportation Review, 209*, 104730. https://doi.org/10.1016/j.tre.2026.104730

Xu, R., Liu, C., Cao, Z., Wang, Y., & Qian, H. (2024). A manipulator-assisted multiple UAV landing system for USV subject to disturbance. *Ocean Engineering, 299*, 117306. https://doi.org/10.1016/j.oceaneng.2024.117306

Yang, Y., Hao, X., & Wang, S. (2025). The drone scheduling problem in shore-to-ship delivery: A time discretization-based model with an exact solving approach. *Transportation Research Part B: Methodological, 191*, 102990. https://doi.org/10.1016/j.trb.2024.102990

★ Yu, X., & Shen, S. (2022). Multistage distributionally robust mixed-integer programming with decision-dependent moment-based ambiguity sets. *Mathematical Programming, 196*, 1025–1064. https://doi.org/10.1007/s10107-021-01742-y [待核verify: DOI 请投稿前对照出版社页面核实]

Zhang, J., Campbell, J. F., Sweeney II, D. C., & Hupman, A. C. (2021). Energy consumption models for delivery drones: A comparison and assessment. *Transportation Research Part D: Transport and Environment, 90*, 102668. https://doi.org/10.1016/j.trd.2020.102668

Zhao, Y., Chen, Z., Lim, A., & Zhang, Z. (2022). Vessel deployment with limited information: Distributionally robust chance constrained models. *Transportation Research Part B: Methodological, 161*, 197–217. https://doi.org/10.1016/j.trb.2022.05.006


## 6. 当前算法定位

当前正式实现仍应表述为“基于隐式路线列生成与精确定价的精确分支–定价–割算法”。它避免完整路线集合的预物化，但 exact pricing 仍是最坏情况下完备遍历组合路线空间的 elementary DFS；本轮没有宣称或实现未经证明的 RCSP/ESPPRC dominance。Multi-column 接纳和 route-pool 复用属于 exactness-preserving acceleration，不能替代最终定价证书。


## Certificate provenance as part of exactness

本项目把“算法对一个有限列宇宙求到全局最优”与“该列宇宙正是论文声明的正式物理路线宇宙”区分为两个证明层。前者是 Branch-Price-and-Cut 的 algorithmic closure，后者是 route-universe provenance。synthetic route fixtures 可用于独立 oracle 与 mutation testing，但不能继承 physical-model certificate。这一划分避免将测试替身、受限路线池或研究枚举器的 exactness 错译为原 physical/DRCC 模型的 exactness。

同理，column identity 不只用于哈希：exact-pattern cuts 等逻辑要求一个 signature 在整个搜索期间表示同一个数学变量。因此正式实现采用 semantic-invariance fail-closed，而不是对同 signature 的不同能耗 representation 做经验 dominance。当前 exact pricing 仍是 implicit exhaustive elementary DFS；没有由于本轮证书强化而宣称新的 RCSP/ESPPRC dominance。

## 7. Exact optimization 与 statistical/model assumptions

The implementation distinguishes exact optimization from statistical/model
assumptions. Exactness is conditional on the declared finite binary64
single-vessel ambiguity model; it is not a claim that pooling heterogeneous
vessels preserves unimodality or that the physical world is globally
optimized. This separation is important when comparing DD-DRO/DRCC routing
studies that report robustness but do not bind the prediction-error population
to the operated asset.

Likewise, heuristic route generation is used only as primal initialization.
This follows the branch-price principle that a good column pool may accelerate
incumbent discovery while only a complete pricing argument can certify the
full route universe. The new physical cache and service-floor pruning are
exact-search accelerations rather than empirical dominance assumptions.

## 8. E1 staged certification 的方法学定位

The final E1 scheduler separates three proof tasks that are often conflated in
heuristic fleet-routing studies: finding a high-coverage incumbent, proving a
coverage threshold infeasible at a predecessor resource point, and proving
energy optimality conditional on the selected coverage. The current scheduler uses a dedicated
exact target-feasibility BPC certificate for the second task and reserves the
full lexicographic certificate for the final knee only. This preserves exact
model claims while avoiding repeated full solves of resource cells that cannot
affect the selected knee.

The empirical validation layer likewise distinguishes the internally allocated
Bonferroni/DRCC budget from the externally stated 5% mission requirement rather
than treating an empirical point estimate or a looser mission threshold as an
optimization certificate.
