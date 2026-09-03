#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step9_model.py — 船载无人机海上风机巡检 DRCC 调度模型（参数、物理、模糊集）。

严格对应 model.md(符号一致)与 params.md(数值来源)。本文件只负责:
  1. 数据加载    : 风机坐标、天气(风/浪)、ξ 误差矩(xi_moments_caseB.csv)
  2. 参数容器    : Params(全部来自 params.md;[待定]/SCN 用占位让代码跑通)
  3. 几何/物理   : §6 能耗 E_r(ξ)、时间 T_r(ξ)、3D 高度飞行、高度风廓线、地速
  4. 着舰门      : §7 简化 6DOF 甲板运动门 L_r 与有效捕获半径
  5. 模糊集      : §4 由矩信息(支持集 Ξ、均值 μ、协方差 Σ)构造 P_{h,c}

**不在本文件**:DRCC 可处理化与 Gurobi 求解 → step10_model_routing.py + step11/step12(配套)。
本文件可独立运行做自检(python step9_model.py),用占位/真实数据跑通能耗-时间-门链路。

依赖: pip install numpy pandas   (Gurobi 仅 step11/step12 主问题需要)

数据契约(列名以作者本机实测为准):
  turbines_*_clean.csv : turbine_id, lon, lat, farm, osm_id
  waves_*.csv          : time, Hs_m, wave_Tm_s, wave_dir_deg
  weather_*.csv(ERA5) : time, wind10_ms, wind100_ms, wind_dir_from_deg[, Hs_m, ...]
  xi_moments_caseB.csv : mmsi,h_min,c_state,n,mu_e_m,mu_n_m,sigma_ee,sigma_en,sigma_nn,
                         rms_norm_m,p50_norm_m,p95_norm_m,max_norm_m,mad_e_m,mad_n_m
"""
from __future__ import annotations

import logging
import math
from fractions import Fraction
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("model")

R_EARTH = 6_371_000.0  # m
G = 9.80665            # m/s^2
KN = 0.514444          # 1 knot in m/s

XI_PREDICTOR_CONTRACTS = {"cv_noleak": "cv_noleak_backward_window_epoch_seconds"}
XI_TIMESTAMP_EPOCH_CONTRACT = "utc_datetime64_ns_to_epoch_seconds"
XI_FORMAL_HORIZON_GRID_MIN = tuple(range(5, 61, 5))
XI_FORMAL_GRID_CONTRACT = "exact-binary64-horizon-grid-subset-5to60-step5-v2"
XI_COVARIANCE_CONTRACT = "symmetric-2x2-binary64-as-real-psd-v1"


def _binary64_psd_cov2(see: float, sen: float, snn: float) -> bool:
    """Exact PSD test for a symmetric 2x2 covariance whose entries are binary64.

    The formal finite model treats loaded floats as exact real numbers.  For
    [[see,sen],[sen,snn]], PSD is therefore exactly see>=0, snn>=0 and
    see*snn-sen**2>=0 in rational arithmetic.  No tolerance is allowed to
    turn an invalid covariance into a certified ambiguity set.
    """
    vals = (float(see), float(sen), float(snn))
    if not all(math.isfinite(v) for v in vals):
        return False
    see_f, sen_f, snn_f = vals
    if see_f < 0.0 or snn_f < 0.0:
        return False
    return (Fraction.from_float(see_f) * Fraction.from_float(snn_f)
            >= Fraction.from_float(sen_f) * Fraction.from_float(sen_f))


def _canonicalize_binary64_psd_cov2(matrix: np.ndarray) -> np.ndarray:
    """Canonicalize an internally derived 2x2 covariance to exact binary64 PSD.

    This function is for *derived* matrices (hierarchical shrinkage/interpolation),
    not for accepting an invalid formal CSV.  It symmetrizes, projects negative
    numerical eigenvalues to zero, then shrinks the single off-diagonal value
    toward zero until the exact rational determinant is nonnegative.
    """
    S = np.asarray(matrix, dtype=float)
    if S.shape != (2, 2) or not bool(np.isfinite(S).all()):
        raise ValueError("派生 covariance 必须是有限 2x2 矩阵。")
    S = 0.5 * (S + S.T)
    vals, vecs = np.linalg.eigh(S)
    vals = np.maximum(vals, 0.0)
    S = (vecs * vals) @ vecs.T
    S = 0.5 * (S + S.T)
    a = max(float(S[0, 0]), 0.0)
    c = max(float(S[1, 1]), 0.0)
    b = float(S[0, 1])
    if a == 0.0 or c == 0.0:
        b = 0.0
    else:
        target = Fraction.from_float(a) * Fraction.from_float(c)
        while Fraction.from_float(b) * Fraction.from_float(b) > target:
            b = math.nextafter(b, 0.0)
    out = np.array([[a, b], [b, c]], dtype=float)
    if not _binary64_psd_cov2(a, b, c):
        raise ArithmeticError("binary64 PSD canonicalization failed")
    return out


# =============================================================================
# 0. 参数容器(全部对应 params.md;来源标签见注释)
# =============================================================================
@dataclass
class Params:
    """所有标量参数。数值来自 params.md;[待定]/SCN 为占位,实验前替换/扫描。"""
    # --- §2 UAV(DJI M350 RTK;能量/速度/重量=OEM,功率=SCN 锚定能量包络) ---
    B_k: float = 526.4          # Wh 电池总能量 (OEM, 2x TB65 @263.2)
    safe_reserve: float = 0.20  # 任务结束不可动用 SOC 下限比例 (SCN) -> B_use=(1-reserve)*B_k
    v_cr: float = 15.0          # m/s 巡航空速 (SCN, <= v_max 23)
    v_max: float = 23.0         # m/s 最大平飞 (OEM)
    v_z: float = 5.0            # m/s 爬升速度 (SCN, <= 6)
    P_cr: float = 900.0         # W 巡航功率 (legacy SCN; use_zeng=True 时被 P_zeng 取代, 仅作校准锚点)
    P_hov: float = 1000.0       # W 悬停功率 (legacy SCN; 巡检改用 P_zeng(v_orbit))
    P_climb: float = 1300.0     # W 爬升功率 (legacy SCN)
    P_wait: float = 1000.0      # W 等待功率 (legacy SCN; 等待改用 P_zeng(v_loiter), 见下)
    # --- 功率-速度模型 (更新, PR-1): Zeng, Xu & Zhang (2019, IEEE TWC) 旋翼 UAV 功率模型 ---
    #   P(V) = P0·(1+3V²/U_tip²) + Pi·(√(1+V⁴/(4v0⁴)) − V²/(2v0²))^½ + ½·d0·ρ·s·A·V³ + P_elec
    #   参数标定到 M350 RTK + H20T (7.27kg): P(0)≈833W(悬停,续航~38min与实测吻合), P(15)≈496W(最省).
    #   标注 CAL: 数值经 OEM 续航包络反推, 非 DJI 官方精确功率; 待飞行日志校准 (见 doc_params §Zeng).
    use_zeng: bool = True       # True: leg_kinematics/巡检/等待用 P_zeng; False: 回退 legacy 常数功率(A/B 对照)
    # 更新: UAV 档位功率缩放 —— P(V) = power_scale · P_zeng_base(V), 锚定 OEM 悬停功率
    #   scale = P_hover_OEM / P_zeng_base(0)。曲线形状(U 型/最省速度)不变, 仅整体幅值缩放;
    #   与作者 OEM 表的"悬停功率 = Wh/续航"推导自洽(见 doc_params §UAV 档位)。默认 1.0 = M350 基线。
    power_scale: float = 1.0
    zeng_P0: float = 49.9       # W 叶片型阻功率系数 (blade profile)
    zeng_Pi: float = 702.9      # W 诱导功率系数 (induced)
    zeng_Utip: float = 61.4     # m/s 桨尖速度
    zeng_v0: float = 7.9        # m/s 悬停诱导速度
    zeng_d0: float = 0.6        # 机身阻力比 (fuselage drag ratio)
    zeng_s: float = 0.083       # 桨叶实度 (rotor solidity)
    zeng_A: float = 0.896       # m² 桨盘总面积 (4×21in)
    zeng_rho: float = 1.225     # kg/m³ 海面空气密度
    zeng_Pelec: float = 80.0    # W 航电/图传/传感器电子负载
    v_orbit: float = 3.0        # m/s 绕飞巡检速度 (LIT: 自动化叶片巡检 1-4 m/s); 巡检功率=P_zeng(v_orbit)
    v_loiter: float = 13.0      # m/s legacy 盘旋等待速度；正式模型改用船尾伴飞能耗，保留用于消融
    # --- 固定接地返程空速 recourse（step13 默认开启；Params 单独构造仍保持 wait_only 兼容） ---
    #   核心(用户洞见): 回收时长 h 固定, 无人机以"恰好在 t_R 到达预测回收点"的速度飞行; 鲁棒性 = 船漂移 ξ 后
    #   无人机能否在 v_max 内加速追上。chance constraint 从"预留时间"变为"所需空速 ≤ v_max":
    #     P( (d_ret0 + ⟨g,ξ⟩)/T_ret_budget > v_max ) ≤ ε_T
    #   打破"小 h 时间不够、大 h 等待耗电"的双向挤压, 使 3-5 台串巡可行。
    #   speed_adjustable=False(默认): 保留 更新 的固定 v_cr + 预留时间 SOC(已测/向后兼容)。
    time_recourse_mode: str = "wait_only"  # wait_only | wait_and_speed
    speed_adjustable: bool = False  # compatibility mirror of time_recourse_mode
    v_air_floor: float = 3.0    # m/s 巡航最低空速(可变速度模式; 低于此改为盘旋等待)
    W_max: float = 12.0         # m/s 最大抗风 (OEM)
    # 空速包络(逐航段风三角可行性, model.md §6.1): 起飞时刻 τ 选定各腿风场后,
    # 每段需检查保持地面航迹所需空速 ∈ [v_air_min, v_air_max], 不能只用固定巡航速度。
    v_air_min: float = 3.0      # m/s 维持航迹的最小空速 (SCN; 低于此视为无法稳定保持航向)
    v_air_max: float = 23.0     # m/s 最大平飞空速 (= v_max, OEM)
    r_cap: float = 3.0          # m 静甲板捕获半径 (SCN; 仅用于 landing_gate r_cap_eff 报告)
    z_cruise: float = 60.0      # m 巡航(外飞/返程)飞行高度 (SCN)
    # --- §7.2 两阶段会合: 末端实时导引对接(更新 新增) ---
    # 远端(DRCC 阶段): 概率裕度把无人机送到回收区附近;
    # 近端: 船载 GNSS/RTK 外层导航后由视觉建立相对定位，进入船尾伴飞并完成着舰。
    # 资源物理口径: 对接储备【状态依赖】—— 时间随甲板运动恶化增长(贴门限≈×(1+γ))、
    #   功率随回收风速按 Zeng 取(顶风驻停≈以风速为空速飞行), 由 dock_reserve() 计算。
    #   旧常数 t_dock_s=120 / E_dock_Wh=33.3(后者=已废弃的 P_wait 1000W×2min 遗迹)已删除;
    #   标定锚: 中位天气 dock_reserve ≈ 28–32Wh, 与旧值平滑衔接(数值迁移最小)。
    # 注: ξ 量级 186–989m >> r_cap=3m; 末端实时导引消除剩余偏差, 不由 r_cap 捕获约束描述(见 doc_model §7.2)。
    t_dock_base_s: float = 120.0   # s 静稳海况末端对接基准时长 (SCN: 视觉锁定+对准+降落 ≈2min)
    dock_gamma: float = 1.0        # 对接时长的甲板运动放大 (SCN, 作者拍板: 贴门限时 t_dock×2, 待 CAL)
    dock_beta: float = 1.15        # 对接功率控制开销 (SCN, 作者拍板: 驻停修正+突风控制 +15%, 待 CAL)
    # --- 终端回收目标与船尾伴飞（正式语义） ---
    # 回收目标不是额外的随机“视觉获取点”：离散回收时长 h 是路线决策的一部分，
    # 计划回收点由 ship.predicted_at(h) 唯一确定；实现回收点的不确定性由同一 h 下
    # 的真实 AIS/CV 船位预测误差 xi_h 描述。传感器级视觉/RTK acquisition error
    # 不属于当前有限模型，也不得以零误差或合成误差偷偷加入正式可靠性声明。
    recovery_target_model: str = "discrete_horizon_ship_prediction"
    terminal_sensor_error_mode: str = "out_of_scope"
    escort_mode: str = "stern_follow"      # 保持船体坐标系下船尾相对位置
    escort_offset_m: float = 20.0           # m 船尾等待点相对甲板参考点距离
    escort_power_beta: float = 1.10         # 伴飞控制、阵风修正功率倍率（SCN，待飞行日志标定）
    recovery_forbidden_states: tuple[str, ...] = ("转弯",)  # 转弯状态禁止进入末端伴飞/回收
    # 起飞爬升由 to_land_energy_time() 计算；最终下降完整归入 dock_reserve()，不得重复计量。
    # --- §3 任务 ---
    tau_insp: float = 5 * 60.0  # s 单机巡检悬停时长 (SCN, 待 LIT)
    # --- §4 甲板运动门(限值=SCN,门限 Hs=LIT) ---
    Hs_op: float = 1.5          # m CTV 作业波高门限 (LIT)
    s_heave_max: float = 0.5    # m 着舰垂荡限 (SCN)
    s_roll_max: float = 3.0     # deg 横摇限 (SCN)
    s_pitch_max: float = 2.0    # deg 纵摇限 (SCN)
    w_land_max: float = 12.0    # m/s 着舰风限 (SCN, <= W_max)
    kappa_h: float = 1.0        # 捕获半径随垂荡收缩 (SCN)
    kappa_rp: float = 0.05      # 捕获半径随横摇+纵摇收缩 (SCN, 每度)
    # 简化"运动~Hs"经验传递系数(SCN;真实 RAO 待 LIT):
    #   s_heave ≈ a_heave * Hs ; s_roll ≈ a_roll * Hs * f(beam sea) ...
    a_heave: float = 0.45       # 垂荡幅值/Hs (SCN, 双体船经验起步)
    a_roll: float = 2.5         # 横摇幅(deg)/Hs (SCN)
    a_pitch: float = 1.5        # 纵摇幅(deg)/Hs (SCN)
    # --- §5 DRCC ---
    # ：每架次“整体成功率 ≥95%”。当前有限模型没有独立 acquisition 随机事件；
    # 活跃 Bonferroni 分量总和不得超过 0.05，允许保守地留有未分配预算。
    # 原 eps_cap 名称为兼容保留，但语义改为“浪高门尾概率”，不再冒充末端捕获概率。
    mission_failure_budget: float = 0.05
    eps_E: float = 0.0125       # 能量不足
    eps_T: float = 0.0125       # 超时
    eps_cap: float = 0.0050     # 浪高门（legacy name）
    eps_gate: float = 0.0050    # 着舰风门
    eps_air: float = 0.0050     # 航段空速包络
    eps_dock: float = 0.0025    # 末端着舰储备风下侧
    eps_escort: float = 0.0025  # 船尾伴飞空速/跟踪包线
    # --- 实体电池组与分离式甲板资源 ---
    battery_reuse_mode: str = "exact_soc"  # exact_soc=逐电池组分配、允许剩余 SOC 顺序复用
    battery_energy_mode: str = "robust_required"  # 电池累计占用使用 DRCC 所需能量而非名义能量
    battery_binding_mode: str = "horizon_fixed_uav"  # 首次使用后在规划窗内绑定同一 UAV，不跨机转移
    landing_clear_min: float = 1.0          # min 接地后停桨并移出着陆区的清场时间(SCN,待实测)
    quick_inspection_min: float = 1.0        # min 原电池继续使用前的快速检查时间(SCN,待实测)
    quick_inspection_capacity: int = 1       # 并行快速检查工位数(SCN,可扫描)
    swap_station_capacity: int = 1          # 非着陆区换电工位数(SCN,可扫描)
    # --- §5b SOC 线性化几何校正(正式默认 geo2d; none 仅消融复现旧口径) ---
    #   'none'  : 主线一阶线性化 SOC(全部结果口径);
    #   'geo2d' : 2-D 精确几何界 d(ξ)=√(L²+ξ⊥²) ≤ √(L_abs²+t⊥²), 沿/垂返程两方向
    #             各用双侧 VP 盒 + Bonferroni 拆 ε —— 消除一阶泰勒对凸距离的系统性低估
    #             (q=0.8 强风窗 vp 判据 emp_viol 贴线越界的根因之一, 见 历史变更记录)。
    soc_correction: str = "geo2d"
    soc_share_lin: float = 0.6   # geo2d: 沿返程方向双侧盒占 ε 的份额(其余给垂向; 见 _soc_margin_geo2d)
    soc_share_wind: float = 0.2  # fixed 模式兼容份额；optimized 模式把它作为必含候选而非硬编码最优值
    soc_risk_allocation: str = "optimized"  # optimized=逐路线优化 Bonferroni 份额；fixed=旧 0.2/0.6 口径
    # --- 风廓线(对数律,中性) ---
    z0: float = 0.0002          # m 海面粗糙度 (LIT, 开阔海)

    @property
    def B_use(self) -> float:
        """可调度能量 B_k^use=(1-safe_reserve)B_k；safe_reserve 是结束 SOC 硬下限。"""
        return (1.0 - self.safe_reserve) * self.B_k

    def validate_contract(self, formal: bool = False) -> None:
        """纯校验本模型的核心语义契约，不修改任何参数。

        ``speed_adjustable`` 是 ``time_recourse_mode`` 的兼容镜像；调用方必须在
        构造/规范化参数时显式保持二者一致，校验器不会再静默改写模型对象。
        当前正式模型把回收目标定义为离散 h 对应的预测船位，传感器级 acquisition
        error 明确在模型范围之外，因此不存在“填零/合成 acquisition 参数”这一入口。
        """
        if not (0.0 < self.safe_reserve < 1.0):
            raise ValueError("safe_reserve 必须位于 (0,1)，并表示任务结束不可动用 SOC 下限。")
        # All scalar numeric model fields must be finite before any formal
        # physics can consume them.  This blocks NaN/Inf from silently turning
        # comparisons into false branches while still being echoed in a model
        # certificate.
        for _name in self.__dataclass_fields__:
            _value = getattr(self, _name)
            if isinstance(_value, bool):
                continue
            if isinstance(_value, (int, float)) and not math.isfinite(float(_value)):
                raise ValueError(f"Params.{_name} 必须是有限数值。")
        for _name in ("use_zeng", "speed_adjustable"):
            if not isinstance(getattr(self, _name), bool):
                raise ValueError(f"Params.{_name} 必须是布尔值。")

        # Physical-domain contract.  Merely being finite is not enough: zero or
        # negative denominators/energies can silently create a different feasible
        # region (or trigger division/logarithm failures) while a solver certificate
        # still refers to the malformed request.
        _positive = (
            "B_k", "v_cr", "v_max", "v_z", "power_scale",
            "zeng_Utip", "zeng_v0", "zeng_A", "zeng_rho",
            "v_air_floor", "v_air_max", "z_cruise", "z0",
            "dock_beta", "escort_power_beta",
        )
        for _name in _positive:
            if float(getattr(self, _name)) <= 0.0:
                raise ValueError(f"Params.{_name} 必须为正数。")

        _nonnegative = (
            "P_cr", "P_hov", "P_climb", "P_wait",
            "zeng_P0", "zeng_Pi", "zeng_d0", "zeng_s", "zeng_Pelec",
            "v_orbit", "v_loiter", "W_max", "v_air_min", "r_cap",
            "t_dock_base_s", "dock_gamma", "escort_offset_m",
            "tau_insp", "Hs_op", "s_heave_max", "s_roll_max",
            "s_pitch_max", "w_land_max", "kappa_h", "kappa_rp",
            "a_heave", "a_roll", "a_pitch",
            "quick_inspection_min", "landing_clear_min",
        )
        for _name in _nonnegative:
            if float(getattr(self, _name)) < 0.0:
                raise ValueError(f"Params.{_name} 不得为负。")

        if float(self.v_cr) > float(self.v_max):
            raise ValueError("v_cr 不得超过 v_max。")
        if float(self.v_air_min) > float(self.v_air_max):
            raise ValueError("v_air_min 不得超过 v_air_max。")
        if float(self.v_air_max) > float(self.v_max):
            raise ValueError("v_air_max 不得超过 v_max。")
        if float(self.v_air_floor) > float(self.v_air_max):
            raise ValueError("v_air_floor 不得超过 v_air_max。")
        if float(self.w_land_max) > float(self.W_max):
            raise ValueError("w_land_max 不得超过 W_max。")
        if not (0.0 < float(self.z0) < 10.0):
            raise ValueError("z0 必须位于 (0,10)m，保证 10m 对数风廓线分母有定义。")
        if float(self.z_cruise) <= float(self.z0):
            raise ValueError("z_cruise 必须严格高于 z0。")
        for _name in ("quick_inspection_capacity", "swap_station_capacity"):
            _value = getattr(self, _name)
            if isinstance(_value, bool) or int(_value) != _value or int(_value) < 1:
                raise ValueError(f"{_name} 必须是至少为 1 的整数。")
        if self.battery_binding_mode != "horizon_fixed_uav":
            raise ValueError("battery_binding_mode 当前只支持 horizon_fixed_uav。")
        if str(self.battery_reuse_mode) != "exact_soc":
            raise ValueError("battery_reuse_mode 当前正式物理/资源合同只支持 exact_soc。")
        if str(self.battery_energy_mode) != "robust_required":
            raise ValueError("battery_energy_mode 当前只支持 robust_required。")
        if str(self.soc_correction) not in {"none", "geo2d"}:
            raise ValueError("soc_correction 只支持 none 或 geo2d。")
        if str(self.soc_risk_allocation) not in {"fixed", "optimized"}:
            raise ValueError("soc_risk_allocation 只支持 fixed 或 optimized。")
        if str(self.soc_risk_allocation) == "fixed":
            if not (0.05 <= float(self.soc_share_lin) <= 0.95):
                raise ValueError("fixed soc_share_lin 必须位于 [0.05,0.95]，不得依赖内部夹断。")
            if not (0.05 <= float(self.soc_share_wind) <= 0.90):
                raise ValueError("fixed soc_share_wind 必须位于 [0.05,0.90]，不得依赖内部夹断。")
        else:
            if not (0.001 <= float(self.soc_share_lin) <= 0.999):
                raise ValueError("optimized soc_share_lin 必须位于 [0.001,0.999]。")
            if not (0.001 <= float(self.soc_share_wind) <= 0.999):
                raise ValueError("optimized soc_share_wind 必须位于 [0.001,0.999]。")
        _single_modes = {
            "recovery_target_model": "discrete_horizon_ship_prediction",
            "terminal_sensor_error_mode": "out_of_scope",
            "escort_mode": "stern_follow",
        }
        for _name, _supported in _single_modes.items():
            if str(getattr(self, _name)) != _supported:
                raise ValueError(f"{_name} 当前只支持 {_supported}。")
        _risk_names = ("eps_E", "eps_T", "eps_cap", "eps_gate",
                       "eps_air", "eps_dock", "eps_escort")
        for _name in _risk_names:
            _eps = float(getattr(self, _name))
            if not (0.0 < _eps < 1.0):
                raise ValueError(f"{_name} 必须位于 (0,1)。")
        if not (0.0 < float(self.mission_failure_budget) < 1.0):
            raise ValueError("mission_failure_budget 必须位于 (0,1)。")
        _risk_values = (self.eps_E, self.eps_T, self.eps_cap, self.eps_gate,
                        self.eps_air, self.eps_dock, self.eps_escort)
        _eps_exact = sum((Fraction.from_float(float(v)) for v in _risk_values), Fraction(0))
        _budget_exact = Fraction.from_float(float(self.mission_failure_budget))
        # Acquisition is outside the finite model.  The active Bonferroni split
        # is therefore allowed to be conservative (sum < mission budget), but
        # binary64-as-real over-allocation is never accepted.
        if _eps_exact > _budget_exact:
            eps_sum = float(sum(float(v) for v in _risk_values))
            raise ValueError(
                f"任务风险预算超配: components={eps_sum:.17g}, "
                f"mission={float(self.mission_failure_budget):.17g}")
        if str(getattr(self, "time_recourse_mode", "wait_and_speed")) not in {"wait_only", "wait_and_speed"}:
            raise ValueError("time_recourse_mode 必须是 wait_only|wait_and_speed。")
        _expected_speed_adjustable = (str(self.time_recourse_mode) == "wait_and_speed")
        if bool(self.speed_adjustable) != _expected_speed_adjustable:
            raise ValueError(
                "speed_adjustable 必须与 time_recourse_mode 一致；"
                "wait_and_speed=>True, wait_only=>False。")


# =============================================================================
# 1b. UAV 档位(更新, E1 三轴实验的"无人机种类"轴; 数值=作者整理 OEM 表, 见 doc_params §UAV)
#     参数化原则: 全部留在多旋翼族内 ⇒ Zeng 曲线形状不变, 仅 (容量 B_k, 悬停功率锚点, 风限,
#     甲板时间) 四轴差异; 速度包络同型(SCN, 避免混淆变量)。hover_W = OEM 电量/续航 代理
#     (与作者表推导一致): S=263.2/36min≈439W, M=474/38≈748W, L=977/53≈1106W。
#     w_land_max 用 OEM【起降】风限(着舰门用), W_max 用 OEM 抗风。t_swap/t_launch=SCN(作者表甲板时间区间)。
# =============================================================================
UAV_PROFILES = {
    # key: (label, B_k_Wh, hover_W, W_max, w_land_max, t_swap_min, t_launch_min)
    "S":    dict(label="DJI Matrice 30T",  B_k=263.2, hover_W=439.0,  W_max=12.0, w_land_max=12.0,
                 t_swap_min=4.0, t_launch_min=2.5,
                 source_type="mixed-OEM-and-scenario", battery_mass_kg=None, acquisition_cost=None),
    "M":    dict(label="Autel Alpha",      B_k=474.0, hover_W=748.0,  W_max=12.0, w_land_max=10.7,
                 t_swap_min=4.0, t_launch_min=3.0,
                 source_type="mixed-OEM-and-scenario", battery_mass_kg=None, acquisition_cost=None),
    "L":    dict(label="DJI Matrice 400",  B_k=977.0, hover_W=1106.0, W_max=12.0, w_land_max=12.0,
                 t_swap_min=5.0, t_launch_min=3.0,
                 source_type="mixed-OEM-and-scenario", battery_mass_kg=None, acquisition_cost=None),
    # 向后兼容基线(=旧 Params 默认, power_scale=1.0 逐位不变), 供回归对照
    "M350": dict(label="DJI M350 RTK(旧基线)", B_k=526.4, hover_W=None, W_max=12.0, w_land_max=12.0,
                 t_swap_min=4.0, t_launch_min=2.5,
                 source_type="legacy-baseline", battery_mass_kg=None, acquisition_cost=None),
}


def uav_parameter_audit(key: str) -> dict:
    """返回机型参数来源审计信息。未知的质量/成本保持 ``None``，绝不臆造。"""
    if key not in UAV_PROFILES:
        raise KeyError(key)
    prof = UAV_PROFILES[key]
    return {
        "uav": key,
        "label": prof["label"],
        "source_type": prof.get("source_type", "unknown"),
        "battery_capacity_Wh": float(prof["B_k"]),
        "battery_mass_kg": prof.get("battery_mass_kg"),
        "acquisition_cost": prof.get("acquisition_cost"),
        "scenario_assumptions": ["t_swap_min", "t_launch_min"],
        "normalization_ready": {
            "inventory_kWh": True,
            "battery_mass": prof.get("battery_mass_kg") is not None,
            "acquisition_cost": prof.get("acquisition_cost") is not None,
        },
    }


def apply_uav_profile(p: "Params", key: str) -> "Params":
    """返回按 UAV 档位覆盖后的 Params 深拷贝(原对象不动)。缩放锚: power_scale=hover_W/P_zeng_base(0)。
    附加属性 p.uav_key/p.uav_label 供实验 provenance。key 不在表中 ⇒ 报错(拒绝静默)。"""
    import copy
    if key not in UAV_PROFILES:
        raise SystemExit(f"未知 UAV 档位 '{key}'; 可选: {sorted(UAV_PROFILES)}")
    prof = UAV_PROFILES[key]
    q = copy.deepcopy(p)
    q.B_k = float(prof["B_k"])
    if prof["hover_W"] is not None:
        base_hover = P_zeng(0.0, Params())          # 基线悬停 ≈832.8W(power_scale=1)
        q.power_scale = float(prof["hover_W"]) / base_hover
    else:
        q.power_scale = 1.0
    # legacy 常数功率同步缩放(仅 use_zeng=False 路径消费; 保持 A/B 对照一致)
    for f in ("P_cr", "P_hov", "P_climb", "P_wait"):
        setattr(q, f, getattr(q, f) * q.power_scale)
    q.W_max = float(prof["W_max"])
    q.w_land_max = float(prof["w_land_max"])
    q.uav_key = key
    q.uav_label = prof["label"]
    return q


# =============================================================================
# 1. 几何工具
# =============================================================================
def latlon_to_local_m(lat, lon, lat0, lon0):
    """本地等距投影 -> 米 (east, north)。几十 km 尺度误差可忽略。"""
    x = R_EARTH * math.radians(lon - lon0) * math.cos(math.radians(lat0))
    y = R_EARTH * math.radians(lat - lat0)
    return np.array([x, y], dtype=float)


def wind_at_height(w10: float, z: float, z0: float) -> float:
    """对数律把 10m 风外推到高度 z (m)。中性层结。z<=10 返回按比例。"""
    if z <= 0:
        return 0.0
    z = max(z, z0 * 1.001)
    return w10 * math.log(z / z0) / math.log(10.0 / z0)


def ground_speed(v_air: float, wind_speed: float, wind_dir_from_deg: float,
                 heading_deg: float, v_floor: float = 1.0) -> float:
    r"""地速(风三角, 含横风偏流损失)。model.md §6/§6.1。
    **更新 审计 P0 修复**: 此前把【气象来向(北=0 顺时针)】与【数学航向(东=0 逆时针)】两套坐标
    直接相减(wind_to − heading), 导致顺风被当成横风(顺风地速本应 v_air+w, 却被算成 √(v_air²−w²))。
    现统一用【矢量】: 风吹向矢量(东,北)=`wind_vector_from`, 航向单位矢量=(cos,sin)(数学航向),
    $w_\parallel=\mathbf w\!\cdot\!\hat e$、$w_\perp=\lVert\mathbf w-w_\parallel\hat e\rVert$,
    地速 $v^g=w_\parallel+\sqrt{v_{air}^2-w_\perp^2}$(与 `leg_airspeed_feasibility` 完全同口径)。
    wind_dir_from_deg: 风来向(气象);heading_deg: 数学航向(东=0 逆时针, 同 `heading_deg()` 输出)。
    """
    w_vec = wind_vector_from(wind_speed, wind_dir_from_deg)   # 风吹向矢量 (east, north)
    hr = math.radians(heading_deg)
    e = np.array([math.cos(hr), math.sin(hr)])               # 航向单位矢量 (east, north)
    w_along = float(np.dot(w_vec, e))                        # 顺风为正(吹向==航向)
    w_cross = float(np.linalg.norm(w_vec - w_along * e))     # 横风分量
    fwd = math.sqrt(max(v_air ** 2 - w_cross ** 2, 0.0))     # 抵消横风后余下前向空速
    return max(w_along + fwd, v_floor)


def wind_vector_from(speed: float, dir_from_deg: float) -> np.ndarray:
    """气象"来向"(0=N 顺时针) → 风吹【去向】矢量 (east, north), m/s。"""
    to_deg = (dir_from_deg + 180.0) % 360.0
    return np.array([speed * math.sin(math.radians(to_deg)),
                     speed * math.cos(math.radians(to_deg))])


def P_zeng(V: float, p) -> float:
    r"""Zeng, Xu & Zhang (2019, IEEE TWC) 旋翼 UAV 总功率(W) at 空速 V(m/s)。
    P(V)=P0·(1+3V²/U_tip²)+Pi·(√(1+V⁴/(4v0⁴))−V²/(2v0²))^½+½·d0·ρ·s·A·V³+P_elec。
    三项: 叶片型阻(随 V² 增)、诱导(随 V 降)、机身寄生(随 V³ 增) ⇒ U 型曲线。
    参数在 Params.zeng_* (CAL, 标定 M350+H20T)。V<0.01 视为悬停(取 V=0.01 避免除零, ≈P0+Pi+P_elec)。"""
    V = max(float(V), 0.01)
    U = p.zeng_Utip; v0 = p.zeng_v0
    blade = p.zeng_P0 * (1.0 + 3.0 * V * V / (U * U))
    ind_inner = math.sqrt(1.0 + V**4 / (4.0 * v0**4)) - V * V / (2.0 * v0 * v0)
    induced = p.zeng_Pi * math.sqrt(max(ind_inner, 0.0))
    parasite = 0.5 * p.zeng_d0 * p.zeng_rho * p.zeng_s * p.zeng_A * V**3
    # 更新: UAV 档位缩放(power_scale=1.0 时逐位等于旧值, 完全向后兼容)
    return getattr(p, "power_scale", 1.0) * (blade + induced + parasite + p.zeng_Pelec)


def leg_power(p, v_air: float) -> float:
    """一段巡航腿在空速 v_air 下的功率(W)。use_zeng=True 用 Zeng P(V); 否则 legacy 立方 P_cr·(v/v_cr)³。"""
    if getattr(p, "use_zeng", False):
        return P_zeng(v_air, p)
    return p.P_cr * (v_air / p.v_cr) ** 3


def leg_airspeed_feasibility(v_air_cr: float, v_air_max: float, w_vec: np.ndarray,
                             e_unit: np.ndarray, v_floor: float = 1.0,
                             v_air_min: float = 0.0) -> tuple[bool, float, float]:
    """逐航段风三角可行性 (model.md §6.1)。给定巡航空速 v_air_cr、最大空速 v_air_max、
    该腿高度处风矢量 w_vec(吹向, m/s)、地面航迹单位向量 e_unit, 判断该腿能否飞:
      - 沿航迹分量 w_along = w·e(顺风为正); 横风分量 w_cross = |w - w_along·e|。
      - 维持航迹需抵消横风 ⇒ 必须 v_air_max ≥ w_cross(否则被横风吹离航迹, 不可行)。
      - 最大可达地速 vg_max = w_along + √(v_air_max² − w_cross²)(需 v_air_max≥w_cross); 须 > v_floor 才能前进。
      - 用模型选定地速 vg(由 ground_speed 给, 限制在 [v_floor, vg_max])反推所需空速
        V_req = |vg·e − w| = √((vg − w_along)² + w_cross²)。
      - **空速下界(更新 审计 claim6, model.md §6.1)**: 须 V_req ≥ v_air_min(默认 0; 多旋翼可悬停,
        故 v_air_min 通常取 0/很小; 若按固定翼/最小可控速度设 v_air_min>0, 则 V_req<v_air_min 的腿不可行)。
    返回 (flyable, V_req[m/s], vg_max[m/s])。flyable=False ⇒ 该腿空速越界(上界或下界), 路由不可行。
    """
    e_norm = float(np.linalg.norm(e_unit))
    if e_norm == 0.0:
        V_req = 0.0
        return bool(V_req >= float(v_air_min)), V_req, float("inf")
    e = e_unit / e_norm
    w_along = float(np.dot(w_vec, e))
    w_cross = float(np.linalg.norm(w_vec - w_along * e))
    if v_air_max < w_cross:                       # 横风超过最大空速 → 无法保持航迹
        return False, w_cross, -1.0
    vg_max = w_along + math.sqrt(max(v_air_max ** 2 - w_cross ** 2, 0.0))
    if vg_max <= v_floor:                         # 逆风过强 → 无法前进
        return False, math.hypot(v_floor - w_along, w_cross), vg_max
    # 模型飞巡航空速 v_air_cr(风三角地速, 与 ground_speed 同口径); 横风超过巡航空速则须加速到 v_air_max 保持航迹
    v_eff = v_air_cr if w_cross <= v_air_cr else v_air_max
    vg_model = max(w_along + math.sqrt(max(v_eff ** 2 - w_cross ** 2, 0.0)), v_floor)
    V_req = math.sqrt((vg_model - w_along) ** 2 + w_cross ** 2)
    if V_req < v_air_min:                         # 严格 binary64 下界；不得用正 tolerance 扩大可行域
        return False, V_req, vg_max
    return True, V_req, vg_max


def leg_kinematics(p, w_vec: np.ndarray, e_unit: np.ndarray,
                   v_floor: float = 1.0) -> tuple:
    r"""**单一逐航段运动学(更新 审计 P0-6)**: 返回 (feasible, v_air_used, vg, power_W)。
    与 `leg_airspeed_feasibility` 同一套风三角与坐标, 供 `route_nominal_ET` 的时间/能耗使用 ——
    消除"可行性按 $v_{air}^{\max}$ 判可飞、而 ET 仍按 $v_{cr}$ 算地速(强横风下 √(v_cr²−w_⊥²) 截到 v_floor)"的不一致。
      - 横风 $w_\perp\le v_{cr}$: 飞巡航空速 $v_{eff}=v_{cr}$, $v^g=w_\parallel+\sqrt{v_{cr}^2-w_\perp^2}$, 功率 $P_{cr}$。
      - $v_{cr}<w_\perp\le v_{air}^{\max}$: 须加速至 $v_{eff}=v_{air}^{\max}$ 才能保持航迹, $v^g=w_\parallel+\sqrt{(v_{air}^{\max})^2-w_\perp^2}$,
        功率按视速立方缩放 $P=P_{cr}(v_{eff}/v_{cr})^3$(保守, 巡航时恰为 $P_{cr}$)。
      - $w_\perp>v_{air}^{\max}$ 或 $v^g\le v_{floor}$(强逆风): 不可行。
    """
    L = float(np.linalg.norm(e_unit))
    if L == 0.0:
        return True, p.v_cr, p.v_cr, leg_power(p, p.v_cr)
    e = e_unit / L
    w_along = float(np.dot(w_vec, e))
    w_cross = float(np.linalg.norm(w_vec - w_along * e))
    if p.v_air_max < w_cross:                                   # 横风超最大空速 → 无法保持航迹
        return False, p.v_air_max, v_floor, leg_power(p, p.v_air_max)
    v_eff = p.v_cr if w_cross <= p.v_cr else p.v_air_max        # 横风超巡航 → 加速保持航迹
    vg_raw = w_along + math.sqrt(max(v_eff ** 2 - w_cross ** 2, 0.0))
    if vg_raw <= v_floor:                                       # 逆风过强 → 实际无法前进
        return False, v_eff, max(vg_raw, v_floor), leg_power(p, v_eff)
    power = leg_power(p, v_eff)                                 # 视速功率(Zeng P(V) 或 legacy 立方)
    return True, v_eff, vg_raw, power


def heading_deg(p_from: np.ndarray, p_to: np.ndarray) -> float:
    """从 p_from 指向 p_to 的航向(度,正东=0,逆时针为正,数学约定)。"""
    d = p_to - p_from
    return math.degrees(math.atan2(d[1], d[0]))


# =============================================================================
# 2. 数据加载
# =============================================================================
@dataclass
class Turbine:
    tid: str
    lonlat: np.ndarray   # [lon, lat]
    H_hub: float
    H_tip: float


# 风机机型规格(params.md §1, LIT)。按 farm 名给 (H_hub, H_tip)。
FARM_TURBINE_GEOM = {
    "Anholt":     dict(H_hub=81.6, H_tip=141.6),
    "Nysted":     dict(H_hub=69.0, H_tip=110.0),
    "Rodsand_II": dict(H_hub=68.5, H_tip=115.0),
}


def load_turbines(csv_path: Path, farm: Optional[str] = None) -> list[Turbine]:
    df = pd.read_csv(csv_path)
    out = []
    for _, r in df.iterrows():
        fname = str(r.get("farm", farm or "Rodsand_II"))
        geom = FARM_TURBINE_GEOM.get(fname, FARM_TURBINE_GEOM["Rodsand_II"])
        out.append(Turbine(tid=str(r["turbine_id"]),
                           lonlat=np.array([float(r["lon"]), float(r["lat"])]),
                           H_hub=geom["H_hub"], H_tip=geom["H_tip"]))
    log.info("载入 %d 台风机 (%s)", len(out), csv_path.name)
    return out


class ShipTrack:
    """真实母船 AIS 航迹: 时间序列 ``(t_sec, P_local)``。

    ``absolute_start`` 保留原始 UTC 起始时间；过去只保存相对秒，导致 AIS 与再分析天气
    无法按历史时间对齐。合成/无时间列航迹的该字段为 ``None``，并通过
    ``time_source`` 明确标记，防止被误称为真实联合回放。
    """
    def __init__(self, t_sec, P, absolute_start=None, time_source="relative"):
        import numpy as _np
        self.t = _np.asarray(t_sec, float)
        self.P = _np.asarray(P, float)             # (N,2)
        self.t0 = float(self.t[0]) if len(self.t) else 0.0
        self.absolute_start = (pd.Timestamp(absolute_start) if absolute_start is not None else None)
        self.time_source = str(time_source)

    def absolute_time(self, t_sec):
        """返回相对航迹时刻对应的 UTC 时间；无绝对时间时返回 ``None``。"""
        if self.absolute_start is None:
            return None
        return self.absolute_start + pd.to_timedelta(float(t_sec), unit="s")

    def __len__(self):
        return len(self.t)

    def duration_sec(self):
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0

    def pos(self, t):
        import numpy as _np
        if len(self.t) == 1:
            return self.P[0].copy()
        tt = float(_np.clip(t, self.t[0], self.t[-1]))
        ex = _np.interp(tt, self.t, self.P[:, 0]); ny = _np.interp(tt, self.t, self.P[:, 1])
        return _np.array([ex, ny])

    def vel(self, t, dt=30.0):
        return (self.pos(t + dt) - self.pos(t - dt)) / (2.0 * dt)


def load_ship_track(track_csv: Optional[Path], lat0: float, lon0: float) -> Optional["ShipTrack"]:
    """读真实 AIS 母船航迹 CSV → ShipTrack(本地米)。灵活识别列:
      时间列 ∈ {time,timestamp,t,datetime,utc};纬度 ∈ {lat,latitude,y};经度 ∈ {lon,longitude,x}。
    缺文件/列 → None(调用方回退合成航迹)。时间转为相对首点的秒。"""
    if not track_csv or not Path(track_csv).is_file():
        return None
    try:
        df = pd.read_csv(track_csv)
        cols = {c.lower(): c for c in df.columns}
        tcol = next((cols[k] for k in ("time", "timestamp", "t", "datetime", "utc") if k in cols), None)
        ycol = next((cols[k] for k in ("lat", "latitude", "y") if k in cols), None)
        xcol = next((cols[k] for k in ("lon", "longitude", "x") if k in cols), None)
        if ycol is None or xcol is None:
            log.warning("航迹文件缺 lat/lon 列, 回退合成航迹。")
            return None
        df = df.dropna(subset=[ycol, xcol])
        absolute_start = None
        time_source = "assumed-1min"
        if tcol is not None:
            try:
                # 统一到 UTC，避免本地时区/夏令时造成 AIS—天气错配。
                ts = pd.to_datetime(df[tcol], utc=True, errors="raise")
                absolute_start = ts.iloc[0]
                time_source = f"column:{tcol}:utc"
                t_sec = (ts - absolute_start).dt.total_seconds().to_numpy(dtype=float)
            except Exception:
                t_sec = np.arange(len(df), dtype=float) * 60.0
                log.warning("航迹时间列 %s 无法解析，按 1min 等间隔处理；不可用于历史天气同步。", tcol)
        else:
            t_sec = np.arange(len(df), dtype=float) * 60.0      # 无时间列: 假设等间隔 1min
        order = np.argsort(t_sec)
        t_sec = t_sec[order]
        P = np.array([latlon_to_local_m(float(df[ycol].iloc[i]), float(df[xcol].iloc[i]), lat0, lon0)
                      for i in order])
        log.info("载入母船航迹 %s (%d 点, 时长 %.1f min)", Path(track_csv).name, len(t_sec), (t_sec[-1]-t_sec[0])/60.0)
        return ShipTrack(t_sec, P, absolute_start=absolute_start, time_source=time_source)
    except Exception as exc:
        log.warning("航迹读取失败(%s), 回退合成航迹。", type(exc).__name__)
        return None


def _normalize_weather_source(df: pd.DataFrame, source_name: str, keep: list[str]) -> pd.DataFrame:
    """Normalize one weather source to a unique, monotone UTC index.

    Time ambiguity is not repaired silently: unparsable timestamps, duplicate UTC instants and
    non-numeric required fields are surfaced to the caller.  Missing values remain missing so the
    mission-window audit can fail closed instead of replacing isolated observations with constants.
    """
    if "time" not in df.columns:
        raise ValueError(f"{source_name} 缺少 time 列。")
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="raise")
    missing = [c for c in keep if c not in out.columns]
    if missing:
        raise ValueError(f"{source_name} 缺少必要列: {missing}")
    out = out.set_index("time")[keep].sort_index()
    if out.index.has_duplicates:
        dup = out.index[out.index.duplicated()].unique()[:5]
        raise ValueError(f"{source_name} 存在重复 UTC 时间戳，例如 {list(map(str, dup))}")
    for col in keep:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_weather(wave_csv: Optional[Path] = None,
                 wind_csv: Optional[Path] = None) -> pd.DataFrame:
    """Load and align CMEMS wave and ERA5 wind data on a UTC time axis.

    Whole missing sources/columns may still use the historical mechanism-test placeholders, but
    isolated NaNs and wind/wave timestamp mismatches are never filled here.  Downstream mission
    construction selects only complete rows and records the number of discarded/misaligned rows.
    """
    frames = []
    supplied = {"wave": bool(wave_csv and wave_csv.is_file()),
                "wind": bool(wind_csv and wind_csv.is_file())}
    if supplied["wave"]:
        raw = pd.read_csv(wave_csv)
        w = _normalize_weather_source(raw, str(wave_csv),
                                      ["Hs_m", "wave_Tm_s", "wave_dir_deg"])
        frames.append(w)
        log.info("载入浪 %d 行 (%s, UTC)", len(w), wave_csv.name)
    if supplied["wind"]:
        raw = pd.read_csv(wind_csv)
        keep = ["wind10_ms", "wind100_ms", "wind_dir_from_deg"]
        # wind100 is optional in the routing model; retain the old placeholder only when the whole
        # column is absent, not for individual missing observations.
        if "wind100_ms" not in raw.columns:
            raw["wind100_ms"] = 9.0
            log.warning("天气缺整列 wind100_ms，机制兼容占位 9.0 m/s。")
        a = _normalize_weather_source(raw, str(wind_csv), keep)
        frames.append(a)
        log.info("载入风 %d 行 (%s, UTC)", len(a), wind_csv.name)
    if not frames:
        idx = pd.date_range("2025-06-01", periods=720, freq="h", tz="UTC")
        df = pd.DataFrame(index=idx)
        df["Hs_m"] = 0.5; df["wave_Tm_s"] = 2.1; df["wave_dir_deg"] = 200.0
        df["wind10_ms"] = 6.7; df["wind100_ms"] = 9.0; df["wind_dir_from_deg"] = 230.0
        df.attrs.update(weather_source_mode="all-placeholder",
                        weather_input_rows=0, weather_complete_rows=len(df),
                        weather_default_columns=list(df.columns))
        log.warning("无天气文件,使用占位天气(720h,Hs=0.5m,wind10=6.7m/s)。")
        return df

    df = pd.concat(frames, axis=1, join="outer").sort_index()
    defaults = []
    for col, val in [("wind10_ms", 6.7), ("wind100_ms", 9.0),
                     ("wind_dir_from_deg", 230.0), ("Hs_m", 0.5),
                     ("wave_Tm_s", 2.1), ("wave_dir_deg", 200.0)]:
        if col not in df.columns:
            df[col] = val
            defaults.append(col)
            log.warning("天气缺整列 %s,机制兼容占位 %.1f。", col, val)

    required = ["wind10_ms", "wind_dir_from_deg", "Hs_m", "wave_Tm_s", "wave_dir_deg"]
    complete = df[required].notna().all(axis=1)
    df.attrs.update(
        weather_source_mode=("wind+wave" if all(supplied.values()) else
                             ("wind-only" if supplied["wind"] else "wave-only")),
        weather_input_rows=int(len(df)),
        weather_complete_rows=int(complete.sum()),
        weather_incomplete_rows=int((~complete).sum()),
        weather_default_columns=defaults,
        weather_index_timezone=str(df.index.tz),
    )
    if not bool(complete.any()):
        raise ValueError("风浪合并后没有任何完整 UTC 时刻；请检查时区、时间戳和数据覆盖。")
    if bool((~complete).any()):
        log.warning("风浪 UTC 合并后 %d/%d 行不完整；不会静默填值，任务窗口仅使用完整行。",
                    int((~complete).sum()), len(df))
    return df


# =============================================================================
# 3. ξ 误差模糊集 (model.md §4) —— 由 xi_moments_caseB.csv 的矩信息构造
# =============================================================================
@dataclass
class XiCell:
    """单个 (h,c) 分组的模糊集矩信息(2D,单位米;east/north)。"""
    h_min: int
    c_state: str
    n: int
    mu: np.ndarray            # (2,) 均值 [mu_e, mu_n]
    Sigma: np.ndarray         # (2,2) 协方差
    support_radius: float     # 支持集半径(取 max_norm),∞-型约束用
    p95_norm: float
    rms_norm: float
    state_change_rate: float = float("nan")
    actual_recovery_state_mode: str = "unknown"
    launch_speed_p50_ms: float = float("nan")
    launch_speed_p95_ms: float = float("nan")

    def mean_norm(self) -> float:
        return float(np.linalg.norm(self.mu))


def validate_xi_ambiguity_math(xi_amb) -> None:
    """Validate the mathematical Xi object consumed by a certified solver.

    This is deliberately independent of CSV provenance: callers may construct
    ``XiAmbiguity`` in memory, but a global certificate must never be issued for
    a non-covariance matrix or a key/cell mismatch.
    """
    cells = getattr(xi_amb, "cells", None)
    if not isinstance(cells, dict):
        raise ValueError("xi_amb.cells 必须是 dict。")
    horizons = getattr(xi_amb, "horizons", None)
    if not isinstance(horizons, (list, tuple)):
        raise ValueError("xi_amb.horizons 必须是有序 horizon 序列。")
    h_list = [float(h) for h in horizons]
    if not all(math.isfinite(h) for h in h_list):
        raise ValueError("xi_amb.horizons 必须全部有限。")
    if len({h.hex() for h in h_list}) != len(h_list):
        raise ValueError("xi_amb.horizons 存在重复 binary64 horizon。")
    key_h = {float(k[0]).hex() for k in cells if isinstance(k, tuple) and len(k) == 2}
    if key_h != {h.hex() for h in h_list}:
        raise ValueError("xi_amb.horizons 与实际 Xi cell horizon 集合不一致。")
    for key, cell in cells.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(f"非法 Xi cell key: {key!r}")
        h_key = float(key[0])
        h_cell = float(getattr(cell, "h_min", float("nan")))
        if (not math.isfinite(h_key) or not math.isfinite(h_cell)
                or h_key.hex() != h_cell.hex()):
            raise ValueError(f"Xi key/cell horizon 不一致: key={h_key!r}, cell={h_cell!r}")
        if str(key[1]) != str(getattr(cell, "c_state", "")):
            raise ValueError(f"Xi key/cell state 不一致: key={key[1]!r}, cell={getattr(cell, 'c_state', None)!r}")
        mu = np.asarray(getattr(cell, "mu", None), dtype=float)
        Sigma = np.asarray(getattr(cell, "Sigma", None), dtype=float)
        if mu.shape != (2,) or not bool(np.isfinite(mu).all()):
            raise ValueError(f"Xi cell {key!r} 的 mu 必须是有限二维向量。")
        if Sigma.shape != (2, 2) or not bool(np.isfinite(Sigma).all()):
            raise ValueError(f"Xi cell {key!r} 的 Sigma 必须是有限 2x2 矩阵。")
        if float(Sigma[0, 1]).hex() != float(Sigma[1, 0]).hex():
            raise ValueError(f"Xi cell {key!r} 的 covariance 必须按模型语义对称。")
        if not _binary64_psd_cov2(float(Sigma[0, 0]), float(Sigma[0, 1]), float(Sigma[1, 1])):
            raise ValueError(f"Xi cell {key!r} 的 covariance 不是 binary64-as-real PSD。")
        radius = float(getattr(cell, "support_radius", float("nan")))
        if not math.isfinite(radius) or radius < 0.0:
            raise ValueError(f"Xi cell {key!r} 的 support_radius 必须非负且有限。")
        n = getattr(cell, "n", None)
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 0:
            raise ValueError(f"Xi cell {key!r} 的 n 必须是非负整数。")


class XiAmbiguity:
    """ξ_{h,c} 模糊集集合。键 = (h_min, c_state)。

    提供给 step10/step11/step12 的接口:
      get(h, c) -> XiCell                      严格命中已观测 h 与指定状态
      second_moment(h,c) -> E[ξξ^T] = Σ + μμ^T 矩 DRO 常用
    模糊集本身(矩信息 μ,Σ,支持集)即 model.md §4 所述,不假设分布族。
    """
    def __init__(self, cells: dict[tuple[int, str], XiCell], horizons: list[int]):
        self.cells = cells
        self.horizons = sorted(horizons)

    @classmethod
    def from_csv(cls, path: Path, mmsi: str = "ALL", formal: bool = False) -> "XiAmbiguity":
        df = pd.read_csv(path)
        df = df[df["mmsi"].astype(str) == str(mmsi)].copy()
        if df.empty:
            raise ValueError(f"{path.name} 中无 mmsi={mmsi} 的行(可选 ALL 或具体 MMSI)。")
        if formal:
            required = {"predictor", "predictor_contract", "timestamp_epoch_contract",
                        "moments_source", "valid_for_formal", "purge_min",
                        "min_cell_n", "sample_rule", "source_states", "n_effective",
                        "sample_overlap_policy", "state_merge_policy", "t0_min_iso", "t0_max_iso"}
            missing = sorted(required - set(df.columns))
            if missing:
                raise ValueError(f"正式 ξ 矩文件缺数据契约列: {missing}。请用阶段3 step7重新生成。")
            valid = df["valid_for_formal"].astype(str).str.lower().isin({"true", "1", "yes"})
            if not bool(valid.all()):
                raise ValueError("正式 ξ 矩文件包含 valid_for_formal=False 的统计格。")
            if set(df["predictor"].astype(str)) != {"cv_noleak"}:
                raise ValueError("正式 ξ 矩必须使用 predictor=cv_noleak。")
            if set(df["moments_source"].astype(str)) != {"train"}:
                raise ValueError("正式 ξ 矩只能由 purged train 样本估计。")
            if set(df["sample_overlap_policy"].astype(str)) != {"nonoverlap"}:
                raise ValueError("正式 ξ 矩必须使用 sample_overlap_policy=nonoverlap。")
            if not bool((pd.to_numeric(df["n_effective"], errors="coerce")
                         == pd.to_numeric(df["n"], errors="coerce")).all()):
                raise ValueError("正式 ξ 矩的 n 必须等于非重叠有效样本数 n_effective。")
            if set(df["predictor_contract"].astype(str)) != {XI_PREDICTOR_CONTRACTS["cv_noleak"]}:
                raise ValueError("正式 ξ 矩 predictor_contract 不是当前 epoch-seconds 修正版。")
            if set(df["timestamp_epoch_contract"].astype(str)) != {XI_TIMESTAMP_EPOCH_CONTRACT}:
                raise ValueError("正式 ξ 矩时间戳合同不是 UTC 纳秒归一化 epoch-seconds 合同。")
            purge_vals = pd.to_numeric(df["purge_min"], errors="coerce")
            if not bool(np.isfinite(purge_vals.to_numpy(float)).all()) or not bool((purge_vals > 0).all()):
                raise ValueError("正式 ξ 矩必须记录正的 purge_min。")
            max_h_file = float(pd.to_numeric(df["h_min"], errors="coerce").max())
            if not bool((purge_vals >= max_h_file).all()):
                raise ValueError("正式 ξ 矩要求 purge_min 至少覆盖文件中的最大预测 horizon。")

            numeric_cols = ["h_min", "n", "n_effective", "min_cell_n", "mu_e_m", "mu_n_m",
                            "sigma_ee", "sigma_en", "sigma_nn", "max_norm_m",
                            "p95_norm_m", "rms_norm_m"]
            num = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
            if not bool(np.isfinite(num.to_numpy(float)).all()):
                raise ValueError("正式 ξ 矩包含缺失或非有限数值。")
            if not bool((num["n"] >= num["min_cell_n"]).all()):
                raise ValueError("正式 ξ 矩存在未达到 min_cell_n 的统计格。")
            if bool((num[["n", "min_cell_n", "sigma_ee", "sigma_nn", "max_norm_m",
                          "p95_norm_m", "rms_norm_m"]] < 0).any().any()):
                raise ValueError("正式 ξ 矩包含负样本数、负方差或负半径。")

            expected_h = list(XI_FORMAL_HORIZON_GRID_MIN)
            expected_h_hex = {float(h).hex(): int(h) for h in expected_h}
            raw_h = [float(x) for x in num["h_min"].to_numpy(float)]
            bad_h = [x for x in raw_h if x.hex() not in expected_h_hex]
            if bad_h:
                sample = [f"{x!r} ({x.hex()})" for x in bad_h[:5]]
                raise ValueError(
                    "正式 ξ h_min 必须逐 binary64 精确命中 5,10,...,60 分钟网格；"
                    f"禁止 round/int/nearest 吸附。非法示例={sample}")
            canonical_h = [expected_h_hex[x.hex()] for x in raw_h]
            hvals = sorted(set(canonical_h))
            if not hvals:
                raise ValueError("正式 ξ 时长集合不能为空。")

            for name in ("n", "n_effective", "min_cell_n"):
                vals = [float(x) for x in num[name].to_numpy(float)]
                if any((not v.is_integer()) for v in vals):
                    raise ValueError(f"正式 ξ {name} 必须是精确整数，禁止 float→int 截断。")

            keys = list(zip(canonical_h, df["c_state"].astype(str)))
            if len(keys) != len(set(keys)):
                raise ValueError("正式 ξ 矩存在重复 (h,c) 键；禁止静默覆盖。")
            allowed_states = {"直航", "转弯", "低速", "动力定位"}
            bad_states = sorted(set(df["c_state"].astype(str)) - allowed_states)
            if bad_states:
                raise ValueError(f"正式 ξ 矩包含未知分类状态: {bad_states}")

            for idx, r in df.iterrows():
                state = str(r["c_state"])
                rule = str(r["sample_rule"])
                policy = str(r["state_merge_policy"])
                sources = {x for x in str(r["source_states"]).split("+") if x}
                if policy not in {"none", "low_speed_pair"}:
                    raise ValueError(f"行 {idx}: 非法 state_merge_policy={policy!r}")
                if rule == "raw_state":
                    if sources != {state}:
                        raise ValueError(f"行 {idx}: raw_state 必须只使用自身状态，得到 {sources}")
                elif rule == "merged_low_speed_pair":
                    if (policy != "low_speed_pair" or state not in {"低速", "动力定位"}
                            or sources != {"低速", "动力定位"}):
                        raise ValueError(f"行 {idx}: 非法低速/动力定位合并声明。")
                else:
                    raise ValueError(f"行 {idx}: 非法 sample_rule={rule!r}")
                if state in {"直航", "转弯"} and rule != "raw_state":
                    raise ValueError(f"行 {idx}: 直航/转弯禁止合并其他状态。")
                see = float(r["sigma_ee"]); sen = float(r["sigma_en"]); snn = float(r["sigma_nn"])
                if not _binary64_psd_cov2(see, sen, snn):
                    det_exact = (Fraction.from_float(see) * Fraction.from_float(snn)
                                 - Fraction.from_float(sen) * Fraction.from_float(sen))
                    raise ValueError(
                        f"行 {idx}: ξ 协方差不是 binary64-as-real 半正定矩阵；"
                        f"exact_det={det_exact}")
                t0a = pd.to_datetime(r["t0_min_iso"], errors="coerce", utc=True)
                t0b = pd.to_datetime(r["t0_max_iso"], errors="coerce", utc=True)
                if pd.isna(t0a) or pd.isna(t0b) or t0a > t0b:
                    raise ValueError(f"行 {idx}: 非法统计时间范围。")
        cells = {}
        _formal_h_map = ({float(h).hex(): int(h) for h in range(5, 61, 5)}
                         if formal else None)
        for _, r in df.iterrows():
            h_raw = float(r["h_min"])
            h = (_formal_h_map[h_raw.hex()] if formal else int(h_raw))
            c = str(r["c_state"])
            mu = np.array([r["mu_e_m"], r["mu_n_m"]], float)
            Sigma = np.array([[r["sigma_ee"], r["sigma_en"]],
                              [r["sigma_en"], r["sigma_nn"]]], float)
            cells[(h, c)] = XiCell(
                h_min=h, c_state=c, n=int(r["n"]), mu=mu, Sigma=Sigma,
                support_radius=float(r["max_norm_m"]),
                p95_norm=float(r["p95_norm_m"]), rms_norm=float(r["rms_norm_m"]),
                state_change_rate=float(r.get("launch_to_recovery_state_change_rate", float("nan"))),
                actual_recovery_state_mode=str(r.get("actual_recovery_state_mode", "unknown")),
                launch_speed_p50_ms=float(r.get("launch_speed_p50_ms", float("nan"))),
                launch_speed_p95_ms=float(r.get("launch_speed_p95_ms", float("nan"))))
        horizons = sorted({h for (h, _) in cells})
        if str(mmsi).upper() == "ALL" and not formal:
            log.info(
                "载入 ξ 模糊集 bootstrap: %d 个 (h,c) 格 (mmsi=ALL, horizons=%s); "
                "formal step13 若提供 train samples 会在构造任务实例前以具体 MMSI 重新估计并覆盖。",
                len(cells), horizons)
        else:
            log.info("载入 ξ 模糊集: %d 个 (h,c) 格 (mmsi=%s, horizons=%s)",
                     len(cells), mmsi, horizons)
        obj = cls(cells, horizons)
        obj.formal_validated = bool(formal)
        obj.moments_source = ("train" if formal else None)
        obj.source_path = str(path)
        obj.selected_mmsi = str(mmsi)
        predictors = sorted({str(x).strip() for x in df["predictor"].dropna()}) if "predictor" in df.columns else []
        predictor_contracts = sorted({str(x).strip() for x in df["predictor_contract"].dropna()}) if "predictor_contract" in df.columns else []
        if len(predictors) > 1:
            raise ValueError(f"{path.name} 的 mmsi={mmsi} 混有多个 predictor: {predictors}")
        if len(predictor_contracts) > 1:
            raise ValueError(f"{path.name} 的 mmsi={mmsi} 混有多个 predictor_contract: {predictor_contracts}")
        obj.predictor = predictors[0] if predictors else "unknown"
        obj.predictor_contract = predictor_contracts[0] if predictor_contracts else "unknown"
        obj.timestamp_epoch_contract = (str(df["timestamp_epoch_contract"].iloc[0])
                                        if "timestamp_epoch_contract" in df.columns else "unknown")
        obj.sample_overlap_policy = (str(df["sample_overlap_policy"].iloc[0])
                                     if "sample_overlap_policy" in df.columns else "unknown")
        obj.formal_horizon_grid_contract = XI_FORMAL_GRID_CONTRACT if formal else "legacy-nonformal"
        obj.covariance_contract = XI_COVARIANCE_CONTRACT if formal else "legacy-nonformal"
        obj.valid_for_formal_data = (bool(df["valid_for_formal"].astype(str).str.lower()
                                          .isin({"true", "1", "yes"}).all())
                                     if "valid_for_formal" in df.columns else False)
        obj.available_columns = tuple(str(c) for c in df.columns)
        return obj

    @classmethod
    def from_csv_hierarchical(cls, path: Path, mmsi: str, formal: bool = False,
                              shrinkage_equivalent_n: float = 100.0) -> "XiAmbiguity":
        """优先使用当前母船的 ξ 矩，并向 ``ALL`` 总体做层级收缩。

        旧实验无条件使用 ``mmsi=ALL``，会把不同船舶操纵特性混成一个分布。这里对每个
        ``(h,c)`` 使用当前 MMSI 的均值/协方差；有限样本时按
        ``lambda=n/(n+shrinkage_equivalent_n)`` 向 ALL 收缩。若具体格缺失，机制模式允许
        明确标记的 ALL 回退，正式模式则 fail-closed。
        """
        all_obj = cls.from_csv(path, mmsi="ALL", formal=formal)
        try:
            exact_obj = cls.from_csv(path, mmsi=str(mmsi), formal=formal)
        except ValueError:
            if formal:
                raise
            all_obj.selected_mmsi = "ALL"
            all_obj.requested_mmsi = str(mmsi)
            all_obj.hierarchical_sources = {k: "ALL-fallback-no-exact-mmsi" for k in all_obj.cells}
            log.warning("ξ 文件无 mmsi=%s，机制模式回退 ALL；正式模式会拒绝。", mmsi)
            return all_obj

        cells = {}
        sources = {}
        keys = sorted(set(all_obj.cells) | set(exact_obj.cells))
        for key in keys:
            ec = exact_obj.cells.get(key)
            ac = all_obj.cells.get(key)
            if ec is None:
                if formal:
                    raise ValueError(f"正式 ξ 缺少当前 MMSI={mmsi} 的统计格 {key}。")
                cells[key] = ac
                sources[key] = "ALL-fallback-missing-exact-cell"
                continue
            if ac is None:
                cells[key] = ec
                sources[key] = "exact-only"
                continue
            lam = float(ec.n) / (float(ec.n) + max(float(shrinkage_equivalent_n), 1e-9))
            mu = lam * ec.mu + (1.0 - lam) * ac.mu
            delta = np.asarray(ec.mu - ac.mu, float)
            Sigma = (lam * ec.Sigma + (1.0 - lam) * ac.Sigma
                     + lam * (1.0 - lam) * np.outer(delta, delta))
            Sigma = _canonicalize_binary64_psd_cov2(Sigma)
            cells[key] = XiCell(
                h_min=int(ec.h_min), c_state=str(ec.c_state), n=int(ec.n),
                mu=np.asarray(mu, float), Sigma=np.asarray(Sigma, float),
                support_radius=float(max(ec.support_radius, ac.support_radius)),
                p95_norm=float(max(ec.p95_norm, ac.p95_norm)),
                rms_norm=float(math.sqrt(max(lam * ec.rms_norm ** 2
                                             + (1.0 - lam) * ac.rms_norm ** 2, 0.0))),
                state_change_rate=float(lam * ec.state_change_rate + (1.0 - lam) * ac.state_change_rate)
                    if math.isfinite(ec.state_change_rate) and math.isfinite(ac.state_change_rate) else float("nan"),
                actual_recovery_state_mode=str(ec.actual_recovery_state_mode),
                launch_speed_p50_ms=float(ec.launch_speed_p50_ms),
                launch_speed_p95_ms=float(ec.launch_speed_p95_ms))
            sources[key] = f"exact-shrunk-to-ALL(lambda={lam:.6f})"
        obj = cls(cells, sorted({h for h, _ in cells}))
        obj.formal_validated = bool(formal)
        obj.moments_source = getattr(exact_obj, "moments_source", None)
        obj.source_path = str(path)
        obj.selected_mmsi = str(mmsi)
        obj.requested_mmsi = str(mmsi)
        exact_predictor = str(getattr(exact_obj, "predictor", "unknown"))
        all_predictor = str(getattr(all_obj, "predictor", "unknown"))
        exact_contract = str(getattr(exact_obj, "predictor_contract", "unknown"))
        all_contract = str(getattr(all_obj, "predictor_contract", "unknown"))
        if exact_predictor != all_predictor:
            raise ValueError(f"具体 MMSI 与 ALL 的 predictor 不一致: {exact_predictor!r} != {all_predictor!r}")
        if exact_contract != all_contract:
            raise ValueError(f"具体 MMSI 与 ALL 的 predictor_contract 不一致: {exact_contract!r} != {all_contract!r}")
        obj.predictor = exact_predictor
        obj.predictor_contract = exact_contract
        exact_epoch = str(getattr(exact_obj, "timestamp_epoch_contract", "unknown"))
        all_epoch = str(getattr(all_obj, "timestamp_epoch_contract", "unknown"))
        if exact_epoch != all_epoch:
            raise ValueError(f"具体 MMSI 与 ALL 的 timestamp_epoch_contract 不一致: {exact_epoch!r} != {all_epoch!r}")
        obj.timestamp_epoch_contract = exact_epoch
        obj.sample_overlap_policy = str(getattr(exact_obj, "sample_overlap_policy", "unknown"))
        obj.valid_for_formal_data = bool(getattr(exact_obj, "valid_for_formal_data", False)
                                         and getattr(all_obj, "valid_for_formal_data", False))
        obj.hierarchical_sources = sources
        obj.shrinkage_equivalent_n = float(shrinkage_equivalent_n)
        log.info("载入分层 ξ 模糊集: mmsi=%s, %d 格；具体船矩向 ALL 收缩等效样本=%.0f。",
                 mmsi, len(cells), shrinkage_equivalent_n)
        return obj

    def nearest_h(self, h_min: float) -> int:
        """旧接口名保留，但正式语义只允许精确锚点，禁止最近时长吸附。"""
        hq = float(h_min)
        hit = next((int(h) for h in self.horizons
                    if float(h).hex() == hq.hex()), None)
        if hit is None:
            raise KeyError(f"h={hq:g} 不是已观测统计锚点；nearest_h 的吸附语义已禁用。")
        return hit

    def get(self, h_min: float, c_state: str) -> XiCell:
        """严格取得一个已观测的 ``(h,c)`` 统计格。

        正式模型禁止把请求 horizon 静默吸附到最近格，也禁止在状态缺失时借用其他状态。
        非锚点查询必须显式调用 :meth:`get_interp`，且仅允许在同一状态的相邻锚点之间插值。
        """
        hq = float(h_min)
        hit = next((int(h) for h in self.horizons
                    if float(h).hex() == hq.hex()), None)
        if hit is None:
            raise KeyError(f"h={hq:g} 不是已观测统计锚点；请在支持区间内使用 get_interp。")
        key = (hit, str(c_state))
        if key not in self.cells:
            raise KeyError(f"缺少统计格 (h={hit}, c={c_state})；禁止跨状态回退。")
        return self.cells[key]

    def second_moment(self, h_min: float, c_state: str) -> np.ndarray:
        cell = self.get(h_min, c_state)
        return cell.Sigma + np.outer(cell.mu, cell.mu)

    # -------------------------------------------------------------------------
    # 双层 h 网格：统计层只在数据声明的 horizon 上估矩；决策层仅可在同一状态的
    # 相邻锚点之间插值。禁止区间外外推，禁止跨状态借格。
    # -------------------------------------------------------------------------
    def get_interp(self, h_query: float, c_state: str) -> XiCell:
        """在统计支持区间内严格查询/插值一个 ``(h,c)`` 矩信息格。

        - 命中锚点：要求该状态格存在；
        - 位于两个锚点之间：要求同一状态在上下锚点均存在；
        - 超出全局统计闭区间、或任一同状态锚点缺失：直接拒绝。
        """
        H = [float(h) for h in self.horizons]
        if not H:
            raise KeyError("模糊集无任何 horizon。")
        hq = float(h_query)
        if not math.isfinite(hq):
            raise ValueError("h_query 必须为有限数。")
        if hq < H[0] or hq > H[-1]:
            raise KeyError(f"h={hq:g} 超出统计支持区间 [{H[0]:g},{H[-1]:g}]；禁止外推。")

        for h in H:
            if float(h).hex() == hq.hex():
                return self.get(h, c_state)

        below = [h for h in H if h < hq]
        above = [h for h in H if h > hq]
        if not below or not above:
            raise KeyError(f"h={hq:g} 无相邻统计锚点；禁止外推。")
        h_lo, h_hi = max(below), min(above)
        c_lo = self.get(h_lo, c_state)
        c_hi = self.get(h_hi, c_state)

        w = (hq - h_lo) / (h_hi - h_lo)
        mu = (1.0 - w) * c_lo.mu + w * c_hi.mu
        # PSD、端点连续，并在下锚点侧保留方差随预测跨度增长的保守项。
        s_lo = hq / h_lo if h_lo > 0 else 1.0
        Sigma = (1.0 - w) * (c_lo.Sigma * s_lo) + w * c_hi.Sigma
        Sigma = _canonicalize_binary64_psd_cov2(Sigma)
        rad = (1.0 - w) * c_lo.support_radius + w * c_hi.support_radius
        p95 = (1.0 - w) * c_lo.p95_norm + w * c_hi.p95_norm
        rms = (1.0 - w) * c_lo.rms_norm + w * c_hi.rms_norm
        n_eff = min(c_lo.n, c_hi.n)
        return XiCell(h_min=hq, c_state=str(c_state), n=n_eff,
                      mu=mu, Sigma=Sigma, support_radius=rad,
                      p95_norm=p95, rms_norm=rms,
                      state_change_rate=((1.0 - w) * c_lo.state_change_rate + w * c_hi.state_change_rate
                                         if math.isfinite(c_lo.state_change_rate) and math.isfinite(c_hi.state_change_rate)
                                         else float("nan")),
                      actual_recovery_state_mode=str(c_lo.actual_recovery_state_mode),
                      launch_speed_p50_ms=((1.0 - w) * c_lo.launch_speed_p50_ms + w * c_hi.launch_speed_p50_ms
                                           if math.isfinite(c_lo.launch_speed_p50_ms) and math.isfinite(c_hi.launch_speed_p50_ms)
                                           else float("nan")),
                      launch_speed_p95_ms=((1.0 - w) * c_lo.launch_speed_p95_ms + w * c_hi.launch_speed_p95_ms
                                           if math.isfinite(c_lo.launch_speed_p95_ms) and math.isfinite(c_hi.launch_speed_p95_ms)
                                           else float("nan")))


# =============================================================================
# 4. 【LEGACY / 仅对照 + 自检】单架次(单风机)结构与能耗/时间 (model.md §16 对照)
#    ───────────────────────────────────────────────────────────────────────
#    legacy single-turbine sortie utilities; retained ONLY for the nominal/single-stop
#    comparison (model.md §16) and this file's self-test —— **NOT the full model**.
#    本项目唯一正式模型 = 多风机时空航路(step10+: 起飞时刻 τ + 序列 ω + 回收时长 h + ξ + 风浪 + 着舰门 + 等待)。
#    Sortie / sortie_distances / energy_time / nominal_feasible 不被 step10–12 主模型流程调用。
# =============================================================================
@dataclass
class Sortie:
    """[LEGACY/对照] 单架次 r=(k, t_L, i, t_R)。仅供 model.md §16 名义/单停靠对照与自检, 非正式模型。"""
    rid: int
    turbine: Turbine
    t_L: pd.Timestamp          # 起飞时刻
    t_R: pd.Timestamp          # 计划回收时刻
    t0: pd.Timestamp           # 决策时刻 (用于 h(r)=t_R - t0)
    P_launch: np.ndarray       # (2,) 起飞点本地米 = \hat P_v(t_L)
    P_recover_pred: np.ndarray # (2,) 预测回收点本地米 = \hat P_v(t_R)
    c_state: str               # 回收时船舶状态分组 c(r)

    def h_min(self) -> float:
        return (self.t_R - self.t0).total_seconds() / 60.0

    def planned_flight_s(self) -> float:
        return (self.t_R - self.t_L).total_seconds()


def insp_vertical_span(tb: Turbine, z_cruise: float) -> float:
    """巡检竖向跨度 Δz^insp:从巡航高度爬到叶尖并覆盖塔身。model.md §6。
    取 (H_tip - z_cruise) 的非负值作为竖向爬升量(覆盖塔身的悬停在 τ 内)。"""
    return max(tb.H_tip - z_cruise, 0.0)


def to_land_energy_time(p: Params) -> tuple[float, float, float, float]:
    """返回 ``(E_to, E_land_legacy, T_to, T_land_legacy)``。

    正式口径中，本函数只计从甲板爬升至巡航高度的起飞阶段。最终下降、接地与可能复飞
    全部包含在 :func:`dock_reserve` 中，因此后两个兼容返回值恒为 0。保留四元组签名是为了
    让现有 step10--step12 在第一阶段即可自动停止重复计量，而不提前改动其接口。
    """
    T_to = p.z_cruise / p.v_z
    if getattr(p, "use_zeng", False):
        P_hover0 = P_zeng(0.0, p)
        P_up = P_hover0 + 7.27 * 9.81 * p.v_z
        E_to = P_up * T_to / 3600.0
    else:
        E_to = p.P_climb * T_to / 3600.0
    return float(E_to), 0.0, float(T_to), 0.0


def sortie_distances(s: Sortie, xi: np.ndarray) -> tuple[float, float]:
    """外飞/返程水平距离 (m)。返程随 ξ。model.md §6。"""
    q = latlon_to_local_m_pair(s.turbine.lonlat, s._lat0, s._lon0) \
        if hasattr(s, "_lat0") else s.turbine_local
    d_out = float(np.linalg.norm(s.P_launch - s.turbine_local))
    P_recover = s.P_recover_pred + xi   # model.md §5
    d_ret = float(np.linalg.norm(s.turbine_local - P_recover))
    return d_out, d_ret


def energy_time(s: Sortie, xi: np.ndarray, p: Params, wx: dict) -> tuple[float, float]:
    """LEGACY 单停靠对照的完整计划能耗/时间。

    起飞爬升独立计量；最终目标锁定、对准、下降、接地和复飞储备统一由
    :func:`dock_reserve` 计量，避免 ``E_land/T_land`` 与 ``E_dock/t_dock`` 重复。
    """
    d_out, d_ret = sortie_distances(s, xi)
    w10 = wx["wind10"]; wdir = wx["wind_dir_from"]
    if w10 is None or (isinstance(w10, float) and math.isnan(w10)):
        w10 = 6.7
    if wdir is None or (isinstance(wdir, float) and math.isnan(wdir)):
        wdir = 230.0
    w_cruise = wind_at_height(w10, p.z_cruise, p.z0)
    hd_out = heading_deg(s.P_launch, s.turbine_local)
    P_recover = s.P_recover_pred + xi
    hd_ret = heading_deg(s.turbine.local if hasattr(s.turbine, "local") else s.turbine_local, P_recover)
    vg_out = ground_speed(p.v_cr, w_cruise, wdir, hd_out)
    vg_ret = ground_speed(p.v_cr, w_cruise, wdir, hd_ret)
    E_to, _E_land, T_to, _T_land = to_land_energy_time(p)
    dz = insp_vertical_span(s.turbine, p.z_cruise)
    if getattr(p, "use_zeng", False):
        P_up = P_zeng(0.0, p) + 7.27 * 9.81 * p.v_z
        E_insp = P_up * dz / p.v_z / 3600.0 + P_zeng(p.v_orbit, p) * p.tau_insp / 3600.0
        E_out = leg_power(p, p.v_cr) * (d_out / vg_out) / 3600.0
        E_ret = leg_power(p, p.v_cr) * (d_ret / vg_ret) / 3600.0
    else:
        E_insp = p.P_climb * dz / p.v_z / 3600.0 + p.P_hov * p.tau_insp / 3600.0
        E_out = p.P_cr * (d_out / vg_out) / 3600.0
        E_ret = p.P_cr * (d_ret / vg_ret) / 3600.0
    motion = deck_motion(float(wx.get("Hs", 0.5)), float(wx.get("Tp", 2.1)),
                         float(wx.get("wave_dir", 200.0)) - float(wx.get("ship_heading", 0.0)), p)
    t_dock, E_dock = dock_reserve(p, motion, float(w10))
    E = E_to + E_out + E_insp + E_ret + E_dock
    T = T_to + d_out / vg_out + p.tau_insp + d_ret / vg_ret + t_dock
    return float(E), float(T)


# helper used above (kept separate for clarity)
def latlon_to_local_m_pair(lonlat, lat0, lon0):
    return latlon_to_local_m(lonlat[1], lonlat[0], lat0, lon0)


# =============================================================================
# 5. 简化 6DOF 甲板运动门 (model.md §7)
# =============================================================================
def deck_motion(Hs: float, Tp: float, rel_dir_deg: float, p: Params) -> dict:
    """海况 -> 简化垂荡/横摇/纵摇 (model.md §7, 经验传递, SCN)。
    rel_dir_deg = 浪向 - 船艏向(横浪 ~90° 横摇大,迎浪 ~0/180° 纵摇大)。"""
    beam = abs(math.sin(math.radians(rel_dir_deg)))   # 横浪权重
    head = abs(math.cos(math.radians(rel_dir_deg)))   # 迎浪权重
    s_heave = p.a_heave * Hs
    s_roll = p.a_roll * Hs * beam
    s_pitch = p.a_pitch * Hs * head
    return dict(heave=s_heave, roll=s_roll, pitch=s_pitch)


def landing_gate(Hs: float, Tp: float, wave_dir: float, ship_heading: float,
                 w10: float, p: Params) -> tuple[int, float]:
    """着舰门 L_r ∈ {0,1} 与有效捕获半径 r_cap_eff。model.md §7。
    返回 (L, r_cap_eff)。L=0 表示该回收时刻不可着舰。"""
    m = deck_motion(Hs, Tp, wave_dir - ship_heading, p)
    L = int(m["heave"] <= p.s_heave_max and
            m["roll"] <= p.s_roll_max and
            m["pitch"] <= p.s_pitch_max and
            w10 <= p.w_land_max and
            Hs <= p.Hs_op)
    r_eff = p.r_cap - p.kappa_h * m["heave"] - p.kappa_rp * (m["roll"] + m["pitch"])
    return L, max(r_eff, 0.0)


def dock_reserve(p: Params, motion: dict, w10_low: float) -> tuple[float, float]:
    r"""完整末端阶段储备 ``(t_dock_s, E_dock_Wh)``。

    该阶段从近端目标锁定开始，包含船尾对准/伴飞末段、等待合适甲板相位、从巡航高度
    完成最终下降、接地以及一次复飞机动储备。它是最终下降的唯一计量位置。

    ``t_dock = t_base(1+gamma*S)``，其中 ``S`` 是甲板运动最大占限比。能量将总时间拆成
    对准/等待段与最终下降段：前者采用风下侧对应的驻位功率，后者采用悬停基线功率；
    两段统一乘控制开销 ``dock_beta``。
    """
    S = max(motion["heave"] / max(p.s_heave_max, 1e-9),
            motion["roll"] / max(p.s_roll_max, 1e-9),
            motion["pitch"] / max(p.s_pitch_max, 1e-9))
    S = min(max(S, 0.0), 1.0)
    t_dock = p.t_dock_base_s * (1.0 + p.dock_gamma * S)
    t_descent = min(p.z_cruise / max(p.v_z, 1e-9), t_dock)
    t_guidance = max(t_dock - t_descent, 0.0)
    if getattr(p, "use_zeng", False):
        P_guidance = P_zeng(max(w10_low, 0.0), p)
        P_descent = P_zeng(0.0, p)
    else:
        P_guidance = p.P_wait
        P_descent = p.P_hov
    E_dock = p.dock_beta * (P_guidance * t_guidance + P_descent * t_descent) / 3600.0
    return float(t_dock), float(E_dock)


def max_flight_radius_m(p: Params, h_max_min: float, dz_insp_m: float = 55.0,
                        s_motion: float = 0.5) -> dict:
    r"""【更新 问题1】无人机最大作业半径(m), 由该 UAV 档位的物理参数推导 —— 取代
    此前无依据的 pair_radius=8km 常数(它决定实验窗起点、航迹评分与 reach 可达集)。

    语义 = 【静态对称诊断半径】: 单台最小架次(出航 d → 巡检 1 台 → 对称返航 d)在名义物理下
    "能量与时间预算都装得下"的最远 d。移动母船可能显著缩短真实返航腿，因此该量不能证明
    正式模型的可达集外包络；v17 formal Step13 禁止用它提前剪风机，只保留日志/诊断用途。

        R_energy = E_avail / (2·e_m),  e_m = P_zeng(v_cr)/v_cr/3600  (Wh/m, 巡航单程每米)
        E_avail  = B_use − E_to − E_insp(1台) − E_dock_env
        R_time   = v_cr · T_avail / 2
        T_avail  = 60·h_max − T_to − (τ_insp + Δz/v_z) − t_dock_env
        R_max    = min(R_energy, R_time)

    E_dock_env/t_dock_env 取包络值: 甲板运动占限比 s_motion=0.5(中位偏上)、驻停功率取
    静风上界 P_zeng(0)(能量最坏面=风小, 见 dock_reserve)。dz_insp_m 默认 55 = Rødsand II
    (H_tip 115 − z_cruise 60); 换场请传对应 Δz。
    返回 dict(R_max_m, R_energy_m, R_time_m, 及全部中间量) 供日志/复核。"""
    E_to, _E_land, T_to, _T_land = to_land_energy_time(p)
    # 单台巡检(竖向爬升 + 绕飞悬停), 与 route_nominal_ET 同一公式
    P_up = P_zeng(0.0, p) + 7.27 * 9.81 * p.v_z
    E_insp1 = P_up * dz_insp_m / p.v_z / 3600.0 + P_zeng(p.v_orbit, p) * p.tau_insp / 3600.0
    T_insp1 = dz_insp_m / p.v_z + p.tau_insp
    # 对接包络与正式 dock_reserve 同一分段口径：最终下降只在此处计量。
    s_env = min(max(float(s_motion), 0.0), 1.0)
    motion_env = dict(heave=s_env * p.s_heave_max, roll=0.0, pitch=0.0)
    t_dock_env, E_dock_env = dock_reserve(p, motion_env, 0.0)
    E_avail = p.B_use - E_to - E_insp1 - E_dock_env
    e_per_m = P_zeng(p.v_cr, p) / max(p.v_cr, 1e-9) / 3600.0
    R_energy = max(E_avail, 0.0) / (2.0 * e_per_m)
    T_avail = 60.0 * float(h_max_min) - T_to - T_insp1 - t_dock_env
    R_time = max(T_avail, 0.0) * p.v_cr / 2.0
    return dict(R_max_m=float(min(R_energy, R_time)),
                R_energy_m=float(R_energy), R_time_m=float(R_time),
                E_avail_Wh=float(E_avail), e_per_m_Wh=float(e_per_m),
                T_avail_s=float(T_avail), E_insp1_Wh=float(E_insp1),
                E_dock_env_Wh=float(E_dock_env), h_max_min=float(h_max_min),
                # A moving recovery ship can shorten the return leg, so this
                # symmetric launch-distance radius is a useful diagnostic but
                # is NOT a theorem-level outer bound for formal preprocessing.
                formal_outer_bound_certified=False,
                bound_scope="static-symmetric-diagnostic-only")


# =============================================================================
# 6. 名义(确定性)可行性 —— 仅作【对照】(nominal comparison), 非基础模型 (ξ=μ 或 ξ=0)
#    注: 本项目唯一正式模型是完整航路模型(step10+); 单停靠 + 名义点仅供 exp 对照,
#    不作为基础模型或阶段性主模型(见 README "去基础模型")。
# =============================================================================
def nominal_feasible(s: Sortie, p: Params, wx: dict, xi_amb: XiAmbiguity) -> dict:
    """用 ξ=μ_{h,c}(名义偏置)评估能量/时间/门是否满足。返回诊断 dict。
    这是 exp 名义【对照】(忽略 ξ 分布、只用名义点)的核心评估, 不是基础模型。"""
    cell = xi_amb.get(s.h_min(), s.c_state)
    E, T = energy_time(s, cell.mu, p, wx)
    L, r_eff = landing_gate(wx["Hs"], wx.get("Tp", 2.1), wx.get("wave_dir", 200.0),
                            wx.get("ship_heading", 0.0), wx["wind10"], p)
    return dict(
        E_Wh=E, B_use=p.B_use, energy_ok=E <= p.B_use,
        T_s=T, T_plan_s=s.planned_flight_s(), time_ok=T <= s.planned_flight_s(),
        gate=L, r_cap_eff=r_eff,
        xi_mean_norm=cell.mean_norm(), xi_p95=cell.p95_norm, n=cell.n,
    )


# =============================================================================
# 7. 自检 / 演示(python step9_model.py)
# =============================================================================
def _selftest():
    here = Path(__file__).resolve().parent
    # 数据路径优先用本机常见位置;不存在则占位
    turb_csv = here / "data" / "turbines_Rodsand_II_clean.csv"
    xi_csv = _first_existing([
        here / "xi_moments_caseB.csv",
        here / "tracks" / "xi_moments_caseB.csv",
        Path("/mnt/user-data/uploads/xi_moments_caseB.csv"),
    ])
    wave_csv = _first_existing([
        here / "weather" / "waves_Rodsand_II.csv",
        Path("/mnt/user-data/uploads/waves_Rodsand_II.csv"),
    ])
    wind_csv = _first_existing([
        here / "weather" / "weather_Rodsand_II.csv",
        Path("/mnt/user-data/uploads/weather_Rodsand_II.csv"),
    ])

    p = Params()
    p.validate_contract(formal=False)
    log.info("B_use = %.1f Wh (= %.0f%% of %.1f)", p.B_use, 100*(1-p.safe_reserve), p.B_k)

    # 风机
    if turb_csv.is_file():
        turbines = load_turbines(turb_csv, farm="Rodsand_II")
    else:
        turbines = [Turbine("DEMO", np.array([11.55, 54.55]), 68.5, 115.0)]
        log.warning("无风机文件,使用 1 台占位风机。")
    lat0, lon0 = turbines[0].lonlat[1], turbines[0].lonlat[0]
    for tb in turbines:
        tb_local = latlon_to_local_m(tb.lonlat[1], tb.lonlat[0], lat0, lon0)
        setattr(tb, "local", tb_local)

    # 天气
    wx_df = load_weather(wave_csv, wind_csv)
    # 选一个风非缺测的行做样点(老 6 月文件部分行 wind10 为 NaN)
    valid = wx_df.dropna(subset=["wind10_ms"]) if "wind10_ms" in wx_df.columns else wx_df
    row = (valid.iloc[len(valid) // 2] if len(valid) else wx_df.iloc[len(wx_df) // 2])
    def _f(v, d):
        try:
            return float(v) if not pd.isna(v) else d
        except Exception:
            return d
    wx = dict(wind10=_f(row.get("wind10_ms"), 6.7),
              wind_dir_from=_f(row.get("wind_dir_from_deg"), 230.0),
              Hs=_f(row.get("Hs_m"), 0.5), Tp=_f(row.get("wave_Tm_s"), 2.1),
              wave_dir=_f(row.get("wave_dir_deg"), 200.0), ship_heading=90.0)

    # ξ 模糊集
    if xi_csv:
        xi_amb = XiAmbiguity.from_csv(xi_csv, mmsi="ALL")
    else:
        # 占位：仅供自检，显式提供所用的 h=20 锚点；不依赖最近格回退。
        demo = {(20, "直航"): XiCell(20, "直航", 1000, np.array([0., 0.]),
                                    np.array([[1e6, 0], [0, 1e6]]), 5000., 2000., 1000.)}
        xi_amb = XiAmbiguity(demo, [20])
        log.warning("无 ξ 文件,使用占位模糊集。")

    # 构造一个候选架次:决策时刻 t0,起飞 +2min,回收 +20min
    t0 = wx_df.index[len(wx_df) // 2]
    tb = turbines[0]
    # 假设船在风机西南 ~3km 处起飞,回收点在风机西 ~4km(船在动)
    P_launch = tb.local + np.array([-3000.0, -2000.0])
    P_recover_pred = tb.local + np.array([-4500.0, 500.0])
    s = Sortie(rid=0, turbine=tb, t_L=t0 + pd.Timedelta(minutes=2),
               t_R=t0 + pd.Timedelta(minutes=20), t0=t0,
               P_launch=P_launch, P_recover_pred=P_recover_pred, c_state="直航")
    setattr(s, "turbine_local", tb.local)

    print("\n================ step9_model.py 自检 ================")
    print(f"风机数: {len(turbines)} | 决策时刻 t0={t0} | h(r)={s.h_min():.0f} min | 状态 c={s.c_state}")
    print(f"天气样点: Hs={wx['Hs']:.2f}m wind10={wx['wind10']:.1f}m/s wave_dir={wx['wave_dir']:.0f}°")

    # 名义评估(ξ=μ)
    diag = nominal_feasible(s, p, wx, xi_amb)
    print("\n--- 名义可行性 (ξ = μ_{h,c}) ---")
    print(f"  能量 E={diag['E_Wh']:.1f} Wh  vs B_use={diag['B_use']:.1f} Wh  -> {'OK' if diag['energy_ok'] else '超限'}")
    print(f"  时间 T={diag['T_s']:.0f} s  vs 计划={diag['T_plan_s']:.0f} s  -> {'OK' if diag['time_ok'] else '超限'}")
    print(f"  着舰门 L={diag['gate']}  有效捕获半径={diag['r_cap_eff']:.2f} m")
    print(f"  ξ 该格: 均值范数={diag['xi_mean_norm']:.1f}m  p95={diag['xi_p95']:.0f}m  样本n={diag['n']}")

    # ξ 不确定性扫一遍:用 p95 半径方向最坏点看能量/时间敏感性
    cell = xi_amb.get(s.h_min(), s.c_state)
    print("\n--- ξ 敏感性(沿 ±p95 各方向的返程能量/时间)---")
    for ang in [0, 90, 180, 270]:
        xi = cell.p95_norm * np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
        E, T = energy_time(s, xi, p, wx)
        print(f"  ξ@{ang:3d}° |ξ|={cell.p95_norm:.0f}m -> E={E:.1f}Wh T={T:.0f}s "
              f"{'(E超)' if E>p.B_use else ''}{'(T超)' if T>s.planned_flight_s() else ''}")

    print("\n自检完成。能耗/时间/门/模糊集链路已跑通。")
    print("下一步: step10_model_routing.py + step11/step12 用 XiAmbiguity 的 (μ,Σ,支持集) 做 DRCC 可处理化 + Gurobi 求解。")


def _first_existing(paths: list[Path]) -> Optional[Path]:
    for p in paths:
        try:
            if p.is_file():
                return p
        except OSError:      # 更新: 网络/临时挂载点可能抖动(Errno 5 等) —— 视为不存在, 继续探测
            continue
    return None


if __name__ == "__main__":
    _selftest()

# =============================================================================
# Shared experiment integrity/statistics utilities (merged here to preserve the
# original 27-file package layout; imported as step9_model by experiment scripts).
# =============================================================================
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        return {str(k): _jsonable(v) for k, v in vars(value).items()
                if not str(k).startswith("_")}
    return str(value)


def sha256_file(path: str | Path | None, chunk_size: int = 1024 * 1024) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _git_commit(cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(cwd), stderr=subprocess.DEVNULL,
            text=True, timeout=3).strip()
    except Exception:
        return None


def _package_availability(names: Iterable[str]) -> dict[str, bool]:
    """Record import availability for the execution manifest."""
    import importlib.util
    return {str(name): importlib.util.find_spec(str(name)) is not None for name in names}


def _package_versions(names: Iterable[str]) -> dict[str, str | None]:
    """Resolve installed distribution versions without importing heavy packages."""
    try:
        import importlib.metadata as metadata
    except Exception:  # pragma: no cover - Python >=3.8 in supported formal runs
        return {str(name): None for name in names}
    out: dict[str, str | None] = {}
    for name in names:
        key = str(name)
        try:
            out[key] = metadata.version(key)
        except Exception:
            out[key] = None
    return out


def source_tree_sha256(root: str | Path | None = None) -> str:
    """Deterministically bind the auditable project source/configuration tree.

    Relative path names are hashed together with file bytes, so exchanging the
    contents of two files cannot preserve the digest. Runtime artefacts and
    caches are deliberately excluded.
    """
    import hashlib
    base = Path(root or Path(__file__).resolve().parent).resolve()
    excluded_parts = {".git", "__pycache__", "results", "cache", ".pytest_cache"}
    allowed_suffixes = {".py", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".md"}
    h = hashlib.sha256()
    for f in sorted((x for x in base.rglob("*") if x.is_file()),
                    key=lambda x: x.relative_to(base).as_posix()):
        rel = f.relative_to(base)
        if any(part in excluded_parts for part in rel.parts):
            continue
        if f.suffix.lower() not in allowed_suffixes:
            continue
        h.update(rel.as_posix().encode("utf-8")); h.update(b"\0")
        h.update(f.read_bytes()); h.update(b"\0")
    return h.hexdigest()


def martingale_conditional_risk_upper_bound(failures: Iterable[int | bool],
                                             confidence: float = 0.95) -> float:
    r"""One-sided Hoeffding-Azuma bound for average conditional failure risk.

    For adapted binary failures X_i and mu_i=E[X_i | F_{i-1}], with probability
    at least ``confidence``::

        mean(mu_i) <= mean(X_i) + sqrt(log(1/alpha)/(2*n)).

    Unlike the binomial/Clopper-Pearson diagnostic this does not assume iid
    Bernoulli trials or a common fixed failure probability. It still does not
    protect against arbitrary post-test distribution shift.
    """
    xs = [int(x) for x in failures]
    if not xs:
        raise ValueError("at least one binary failure observation is required")
    if any(x not in (0, 1) for x in xs):
        raise ValueError("failures must be binary (0/1 or bool)")
    c = float(confidence)
    if not (0.0 < c < 1.0):
        raise ValueError("confidence must be in (0,1)")
    n = len(xs)
    alpha = 1.0 - c
    return min(1.0, sum(xs) / n + math.sqrt(math.log(1.0 / alpha) / (2.0 * n)))


def write_run_manifest(outdir: str | Path, experiment: str, args: Any = None,
                       input_paths: Iterable[str | Path] | None = None,
                       extra: Mapping[str, Any] | None = None,
                       filename: str = "run_manifest.json") -> Path:
    """Write an auditable manifest next to experiment outputs.

    The file is overwritten atomically for each checkpoint, so interrupted runs
    always leave a valid JSON document describing the latest invocation.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    inputs = []
    for raw in input_paths or ():
        p = Path(raw)
        inputs.append({
            "path": str(p),
            "exists": p.exists(),
            "size_bytes": p.stat().st_size if p.is_file() else None,
            "sha256": sha256_file(p),
        })
    manifest = {
        "experiment": str(experiment),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_dir": str(here),
        "git_commit": _git_commit(here),
        "command": [sys.executable, *sys.argv],
        "arguments": _jsonable(args),
        "python_runtime": {
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "version": sys.version,
            "version_info": list(sys.version_info[:5]),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "environment": {k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "PYTHONHASHSEED", "TZ", "LANG", "LC_ALL")},
        "package_availability": _package_availability(
            ["numpy", "pandas", "scipy", "matplotlib", "gurobipy", "highspy"]),
        "package_versions": _package_versions(
            ["numpy", "pandas", "scipy", "matplotlib", "gurobipy", "highspy"]),
        "source_tree_sha256": source_tree_sha256(here),
        "inputs": inputs,
        "extra": _jsonable(extra or {}),
    }
    target = out / filename
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)
    return target


def binomial_interval(successes: int, trials: int, confidence: float = 0.95,
                      method: str = "clopper-pearson") -> tuple[float | None, float | None]:
    """Two-sided binomial confidence interval.

    Uses exact Clopper-Pearson when SciPy is available and Wilson otherwise.
    Returns ``(None, None)`` for zero trials.
    """
    k, n = int(successes), int(trials)
    if n <= 0:
        return None, None
    if k < 0 or k > n:
        raise ValueError(f"successes must satisfy 0 <= k <= n, got {k}/{n}")
    alpha = 1.0 - float(confidence)
    if method.lower().replace("_", "-") in {"clopper-pearson", "exact"}:
        try:
            from scipy.stats import beta
            lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
            hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
            return lo, hi
        except Exception:
            pass
    # Wilson score interval fallback.
    try:
        from statistics import NormalDist
        z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    except Exception:  # 95% fallback
        z = 1.959963984540054
    phat = k / n
    den = 1.0 + z * z / n
    ctr = (phat + z * z / (2.0 * n)) / den
    half = z * ((phat * (1.0 - phat) / n + z * z / (4.0 * n * n)) ** 0.5) / den
    return max(0.0, ctr - half), min(1.0, ctr + half)


def binomial_upper_bound(successes: int, trials: int, confidence: float = 0.95,
                         method: str = "clopper-pearson") -> float | None:
    """One-sided upper confidence bound for a binomial probability."""
    k, n = int(successes), int(trials)
    if n <= 0:
        return None
    if k < 0 or k > n:
        raise ValueError(f"successes must satisfy 0 <= k <= n, got {k}/{n}")
    alpha = 1.0 - float(confidence)
    if method.lower().replace("_", "-") in {"clopper-pearson", "exact"}:
        try:
            from scipy.stats import beta
            return 1.0 if k == n else float(beta.ppf(1.0 - alpha, k + 1, n - k))
        except Exception:
            pass
    # Conservative fallback from the two-sided Wilson interval.
    return binomial_interval(k, n, confidence=confidence, method="wilson")[1]


def matrix_completion(expected: Iterable[tuple[Any, ...]], observed: Iterable[tuple[Any, ...]]) -> dict[str, Any]:
    exp = {tuple(map(str, x)) for x in expected}
    obs = {tuple(map(str, x)) for x in observed}
    missing = sorted(exp - obs)
    unexpected = sorted(obs - exp)
    return {
        "expected_count": len(exp),
        "observed_count": len(obs & exp),
        "complete": not missing,
        "missing": missing,
        "unexpected": unexpected,
    }
