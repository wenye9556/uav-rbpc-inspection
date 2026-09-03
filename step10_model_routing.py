#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step10_model_routing.py — 航路化多机巡检 + 决策依赖模糊集的物理层。

这是完整航路模型的核心物理层文件:

  航路化:一个架次(sortie)= 无人机一次出动, 从母船起飞 → 顺序巡检【多台】风机
            → 返回【仍在移动】的母船着舰回收。能耗/时间因此【依赖访问序列】,
            可行路径数随风机数指数增长 → 这才需要列生成/分支定价等高级算法。

  决策依赖模糊集:回收时长 h = t_R − t_0 现在是【决策】(选多晚回收), 而:
            (a) 船位预测误差模糊集 P_{h,c} 的矩 (μ_h, Σ_h) 随 h 增大而增大;
            (b) 预测回收点 P̂_v(t_0+h) 随 h 沿船航迹移动 → 返程几何随 h 变;
            (c) 时间裕度 b_T = h − T_route(0) 随 h 增大而增大。
            => 选更晚回收 ⇒ 时间裕度更大(利), 但预测不确定性更大、船跑更远(弊)。
            模糊集本身随决策 h【移动】, 这正是决策依赖 DRO(DD-DRO)的核心张力。
            把 t_R 离散成固定候选(传统做法)回避了它; 本模型让 h 成为真决策, 逐 h 权衡。

本文件只负责【给定一条路由 + 一个 h】如何算 E_route(ξ)、T_route(ξ)、DRCC 可行性。
搜索/选路/求解在 step11_algorithm_route_drcc.py。复用 step9_model.py 的参数容器、
几何工具、能耗原语(起降/巡检/地速/高度风)、着舰门与 XiAmbiguity, 不重复实现。

依赖: pip install numpy pandas    (Gurobi 仅 step14 的精确主问题需要)
自检: python step10_model_routing.py   (占位船航迹 + data/ 真实风机 + 占位 ξ)

与 model.md §12(航路模型)严格对应。符号见 model.md / params.md。
"""
from __future__ import annotations

import logging
import math
import time
from fractions import Fraction
from statistics import NormalDist
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

import step9_model as M  # 复用 step9 的物理原语与数据结构

log = logging.getLogger("route")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Fixed-touchdown time-accounting contract.  The nominal stern escort interval is
# recourse: flight/dock delay consumes it before it can create a touchdown overrun.
WAIT_ONLY_TIME_CONTRACT = "fixed_touchdown_wait_recourse"
SPEED_RECOURSE_TIME_CONTRACT = "fixed_touchdown_wait_speed_recourse"
TIME_CONTRACT = SPEED_RECOURSE_TIME_CONTRACT
WAIT_IS_RECOURSE = True
SPEED_IS_RECOURSE = True
DOCK_RISK_CONTRACT = "risk_adjusted_dock_in_core_no_extra_dock_risk"
GEO_RISK_ALLOCATION_CONTRACT = "geo2d_bonferroni_route_optimized_grid"
SPEED_RECOURSE_CONTRACT = "required_airspeed_vector_geo2d_with_nonreturn_weather"
ENERGY_SPEED_RECOURSE_CONTRACT = "fixed_nonreturn_plus_power_envelope_remaining_time"
POWER_ENVELOPE_CONTRACT = "analytic-zeng-endpoint-maximum-unimodality-proof"
# Formal certificate semantics: feasibility comparisons are exact comparisons of
# the binary64 values produced by the declared finite physical model.  Positive
# tolerances may be used for diagnostics only; they must never enlarge the
# certified feasible set.
FORMAL_PHYSICAL_NUMERIC_CONTRACT = "strict-binary64-feasibility-no-positive-tolerance-v2"
WEATHER_PREDICTOR_CONTRACTS = {
    "weather_speed_primary_coherent_noleak": "weather_backward_linear_speed_primary_coherent_epoch_seconds_v2",
}
WEATHER_TIMESTAMP_EPOCH_CONTRACT = M.XI_TIMESTAMP_EPOCH_CONTRACT
WEATHER_FORMAL_DATA_CONTRACT = "real-history-noleak-weather-residuals-global-weather-nonoverlap-v3-coherent-wind"
WEATHER_TRUTH_CONTRACT = "era5-reanalysis-wind+cmems-historical-wave-hourly-linear-truth-v1"
TIME_TOL_S = 1e-7  # diagnostic/assertion tolerance only; never used to enlarge formal feasibility.


def _strict_finite_leq(value: float, limit: float) -> bool:
    """Fail-closed binary64 comparison used by formal physical predicates."""
    try:
        value = float(value)
        limit = float(limit)
    except (TypeError, ValueError):
        return False
    return bool(math.isfinite(value) and math.isfinite(limit) and value <= limit)


def _strict_probability(eps: float, name: str = "eps") -> float:
    """Return an unchanged finite probability in (0,1), never silently clip it."""
    value = float(eps)
    if not math.isfinite(value) or not (0.0 < value < 1.0):
        raise ValueError(f"{name} must be finite and lie in (0,1)")
    return value


@dataclass(frozen=True)
class RiskPolicy:
    """Immutable DRCC risk contract for one certified physical evaluation.

    Formal exact optimization passes this object through the complete physical
    call chain so no helper can silently consult the mutable module-level
    ``kappa`` selector.  Legacy/research callers that omit it retain the old
    module-global behavior for backward compatibility, but that path is not a
    formal certificate contract.
    """
    mode: str
    one_sided: Callable[[float], float]
    two_sided: Callable[[float], float]


def risk_policy_for_mode(mode: str) -> RiskPolicy:
    key = str(mode).strip().lower()
    if key == "nominal":
        return RiskPolicy(key, lambda _eps: 0.0, lambda _eps: 0.0)
    if key not in KAPPA_MODES:
        raise ValueError(f"unknown risk mode={mode!r}")
    one = KAPPA_MODES[key]
    if key == "gaussian":
        def two(eps):
            e = _strict_probability(eps)
            try:
                from scipy.stats import norm
                return float(norm.ppf(1.0 - e / 2.0))
            except Exception:
                return float(NormalDist().inv_cdf(1.0 - e / 2.0))
    elif key == "vp_unimodal":
        def two(eps):
            e = _strict_probability(eps)
            kvp = math.sqrt(4.0 / (9.0 * e))
            return kvp if kvp >= math.sqrt(8.0 / 3.0) else 1.0 / math.sqrt(e)
    else:  # Cantelli is one-sided; the valid two-sided companion is Chebyshev.
        def two(eps):
            e = _strict_probability(eps)
            return 1.0 / math.sqrt(e)
    return RiskPolicy(key, one, two)


def _risk_policy_from_inputs(risk_policy=None, kappa_fn=None) -> RiskPolicy:
    if risk_policy is not None:
        if not isinstance(risk_policy, RiskPolicy):
            raise TypeError("risk_policy must be a RiskPolicy")
        return risk_policy
    if kappa_fn is not None:
        for key, fn in KAPPA_MODES.items():
            if kappa_fn is fn:
                return risk_policy_for_mode(key)
        try:
            if abs(float(kappa_fn(0.05))) < 1e-12:
                return risk_policy_for_mode("nominal")
        except Exception:
            pass
        # A custom one-sided function has no derivable two-sided theorem.  Use
        # distribution-free Chebyshev rather than consulting mutable globals.
        return RiskPolicy("custom", kappa_fn,
                          lambda eps: 1.0 / math.sqrt(_strict_probability(eps)))
    # Legacy/research compatibility only: capture the current global selector
    # once, then keep the evaluation internally immutable.
    for key, fn in KAPPA_MODES.items():
        if kappa is fn:
            return risk_policy_for_mode(key)
    try:
        if abs(float(kappa(0.05))) < 1e-12:
            return risk_policy_for_mode("nominal")
    except Exception:
        pass
    return RiskPolicy("legacy-custom", kappa,
                      lambda eps: 1.0 / math.sqrt(_strict_probability(eps)))


def _check_deadline(deadline):
    """Cooperative wall-clock cancellation for physical route evaluation.

    This cannot pre-empt an arbitrary third-party blocking call, but all loops
    and major phases in this module can observe the shared solver deadline.
    """
    if deadline is not None and time.monotonic() >= float(deadline):
        raise TimeoutError("global deadline reached inside route_feasible_at_h")


def time_contract_for(p) -> str:
    """Return the exact fixed-touchdown contract selected by the model parameters."""
    return (SPEED_RECOURSE_TIME_CONTRACT
            if bool(getattr(p, "speed_adjustable", False))
            else WAIT_ONLY_TIME_CONTRACT)


def fixed_touchdown_time_accounting(h_s: float, time_core_nom_s: float,
                                     time_drcc_tightening_s: float = 0.0,
                                     tol: float = TIME_TOL_S) -> dict:
    """Apply the fixed-touchdown/wait-recourse time contract in seconds.

    ``time_core_nom_s`` contains flight, inspection and the single dock component,
    but never stern waiting.  Positive uncertainty first compresses waiting.
    """
    h_s = float(h_s)
    core = float(time_core_nom_s)
    tightening = float(time_drcc_tightening_s)
    if not all(math.isfinite(v) for v in (h_s, core, tightening)):
        raise ValueError("fixed-touchdown time inputs must be finite seconds")
    nominal_margin = h_s - core
    wait_nom = max(0.0, nominal_margin)
    safe_core = core + tightening
    drcc_margin = h_s - safe_core
    wait_safe = max(0.0, drcc_margin)
    assert safe_core == core + tightening
    return dict(
        time_contract=WAIT_ONLY_TIME_CONTRACT,
        time_contract_id=WAIT_ONLY_TIME_CONTRACT,
        wait_is_recourse=WAIT_IS_RECOURSE,
        dock_risk_contract=DOCK_RISK_CONTRACT,
        time_core_nom_s=core,
        time_wait_nom_s=wait_nom,
        time_drcc_tightening_s=tightening,
        time_safe_core_s=safe_core,
        time_wait_safe_s=wait_safe,
        nominal_time_margin_s=nominal_margin,
        time_drcc_margin_s=drcc_margin,
        nominal_time_failed=bool(nominal_margin < 0.0),
        time_drcc_failed=bool(drcc_margin < 0.0),
    )


def realized_fixed_touchdown_time(h_s: float, realized_core_time_s: float,
                                    tol: float = TIME_TOL_S) -> dict:
    """Realized counterpart used by replay; scheduled touchdown remains ``tau+h``."""
    h_s = float(h_s)
    core = float(realized_core_time_s)
    if not all(math.isfinite(v) for v in (h_s, core)):
        raise ValueError("realized fixed-touchdown inputs must be finite seconds")
    overrun = max(0.0, core - h_s)
    return dict(
        time_contract=TIME_CONTRACT, wait_is_recourse=WAIT_IS_RECOURSE,
        realized_core_time_s=core, realized_wait_s=max(0.0, h_s - core),
        scheduled_touchdown_s=h_s, time_overrun_s=overrun,
        time_violation=bool(core > h_s),
    )


# =============================================================================
# 0. Cantelli 系数(与 step10 一致, 这里独立放一份避免循环 import)
# =============================================================================
def kappa_cantelli(eps: float) -> float:
    """κ=√((1-ε)/ε); formal inputs are validated, never clipped."""
    eps = _strict_probability(eps)
    return math.sqrt((1.0 - eps) / eps)


def kappa_vp_unimodal(eps: float) -> float:
    r"""单边 Vysochanskij–Petunin(**单峰**分布族): P(X-μ≥λσ) ≤ 4/(9(1+λ²)), 对 λ²≥5/3 成立。
    令上界=ε 解得 λ=√(4/(9ε)−1)(适用 ε≤1/6; 超域回退 Cantelli)。
    要点: 比 Cantelli 紧(同 ε 下 κ 更小 ⇒ DR 可行集更大、覆盖更多), 但**仍分布无关**——只额外假设
    单峰(重尾 t、对称椭圆分布的 1D 投影均单峰), 故安全性远强于高斯正态假设。
    ε=0.05: κ_VP=2.81 vs Cantelli=4.36 vs 高斯=1.64。"""
    eps = _strict_probability(eps)
    val = 4.0 / (9.0 * eps) - 1.0
    if val < 5.0 / 3.0:                      # ε>1/6 出 VP 适用域 → 回退 Cantelli(保守但安全)
        return kappa_cantelli(eps)
    return math.sqrt(val)


def kappa_gaussian(eps: float) -> float:
    """κ=Φ⁻¹(1-ε), 正态假设(**非**分布无关; 重尾下欠保守, 仅作对照)。"""
    eps = _strict_probability(eps)
    try:
        from scipy.stats import norm
        return float(norm.ppf(1.0 - eps))
    except Exception:
        # Python's NormalDist provides a deterministic binary64 fallback; never
        # round ε to a lookup-table key because that changes the requested model.
        return float(NormalDist().inv_cdf(1.0 - eps))


# κ 模式注册表(供实验选择 / 消融; 默认 Cantelli 向后兼容)
KAPPA_MODES = {"cantelli": kappa_cantelli, "vp_unimodal": kappa_vp_unimodal, "gaussian": kappa_gaussian}


def kappa(eps: float) -> float:
    """默认 κ(ε)=√((1-ε)/ε)(Cantelli, 分布无关)。实验可 monkey-patch 本符号或用 KAPPA_MODES 选别的界。"""
    return kappa_cantelli(eps)


# =============================================================================
# 0b. 双层 h 网格(model.md §12.3): 决策层细化 —— 回收时长 h 是决策变量,
#     决策层在统计 horizon 覆盖区间内逐 h 评估；统计层可用粗格估矩，二者由
#     XiAmbiguity.get_interp 在相邻锚点间插值接驳，但正式模型禁止区间外外推。
#     细化的意义: (a) 在有统计支撑的区间内提高 h 决策分辨率;
#                (b) 候选路由列空间显著增大 → 精确算法 vs Gurobi 整体 MISOCP 的
#                    速度差距、vs 启发式的精度差距都更明显(算法实验更有说服力)。
# =============================================================================
DECISION_HORIZONS = list(range(5, 61, 5))   # 候选模板；实际候选严格裁剪到 xi 统计支持闭区间。


def decision_horizons_of(xi_amb: "M.XiAmbiguity", h_grid=None) -> list:
    """返回允许进入优化的回收时长。

    正式口径只允许在统计层已覆盖的 ``[min(h), max(h)]`` 内取值；相邻统计格之间可插值，
    但禁止向区间外做布朗外推。显式 ``h_grid`` 只要包含越界值就立即报错，避免实验静默
    使用没有直接统计支撑的误差分布。
    """
    H = sorted(float(h) for h in getattr(xi_amb, "horizons", []) if math.isfinite(float(h)))
    if not H:
        raise ValueError("xi_amb 不含任何统计 horizon。")
    h_min, h_max = min(H), max(H)
    if h_grid is not None:
        grid = sorted({float(h) for h in h_grid})
        bad = [h for h in grid if h < h_min or h > h_max]
        if bad:
            raise ValueError(f"决策回收时长 {bad} 超出统计支持区间 [{h_min:g},{h_max:g}] min；禁止外推。")
        return [int(h) if float(h).is_integer() else h for h in grid]
    grid = [float(h) for h in DECISION_HORIZONS if h_min <= float(h) <= h_max]
    grid.extend(H)
    out = sorted(set(grid))
    return [int(h) if float(h).is_integer() else h for h in out]


def _xi_cell_strict(xi_amb: "M.XiAmbiguity", h_query: float, c_state: str) -> "M.XiCell":
    """无外推、无状态回退地取得 ``(h,c)`` 矩信息。

    精确统计格必须存在对应状态；内插时上下两个锚点都必须存在该状态。缺格由调用方拒绝
    当前候选时长，而不是自动借用样本最多的其他船舶状态。
    """
    H = sorted(float(h) for h in getattr(xi_amb, "horizons", []))
    if not H:
        raise KeyError("模糊集无任何 horizon。")
    hq = float(h_query)
    if hq < H[0] or hq > H[-1]:
        raise KeyError(f"h={hq:g} 超出统计支持区间 [{H[0]:g},{H[-1]:g}]，禁止外推。")
    exact = next((h for h in H if float(h).hex() == float(hq).hex()), None)
    if exact is not None:
        key = (int(exact) if float(exact).is_integer() else exact, str(c_state))
        if key not in xi_amb.cells:
            raise KeyError(f"缺少统计格 (h={exact:g}, c={c_state})。")
        return xi_amb.cells[key]
    h_lo = max(h for h in H if h < hq)
    h_hi = min(h for h in H if h > hq)
    for hh in (h_lo, h_hi):
        key = (int(hh) if float(hh).is_integer() else hh, str(c_state))
        if key not in xi_amb.cells:
            raise KeyError(f"内插 h={hq:g} 缺少锚点 (h={hh:g}, c={c_state})。")
    return xi_amb.get_interp(hq, str(c_state))


# =============================================================================
# 1. 船位预测器接口(给定起飞点与 t_L, 返回各 h 的预测回收点)
# =============================================================================
def _cv_recovery_state(v_ship: np.ndarray, launch_state: str) -> str:
    """从 CV 预测轨迹本身推导未来状态，不把起飞状态机械持续到回收时刻。

    CV 的预测转弯率恒为零，因此起飞时为“转弯”的船在该预测器下应预测为直航（若速度足够），
    而不是在所有 h 上继续标为转弯。ξ 仍按起飞状态索引，不改变误差数据契约。
    非标准夹具状态（如测试中的 ``DP``）保持原值以兼容外部调用。
    """
    state = str(launch_state)
    if state not in {"直航", "转弯", "低速", "动力定位"}:
        return state
    speed_kn = float(np.linalg.norm(np.asarray(v_ship, float))) / float(M.KN)
    if speed_kn < 0.3:
        return "动力定位"
    if speed_kn < 1.0:
        return "低速"
    return "直航"


@dataclass
class ShipPrediction:
    """某次出动的船位预测: 起飞点 P_launch(本地米) + 各回收时长 h(分钟)的预测回收点。

    起飞时刻 τ = 决策/预测起点 t_0(本类的 t0)。**c_state = c(τ) 是【起飞/决策时刻
    可观测】的船舶运动状态**, 用作 ξ 模糊集索引 𝒫_{h,c(τ)} —— 与 step7 按【预测起点 t_0
    状态】归组 ξ 矩完全一致, 无未来信息泄漏。**严禁用回收时刻真实状态 c(τ+h) 索引**(规划时
    τ+h 的真实状态尚未发生 → 泄漏)。回收转弯门必须使用逐时长的【预测回收状态
    ĉ_R(τ,h)】；该预测及来源由上游显式写入，缺失时路线 fail-closed。回收处能否着舰由
    【着舰门】(海况/甲板运动)判定, 与 ξ 索引解耦。

    真实使用: 由 step8 类脚本对每个起飞时刻 τ, 用 t≤τ 的 AIS 建立 CV 预测器，
    在 step7 已统计支持的各 h 上给出 pred_by_h[h] = P̂_v(τ+h)；c_state=classify_state(τ)。
    自检使用: 用恒速航位推算 CV 合成 P̂_v(τ+h)=P_launch + v_ship·(h·60)。
    """
    P_launch: np.ndarray                 # (2,) 本地米, = \hat P_v(τ)
    pred_by_h: dict[int, np.ndarray]     # h(min) -> 预测回收点(2,) 本地米, = \hat P_v(τ+h)
    c_state: str                          # τ(起飞/决策)时刻可观测船舶状态 c(τ) —— ξ 模糊集索引(无泄漏)
    t0: pd.Timestamp = None              # 决策时刻 = 起飞时刻 τ
    recovery_state_by_h: dict[float, str] = field(default_factory=dict)
    recovery_state_source_by_h: dict[float, str] = field(default_factory=dict)

    @classmethod
    def from_cv(cls, P_launch: np.ndarray, v_ship: np.ndarray,
                horizons: list[int], c_state: str, t0=None) -> "ShipPrediction":
        """恒速航位推算合成预测点(自检/缺真实预测时用)。v_ship: 船速向量 m/s。"""
        pred = {int(h): P_launch + v_ship * (h * 60.0) for h in horizons}
        # 回收状态由 CV 预测运动本身推导；不再把起飞“转弯”等状态机械持续到所有 h。
        predicted_state = _cv_recovery_state(np.asarray(v_ship, float), str(c_state))
        rs = {int(h): predicted_state for h in horizons}
        rss = {int(h): 'cv-noleak-state-from-predicted-motion' for h in horizons}
        sp = cls(P_launch=np.asarray(P_launch, float), pred_by_h=pred,
                 c_state=c_state, t0=t0, recovery_state_by_h=rs,
                 recovery_state_source_by_h=rss)
        sp._v_ship = np.asarray(v_ship, float)   # 保留同一无泄漏 CV 速度，供支持区间内任意 h 预测
        return sp

    def predicted_at(self, h: float) -> np.ndarray:
        """统计支持区间内任意回收时长 h 的预测回收点。

        命中已存格点直接取；CV 场景使用起飞时刻已估计的同一速度计算；没有 CV 速度时只允许
        在相邻预测格点之间线性插值。候选 h 的统计区间约束由 ``decision_horizons_of`` 和
        ``_xi_cell_strict`` 双重执行，因此这里不为优化提供统计区间外的预测支撑。
        """
        hq = float(h)
        exact_key = next((k for k in self.pred_by_h
                          if float(k).hex() == hq.hex()), None)
        if exact_key is not None:
            return self.pred_by_h[exact_key]
        # ---- 更新 修复(审计 P0-1 无泄漏): 真实未来航迹插值【仅】在显式选择
        #   predictor='true_track'(解释 A: 已知计划航线)时使用; 默认 cv_noleak 口径下
        #   一律 CV 外推(与 step7 ξ 统计的预测器同一语义, 严格只用 t≤τ 信息)。
        #   旧口径把 track.pos(τ+h) 当"预测", 与"回收点=起飞时可得预测"的声明矛盾。
        trk = getattr(self, "_track", None)
        if (trk is not None and getattr(self, "_t_launch_sec", None) is not None
                and getattr(self, "_predictor", "cv_noleak") == "true_track"):
            return trk.pos(self._t_launch_sec + float(h) * 60.0)
        # CV 外推(默认; 与 step7 预测器一致, 无未来信息)
        v = getattr(self, "_v_ship", None)
        if v is not None:
            return self.P_launch + v * (float(h) * 60.0)
        # 无船速 → 在粗格间线性插值预测点
        Hs = sorted(self.pred_by_h)
        below = [x for x in Hs if x <= h]
        above = [x for x in Hs if x >= h]
        if below and above and max(below) != min(above):
            lo, hh = max(below), min(above)
            w = (h - lo) / (hh - lo)
            return (1 - w) * self.pred_by_h[lo] + w * self.pred_by_h[hh]
        # 无 CV 速度且超出预测点覆盖范围时拒绝外推。
        raise KeyError(f"预测回收点 h={float(h):g} 超出已有区间 [{min(Hs):g},{max(Hs):g}] min。")

    def recovery_state_at(self, h: float) -> tuple[str, str]:
        """返回逐时长、无未来泄漏的预测回收状态及来源。

        分类状态禁止插值、最近时长吸附和起飞状态隐式回退。上游必须对每个候选 ``h``
        显式写入预测状态；缺失时由路线可行性层 fail-closed。
        """
        if not self.recovery_state_by_h:
            raise KeyError(f"缺少回收状态预测 h={float(h):g}；禁止使用起飞状态代理。")
        hq = float(h)
        raw_key = next((k for k in self.recovery_state_by_h
                        if float(k).hex() == hq.hex()), None)
        if raw_key is None:
            raise KeyError(f"缺少回收状态预测 h={hq:g}；分类状态禁止吸附到最近时长。")
        exact = float(raw_key)
        source = self.recovery_state_source_by_h.get(
            raw_key, self.recovery_state_source_by_h.get(exact, "predicted_by_h"))
        source = str(source).strip() or "predicted_by_h"
        if source in {"launch_state_proxy", "nearest_state", "future_realized_state"}:
            raise ValueError(f"非法回收状态来源: {source}")
        return str(self.recovery_state_by_h[raw_key]), source

    # ---- 更新 新增: 事后审计接口(不参与规划; 仅当绑定了真实航迹时可用) ----
    def realized_at(self, h: float):
        """真实回收时刻船位 P_track(τ+h)。仅供【事后】回放/审计, 规划严禁调用。
        未绑定航迹(合成 CV 场景)返回 None。"""
        trk = getattr(self, "_track", None)
        t0 = getattr(self, "_t_launch_sec", None)
        if trk is None or t0 is None:
            return None
        return trk.pos(float(t0) + float(h) * 60.0)

    def xi_realized(self, h: float):
        """该列的【单次真实实现误差】ξ_real = P_track(τ+h) − P̂(τ+h)(cv_noleak 口径下
        即 CV 预测误差的一次真实抽样)。供回放做逐列 realized 审计; 无航迹返回 None。"""
        real = self.realized_at(h)
        if real is None:
            return None
        return np.asarray(real, float) - np.asarray(self.predicted_at(float(h)), float)

    def weather_at_h(self, h: float, fallback: dict | None = None) -> dict:
        """Return the nominal weather attached to this launch at elapsed ``h`` minutes.

        Launch-grid builders attach exact values for every decision horizon.  When that strict
        contract is present, a missing horizon is a data error rather than a reason to reuse launch
        weather.  Legacy/synthetic callers without a weather timeline retain the historical fallback.
        """
        hq = float(h)
        if hq == 0.0 and isinstance(getattr(self, "wx_tau", None), dict):
            return dict(self.wx_tau)
        table = getattr(self, "weather_by_h", None)
        if isinstance(table, dict) and table:
            exact = next((k for k in table if float(k).hex() == hq.hex()), None)
            if exact is not None:
                return dict(table[exact])
            if bool(getattr(self, "_weather_contract_strict", False)):
                raise KeyError(f"缺少回收天气 h={hq:g} min；禁止复用起飞天气。")
        if fallback is not None:
            return dict(fallback)
        if isinstance(getattr(self, "wx_tau", None), dict):
            return dict(self.wx_tau)
        raise KeyError(f"未绑定天气 h={hq:g} min。")


# =============================================================================
# 1b. 起飞时刻网格(model.md §6.2 起飞—回收协同定时): τ 是决策变量。
#     不同 τ ⇒ 不同起飞船位 P^L=P̂_v(τ)、不同可观测状态 c(τ)、不同时刻风浪环境。
#     与 h 分工: τ 选【作业时空环境】, h 选【回收风险与未来会合点】。t_R=τ+h。
#     列结构升级为 r=(τ,ω,h)。求解时逐 τ 定价/枚举(见 step11/step12)。
# =============================================================================
@dataclass
class LaunchOption:
    """一个候选起飞时刻 τ 及其船位预测与该时刻天气。"""
    tau_min: float                  # 起飞时刻相对基准 t0 的偏移(分钟); 真实用绝对 τ
    ship: ShipPrediction            # 该 τ 的船位预测(c_state=c(τ))
    wx: dict                        # 该 τ 时刻的天气(各腿/着舰门用; 时变天气时随 τ 不同)


def _track_backvel(track, tk_sec: float, window_s: float = 60.0) -> np.ndarray:
    """【更新 无泄漏】只用 t≤tk 的航迹样本估起飞时刻船速(后向窗差分, 与 step7 CV 预测器
    的 vel-window 语义一致)。窗内位移过短/在航迹起点则退化为零速(诚实: 无信息即不外推)。"""
    t0 = float(track.t[0])
    tk = float(tk_sec)
    ta = max(tk - float(window_s), t0)
    el = tk - ta
    if el < 1e-6:
        return np.zeros(2)
    return (np.asarray(track.pos(tk), float) - np.asarray(track.pos(ta), float)) / el


def build_launch_grid_from_track(track, slot_times_sec, horizons, c_state="动力定位",
                                 wx_base=None, wx_of_t=None, wx_forecast_of=None,
                                 c_state_of_t=None, recovery_state_predictor=None,
                                 predictor: str = "cv_noleak", vel_window_s: float = 60.0,
                                 mission_origin_sec: float = 0.0):
    """【Phase2】在真实母船 AIS 航迹上构造【K 个起飞时隙】的 LaunchOption。
      track          : step9.ShipTrack(真实航迹, 提供 pos(t)/vel(t))。
      slot_times_sec : K 个起飞时隙的时刻(相对航迹首点的秒); 每个 = 一次架次的起飞时机。
      horizons       : 回收时长 h(分钟)预测格；细 h 仅在覆盖区间内插值或用同一 CV 预测器计算。
      predictor      : 回收点名义预测的口径(更新 修复, 审计 P0-1):
        'cv_noleak'(默认, 正式口径) —— P̂(τ+h)=P(τ)+v̂(τ)·h·60, v̂(τ) 用【仅 t≤τ】的后向窗
            差分估计(_track_backvel, 窗=vel_window_s)。与 step7 估 ξ 矩的 CV 预测器同一语义,
            名义回收点与 ξ 分布严格自洽、无未来信息泄漏; 真实航迹仅经 realized_at/xi_realized
            供【事后】回放审计。
        'true_track'(仅消融/『已知计划航线』解释 A) —— pred_by_h[h]=track.pos(τ+h·60) 沿真实
            未来航迹。⚠ 该口径把未来真值当预测, 与"无泄漏预测"声明不相容, 使用时必须在论文中
            改述为"船舶计划航线已知、ξ 为对计划的偏差", 且 ξ 矩须重估为对计划航线的偏差。
    每个时隙 k: 起飞船位 P_launch=track.pos(t_k)、可观测状态 c(t_k) 按【launch-asof】列口径构造。
    注意: 在一次性静态 master 同时选择未来多个 τ 时，这不等同于 mission-start nonanticipativity；
    v17 会在结果中显式标记该全局性质未获证明。正式天气可通过 ``wx_forecast_of(issue_sec,target_sec)``
    注入，并要求只使用 issue_sec 以前的观测；``wx_of_t`` 仅保留为机制/旧接口。返回 [LaunchOption,...]。"""
    if predictor not in ("cv_noleak", "true_track"):
        raise ValueError(f"predictor 必须是 cv_noleak|true_track, 得到 {predictor!r}")
    wx_base = wx_base or {}
    horizons = list(horizons)
    mission_origin_sec = float(mission_origin_sec)
    opts = []
    for k, tk in enumerate(slot_times_sec):
        P_launch = np.asarray(track.pos(tk), float)
        if predictor == "cv_noleak":
            v_hat = _track_backvel(track, tk, vel_window_s)
            pred = {int(h): P_launch + v_hat * (float(h) * 60.0) for h in horizons}
        else:  # true_track(消融)
            v_hat = track.vel(tk)
            pred = {int(h): track.pos(tk + float(h) * 60.0) for h in horizons}
        cs = (c_state_of_t(tk) if callable(c_state_of_t) else (c_state(tk) if callable(c_state) else c_state))
        rs, rss = {}, {}
        for hh in horizons:
            if callable(recovery_state_predictor):
                pred_state = recovery_state_predictor(float(tk), float(hh), np.asarray(v_hat, float), str(cs))
                if isinstance(pred_state, tuple):
                    state_h, source_h = pred_state
                else:
                    state_h, source_h = pred_state, 'declared-noleak-state-predictor'
            else:
                # 由预测轨迹的速度/转率语义分类；CV 转率为零，禁止机械持续起飞转弯状态。
                state_h = _cv_recovery_state(np.asarray(v_hat, float), str(cs))
                source_h = 'cv-noleak-state-from-predicted-motion'
            rs[int(hh)] = str(state_h); rss[int(hh)] = str(source_h)
        launch_abs = None
        if getattr(track, "absolute_start", None) is not None:
            launch_abs = pd.Timestamp(track.absolute_start) + pd.to_timedelta(float(tk), unit="s")
        sp = ShipPrediction(P_launch=P_launch, pred_by_h=pred, c_state=cs, t0=launch_abs,
                            recovery_state_by_h=rs, recovery_state_source_by_h=rss)
        sp._v_ship = np.asarray(v_hat, float)      # 支持区间内细 h 使用同一 v̂(cv_noleak 下无泄漏)
        # 绑定真实航迹: cv_noleak 下【仅】供 realized_at/xi_realized 事后审计;
        # true_track 下 predicted_at 才会沿真实航迹插值(由 _predictor 门控)。
        sp._track = track; sp._t_launch_sec = float(tk); sp._predictor = predictor
        tau_rel_min = (float(tk) - mission_origin_sec) / 60.0
        if tau_rel_min < 0.0:
            raise ValueError("起飞时隙早于任务窗口起点。")
        sp.tau_min = float(tau_rel_min); sp.slot = int(k)
        sp._mission_origin_sec = mission_origin_sec
        if callable(wx_forecast_of):
            # Formal path: every target weather value is issued from information
            # available at launch time tk; future realized weather never enters
            # route-column construction.
            wx_tau = dict(wx_forecast_of(float(tk), float(tk)))
            sp.weather_by_h = {
                (int(hh) if float(hh).is_integer() else float(hh)):
                dict(wx_forecast_of(float(tk), float(tk) + float(hh) * 60.0))
                for hh in horizons
            }
            sp._weather_information_mode = "launch_asof_forecast"
        else:
            wx_tau = (wx_of_t(tk) if callable(wx_of_t) else dict(wx_base))
            sp.weather_by_h = {
                (int(hh) if float(hh).is_integer() else float(hh)):
                (dict(wx_of_t(float(tk) + float(hh) * 60.0)) if callable(wx_of_t) else dict(wx_tau))
                for hh in horizons
            }
            sp._weather_information_mode = ("realized-target-series" if callable(wx_of_t)
                                            else "constant-weather")
        sp.wx_tau = dict(wx_tau)
        sp._weather_contract_strict = bool(callable(wx_forecast_of) or callable(wx_of_t))
        sp._ship_information_mode = ("launch_asof_cv" if predictor == "cv_noleak"
                                     else "declared-plan-or-hindsight-true-track")
        sp._column_information_cutoff_sec = float(tk)
        opts.append(LaunchOption(tau_min=float(tau_rel_min), ship=sp, wx=dict(wx_tau)))
    return opts


def build_launch_grid(P0: np.ndarray, v_ship: np.ndarray, launch_offsets_min: list,
                      horizons: list[int], c_state, wx_base: dict,
                      wx_of_tau=None, t0_base=None) -> list:
    """构造起飞时刻网格(CV 合成; 真实由 step8 对每个 τ 用 t≤τ 的 AIS 给预测点+状态)。
      P0           : 基准时刻(τ=0)的船位(本地米)。
      v_ship       : 船速矢量 m/s(CV)。
      launch_offsets_min: 候选 τ 相对基准的偏移(分钟), 如 [0,10,20,30]。
      c_state      : 标量(各 τ 同一可观测状态)或可调用 c_state(tau_min)->str(状态随 τ 变)。
      wx_base      : 基准天气; wx_of_tau(tau_min)->dict 给时变天气(默认全程同一)。
    返回 [LaunchOption,...]。每个 τ: P_launch=P0+v·(τ·60), pred_by_h[h]=P_launch+v·(h·60),
    c(τ)=可观测状态(无泄漏), 用作 ξ 模糊集索引 𝒫_{h,c(τ)}。
    """
    P0 = np.asarray(P0, float); v_ship = np.asarray(v_ship, float)
    opts = []
    for tau in launch_offsets_min:
        P_launch = P0 + v_ship * (float(tau) * 60.0)
        cs = c_state(tau) if callable(c_state) else c_state
        sp = ShipPrediction.from_cv(P_launch, v_ship, horizons, c_state=cs, t0=t0_base)
        sp.tau_min = float(tau)
        wx_tau = wx_of_tau(tau) if callable(wx_of_tau) else dict(wx_base)
        # 更新 审计 P0-10(任务 #7): 把该 τ 的天气挂到 ship 上(sp.wx_tau), 使求解器逐 τ 列生成/闭合
        # 时【用该 τ 的风浪场】—— τ 不再只选起飞船位/状态, 也选【作业风浪窗】(时变天气透传到定价/闭合)。
        # 求解器侧 _wx_of_ship/_wx_of_route 读取此字段; 缺失则退回全局 wx(向后兼容, 单一天气场字节一致)。
        sp.wx_tau = dict(wx_tau)
        sp.weather_by_h = {
            (int(hh) if float(hh).is_integer() else float(hh)):
            (dict(wx_of_tau(float(tau) + float(hh))) if callable(wx_of_tau) else dict(wx_tau))
            for hh in horizons
        }
        sp._weather_contract_strict = bool(callable(wx_of_tau))
        opts.append(LaunchOption(tau_min=float(tau), ship=sp, wx=dict(wx_tau)))
    return opts


# =============================================================================
# 2. 路由(一个候选架次 = 一条访问多台风机的路径)
# =============================================================================
@dataclass
class Route:
    """一条候选路由: 母船起飞 → 顺序巡检 turbines → 返回移动母船。

    决策含义: 选哪些风机、什么顺序、何时回收(h)。本类只持有"选哪些/什么顺序",
    回收时长 h 在 DRCC 评估时作为决策逐个尝试(见 route_drcc_feasible)。
    """
    rid: int
    turbines: list                      # list[M.Turbine], 按访问顺序(已 setattr .local)
    ship: ShipPrediction                # 该出动的船位预测
    fixed_h: Optional[int] = None       # 若已定 h(求解后回填), 否则 None

    def turbine_ids(self) -> list[str]:
        return [t.tid for t in self.turbines]

    def n_stops(self) -> int:
        return len(self.turbines)


# =============================================================================
# 3. 路由的能耗/时间(序列相关; 仅返程依赖 ξ) —— model.md §12.2
# =============================================================================
def _leg_ground_speed(p: M.Params, w_cruise: float, wdir: float,
                      p_from: np.ndarray, p_to: np.ndarray) -> tuple:
    """返回 (vg, power_W): 与 leg_airspeed_feasibility 同口径的风三角地速 + 视速功率(更新 P0-6)。
    强横风(>v_cr)下飞 v_air_max 保持航迹, 地速与功率随之一致 —— 不再出现"可行性按 v_air_max、ET 按 v_cr"。"""
    w_vec = M.wind_vector_from(w_cruise, wdir)
    e = p_to - p_from
    _ok, _v_eff, vg, power = M.leg_kinematics(p, w_vec, e)
    return vg, power


def _wind_of(wx: dict, default_w10=6.7, default_wdir=230.0):
    """从 wx dict 取 (10m 风速, 来向), 带 NaN 兜底。"""
    w10 = wx.get("wind10") if wx else None
    wdir = wx.get("wind_dir_from") if wx else None
    if w10 is None or (isinstance(w10, float) and math.isnan(w10)): w10 = default_w10
    if wdir is None or (isinstance(wdir, float) and math.isnan(wdir)): wdir = default_wdir
    return float(w10), float(wdir)


def route_nominal_ET(route: Route, h: int, p: M.Params, wx: dict,
                     wind_delta=None, t_dock_s: float = 0.0) -> dict:
    """名义(ξ=0)下整条路由的能耗 E0[Wh]、时间 T0[s], 以及返程线性化所需量。

    wind_delta(更新 修复, 审计 P1-风敏感度): 可选 2D【10m 风矢量增量】(east,north, m/s)。
    给定时, 每条腿在解析出该腿风(全局 wx 或风机本地 wx_local)后、升尺度到巡航高度前,
    先把 10m 风矢量加上 wind_delta —— 这样 route_wind_sensitivity 的有限差分对
    【逐风机本地风】同样生效(旧口径只扰全局 wx, 一旦挂载 wx_local 风敏感度会错误地趋零)。
    正式口径: ``E0`` = 起飞爬升 + 巡航/巡检/返程 + 船尾伴飞，不含 ``E_dock``；
    最终下降只在 ``dock_reserve`` 中计量，完整第二层成本由调用方取 ``E0 + E_dock``。

    逐风机天气(model.md §15): 若风机对象带 `wx_local`(由 step15 --per-turbine 注入), 每段巡航腿用其
    【目的风机的本地风】、返程用【末端风机的本地风】算地速; 否则全程用代表性 wx(向后兼容)。

    返回 dict(E0, T0, d_ret0, g, v_ret, c_E, c_T, P_recover_pred,
             E_wait, T_flight, speed_feasible, max_required_airspeed_ms, speed_fail_leg)。
    新增(critique 必改 3/4):
      - 逐航段风三角【空速可行性】: 每段(巡航腿 + 返程)按该腿风矢量检查保持航迹所需空速
        ∈ [_, v_air_max], 横风>v_air_max 或逆风过强 → speed_feasible=False, speed_fail_leg 记腿号。
      - 船尾伴飞能耗(提前到达): E_escort = P_escort·max(0, h·60 − T_flight − t_dock)。
        注: 这是【名义近似】(基于名义到达时间), 非无条件鲁棒上界 —— 含风不确定性时顺风可能使
        到达更早、等待更久, 故名义等待不必然是上界。严格鲁棒上界须用【最早可能到达时间 t_A^earliest】
        算最大等待 T^wait,max=h·60−t_A^earliest(见 model.md §6.2; 留作收紧选项)。
    """
    h_s = 60.0 * float(h)
    w10_g, wdir_g = _wind_of(wx)             # 代表性(farm 级)风, 作兜底
    _wd = None if wind_delta is None else np.asarray(wind_delta, float)

    def _leg_wc(turbine):
        """该腿用的巡航高度风(优先目的风机本地风; 统一施加 wind_delta 后再升尺度)。
        返回 (风速 m/s @巡航高度, 来向 deg)。"""
        loc = getattr(turbine, "wx_local", None)
        if loc is None:
            w10, wdir = w10_g, wdir_g
        else:
            w10, wdir = _wind_of(loc, w10_g, wdir_g)
        if _wd is not None:                          # 10m 风矢量增量(全局与本地风同扰)
            w10, wdir = _speed_dir_from_vec(_wind_vec(float(w10), float(wdir)) + _wd)
        return M.wind_at_height(w10, p.z_cruise, p.z0), wdir

    # 逐航段空速可行性累计(critique 必改 4)。更新(审计修复#8-回放#3): 拆分
    # 【出程/台间腿】与【返程腿】两个旗标 —— 回放在 ξ 实现后按【实际回收方向】重判返程腿,
    # 只需替换 ret_leg_ok, 出程侧结论(与 ξ 无关)可直接沿用。
    speed_feasible = True
    speed_ok_outbound = True
    ret_leg_ok = True
    max_req_air = 0.0
    max_req_air_out = 0.0
    speed_fail_leg = -1
    leg_air_records = []

    def _check_airspeed(leg_idx, w_speed, wdir, p_from, p_to, is_return=False):
        nonlocal speed_feasible, speed_ok_outbound, ret_leg_ok
        nonlocal max_req_air, max_req_air_out, speed_fail_leg
        e = np.asarray(p_to - p_from, float)
        length = float(np.linalg.norm(e))
        if length == 0.0:
            return
        ehat = e / length
        nhat = np.array([-ehat[1], ehat[0]], float)
        w_vec = M.wind_vector_from(w_speed, wdir)
        ok, V_req, vg_max = M.leg_airspeed_feasibility(p.v_cr, p.v_air_max, w_vec, e, v_air_min=p.v_air_min)
        leg_air_records.append(dict(
            leg_index=int(leg_idx), is_return=bool(is_return), length_m=length,
            direction=np.asarray(ehat, float), normal=np.asarray(nhat, float),
            wind_vector=np.asarray(w_vec, float),
            wind_along_ms=float(np.dot(w_vec, ehat)),
            wind_cross_signed_ms=float(np.dot(w_vec, nhat)),
            nominal_required_airspeed_ms=float(V_req), nominal_vg_max_ms=float(vg_max),
            nominal_feasible=bool(ok)))
        if V_req > max_req_air:
            max_req_air = V_req
        if (not is_return) and V_req > max_req_air_out:
            max_req_air_out = V_req
        if not ok:
            if is_return:
                ret_leg_ok = False
            else:
                speed_ok_outbound = False
            if speed_feasible:
                speed_feasible = False; speed_fail_leg = leg_idx

    # 起飞爬升只计一次；最终下降完整归入 dock_reserve。
    E_to, _E_land, T_to, _T_land = M.to_land_energy_time(p)
    E_land = 0.0; T_land = 0.0

    # 节点序列(本地米): 起飞点 → 各风机 → (返程目标=预测回收点)
    pts = [route.ship.P_launch] + [t.local for t in route.turbines]

    # 巡航腿(起飞→t1, t1→t2, ...): 第 i 段目的地为 route.turbines[i], 用其本地风
    E_cruise = 0.0; T_cruise = 0.0; E_insp = 0.0; T_insp = 0.0
    for i in range(len(pts) - 1):
        d = float(np.linalg.norm(pts[i + 1] - pts[i]))
        wc_i, wdir_i = _leg_wc(route.turbines[i])
        _check_airspeed(i, wc_i, wdir_i, pts[i], pts[i + 1])
        vg, pw_leg = _leg_ground_speed(p, wc_i, wdir_i, pts[i], pts[i + 1])
        E_cruise += pw_leg * (d / vg) / 3600.0
        T_cruise += d / vg
    for t in route.turbines:                 # 巡检(竖向爬升 + 绕飞悬停/慢飞)
        dz = M.insp_vertical_span(t, p.z_cruise)
        if getattr(p, "use_zeng", False):
            # 更新: 巡检爬升用 P_zeng(0)+提升做功; 绕飞巡检用 P_zeng(v_orbit)(≪悬停)
            P_up = M.P_zeng(0.0, p) + 7.27 * 9.81 * p.v_z
            E_insp += P_up * dz / p.v_z / 3600.0 + M.P_zeng(p.v_orbit, p) * p.tau_insp / 3600.0
        else:
            E_insp += p.P_climb * dz / p.v_z / 3600.0 + p.P_hov * p.tau_insp / 3600.0
        T_insp += dz / p.v_z + p.tau_insp

    # 返程(最后一台风机 → 预测回收点 P̂_v(τ+h)); 随 ξ 的入口; 用末端风机本地风
    q_last = route.turbines[-1].local
    P_rec = route.ship.predicted_at(float(h))   # 支持细网格 h(双层网格决策层)
    diff = q_last - P_rec
    d_ret0 = float(np.linalg.norm(diff))
    g = (-diff / d_ret0 if d_ret0 > 0.0 else np.zeros(2, dtype=float))  # binary64 exact normalization
    wc_r, wdir_r = _leg_wc(route.turbines[-1])
    _check_airspeed(len(pts) - 1, wc_r, wdir_r, q_last, P_rec, is_return=True)   # 返程腿空速可行性
    v_ret, pw_ret = _leg_ground_speed(p, wc_r, wdir_r, q_last, P_rec)
    E_ret0 = pw_ret * (d_ret0 / v_ret) / 3600.0
    T_ret0 = d_ret0 / v_ret

    T_fixed = T_to + T_cruise + T_insp + T_land
    E_fixed = E_to + E_cruise + E_insp + E_land
    T_flight = T_fixed + T_ret0                  # 名义飞行时间(不含伴飞/对接)
    # 船尾伴飞能量关于返程距离是两条仿射函数的最大值：
    #   无伴飞分支 E_f + P_ret*d/v；伴飞分支再加 P_esc*(C-d/v)。
    # 机会约束层将分别约束两个分支并在 eps_E 内部做 Bonferroni 二分，避免折点线性化漏保。
    escort = escort_state(route, h, p, wx, wind_delta=_wd)
    escort_available_s = h_s - T_fixed - max(float(t_dock_s), 0.0)
    E_branch_noescort = E_fixed + E_ret0
    E_branch_escort_affine = (E_branch_noescort
                              + escort["power_W"] * (escort_available_s - T_ret0) / 3600.0)
    E0 = max(E_branch_noescort, E_branch_escort_affine)
    T_escort = max(0.0, escort_available_s - T_ret0)
    E_escort = max(0.0, E0 - E_branch_noescort)
    E_flight = E_branch_noescort
    T0 = T_flight
    c_E_noescort = pw_ret / v_ret / 3600.0
    c_E_escort = (pw_ret - float(escort["power_W"])) / v_ret / 3600.0
    c_E = c_E_escort if T_escort > 0.0 else c_E_noescort
    c_T = 1.0 / v_ret                            # dT_ret/d(d_ret), s/m
    time_flight_nom_s = float(T_to + T_cruise + T_ret0 + T_land)
    time_inspection_s = float(T_insp)
    assert abs(T0 - (time_flight_nom_s + time_inspection_s)) <= TIME_TOL_S
    return dict(E0=E0, E_flight=E_flight, E_escort=E_escort,
                E_fixed_nonreturn_Wh=float(E_fixed),
                E_wait=E_escort, T_escort=T_escort,
                h_s=float(h_s), time_flight_nom_s=time_flight_nom_s,
                time_inspection_s=time_inspection_s,
                E_branch_noescort_Wh=float(E_branch_noescort),
                E_branch_escort_affine_Wh=float(E_branch_escort_affine),
                c_E_noescort=float(c_E_noescort), c_E_escort=float(c_E_escort),
                escort_power_W=float(escort["power_W"]),
                escort_available_s=float(escort_available_s),
                escort_required_airspeed_ms=float(escort["required_airspeed_ms"]),
                escort_speed_feasible=bool(escort["feasible"]),
                T0=T0, d_ret0=d_ret0, g=g, v_ret=v_ret,
                c_E=c_E, c_T=c_T, P_recover_pred=P_rec,
                T_flight=T_flight, t_dock_s_used=float(max(float(t_dock_s), 0.0)),
                speed_feasible=speed_feasible,
                speed_ok_outbound=speed_ok_outbound, ret_leg_ok=ret_leg_ok,
                max_required_airspeed_ms=float(max_req_air),
                max_required_airspeed_out_ms=float(max_req_air_out),
                speed_fail_leg=speed_fail_leg,
                leg_air_records=leg_air_records)


def route_energy_time(route: Route, h: int, xi: np.ndarray,
                      p: M.Params, wx: dict, detail: bool = False,
                      wind_delta=None, t_dock_s: float = 0.0):
    """给定误差实现 ξ(及可能已扰动的天气 wx), 整条路由的实现 (E[Wh], T[s])。
    用真实(非线性)返程距离; 供回放/校核线性化精度; DRCC 重构本身用 route_nominal_ET 的一阶量。

    相对旧盘旋口径的三处统一:
      ① 返程腿风与 route_nominal_ET 同口径(优先末端风机本地风);
      ② 船尾伴飞按【实现到达时刻】重算，ξ 改变返程时长后伴飞时长同步变化;
      ③ detail=True 返回 E_escort，并保留 E_wait 兼容别名，供回放检查能量与空速。
    注意: 返回的 T 一律为【飞行时间】(不含等待/对接), 与规划侧 b_T = 60h − T0 − t_dock 同口径;
    对接储备由调用方(step15.replay_routes)按实现海况另行加入。"""
    h_s = 60.0 * float(h)
    _wd = None if wind_delta is None else np.asarray(wind_delta, float)
    nom = route_nominal_ET(route, h, p, wx, wind_delta=_wd, t_dock_s=t_dock_s)
    q_last = route.turbines[-1].local
    P_rec = nom["P_recover_pred"] + np.asarray(xi, float)
    d_ret = float(np.linalg.norm(q_last - P_rec))
    # 返程腿风: 与 route_nominal_ET._leg_wc 同口径(优先末端风机本地风; 兜底全局)。
    # 更新(审计修复#8-回放#4): wind_delta 在【解析出全局/本地风之后、升尺度之前】统一施加
    # —— 风机挂 wx_local 时回放风扰动不再被本地天气绕过(与规划侧 wind_delta 通道同一位置)。
    last = route.turbines[-1]
    loc = getattr(last, "wx_local", None)
    w10_g, wdir_g = _wind_of(wx)
    w10, wdir = (_wind_of(loc, w10_g, wdir_g) if loc is not None else (w10_g, wdir_g))
    if _wd is not None:
        w10, wdir = _speed_dir_from_vec(_wind_vec(float(w10), float(wdir)) + _wd)
    w_cruise = M.wind_at_height(w10, p.z_cruise, p.z0)
    vg, pw = _leg_ground_speed(p, w_cruise, wdir, q_last, P_rec)            # 扰动后返程
    vg0, pw0 = _leg_ground_speed(p, w_cruise, wdir, q_last, nom["P_recover_pred"])  # 名义返程(同口径, 供扣除)
    # 飞行段: 名义扣去名义返程、加回实现返程；伴飞按实现到达时刻和实现风重算。
    E_fl = float(nom.get("E_flight", nom["E0"] - nom.get("E_escort", nom.get("E_wait", 0.0)))) \
        - pw0 * (nom["d_ret0"] / vg0) / 3600.0 + pw * (d_ret / vg) / 3600.0
    T = nom["T0"] - nom["d_ret0"] / vg0 + d_ret / vg
    escort = escort_state(route, h, p, wx, wind_delta=_wd)
    T_escort = max(0.0, h_s - T - max(float(t_dock_s), 0.0))
    E_escort = escort["power_W"] * T_escort / 3600.0
    E = E_fl + E_escort
    # 更新(审计修复#8-回放#3): 按【实际回收方向】(ξ 平移后的 P_rec)重判返程腿空速。
    # 旧口径 speed_feasible 沿用名义回收方向的检查 —— 横向 ξ 可能使实际返程所需空速
    # 超上限而回放发现不了。出程/台间腿与 ξ 无关, 沿用 nom 的 outbound 结论。
    ret_ok_real = True
    V_req_ret = 0.0
    if d_ret > 0.0:
        w_vec_ret = M.wind_vector_from(w_cruise, wdir)
        ret_ok_real, V_req_ret, _ = M.leg_airspeed_feasibility(
            p.v_cr, p.v_air_max, w_vec_ret, P_rec - q_last, v_air_min=p.v_air_min)
    speed_ok_real = bool(nom.get("speed_ok_outbound", nom["speed_feasible"]) and ret_ok_real)
    max_req_real = max(float(nom.get("max_required_airspeed_out_ms", 0.0)), float(V_req_ret))
    if detail:
        return dict(E=float(E), T=float(T), E_escort=float(E_escort), E_wait=float(E_escort),
                    T_escort=float(T_escort), d_ret=d_ret,
                    escort_required_airspeed_ms=float(escort["required_airspeed_ms"]),
                    escort_speed_feasible=bool(escort["feasible"]),
                    speed_feasible=speed_ok_real,
                    speed_feasible_nominal_dir=bool(nom["speed_feasible"]),
                    ret_leg_ok_realized=bool(ret_ok_real),
                    max_required_airspeed_ms=float(max_req_real))
    return E, T


def route_nominal_schedule(route: Route, h: float, p: M.Params, wx: dict,
                           *, t_dock_s: float = 0.0, wind_delta=None) -> dict:
    """Return the exact nominal phase timeline used by the physical route model.

    The schedule is a visualization/export contract, not a second kinematic model.
    It uses the same per-leg wind resolution, ground-speed calculation, inspection
    duration, predicted recovery point, escort interval and dock reserve as
    :func:`route_nominal_ET`.  Times are seconds relative to sortie launch.
    """
    _wd = None if wind_delta is None else np.asarray(wind_delta, float)

    def _leg_wc(turbine):
        loc = getattr(turbine, "wx_local", None)
        w10_g, wdir_g = _wind_of(wx)
        w10, wdir = (_wind_of(loc, w10_g, wdir_g)
                      if loc is not None else (w10_g, wdir_g))
        if _wd is not None:
            w10, wdir = _speed_dir_from_vec(
                _wind_vec(float(w10), float(wdir)) + _wd)
        return M.wind_at_height(w10, p.z_cruise, p.z0), wdir

    total_s = float(h) * 60.0
    dock_s = max(float(t_dock_s), 0.0)
    dock_start_s = total_s - dock_s
    if dock_start_s < -1e-9:
        raise ValueError("dock reserve exceeds sortie duration")

    P0 = np.asarray(route.ship.P_launch, float)
    waypoints = [dict(t_s=0.0, x_m=float(P0[0]), y_m=float(P0[1]),
                      phase="launch")]
    path = [P0.tolist()]
    inspections = []
    phases = []
    t = 0.0

    _E_to, _E_land, T_to, _T_land = M.to_land_energy_time(p)
    if T_to > 0:
        phases.append(dict(phase="takeoff", start_s=t, end_s=t + float(T_to)))
        t += float(T_to)
        waypoints.append(dict(t_s=t, x_m=float(P0[0]), y_m=float(P0[1]),
                              phase="takeoff_complete"))

    cur = P0
    for idx, turb in enumerate(route.turbines):
        q = np.asarray(turb.local, float)
        wc, wdir = _leg_wc(turb)
        vg, _pw = _leg_ground_speed(p, wc, wdir, cur, q)
        leg_s = float(np.linalg.norm(q - cur)) / max(float(vg), 1e-9)
        phases.append(dict(phase="outbound_leg", leg_index=int(idx),
                           start_s=t, end_s=t + leg_s, turbine_id=str(turb.tid)))
        t += leg_s
        waypoints.append(dict(t_s=t, x_m=float(q[0]), y_m=float(q[1]),
                              phase="arrive_turbine", turbine_id=str(turb.tid)))
        path.append(q.tolist())
        dz = M.insp_vertical_span(turb, p.z_cruise)
        insp_s = float(dz / p.v_z + p.tau_insp)
        inspections.append(dict(turbine_id=str(turb.tid), start_s=t, end_s=t + insp_s))
        phases.append(dict(phase="inspection", start_s=t, end_s=t + insp_s,
                           turbine_id=str(turb.tid)))
        t += insp_s
        waypoints.append(dict(t_s=t, x_m=float(q[0]), y_m=float(q[1]),
                              phase="inspection_complete", turbine_id=str(turb.tid)))
        cur = q

    P_rec = np.asarray(route.ship.predicted_at(float(h)), float)
    wc_r, wdir_r = _leg_wc(route.turbines[-1])
    vg_r, _pw_r = _leg_ground_speed(p, wc_r, wdir_r, cur, P_rec)
    ret_s = float(np.linalg.norm(P_rec - cur)) / max(float(vg_r), 1e-9)
    phases.append(dict(phase="return_leg", start_s=t, end_s=t + ret_s))
    t += ret_s
    waypoints.append(dict(t_s=t, x_m=float(P_rec[0]), y_m=float(P_rec[1]),
                          phase="arrive_recovery_target"))
    path.append(P_rec.tolist())

    if t > dock_start_s + 1e-6:
        raise ValueError(
            f"nominal route reaches dock phase late: arrival={t:.6f}s, "
            f"dock_start={dock_start_s:.6f}s")
    escort_start_s = t
    if dock_start_s > t + 1e-9:
        phases.append(dict(phase="escort", start_s=t, end_s=dock_start_s))
        waypoints.append(dict(t_s=dock_start_s, x_m=float(P_rec[0]), y_m=float(P_rec[1]),
                              phase="escort_complete"))
    if dock_s > 0:
        phases.append(dict(phase="dock", start_s=dock_start_s, end_s=total_s))
    waypoints.append(dict(t_s=total_s, x_m=float(P_rec[0]), y_m=float(P_rec[1]),
                          phase="touchdown"))
    return dict(
        result_contract="route-nominal-schedule", h_s=total_s,
        predicted_recovery_point_m=P_rec.tolist(), waypoints=waypoints,
        inspections=inspections, phases=phases, path_m=path,
        return_arrival_s=float(escort_start_s), escort_start_s=float(escort_start_s),
        dock_start_s=float(dock_start_s), touchdown_s=float(total_s),
        t_dock_s=float(dock_s))


# =============================================================================
# 4. 决策依赖 DRCC: 逐 h 评估, 选最优回收时长 —— model.md §12.3 / §14(多源不确定性)
# =============================================================================
@dataclass
class WeatherUncertainty:
    """按预测提前量标定的天气残差矩。

    ``wind_bias``/``wind_cov`` 描述二维风矢量残差，供航段时间、能量和可控性约束使用；
    ``wind_speed_bias``/``wind_speed_std`` 描述标量风速大小残差
    ``|w_true|-|w_nominal|``，只供着舰风门和对接风上下侧使用。二者不能互相替代。
    机制实验可由相邻再分析差分构造 persistence proxy；正式实验必须由 forecast-truth
    配对残差构造，并把 ``formal_eligible`` 置为 True。
    """
    wind_cov: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    wind_bias: np.ndarray = field(default_factory=lambda: np.zeros(2))
    wind_speed_std: float = 0.0
    wind_speed_bias: float = 0.0
    hs_std: float = 0.0
    hs_bias: float = 0.0
    n_samples: int = 0
    source: str = "unknown"
    formal_eligible: bool = False
    regularization_weight: float = 0.0
    vector_floor_ms: float = 0.0
    speed_floor_ms: float = 0.0
    hs_floor_m: float = 0.0


@dataclass
class WeatherAmbiguity:
    """按回收提前量 h 索引的天气残差矩集合。"""
    by_h: dict = field(default_factory=dict)
    horizons: list = field(default_factory=list)
    source: str = "unknown"
    formal_eligible: bool = False
    predictor: str = "unknown"
    predictor_contract: str = "unknown"
    timestamp_epoch_contract: str = "unknown"
    truth_contract: str = "unknown"
    weather_data_contract: str = "unknown"
    sample_overlap_policy: str = "unknown"
    purge_min: float = 0.0
    source_path: str = ""
    source_sha256: str = ""
    weather_source_sha256: str = ""
    xi_train_source_sha256: str = ""

    def get(self, h: float) -> "WeatherUncertainty":
        """Return the exact binary64 horizon cell; never nearest-neighbour snap.

        Weather uncertainty is part of the certified finite instance.  A missing
        horizon is missing model support, not permission to borrow a nearby cell
        or silently drop uncertainty.
        """
        if not self.horizons:
            raise KeyError("WeatherAmbiguity contains no horizon support")
        hq = float(h)
        matches = [hh for hh in self.horizons if float(hh).hex() == hq.hex()]
        if len(matches) != 1:
            raise KeyError(f"weather uncertainty missing exact horizon h={hq!r}")
        hh = matches[0]
        # by_h may use an equivalent numeric key with a different Python scalar type.
        keys = [k for k in self.by_h if float(k).hex() == float(hh).hex()]
        if len(keys) != 1:
            raise KeyError(f"weather uncertainty mapping ambiguous/missing for h={hq!r}")
        return self.by_h[keys[0]]


def _regularized_cov2(samples: np.ndarray, floor_std: float,
                      shrinkage_equivalent_n: float = 30.0) -> tuple[np.ndarray, np.ndarray, float]:
    """返回 ``(mean,cov,lambda)``。

    旧实现逐分量硬截断 ``sigma=max(empirical,floor)``，在大量样本且经验波动很小时仍把
    协方差固定为 ``floor² I``，随后二维 Markov 半径被放大成全局关闭开关。这里改成可审计
    的有限样本收缩：经验协方差与各向同性目标做凸组合，样本越多收缩权重越小。floor 只定义
    正则化目标，不再作为每个分量的永久硬下限。
    """
    x = np.asarray(samples, float)
    x = x[np.all(np.isfinite(x), axis=1)]
    if len(x) < 2:
        return np.zeros(2), np.eye(2) * float(floor_std) ** 2, 1.0
    mu = np.mean(x, axis=0)
    cov = np.asarray(np.cov(x.T, ddof=1), float)
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    cov = (vecs * np.maximum(vals, 0.0)) @ vecs.T
    lam = min(1.0, float(shrinkage_equivalent_n) / max(float(len(x)), 1.0))
    target_var = max(float(np.trace(cov)) / 2.0, float(floor_std) ** 2)
    reg = (1.0 - lam) * cov + lam * target_var * np.eye(2)
    return np.asarray(mu, float), np.asarray(reg, float), float(lam)


def _regularized_scalar(samples: np.ndarray, floor_std: float,
                        shrinkage_equivalent_n: float = 30.0) -> tuple[float, float, float]:
    x = np.asarray(samples, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return 0.0, float(floor_std), 1.0
    mu = float(np.mean(x))
    var = float(np.var(x, ddof=1))
    lam = min(1.0, float(shrinkage_equivalent_n) / max(float(len(x)), 1.0))
    reg_var = (1.0 - lam) * max(var, 0.0) + lam * max(var, float(floor_std) ** 2)
    return mu, math.sqrt(max(reg_var, 0.0)), float(lam)


def weather_ambiguity_from_series(wx_df, horizons, scale: float = 1.0,
                                  floor_wind_ms: float = 0.5,
                                  floor_hs_m: float = 0.05,
                                  floor_wind_speed_ms: float = 0.25,
                                  source: str = "adjacent_reanalysis_difference_proxy") -> "WeatherAmbiguity":
    r"""由时序构造 persistence-proxy 天气残差矩。

    风矢量残差 ``[du,dv]``、标量风速残差 ``d|w|`` 和波高残差 ``dHs`` 分开估计；
    保留偏置、完整二维协方差及样本量。对提前量 h 仍采用 ``sqrt(h/native_step)`` 的机制
    缩放。该函数的默认来源不是 forecast-truth，因此 ``formal_eligible=False``。
    """
    by_h = {}
    horizons = sorted(int(h) for h in horizons)
    step_min = 60.0
    vec_res = np.empty((0, 2), float)
    speed_res = np.empty(0, float)
    hs_res = np.empty(0, float)
    try:
        if wx_df is not None and len(wx_df) >= 8:
            df = wx_df.copy()
            if "time" in getattr(df, "columns", []):
                df = df.sort_values("time")
                tv = np.asarray(pd.to_datetime(df["time"], errors="coerce", utc=True).values,
                                dtype="datetime64[m]")
                dt = np.median(np.diff(tv).astype(float))
                if np.isfinite(dt) and dt > 0:
                    step_min = float(dt)
            else:
                df = df.sort_index()
                if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 1:
                    dt = np.median(np.diff(df.index.values.astype("datetime64[m]")).astype(float))
                    if np.isfinite(dt) and dt > 0:
                        step_min = float(dt)
            wcol = next((c for c in ("wind10_ms", "wind10") if c in df.columns), None)
            dcol = next((c for c in ("wind_dir_from_deg", "wind_dir_from", "wind_dir")
                         if c in df.columns), None)
            if wcol is not None and dcol is not None:
                vals = pd.to_numeric(df[wcol], errors="coerce").to_numpy(float)
                dirs = pd.to_numeric(df[dcol], errors="coerce").to_numpy(float)
                good = np.isfinite(vals) & np.isfinite(dirs)
                vecs = np.array([_wind_vec(float(v), float(d)) for v, d in zip(vals[good], dirs[good])])
                speeds = vals[good]
                if len(vecs) >= 5:
                    vec_res = np.diff(vecs, axis=0)
                    speed_res = np.diff(speeds)
            hcol = next((c for c in ("Hs_m", "Hs") if c in df.columns), None)
            if hcol is not None:
                hv = pd.to_numeric(df[hcol], errors="coerce").to_numpy(float)
                hv = hv[np.isfinite(hv)]
                if len(hv) >= 5:
                    hs_res = np.diff(hv)
    except Exception as exc:
        log.warning("天气残差代理估计失败，使用正则化占位: %s", exc)

    vec_mu, vec_cov, lam_vec = _regularized_cov2(vec_res, floor_wind_ms)
    speed_mu, speed_std, lam_speed = _regularized_scalar(speed_res, floor_wind_speed_ms)
    hs_mu, hs_std, lam_hs = _regularized_scalar(hs_res, floor_hs_m)
    n = int(min(len(vec_res), len(speed_res))) if len(vec_res) and len(speed_res) else 0
    reg_weight = float(max(lam_vec, lam_speed, lam_hs))
    for h in horizons:
        ratio = max(float(h), 1e-6) / max(step_min, 1e-6)
        f_std = math.sqrt(ratio) * float(scale)
        f_bias = ratio * float(scale)
        by_h[int(h)] = WeatherUncertainty(
            wind_cov=np.asarray(vec_cov, float) * f_std * f_std,
            wind_bias=np.asarray(vec_mu, float) * f_bias,
            wind_speed_std=float(speed_std) * f_std,
            wind_speed_bias=float(speed_mu) * f_bias,
            hs_std=float(hs_std) * f_std,
            hs_bias=float(hs_mu) * f_bias,
            n_samples=n,
            source=str(source),
            formal_eligible=False,
            regularization_weight=reg_weight,
            vector_floor_ms=float(floor_wind_ms),
            speed_floor_ms=float(floor_wind_speed_ms),
            hs_floor_m=float(floor_hs_m),
        )
    return WeatherAmbiguity(by_h=by_h, horizons=horizons, source=str(source),
                            formal_eligible=False)


def _strict_weather_bool(value, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"{name} must be literal True/False, got {value!r}")


def _weather_cov2_binary64_psd(see: float, sen: float, snn: float) -> bool:
    vals = tuple(float(x) for x in (see, sen, snn))
    if not all(math.isfinite(x) for x in vals):
        return False
    see, sen, snn = (Fraction.from_float(x) for x in vals)
    return bool(see >= 0 and snn >= 0 and see * snn - sen * sen >= 0)


def weather_ambiguity_from_moments_csv(path: Path | str, horizons=None,
                                       formal: bool = True) -> "WeatherAmbiguity":
    """Load no-leak real-history weather residual moments.

    Formal semantics are fail-closed: exact binary64 horizon support, train-only
    moments, inherited non-overlap/purge provenance, finite moments, and an exact
    binary64-as-real PSD check of the 2-D wind covariance.  No adjacent-difference
    or Gaussian fallback is performed here.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {
        "h_min", "n", "wind_bias_e_ms", "wind_bias_n_ms",
        "wind_sigma_ee", "wind_sigma_en", "wind_sigma_nn",
        "wind_speed_bias_ms", "wind_speed_std_ms", "hs_bias_m", "hs_std_m",
        "predictor", "predictor_contract", "timestamp_epoch_contract",
        "truth_contract", "weather_data_contract", "moments_source",
        "sample_overlap_policy", "purge_min", "valid_for_formal",
        "weather_source_sha256", "xi_train_source_sha256",
    }
    miss = sorted(required - set(df.columns))
    if miss:
        raise ValueError(f"weather moments missing formal columns: {miss}")
    if df.empty:
        raise ValueError("weather moments file is empty")
    numeric = ["h_min", "n", "wind_bias_e_ms", "wind_bias_n_ms", "wind_sigma_ee",
               "wind_sigma_en", "wind_sigma_nn", "wind_speed_bias_ms",
               "wind_speed_std_ms", "hs_bias_m", "hs_std_m", "purge_min"]
    num = df[numeric].apply(pd.to_numeric, errors="coerce")
    if not bool(np.isfinite(num.to_numpy(float)).all()):
        raise ValueError("weather moments contain non-finite numeric values")
    nvals = num["n"].to_numpy(float)
    if any((not float(v).is_integer()) or v < 2 for v in nvals):
        raise ValueError("weather moments n must be exact integers >=2")
    if bool((num[["wind_speed_std_ms", "hs_std_m", "wind_sigma_ee", "wind_sigma_nn"]] < 0).any().any()):
        raise ValueError("weather moments contain negative std/variance")
    allowed = {float(h).hex(): int(h) for h in M.XI_FORMAL_HORIZON_GRID_MIN}
    raw_h = [float(x) for x in num["h_min"].to_numpy(float)]
    bad_h = [h for h in raw_h if h.hex() not in allowed]
    if bad_h:
        raise ValueError(f"weather moments contain off-grid horizons: {bad_h[:5]}")
    canonical_h = [allowed[h.hex()] for h in raw_h]
    if len(set(canonical_h)) != len(canonical_h):
        raise ValueError("weather moments contain duplicate horizon cells")
    if horizons is not None:
        requested = [float(h) for h in horizons]
        requested_hex = {h.hex() for h in requested}
        got_hex = {float(h).hex() for h in canonical_h}
        if requested_hex != got_hex:
            raise ValueError(f"weather horizon support mismatch: requested={requested}, got={canonical_h}")
    max_h = float(max(canonical_h))
    if not bool((num["purge_min"] >= max_h).all()):
        raise ValueError("weather moments purge_min must cover maximum horizon")
    if formal:
        for col in ("weather_source_sha256", "xi_train_source_sha256"):
            vals = set(df[col].astype(str))
            if len(vals) != 1:
                raise ValueError(f"weather moments {col} must be constant")
            value = next(iter(vals))
            if len(value) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in value):
                raise ValueError(f"weather moments {col} is not a SHA-256 hex digest")
        if not all(_strict_weather_bool(x, "valid_for_formal") for x in df["valid_for_formal"]):
            raise ValueError("weather moments include valid_for_formal=False")
        if set(df["predictor"].astype(str)) != {"weather_speed_primary_coherent_noleak"}:
            raise ValueError("formal weather moments require predictor=weather_speed_primary_coherent_noleak")
        if set(df["predictor_contract"].astype(str)) != {WEATHER_PREDICTOR_CONTRACTS["weather_speed_primary_coherent_noleak"]}:
            raise ValueError("weather predictor contract mismatch")
        if set(df["timestamp_epoch_contract"].astype(str)) != {WEATHER_TIMESTAMP_EPOCH_CONTRACT}:
            raise ValueError("weather timestamp epoch contract mismatch")
        if set(df["truth_contract"].astype(str)) != {WEATHER_TRUTH_CONTRACT}:
            raise ValueError("weather truth contract mismatch")
        if set(df["weather_data_contract"].astype(str)) != {WEATHER_FORMAL_DATA_CONTRACT}:
            raise ValueError("weather data contract mismatch")
        if set(df["moments_source"].astype(str)) != {"train"}:
            raise ValueError("formal weather moments must be train-only")
        if set(df["sample_overlap_policy"].astype(str)) != {"weather_timeline_global_nonoverlap"}:
            raise ValueError("formal weather moments must use global non-overlap sampling on the weather timeline")
    by_h = {}
    for i, row in df.reset_index(drop=True).iterrows():
        see = float(row["wind_sigma_ee"]); sen = float(row["wind_sigma_en"]); snn = float(row["wind_sigma_nn"])
        if not _weather_cov2_binary64_psd(see, sen, snn):
            raise ValueError(f"weather covariance at h={canonical_h[i]} is not binary64-as-real PSD")
        cov = np.array([[see, sen], [sen, snn]], dtype=float)
        cell = WeatherUncertainty(
            wind_cov=cov,
            wind_bias=np.array([float(row["wind_bias_e_ms"]), float(row["wind_bias_n_ms"])], dtype=float),
            wind_speed_std=float(row["wind_speed_std_ms"]),
            wind_speed_bias=float(row["wind_speed_bias_ms"]),
            hs_std=float(row["hs_std_m"]), hs_bias=float(row["hs_bias_m"]),
            n_samples=int(float(row["n"])),
            source="real-history-weather-speed-primary-coherent-noleak-residuals",
            formal_eligible=bool(formal), regularization_weight=0.0,
            vector_floor_ms=0.0, speed_floor_ms=0.0, hs_floor_m=0.0,
        )
        by_h[canonical_h[i]] = cell
    return WeatherAmbiguity(
        by_h=by_h, horizons=sorted(by_h), source="real-history-weather-speed-primary-coherent-noleak-residuals",
        formal_eligible=bool(formal), predictor="weather_speed_primary_coherent_noleak",
        predictor_contract=WEATHER_PREDICTOR_CONTRACTS["weather_speed_primary_coherent_noleak"],
        timestamp_epoch_contract=WEATHER_TIMESTAMP_EPOCH_CONTRACT,
        truth_contract=WEATHER_TRUTH_CONTRACT, weather_data_contract=WEATHER_FORMAL_DATA_CONTRACT,
        sample_overlap_policy="weather_timeline_global_nonoverlap", purge_min=float(num["purge_min"].min()),
        source_path="", source_sha256=str(M.sha256_file(path) or ""),
        weather_source_sha256=str(df["weather_source_sha256"].iloc[0]),
        xi_train_source_sha256=str(df["xi_train_source_sha256"].iloc[0]),
    )


def _resolve_weather_unc(weather_unc, h: float):
    """把 weather_unc 解析成【当前 h 的】WeatherUncertainty: 支持 None / WeatherUncertainty(固定) / WeatherAmbiguity(按 h)。"""
    if weather_unc is None:
        return None
    if isinstance(weather_unc, WeatherAmbiguity):
        return weather_unc.get(h)
    return weather_unc


def _wind_vec(speed: float, dir_from_deg: float) -> np.ndarray:
    """气象"来向"(0=N 顺时针) → 风速矢量(指向去向)的 (east,north) 分量, m/s。"""
    to_deg = (dir_from_deg + 180.0) % 360.0
    return np.array([speed * math.sin(math.radians(to_deg)),
                     speed * math.cos(math.radians(to_deg))])


def _speed_dir_from_vec(w: np.ndarray) -> tuple[float, float]:
    speed = float(np.hypot(w[0], w[1]))
    to_deg = math.degrees(math.atan2(w[0], w[1])) % 360.0
    return speed, (to_deg - 180.0) % 360.0


def _ship_velocity_at(ship: "ShipPrediction", h_min: float) -> np.ndarray:
    """Return the predicted vessel ground-velocity vector at recovery.

    Prefer the CV predictor's stored velocity.  Otherwise use a centred finite
    difference of the *predicted* trajectory only, so no realised future AIS is
    leaked into planning.
    """
    v = getattr(ship, "_v_ship", None)
    if v is not None:
        return np.asarray(v, float)
    dh = 0.25  # min; small enough for the decision grid, large enough for numerical stability
    keys = sorted(float(k) for k in ship.pred_by_h)
    if not keys:
        return np.zeros(2)
    h0 = max(float(h_min) - dh, keys[0])
    h1 = min(float(h_min) + dh, keys[-1])
    if h1 == h0:
        return np.zeros(2)
    dt = (h1 - h0) * 60.0
    return (np.asarray(ship.predicted_at(h1), float) - np.asarray(ship.predicted_at(h0), float)) / dt


def escort_state(route: Route, h: float, p: M.Params, wx: dict,
                 wind_delta=None) -> dict:
    """Nominal/realised stern-follow state for the pre-landing loiter interval.

    In ``stern_follow`` mode the UAV keeps a fixed vessel-frame offset, hence its
    ground velocity equals the vessel velocity.  Required airspeed is
    ``v_ship - w_10m``.  This replaces the old fixed-speed circular loiter model.
    """
    mode = str(getattr(p, "escort_mode", "stern_follow")).strip().lower()
    if mode not in ("stern_follow", "legacy_loiter"):
        raise ValueError(f"unknown escort_mode={mode!r}")
    if mode == "legacy_loiter":
        v_req = float(getattr(p, "v_loiter", 13.0))
        power = M.P_zeng(v_req, p) if getattr(p, "use_zeng", False) else p.P_wait
        return dict(mode=mode, ship_velocity=np.zeros(2), wind_vector=np.zeros(2),
                    required_airspeed_ms=v_req, power_W=float(power), feasible=True)

    P_rec = route.ship.predicted_at(float(h))
    gwx = recovery_gate_wx(route, wx, P_rec)
    w10 = float(gwx.get("wind10", wx.get("wind10", 0.0)) or 0.0)
    wdir = float(gwx.get("wind_dir_from", wx.get("wind_dir_from", 0.0)) or 0.0)
    w_vec = _wind_vec(w10, wdir)
    if wind_delta is not None:
        w_vec = w_vec + np.asarray(wind_delta, float)
    v_ship = _ship_velocity_at(route.ship, float(h))
    v_air_vec = v_ship - w_vec
    v_req = float(np.linalg.norm(v_air_vec))
    feasible = _strict_finite_leq(v_req, p.v_air_max)
    if getattr(p, "use_zeng", False):
        power = float(getattr(p, "escort_power_beta", 1.0)) * M.P_zeng(v_req, p)
    else:
        # Legacy power curve is only calibrated around cruise; use the old wait
        # power with the same control multiplier rather than inventing a new law.
        power = float(getattr(p, "escort_power_beta", 1.0)) * float(p.P_wait)
    return dict(mode=mode, ship_velocity=v_ship, wind_vector=w_vec,
                required_airspeed_ms=v_req, power_W=float(power), feasible=feasible)


def route_wind_sensitivity(route: Route, h: int, p: M.Params, wx: dict,
                           delta: float = 0.15, t_dock_s: float = 0.0,
                           energy_branch: str = 'actual',
                           time_branch: str = 'actual') -> tuple[np.ndarray, np.ndarray]:
    """整条路由名义能耗/时间对【10m 风矢量】的一阶灵敏度 (∂E/∂w, ∂T/∂w), 各 2D(east,north)。
    数值中心差分: 扰动 10m 风矢量 ±delta m/s, 重算 route_nominal_ET。逆风降地速→能耗/时间↑。
    更新(采纳外部审计 6.5): 差分必须在与主判据【同一工作点】上做 —— 主判据的名义评估
    传入了 t_dock_s(等待窗 = 60h−T_flight−t_dock), 差分若用缺省 t_dock_s=0, 在
    60h−T_flight−t_dock≈0 的折点附近会落在 max(·) 的另一支, 使联合 SOC 的风灵敏度与
    实际 b_E/b_T 口径不一致; 现由 route_feasible_at_h 传入同一 t_dock。"""
    # 更新 修复(审计 P1-风敏感度): 改经 route_nominal_ET 的 wind_delta 通道做中心差分 ——
    # 增量对【每条腿实际使用的风】(全局 wx 或风机本地 wx_local)统一生效。旧口径只改全局
    # wx 字典, 一旦挂载逐风机天气(attach_per_turbine_weather), 各腿仍读 wx_local, 差分
    # 前后逐腿风不变 ⇒ 风敏感度错误地趋零(联合 SOC 的风保护随之失效)。无 wx_local 时
    # 本实现与旧口径逐位等价(矢量加增量 = 改 wind10/wind_dir_from)。
    # 注: E0 含 E_wait(名义到达近似), 故差分同时捕捉风经飞行时间对等待能耗的影响
    # (max(·) 折点处为割线斜率; 严格鲁棒上界见 route_nominal_ET 文档的最早到达选项)。
    aE = np.zeros(2); aT = np.zeros(2)
    for k in range(2):
        dvec = np.zeros(2); dvec[k] = delta
        np_ = route_nominal_ET(route, h, p, wx, wind_delta=+dvec, t_dock_s=t_dock_s)
        nn = route_nominal_ET(route, h, p, wx, wind_delta=-dvec, t_dock_s=t_dock_s)
        ekey = {'actual': 'E0', 'noescort': 'E_branch_noescort_Wh',
                'escort': 'E_branch_escort_affine_Wh',
                'fixed_nonreturn': 'E_fixed_nonreturn_Wh'}.get(str(energy_branch))
        if ekey is None:
            raise ValueError(f'unknown energy_branch={energy_branch!r}')
        aE[k] = (np_[ekey] - nn[ekey]) / (2 * delta)
        if str(time_branch) == "actual":
            tp = float(np_["T0"]); tn = float(nn["T0"])
        elif str(time_branch) == "fixed_nonreturn":
            tp = float(np_["T0"]) - float(np_["d_ret0"]) / max(float(np_["v_ret"]), 1e-12)
            tn = float(nn["T0"]) - float(nn["d_ret0"]) / max(float(nn["v_ret"]), 1e-12)
        else:
            raise ValueError(f"unknown time_branch={time_branch!r}")
        aT[k] = (tp - tn) / (2 * delta)
    return aE, aT


def _soc_margin(a: np.ndarray, b: float, cell: "M.XiCell", eps: float,
                risk_policy: RiskPolicy | None = None) -> float:
    """SOC margin under the explicit risk policy used by this route."""
    rp = _risk_policy_from_inputs(risk_policy)
    quad = max(float(a @ cell.Sigma @ a), 0.0)
    return b - (float(a @ cell.mu) + rp.one_sided(eps) * math.sqrt(quad))


def _soc_margin_joint(a_xi: np.ndarray, a_w: np.ndarray, b: float, cell: "M.XiCell",
                      wu: "WeatherUncertainty", eps: float,
                      risk_policy: RiskPolicy | None = None) -> float:
    """联合 SOC 余量: 堆叠不确定性 u=[ξ; Δw], a=[a_ξ; a_w], 矩 blkdiag(Σ_ξ,Σ_w)、均值 [μ_ξ;b_w]。
    b − [aᵀμ + κ(ε)·√(aᵀΣa)]; ξ⊥Δw 设分块对角。"""
    a = np.concatenate([a_xi, a_w])
    mu = np.concatenate([cell.mu, wu.wind_bias])
    Sig = np.zeros((4, 4))
    Sig[:2, :2] = cell.Sigma
    Sig[2:, 2:] = wu.wind_cov
    quad = max(float(a @ Sig @ a), 0.0)
    rp = _risk_policy_from_inputs(risk_policy)
    return b - (float(a @ mu) + rp.one_sided(eps) * math.sqrt(quad))


# =============================================================================
# 3a-bis.【更新 问题3】SOC 线性化的 2-D 精确几何校正(geo2d)
#   动因(E2 实测): q=0.8 强风窗 vp 判据 emp_viol=10.22% 贴线越过 2ε —— 根因是主线
#   把凸的返程距离 d(ξ)=‖v−ξ‖ 一阶泰勒成 d0+gᵀξ(切平面在凸函数下方 ⇒ 系统性【低估】),
#   σ⊥ 大 / d0 小 / 强风地速敏感时低估被放大。
#   校正原理(2-D 精确恒等式, 非近似): 记 L(ξ)=d0+gᵀξ(沿返程方向分量), ξ⊥=g⊥ᵀξ(垂向标量),
#       d(ξ) = √(L(ξ)² + ξ⊥²)          —— 精确, 无泰勒。
#   在两个高概率事件上取界(Bonferroni 拆 ε = ε₁+ε₂):
#       A(双侧, prob≥1−ε₁): |gᵀ(ξ−μ)| ≤ κ₂(ε₁)·σ_L  ⇒ L∈[L_lo,L_ub], |L|≤L_abs
#       B(双侧, prob≥1−ε₂): |g⊥ᵀ(ξ−μ)| ≤ κ₂(ε₂)·σ⊥  ⇒ |ξ⊥| ≤ t⊥
#   ⇒ P( d(ξ) > √(L_abs²+t⊥²) ) ≤ ε。κ₂ 用双侧 Vysochanskij–Petunin 4/(9κ²)
#   (与主线同一"单峰投影"假设族; κ<√(8/3) 时回退双侧 Chebyshev 1/κ²=分布无关)。
#   性质: ①严格上界(含 L<0 过冲情形, 由 |L|≤L_abs 覆盖); ②σ⊥→0,μ⊥→0 时退化为
#   "双侧 κ₂(ε₁) 的沿向盒"(比主线单侧 κ(ε) 略保守 —— 这是覆盖曲率的已知代价);
#   ③默认关闭(soc_correction='none'), 全部数字不受影响 —— 开启即换口径, 须整批重跑。
# =============================================================================
def _kappa_two_sided(eps: float, risk_policy: RiskPolicy | None = None) -> float:
    """Two-sided constant from the same immutable route risk contract."""
    return float(_risk_policy_from_inputs(risk_policy).two_sided(float(eps)))


def _risk_fraction_grid(fixed: float) -> tuple[float, ...]:
    """Deterministic candidate fractions for route-wise Bonferroni allocation.

    Every candidate is itself a valid allocation and the legacy fixed share is always included,
    so optimized mode can never be more conservative than the old fixed split.  This changes
    only the allocation *inside* an unchanged event budget; it never enlarges eps.
    """
    vals = {0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30,
            0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.94, 0.96,
            0.98, 0.99, max(0.001, min(0.999, float(fixed)))}
    return tuple(sorted(vals))


def _geo2d_dist_at_split(cell: "M.XiCell", d0: float, g: np.ndarray,
                         eps_along: float, eps_cross: float,
                         risk_policy: RiskPolicy | None = None) -> dict:
    if eps_along <= 0.0 or eps_cross <= 0.0:
        raise ValueError("geo2d Bonferroni parts must be positive")
    g = np.asarray(g, float)
    g_norm = float(np.linalg.norm(g))
    if g_norm == 0.0:
        raise ValueError("geo2d direction must be nonzero")
    g = g / g_norm
    g_perp = np.array([-g[1], g[0]])
    mL = float(g @ cell.mu); sL = math.sqrt(max(float(g @ cell.Sigma @ g), 0.0))
    mP = float(g_perp @ cell.mu); sP = math.sqrt(max(float(g_perp @ cell.Sigma @ g_perp), 0.0))
    k1 = _kappa_two_sided(float(eps_along), risk_policy)
    k2 = _kappa_two_sided(float(eps_cross), risk_policy)
    L_lo, L_ub = float(d0) + mL - k1 * sL, float(d0) + mL + k1 * sL
    L_abs = max(abs(L_lo), abs(L_ub))
    t_perp = abs(mP) + k2 * sP
    return dict(bound_m=math.hypot(L_abs, t_perp), eps_along=float(eps_along),
                eps_cross=float(eps_cross), mean_along_m=mL, mean_cross_m=mP,
                std_along_m=sL, std_cross_m=sP, kappa_along=k1, kappa_cross=k2)


def _geo2d_dist_bound_details(cell: "M.XiCell", eps: float, d0: float, g: np.ndarray,
                              share_lin: float = 0.6,
                              allocation_mode: str = "fixed",
                              risk_policy: RiskPolicy | None = None) -> dict:
    """Return a valid 2-D distance certificate and its selected risk allocation."""
    eps = float(eps)
    if not (0.0 < eps < 1.0):
        raise ValueError("geo2d eps must be in (0,1)")
    fixed = (max(min(float(share_lin), 0.95), 0.05)
             if str(allocation_mode) == "fixed"
             else max(min(float(share_lin), 0.999), 0.001))
    fractions = (fixed,) if str(allocation_mode) == "fixed" else _risk_fraction_grid(fixed)
    best = None
    for frac in fractions:
        e1 = eps * frac; e2 = eps - e1
        cand = _geo2d_dist_at_split(cell, d0, g, e1, e2, risk_policy)
        cand["along_fraction"] = float(frac)
        if best is None or cand["bound_m"] < best["bound_m"]:
            best = cand
    best["allocation_mode"] = str(allocation_mode)
    best["eps_total"] = eps
    assert abs(best["eps_along"] + best["eps_cross"] - eps) <= 1e-12
    return best


def _geo2d_dist_bound(cell: "M.XiCell", eps: float, d0: float, g: np.ndarray,
                      share_lin: float, allocation_mode: str = "fixed",
                      risk_policy: RiskPolicy | None = None) -> float:
    return float(_geo2d_dist_bound_details(cell, eps, d0, g, share_lin,
                                           allocation_mode, risk_policy)["bound_m"])


def _soc_margin_geo2d(a: np.ndarray, b: float, cell: "M.XiCell", eps: float,
                      d0: float, share_lin: float = 0.6,
                      allocation_mode: str = "fixed",
                      risk_policy: RiskPolicy | None = None) -> float:
    c = float(np.linalg.norm(a))
    if c == 0.0:
        return b
    det = _geo2d_dist_bound_details(cell, eps, float(d0), np.asarray(a) / c,
                                    share_lin, allocation_mode, risk_policy)
    return b - c * (float(det["bound_m"]) - float(d0))


def _soc_margin_geo2d_joint_details(a_xi: np.ndarray, a_w: np.ndarray, b: float,
                                    cell: "M.XiCell", wu: "WeatherUncertainty", eps: float,
                                    d0: float, share_lin: float = 0.6,
                                    share_wind: float = 0.2,
                                    allocation_mode: str = "fixed",
                                    risk_policy: RiskPolicy | None = None) -> dict:
    """Joint geo2d/weather certificate with route-wise Bonferroni allocation.

    The total event budget remains exactly ``eps``.  Optimized mode evaluates a deterministic
    set of valid splits and retains the least conservative one, always including the legacy
    fixed split as a candidate.
    """
    eps = float(eps); a_xi = np.asarray(a_xi, float); a_w = np.asarray(a_w, float)
    c = float(np.linalg.norm(a_xi))
    sd_w = math.sqrt(max(float(a_w @ wu.wind_cov @ a_w), 0.0))
    mean_w = float(a_w @ wu.wind_bias)
    fixed_w = (max(min(float(share_wind), 0.9), 0.05)
               if str(allocation_mode) == "fixed"
               else max(min(float(share_wind), 0.999), 0.001))
    wind_fracs = (fixed_w,) if str(allocation_mode) == "fixed" else _risk_fraction_grid(fixed_w)
    # If weather has no stochastic contribution, give the complete budget to xi exactly.
    if sd_w == 0.0:
        wind_fracs = (0.0,)
    best = None
    for fw in wind_fracs:
        ew = eps * float(fw)
        ex = eps - ew
        if ex <= 0.0:
            continue
        xi_det = (_geo2d_dist_bound_details(cell, ex, float(d0), a_xi / c,
                                            share_lin, allocation_mode, risk_policy)
                  if c > 0.0 else
                  dict(bound_m=float(d0), eps_along=0.0, eps_cross=0.0,
                       allocation_mode=str(allocation_mode), along_fraction=float('nan'),
                       std_along_m=0.0, std_cross_m=0.0, mean_along_m=0.0, mean_cross_m=0.0))
        xi_term = c * (float(xi_det["bound_m"]) - float(d0))
        rp = _risk_policy_from_inputs(risk_policy)
        wind_std = (rp.one_sided(ew) * sd_w if ew > 0.0 and sd_w > 0.0 else 0.0)
        total = xi_term + mean_w + wind_std
        cand = dict(xi_det)
        cand.update(margin=float(b) - total, total_tightening=total,
                    xi_tightening=xi_term, weather_mean=mean_w, weather_std=wind_std,
                    eps_total=eps, eps_weather=ew, eps_xi=ex,
                    wind_fraction=float(fw))
        if best is None or cand["total_tightening"] < best["total_tightening"]:
            best = cand
    if best is None:
        raise RuntimeError("no valid geo2d risk allocation candidate")
    assert abs(best["eps_weather"] + best["eps_xi"] - eps) <= 1e-12
    if c > 0.0:
        assert abs(best["eps_along"] + best["eps_cross"] - best["eps_xi"]) <= 1e-12
    best["risk_allocation_contract"] = GEO_RISK_ALLOCATION_CONTRACT
    return best


def _soc_margin_geo2d_joint(a_xi: np.ndarray, a_w: np.ndarray, b: float,
                            cell: "M.XiCell", wu: "WeatherUncertainty", eps: float,
                            d0: float, share_lin: float = 0.6,
                            share_wind: float = 0.2,
                            allocation_mode: str = "fixed",
                            risk_policy: RiskPolicy | None = None) -> float:
    return float(_soc_margin_geo2d_joint_details(
        a_xi, a_w, b, cell, wu, eps, d0, share_lin, share_wind,
        allocation_mode, risk_policy)["margin"])


# =============================================================================
# 3b. 外部 baseline 判据(论文对照; doc_related_work §对标, doc_model §16)
#     —— 与主线 DRCC 用【完全相同的物理层/列生成/主问题】, 只替换"约束左端"判据,
#        保证公平可比(同数据、同几何、同 ε)。两 baseline:
#       (1) SAA 样本机会约束(对标 multi-visit 2024): 用 ξ 矩重建样本, 要求"样本违反比例 ≤ ε"。
#       (2) Bertsimas–Sim 预算鲁棒(对标 Robust UAV-USV 2025): ξ 落在椭球预算不确定集,
#           最坏点硬约束(无概率语义, 保守度由 Γ 控制)。
# =============================================================================
def _saa_samples_from_cell(cell: "M.XiCell", n_samp: int = 200, seed: int = 12345) -> np.ndarray:
    """从 (h,c) 模糊集的矩 (μ,Σ) 重建 SAA 场景样本(2D)。
    诚实标注: 无原始 ξ 样本文件时, SAA 用矩重建的【高斯样本】近似(轻尾)——这正是 SAA 的
    典型软肋: 它对【建模时的样本分布】优化, 故在 out-of-sample 重尾(多元 t)回放下会欠保护。
    作者本机若有 step7 --dump-samples 的原始样本, 可换为经验样本(留作接口扩展)。
    更新: 接口已闭合 —— load_saa_empirical 登记后, 精确命中 (h,c) 统计格时用经验样本
    (决策细格 h 未命中仍回退矩重建, 如实; 经验样本使 SAA 与 gaussian 判据真正解耦)。"""
    _emp = SAA_EMPIRICAL.get((int(round(float(cell.h_min))), str(cell.c_state)))
    if _emp is not None:
        return _emp
    if not bool(SAA_ALLOW_MOMENT_FALLBACK):
        raise ValueError(
            f"formal SAA 缺少经验样本格 (h={cell.h_min}, c={cell.c_state}); "
            "禁止回退矩重建高斯。")
    rng = np.random.default_rng(seed)
    # 数值稳健的对称化 + Cholesky(退化时退回对角根)
    Sig = 0.5 * (cell.Sigma + cell.Sigma.T)
    try:
        L = np.linalg.cholesky(Sig + 1e-9 * np.eye(2))
    except np.linalg.LinAlgError:
        L = np.diag(np.sqrt(np.clip(np.diag(Sig), 0.0, None)))
    z = rng.standard_normal((n_samp, 2))
    return cell.mu[None, :] + z @ L.T


def _saa_margin(a: np.ndarray, b: float, samples: np.ndarray, eps: float) -> float:
    """SAA 样本机会约束余量: 要求 P̂(aᵀξ ≤ b) ≥ 1−ε, 即【样本上第 (1−ε) 分位的 aᵀξ ≤ b】。
    余量 = b − quantile_{1−ε}(aᵀξ_samples); ≥0 即该条样本机会约束成立(经验)。
    与 SOC 同符号约定(余量≥0 ⇒ 可行), 故可直接替换进 route_feasible_at_h。"""
    proj = samples @ a                       # (n,) 每个样本的 aᵀξ
    q = float(np.quantile(proj, 1.0 - eps))  # (1−ε) 经验分位 → 样本机会约束阈值
    return b - q


def _saa_margin_joint(a_xi: np.ndarray, a_w: np.ndarray, b: float, samples: np.ndarray,
                      wu: "WeatherUncertainty", eps: float, seed: int = 12345) -> float:
    """SAA 联合余量(ξ+风): 把 ξ 样本与独立采样的风误差 Δw 堆叠, 取 (1−ε) 经验分位。"""
    rng = np.random.default_rng(seed + 1)
    n = samples.shape[0]
    Lw = np.linalg.cholesky(0.5 * (wu.wind_cov + wu.wind_cov.T) + 1e-9 * np.eye(2))
    dw = wu.wind_bias[None, :] + rng.standard_normal((n, 2)) @ Lw.T
    proj = samples @ a_xi + dw @ a_w
    q = float(np.quantile(proj, 1.0 - eps))
    return b - q


def _budget_margin(a: np.ndarray, b: float, cell: "M.XiCell", gamma: float) -> float:
    """Bertsimas–Sim 椭球预算鲁棒余量: ξ ∈ {μ + Σ^{1/2} u : ‖u‖₂ ≤ Γ}(预算 Γ 控制保守度)。
    最坏点 max_{‖u‖≤Γ} aᵀξ = aᵀμ + Γ·‖Σ^{1/2}a‖₂ (硬约束, 无概率语义)。
    余量 = b − [aᵀμ + Γ·‖Σ^{1/2}a‖₂]; ≥0 即最坏点可行。
    注: 形式与 Cantelli SOC 同构, 差异在 Γ 是【外生预算参数】(拍脑袋调), 不像 κ(ε) 有"违反概率 ≤ ε"语义。"""
    quad = max(float(a @ cell.Sigma @ a), 0.0)
    return b - (float(a @ cell.mu) + gamma * math.sqrt(quad))


def _budget_margin_joint(a_xi: np.ndarray, a_w: np.ndarray, b: float, cell: "M.XiCell",
                         wu: "WeatherUncertainty", gamma: float) -> float:
    """预算鲁棒联合余量(ξ+风): u=[ξ;Δw] 落在 4D 椭球 ‖Σ_u^{-1/2}(u−μ_u)‖≤Γ。"""
    a = np.concatenate([a_xi, a_w])
    mu = np.concatenate([cell.mu, wu.wind_bias])
    Sig = np.zeros((4, 4)); Sig[:2, :2] = cell.Sigma; Sig[2:, 2:] = wu.wind_cov
    quad = max(float(a @ Sig @ a), 0.0)
    return b - (float(a @ mu) + gamma * math.sqrt(quad))


# Bertsimas–Sim 预算默认值: Γ 与 ε 无内在对应; 取 Γ=2.0 为常见"温和保守"起步值(高斯下≈97.7%单边)。
# 这正是要论证的点: Γ 调小欠保护、调大过保守, 缺概率可解释性(对照 κ(ε))。
BUDGET_GAMMA_DEFAULT = 2.0

# 更新: box(支持集)鲁棒的天气侧代理倍率 —— ξ 有经验支持半径(XiCell.support_radius=样本 max_norm),
# 但风/浪【预报误差】无经验支持统计, 用 3σ 盒作 SCN 代理(诚实标注; 待预报-分析残差标定)。
BOX_WEATHER_MULT = 3.0


def _box_margin(a: np.ndarray, b: float, cell: "M.XiCell") -> float:
    r"""【更新, E2 经典 RO-支持集基线】box(支持集)鲁棒余量:
    ξ ∈ {μ + u : ‖u‖₂ ≤ r},  r = cell.support_radius(该 (h,c) 格样本最大范数, 经验支持集)。
    最坏点 max aᵀξ = aᵀμ + r·‖a‖₂; 余量 = b − 该最坏值。无概率语义(硬最坏), 最保守的经典 RO 端点。"""
    r = max(float(getattr(cell, "support_radius", 0.0)), 0.0)
    return b - (float(a @ cell.mu) + r * float(np.linalg.norm(a)))


def _box_margin_joint(a_xi: np.ndarray, a_w: np.ndarray, b: float, cell: "M.XiCell",
                      wu: "WeatherUncertainty", mult: float = BOX_WEATHER_MULT) -> float:
    """box 联合余量(ξ+风): ξ 用经验支持球(半径 r_ξ), 风用 mult·√λmax(Σ_w) 盒代理(SCN, 见 BOX_WEATHER_MULT)。
    独立范数球最坏点可分离: max = aᵀμ + r_ξ‖a_ξ‖ + r_w‖a_w‖。"""
    r_xi = max(float(getattr(cell, "support_radius", 0.0)), 0.0)
    r_w = mult * _wind_sigma_max(wu)
    val = float(a_xi @ cell.mu) + float(a_w @ wu.wind_bias) \
        + r_xi * float(np.linalg.norm(a_xi)) + r_w * float(np.linalg.norm(a_w))
    return b - val


# ── 更新: SAA 经验样本登记表(任务0: 作者本地有真实 AIS ⇒ step7 --dump-samples 可产
#    xi_samples_caseB.csv)。登记后 SAA 变成【真正的样本基线】(此前用矩重建高斯样本 ⇒ 与
#    gaussian 判据几乎逐位相同, 不构成独立对照 —— 更新 修复的核心口径问题之一)。 ──
SAA_EMPIRICAL: dict = {}          # {(h_int, c_state): np.ndarray (n,2)}
SAA_SOURCE: str = "moment-gaussian(矩重建, 无经验样本文件)"
SAA_ALLOW_MOMENT_FALLBACK: bool = True


def load_saa_empirical(csv_path, mmsi: str = "ALL", min_n: int = 30, *,
                       require_current_contract: bool = False,
                       allow_pooled_fallback: bool = True) -> int:
    """登记经验 ξ 场景。

    Mechanism/research mode may optionally use explicitly reported pooled fallback.
    Formal single-vessel experiments must call with ``allow_pooled_fallback=False``:
    a sparse vessel-specific cell then remains unavailable rather than silently
    importing another vessel's samples.

    旧实现先按 MMSI 整表过滤，导致当前船某个 ``(h,c)`` 稀疏时该格静默退回矩重建高斯。
    现在逐格执行 exact-first：具体船样本达到门槛即采用；否则使用所有船的 pooled 样本，
    并把回退格数写入 ``SAA_SOURCE``。这只影响 SAA 对照，不改变 DRCC 主判据。
    """
    global SAA_SOURCE, SAA_ALLOW_MOMENT_FALLBACK
    import pandas as _pd
    SAA_EMPIRICAL.clear()
    SAA_SOURCE = "moment-gaussian(矩重建, 无经验样本文件)"
    SAA_ALLOW_MOMENT_FALLBACK = bool(allow_pooled_fallback)
    full = _pd.read_csv(csv_path)
    required = {"h_min", "c_state", "xi_e_m", "xi_n_m"}
    missing = sorted(required - set(full.columns))
    if missing:
        log.warning("SAA 经验样本缺列 %s，回退矩重建高斯。", missing)
        return 0
    expected_predictor = "cv_noleak"
    expected_contract = M.XI_PREDICTOR_CONTRACTS.get(expected_predictor, "unknown")
    if "predictor" in full.columns:
        found = sorted(set(full["predictor"].dropna().astype(str)))
        if found != [expected_predictor]:
            raise ValueError(
                f"SAA ξ 样本 predictor 与当前运行合同不一致: expected={expected_predictor!r}, found={found!r}。")
    if "predictor_contract" in full.columns:
        found = sorted(set(full["predictor_contract"].dropna().astype(str)))
        if found != [expected_contract]:
            raise ValueError(
                f"SAA ξ 样本 predictor_contract 与当前运行合同不一致: expected={expected_contract!r}, found={found!r}。")
    if "timestamp_epoch_contract" in full.columns:
        found = sorted(set(full["timestamp_epoch_contract"].dropna().astype(str)))
        if found != [M.XI_TIMESTAMP_EPOCH_CONTRACT]:
            raise ValueError(
                "SAA ξ 样本时间戳合同与当前运行合同不一致: "
                f"expected={M.XI_TIMESTAMP_EPOCH_CONTRACT!r}, found={found!r}。"
                "请用当前 step7_compute_xi.py --dump-samples 重新生成经验样本。")
    missing_contract = sorted({"predictor_contract", "timestamp_epoch_contract"} - set(full.columns))
    if missing_contract:
        if require_current_contract:
            raise ValueError(
                f"SAA ξ 样本缺当前合同列 {missing_contract}；formal SAA 禁止使用无法验证 provenance 的旧样本。")
        log.warning("SAA ξ 样本缺 predictor_contract/epoch_contract；仅允许旧机制对照，禁止正式验证。")

    # step7 的 SAA 样本与正式 ξ 矩共用同一离散 horizon 支持。不能把
    # nextafter/off-grid 值通过 round()/astype(int) 吸附到另一个有限状态。
    h_grid = {float(h).hex(): int(h) for h in range(5, 61, 5)}
    h_raw = pd.to_numeric(full["h_min"], errors="coerce").to_numpy(float)
    if not bool(np.isfinite(h_raw).all()):
        raise ValueError("SAA ξ 样本 h_min 包含缺失或非有限值。")
    bad_h = [float(h) for h in h_raw if float(h).hex() not in h_grid]
    if bad_h:
        sample = [f"{h!r} ({h.hex()})" for h in bad_h[:5]]
        raise ValueError(
            "SAA ξ 样本 h_min 必须逐 binary64 精确命中 5,10,...,60 分钟网格；"
            f"禁止 round/int/nearest 吸附。非法示例={sample}")
    full = full.copy()
    full["_h_exact"] = [h_grid[float(h).hex()] for h in h_raw]
    for name in ("xi_e_m", "xi_n_m"):
        vals = pd.to_numeric(full[name], errors="coerce").to_numpy(float)
        if not bool(np.isfinite(vals).all()):
            raise ValueError(f"SAA ξ 样本 {name} 包含缺失或非有限值。")
        full[name] = vals

    exact = full
    if mmsi != "ALL" and "mmsi" in full.columns:
        exact = full[full["mmsi"].astype(str) == str(mmsi)].copy()
    keys = sorted({(int(h), str(c))
                   for h, c in full[["_h_exact", "c_state"]].itertuples(index=False, name=None)})
    exact_n = pooled_n = 0
    for h, c in keys:
        ge = exact[(exact["_h_exact"] == int(h))
                   & (exact["c_state"].astype(str) == str(c))]
        source = ge
        if len(ge) >= int(min_n):
            exact_n += 1
        else:
            if str(mmsi).upper() != "ALL" and not bool(allow_pooled_fallback):
                continue
            gp = full[(full["_h_exact"] == int(h))
                      & (full["c_state"].astype(str) == str(c))]
            if len(gp) < int(min_n):
                continue
            source = gp
            pooled_n += 1
        SAA_EMPIRICAL[(int(h), str(c))] = source[["xi_e_m", "xi_n_m"]].to_numpy(float)
    n_cell = len(SAA_EMPIRICAL)
    if require_current_contract and str(mmsi).upper() != "ALL" and pooled_n:
        SAA_EMPIRICAL.clear()
        raise ValueError(
            f"formal SAA 禁止跨船 pooled fallback: mmsi={mmsi}, pooled_fallback={pooled_n}")
    if n_cell:
        SAA_SOURCE = (f"empirical({getattr(csv_path, 'name', csv_path)}, {n_cell} 格, "
                      f"mmsi={mmsi}, exact={exact_n}, pooled_fallback={pooled_n})")
        log.info("SAA 经验样本已登记 %d 格(exact=%d, pooled回退=%d) ← %s",
                 n_cell, exact_n, pooled_n, csv_path)
    else:
        log.warning("SAA 经验样本文件无有效格(min_n=%d), 回退矩重建高斯。", min_n)
    return n_cell


def _solve_airspeed_for_time(legs, wind_vec, T_target, v_max, v_floor=3.0):
    """给定各腿 (距离, 单位航向向量) + 风矢量 + 目标总时间, 二分求满足 Σ(d/vg(v))=T_target 的空速 v。
    返回 (v, feasible, actual_time)。vg(v)=w_along+√(v²−w_cross²)(风三角), 随 v 单调增 ⇒ 总时间随 v 单调减。
      - v_max 下仍太慢(total_time>T_target) → 追不上, infeasible。
      - v_floor 下已太快(total_time<T_target) → 飞 v_floor 再盘旋等待, 返回 v_floor。"""
    def total_time(v):
        T = 0.0
        for d, e in legs:
            w_along = float(np.dot(wind_vec, e))
            w_cross = float(np.linalg.norm(wind_vec - w_along * e))
            if v <= w_cross:
                return float("inf")
            vg = w_along + math.sqrt(max(v * v - w_cross * w_cross, 0.0))
            if vg <= 0.5:
                return float("inf")
            T += d / vg
        return T
    t_max = total_time(v_max)
    if t_max > T_target:                       # 最大速度仍太慢
        return v_max, False, t_max
    t_floor = total_time(v_floor)
    if t_floor <= T_target:                    # 最低速度已够快 → 飞 v_floor + 盘旋
        return v_floor, True, t_floor
    lo, hi = v_floor, v_max
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if total_time(mid) > T_target:
            lo = mid
        else:
            hi = mid
    v = 0.5 * (lo + hi)
    return v, True, total_time(v)


def _return_leg_wind_at_height(route: "Route", p: M.Params, wx: dict) -> tuple[np.ndarray, float]:
    """Return nominal cruise-height wind vector and 10m→cruise linear scale for the return leg."""
    last = route.turbines[-1]
    loc = getattr(last, "wx_local", None)
    w10_g, wdir_g = _wind_of(wx)
    w10, wdir = (_wind_of(loc, w10_g, wdir_g) if loc is not None else (w10_g, wdir_g))
    alpha = float(M.wind_at_height(1.0, p.z_cruise, p.z0))
    return M.wind_vector_from(float(w10) * alpha, float(wdir)), alpha


def _required_airspeed_geo2d_certificate(route: "Route", h: int, p: M.Params, wx: dict,
                                          cell: "M.XiCell", weather_unc,
                                          return_budget_s: float, eps: float,
                                          allocation_mode: str = "optimized",
                                          risk_policy: RiskPolicy | None = None) -> dict:
    r"""Certificate for the recourse airspeed needed to hit the fixed touchdown time.

    At the start of the return leg, all non-return phases have already consumed deterministic
    time.  The remaining return-time budget ``T`` is fixed.  The constant ground-velocity vector
    required to meet the moving ship is

        v_g = (P_hat + xi - q_last) / T,

    and the required airspeed is ``v_air = v_g - (w_nom + Delta w)``.  Thus the random required
    airspeed is a two-dimensional affine vector.  We apply the same geo2d moment certificate
    directly to its norm; this is materially different from converting position uncertainty to
    a delay at a frozen nominal ground speed.
    """
    T = float(return_budget_s)
    if not math.isfinite(T) or T <= 0.0:
        return dict(ok=False, safe_required_airspeed_ms=float("inf"),
                    nominal_required_airspeed_ms=float("inf"), return_budget_s=T,
                    margin_ms=-float("inf"), reason="nonpositive_return_budget")
    q_last = np.asarray(route.turbines[-1].local, float)
    P_rec = np.asarray(route.ship.predicted_at(float(h)), float)
    r0 = P_rec - q_last
    w_nom, alpha = _return_leg_wind_at_height(route, p, wx)
    base = r0 / T - w_nom
    eta_mu = np.asarray(cell.mu, float) / T
    eta_cov = np.asarray(cell.Sigma, float) / (T * T)
    if weather_unc is not None:
        eta_mu = eta_mu - alpha * np.asarray(weather_unc.wind_bias, float)
        eta_cov = eta_cov + (alpha * alpha) * np.asarray(weather_unc.wind_cov, float)
    eta_cov = 0.5 * (eta_cov + eta_cov.T)
    vals, vecs = np.linalg.eigh(eta_cov)
    eta_cov = (vecs * np.maximum(vals, 0.0)) @ vecs.T
    d0 = float(np.linalg.norm(base))
    g = base / d0 if d0 > 0.0 else np.array([1.0, 0.0])
    tmp = M.XiCell(h_min=int(h), c_state=str(cell.c_state), n=int(cell.n),
                   mu=np.asarray(eta_mu, float), Sigma=np.asarray(eta_cov, float),
                   support_radius=float(cell.support_radius) / T,
                   p95_norm=float(cell.p95_norm) / T, rms_norm=float(cell.rms_norm) / T)
    if getattr(p, "soc_correction", "none") == "geo2d":
        det = _geo2d_dist_bound_details(tmp, float(eps), d0, g,
                                        float(getattr(p, "soc_share_lin", 0.6)),
                                        str(allocation_mode), risk_policy)
        safe = float(det["bound_m"])
    else:
        mean = float(g @ tmp.mu)
        std = math.sqrt(max(float(g @ tmp.Sigma @ g), 0.0))
        safe = d0 + mean + _risk_policy_from_inputs(risk_policy).one_sided(float(eps)) * std
        det = dict(bound_m=safe, eps_total=float(eps), eps_along=float(eps),
                   eps_cross=0.0, mean_along_m=mean, mean_cross_m=0.0,
                   std_along_m=std, std_cross_m=0.0,
                   allocation_mode="linear_projection")
    nominal = float(np.linalg.norm(base))
    return dict(ok=_strict_finite_leq(safe, p.v_air_max),
                contract=SPEED_RECOURSE_CONTRACT,
                return_budget_s=T, nominal_displacement_m=float(np.linalg.norm(r0)),
                nominal_wind_vector_ms=np.asarray(w_nom, float), wind_height_scale=alpha,
                nominal_required_airspeed_ms=nominal,
                safe_required_airspeed_ms=float(safe),
                margin_ms=float(p.v_air_max) - float(safe),
                combined_mean_ms=np.asarray(eta_mu, float),
                combined_cov_ms2=np.asarray(eta_cov, float),
                eps_total=float(eps), eps_along=float(det.get("eps_along", float("nan"))),
                eps_cross=float(det.get("eps_cross", float("nan"))),
                geo_detail=det)


def _required_airspeed_joint_certificate(route: "Route", h: int, p: M.Params, wx: dict,
                                           cell: "M.XiCell", weather_unc,
                                           nominal_return_budget_s: float, eps_total: float,
                                           t_dock_s: float, allocation_mode: str = "optimized",
                                           risk_policy: RiskPolicy | None = None) -> dict:
    r"""Bonferroni certificate for outbound-time delay and random return airspeed.

    Weather affects two distinct pieces of the fixed-touchdown policy: it can delay the
    already-flown outbound/inter-turbine legs, thereby reducing the remaining return budget,
    and it changes the wind vector on the return leg.  The two events may be dependent; a
    Bonferroni split is therefore used and does not assume independence.
    """
    eps_total = float(eps_total)
    T_nom = float(nominal_return_budget_s)
    if not (0.0 < eps_total < 1.0):
        raise ValueError("speed-recourse eps_total must be in (0,1)")
    if weather_unc is None:
        cert = _required_airspeed_geo2d_certificate(
            route, h, p, wx, cell, None, T_nom, eps_total, allocation_mode, risk_policy)
        cert.update(eps_total=eps_total, eps_nonreturn_weather=0.0,
                    eps_return_required_airspeed=eps_total,
                    nonreturn_weather_mean_shift_s=0.0,
                    nonreturn_weather_std_term_s=0.0,
                    nonreturn_weather_reserve_s=0.0,
                    nominal_return_budget_s=T_nom, safe_return_budget_s=T_nom,
                    risk_allocation_contract=SPEED_RECOURSE_CONTRACT)
        return cert

    _, aT_nonret = route_wind_sensitivity(
        route, h, p, wx, t_dock_s=float(t_dock_s),
        energy_branch="fixed_nonreturn", time_branch="fixed_nonreturn")
    mean_delay = float(aT_nonret @ np.asarray(weather_unc.wind_bias, float))
    sd_delay = math.sqrt(max(float(aT_nonret @ np.asarray(weather_unc.wind_cov, float)
                                   @ aT_nonret), 0.0))
    fixed_nonret_fraction = 0.20
    if sd_delay == 0.0:
        fractions = (0.0,)
    elif str(allocation_mode) == "fixed":
        fractions = (fixed_nonret_fraction,)
    else:
        fractions = _risk_fraction_grid(fixed_nonret_fraction)
    best = None
    for frac in fractions:
        eps_nonret = eps_total * float(frac)
        eps_return = eps_total - eps_nonret
        if eps_return <= 0.0:
            continue
        std_term = (_risk_policy_from_inputs(risk_policy).one_sided(eps_nonret) * sd_delay
                    if eps_nonret > 0.0 and sd_delay > 0.0 else 0.0)
        reserve = mean_delay + std_term
        T_safe = T_nom - reserve
        cert = _required_airspeed_geo2d_certificate(
            route, h, p, wx, cell, weather_unc, T_safe, eps_return, allocation_mode, risk_policy)
        cand = dict(cert)
        cand.update(eps_total=eps_total, eps_nonreturn_weather=float(eps_nonret),
                    eps_return_required_airspeed=float(eps_return),
                    nonreturn_weather_mean_shift_s=float(mean_delay),
                    nonreturn_weather_std_term_s=float(std_term),
                    nonreturn_weather_reserve_s=float(reserve),
                    nominal_return_budget_s=T_nom, safe_return_budget_s=float(T_safe),
                    nonreturn_time_wind_gradient_s_per_ms=np.asarray(aT_nonret, float),
                    risk_allocation_contract=SPEED_RECOURSE_CONTRACT)
        score = float(cand.get("safe_required_airspeed_ms", float("inf")))
        if best is None or score < float(best.get("safe_required_airspeed_ms", float("inf"))):
            best = cand
    if best is None:
        raise RuntimeError("no valid speed-recourse risk allocation candidate")
    assert abs(float(best["eps_nonreturn_weather"])
               + float(best["eps_return_required_airspeed"]) - eps_total) <= 1e-12
    return best


def _max_power_on_interval(p: M.Params, lo: float, hi: float, n: int = 257) -> float:
    r"""Certified continuous maximum of ``leg_power`` on a speed interval.

    ``n`` is retained only for API compatibility; no sampling is used.  The
    legacy cubic law is monotone for the validated nonnegative parameters.  For
    the Zeng law, let ``x=V**2/(2*v0**2)`` and

        f(V) = sqrt(sqrt(1+x**2)-x) = exp(-asinh(x)/2).

    Apart from the nonnegative scale and constants, the derivative is

        P'(V) = V * [6*P0/Utip**2 + 3*c_drag*V
                     - Pi*f(V)/(2*v0**2*sqrt(1+x**2))].

    The bracket is nondecreasing: its first two terms are nondecreasing, while
    ``f(V)/sqrt(1+x**2)`` is positive and nonincreasing.  Hence the derivative
    changes sign at most once, from negative to positive, so the Zeng curve is
    decreasing-then-increasing and its maximum on every compact interval is at
    an endpoint.  ``P_zeng``'s 0.01 m/s clamp is constant below that speed and
    preserves the endpoint result.  A small outward floating-point guard keeps
    the implemented bound from rounding below the endpoint maximum.
    """
    del n
    lo = max(0.0, float(lo))
    hi = max(lo, float(hi))
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError("power-envelope interval endpoints must be finite")

    if not bool(getattr(p, "use_zeng", False)):
        vals = (float(p.v_cr), float(p.P_cr))
        if not all(math.isfinite(v) for v in vals) or vals[0] <= 0.0 or vals[1] < 0.0:
            raise ValueError("cannot certify legacy power envelope with nonphysical parameters")
    else:
        scale = float(getattr(p, "power_scale", 1.0))
        vals = (scale, float(p.zeng_P0), float(p.zeng_Pi),
                float(p.zeng_Utip), float(p.zeng_v0), float(p.zeng_d0),
                float(p.zeng_rho), float(p.zeng_s), float(p.zeng_A),
                float(p.zeng_Pelec))
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("cannot certify Zeng power envelope with nonfinite parameters")
        scale, P0, Pi, U, v0, d0, rho, solidity, area, Pelec = vals
        if (scale < 0.0 or P0 < 0.0 or Pi < 0.0 or U <= 0.0 or v0 <= 0.0
                or d0 < 0.0 or rho < 0.0 or solidity < 0.0 or area < 0.0
                or Pelec < 0.0):
            raise ValueError("cannot certify Zeng power envelope with nonphysical parameters")

    upper = max(float(M.leg_power(p, lo)), float(M.leg_power(p, hi)))
    guard = 64.0 * np.finfo(float).eps * max(1.0, abs(upper))
    return float(math.nextafter(upper + guard, math.inf))


def _feasible_speed_adjustable(route: "Route", h: int, p: M.Params, wx: dict,
                               cell: "M.XiCell", weather_unc=None,
                               wx_recovery: dict | None = None,
                               risk_policy: RiskPolicy | None = None, deadline=None) -> dict:
    r"""Fixed-touchdown feasibility with both wait and return-speed recourse.

    The old implementation converted every metre of vessel-position uncertainty into seconds at
    the *nominal* return ground speed, even though the route had already certified controllability
    up to ``v_air_max``.  This function instead reserves the deterministic non-return phases first
    and certifies the random airspeed vector required to cover the return displacement inside the
    remaining fixed time.  No UAV speed limit or risk budget is changed.
    """
    _check_deadline(deadline)
    rp = _risk_policy_from_inputs(risk_policy)
    h_s = 60.0 * float(h)
    # Terminal gate/dock uses exactly the same worst-case candidate contract as the wait-only path.
    cand_wx = gate_weather_candidates(route, wx_recovery or wx)
    r_gate = wind_speed_upper_shift("drcc", weather_unc,
                                    float(getattr(p, "eps_gate", p.eps_cap)),
                                    kappa_fn=rp.one_sided)
    r_dock_low = wind_speed_lower_shift("drcc", weather_unc,
                                        float(getattr(p, "eps_dock", p.eps_cap)),
                                        kappa_fn=rp.one_sided)
    checked = []
    for cw in cand_wx:
        _check_deadline(deadline)
        hs_eff = float(cw["Hs"]) + ((weather_unc.hs_bias + rp.one_sided(p.eps_cap) * weather_unc.hs_std)
                                     if weather_unc is not None else 0.0)
        w10 = float(cw["wind10"])
        motion = M.deck_motion(hs_eff, cw.get("Tp", 2.1),
                               cw.get("wave_dir", 200.0) - cw.get("ship_heading", 0.0), p)
        td, Ed = M.dock_reserve(p, motion, max(0.0, w10 + r_dock_low))
        L, reff = M.landing_gate(hs_eff, cw.get("Tp", 2.1), cw.get("wave_dir", 200.0),
                                cw.get("ship_heading", 0.0), w10 + r_gate, p)
        checked.append(dict(Hs_eff=hs_eff, t_dock=float(td), E_dock=float(Ed), L=int(L),
                            r_eff=float(reff), w10_gate=w10 + r_gate,
                            wave_ok=bool(motion["heave"] <= p.s_heave_max
                                         and motion["roll"] <= p.s_roll_max
                                         and motion["pitch"] <= p.s_pitch_max
                                         and hs_eff <= p.Hs_op),
                            wind_ok=bool(w10 + r_gate <= p.w_land_max)))
    t_dock = max(c["t_dock"] for c in checked)
    t_dock_wait = min(c["t_dock"] for c in checked)
    E_dock = max(c["E_dock"] for c in checked)
    gate_open = all(c["L"] == 1 for c in checked)
    gate_weather_proof = dict(selector="nearest-route-turbine", candidate_count=len(cand_wx),
                              candidates_checked=len(checked), all_candidates_checked=True,
                              switching_proven_safe=True,
                              worst_case_aggregation="gate=forall-open,t_dock=max,E_dock=max")

    nom = route_nominal_ET(route, h, p, wx, t_dock_s=t_dock_wait)
    d0 = float(nom["d_ret0"]); T_ret_nom = d0 / max(float(nom["v_ret"]), 1e-12)
    T_nonreturn = float(nom["T0"]) - T_ret_nom
    return_budget = h_s - float(t_dock) - T_nonreturn
    physical_core_nom = float(nom["T0"]) + float(t_dock)
    physical_wait_nom = max(0.0, h_s - physical_core_nom)

    alloc = str(getattr(p, "soc_risk_allocation", "optimized"))
    time_cert = _required_airspeed_joint_certificate(
        route, h, p, wx, cell, weather_unc, return_budget, float(p.eps_T),
        t_dock_s=t_dock_wait, allocation_mode=alloc, risk_policy=rp)
    safe_return_budget = float(time_cert.get("safe_return_budget_s", return_budget))
    # Nominal feasibility is evaluated at the nominal schedule and nominal wind, before any DRCC
    # reserve is taken.  Keep it separate from the base vector used by the safe certificate.
    q_last_nom = np.asarray(route.turbines[-1].local, float)
    P_rec_nom = np.asarray(route.ship.predicted_at(float(h)), float)
    w_nom_ret, _alpha_nom = _return_leg_wind_at_height(route, p, wx)
    nominal_required_schedule = (float(np.linalg.norm((P_rec_nom - q_last_nom) / return_budget
                                                       - w_nom_ret))
                                 if return_budget > 0.0 else float("inf"))
    time_cert["base_required_airspeed_at_safe_budget_ms"] = float(
        time_cert.get("nominal_required_airspeed_ms", float("inf")))
    time_cert["nominal_schedule_required_airspeed_ms"] = float(nominal_required_schedule)
    # Energy is bounded for every realization that passes the physical airspeed limit.  This avoids
    # adding an independent "maximum flight energy + full nominal waiting" combination.
    escort = escort_state(route, h, p, wx)
    p_env = max(float(escort["power_W"]),
                _max_power_on_interval(p, 0.0, float(p.v_air_max)))
    E_return_envelope = max(return_budget, 0.0) * p_env / 3600.0
    E_ret_nom = float(nom["c_E_noescort"]) * d0
    E_fixed_nonreturn = float(nom["E_branch_noescort_Wh"]) - E_ret_nom
    # Coupled horizontal-time envelope.  For every realization, the time left after the fixed
    # non-return phases is allocated between return flight and escort/wait, each with power no
    # greater than p_env.  Therefore the weather-sensitive upper-bound function is
    #   E_fixed_nonreturn(w) + p_env * [h - dock - T_nonreturn(w)].
    # Its gradient contains the crucial -p_env*dT_nonreturn term; omitting it would combine
    # worst non-return energy with a full nominal return/wait duration that cannot co-occur.
    weather_E_mean = weather_E_std = 0.0
    energy_coupled_gradient = np.zeros(2)
    if weather_unc is not None:
        aE_w, aT_nonret_E = route_wind_sensitivity(
            route, h, p, wx, t_dock_s=t_dock_wait,
            energy_branch="fixed_nonreturn", time_branch="fixed_nonreturn")
        energy_coupled_gradient = np.asarray(aE_w, float) - (p_env / 3600.0) * np.asarray(aT_nonret_E, float)
        weather_E_mean = float(energy_coupled_gradient @ np.asarray(weather_unc.wind_bias, float))
        weather_E_std = rp.one_sided(float(p.eps_E)) * math.sqrt(max(float(
            energy_coupled_gradient @ weather_unc.wind_cov @ energy_coupled_gradient), 0.0))
    E_safe = E_fixed_nonreturn + E_return_envelope + weather_E_mean + weather_E_std + E_dock
    margin_E = float(p.B_use) - float(E_safe)

    # Outbound/inter-turbine legs retain their own weather controllability event; the random return
    # direction is already included in the required-airspeed vector certificate above.
    air_diag = route_airspeed_projection_check(
        nom, p, weather_unc, float(getattr(p, "eps_air", p.eps_cap)),
        chance_mode="drcc", include_return=False, kappa_fn=rp.one_sided) if weather_unc is not None else dict(
            ok=bool(nom.get("speed_ok_outbound", True)),
            margin_ms=float(p.v_air_max) - float(nom.get("max_required_airspeed_out_ms", 0.0)),
            worst_leg=None, leg_checks=[])
    outbound_ok = bool(nom.get("speed_ok_outbound", True) and air_diag["ok"])

    escort_margin = float(p.v_air_max) - float(escort["required_airspeed_ms"])
    if weather_unc is not None:
        escort_margin -= wind_delta_radius("drcc", weather_unc,
                                           float(getattr(p, "eps_escort", p.eps_cap)))
    escort_ok = bool(escort["feasible"] and escort_margin >= 0.0)

    nominal_speed_margin = float(p.v_air_max) - float(nominal_required_schedule)
    safe_speed_margin = float(time_cert["margin_ms"])
    # Convert speed-capacity slack to an explicitly labelled equivalent seconds diagnostic.  The
    # hard feasibility test remains the airspeed margin, not this reporting transform.
    nominal_margin_eq_s = (max(return_budget, 0.0) * nominal_speed_margin / float(p.v_air_max)
                           if math.isfinite(nominal_speed_margin) else -1.0e30)
    safe_margin_eq_s = (max(safe_return_budget, 0.0) * safe_speed_margin / float(p.v_air_max)
                        if math.isfinite(safe_speed_margin) else -1.0e30)
    tightening_eq_s = nominal_margin_eq_s - safe_margin_eq_s
    nominal_failed = bool(
        return_budget <= 0.0
        or not _strict_finite_leq(nominal_required_schedule, p.v_air_max))
    time_failed = bool(
        return_budget <= 0.0
        or not _strict_finite_leq(
            time_cert.get("safe_required_airspeed_ms", float("nan")), p.v_air_max))
    L = 1 if gate_open else 0
    landing_wave_ok = all(c["wave_ok"] for c in checked)
    landing_wind_ok = all(c["wind_ok"] for c in checked)
    feasible = bool(not nominal_failed and not time_failed and margin_E >= 0.0 and L == 1
                    and outbound_ok and escort_ok)
    reason = None
    if not feasible:
        if L != 1: reason = "landing_gate_closed"
        elif not outbound_ok: reason = "route_airspeed_infeasible"
        elif not escort_ok: reason = "escort_speed_infeasible"
        elif margin_E < 0.0: reason = "energy_margin_negative"
        else: reason = "time_margin_negative"

    safe_core_eq = h_s - safe_margin_eq_s
    # Preserve the original position-error diagnostics alongside the airspeed-domain certificate.
    # This lets E1 distinguish a genuinely wide Xi cell from an overly conservative mapping.
    rhat = ((np.asarray(route.ship.predicted_at(float(h)), float)
             - np.asarray(route.turbines[-1].local, float)))
    rnorm = float(np.linalg.norm(rhat))
    rhat = rhat / (rnorm + 1e-15) if rnorm > 0.0 else np.array([1.0, 0.0])
    phat = np.array([-rhat[1], rhat[0]])
    xi_mean_along = float(rhat @ np.asarray(cell.mu, float))
    xi_mean_cross = float(phat @ np.asarray(cell.mu, float))
    xi_std_along = math.sqrt(max(float(rhat @ np.asarray(cell.Sigma, float) @ rhat), 0.0))
    xi_std_cross = math.sqrt(max(float(phat @ np.asarray(cell.Sigma, float) @ phat), 0.0))
    xi_state_change_rate = float(getattr(cell, "state_change_rate", float("nan")))
    xi_recovery_mode = str(getattr(cell, "actual_recovery_state_mode", ""))
    return dict(
        feasible=feasible, reason=reason, primary_reason=reason,
        h=h, c=str(route.ship.c_state), n=int(cell.n),
        E0=float(E_safe - E_dock), E_flight_Wh=float(E_fixed_nonreturn + E_return_envelope),
        E_escort_Wh=0.0, E_wait=0.0, E_plan_Wh=float(E_safe),
        E_soc_required_Wh=float(E_safe), E_uncertainty_buffer_Wh=float(max(E_safe - (float(nom["E0"]) + E_dock), 0.0)),
        T0=float(nom["T0"]), T_flight=float(nom["T0"]), T_escort_s=float(physical_wait_nom),
        margin_E=float(margin_E), margin_T=float(safe_margin_eq_s),
        margin_return_airspeed_ms=float(safe_speed_margin),
        M_omega=float(min(margin_E, safe_margin_eq_s, safe_speed_margin, escort_margin,
                          float(air_diag["margin_ms"]))),
        d_ret0=d0, v_ret=float(nom["v_ret"]), speed_feasible=bool(outbound_ok and not time_failed),
        xi_mean_along_m=float(xi_mean_along), xi_mean_cross_m=float(xi_mean_cross),
        xi_std_along_m=float(xi_std_along), xi_std_cross_m=float(xi_std_cross),
        xi_state_change_rate=float(xi_state_change_rate), xi_recovery_mode=xi_recovery_mode,
        xi_launch_to_recovery_state_change_rate=float(xi_state_change_rate),
        xi_actual_recovery_state_mode=xi_recovery_mode,
        xi_launch_speed_p50_ms=float(getattr(cell, "launch_speed_p50_ms", float("nan"))),
        xi_launch_speed_p95_ms=float(getattr(cell, "launch_speed_p95_ms", float("nan"))),
        max_required_airspeed_ms=float(max(float(nom.get("max_required_airspeed_out_ms", 0.0)),
                                               float(time_cert["safe_required_airspeed_ms"]))),
        airspeed_margin_ms=float(min(float(air_diag["margin_ms"]), safe_speed_margin)),
        route_airspeed_diag=air_diag,
        return_time_budget_s=float(return_budget),
        return_time_budget_safe_s=float(safe_return_budget),
        nonreturn_weather_mean_shift_s=float(time_cert.get("nonreturn_weather_mean_shift_s", 0.0)),
        nonreturn_weather_std_term_s=float(time_cert.get("nonreturn_weather_std_term_s", 0.0)),
        nonreturn_weather_reserve_s=float(time_cert.get("nonreturn_weather_reserve_s", 0.0)),
        eps_time_nonreturn_weather=float(time_cert.get("eps_nonreturn_weather", 0.0)),
        eps_time_return_required_airspeed=float(time_cert.get("eps_return_required_airspeed", p.eps_T)),
        return_required_airspeed_nom_ms=float(nominal_required_schedule),
        return_required_airspeed_base_safe_budget_ms=float(time_cert.get("base_required_airspeed_at_safe_budget_ms", float("nan"))),
        return_required_airspeed_safe_ms=float(time_cert["safe_required_airspeed_ms"]),
        return_airspeed_margin_ms=float(safe_speed_margin),
        return_speed_recourse_contract=SPEED_RECOURSE_CONTRACT,
        return_power_envelope_W=float(p_env),
        return_power_envelope_contract=POWER_ENVELOPE_CONTRACT,
        return_energy_envelope_Wh=float(E_return_envelope),
        energy_weather_mean_shift_Wh=float(weather_E_mean),
        energy_weather_std_term_Wh=float(weather_E_std),
        energy_coupled_wind_gradient_Wh_per_ms=np.asarray(energy_coupled_gradient, float),
        energy_recourse_contract=ENERGY_SPEED_RECOURSE_CONTRACT,
        landing=L, gate=L, Hs_eff=max(c["Hs_eff"] for c in checked),
        w10_gate_eff=max(c["w10_gate"] for c in checked), r_cap_eff=min(c["r_eff"] for c in checked),
        E_dock_Wh=float(E_dock), t_dock_s=float(t_dock), t_dock_wait_s=float(t_dock_wait),
        escort_required_airspeed_ms=float(escort["required_airspeed_ms"]),
        escort_speed_feasible=bool(escort_ok), escort_margin_ms=float(escort_margin),
        gate_weather_proof=gate_weather_proof,
        time_contract=SPEED_RECOURSE_TIME_CONTRACT,
        time_contract_id=SPEED_RECOURSE_TIME_CONTRACT,
        time_feasibility_basis="return_required_airspeed",
        wait_is_recourse=True, speed_is_recourse=True, dock_risk_contract=DOCK_RISK_CONTRACT,
        time_flight_nom_s=float(nom.get("time_flight_nom_s", nom["T0"])),
        time_inspection_s=float(nom.get("time_inspection_s", 0.0)),
        time_dock_nom_s=float(t_dock), time_core_nom_s=float(physical_core_nom),
        time_wait_nom_s=float(physical_wait_nom),
        nominal_time_margin_s=float(nominal_margin_eq_s),
        time_drcc_margin_s=float(safe_margin_eq_s),
        time_drcc_tightening_s=float(tightening_eq_s),
        time_safe_core_s=float(safe_core_eq), time_wait_safe_s=float(max(0.0, safe_margin_eq_s)),
        time_xi_mean_shift_s=0.0, time_xi_std_term_s=0.0,
        time_weather_mean_shift_s=0.0, time_weather_std_term_s=0.0,
        time_geometry_remainder_s=float(tightening_eq_s),
        eps_time_total=float(p.eps_T),
        eps_time_xi=float(time_cert.get("eps_return_required_airspeed", p.eps_T)),
        eps_time_weather=float(time_cert.get("eps_nonreturn_weather", 0.0)),
        eps_time_along=float(time_cert.get("eps_along", float("nan"))),
        eps_time_cross=float(time_cert.get("eps_cross", float("nan"))),
        geo_risk_allocation_mode=alloc,
        geo_risk_allocation_contract=SPEED_RECOURSE_CONTRACT,
        failure_flags=dict(
            missing_recovery_state=False, forbidden_recovery_state=False,
            missing_recovery_prediction=False, missing_recovery_weather=False,
            missing_xi_cell=False, risk_budget_invalid=False,
            escort_airspeed_failed=not escort_ok, landing_gate_failed=L != 1,
            landing_wind_failed=not landing_wind_ok, landing_wave_failed=not landing_wave_ok,
            nominal_airspeed_failed=not bool(nom.get("speed_ok_outbound", True)),
            route_airspeed_failed=not outbound_ok,
            energy_drcc_failed=bool(margin_E < 0.0),
            nominal_time_failed=nominal_failed, time_drcc_failed=time_failed),
        margins=dict(energy_Wh=float(margin_E), time_s=float(safe_margin_eq_s),
                     return_airspeed_ms=float(safe_speed_margin),
                     route_airspeed_ms=float(air_diag["margin_ms"]),
                     landing_wind_ms=float(p.w_land_max) - max(c["w10_gate"] for c in checked),
                     escort_airspeed_ms=float(escort_margin)),
        time_decomposition=dict(
            time_contract=SPEED_RECOURSE_TIME_CONTRACT,
            return_speed_recourse_contract=SPEED_RECOURSE_CONTRACT,
            return_time_budget_s=float(return_budget),
            return_time_budget_safe_s=float(safe_return_budget),
            nonreturn_weather_mean_shift_s=float(time_cert.get("nonreturn_weather_mean_shift_s", 0.0)),
            nonreturn_weather_std_term_s=float(time_cert.get("nonreturn_weather_std_term_s", 0.0)),
            nonreturn_weather_reserve_s=float(time_cert.get("nonreturn_weather_reserve_s", 0.0)),
            nominal_required_airspeed_ms=float(nominal_required_schedule),
            base_required_airspeed_at_safe_budget_ms=float(time_cert.get("base_required_airspeed_at_safe_budget_ms", float("nan"))),
            safe_required_airspeed_ms=float(time_cert["safe_required_airspeed_ms"]),
            return_airspeed_margin_ms=float(safe_speed_margin),
            nominal_margin_equiv_s=float(nominal_margin_eq_s),
            final_margin_s=float(safe_margin_eq_s),
            total_tightening_s=float(tightening_eq_s),
            eps_time_total=float(p.eps_T),
            eps_time_nonreturn_weather=float(time_cert.get("eps_nonreturn_weather", 0.0)),
            eps_time_return_required_airspeed=float(time_cert.get("eps_return_required_airspeed", p.eps_T)),
            eps_time_along=float(time_cert.get("eps_along", float("nan"))),
            eps_time_cross=float(time_cert.get("eps_cross", float("nan"))),
            combined_required_airspeed_mean_ms=np.asarray(time_cert.get("combined_mean_ms", np.zeros(2))).tolist(),
            combined_required_airspeed_cov_ms2=np.asarray(time_cert.get("combined_cov_ms2", np.zeros((2,2)))).tolist(),
            geo_detail=dict(time_cert.get("geo_detail", {}))))

def _wind_sigma_max(wu: "WeatherUncertainty") -> float:
    """更新: 风矢量协方差的最大方向标准差 √λmax(用于对接储备的风下侧 w10_low)。"""
    try:
        return float(np.sqrt(max(np.linalg.eigvalsh(wu.wind_cov).max(), 0.0)))
    except Exception:
        return 0.0


def recovery_gate_wx(route: "Route", wx: dict, P_rec: np.ndarray) -> dict:
    """更新(口径统一): 回收点处的着舰门/对接天气 —— 有逐风机天气时取离预测回收点最近
    风机的本地天气, 否则代表性 wx。规划(route_feasible_at_h)与回放(step15)共用本函数,
    保证两侧同一 gate 天气解析口径。"""
    gate_wx = wx
    if any(getattr(t, "wx_local", None) is not None for t in route.turbines):
        near = min(route.turbines, key=lambda t: float(np.linalg.norm(t.local - P_rec)))
        loc = getattr(near, "wx_local", None)
        if loc is not None:
            gate_wx = {**wx, **{k: loc[k] for k in ("Hs", "Tp", "wave_dir", "wind10") if k in loc}}
    return gate_wx


def gate_weather_candidates(route: "Route", wx: dict) -> list:
    r"""更新(C-01): 回收门/对接【可能实现】的天气场候选集合。

    recovery_gate_wx 是关于实现回收点 P_rec 的【不连续最近邻选择器】——
    P_rec = 预测点 + ξ, 而 ξ 的矩模糊集(μ,Σ)【没有有界支持】: 任意满足二阶矩的分布
    都可把非零概率质量放在最近邻边界另一侧(两点分布反例: 门失败概率可达 0.5 而
    ξ-SOC 全部通过)。因此规划侧不能只用预测回收点选一次场; 对该路线, 实现时可能被
    选中的场恰是: {每台挂 wx_local 的风机的合成场} ∪ {代表场 wx, 若存在无 wx_local
    的风机}(recovery_gate_wx 的取值域, 逐风机穷举)。无任何 wx_local ⇒ 恒为 [wx],
    不存在切换事件。route_feasible_at_h 对返回的【全部】候选做最坏情况检查
    (门须全开; t_dock/E_dock 取最坏), 使规划判据支配任何实现选择 —— 这是在无支持集
    信息下唯一保真的处理(候选逐场的 ε 细分配属未来工作, 见 MODIFICATIONS 更新)。
    返回 [dict], 至少 1 个; 按 (Hs,Tp,wave_dir,wind10) 去重(与 recovery_gate_wx
    的合成键完全一致)。"""
    turbs = list(getattr(route, "turbines", []) or [])
    if not any(getattr(t, "wx_local", None) is not None for t in turbs):
        return [dict(wx)]
    cands, keys = [], set()
    for t in turbs:
        loc = getattr(t, "wx_local", None)
        if loc is None:
            cw = dict(wx)
        else:
            cw = {**wx, **{k: loc[k] for k in ("Hs", "Tp", "wave_dir", "wind10") if k in loc}}
        key = tuple(repr(cw.get(k)) for k in ("Hs", "Tp", "wave_dir", "wind10"))
        if key in keys:
            continue
        keys.add(key)
        cands.append(cw)
    return cands


def wind_delta_radius(chance_mode: str, weather_unc: "WeatherUncertainty", eps: float,
                      budget_gamma: float = BUDGET_GAMMA_DEFAULT,
                      saa_n: int = 200, saa_seed: int = 12345) -> float:
    r"""10m 风矢量扰动范数的收紧半径 r。更新(采纳外部审计 6.4)明确各族【语义并不相同】,
    只有 drcc 是分布无关的人口概率界, 其余为消融基线, 不得在论文/证书中混称"P(|Δw|>r)≤eps":
      drcc  : r=|b|+√(tr Σ_w/eps)。由 |Δw|≤|b|+|Δw−b| 与对 ‖Δw−b‖²(均值0, E‖·‖²=trΣ)
              的 Markov 不等式 ⇒ 对【任意】二阶矩匹配的分布 P(|Δw|>r) ≤ eps(人口界 ✓)。
      saa   : 高斯重构样本 |Δw| 的经验 (1−eps) 分位 —— 仅样本内经验覆盖; 对重尾真分布
              (回放用 t3)无人口保证(实测可违反, 见审计 MC)。语义 = empirical。
      budget: r=|b|+Γ·√λmax —— 不确定集半径, 与 eps 无关, 无概率语义。semantics = set。
      box   : r=|b|+M·√tr —— 同为不确定集半径, 无一般 eps 保证。semantics = set。
    用途: ① 空速机会约束(r_air, 使用前须按对数风廓线升尺度到巡航高度, 审计#6);
          ② 着舰门风上侧收紧(r_gate, 门以 10m 风判定, 不升尺度, 审计#5)。"""
    if weather_unc is None:
        return 0.0
    _b = float(np.linalg.norm(np.asarray(weather_unc.wind_bias, float)))
    _cov = np.asarray(weather_unc.wind_cov, float)
    _tr = max(float(np.trace(_cov)), 0.0)
    eps = _strict_probability(eps)
    if chance_mode == "saa":
        rng_a = np.random.default_rng(saa_seed + 7)
        try:
            Lc = np.linalg.cholesky(0.5 * (_cov + _cov.T) + 1e-12 * np.eye(2))
        except np.linalg.LinAlgError:
            Lc = np.diag(np.sqrt(np.maximum(np.diag(_cov), 0.0)))
        dw = (rng_a.standard_normal((saa_n, 2)) @ Lc.T) + np.asarray(weather_unc.wind_bias, float)
        return float(np.quantile(np.linalg.norm(dw, axis=1), 1.0 - eps))
    if chance_mode == "budget":
        lam = float(np.max(np.linalg.eigvalsh(0.5 * (_cov + _cov.T)))) if _tr > 0 else 0.0
        return _b + budget_gamma * math.sqrt(max(lam, 0.0))
    if chance_mode == "box":
        return _b + BOX_WEATHER_MULT * math.sqrt(_tr)
    return _b + math.sqrt(_tr / eps)   # drcc(二维范数人口界; 仅供确实需要范数事件的约束)


def scalar_weather_radius(chance_mode: str, std: float, eps: float,
                          budget_gamma: float = BUDGET_GAMMA_DEFAULT,
                          saa_n: int = 200, saa_seed: int = 12345,
                          kappa_fn=None) -> float:
    """标量天气残差的单侧收紧半径，不包含偏置。

    DRCC 使用当前实验选定的单侧 ``kappa``（Cantelli/VP/Gaussian 对照）；SAA、预算鲁棒、
    box 与现有基线语义一致。该函数用于标量风速大小与方向投影，禁止用二维 trace 替代。
    """
    sd = max(float(std), 0.0)
    eps = _strict_probability(eps)
    if chance_mode == "saa":
        rng = np.random.default_rng(int(saa_seed) + 101)
        return float(np.quantile(sd * rng.standard_normal(max(int(saa_n), 2)), 1.0 - eps))
    if chance_mode == "budget":
        return float(budget_gamma) * sd
    if chance_mode == "box":
        return float(BOX_WEATHER_MULT) * sd
    kappa_eval = kappa if kappa_fn is None else kappa_fn
    return float(kappa_eval(eps)) * sd


def wind_speed_upper_shift(chance_mode: str, weather_unc: "WeatherUncertainty", eps: float,
                           budget_gamma: float = BUDGET_GAMMA_DEFAULT,
                           saa_n: int = 200, saa_seed: int = 12345,
                           kappa_fn=None) -> float:
    """``|w_true|-|w_nominal|`` 的上侧分位/鲁棒位移。"""
    if weather_unc is None:
        return 0.0
    return (float(getattr(weather_unc, "wind_speed_bias", 0.0))
            + scalar_weather_radius(chance_mode,
                                    float(getattr(weather_unc, "wind_speed_std", 0.0)), eps,
                                    budget_gamma=budget_gamma, saa_n=saa_n, saa_seed=saa_seed,
                                    kappa_fn=kappa_fn))


def wind_speed_lower_shift(chance_mode: str, weather_unc: "WeatherUncertainty", eps: float,
                           budget_gamma: float = BUDGET_GAMMA_DEFAULT,
                           saa_n: int = 200, saa_seed: int = 12345,
                           kappa_fn=None) -> float:
    """标量风速残差的保守下侧位移（可为负）。"""
    if weather_unc is None:
        return 0.0
    return (float(getattr(weather_unc, "wind_speed_bias", 0.0))
            - scalar_weather_radius(chance_mode,
                                    float(getattr(weather_unc, "wind_speed_std", 0.0)), eps,
                                    budget_gamma=budget_gamma, saa_n=saa_n, saa_seed=saa_seed,
                                    kappa_fn=kappa_fn))


def route_airspeed_projection_check(nom: dict, p: M.Params,
                                    weather_unc: "WeatherUncertainty",
                                    eps_air: float, chance_mode: str = "drcc",
                                    budget_gamma: float = BUDGET_GAMMA_DEFAULT,
                                    saa_n: int = 200, saa_seed: int = 12345,
                                    include_return: bool = True, kappa_fn=None) -> dict:
    r"""逐航段可控性机会约束。

    对每条航段把巡航高度风残差投影到沿航向和横航向。总路线预算 ``eps_air`` 先在航段
    间 Bonferroni 分配，再在沿向下尾与横向双侧事件之间分配。判据直接验证：在最大空速
    内是否仍能抵消横风并保持 ``v_floor`` 以上的前进地速。它与 ``leg_kinematics`` 的
    15/23 m/s 分段控制策略描述同一个物理可控性问题，不再使用
    ``nominal_required_airspeed + ||Delta w||`` 的固定全局关闭式。
    """
    records = list(nom.get("leg_air_records", []) or [])
    if not include_return:
        records = [r for r in records if not bool(r.get("is_return", False))]
    if weather_unc is None or not records:
        margin = float(p.v_air_max) - float(nom.get("max_required_airspeed_ms", 0.0))
        return dict(ok=bool(nom.get("speed_feasible", True)), margin_ms=margin,
                    worst_leg=None, leg_checks=[])
    alpha = float(M.wind_at_height(1.0, p.z_cruise, p.z0))
    mu = alpha * np.asarray(weather_unc.wind_bias, float)
    cov = (alpha ** 2) * np.asarray(weather_unc.wind_cov, float)
    nleg = max(len(records), 1)
    eps_leg = _strict_probability(float(eps_air), "eps_air") / nleg
    eps_along = eps_leg / 2.0
    eps_cross_side = eps_leg / 4.0  # 横向绝对值拆成正、负两侧
    checks = []
    for rec in records:
        e = np.asarray(rec["direction"], float)
        n = np.asarray(rec["normal"], float)
        wa = float(rec["wind_along_ms"])
        wc = float(rec["wind_cross_signed_ms"])
        ma = float(e @ mu); mc = float(n @ mu)
        sda = math.sqrt(max(float(e @ cov @ e), 0.0))
        sdc = math.sqrt(max(float(n @ cov @ n), 0.0))
        ra = scalar_weather_radius(chance_mode, sda, eps_along,
                                   budget_gamma=budget_gamma, saa_n=saa_n,
                                   saa_seed=saa_seed + 13 * int(rec["leg_index"]),
                                   kappa_fn=kappa_fn)
        rc = scalar_weather_radius(chance_mode, sdc, eps_cross_side,
                                   budget_gamma=budget_gamma, saa_n=saa_n,
                                   saa_seed=saa_seed + 17 * int(rec["leg_index"]),
                                   kappa_fn=kappa_fn)
        cross_safe = abs(wc + mc) + max(float(rc), 0.0)
        cross_margin = float(p.v_air_max) - cross_safe
        cross_ok = bool(cross_safe <= float(p.v_air_max))
        if not cross_ok:
            forward_safe = -float("inf")
            forward_margin = -float("inf")
            forward_ok = False
        else:
            along_lower = wa + ma - max(float(ra), 0.0)
            forward_safe = along_lower + math.sqrt(max(float(p.v_air_max) ** 2 - cross_safe ** 2, 0.0))
            forward_margin = forward_safe - 1.0  # leg_kinematics requires strictly > v_floor
            forward_ok = bool(forward_safe > 1.0)
        margin = min(cross_margin, forward_margin)
        checks.append(dict(
            leg_index=int(rec["leg_index"]), is_return=bool(rec["is_return"]),
            eps_leg=float(eps_leg), nominal_along_ms=wa, nominal_cross_ms=wc,
            mean_along_ms=ma, mean_cross_ms=mc,
            std_along_ms=sda, std_cross_ms=sdc,
            radius_along_ms=float(ra), radius_cross_ms=float(rc),
            safe_crosswind_ms=float(cross_safe), safe_forward_speed_ms=float(forward_safe),
            crosswind_margin_ms=float(cross_margin), forward_margin_ms=float(forward_margin),
            margin_ms=float(margin), ok=bool(cross_ok and forward_ok)))
    worst = min(checks, key=lambda x: x["margin_ms"])
    return dict(ok=bool(nom.get("speed_feasible", True) and all(x["ok"] for x in checks)),
                margin_ms=float(worst["margin_ms"]), worst_leg=int(worst["leg_index"]),
                leg_checks=checks)


def mission_risk_allocation(p: M.Params, weather_on: bool) -> dict:
    """Return the active per-sortie Bonferroni allocation.

    The recovery target is the ship prediction at the discrete decision horizon h;
    its realized spatial uncertainty is already represented by xi_h.  Sensor-level
    terminal acquisition error is outside this finite model and is therefore not a
    separate chance event.  Weather-specific events are included only when a formal
    weather ambiguity object is active.
    """
    out = {
        "energy": float(p.eps_E),
        "time": float(p.eps_T),
    }
    if weather_on:
        out.update({
            "wave_gate": float(p.eps_cap),
            "wind_gate": float(getattr(p, "eps_gate", p.eps_cap)),
            "route_airspeed": float(getattr(p, "eps_air", p.eps_cap)),
            "dock_reserve": float(getattr(p, "eps_dock", p.eps_cap)),
            "stern_escort": float(getattr(p, "eps_escort", p.eps_cap)),
        })
    return out


def mission_eps_budget(p: M.Params, weather_on: bool) -> float:
    r"""Per-sortie joint failure upper bound under the active event split.

    The active split may conservatively sum to less than ``mission_failure_budget``;
    formal feasibility only requires that the exact binary64-as-real sum does not
    exceed the declared mission budget.
    """
    return float(sum(mission_risk_allocation(p, weather_on).values()))


def mission_budget_compliant(p: M.Params, weather_on: bool, tol: float = 0.0) -> bool:
    """Exact binary64-as-real Bonferroni budget check; ``tol`` is ignored for compatibility."""
    parts = mission_risk_allocation(p, weather_on)
    total = sum((Fraction.from_float(float(v)) for v in parts.values()), Fraction(0))
    budget = Fraction.from_float(float(getattr(p, "mission_failure_budget", 0.05)))
    return bool(total <= budget)


def mean_relax_free(xi_amb, weather_unc, tol: float = 0.0) -> bool:
    r"""Return whether signed mean terms are exactly zero in the finite model.

    This predicate guards *exclusion* bounds used by formal pricing.  Treating a
    small favorable mean as zero can make those bounds unsafe and remove a truly
    feasible route.  Therefore the certificate path uses component-wise exact
    binary64 zero tests; ``tol`` is retained only for API compatibility and is
    intentionally ignored.
    """
    def _all_exact_zero(values) -> bool:
        try:
            arr = np.asarray(values, float).ravel()
        except Exception:
            return False
        return bool(np.all(np.isfinite(arr)) and all(float(v) == 0.0 for v in arr))

    if weather_unc is not None:
        if isinstance(weather_unc, WeatherAmbiguity):
            try:
                if any(not _all_exact_zero(w.wind_bias) for w in weather_unc.by_h.values()):
                    return False
            except Exception:
                return False
        else:
            try:
                if not _all_exact_zero(weather_unc.wind_bias):
                    return False
            except Exception:
                return False
    if xi_amb is None:
        return False
    try:
        for cell in xi_amb.cells.values():
            if not _all_exact_zero(cell.mu):
                return False
    except Exception:
        return False
    return True

def _route_weather_at_h(route: "Route", h: float, fallback: dict) -> dict:
    """Resolve launch/recovery nominal weather from the route's launch-time contract."""
    ship = getattr(route, "ship", None)
    if ship is not None and hasattr(ship, "weather_at_h"):
        return ship.weather_at_h(float(h), fallback=fallback)
    return dict(fallback)


def _rejected_route_diag(route: Route, h: float, p: M.Params, reason: str,
                         recovery_state: str = "unknown", recovery_state_source: str = "unknown",
                         error: str | None = None) -> dict:
    """构造下游可安全消费的候选拒绝诊断。"""
    bad = -1.0e30
    risk_budget = mission_eps_budget(p, False)
    failure_flags = {
        "missing_recovery_state": reason == "missing_recovery_state_support",
        "forbidden_recovery_state": reason == "recovery_state_forbidden",
        "missing_recovery_prediction": reason == "missing_recovery_prediction",
        "missing_recovery_weather": reason == "missing_recovery_weather",
        "missing_xi_cell": reason == "missing_xi_support",
        "risk_budget_invalid": reason == "mission_risk_budget_invalid",
        "escort_airspeed_failed": reason == "escort_speed_infeasible",
        "landing_gate_failed": reason == "landing_gate_closed",
        "landing_wind_failed": False, "landing_wave_failed": False,
        "nominal_airspeed_failed": False, "route_airspeed_failed": False,
        "energy_drcc_failed": False, "time_drcc_failed": False,
        "nominal_time_failed": False,
    }
    return dict(feasible=False, reason=str(reason), primary_reason=str(reason),
                failure_flags=failure_flags,
                margins=dict(energy_Wh=bad, time_s=bad, route_airspeed_ms=bad,
                             landing_wind_ms=bad, escort_airspeed_ms=bad),
                error=(str(error) if error else None),
                h=h, c=str(route.ship.c_state), n=0, margin_E=bad, margin_T=bad,
                M_omega=bad, slack_T=bad, gate=0, r_cap_eff=0.0,
                recovery_target_model=str(getattr(p, "recovery_target_model", "discrete_horizon_ship_prediction")),
                terminal_sensor_error_mode=str(getattr(p, "terminal_sensor_error_mode", "out_of_scope")),
                recovery_state=str(recovery_state), recovery_state_source=str(recovery_state_source),
                recovery_state_ok=False,
                E0=1.0e30, E_flight_Wh=1.0e30, E_escort_Wh=0.0, E_plan_Wh=1.0e30,
                E_soc_required_Wh=1.0e30, E_uncertainty_buffer_Wh=0.0, T0=1.0e30,
                d_ret0=1.0e30, E_wait=0.0, T_flight=1.0e30, T_escort_s=0.0,
                escort_required_airspeed_ms=0.0, escort_speed_feasible=False,
                escort_margin_ms=bad, E_dock_Wh=0.0, t_dock_s=0.0, t_dock_wait_s=0.0,
                gate_weather_proof=dict(all_candidates_checked=False, switching_proven_safe=False),
                speed_feasible=False, airspeed_margin_ms=None, max_required_airspeed_ms=0.0,
                speed_fail_leg=-1, sigma_trace=0.0, Hs_eff=float("nan"),
                w10_gate_eff=float("nan"), eps_gate_used=float(getattr(p, "eps_gate", p.eps_cap)),
                time_contract=TIME_CONTRACT, time_contract_id=TIME_CONTRACT,
                wait_is_recourse=WAIT_IS_RECOURSE, dock_risk_contract=DOCK_RISK_CONTRACT,
                time_flight_nom_s=1.0e30, time_inspection_s=0.0, time_dock_nom_s=0.0,
                time_core_nom_s=1.0e30, time_wait_nom_s=0.0,
                time_xi_mean_shift_s=0.0, time_xi_std_term_s=0.0,
                time_weather_mean_shift_s=0.0, time_weather_std_term_s=0.0,
                time_geometry_remainder_s=0.0, time_drcc_tightening_s=0.0,
                time_safe_core_s=1.0e30, time_wait_safe_s=0.0,
                nominal_time_margin_s=bad, time_drcc_margin_s=bad,
                chance_mode="rejected", mission_failure_budget=float(p.mission_failure_budget),
                mission_eps_budget=float(risk_budget),
                mission_budget_compliant=mission_budget_compliant(p, False),
                risk_allocation=mission_risk_allocation(p, False),
                risk_budget_unallocated=float(p.mission_failure_budget) - float(risk_budget),
                soc_correction=getattr(p, "soc_correction", "none"),
                soc_risk_allocation=getattr(p, "soc_risk_allocation", "fixed"))

def route_feasible_at_h(route: Route, h: int, p: M.Params, wx: dict,
                        xi_amb: "M.XiAmbiguity",
                        weather_unc: "WeatherUncertainty" = None,
                        chance_mode: str = "drcc",
                        budget_gamma: float = BUDGET_GAMMA_DEFAULT,
                        saa_n: int = 200, saa_seed: int = 12345,
                        formal: bool = False, deadline=None, kappa_fn=None,
                        risk_policy: RiskPolicy | None = None) -> dict:
    """固定回收时长 h, 判定该路由的 DRCC(能量+时间, Bonferroni 拆分)+ 着舰门。
    决策依赖核心: 这里用【该 h 的】模糊集 cell=(μ_h,Σ_h) 与【该 h 的】预测回收点。

    三阶段终端链路(doc_model §7.2):
      1) 离散回收目标: h 是路线决策；计划回收点为 ``ship.predicted_at(h)``，实现位置
         不确定性由同一 h 的真实 AIS/CV 船位预测误差 xi_h 统一描述。
      2) 船尾伴飞: 到达实现回收区后维持船体坐标系下的船尾等待点；转弯或未解析
         的回收状态拒绝。传感器级 acquisition error 明确在当前有限模型范围之外。
      3) 完整着舰: ``dock_reserve`` 统一计量对准、等待甲板相位、最终下降、接地与复飞储备。
      能量和时间预算均扣除状态相关的 ``E_dock/t_dock``，最终下降不再单独计量。

    weather_unc(可选, model.md §14): 完整模型多源不确定性的风、浪两源(与 ξ 并行, 非"扩展"):
      - 风预报误差 → 地速 → 能耗/时间, 用联合 SOC([ξ;Δw] 堆叠);
      - 浪 Hs 预报误差 → 着舰门收紧 Hs_eff = Hs + bias + κ(ε)·σ_hs(Cantelli 单边)。
      为 None 时退化为仅 ξ 的主线 DRCC(向后兼容)。

    chance_mode(论文 baseline 对照; 默认 'drcc' = 本文方法, 向后兼容):
      'drcc'   : 矩模糊集 DRCC-SOC(本文; κ(ε) 由 KAPPA_MODES 选 Cantelli/VP/高斯);
      'saa'    : SAA 样本机会约束(对标 multi-visit 2024; 用 ξ 矩重建样本取 (1−ε) 经验分位);
      'budget' : Bertsimas–Sim 椭球预算鲁棒(对标 Robust UAV-USV 2025; 最坏点硬约束, 预算 Γ=budget_gamma);
      'box'    : 【更新】支持集鲁棒(经典 RO 最保守端: ξ 经验支持球 support_radius, 天气 3σ 盒代理)。
    各模式【只替换约束左端判据】, 物理层/能耗/时间/着舰门/空速完全一致 ⇒ 公平对比。
    """
    _check_deadline(deadline)
    rp = _risk_policy_from_inputs(risk_policy, kappa_fn)
    kappa_eval = rp.one_sided
    p.validate_contract(formal=formal)
    h_s = 60.0 * float(h)
    try:
        recovery_state, recovery_state_source = route.ship.recovery_state_at(float(h))
    except (KeyError, ValueError) as exc:
        return _rejected_route_diag(
            route, h, p, "missing_recovery_state_support",
            str(getattr(route.ship, "c_state", "unknown")), "missing_predicted_by_h",
            str(exc))
    recovery_state_ok = str(recovery_state) not in set(getattr(p, "recovery_forbidden_states", ()))
    if not recovery_state_ok:
        return _rejected_route_diag(route, h, p, "recovery_state_forbidden",
                                    recovery_state, recovery_state_source)

    _check_deadline(deadline)
    try:
        weather_unc = _resolve_weather_unc(weather_unc, h)
    except (KeyError, ValueError) as exc:
        return _rejected_route_diag(
            route, h, p, "missing_weather_uncertainty_support",
            recovery_state, recovery_state_source, str(exc))
    try:
        # xi 只描述外层船位预测误差；严格禁止 horizon 外推和跨状态回退。
        cell = _xi_cell_strict(xi_amb, float(h), route.ship.c_state)
    except (KeyError, ValueError) as exc:
        return _rejected_route_diag(route, h, p, "missing_xi_support",
                                    recovery_state, recovery_state_source, str(exc))

    weather_on = weather_unc is not None
    risk_budget = mission_eps_budget(p, weather_on)
    risk_budget_ok = mission_budget_compliant(p, weather_on)
    try:
        wx_launch = _route_weather_at_h(route, 0.0, wx)
        wx_recovery = _route_weather_at_h(route, float(h), wx_launch)
    except (KeyError, ValueError) as exc:
        return _rejected_route_diag(route, h, p, "missing_recovery_weather",
                                    recovery_state, recovery_state_source, str(exc))

    # Return-speed recourse currently has a complete population-moment certificate only for
    # the DRCC family (including nominal/Gaussian/Cantelli/VP through the selected kappa).
    # SAA/set-robust counterparts must not silently fall back to wait-only, because that would
    # compare different physical policies under one provenance signature.
    if getattr(p, "speed_adjustable", False) and chance_mode != "drcc":
        return _rejected_route_diag(
            route, h, p, "speed_recourse_chance_mode_not_certified",
            recovery_state, recovery_state_source,
            f"chance_mode={chance_mode}; use --time-recourse wait_only for this baseline")
    # 可变速度分支与固定速度分支共享同一离散回收目标、回收状态和风险预算语义。
    if getattr(p, "speed_adjustable", False) and chance_mode == "drcc":
        d = _feasible_speed_adjustable(route, h, p, wx_launch, cell, weather_unc,
                                       wx_recovery=wx_recovery, risk_policy=rp,
                                       deadline=deadline)
        E_plan = float(d.get("E0", 1.0e30) + d.get("E_dock_Wh", 0.0))
        margin_E = float(d.get("margin_E", -1.0e30))
        d.update(
            feasible=bool(d.get("feasible", False) and recovery_state_ok and risk_budget_ok),
            reason=(d.get("reason") if not d.get("feasible", False)
                    else ("mission_risk_budget_invalid" if not risk_budget_ok else None)),
            recovery_state=str(recovery_state), recovery_state_source=str(recovery_state_source),
            recovery_state_ok=bool(recovery_state_ok),
            recovery_target_model=str(getattr(p, "recovery_target_model", "discrete_horizon_ship_prediction")),
            terminal_sensor_error_mode=str(getattr(p, "terminal_sensor_error_mode", "out_of_scope")),
            E_flight_Wh=float(d.get("E_flight_Wh", d.get("E0", 1.0e30))),
            E_escort_Wh=float(d.get("E_escort_Wh", 0.0)), E_plan_Wh=E_plan,
            E_soc_required_Wh=float(max(E_plan, p.B_use - margin_E)),
            E_uncertainty_buffer_Wh=float(max((p.B_use - margin_E) - E_plan, 0.0)),
            mission_failure_budget=float(p.mission_failure_budget),
            mission_eps_budget=float(risk_budget),
            mission_budget_compliant=bool(risk_budget_ok),
            risk_allocation=mission_risk_allocation(p, weather_on),
            risk_budget_unallocated=float(p.mission_failure_budget) - float(risk_budget),
            soc_correction=getattr(p, "soc_correction", "none"), chance_mode=chance_mode)
        flags = dict(d.get("failure_flags", {}))
        flags["risk_budget_invalid"] = not risk_budget_ok
        flags["forbidden_recovery_state"] = not recovery_state_ok
        d["failure_flags"] = flags
        d["M_omega"] = float(min(float(d.get("M_omega", -1.0e30)),
                                  float(d.get("escort_margin_ms", 1.0e30))))
        return d

    # 着舰门用【回收点】天气(更新: 规划/回放共用 recovery_gate_wx, 同口径)。
    # 更新(审计修复#8-等待/对接): 对接储备须先于名义评估计算 —— T_wait 要扣它;
    # P_rec 可直接由 ship.predicted_at(h) 得到, 不依赖 nom。
    try:
        P_rec = route.ship.predicted_at(float(h))
    except (KeyError, ValueError) as exc:
        return _rejected_route_diag(route, h, p, "missing_recovery_prediction",
                                    recovery_state, recovery_state_source, str(exc))
    gate_wx = recovery_gate_wx(route, wx, P_rec)   # 预测点最近场(仅诊断参照, 判据不再单点依赖它)
    # ================= 更新(C-01, 致命修复) =================
    # 旧口径只在【预测】回收点选一次 gate 天气场; 但 recovery_gate_wx 是关于实现回收点的
    # 不连续最近邻选择器, ξ(矩模糊集, 无有界支持)可把实现回收点推过最近邻边界 ⇒ 实际
    # 使用完全不同的 Hs/wind10 —— 该离散切换事件此前无任何概率预算(可执行反例: 两点
    # 分布下门失败概率 0.5 而证书为真)。修复: 对该路线【所有可能实现】的候选场
    # (gate_weather_candidates)做最坏情况检查:
    #   · 着舰门须对全部候选开启;  · t_dock 取候选最大(时间预算);
    #   · E_dock 取候选最大(能量预算);  · 等待窗按候选最小 t_dock 计
    #     (E_plan = E_flight + P_escort·max(0, W−min t_dock) + max E_dock ≥ 任一候选实现的
    #      E_flight + P_escort·max(0, W−t_dock_c) + E_dock_c —— 逐项支配, 排除方向保真)。
    # 无 wx_local ⇒ 候选=[wx], 与旧口径逐位一致。运行时证据 gate_weather_proof 由
    # 【实际检查的候选逐一累积】(证书据此判定 gate_weather_switch_proven_safe)。
    cand_wx = gate_weather_candidates(route, wx_recovery)
    eps_gate = float(getattr(p, "eps_gate", p.eps_cap))
    eps_dock = float(getattr(p, "eps_dock", p.eps_cap))   # 更新 H-03: 对接风下侧独立预算
    # 着舰门判的是标量风速大小，不再复用二维风矢量范数 Markov 半径。
    # 上侧用于风门，下侧用于对接储备功率；两者均基于 d|w| 的独立标量残差矩。
    r_gate = wind_speed_upper_shift(chance_mode, weather_unc, eps_gate,
                                    budget_gamma=budget_gamma, saa_n=saa_n, saa_seed=saa_seed,
                                    kappa_fn=kappa_eval)
    r_dock_low = wind_speed_lower_shift(chance_mode, weather_unc, eps_dock,
                                        budget_gamma=budget_gamma, saa_n=saa_n, saa_seed=saa_seed + 5,
                                        kappa_fn=kappa_eval)
    _hs_draw = None
    if weather_unc is not None and chance_mode == "saa":
        _rng_hs = np.random.default_rng(saa_seed + 2)
        _hs_draw = weather_unc.hs_bias + weather_unc.hs_std * _rng_hs.standard_normal(saa_n)

    def _hs_eff_of(cw_hs: float) -> float:
        if weather_unc is None:
            return float(cw_hs)
        if chance_mode == "saa":
            return float(np.quantile(cw_hs + _hs_draw, 1.0 - p.eps_cap))
        if chance_mode == "budget":
            return float(cw_hs + weather_unc.hs_bias + budget_gamma * weather_unc.hs_std)
        if chance_mode == "box":   # 更新: 支持集最坏(天气侧 3σ 盒代理)
            return float(cw_hs + weather_unc.hs_bias + BOX_WEATHER_MULT * weather_unc.hs_std)
        return float(cw_hs + weather_unc.hs_bias + kappa_eval(p.eps_cap) * weather_unc.hs_std)  # drcc

    _check_deadline(deadline)
    _checked = []
    for cw in cand_wx:
        _check_deadline(deadline)
        _c_hs = _hs_eff_of(float(cw["Hs"]))
        _c_w10 = float(cw["wind10"])
        # 更新(H-03): 风下侧统一用 r_dock(drcc 族有人口概率语义; 其余按各自家族口径),
        # 旧 κ(eps_cap)·√λmax 口径废弃 —— 该事件此前与浪门共用配额且无独立证明。
        _c_w10_low = max(0.0, _c_w10 + r_dock_low)
        _c_motion = M.deck_motion(_c_hs, cw.get("Tp", 2.1),
                                  cw.get("wave_dir", 200.0) - cw.get("ship_heading", 0.0), p)
        _c_td, _c_Ed = M.dock_reserve(p, _c_motion, _c_w10_low)
        _c_w10_gate = _c_w10 + r_gate
        _c_L, _c_reff = M.landing_gate(_c_hs, cw.get("Tp", 2.1), cw.get("wave_dir", 200.0),
                                       cw.get("ship_heading", 0.0), _c_w10_gate, p)
        _wave_ok = bool(_c_motion["heave"] <= p.s_heave_max
                        and _c_motion["roll"] <= p.s_roll_max
                        and _c_motion["pitch"] <= p.s_pitch_max
                        and _c_hs <= p.Hs_op)
        _wind_ok = bool(_c_w10_gate <= p.w_land_max)
        _checked.append(dict(Hs_eff=_c_hs, w10_low=_c_w10_low, t_dock=_c_td, E_dock=_c_Ed,
                             w10_gate=_c_w10_gate, L=int(_c_L), r_eff=float(_c_reff),
                             wave_ok=_wave_ok, wind_ok=_wind_ok))
    _all_checked = bool(len(_checked) == len(cand_wx) and len(_checked) >= 1)
    t_dock = max(c["t_dock"] for c in _checked)          # 最坏对接时长(时间预算侧)
    t_dock_wait = min(c["t_dock"] for c in _checked)     # 最短对接 ⇒ 最长等待(能量预算侧)
    E_dock = max(c["E_dock"] for c in _checked)          # 最坏对接能耗
    Hs_eff = max(c["Hs_eff"] for c in _checked)
    w10_gate = max(c["w10_gate"] for c in _checked)
    gate_all_open = all(c["L"] == 1 for c in _checked)
    r_eff = min(c["r_eff"] for c in _checked)
    gate_weather_proof = dict(
        selector="nearest-route-turbine",
        candidate_count=int(len(cand_wx)),
        candidates_checked=int(len(_checked)),
        all_candidates_checked=_all_checked,
        switching_proven_safe=bool(_all_checked),
        worst_case_aggregation="gate=∀open, t_dock=max, E_dock=max, wait=min-t_dock")
    # 名义评估在 t_dock 已知后进行，船尾伴飞窗 = 60h − T_flight − t_dock。
    # 对接期不再重复计伴飞能耗；E_dock 仍单独从 b_E 扣除。伴飞窗
    # 用候选最小 t_dock(见上支配论证); 单候选时 t_dock_wait == t_dock, 与旧口径逐位一致。
    nom = route_nominal_ET(route, h, p, wx_launch, t_dock_s=t_dock_wait)
    # Fixed touchdown contract: dock reserve is already risk-adjusted and enters the
    # core exactly once; nominal stern waiting is recourse and is excluded from b_T.
    time_flight_nom_s = float(nom.get("time_flight_nom_s", nom["T0"] - nom.get("time_inspection_s", 0.0)))
    time_inspection_s = float(nom.get("time_inspection_s", 0.0))
    time_dock_nom_s = float(t_dock)
    time_core_nom_s = time_flight_nom_s + time_inspection_s + time_dock_nom_s
    assert abs(time_core_nom_s - (float(nom["T0"]) + time_dock_nom_s)) <= TIME_TOL_S
    nominal_time_margin_s = h_s - time_core_nom_s
    b_T = nominal_time_margin_s
    a_T = np.asarray(nom["c_T"] * nom["g"], float)
    if a_T.shape != (2,) or not np.all(np.isfinite(a_T)):
        raise ValueError("time xi gradient must be a finite 2-vector in seconds/metre")

    # 完整计划能量关于返程距离是两条仿射分支的最大值。分别约束两条分支，
    # 并把能量事件预算 eps_E 在分支间二分；由 union bound 仍保证总能量失败概率 ≤ eps_E。
    eps_E_branch = float(p.eps_E) / 2.0
    energy_branches = [
        ("noescort", float(nom["E_branch_noescort_Wh"]), float(nom["c_E_noescort"])),
        ("escort", float(nom["E_branch_escort_affine_Wh"]), float(nom["c_E_escort"])),
    ]
    branch_margins = {}

    def _branch_margin(name, e0_branch, c_branch):
        b = float(p.B_use) - float(e0_branch) - float(E_dock)
        a = float(c_branch) * nom["g"]
        if chance_mode == "saa":
            if weather_unc is None:
                return _saa_margin(a, b, samples, eps_E_branch)
            aw, _ = route_wind_sensitivity(route, h, p, wx_launch, t_dock_s=t_dock_wait,
                                           energy_branch=name)
            return _saa_margin_joint(a, aw, b, samples, weather_unc,
                                     eps_E_branch, seed=saa_seed)
        if chance_mode == "budget":
            if weather_unc is None:
                return _budget_margin(a, b, cell, budget_gamma)
            aw, _ = route_wind_sensitivity(route, h, p, wx_launch, t_dock_s=t_dock_wait,
                                           energy_branch=name)
            return _budget_margin_joint(a, aw, b, cell, weather_unc, budget_gamma)
        if chance_mode == "box":
            if weather_unc is None:
                return _box_margin(a, b, cell)
            aw, _ = route_wind_sensitivity(route, h, p, wx_launch, t_dock_s=t_dock_wait,
                                           energy_branch=name)
            return _box_margin_joint(a, aw, b, cell, weather_unc)
        # DRCC branch
        _geo = (getattr(p, "soc_correction", "none") == "geo2d")
        _sl = float(getattr(p, "soc_share_lin", 0.6))
        _sw = float(getattr(p, "soc_share_wind", 0.2))
        _alloc = str(getattr(p, "soc_risk_allocation", "fixed"))
        if weather_unc is None:
            if _geo:
                return _soc_margin_geo2d(a, b, cell, eps_E_branch, nom["d_ret0"], _sl, _alloc, rp)
            return _soc_margin(a, b, cell, eps_E_branch, rp)
        aw, _ = route_wind_sensitivity(route, h, p, wx_launch, t_dock_s=t_dock_wait,
                                       energy_branch=name)
        if _geo:
            return _soc_margin_geo2d_joint(a, aw, b, cell, weather_unc,
                                           eps_E_branch, nom["d_ret0"], _sl, _sw, _alloc, rp)
        return _soc_margin_joint(a, aw, b, cell, weather_unc, eps_E_branch, rp)

    if chance_mode == "saa":
        samples = _saa_samples_from_cell(cell, n_samp=saa_n, seed=saa_seed)
    _check_deadline(deadline)
    for _bn, _be, _bc in energy_branches:
        _check_deadline(deadline)
        branch_margins[_bn] = float(_branch_margin(_bn, _be, _bc))
    mE = min(branch_margins.values())

    # 时间事件只有一个仿射分支，继续使用完整 eps_T。
    if chance_mode == "saa":
        if weather_unc is None:
            mT = _saa_margin(a_T, b_T, samples, p.eps_T)
        else:
            _, aT_w = route_wind_sensitivity(route, h, p, wx_launch, t_dock_s=t_dock_wait)
            mT = _saa_margin_joint(a_T, aT_w, b_T, samples, weather_unc,
                                   p.eps_T, seed=saa_seed)
    elif chance_mode == "budget":
        if weather_unc is None:
            mT = _budget_margin(a_T, b_T, cell, budget_gamma)
        else:
            _, aT_w = route_wind_sensitivity(route, h, p, wx_launch, t_dock_s=t_dock_wait)
            mT = _budget_margin_joint(a_T, aT_w, b_T, cell, weather_unc, budget_gamma)
    elif chance_mode == "box":
        if weather_unc is None:
            mT = _box_margin(a_T, b_T, cell)
        else:
            _, aT_w = route_wind_sensitivity(route, h, p, wx_launch, t_dock_s=t_dock_wait)
            mT = _box_margin_joint(a_T, aT_w, b_T, cell, weather_unc)
    else:
        _geo = (getattr(p, "soc_correction", "none") == "geo2d")
        _sl = float(getattr(p, "soc_share_lin", 0.6))
        _sw = float(getattr(p, "soc_share_wind", 0.2))
        _alloc = str(getattr(p, "soc_risk_allocation", "fixed"))
        if weather_unc is None:
            mT = (_soc_margin_geo2d(a_T, b_T, cell, p.eps_T, nom["d_ret0"], _sl, _alloc, rp)
                  if _geo else _soc_margin(a_T, b_T, cell, p.eps_T, rp))
        else:
            _, aT_w = route_wind_sensitivity(route, h, p, wx_launch, t_dock_s=t_dock_wait)
            mT = (_soc_margin_geo2d_joint(a_T, aT_w, b_T, cell, weather_unc,
                                          p.eps_T, nom["d_ret0"], _sl, _sw, _alloc, rp)
                  if _geo else _soc_margin_joint(a_T, aT_w, b_T, cell, weather_unc, p.eps_T, rp))
    # Structured time tightening.  All components are seconds and close exactly to
    # ``b_T-mT``; any non-linear/baseline residual is isolated as geometry remainder.
    time_drcc_tightening_s = float(b_T - mT)
    xi_mean = xi_std = weather_mean = weather_std = 0.0
    geometry_remainder = 0.0
    xi_geo_total = 0.0
    eps_time_weather = 0.0
    eps_time_xi = float(p.eps_T)
    eps_time_along = float("nan")
    eps_time_cross = float("nan")
    geo_risk_allocation_mode = str(getattr(p, "soc_risk_allocation", "fixed"))
    geo_risk_allocation_contract = GEO_RISK_ALLOCATION_CONTRACT
    aT_w_diag = np.zeros(2, dtype=float)
    if weather_unc is not None:
        _, aT_w_diag = route_wind_sensitivity(route, h, p, wx_launch,
                                              t_dock_s=t_dock_wait)
        aT_w_diag = np.asarray(aT_w_diag, float)
        if aT_w_diag.shape != (2,) or not np.all(np.isfinite(aT_w_diag)):
            raise ValueError("weather time gradient must be finite seconds per (m/s)")
    if chance_mode == "drcc":
        geo_time = (getattr(p, "soc_correction", "none") == "geo2d")
        if geo_time:
            share_lin = float(getattr(p, "soc_share_lin", 0.6))
            share_wind = float(getattr(p, "soc_share_wind", 0.2))
            allocation_mode = str(getattr(p, "soc_risk_allocation", "fixed"))
            cT = float(np.linalg.norm(a_T))
            if weather_unc is not None:
                geo_detail = _soc_margin_geo2d_joint_details(
                    a_T, aT_w_diag, b_T, cell, weather_unc, float(p.eps_T),
                    float(nom["d_ret0"]), share_lin, share_wind, allocation_mode, rp)
                assert abs(float(geo_detail["margin"]) - float(mT)) <= 1e-6
                ew = float(geo_detail["eps_weather"]); ex = float(geo_detail["eps_xi"])
                D = float(geo_detail["bound_m"])
            else:
                if cT == 0.0:
                    ew, ex = 0.0, float(p.eps_T)
                    D = float(nom["d_ret0"])
                    geo_detail = dict(
                        bound_m=D, eps_total=ex, eps_along=0.0, eps_cross=0.0,
                        allocation_mode=str(allocation_mode), along_fraction=float("nan"),
                        std_along_m=0.0, std_cross_m=0.0,
                        mean_along_m=0.0, mean_cross_m=0.0,
                        eps_weather=0.0, eps_xi=ex, weather_mean=0.0, weather_std=0.0,
                        risk_allocation_contract=GEO_RISK_ALLOCATION_CONTRACT)
                else:
                    xi_detail = _geo2d_dist_bound_details(
                        cell, float(p.eps_T), float(nom["d_ret0"]), a_T / cT,
                        share_lin, allocation_mode, rp)
                    ew, ex = 0.0, float(p.eps_T); D = float(xi_detail["bound_m"])
                    geo_detail = dict(**xi_detail, eps_weather=0.0, eps_xi=ex,
                                      weather_mean=0.0, weather_std=0.0,
                                      risk_allocation_contract=GEO_RISK_ALLOCATION_CONTRACT)
            eps_time_weather = ew; eps_time_xi = ex
            eps_time_along = float(geo_detail.get("eps_along", float("nan")))
            eps_time_cross = float(geo_detail.get("eps_cross", float("nan")))
            xi_geo_total = cT * (D - float(nom["d_ret0"]))
            # Audit decomposition only: retain the exact geo2d total, while exposing
            # the familiar linear mean/std baseline and isolating the true nonlinear/
            # two-sided correction in geometry_remainder.  The feasibility formula is
            # unchanged and still consumes xi_geo_total exactly once.
            xi_mean = float(a_T @ np.asarray(cell.mu, float))
            qx = max(float(a_T @ cell.Sigma @ a_T), 0.0)
            xi_std = kappa_eval(float(ex)) * math.sqrt(qx) if ex > 0 else 0.0
            geometry_remainder = xi_geo_total - xi_mean - xi_std
            if weather_unc is not None:
                weather_mean = float(aT_w_diag @ np.asarray(weather_unc.wind_bias, float))
                weather_std = (kappa_eval(ew) * math.sqrt(max(float(aT_w_diag @ weather_unc.wind_cov
                                                            @ aT_w_diag), 0.0))
                               if ew > 0 else 0.0)
        else:
            xi_mean = float(a_T @ np.asarray(cell.mu, float))
            qx = max(float(a_T @ cell.Sigma @ a_T), 0.0)
            if weather_unc is None:
                xi_std = kappa_eval(float(p.eps_T)) * math.sqrt(qx)
            else:
                weather_mean = float(aT_w_diag @ np.asarray(weather_unc.wind_bias, float))
                qw = max(float(aT_w_diag @ weather_unc.wind_cov @ aT_w_diag), 0.0)
                joint_std = kappa_eval(float(p.eps_T)) * math.sqrt(qx + qw)
                qsum = qx + qw
                if qsum > 0.0:
                    xi_std = joint_std * qx / qsum
                    weather_std = joint_std * qw / qsum
            xi_geo_total = xi_mean + xi_std
    cT_diag = float(np.linalg.norm(a_T))
    if cT_diag > 1e-15:
        g_time = np.asarray(a_T, float) / cT_diag
        g_time_perp = np.array([-g_time[1], g_time[0]], float)
        xi_mean_along_m = float(g_time @ np.asarray(cell.mu, float))
        xi_mean_cross_m = float(g_time_perp @ np.asarray(cell.mu, float))
        xi_std_along_m = math.sqrt(max(float(g_time @ cell.Sigma @ g_time), 0.0))
        xi_std_cross_m = math.sqrt(max(float(g_time_perp @ cell.Sigma @ g_time_perp), 0.0))
        xi_geo_bound_extra_m = float(xi_geo_total / cT_diag) if chance_mode == "drcc" else float("nan")
    else:
        xi_mean_along_m = xi_mean_cross_m = 0.0
        xi_std_along_m = xi_std_cross_m = 0.0
        xi_geo_bound_extra_m = 0.0

    # Close the decomposition exactly.  For SAA/box/budget this residual is the
    # corresponding empirical/set robust term; for DRCC it is normally numerical zero
    # (or the explicit geo2d correction).
    geometry_remainder += (time_drcc_tightening_s
                           - (xi_mean + xi_std + weather_mean + weather_std
                              + geometry_remainder))
    decomp_sum = xi_mean + xi_std + weather_mean + weather_std + geometry_remainder
    assert abs(decomp_sum - time_drcc_tightening_s) <= max(TIME_TOL_S, 1e-6)
    time_contract = fixed_touchdown_time_accounting(
        h_s, time_core_nom_s, time_drcc_tightening_s, tol=TIME_TOL_S)
    assert abs(time_contract["time_drcc_margin_s"] - float(mT)) <= max(TIME_TOL_S, 1e-5)
    time_decomposition = dict(
        nominal_margin_s=float(time_contract["nominal_time_margin_s"]),
        xi_mean_shift_s=float(xi_mean), xi_std_term_s=float(xi_std),
        weather_mean_shift_s=float(weather_mean), weather_std_term_s=float(weather_std),
        geometry_remainder_s=float(geometry_remainder),
        xi_geo_total_s=float(xi_geo_total),
        weather_total_s=float(weather_mean + weather_std),
        xi_mean_along_m=float(xi_mean_along_m), xi_mean_cross_m=float(xi_mean_cross_m),
        xi_std_along_m=float(xi_std_along_m), xi_std_cross_m=float(xi_std_cross_m),
        xi_geo_bound_extra_m=float(xi_geo_bound_extra_m),
        eps_time_total=float(p.eps_T), eps_time_xi=float(eps_time_xi),
        eps_time_weather=float(eps_time_weather), eps_time_along=float(eps_time_along),
        eps_time_cross=float(eps_time_cross),
        total_tightening_s=float(time_drcc_tightening_s),
        final_margin_s=float(time_contract["time_drcc_margin_s"]),
        time_core_nom_s=float(time_core_nom_s),
        time_wait_nom_s=float(time_contract["time_wait_nom_s"]),
        time_safe_core_s=float(time_contract["time_safe_core_s"]),
        time_wait_safe_s=float(time_contract["time_wait_safe_s"]),
        xi_gradient_s_per_m=np.asarray(a_T, float).tolist(),
        weather_gradient_s_per_ms=np.asarray(aT_w_diag, float).tolist(),
        xi_gradient_unit="s/m", weather_gradient_unit="s/(m/s)",
        soc_correction=str(getattr(p, "soc_correction", "none")),
        chance_mode=str(chance_mode), time_contract=TIME_CONTRACT,
        wait_is_recourse=True, dock_risk_contract=DOCK_RISK_CONTRACT)

    # 着舰门(回收时刻天气; 含浪不确定性收紧的 Hs_eff)。
    # 更新(审计修复#8-着舰门#5): 风侧按规划侧【上侧】收紧 w10_gate = wind10 + r_gate
    # (ε_gate 预算, 门以 10m 风判定不升尺度), 已在上方逐候选场计算。
    # 更新(C-01): 门判据 = 所有候选场全部开门(最坏情况; 见 gate_weather_proof)。
    L = 1 if gate_all_open else 0
    # 可行 = 能量/时间 SOC + 着舰门 + 非负时间裕度 + 【逐航段空速可行】(critique 必改 4)
    spd_ok = bool(nom["speed_feasible"])
    # ---- 空速机会约束：与当前“巡航空速/最大空速可调”控制策略保持同一对象。
    #   对每条航段把风误差投影到顺航向和横航向，分别计算单侧/双侧安全界；在最大空速下
    #   检查横风可控性及最小前进地速。风险预算在航段与投影事件间 Bonferroni 分配。
    #   因而裕度随航段方向、风向和路线顺序变化，不再是 max_required_airspeed 加一个全局
    #   二维范数半径的固定关闭开关。
    air_margin = float(p.v_air_max) - float(nom.get("max_required_airspeed_ms", 0.0))
    air_diag = dict(ok=bool(spd_ok), margin_ms=float(air_margin), worst_leg=None, leg_checks=[])
    if weather_unc is not None and getattr(p, "airspeed_cc", "on") != "off":
        eps_air = float(getattr(p, "eps_air", p.eps_cap))
        air_diag = route_airspeed_projection_check(
            nom, p, weather_unc, eps_air, chance_mode=chance_mode,
            budget_gamma=budget_gamma, saa_n=saa_n, saa_seed=saa_seed,
            kappa_fn=kappa_eval)
        air_margin = float(air_diag["margin_ms"])
        spd_ok = bool(spd_ok and air_diag["ok"])

    # 船尾伴飞所需空速同样受 10m 风误差影响；范数对风矢量是 1-Lipschitz。
    escort_margin = float(p.v_air_max) - float(nom.get("escort_required_airspeed_ms", 0.0))
    escort_ok = bool(nom.get("escort_speed_feasible", True))
    if weather_unc is not None:
        eps_escort = float(getattr(p, "eps_escort", p.eps_cap))
        r_escort = wind_delta_radius(chance_mode, weather_unc, eps_escort,
                                     budget_gamma=budget_gamma, saa_n=saa_n, saa_seed=saa_seed + 37)
        escort_margin -= r_escort
        escort_ok = escort_ok and (escort_margin >= 0.0)

    landing_wave_ok = bool(all(c.get("wave_ok", False) for c in _checked))
    landing_wind_ok = bool(all(c.get("wind_ok", False) for c in _checked))
    nominal_speed_ok = bool(nom.get("speed_feasible", False))
    failure_flags = {
        "missing_recovery_state": False, "forbidden_recovery_state": False,
        "missing_recovery_prediction": False, "missing_recovery_weather": False,
        "missing_xi_cell": False, "risk_budget_invalid": not risk_budget_ok,
        "escort_airspeed_failed": not escort_ok,
        "landing_gate_failed": L != 1,
        "landing_wind_failed": not landing_wind_ok,
        "landing_wave_failed": not landing_wave_ok,
        "nominal_airspeed_failed": not nominal_speed_ok,
        "route_airspeed_failed": not spd_ok,
        "energy_drcc_failed": mE < 0,
        "time_drcc_failed": bool(time_contract["time_drcc_failed"]),
        "nominal_time_failed": bool(time_contract["nominal_time_failed"]),
    }
    feasible = bool(mE >= 0 and not time_contract["time_drcc_failed"]
                    and not time_contract["nominal_time_failed"] and L == 1 and spd_ok
                    and escort_ok and risk_budget_ok)
    reason = None
    if not feasible:
        if not risk_budget_ok:
            reason = "mission_risk_budget_invalid"
        elif not escort_ok:
            reason = "escort_speed_infeasible"
        elif L != 1:
            reason = "landing_gate_closed"
        elif not spd_ok:
            reason = "route_airspeed_infeasible"
        elif mE < 0:
            reason = "energy_margin_negative"
        elif time_contract["time_drcc_failed"] or time_contract["nominal_time_failed"]:
            reason = "time_margin_negative"
        else:
            reason = "physical_constraint_failed"
    # 多物理量裕度仅作排序破同；所有硬约束仍逐项判断。
    M_omega = float(min(mE, float(time_contract["time_drcc_margin_s"]), escort_margin))
    landing_wind_margin = float(p.w_land_max) - float(w10_gate)
    return dict(feasible=feasible, reason=reason, primary_reason=reason,
                failure_flags=failure_flags,
                margins=dict(energy_Wh=float(mE), time_s=float(time_contract["time_drcc_margin_s"]),
                             route_airspeed_ms=(float(air_margin) if math.isfinite(air_margin)
                                                else float(p.v_air_max) - float(nom.get("max_required_airspeed_ms", 0.0))),
                             landing_wind_ms=landing_wind_margin,
                             escort_airspeed_ms=float(escort_margin)),
                weather_launch_time=str(wx_launch.get("time", "unknown")),
                weather_recovery_time=str(wx_recovery.get("time", "unknown")),
                h=h, h_s=float(h_s), c=route.ship.c_state, n=cell.n,
                margin_E=mE, margin_T=float(time_contract["time_drcc_margin_s"]),
                M_omega=M_omega, slack_T=float(time_contract["nominal_time_margin_s"]),
                gate=L, r_cap_eff=r_eff,
                recovery_state=str(recovery_state), recovery_state_source=str(recovery_state_source),
                recovery_state_ok=bool(recovery_state_ok),
                recovery_target_model=str(getattr(p, "recovery_target_model", "discrete_horizon_ship_prediction")),
                terminal_sensor_error_mode=str(getattr(p, "terminal_sensor_error_mode", "out_of_scope")),
                E0=nom["E0"], E_flight_Wh=float(nom.get("E_flight", nom["E0"])),
                E_escort_Wh=float(nom.get("E_escort", 0.0)),
                E_plan_Wh=float(nom["E0"] + E_dock),
                # : 电池 SOC 调度使用“DRCC 所需能量”而非名义计划能量。
                # mE = B_use - E_plan - uncertainty_buffer，因此 B_use-mE
                # 正好是该架次在当前风险口径下需从实体电池组预留的能量。
                E_soc_required_Wh=float(max(nom["E0"] + E_dock, p.B_use - mE)),
                E_uncertainty_buffer_Wh=float(max((p.B_use - mE) - (nom["E0"] + E_dock), 0.0)),
                energy_branch_eps=float(eps_E_branch),
                energy_branch_margins={k: float(v) for k, v in branch_margins.items()},
                T0=nom["T0"], d_ret0=nom["d_ret0"],
                E_wait=nom.get("E_wait", 0.0), T_flight=nom.get("T_flight", nom["T0"]),
                T_escort_s=float(nom.get("T_escort", 0.0)),
                escort_required_airspeed_ms=float(nom.get("escort_required_airspeed_ms", 0.0)),
                escort_speed_feasible=bool(escort_ok),
                escort_margin_ms=float(escort_margin),
                E_dock_Wh=float(E_dock), t_dock_s=float(t_dock),   # 更新: 动态对接储备(该回收时刻)
                t_dock_wait_s=float(t_dock_wait),                  # 更新 C-01: 等待窗侧(候选最小)
                gate_weather_proof=gate_weather_proof,             # 更新 C-01: 运行时切换安全证据
                landing_wind_upper_shift_ms=float(r_gate),
                dock_wind_lower_shift_ms=float(r_dock_low),
                wind_speed_bias_ms=float(getattr(weather_unc, "wind_speed_bias", 0.0)) if weather_unc is not None else 0.0,
                wind_speed_std_ms=float(getattr(weather_unc, "wind_speed_std", 0.0)) if weather_unc is not None else 0.0,
                route_airspeed_diagnostics=air_diag,
                time_contract=TIME_CONTRACT, time_contract_id=TIME_CONTRACT,
                wait_is_recourse=WAIT_IS_RECOURSE, dock_risk_contract=DOCK_RISK_CONTRACT,
                time_flight_nom_s=float(time_flight_nom_s),
                time_inspection_s=float(time_inspection_s),
                time_dock_nom_s=float(time_dock_nom_s),
                time_core_nom_s=float(time_core_nom_s),
                time_wait_nom_s=float(time_contract["time_wait_nom_s"]),
                time_xi_mean_shift_s=float(xi_mean),
                time_xi_std_term_s=float(xi_std),
                time_weather_mean_shift_s=float(weather_mean),
                time_weather_std_term_s=float(weather_std),
                time_geometry_remainder_s=float(geometry_remainder),
                time_geometry_correction_s=float(geometry_remainder),
                time_xi_geo_total_s=float(xi_geo_total),
                time_weather_total_s=float(weather_mean + weather_std),
                xi_mean_along_m=float(xi_mean_along_m),
                xi_mean_cross_m=float(xi_mean_cross_m),
                xi_std_along_m=float(xi_std_along_m),
                xi_std_cross_m=float(xi_std_cross_m),
                xi_geo_bound_extra_m=float(xi_geo_bound_extra_m),
                xi_mu_e_m=float(cell.mu[0]), xi_mu_n_m=float(cell.mu[1]),
                xi_sigma_ee_m2=float(cell.Sigma[0, 0]),
                xi_sigma_en_m2=float(cell.Sigma[0, 1]),
                xi_sigma_nn_m2=float(cell.Sigma[1, 1]),
                xi_launch_to_recovery_state_change_rate=float(getattr(cell, "state_change_rate", float("nan"))),
                xi_actual_recovery_state_mode=str(getattr(cell, "actual_recovery_state_mode", "unknown")),
                xi_launch_speed_p50_ms=float(getattr(cell, "launch_speed_p50_ms", float("nan"))),
                xi_launch_speed_p95_ms=float(getattr(cell, "launch_speed_p95_ms", float("nan"))),
                eps_time_total=float(p.eps_T), eps_time_xi=float(eps_time_xi),
                eps_time_weather=float(eps_time_weather),
                eps_time_along=float(eps_time_along), eps_time_cross=float(eps_time_cross),
                geo_risk_allocation_mode=geo_risk_allocation_mode,
                geo_risk_allocation_contract=geo_risk_allocation_contract,
                time_drcc_tightening_s=float(time_drcc_tightening_s),
                time_safe_core_s=float(time_contract["time_safe_core_s"]),
                time_wait_safe_s=float(time_contract["time_wait_safe_s"]),
                nominal_time_margin_s=float(time_contract["nominal_time_margin_s"]),
                time_drcc_margin_s=float(time_contract["time_drcc_margin_s"]),
                time_decomposition=time_decomposition,
                time_nominal_margin_s=float(time_contract["nominal_time_margin_s"]),
                time_total_tightening_s=float(time_drcc_tightening_s),
                speed_feasible=spd_ok,
                airspeed_margin_ms=(round(float(air_margin), 2) if math.isfinite(air_margin) else None),
                max_required_airspeed_ms=nom.get("max_required_airspeed_ms", 0.0),
                speed_fail_leg=nom.get("speed_fail_leg", -1),
                sigma_trace=float(np.trace(cell.Sigma)), Hs_eff=float(Hs_eff),
                w10_gate_eff=float(w10_gate), eps_gate_used=float(eps_gate),
                chance_mode=chance_mode,
                mission_failure_budget=float(getattr(p, "mission_failure_budget", 0.05)),
                mission_eps_budget=float(risk_budget),
                mission_budget_compliant=bool(risk_budget_ok),
                risk_allocation=mission_risk_allocation(p, weather_on),
                risk_budget_unallocated=float(p.mission_failure_budget) - float(risk_budget),
                soc_correction=getattr(p, "soc_correction", "none"),
                soc_risk_allocation=getattr(p, "soc_risk_allocation", "fixed"))   # 更新: 口径入诊断


def route_drcc_feasible(route: Route, p: M.Params, wx: dict,
                        xi_amb: "M.XiAmbiguity",
                        objective: str = "min_h",
                        weather_unc: "WeatherUncertainty" = None,
                        h_grid=None,
                        chance_mode: str = "drcc",
                        budget_gamma: float = BUDGET_GAMMA_DEFAULT,
                        formal: bool = False) -> dict:
    """对一条路由【优化回收时长 h】(决策依赖): 只在 xi 统计支持区间内的决策网格
    (见 ``decision_horizons_of``)逐 h 检查，选满足 DRCC、目标获取、船尾伴飞和着舰门且符合
    objective 的 h。
    这就是决策依赖——逐 h 权衡"时间裕度↑ vs 不确定性↑"; 细网格让权衡更精细。

    objective:
      'min_h'       最早可行回收(省占位时间, 母船早归队)
      'max_margin'  能量/时间余量之和最大
      'max_robust'  鲁棒安全裕度 M_ω 最大（诊断/消融用；不属于正式两层目标）
    weather_unc(可选): 见 route_feasible_at_h, 把风/浪不确定性纳入 DRCC。
    h_grid(可选): 覆盖默认决策网格(如做 h 网格消融)。
    chance_mode(可选): 'drcc'(本文)/'saa'/'budget' baseline 对照(见 route_feasible_at_h)。
    返回最优 h 的诊断; feasible=False 表示【任何 h 都不可行】。
    """
    cand_h = decision_horizons_of(xi_amb, h_grid)
    _kw = dict(weather_unc=weather_unc, chance_mode=chance_mode, budget_gamma=budget_gamma, formal=formal)
    feas = []
    for h in cand_h:
        d = route_feasible_at_h(route, h, p, wx, xi_amb, **_kw)
        if d["feasible"]:
            feas.append(d)
    if not feas:
        # 仍返回"最不坏"的 h 供诊断(slack 与不确定性的折中点)
        diags = [route_feasible_at_h(route, h, p, wx, xi_amb, **_kw) for h in cand_h]
        best = max(diags, key=lambda d: min(d["margin_E"], d["margin_T"]))
        best = dict(best); best["feasible"] = False
        return best
    if objective == "max_margin":
        return max(feas, key=lambda d: d["margin_E"] + d["margin_T"])
    if objective == "max_robust":
        return max(feas, key=lambda d: d["M_omega"])
    return min(feas, key=lambda d: d["h"])       # min_h 默认


def route_cost(route: Route, h: int, p: M.Params, wx: dict, kind: str = "count",
               xi_amb: "M.XiAmbiguity" = None, weather_unc: "WeatherUncertainty" = None) -> float:
    """路由成本: 'count'=1(最小化架次数) | 'energy'=名义能耗 Wh | 'time'=名义时长 s
    | 'neg_robust'= −M_ω(最大化鲁棒裕度 → 取负作最小化; 需 xi_amb)。"""
    if kind == "count":
        return 1.0
    if kind == "neg_robust":
        if xi_amb is None:
            return 0.0
        d = route_feasible_at_h(route, int(h), p, wx, xi_amb, weather_unc=weather_unc)
        return -float(d["M_omega"])
    if kind == "energy":
        if xi_amb is not None:
            d = route_feasible_at_h(route, int(h), p, wx, xi_amb,
                                    weather_unc=weather_unc)
            if not d.get("feasible", False):
                return float("inf")
            return float(d["E_plan_Wh"])
        # 无 ξ 对象时仍按正式第二层口径加入完整 dock 储备。
        cand = gate_weather_candidates(route, wx)
        checked = []
        for cw in cand:
            motion = M.deck_motion(float(cw["Hs"]), cw.get("Tp", 2.1),
                                   cw.get("wave_dir", 200.0) - cw.get("ship_heading", 0.0), p)
            td, ed = M.dock_reserve(p, motion, float(cw.get("wind10", 0.0)))
            checked.append((float(td), float(ed)))
        t_wait = min(x[0] for x in checked) if checked else 0.0
        e_dock = max(x[1] for x in checked) if checked else 0.0
        nom = route_nominal_ET(route, h, p, wx, t_dock_s=t_wait)
        return float(nom["E0"] + e_dock)
    nom = route_nominal_ET(route, h, p, wx)
    return nom["T0"]


# =============================================================================
# 5. 自检(占位船航迹 + 真实风机 + 占位多 h 模糊集)
# =============================================================================
def _demo_xi(horizons, states):
    """占位模糊集: Σ 随 h 线性增长、直航≈4.3×动力定位(贴近真实数据特征),
    用于让决策依赖的 h 权衡【真的发生】(短 h 不确定性小但 slack 小; 长 h 反之)。"""
    cells = {}
    state_scale = {"直航": 1.0, "转弯": 1.2, "低速": 0.5, "动力定位": 0.23}
    for h in horizons:
        for c in states:
            sd = 120.0 * (h / 5.0) * state_scale.get(c, 1.0)   # m, 标准差随 h 增
            Sigma = np.array([[sd**2, 0.0], [0.0, (0.8 * sd)**2]])
            cells[(h, c)] = M.XiCell(h_min=h, c_state=c, n=50000,
                                     mu=np.array([0.0, 0.0]), Sigma=Sigma,
                                     support_radius=4.0 * sd,
                                     p95_norm=2.0 * sd, rms_norm=1.3 * sd)
    return M.XiAmbiguity(cells, list(horizons))


def _demo_xi_realistic(horizons, states):
    """★测试夹具: 用 Case B 真实 √λmax 量级(从 step11 的 moment_radius/κ 反推, 2026-06-27)。
    h=5 的 √λmax(m): 低速 405 / 动力定位 186 / 直航 727 / 转弯 989; 按 (h/5)^1.38 幂律外推。
    用途: 让 step16/step17 自检在【真实量级】下跑(直航不可回收、动力定位可行), 验证自洽性与算法。
    注意: 这是合成测试矩, 非真实数据; 正式数值仍须作者本机用 xi_moments_caseB.csv 跑。"""
    base5 = {"低速": 405.0, "动力定位": 186.0, "直航": 727.0, "转弯": 989.0}
    cells = {}
    for h in horizons:
        for c in states:
            sl = base5.get(c, 400.0) * (h / 5.0) ** 1.38      # √λmax(h)
            Sigma = np.array([[sl**2, 0.18 * sl * sl], [0.18 * sl * sl, (0.6 * sl)**2]])
            cells[(h, c)] = M.XiCell(h_min=h, c_state=c, n=50000,
                                     mu=np.array([20.0, -15.0]),   # 小偏置(重尾→μ 小)
                                     Sigma=Sigma, support_radius=3.5 * sl,
                                     p95_norm=1.2 * sl, rms_norm=1.5 * sl)
    amb = M.XiAmbiguity(cells, list(horizons))
    amb.predictor = "cv_noleak"
    amb.predictor_contract = M.XI_PREDICTOR_CONTRACTS["cv_noleak"]
    amb.timestamp_epoch_contract = M.XI_TIMESTAMP_EPOCH_CONTRACT
    amb.valid_for_formal = False
    amb.formal_validated = False
    amb.moments_source = "synthetic_fixture"
    amb.sample_overlap_policy = "synthetic_fixture"
    return amb


def _selftest():
    here = Path(__file__).resolve().parent
    turb_csv = M._first_existing([here / "data" / "turbines_Rodsand_II_clean.csv"])
    p = M.Params()
    horizons = [5, 10, 15, 20, 30]
    states = ["直航", "转弯", "低速", "动力定位"]

    # 风机(取前 6 台做演示路由)
    if turb_csv:
        turbines = M.load_turbines(turb_csv, farm="Rodsand_II")[:6]
    else:
        turbines = [M.Turbine(f"DEMO_{i}", np.array([11.55 + 0.008 * i, 54.55]), 68.5, 115.0)
                    for i in range(6)]
    lat0, lon0 = turbines[0].lonlat[1], turbines[0].lonlat[0]
    for t in turbines:
        t.local = M.latlon_to_local_m(t.lonlat[1], t.lonlat[0], lat0, lon0)

    # 占位天气(平静) + 占位多 h 模糊集
    wx = dict(wind10=6.0, wind_dir_from=230.0, Hs=0.5, Tp=2.1, wave_dir=200.0, ship_heading=90.0)
    xi_amb = _demo_xi(horizons, states)

    # 船位预测: 起飞点在风机群西南 ~800m, 船以 3 m/s 向东北航行(CV 合成各 h 预测点)
    P_launch = turbines[0].local + np.array([-800.0, -600.0])
    ship = ShipPrediction.from_cv(P_launch, v_ship=np.array([2.5, 1.5]),
                                  horizons=horizons, c_state="直航")

    print("\n================ step10_model_routing.py 自检 ================")
    print(f"风机 {len(turbines)} 台 | 船状态 c={ship.c_state} | h 候选 {horizons} min")

    # 单台路由 vs 三台路由, 看航路化效果
    r1 = Route(rid=0, turbines=[turbines[0]], ship=ship)
    r3 = Route(rid=1, turbines=turbines[:3], ship=ship)

    for name, r in [("单台路由 [t0]", r1), ("三台路由 [t0,t1,t2]", r3)]:
        d = route_drcc_feasible(r, p, wx, xi_amb, objective="min_h")
        print(f"\n--- {name} ---")
        print(f"  访问 {r.turbine_ids()} | n_stops={r.n_stops()}")
        print(f"  DRCC 最优回收时长 h*={d['h']} min(决策依赖选出) → "
              f"{'可行' if d['feasible'] else '不可行'}")
        print(f"  名义: E0={d['E0']:.1f}Wh(≤{p.B_use:.0f}) T0={d['T0']:.0f}s | "
              f"时间裕度 slack={d['slack_T']:.0f}s")
        print(f"  能量余量={d['margin_E']:+.1f}Wh 时间余量={d['margin_T']:+.0f}s | "
              f"该 h 的 Σ迹={d['sigma_trace']:.0f}")

    # 决策依赖核心演示: 同一条三台路由, 逐 h 看"裕度 vs 不确定性"权衡(细网格采样)
    print("\n--- 决策依赖: 三台路由在统计支持区间内逐 h 权衡(抽样展示)---")
    print("   h(min)  名义T0(s)  slack=h-T0(s)  Σ迹(m²)  能量余量  时间余量  M_ω(min余量)  可行")
    demo_grid = decision_horizons_of(xi_amb)
    show_h = [h for h in demo_grid if h in (5, 10, 15, 18, 20, 22, 24, 26, 30, 36, 42)]
    for h in show_h:
        d = route_feasible_at_h(r3, h, p, wx, xi_amb)
        print(f"   {h:5d}  {d['T0']:8.0f}  {d['slack_T']:11.0f}  {d['sigma_trace']:8.0f}  "
              f"{d['margin_E']:+8.1f}  {d['margin_T']:+8.0f}  {d['M_omega']:+10.1f}  "
              f"{'✓' if d['feasible'] else '✗'}")
    print("\n  解读: h 太小 → slack 不够(时间余量负); h 太大 → Σ 膨胀(余量被 κ√Σ 吃掉)。")
    print(f"       决策层细网格 {len(demo_grid)} 个候选({min(demo_grid)}..{max(demo_grid)}min), 统计层 ξ 矩仍用粗格 {horizons};")
    print("       存在一个最优 h* 平衡二者 —— 细网格让该最优更精确, 也增大候选列空间(算法对比更有戏)。")

    print("\n自检完成。路由能耗/时间(序列相关)+ 决策依赖 DRCC(逐 h 权衡)链路已跑通。")
    print("下一步: step11_algorithm_route_drcc.py 生成可行任务列，并求解最大覆盖—最小完整计划能耗的机队主问题。")


if __name__ == "__main__":
    _selftest()
