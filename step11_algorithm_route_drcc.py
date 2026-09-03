#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step11_algorithm_route_drcc.py — 给定列池的最终资源主问题。

正式入口 ``solve_resource_master`` 使用两层目标：L1 最大可靠覆盖，L2 最小完整计划能耗。
它显式分配实体 UAV 和电池组，累计 E_soc_required_Wh，允许同 UAV/同电池的剩余
SOC 快速复用，并分别约束起降甲板、着陆清场、快速检查工位和非甲板换电工位。
资源可行性子问题通过精确整数模式排除割反馈给列选择主问题；在精确 MILP 与资源搜索
均闭合时，结果对给定候选列池精确。

本文件下部仍保留 ``solve_route_drcc``、历史集合划分和三层目标等兼容接口，仅供旧
实验与回归测试，不代表当前业务模型。全有限路线空间证书必须使用
``step12.solve_fleet_anytime``。
"""
from __future__ import annotations

import itertools
import logging
import math
import time
from fractions import Fraction
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import step9_model as M
import step10_model_routing as RM

log = logging.getLogger("routealg")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")




def _validate_binary_master_vector(x, objective, cover_matrix, cover_lb, cover_ub,
                                   slot_rows=None, extra_rows=None, tol=1e-7):
    """Validate a complete 0-1 master vector without trusting solver status/metadata.

    Returns ``(validated_x, objective_value, reason)``.  ``validated_x`` is rounded
    to exact 0/1 only after shape, finiteness, bounds, integrality and every master
    row have passed.  Any malformed/unknown result fails closed.
    """
    try:
        c = np.asarray(objective, dtype=float).reshape(-1)
        xx = np.asarray(x, dtype=float)
        if xx.ndim != 1 or xx.shape != c.shape:
            return None, None, "bad_shape"
        if not np.all(np.isfinite(xx)) or not np.all(np.isfinite(c)):
            return None, None, "non_finite"
        if np.any(xx < -tol) or np.any(xx > 1.0 + tol):
            return None, None, "bounds"
        if np.any(np.abs(xx - np.rint(xx)) > tol):
            return None, None, "non_integral"
        zz = np.rint(xx).astype(float)

        A = np.asarray(cover_matrix, dtype=float)
        lb = np.asarray(cover_lb, dtype=float).reshape(-1)
        ub = np.asarray(cover_ub, dtype=float).reshape(-1)
        if A.ndim != 2 or A.shape != (lb.size, zz.size) or ub.shape != lb.shape:
            return None, None, "bad_model_shape"
        lhs = A @ zz
        if np.any(lhs < lb - tol) or np.any(lhs > ub + tol):
            return None, None, "cover_constraint"

        for row in slot_rows or ():
            rr = np.asarray(row, dtype=float).reshape(-1)
            if rr.shape != zz.shape or not np.all(np.isfinite(rr)):
                return None, None, "bad_slot_row"
            if float(rr @ zz) > 1.0 + tol:
                return None, None, "slot_constraint"
        for row, row_lb, row_ub in extra_rows or ():
            rr = np.asarray(row, dtype=float).reshape(-1)
            if rr.shape != zz.shape or not np.all(np.isfinite(rr)):
                return None, None, "bad_extra_row"
            value = float(rr @ zz)
            if value < float(row_lb) - tol or value > float(row_ub) + tol:
                return None, None, "extra_constraint"
        return zz, float(c @ zz), None
    except Exception as exc:
        return None, None, f"validation_exception:{type(exc).__name__}"



def _lagrangian_dual_lower_bound(c, lo, hi, Au, bu, Ae, be, im, em):
    """Conservative row-dual Lagrangian lower bound for a bounded LP."""
    try:
        c = np.asarray(c, float).reshape(-1)
        lo = np.asarray(lo, float).reshape(-1)
        hi = np.asarray(hi, float).reshape(-1)
        Au = np.asarray(Au, float).reshape((-1, c.size))
        bu = np.asarray(bu, float).reshape(-1)
        Ae = np.asarray(Ae, float).reshape((-1, c.size))
        be = np.asarray(be, float).reshape(-1)
        y = np.minimum(np.asarray(im, float).reshape(-1), 0.0)
        v = np.asarray(em, float).reshape(-1)
        if (y.shape != bu.shape or v.shape != be.shape
                or not all(np.all(np.isfinite(a)) for a in (c, lo, hi, Au, bu, Ae, be, y, v))):
            return None
        base = [float(bu[j]) * float(y[j]) for j in range(bu.size)]
        base += [float(be[j]) * float(v[j]) for j in range(be.size)]
        terms = list(base)
        eps = np.finfo(float).eps
        mass = 1.0 + sum(abs(t) for t in base)
        for i in range(c.size):
            products = [float(Au[j, i]) * float(y[j]) for j in range(bu.size)]
            products += [float(Ae[j, i]) * float(v[j]) for j in range(be.size)]
            q = math.fsum([float(c[i])] + [-t for t in products])
            q_err = 16.0 * eps * (1.0 + abs(float(c[i])) + sum(abs(t) for t in products))
            q_lo, q_hi = q - q_err, q + q_err
            l, u = float(lo[i]), float(hi[i])
            if math.isfinite(l) and math.isfinite(u):
                val = min(q_lo * l, q_lo * u, q_hi * l, q_hi * u)
            elif math.isfinite(l):
                if q_lo < 0.0:
                    return None
                val = min(q_lo * l, q_hi * l)
            elif math.isfinite(u):
                if q_hi > 0.0:
                    return None
                val = min(q_lo * u, q_hi * u)
            else:
                if q_lo < 0.0 or q_hi > 0.0:
                    return None
                val = 0.0
            terms.append(float(val)); mass += abs(float(val)) + sum(abs(t) for t in products)
        bound = math.fsum(terms) - 128.0 * eps * mass
        bound = math.nextafter(bound, -math.inf)
        return float(bound) if math.isfinite(bound) else None
    except Exception:
        return None


def _safe_integer_ceiling(lower_bound):
    if lower_bound is None or not math.isfinite(float(lower_bound)):
        return 0
    return int(math.ceil(math.nextafter(float(lower_bound), -math.inf)))


def _validate_lp_kkt(res, c, bounds, A_ub=None, b_ub=None,
                     A_eq=None, b_eq=None, tol=1e-7):
    """Validate a HiGHS LP primal/dual certificate without trusting status alone."""
    try:
        c = np.asarray(c, float).reshape(-1); n = c.size
        if (res is None or getattr(res, "success", False) is not True
                or getattr(res, "status", None) != 0 or not np.all(np.isfinite(c))):
            return None
        x = np.asarray(getattr(res, "x", None), float).reshape(-1)
        if x.shape != (n,) or not np.all(np.isfinite(x)) or len(bounds) != n:
            return None
        lo = np.array([-np.inf if b[0] is None else float(b[0]) for b in bounds])
        hi = np.array([ np.inf if b[1] is None else float(b[1]) for b in bounds])
        if np.any(np.isnan(lo)) or np.any(np.isnan(hi)) or np.any(lo > hi):
            return None
        if np.any(x < lo - tol) or np.any(x > hi + tol):
            return None

        Au = np.zeros((0, n)); bu = np.zeros(0)
        if A_ub is not None:
            Au = np.asarray(A_ub, float); bu = np.asarray(b_ub, float).reshape(-1)
            if Au.shape != (bu.size, n) or not np.all(np.isfinite(Au)) or not np.all(np.isfinite(bu)):
                return None
            if np.any(Au @ x > bu + tol * np.maximum(1.0, np.abs(bu))):
                return None
        Ae = np.zeros((0, n)); be = np.zeros(0)
        if A_eq is not None:
            Ae = np.asarray(A_eq, float); be = np.asarray(b_eq, float).reshape(-1)
            if Ae.shape != (be.size, n) or not np.all(np.isfinite(Ae)) or not np.all(np.isfinite(be)):
                return None
            if np.any(np.abs(Ae @ x - be) > tol * np.maximum(1.0, np.abs(be))):
                return None

        fun = float(getattr(res, "fun", np.nan)); calc = float(c @ x)
        if not math.isfinite(fun) or abs(fun - calc) > tol * max(1.0, abs(fun), abs(calc)):
            return None
        im = np.asarray(res.ineqlin.marginals, float).reshape(-1) if bu.size else np.zeros(0)
        em = np.asarray(res.eqlin.marginals, float).reshape(-1) if be.size else np.zeros(0)
        lm = np.asarray(res.lower.marginals, float).reshape(-1)
        um = np.asarray(res.upper.marginals, float).reshape(-1)
        if (im.shape != (bu.size,) or em.shape != (be.size,) or lm.shape != (n,) or um.shape != (n,)
                or not all(np.all(np.isfinite(v)) for v in (im, em, lm, um))):
            return None
        finite_lo = np.isfinite(lo); finite_hi = np.isfinite(hi)
        if (np.any(im > tol) or np.any(lm[finite_lo] < -tol) or np.any(um[finite_hi] > tol)
                or np.any(np.abs(lm[~finite_lo]) > tol) or np.any(np.abs(um[~finite_hi]) > tol)):
            return None
        station = Au.T @ im + Ae.T @ em + lm + um
        if np.any(np.abs(c - station) > tol * max(1.0, np.max(np.abs(c)), np.max(np.abs(station)))):
            return None
        comp_tol = 10.0 * tol * max(1.0, abs(fun))
        if bu.size and np.any(np.abs(im * (bu - Au @ x)) > comp_tol):
            return None
        if np.any(np.abs(lm[finite_lo] * (x[finite_lo] - lo[finite_lo])) > comp_tol):
            return None
        if np.any(np.abs(um[finite_hi] * (hi[finite_hi] - x[finite_hi])) > comp_tol):
            return None
        dual = float(bu @ im) if bu.size else 0.0
        dual += float(be @ em) if be.size else 0.0
        dual += float(lo[finite_lo] @ lm[finite_lo]) if np.any(finite_lo) else 0.0
        dual += float(hi[finite_hi] @ um[finite_hi]) if np.any(finite_hi) else 0.0
        if not math.isfinite(dual) or abs(fun - dual) > 10.0 * tol * max(1.0, abs(fun), abs(dual)):
            return None
        dual_lb = _lagrangian_dual_lower_bound(c, lo, hi, Au, bu, Ae, be, im, em)
        if dual_lb is None or dual_lb > calc + tol * max(1.0, abs(calc)):
            return None
        return x, fun, dual_lb
    except Exception:
        return None


def _lp_relaxation_certifies(c, candidate_value, cover_matrix, cover_lb, cover_ub,
                             slot_rows=(), extra_rows=(), tol=1e-7):
    """Sufficient independent optimality check for a binary restricted master.

    The candidate is certified only when a separately KKT-validated LP relaxation
    supplies a matching lower bound (or, for an integer-valued objective, its safe
    ceiling matches the candidate).  Otherwise the candidate remains a feasible
    incumbent but must not be labelled ``proven_optimal``.
    """
    try:
        from scipy.optimize import linprog
        c = np.asarray(c, float).reshape(-1); n = c.size
        Aub, bub, Aeq, beq = [], [], [], []

        def add_row(row, lb, ub):
            rr = np.asarray(row, float).reshape(-1)
            if rr.shape != (n,) or not np.all(np.isfinite(rr)):
                raise ValueError("bad row")
            lb = float(lb); ub = float(ub)
            if math.isfinite(lb) and math.isfinite(ub) and abs(lb - ub) <= tol:
                Aeq.append(rr); beq.append(0.5 * (lb + ub)); return
            if math.isfinite(ub):
                Aub.append(rr); bub.append(ub)
            if math.isfinite(lb):
                Aub.append(-rr); bub.append(-lb)

        A = np.asarray(cover_matrix, float)
        lb = np.asarray(cover_lb, float).reshape(-1)
        ub = np.asarray(cover_ub, float).reshape(-1)
        if A.shape != (lb.size, n) or ub.shape != lb.shape:
            return False, None, False
        for i in range(lb.size):
            add_row(A[i], lb[i], ub[i])
        for row in slot_rows or ():
            add_row(row, -np.inf, 1.0)
        for row, row_lb, row_ub in extra_rows or ():
            add_row(row, row_lb, row_ub)

        A_ub = np.asarray(Aub, float).reshape((-1, n)) if Aub else None
        b_ub = np.asarray(bub, float) if Aub else None
        A_eq = np.asarray(Aeq, float).reshape((-1, n)) if Aeq else None
        b_eq = np.asarray(beq, float) if Aeq else None
        bounds = [(0.0, 1.0)] * n
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
        checked = _validate_lp_kkt(res, c, bounds, A_ub=A_ub, b_ub=b_ub,
                                   A_eq=A_eq, b_eq=b_eq, tol=tol)
        if checked is None:
            return False, None, False
        x_lp, _primal_value, lb_value = checked
        value = float(candidate_value)
        scale = max(1.0, abs(value), abs(lb_value))
        cert_tol = tol * scale
        if value < lb_value - cert_tol:
            return False, lb_value, False
        integral_objective = bool(np.all(np.abs(c - np.rint(c)) <= tol)
                                  and abs(value - round(value)) <= tol)
        if integral_objective:
            ok = _safe_integer_ceiling(lb_value) >= round(value)
        else:
            ok = (-cert_tol <= value - lb_value <= cert_tol)
        # A near-integral LP vector is not by itself a valid counterexample: after
        # rounding it may violate a tight lexicographic fixing row (notably the
        # L2 energy band).  Refute the solver's optimality claim only with a
        # separately revalidated binary vector whose recomputed objective is
        # strictly better.
        better_integer = False
        if np.all(np.abs(x_lp - np.rint(x_lp)) <= 10.0 * tol):
            z_better, z_value, _ = _validate_binary_master_vector(
                x_lp, c, A, lb, ub, slot_rows=slot_rows,
                extra_rows=extra_rows, tol=tol)
            better_integer = bool(
                z_better is not None
                and z_value < value - 10.0 * tol * scale)
        return bool(ok), float(lb_value), better_integer
    except Exception:
        return False, None, False

def _master_result(chosen, R, uncovered, solver, objective, *,
                   proven_optimal, solution_validated, validation_reason=None,
                   lex_sorties=None, lex_energy=None, lex_robust=None,
                   l1_proven_optimal=None, l2_proven_optimal=None,
                   l3_proven_optimal=None):
    """Create a uniform master result whose proof flags are fail-closed."""
    out = dict(chosen=list(chosen), n_sorties=len(chosen), uncovered=list(uncovered),
               solver=solver, objective=objective,
               routes=[R[k] for k in chosen],
               proven_optimal=bool(proven_optimal),
               solution_validated=bool(solution_validated),
               validation_reason=validation_reason)
    if lex_sorties is not None:
        out.update(lex_sorties=lex_sorties, lex_energy=lex_energy, lex_robust=lex_robust)
    if l1_proven_optimal is not None:
        out["l1_proven_optimal"] = bool(l1_proven_optimal)
    if l2_proven_optimal is not None:
        out["l2_proven_optimal"] = bool(l2_proven_optimal)
    if l3_proven_optimal is not None:
        out["l3_proven_optimal"] = bool(l3_proven_optimal)
    return out


def _wx_of_ship(sp, wx_default: dict) -> dict:
    """列所属起飞时刻 τ 的天气(sp.wx_tau); 缺失退回全局 wx(任务 #7, 与 step12 同口径)。"""
    w = getattr(sp, "wx_tau", None)
    return w if isinstance(w, dict) else wx_default


def _wx_of_route(r, wx_default: dict) -> dict:
    """该列(其 ship 携带 τ)的天气窗; 缺失退回全局 wx(任务 #7)。"""
    sp = getattr(r, "ship", None)
    w = getattr(sp, "wx_tau", None) if sp is not None else None
    return w if isinstance(w, dict) else wx_default


# =============================================================================
# 1. 列生成 / 定价(启发式): 生成 DR 可行候选路由池
# =============================================================================
def _route_ok(turbine_seq, ship, p, wx, xi_amb, objective, weather_unc=None,
              chance_mode="drcc", budget_gamma=RM.BUDGET_GAMMA_DEFAULT):
    """构造一条路由并做决策依赖 DRCC 评估。返回 (Route, 诊断) 或 (None, None)。
    chance_mode: 'drcc'(本文)/'saa'/'budget' baseline 对照(见 step10.route_feasible_at_h)。"""
    r = RM.Route(rid=-1, turbines=list(turbine_seq), ship=ship)
    d = RM.route_drcc_feasible(r, p, wx, xi_amb, objective=objective, weather_unc=weather_unc,
                               chance_mode=chance_mode, budget_gamma=budget_gamma)
    if d["feasible"]:
        r.fixed_h = d["h"]
        return r, d
    return None, None


def gen_singletons(turbines, ship, p, wx, xi_amb, objective, weather_unc=None,
                   chance_mode="drcc", budget_gamma=RM.BUDGET_GAMMA_DEFAULT):
    """每台风机单独成路由(保证可覆盖性; 即单台架次)。"""
    routes = []
    for t in turbines:
        r, d = _route_ok([t], ship, p, wx, xi_amb, objective, weather_unc, chance_mode, budget_gamma)
        if r is not None:
            routes.append((r, d))
    return routes


def gen_nearest_neighbor(turbines, ship, p, wx, xi_amb, objective, max_stops, weather_unc=None,
                         chance_mode="drcc", budget_gamma=RM.BUDGET_GAMMA_DEFAULT):
    """最近邻构造: 从起飞点出发, 反复加入最近的未访问风机; 若加入后任何 h 都不可行,
    则收尾当前路由、从剩余风机里另起一条。产出一组覆盖(尽量)全部风机的 DR 可行路由。"""
    remaining = list(turbines)
    routes = []
    while remaining:
        # 从离起飞点最近的风机起头
        start = min(remaining, key=lambda t: np.linalg.norm(t.local - ship.P_launch))
        seq = [start]
        remaining.remove(start)
        r, d = _route_ok(seq, ship, p, wx, xi_amb, objective, weather_unc, chance_mode, budget_gamma)
        if r is None:                       # 连单台都不可行 → 跳过(该风机此出动覆盖不了)
            continue
        while remaining and len(seq) < max_stops:
            last = seq[-1].local
            nxt = min(remaining, key=lambda t: np.linalg.norm(t.local - last))
            trial = seq + [nxt]
            r2, d2 = _route_ok(trial, ship, p, wx, xi_amb, objective, weather_unc, chance_mode, budget_gamma)
            if r2 is None:
                break                        # 加入后不可行 → 收尾
            seq = trial; r, d = r2, d2
            remaining.remove(nxt)
        routes.append((r, d))
    return routes


def gen_savings_merge(routes, ship, p, wx, xi_amb, objective, max_stops, weather_unc=None,
                      chance_mode="drcc", budget_gamma=RM.BUDGET_GAMMA_DEFAULT):
    """Clarke-Wright 风格节约合并: 尝试把两条路由首尾拼接, 若合并路由 DR 可行且
    架次数更少(2→1), 则合并。重复直到无可合并。每次合并都重评决策依赖 DRCC。"""
    pool = [r for r, _ in routes]
    improved = True
    while improved:
        improved = False
        best = None
        for i, j in itertools.combinations(range(len(pool)), 2):
            ri, rj = pool[i], pool[j]
            if ri.n_stops() + rj.n_stops() > max_stops:
                continue
            for seq in (ri.turbines + rj.turbines, ri.turbines + list(reversed(rj.turbines))):
                r, d = _route_ok(seq, ship, p, wx, xi_amb, objective, weather_unc, chance_mode, budget_gamma)
                if r is not None:
                    # 节约 = 合并前架次成本(2) − 合并后(1) = 1; 取首个可行合并即可
                    best = (i, j, r, d)
                    break
            if best:
                break
        if best:
            i, j, r, d = best
            pool = [pool[k] for k in range(len(pool)) if k not in (i, j)] + [r]
            improved = True
    # 重新附诊断
    out = []
    for r in pool:
        d = RM.route_drcc_feasible(r, p, wx, xi_amb, objective=objective, weather_unc=weather_unc,
                                   chance_mode=chance_mode, budget_gamma=budget_gamma)
        out.append((r, d))
    return out


def generate_routes(turbines, ship, p, wx, xi_amb, objective="min_h",
                    max_stops=8, strategy="full", weather_unc=None,
                    launch_ships=None, chance_mode="drcc",
                    budget_gamma=RM.BUDGET_GAMMA_DEFAULT) -> list:
    """生成候选路由池(列)。strategy:
       'singleton' 仅单台(单台退化对照);
       'nn'        最近邻多台;
       'full'      单台 + 最近邻 + 节约合并(给主问题更丰富的列, 体现 航路化价值)。
    launch_ships(可选, 起飞—回收协同定时 / 任务2): list[ShipPrediction], 每个对应一个候选起飞时刻 τ
       (其 P_launch=P̂_v(τ)、c_state=c(τ))。给定时【逐 τ 生成列并并池】, 每条列携带其 τ 的 ship,
       主问题在 (τ,ω,h) 列上选; 列结构升级为 r=(τ,ω,h)。为 None 时退回单一 ship(向后兼容)。
       注: 短起飞窗内天气近似不变, 故各 τ 共用代表性 wx(时变天气=直接扩展, 见 model.md §6.2)。
    chance_mode(可选, baseline 对照): 'drcc'(本文)/'saa'/'budget'(见 step10.route_feasible_at_h)。
    返回 [(Route, 诊断), ...]; 每条都已通过决策依赖 DRCC(带最优 h)。
    """
    ships = launch_ships if launch_ships is not None else [ship]
    all_pool, seen = [], set()
    for sp in ships:
        wx_sp = _wx_of_ship(sp, wx)   # 任务 #7: 该起飞时刻 τ 的天气窗(无 wx_tau 退回全局)
        pool_s = _generate_routes_one_ship(turbines, sp, p, wx_sp, xi_amb, objective,
                                           max_stops, strategy, weather_unc, chance_mode, budget_gamma)
        for r, d in pool_s:
            # 跨 τ 去重键 = (起飞 τ 标识, 访问序列); 不同 τ 的同序列是不同列
            tau_key = getattr(sp, "tau_min", id(sp))
            key = (tau_key, tuple(r.turbine_ids()))
            if key in seen:
                continue
            seen.add(key); all_pool.append((r, d))
    for i, (r, _) in enumerate(all_pool):
        r.rid = i
    return all_pool


def _generate_routes_one_ship(turbines, ship, p, wx, xi_amb, objective,
                              max_stops, strategy, weather_unc,
                              chance_mode="drcc", budget_gamma=RM.BUDGET_GAMMA_DEFAULT) -> list:
    """单一起飞时刻(单 ship)下生成候选路由池(原 generate_routes 逻辑)。"""
    cm = dict(chance_mode=chance_mode, budget_gamma=budget_gamma)
    if strategy == "singleton":
        return gen_singletons(turbines, ship, p, wx, xi_amb, objective, weather_unc, **cm)
    nn = gen_nearest_neighbor(turbines, ship, p, wx, xi_amb, objective, max_stops, weather_unc, **cm)
    if strategy == "nn":
        return nn
    merged = gen_savings_merge(nn, ship, p, wx, xi_amb, objective, max_stops, weather_unc, **cm)
    # 池 = 节约合并后的多台路由 + 单台路由(保证每台可被覆盖)
    singles = gen_singletons(turbines, ship, p, wx, xi_amb, objective, weather_unc, **cm)
    # 去重(按访问集合 + 顺序)
    seen, pool = set(), []
    for r, d in merged + singles:
        key = tuple(r.turbine_ids())
        if key in seen:
            continue
        seen.add(key); pool.append((r, d))
    for i, (r, _) in enumerate(pool):      # 重新编号, 便于输出
        r.rid = i
    return pool


# =============================================================================
# 2. 主问题: 集合覆盖/划分 MILP, 词典序三层目标(Gurobi 惰性, 贪心回退)
#    第一层 min Σ z_ω (架次数) → 第二层 min Σ E_ω z_ω (总能耗)
#    → 第三层 max Σ M_ω z_ω (鲁棒安全裕度, 取 −Σ M_ω 作最小化)
#    词典序实现: 逐层求解, 把上层最优值固定为约束, 再优化下层(lexicographic)。
# =============================================================================
def solve_master(routes, all_turbine_ids, costs=None, partition=False,
                 lex=False, energy=None, robust=None) -> dict:
    """Select a validated restricted-master integer solution.

    Exact solver outputs are never trusted solely because a backend reports
    success.  SciPy is tried first and performs full primal validation in
    ``_milp_master``.  Gurobi results are independently checked by the same
    row-level validator.  The final greedy fallback is explicitly marked
    ``proven_optimal=False`` and therefore cannot support a global certificate.
    """
    R = [r for r, _ in routes]
    n = len(R)
    costs = np.asarray(costs if costs is not None else [1.0] * n, float)
    energy = np.asarray(energy if energy is not None else [0.0] * n, float)
    robust = np.asarray(robust if robust is not None else [0.0] * n, float)
    if any(v.shape != (n,) for v in (costs, energy, robust)):
        raise ValueError("master objective arrays must match route count")
    if not all(np.all(np.isfinite(v)) for v in (costs, energy, robust)):
        raise ValueError("master objective arrays must be finite")

    cover = {t: [] for t in all_turbine_ids}
    for k, r in enumerate(R):
        tids = list(r.turbine_ids())
        if len(tids) != len(set(tids)):
            raise ValueError(f"route {k} contains duplicate turbine ids")
        for tid in tids:
            if tid in cover:
                cover[tid].append(k)
    uncovered = [t for t, ks in cover.items() if not ks]
    slot_of = [getattr(getattr(r, "ship", None), "slot", None) for r in R]
    has_slots = any(s is not None for s in slot_of)

    # Preferred backend: SciPy MILP with full fail-closed validation.
    # Keep an audit bit for *every* attempted exact-solver output.  A rejected
    # success/optimal result may be ignored as an incumbent, but it must still
    # poison any downstream global-certificate chain for this solve.
    try:
        from scipy.optimize import milp as _scipy_milp_probe  # noqa: F401
        scipy_backend_available = True
    except Exception:
        scipy_backend_available = False
    solver_audit_reasons = []
    all_exact_outputs_validated = True
    res_milp = _milp_master(R, cover, costs, uncovered, partition=partition,
                            lex=lex, energy=energy, robust=robust,
                            slot_of=(slot_of if has_slots else None))
    if res_milp is not None:
        res_milp.update(all_exact_solver_outputs_validated=True,
                        exact_solver_output_rejected=False,
                        solver_audit_reasons=[])
        return res_milp
    if scipy_backend_available:
        all_exact_outputs_validated = False
        solver_audit_reasons.append("scipy_milp_output_rejected")

    def _audited(out):
        out.update(all_exact_solver_outputs_validated=bool(all_exact_outputs_validated),
                   exact_solver_output_rejected=bool(not all_exact_outputs_validated),
                   solver_audit_reasons=list(solver_audit_reasons))
        return out

    # Optional Gurobi fallback, also independently validated.
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as exc:
        log.warning("精确整数主问题不可用(%s); 返回无证书贪心解。", type(exc).__name__)
        out = _greedy_master(R, cover, costs, uncovered, lex=lex,
                             energy=energy, robust=robust,
                             slot_of=(slot_of if has_slots else None))
        out.update(proven_optimal=False, solution_validated=False,
                   validation_reason="exact_solver_unavailable_greedy_fallback")
        return _audited(out)

    tids = [t for t, ks in cover.items() if ks]
    A = np.zeros((len(tids), n), float)
    for ri, tid in enumerate(tids):
        for k in cover[tid]:
            A[ri, k] = 1.0
    cover_lb = np.ones(len(tids), float)
    cover_ub = np.ones(len(tids), float) if partition else np.full(len(tids), np.inf)
    slot_rows = []
    if has_slots:
        from collections import defaultdict as _dd
        sm = _dd(list)
        for k, sl in enumerate(slot_of):
            if sl is not None:
                sm[sl].append(k)
        for ks in sm.values():
            row = np.zeros(n, float); row[ks] = 1.0; slot_rows.append(row)

    def _build():
        model = gp.Model("route_master")
        model.Params.OutputFlag = 0
        z = model.addVars(n, vtype=GRB.BINARY, name="z")
        for tid, ks in cover.items():
            if not ks:
                continue
            expr = gp.quicksum(z[k] for k in ks)
            model.addConstr(expr == 1 if partition else expr >= 1,
                            name=("part_" if partition else "cov_") + str(tid))
        if has_slots:
            from collections import defaultdict as _dd
            sm = _dd(list)
            for k, sl in enumerate(slot_of):
                if sl is not None:
                    sm[sl].append(k)
            for sl, ks in sm.items():
                model.addConstr(gp.quicksum(z[k] for k in ks) <= 1,
                                name=f"slot_{sl}")
        return model, z

    def _validated_values(model, z, c, extra_rows=()):
        try:
            if model.Status != GRB.OPTIMAL or model.SolCount <= 0:
                return None, None, False, "solver_status_not_optimal"
            raw = np.asarray([z[k].X for k in range(n)], float)
            zz, value, reason = _validate_binary_master_vector(
                raw, c, A, cover_lb, cover_ub,
                slot_rows=slot_rows, extra_rows=extra_rows)
            if zz is None:
                return None, None, False, reason
            bound_ok, lp_lb, better_integer = _lp_relaxation_certifies(
                c, value, A, cover_lb, cover_ub,
                slot_rows=slot_rows, extra_rows=extra_rows)
            if better_integer:
                return None, None, False, "solver_optimality_claim_refuted"
            reason = None if bound_ok else "optimality_not_independently_proven"
            return zz, value, bool(bound_ok), reason
        except Exception:
            return None, None, False, "validation_exception"

    try:
        if not lex:
            model, z = _build()
            model.setObjective(gp.quicksum(float(costs[k]) * z[k] for k in range(n)), GRB.MINIMIZE)
            model.optimize()
            zz, value, l1_proven, reason = _validated_values(model, z, costs)
            if zz is not None:
                chosen = [k for k, v in enumerate(zz) if v > 0.5]
                return _audited(_master_result(
                    chosen, R, uncovered, "gurobi", value,
                    proven_optimal=l1_proven, solution_validated=True,
                    validation_reason=reason, l1_proven_optimal=l1_proven))
        else:
            model, z = _build()
            count_expr = gp.quicksum(z[k] for k in range(n))
            model.setObjective(count_expr, GRB.MINIMIZE); model.optimize()
            zz1, n_value, l1_proven, reason1 = _validated_values(model, z, np.ones(n))
            if zz1 is not None:
                n_star = int(round(n_value))
                model.addConstr(count_expr == n_star, name="lex_L1_exact")
                energy_expr = gp.quicksum(float(energy[k]) * z[k] for k in range(n))
                model.setObjective(energy_expr, GRB.MINIMIZE); model.optimize()
                count_row = (np.ones(n), float(n_star), float(n_star))
                zz2, E_star, l2_proven, reason2 = _validated_values(model, z, energy, [count_row])
                if zz2 is not None:
                    e_tol = 1e-7 * max(1.0, abs(float(E_star)))
                    model.addConstr(energy_expr >= E_star - e_tol, name="lex_L2_lb")
                    model.addConstr(energy_expr <= E_star + e_tol, name="lex_L2_ub")
                    robust_expr = gp.quicksum(float(robust[k]) * z[k] for k in range(n))
                    model.setObjective(-robust_expr, GRB.MINIMIZE); model.optimize()
                    energy_row = (energy, E_star - e_tol, E_star + e_tol)
                    zz3, negM, l3_proven, reason3 = _validated_values(
                        model, z, -robust, [count_row, energy_row])
                    if zz3 is not None:
                        chosen = [k for k, v in enumerate(zz3) if v > 0.5]
                        E_check = float(energy @ zz3); M_check = float(robust @ zz3)
                        lex_proven = bool(l1_proven and l2_proven and l3_proven)
                        reasons = [r for r in (reason1, reason2, reason3) if r]
                        return _audited(_master_result(
                            chosen, R, uncovered, "gurobi-lex",
                            (n_star, round(E_check, 1), round(M_check, 2)),
                            proven_optimal=lex_proven, solution_validated=True,
                            validation_reason=(";".join(reasons) if reasons else None),
                            lex_sorties=n_star, lex_energy=round(E_check, 1),
                            lex_robust=round(M_check, 2),
                            l1_proven_optimal=l1_proven,
                            l2_proven_optimal=l2_proven,
                            l3_proven_optimal=l3_proven))
    except Exception as exc:
        log.warning("Gurobi 主问题失败或解未通过验证: %s", type(exc).__name__)

    all_exact_outputs_validated = False
    solver_audit_reasons.append("gurobi_output_rejected_or_failed")
    out = _greedy_master(R, cover, costs, uncovered, lex=lex,
                         energy=energy, robust=robust,
                         slot_of=(slot_of if has_slots else None))
    out.update(proven_optimal=False, solution_validated=False,
               validation_reason="all_exact_solvers_failed_greedy_fallback")
    return _audited(out)




def _milp_master(R, cover, costs, uncovered, partition=False, lex=False,
                 energy=None, robust=None, slot_of=None):
    """Solve the restricted 0-1 master and independently validate every stage.

    A solver result is accepted only when it has ``success=True``, ``status==0``
    and a finite, correctly-sized, integral vector satisfying all coverage,
    partition, slot and lexicographic fixing rows.  No unvalidated result is
    returned to a certificate-producing caller.
    """
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
    except Exception:
        return None

    n = len(R)
    energy = np.asarray(energy if energy is not None else [0.0] * n, float)
    robust = np.asarray(robust if robust is not None else [0.0] * n, float)
    costs = np.asarray(costs if costs is not None else [1.0] * n, float)
    if any(v.shape != (n,) for v in (energy, robust, costs)):
        return None
    if not all(np.all(np.isfinite(v)) for v in (energy, robust, costs)):
        return None

    tids = [t for t, ks in cover.items() if ks]
    A = np.zeros((len(tids), n), float)
    for ri, t in enumerate(tids):
        for k in cover[t]:
            if not isinstance(k, (int, np.integer)) or not (0 <= int(k) < n):
                return None
            A[ri, int(k)] = 1.0
    cover_lb = np.ones(len(tids), float)
    cover_ub = (np.ones(len(tids), float) if partition
                else np.full(len(tids), np.inf, float))

    slot_rows = []
    if slot_of is not None:
        if len(slot_of) != n:
            return None
        from collections import defaultdict as _dd
        slotmap = _dd(list)
        for k, sl in enumerate(slot_of):
            if sl is not None:
                slotmap[sl].append(k)
        for ks in slotmap.values():
            row = np.zeros(n, float)
            row[ks] = 1.0
            slot_rows.append(row)

    if n == 0:
        objective = (0, 0.0, 0.0) if lex else 0.0
        return _master_result([], R, uncovered, "scipy-milp", objective,
                              proven_optimal=True, solution_validated=True,
                              lex_sorties=(0 if lex else None),
                              lex_energy=(0.0 if lex else None),
                              lex_robust=(0.0 if lex else None))

    bounds = Bounds(np.zeros(n), np.ones(n))
    integrality = np.ones(n)
    base_constraints = [LinearConstraint(A, cover_lb, cover_ub)]
    base_constraints += [LinearConstraint(row.reshape(1, -1), -np.inf, 1.0)
                         for row in slot_rows]

    def _solve_and_validate(c, extra_constraints=(), extra_rows=()):
        try:
            res = milp(c=np.asarray(c, float),
                       constraints=base_constraints + list(extra_constraints),
                       integrality=integrality, bounds=bounds)
        except Exception:
            return None, None, "solver_exception", False
        if res is None or getattr(res, "success", False) is not True:
            return None, None, "solver_failure", False
        if getattr(res, "status", None) != 0:
            return None, None, "solver_status_not_optimal", False
        zz, value, reason = _validate_binary_master_vector(
            getattr(res, "x", None), c, A, cover_lb, cover_ub,
            slot_rows=slot_rows, extra_rows=extra_rows)
        if zz is None:
            return None, None, reason, False
        fun = getattr(res, "fun", None)
        if fun is not None:
            try:
                fun = float(fun)
            except Exception:
                return None, None, "bad_objective", False
            if (not math.isfinite(fun) or
                    abs(fun - value) > 1e-6 * max(1.0, abs(value))):
                return None, None, "objective_mismatch", False
        bound_ok, lp_lb, better_integer = _lp_relaxation_certifies(
            c, value, A, cover_lb, cover_ub,
            slot_rows=slot_rows, extra_rows=extra_rows)
        if better_integer:
            return None, None, "solver_optimality_claim_refuted", False
        reason = None if bound_ok else "optimality_not_independently_proven"
        return zz, value, reason, bool(bound_ok)

    if not lex:
        z, value, reason, l1_proven = _solve_and_validate(costs)
        if z is None:
            return None
        chosen = [k for k, v in enumerate(z) if v > 0.5]
        return _master_result(chosen, R, uncovered, "scipy-milp", value,
                              proven_optimal=l1_proven, solution_validated=True,
                              validation_reason=reason,
                              l1_proven_optimal=l1_proven)

    ones = np.ones(n, float)
    z1, value1, reason1, l1_proven = _solve_and_validate(ones)
    if z1 is None:
        return None
    n_star = int(round(value1))
    if abs(value1 - n_star) > 1e-7:
        return None

    count_con = LinearConstraint(ones, float(n_star), float(n_star))
    count_row = (ones, float(n_star), float(n_star))
    z2, E_star, reason2, l2_proven = _solve_and_validate(
        energy, extra_constraints=[count_con], extra_rows=[count_row])
    if z2 is None:
        return None

    e_tol = 1e-7 * max(1.0, abs(float(E_star)))
    energy_con = LinearConstraint(energy, float(E_star) - e_tol,
                                  float(E_star) + e_tol)
    energy_row = (energy, float(E_star) - e_tol, float(E_star) + e_tol)
    z3, neg_robust, reason3, l3_proven = _solve_and_validate(
        -robust, extra_constraints=[count_con, energy_con],
        extra_rows=[count_row, energy_row])
    if z3 is None:
        return None
    chosen = [k for k, v in enumerate(z3) if v > 0.5]
    robust_star = float(robust @ z3)
    # Recompute every reported objective from the validated selected vector.
    n_check = int(np.sum(z3))
    E_check = float(energy @ z3)
    if n_check != n_star or abs(E_check - E_star) > e_tol:
        return None
    lex_proven = bool(l1_proven and l2_proven and l3_proven)
    reasons = [r for r in (reason1, reason2, reason3) if r]
    return _master_result(
        chosen, R, uncovered, "scipy-milp-lex",
        (n_star, round(E_check, 1), round(robust_star, 2)),
        proven_optimal=lex_proven, solution_validated=True,
        validation_reason=(";".join(reasons) if reasons else None),
        lex_sorties=n_star, lex_energy=round(E_check, 1),
        lex_robust=round(robust_star, 2),
        l1_proven_optimal=l1_proven, l2_proven_optimal=l2_proven,
        l3_proven_optimal=l3_proven)




def _greedy_master(R, cover, costs, uncovered, lex=False, energy=None, robust=None, slot_of=None):
    """贪心集合覆盖: 每步选"每单位成本新覆盖最多"的路由。
    lex=True: 词典序近似——主判据仍是新覆盖数/成本(趋向最少架次), 平手时优先
    高鲁棒裕度、再低能耗(贴近词典序 L2/L3 倾向)。无 Gurobi 时的近似, 非严格最优。
    Phase3: slot_of 给定时, 贪心也【遵守时隙打包】(每时隙至多选一条列), 已用时隙的列跳过;
    覆盖不完时把未覆盖风机计入 uncovered(单机时隙有限 ⇒ 可能覆盖不全, 诚实反映)。"""
    if costs is None:
        costs = [1.0] * len(R)
    route_cov = {k: set() for k in range(len(R))}
    for t, ks in cover.items():
        for k in ks:
            route_cov[k].add(t)
    need = {t for t, ks in cover.items() if ks}
    remaining, chosen = set(need), []
    used_slots = set()
    energy = energy if energy is not None else [0.0] * len(R)
    robust = robust if robust is not None else [0.0] * len(R)

    def _avail(k):
        if slot_of is not None and slot_of[k] is not None and slot_of[k] in used_slots:
            return False
        return True

    def _key(k):
        if not _avail(k):
            return (-1.0, 0.0, 0.0)
        gain = len(route_cov[k] & remaining)
        if gain == 0:
            return (-1.0, 0.0, 0.0)
        return (gain / costs[k], robust[k] if lex else 0.0,
                -energy[k] if lex else 0.0)

    while remaining:
        best = max(range(len(R)), key=_key)
        if not _avail(best):
            break
        gain = route_cov[best] & remaining
        if not gain:
            break
        chosen.append(best); remaining -= gain
        if slot_of is not None and slot_of[best] is not None:
            used_slots.add(slot_of[best])
    # 单机时隙有限 ⇒ 剩余未覆盖计入 uncovered(诚实)
    unco= list(uncovered) + sorted(remaining) if slot_of is not None else uncovered
    tot_E = sum(energy[k] for k in chosen)
    tot_M = sum(robust[k] for k in chosen)
    return dict(chosen=chosen, n_sorties=len(chosen), uncovered=unco,
                solver="greedy-lex-slot" if (slot_of is not None) else ("greedy-lex" if lex else "greedy"),
                objective=float(len(chosen)),
                lex_sorties=len(chosen), lex_energy=round(tot_E, 1),
                lex_robust=round(tot_M, 2),
                routes=[R[k] for k in chosen])


# =============================================================================
# 3. 顶层: 生成列 + 解主问题
# =============================================================================
def solve_route_drcc(turbines, ship, p, wx, xi_amb, objective="min_h",
                     max_stops=8, strategy="full", cost_kind="count",
                     partition=True, weather_unc=None, lex=True,
                     launch_ships=None, chance_mode="drcc",
                     budget_gamma=RM.BUDGET_GAMMA_DEFAULT) -> dict:
    """历史航路模型: 生成 DR 可行候选路由 → 历史集合划分主问题；不进入正式车队证书路径。
    lex=True(默认): 词典序三层目标(架次→能耗→鲁棒裕度 M_ω); 每条列携带其
      名义能耗 E_ω 与鲁棒裕度 M_ω, 主问题逐层优化(见 solve_master)。
    lex=False: 退回单目标 min Σ cost(cost_kind 决定 count/energy/...)。
    launch_ships(可选, 起飞—回收协同 / 任务2): list[ShipPrediction], 逐 τ 生成列并并池(见 generate_routes)。
    chance_mode(可选, 论文 baseline 对照): 'drcc'(本文, 默认)/'saa'(样本机会约束)/'budget'(预算鲁棒);
      三模式共用物理层/列生成/主问题, 仅替换 DRCC 判据 ⇒ 公平对比(见 step10.route_feasible_at_h)。
    返回选用路由、架次数、覆盖、各路由的最优 h 与名义能耗/时间/鲁棒裕度汇总。"""
    pool = generate_routes(turbines, ship, p, wx, xi_amb, objective, max_stops, strategy,
                           weather_unc, launch_ships=launch_ships,
                           chance_mode=chance_mode, budget_gamma=budget_gamma)
    # Phase3: 多起飞时隙(launch_ships 带 slot)+ 单机时隙打包下, 严格【划分】常因列池稀疏不可行 ⇒
    # 自动放宽为【覆盖】(≥1, 每台被≥1 架次巡检; 词典序 min 架次/能耗 仍抑制过度覆盖), 以达成"全场覆盖"目标。
    has_slot_pool = any(getattr(r.ship, "slot", None) is not None for r, _ in pool) if pool else False
    if has_slot_pool and partition:
        partition = False        # 转覆盖, 保证单机多时隙可达全场
    all_ids = [t.tid for t in turbines]
    # 每列的能耗 E_ω 与鲁棒裕度 M_ω(用其已选 h*)
    energy_arr, robust_arr = [], []
    for r, d in pool:
        h_use = r.fixed_h if r.fixed_h is not None else d.get("h")
        # L2: complete planned energy = flight + stern escort + docking reserve.
        if "E_plan_Wh" not in d:
            d = RM.route_feasible_at_h(r, int(h_use), p, _wx_of_route(r, wx), xi_amb,
                                       weather_unc=weather_unc, chance_mode=chance_mode,
                                       budget_gamma=budget_gamma)
        energy_arr.append(float(d.get("E_plan_Wh", d.get("E0", 0.0))))
        robust_arr.append(float(d.get("M_omega", 0.0)))
    if lex:
        res = solve_master(pool, all_ids, partition=partition, lex=True,
                           energy=energy_arr, robust=robust_arr)
    else:
        costs = [RM.route_cost(r, r.fixed_h, p, wx, kind=cost_kind) for r, _ in pool]
        res = solve_master(pool, all_ids, costs=costs, partition=partition)
    # 汇总选用路由
    chosen_routes = res["routes"]
    diag = []
    tot_E = 0.0; tot_M = 0.0
    n_multi = 0
    for r in chosen_routes:
        wxr = _wx_of_route(r, wx)   # 任务 #7: 该列 τ 天气
        d = RM.route_drcc_feasible(r, p, wxr, xi_amb, objective=objective, weather_unc=weather_unc,
                                   chance_mode=chance_mode, budget_gamma=budget_gamma)
        nom = RM.route_nominal_ET(r, d["h"], p, wxr, t_dock_s=float(d.get("t_dock_wait_s", 0.0)))
        E_plan = float(d.get("E_plan_Wh", nom["E0"] + float(d.get("E_dock_Wh", 0.0))))
        tot_E += E_plan; tot_M += float(d.get("M_omega", 0.0))
        n_multi += int(r.n_stops() > 1)
        diag.append(dict(rid=r.rid, stops=r.n_stops(), turbines=r.turbine_ids(),
                         h=d["h"], E0=round(nom["E0"], 1),
                         E_flight=round(float(d.get("E_flight_Wh", nom.get("E_flight", 0.0))), 1),
                         E_escort=round(float(d.get("E_escort_Wh", nom.get("E_escort", 0.0))), 1),
                         E_dock=round(float(d.get("E_dock_Wh", 0.0)), 1),
                         E_plan=round(E_plan, 1), T0=round(nom["T0"], 0),
                         M_omega=round(float(d.get("M_omega", 0.0)), 2)))
    res.update(dict(pool_size=len(pool), n_sorties_chosen=len(chosen_routes),
                    n_multi_stop=n_multi, total_energy_Wh=round(tot_E, 1),
                    total_robust_margin=round(tot_M, 2),
                    mean_stops=round(np.mean([r.n_stops() for r in chosen_routes]), 2)
                    if chosen_routes else 0.0,
                    route_diag=diag, objective_kind=("lex" if lex else cost_kind),
                    chance_mode=chance_mode))
    return res


# =============================================================================
# 4. 自检(占位船航迹 + 真实风机 + 占位多 h 模糊集)
# =============================================================================
def _selftest():
    here = Path(__file__).resolve().parent
    turb_csv = M._first_existing([here / "data" / "turbines_Rodsand_II_clean.csv"])
    p = M.Params()
    horizons = [5, 10, 15, 20, 30]
    states = ["直航", "转弯", "低速", "动力定位"]

    if turb_csv:
        turbines = M.load_turbines(turb_csv, farm="Rodsand_II")[:10]
    else:
        turbines = [M.Turbine(f"DEMO_{i}", np.array([11.55 + 0.006 * i, 54.55 + 0.002 * (i % 3)]),
                              68.5, 115.0) for i in range(10)]
    lat0, lon0 = turbines[0].lonlat[1], turbines[0].lonlat[0]
    for t in turbines:
        t.local = M.latlon_to_local_m(t.lonlat[1], t.lonlat[0], lat0, lon0)

    wx = dict(wind10=6.0, wind_dir_from=230.0, Hs=0.5, Tp=2.1, wave_dir=200.0, ship_heading=90.0)
    xi_amb = RM._demo_xi(horizons, states)
    # 起飞点在风机群质心西南; 船向东北慢速航行
    centroid = np.mean([t.local for t in turbines], axis=0)
    ship = RM.ShipPrediction.from_cv(centroid + np.array([-700.0, -500.0]),
                                     v_ship=np.array([2.0, 1.2]), horizons=horizons, c_state="直航")

    print("\n================ step11_algorithm_route_drcc.py 自检 ================")
    print(f"风机 {len(turbines)} 台 | 船状态 {ship.c_state} | h 候选 {horizons} | κ(ε=.05)={RM.kappa(.05):.2f}")

    # 单台路由对照 vs 多台航路
    res_single = solve_route_drcc(turbines, ship, p, wx, xi_amb, strategy="singleton")
    res_route = solve_route_drcc(turbines, ship, p, wx, xi_amb, strategy="full", max_stops=8)

    print("\n--- 单台路由(对照) ---")
    print(f"  候选路由池 {res_single['pool_size']} | 选用架次 {res_single['n_sorties_chosen']} | "
          f"求解器 {res_single['solver']} | 未覆盖 {len(res_single['uncovered'])} | "
          f"计划总能耗 {res_single['total_energy_Wh']}Wh")
    print("\n--- 多台航路化(决策依赖 h) ---")
    print(f"  候选路由池 {res_route['pool_size']} | 选用架次 {res_route['n_sorties_chosen']} | "
          f"多台架次 {res_route['n_multi_stop']} | 均停靠 {res_route['mean_stops']} | "
          f"求解器 {res_route['solver']} | 未覆盖 {len(res_route['uncovered'])} | "
          f"计划总能耗 {res_route['total_energy_Wh']}Wh")

    print("\n--- 对比解读 ---")
    print(f"  架次数: 单台 {res_single['n_sorties_chosen']} → 航路 {res_route['n_sorties_chosen']} "
          f"(航路化合并使一次出动巡检多台, 架次显著下降)。")
    print("\n--- 选用路由明细(航路模型, 前 6 条)---")
    print("   架次  停靠  最优h(min)  名义E0(Wh)  名义T0(s)  访问风机")
    for d in res_route["route_diag"][:6]:
        print(f"   r{d['rid']:<3d}  {d['stops']:^4d}  {d['h']:^9d}  {d['E0']:^9.1f}  "
              f"{d['T0']:^8.0f}  {','.join(t.split('_')[-1] for t in d['turbines'])}")

    print("\n自检完成。历史启发式列生成与历史集合划分主问题链路已跑通。")
    if res_route["solver"] == "greedy":
        print("提示: 未检测到 Gurobi, 主问题用了贪心回退。装 gurobipy 可得精确集合覆盖最优。")
    print("顶刊精确算法待办: 把启发式定价替换为 DR-RCSPP 最优定价 + 分支定价(见文件头 §尾注)。")


if False and __name__ == "__main__":  # legacy research demo: never auto-executed
    _selftest()


# =============================================================================
# 机队资源主问题 —— 移动母船·K 机·池化 2K 电池·两层词典序(作者定案)
# =============================================================================
def deck_indices(tau_min: float, h_min: float, t_swap_min: float, t_launch_min: float,
                 deck_delta_min: float, n_tgrid: int, deck_mode: str = "interval") -> list:
    r"""旧 step12/B&P 兼容用的甲板格点函数，不是当前 ``solve_resource_master`` 的正式资源口径。

    当前阶段4主问题直接使用连续区间：起飞准备 + 着陆清场；换电在非甲板工位，且只有
    相邻任务实际更换电池时才发生。此函数仍保留旧 B&P 的“回收后固定服务”语义，待
    step12 资源定价重构阶段一并替换，不能用来解释阶段4 restricted master。

    deck_mode='interval'(新默认, 物理更对): 起降是有时长的甲板【区间】占用 ——
      更新 修复(审计 P1-τ 语义): 定死 τ=【离舰时刻】(与论文"occupies the deck for launch
      preparation and leaves the vessel at time τ"一致)⇒ 起飞准备占 [max(τ−t_launch,0), τ)
      (窗首起飞的准备越过窗起点时按 0 截断并如实记录), 回收+换电占 [τ+h, τ+h+t_swap)。
      旧口径把准备记在 [τ, τ+t_launch) —— 物理飞行已从 τ 开始, 甲板却在 τ 后仍被记作准备,
      同一架次被双重占用且与论文陈述矛盾。
      正确性口径(更新 收敛措辞): 每条列占用甲板的是【两个不连通区间】(准备 ∪ 回收换电),
      列级冲突图是 2-interval 图而非普通区间图 —— 逐格点容量-1 行是【有效不等式】, 只要
      Δ 网格覆盖全部区间端点即足以精确阻止任何整数解的甲板重叠(整数可行性充分),
      但**不**构成该 stable-set 多面体的完整线性描述(旧注释的"完美图/全部极大团"断言过强,
      LP 根松弛可能比其声称的更松; 证书仍由 B&P 的整数分支闭合)。
    deck_mode='slot'(旧口径, 保留做消融/回归): 起飞/回收各占单个 Δ 槽(瞬时事件, 事件点=τ 与 τ+h)。"""
    idx = set()
    if deck_mode == "slot":
        idx.add(int(round(tau_min / deck_delta_min)))
        idx.add(int(round((tau_min + h_min) / deck_delta_min)))
    else:
        for (a, b) in ((max(tau_min - t_launch_min, 0.0), tau_min),
                       (tau_min + h_min, tau_min + h_min + t_swap_min)):
            i0 = int(math.ceil((a - 1e-9) / deck_delta_min))
            i1 = int(math.floor((b - 1e-9) / deck_delta_min))
            for i in range(max(i0, 0), i1 + 1):
                if a - 1e-9 <= i * deck_delta_min < b - 1e-9:
                    idx.add(i)
    return sorted(i for i in idx if 0 <= i < n_tgrid)


def _e_per_m_lb(p) -> float:
    """更新: 每米能耗的【下界】(零风、全速度网格最优空速)。更新: 仅供 legacy2d 消融保留。"""
    vs = np.linspace(6.0, 22.0, 33)
    return float(min(M.P_zeng(v, p) / max(v, 1e-6) for v in vs)) / 3600.0


def _e_per_m_lb_tailwind(p, w_max: float) -> float:
    """更新(审计修复#2): 顺风最优下的每米【地面距离】能耗下界。
    与仿真同一功率派发 M.leg_power(Zeng 或 legacy 立方, 由 p.use_zeng 决定; 实际 v_eff 只取
    {v_cr, v_air_max}, 扫描连续 v 只会让下界更松 ⇒ 排除方向保真)。
    地速 v_g ≤ v_air + w_max(全顺风), 故 E/米 ≥ min_v leg_power(v)/(v+w_max)。"""
    v_lo = max(0.1, min(3.0, float(getattr(p, "v_air_min", 3.0))))   # 更新: 覆盖可配置 v_air_min
    v_hi = float(getattr(p, "v_air_max", 22.0))
    vs = list(np.linspace(v_lo, v_hi, 40))
    # 更新(外部审计 轻微#4): 有限网格取最小值【不是】连续区间的严格下界 —— 网格可能
    #   跳过真实最优速度, 使"下界"高于实际每米能耗, 理论上可错误排除可行列。
    #   仿真的功率派发只用 v_eff ∈ {v_cr, v_air_max}(见 step9.leg_kinematics), 故把这两个
    #   【真实候选速度】连同端点强制并入扫描集合 —— 固定速度模式下由此得到的最小值恒 ≤
    #   任何真实腿的每米能耗(排除方向保真), 且成本为零。
    #   (实测该修复前网格仅比连续最小高 ~1.8e-4 相对量, 需约 1168 km 单腿才累积 1 Wh 高估,
    #    故属防御性加固而非现网缺陷; speed_adjustable=True 的连续速度档见 tau_reach 说明。)
    for _v in (float(getattr(p, "v_cr", 15.0)), v_hi, v_lo,
               float(getattr(p, "v_air_min", 3.0))):
        if v_lo - 1e-9 <= _v <= v_hi + 1e-9:
            vs.append(_v)
    return float(min(M.leg_power(p, v) / max(v + max(w_max, 0.0), 1e-6) for v in vs)) / 3600.0


def _reach_wind_max_cruise(p, wx: dict | None, turbines) -> float:
    """更新: 排除界所用的巡航高度风速上界 = max(全局 wx, 各风机 wx_local) 升尺度。"""
    ws = []
    if wx is not None:
        w10 = wx.get("wind10")
        if w10 is not None and not (isinstance(w10, float) and math.isnan(w10)):
            ws.append(float(w10))
    for t in turbines:
        loc = getattr(t, "wx_local", None)
        if loc is not None:
            w10 = loc.get("wind10")
            if w10 is not None and not (isinstance(w10, float) and math.isnan(w10)):
                ws.append(float(w10))
    w10_max = max(ws) if ws else 6.7
    return float(M.wind_at_height(w10_max, p.z_cruise, p.z0))


class ReachResult:
    r"""更新(H-02): tau_reach 的【结构化运行时证明对象】。旧口径只返回风机列表, 证书
    装配层无从得知预筛实际以何种模式生效、排除了谁、前提是否成立 —— 只能"再次调用同一
    判断函数"推断(对'删除自动降级'/'强制早剪'类回退完全失明, 两个单点变异实测存活)。
    现在字段全部来自【本次调用内的实际执行事实】:
      turbines        实际返回的风机(本对象可当 list 迭代, 向后兼容);
      requested_mode  调用方请求的 mode;
      effective_mode  实际生效档: 'off' | 'valid-proven' | 'legacy2d'
                      (off = 未做任何排除, 含自动降级; valid-proven = 保真前提成立且做了
                       去程下界排除; legacy2d = 不保真消融);
      excluded_tids   实际被排除的风机 tid(由输出集合直接构造, 不是声明);
      mean_relax_free / speed_adjustable  本次调用观测到的两个保真前提事实;
      proof_complete  正常执行到返回点(异常/截断路径拿不到 True);
      proof_reason    人类可读的档位原因。
    证书条件 reach_filter_proven_safe 只读取这些对象, 并由调用方独立复核
    excluded_tids 与实际返回集合一致(见 step12 solve_branch_cut_price)。"""

    __slots__ = ("turbines", "requested_mode", "effective_mode", "excluded_tids",
                 "mean_relax_free", "speed_adjustable", "proof_complete", "proof_reason")

    def __init__(self, turbines, requested_mode, effective_mode, excluded_tids,
                 mean_relax_free, speed_adjustable, proof_complete, proof_reason):
        self.turbines = list(turbines)
        self.requested_mode = str(requested_mode)
        self.effective_mode = str(effective_mode)
        self.excluded_tids = tuple(sorted(excluded_tids))
        self.mean_relax_free = bool(mean_relax_free)
        self.speed_adjustable = bool(speed_adjustable)
        self.proof_complete = bool(proof_complete)
        self.proof_reason = str(proof_reason)

    # ---- list 兼容(既有调用方按可迭代序列使用) ----
    def __iter__(self):
        return iter(self.turbines)

    def __len__(self):
        return len(self.turbines)

    def __getitem__(self, i):
        return self.turbines[i]

    def __repr__(self):
        return (f"ReachResult(n={len(self.turbines)}, mode={self.requested_mode}→"
                f"{self.effective_mode}, excl={len(self.excluded_tids)}, "
                f"mrf={self.mean_relax_free}, spd={self.speed_adjustable}, "
                f"complete={self.proof_complete})")


def tau_reach(opt, turbines, p, h_max_min: float, mode: str = "valid", wx: dict | None = None,
              xi_amb=None, weather_unc=None) -> "ReachResult":
    r"""该起飞时隙 τ 的可达风机预筛。更新(审计修复#2)提供三种模式:

    mode='valid'(新默认, 保真 ⇒ 不破坏全局最优证书): 只用【去程】下界 ——
      任何服务 j 的列, 其到 j 为止的地面航程 ≥ d = |q_j − P_launch(τ)|(三角不等式,
      与后续访问/回收点无关); 名义地速 ≤ v_air_max + w_max(巡航高全顺风上界)。
      时间下界:  T0 ≥ T_to + T_land + τ_insp + d/(v_air_max + w_max);
      能量下界:  E0 ≥ E_to + E_land + E_insp(j) + d · min_v leg_power(v)/(v+w_max)/3600
      (leg_power 与仿真同派发: use_zeng ? P_zeng : legacy 立方)。
      可行必要条件(b_T≥0 ⇒ T0 ≤ 60·h_max − t_dock ≤ 60·h_max; b_E≥0 ⇒ E0 ≤ B_use −
      E_dock ≤ B_use; 均向松放宽 t_dock/E_dock→0, 排除方向保真)。任一下界超预算 ⇒
      j 在该 τ 的任何列中不可行。
      更新(采纳外部审计 6.1, 并扩展): "DRCC/天气裕度只增不减"只在【无带符号均值
      放松】时成立 —— 联合 SOC 的 a_wᵀ·wind_bias 与 aᵀ·μ(ξ 均值)可为负, 把接受判据
      放松到名义预算之下, 使"名义 E0>B_use ⇒ 必不可行"失效(外部审计给出可执行反例:
      顺风偏置下名义 E0=323Wh>B_use=262Wh 仍 DRCC 可行)。因此 'valid' 仅在
      RM.mean_relax_free(xi_amb, weather_unc) 为真时启用; 否则本函数【自动降级为
      'off'】(返回全集是 valid 的超集 ⇒ 排除方向恒保真, 证书无需撤销, 只是更慢)。
      调用方不传 xi_amb 视为不可判定, 同样降级(保守)。

    mode='off': 不预筛(全部返回)。证书等价于 'valid', 仅更慢。

    mode='legacy2d'(更新 旧口径, 仅消融保留, 【不保真】): 用"闭合回路 ≥ 2d"。
      该论证在【移动母船】下不成立(船可向 j 靠近, 返程 < d), 且时间界未计顺风地速
      > v_air_max —— 可能错误排除可行列 ⇒ 使用它时上层必须撤销全局最优证书。
    """
    # ---- 更新(H-02): 两个保真前提【无条件先观测并记录】(运行事实, 非声明) ----
    #   即使后续降级分支被误删(变异: "tau_reach 不降级"), 这两个字段仍如实记录当时的
    #   数据事实; 上层证书据此判定 valid-proven 档的排除是否有数学依据。
    _mrf = bool(RM.mean_relax_free(xi_amb, weather_unc))
    _spd = bool(getattr(p, "speed_adjustable", False))

    def _proof(out, effective, reason, complete=True):
        excl = sorted({t.tid for t in turbines} - {t.tid for t in out})
        return ReachResult(out, mode, effective, excl, _mrf, _spd, complete, reason)

    if mode == "off":
        return _proof(list(turbines), "off", "requested-off")
    if mode == "valid" and not _mrf:
        # 更新: 带符号均值可放松判据 ⇒ 名义必要条件不保真, 自动 off
        return _proof(list(turbines), "off", "degraded-to-off-nonzero-mean")
    # 更新(外部审计 轻微#4): speed_adjustable=True 时可行性走 _feasible_speed_adjustable,
    #   允许【连续】速度追赶, 而 _e_per_m_lb_tailwind 的有限网格最小值对连续区间没有
    #   经证明的下界性质 ⇒ 该档同样自动降级为 off(返回全集是超集, 排除方向恒保真)。
    if mode == "valid" and _spd:
        return _proof(list(turbines), "off", "degraded-to-off-speed-adjustable")
    P0 = np.asarray(opt.ship.P_launch, float)
    if mode == "legacy2d":
        e_pm = _e_per_m_lb(p)
        v_max = max(float(getattr(p, "v_air_max", getattr(p, "v_cruise", 22.0))), 22.0)
        T_cap = h_max_min * 60.0 - p.tau_insp
        E_cap = p.B_use
        out = []
        for t in turbines:
            d = float(np.linalg.norm(np.asarray(t.local, float) - P0))
            if 2.0 * d / v_max <= T_cap + 1e-9 and 2.0 * d * e_pm <= E_cap + 1e-9:
                out.append(t)
        return _proof(out, "legacy2d", "legacy2d-ablation-not-faithful")
    # ---- mode='valid': 去程-only 保真排除(此处 _mrf=True 且 _spd=False 已证成立) ----
    wx_use = wx if wx is not None else getattr(opt, "wx", None)
    w_max = _reach_wind_max_cruise(p, wx_use, turbines)
    e_pm = _e_per_m_lb_tailwind(p, w_max)
    v_g_max = float(getattr(p, "v_air_max", 22.0)) + w_max
    E_to, E_land, T_to, T_land = M.to_land_energy_time(p)
    T_cap = h_max_min * 60.0
    out = []
    for t in turbines:
        d = float(np.linalg.norm(np.asarray(t.local, float) - P0))
        dz = M.insp_vertical_span(t, p.z_cruise)
        if getattr(p, "use_zeng", False):
            P_up = M.P_zeng(0.0, p) + 7.27 * 9.81 * p.v_z
            E_insp = P_up * dz / p.v_z / 3600.0 + M.P_zeng(p.v_orbit, p) * p.tau_insp / 3600.0
        else:
            E_insp = p.P_climb * dz / p.v_z / 3600.0 + p.P_hov * p.tau_insp / 3600.0
        T_lb = T_to + T_land + p.tau_insp + dz / p.v_z + d / max(v_g_max, 1e-6)
        E_lb = E_to + E_land + E_insp + d * e_pm
        if T_lb <= T_cap + 1e-9 and E_lb <= p.B_use + 1e-9:
            out.append(t)
    return _proof(out, "valid-proven", "outbound-only-lower-bounds")


def enumerate_discrete_routes(turbines, opts, p, xi_amb, T_min, deck_delta_min, max_stops,
                              weather_unc, kappa_mode="vp_unimodal", verbose=False,
                              max_evals=None, reach_mode: str = "valid",
                              deadline: float | None = None):
    """更新【广延式全枚举池】(A2 的 Gurobi 对照臂; 与 B&P 定价空间同一问题):
    每 τ: 可达集(tau_reach) → 全部 ≤max_stops 【有序序列】 → 全部可行 h(τ+h≤T) → 列。
    同 (τ, 有序风机序列, h) 只留能耗最小者；访问集合相同但顺序不同的路线绝不合并。
    返回 (cols, stats: 每 τ 可达/序列/评估数)。规模会先于 B&P 爆炸 —— 这正是 A2 要画的曲线。"""
    import itertools as _it
    horizons = RM.decision_horizons_of(xi_amb)
    if str(kappa_mode) not in set(RM.KAPPA_MODES) | {"nominal"}:
        raise ValueError(f"unknown kappa_mode={kappa_mode!r}")
    risk_policy = RM.risk_policy_for_mode(str(kappa_mode))
    cols, best = [], {}
    n_seq = n_eval = 0
    status = "ok"
    _reach_proofs = []
    try:
        for oi, opt in enumerate(opts):
            if deadline_reached(deadline):
                status = "time_limit"
                break
            if status in ("eval_limit", "time_limit"):
                break
            reach = tau_reach(opt, turbines, p, max(horizons), mode=reach_mode,
                              wx=getattr(opt, "wx", None), xi_amb=xi_amb,
                              weather_unc=weather_unc)
            _reach_proofs.append(reach)
            for k in range(1, max_stops + 1):
                if deadline_reached(deadline):
                    status = "time_limit"
                    break
                if status in ("eval_limit", "time_limit"):
                    break
                for perm in _it.permutations(reach, k):
                    if deadline_reached(deadline):
                        status = "time_limit"
                        break
                    if max_evals is not None and n_eval >= max_evals:
                        # 更新 哨兵: 广延式枚举极易失控(实测 n=20/stops=4 已 2557 万评估/3.9h),
                        # 触限即停并如实标注 —— 该行只能进 scaling diagnostic, 不能进 speedup 主表。
                        log.warning("枚举触达 --max-ext-evals=%d(τ#%d, 列 %d)—— 提前终止, status=eval_limit。",
                                    max_evals, oi, len(best))
                        status = "eval_limit"
                        break
                    n_seq += 1
                    r = RM.Route(rid=-1, turbines=list(perm), ship=opt.ship)
                    for h in horizons:
                        if deadline_reached(deadline):
                            status = "time_limit"
                            break
                        if float(opt.tau_min) + float(h) > float(T_min):
                            break
                        n_eval += 1
                        dd = RM.route_feasible_at_h(
                            r, int(h) if float(h).is_integer() else float(h),
                            p, opt.wx, xi_amb, weather_unc=weather_unc,
                            chance_mode="drcc", risk_policy=risk_policy)
                        if not dd["feasible"]:
                            continue
                        ordered_tids = tuple(t.tid for t in perm)
                        # Exact finite-state identity: the launch-option index is part of
                        # the caller-declared finite route space, and h keeps its binary64
                        # representation.  Do not decimal-round tau or merge ULP-distinct
                        # launch states.
                        key = (int(oi), ordered_tids, float(h).hex())
                        E0 = float(dd.get("E_plan_Wh", dd["E0"] + dd.get("E_dock_Wh", 0.0)))
                        Esoc = float(dd.get("E_soc_required_Wh", E0))
                        if key in best:
                            old = best[key]
                            old_plan = float(old.get("E_plan_Wh", old["E0"]))
                            old_soc = float(old.get("E_soc_required_Wh", old_plan))
                            # Strict Pareto dominance only; no fixed energy tolerance.
                            if old_plan <= E0 and old_soc <= Esoc:
                                continue
                            if not (E0 <= old_plan and Esoc <= old_soc):
                                # Incomparable diagnostics for the same declared state are
                                # retained separately rather than silently deleting one.
                                key = (int(oi), ordered_tids, float(h).hex(),
                                       E0.hex(), Esoc.hex())
                        best[key] = dict(tau=float(opt.tau_min), ship=opt.ship, wx=opt.wx,
                                         route=r, tids=tuple(sorted(ordered_tids, key=str)),
                                         ordered_tids=ordered_tids, h=float(h), E0=E0,
                                         E_plan_Wh=E0, E_soc_required_Wh=Esoc,
                                         E0_nominal=float(dd.get("E0", E0)),
                                         kappa_used=kappa_mode,
                                         gate_proof=dd.get("gate_weather_proof"),
                                         launch_slot=int(round(opt.tau_min / deck_delta_min)),
                                         rec_slot=int(round((opt.tau_min + h) / deck_delta_min)))
            if verbose and (oi + 1) % 10 == 0:
                log.info("枚举进度 %d/%d τ: 序列 %d, 评估 %d, 列 %d", oi + 1, len(opts), n_seq, n_eval, len(best))
    finally:
        pass
    cols = list(best.values())
    # 更新(H-02): reach_effective / anchor_complete 由【实际返回的 ReachResult 运行时
    # 证明】导出, 不再由"再次调用 mean_relax_free 重推理论上应该怎样"得到; 并做调用方侧
    # 独立复核 —— excluded_tids 与实际返回集合逐一致(证明对象撒谎即失效)。
    _effs = {r.effective_mode for r in _reach_proofs} or {"off"}
    _reach_safe = bool(_reach_proofs) and all(
        r.proof_complete and r.effective_mode in ("off", "valid-proven")
        and (r.effective_mode != "off" or len(r.excluded_tids) == 0)
        and (r.effective_mode != "valid-proven"
             or (r.mean_relax_free and not r.speed_adjustable))
        and tuple(sorted({t.tid for t in turbines} - {t.tid for t in r.turbines}))
        == tuple(sorted(r.excluded_tids))
        for r in _reach_proofs)
    if _effs == {"off"}:
        _eff_report = "off"
    elif _effs <= {"off", "valid-proven"}:
        _eff_report = "valid-proven"
    else:
        _eff_report = ",".join(sorted(_effs))
    return cols, dict(n_seq=n_seq, n_eval=n_eval, n_cols=len(cols), status=status,
                      reach_mode=reach_mode,
                      reach_effective=("off" if _eff_report == "off" else
                                       ("valid" if _eff_report == "valid-proven" else _eff_report)),
                      reach_proofs=list(_reach_proofs),
                      reach_filter_proven_safe=_reach_safe,
                      mean_relax_free=bool(all(r.mean_relax_free for r in _reach_proofs)
                                           if _reach_proofs else False),
                      # 更新(审计修复#2)+更新(H-02): 锚点完整性 = 状态 ok 且全部
                      # 排除由运行时证明背书(off 无排除 / valid-proven 前提成立)。
                      anchor_complete=bool(status == "ok" and _reach_safe),
                      route_space_complete=bool(status == "ok" and _reach_safe))




def _column_plan_energy(c: dict) -> float:
    """第二层目标：飞行 + 船尾伴飞 + 末端着舰储备。"""
    return float(c.get("E_plan_Wh", c.get("E0", 0.0)))


def _column_soc_energy(c: dict) -> float:
    """实体电池组累计占用能量；缺字段时保守退回计划能量。"""
    return float(c.get("E_soc_required_Wh", _column_plan_energy(c)))


def _halfopen_overlap(a, b) -> bool:
    """Strict half-open overlap on the binary64 interval endpoints.

    The formal resource model treats the stored endpoints as the finite model
    data.  No positive tolerance may enlarge feasibility: [a0,a1) and [b0,b1)
    overlap iff max(a0,b0) < min(a1,b1).
    """
    return max(a[0], b[0]) < min(a[1], b[1])


def _resource_intervals(c: dict, t_launch_min: float, t_clear_min: float,
                        t_swap_min: float = 0.0, deck_mode: str = "interval",
                        deck_delta_min: float = 2.5) -> dict:
    """固定任务本体资源区间；换电/快速检查由相邻任务转换决定。

    ``tau`` 是离舰时刻，``launch_start`` 是甲板最终准备开始时刻，``clear_end``
    是接地、停桨并移出着陆区的时刻。甲板只包含起飞准备与着陆清场；不能在列级
    预先假定每个架次都换电，因为保留原电池时只需快速检查。
    """
    tau, h = float(c["tau"]), float(c["h"])
    launch_a = max(tau - float(t_launch_min), 0.0)
    rec = tau + h
    clear_end = rec + max(float(t_clear_min), 0.0)
    if deck_mode == "slot":
        d = max(float(deck_delta_min), 1e-6)
        ls = int(round(tau / d)) * d
        rs = int(round(rec / d)) * d
        deck = ((ls, ls + d), (rs, rs + d))
    else:
        deck_parts = []
        if tau > launch_a:
            deck_parts.append((launch_a, tau))
        if clear_end > rec:
            deck_parts.append((rec, clear_end))
        deck = tuple(deck_parts)
    return dict(
        deck=deck,
        active=(launch_a, clear_end),
        launch_start_min=launch_a,
        launch_min=tau,
        recovery_min=rec,
        clear_end_min=clear_end,
    )


def _interval_capacity_rows(intervals, min_size=2):
    """Exact half-open capacity rows at every nonempty interval start."""
    starts = sorted({float(a) for a, b in intervals if float(b) > float(a)})
    rows, seen = [], set()
    for t in starts:
        js = tuple(j for j, (a, b) in enumerate(intervals)
                   if float(a) <= t < float(b))
        if len(js) >= int(min_size) and js not in seen:
            seen.add(js); rows.append(list(js))
    return rows


def _multi_interval_conflict_rows(resources):
    """容量为1的多区间资源（甲板）用连续时间逐对冲突行表达整数可行性。"""
    rows = []
    for j in range(len(resources)):
        for k in range(j + 1, len(resources)):
            if any(_halfopen_overlap(a, b) for a in resources[j] for b in resources[k]):
                rows.append([j, k])
    return rows


def _event_capacity_ok(events, candidate, capacity: int) -> bool:
    """Strict half-open capacity check preserving exact Fraction endpoints."""
    if candidate[1] <= candidate[0]:
        return True
    all_events = list(events) + [candidate]
    starts = sorted({a for a, b in all_events if b > a})
    for t in starts:
        if sum(a <= t < b for a, b in all_events) > int(capacity):
            return False
    return True


def audit_resource_assignment(cols, selected, K: int, B: int, B_use: float,
                              resource_of, quick_min: float, swap_min: float,
                              quick_capacity: int, swap_capacity: int,
                              deadline: float | None = None) -> ResourceAuditResult:
    """Exact tri-state UAV/battery/SOC/turnaround audit.

    ``INFEASIBLE_PROVEN`` is returned only after the complete DFS search closes.
    ``UNKNOWN_TIMEOUT`` never authorizes a resource-pattern exclusion cut.
    """
    return exact_resource_audit(
        cols, selected, uav_count=K, battery_count=B,
        usable_battery_energy_Wh=B_use, resource_of=resource_of,
        quick_inspection_min=quick_min, swap_min=swap_min,
        quick_capacity=quick_capacity, swap_capacity=swap_capacity,
        overlap=_halfopen_overlap, event_capacity_ok=_event_capacity_ok,
        deadline=deadline)


def _solve_logic_benders_anytime(cols, all_tids, K, B, B_use, resource_of,
                                  deck_rows, quick_min, swap_min, quick_capacity,
                                  swap_capacity, *, solver="auto", deadline=None,
                                  capacity_rows=(), pooled_energy_cap=None,
                                  coverage_gap_target_abs=0,
                                  energy_gap_target_rel=0.0,
                                  energy_gap_target_abs_Wh=1e-6):
    """Two-phase set-packing master with exact logic-based Benders cuts."""
    no_goods: list[tuple[int, ...]] = []
    seen_cuts: set[tuple[int, ...]] = set()
    resource_nodes = 0
    unknown_resource_audit = False
    restricted_gap_pct = None
    last_backend = None

    empty_audit = audit_resource_assignment(
        cols, [], K, B, B_use, resource_of, quick_min, swap_min,
        quick_capacity, swap_capacity, deadline=deadline)
    if empty_audit.status is ResourceAuditStatus.UNKNOWN_TIMEOUT:
        return dict(selected=[], audit=None, coverage_incumbent=0,
                    coverage_upper_bound=len(all_tids), coverage_optimal=False,
                    energy_incumbent_Wh=None, energy_lower_bound_Wh=None,
                    energy_optimal=False, no_goods=no_goods,
                    resource_audit_complete=False, termination_reason="resource-audit-time-limit",
                    backend=None, restricted_pool_gap_pct=None, resource_nodes=0)
    incumbent_sel: tuple[int, ...] = ()
    incumbent_audit = empty_audit.assignment
    incumbent_cov = 0
    # Fast resource-feasible seed.  The disjointness test enforces the same
    # set-packing rows used by both exact MILP backends.
    covered_seed: set = set()
    for j in sorted(range(len(cols)),
                    key=lambda q: (-len(cols[q]["tids"]), float(cols[q]["E_plan_Wh"]), q)):
        if deadline_reached(deadline):
            break
        tids = set(cols[j]["tids"])
        if tids & covered_seed:
            continue
        trial = tuple(list(incumbent_sel) + [j])
        seed_audit = audit_resource_assignment(
            cols, trial, K, B, B_use, resource_of, quick_min, swap_min,
            quick_capacity, swap_capacity, deadline=deadline)
        resource_nodes += int(seed_audit.explored_nodes)
        if seed_audit.status is ResourceAuditStatus.UNKNOWN_TIMEOUT:
            unknown_resource_audit = True
            break
        if seed_audit.status is ResourceAuditStatus.FEASIBLE:
            incumbent_sel = trial
            incumbent_audit = seed_audit.assignment
            covered_seed.update(tids)
            incumbent_cov = len(covered_seed)
    coverage_ub = len(all_tids)
    coverage_proven = False
    termination = "coverage-search-not-started"

    if str(solver).lower() == "greedy":
        greedy_energy = float(sum(float(cols[j]["E_plan_Wh"]) for j in incumbent_sel))
        return dict(
            selected=list(incumbent_sel), audit=incumbent_audit,
            coverage_incumbent=int(incumbent_cov), coverage_upper_bound=int(coverage_ub),
            coverage_optimal=False, energy_incumbent_Wh=greedy_energy,
            energy_lower_bound_Wh=None, energy_optimal=False, no_goods=[],
            resource_audit_complete=not unknown_resource_audit,
            termination_reason=("resource-audit-time-limit" if unknown_resource_audit
                                else "greedy-resource-feasible-fallback"),
            backend="greedy", restricted_pool_gap_pct=None,
            resource_nodes=int(resource_nodes))

    while not deadline_reached(deadline):
        min_cov = incumbent_cov + 1 if incumbent_cov > 0 else None
        master = solve_binary_master(
            solver, cols, all_tids, deck_rows, no_goods, phase="coverage",
            deadline=deadline, capacity_rows=capacity_rows,
            pooled_energy_cap=pooled_energy_cap, coverage_min=min_cov)
        last_backend = master.backend
        restricted_gap_pct = master.restricted_pool_gap_pct
        ub = safe_coverage_upper_bound(master, len(all_tids))
        if ub is not None:
            coverage_ub = min(coverage_ub, max(incumbent_cov, ub))
        if master.x is None:
            if master.infeasible_proven:
                coverage_ub = incumbent_cov
                coverage_proven = True
                termination = "coverage-optimum-proven"
            else:
                termination = "coverage-master-time-limit"
            break
        selected = tuple(np.flatnonzero(master.x > 0.5).tolist())
        audit = audit_resource_assignment(
            cols, selected, K, B, B_use, resource_of, quick_min, swap_min,
            quick_capacity, swap_capacity, deadline=deadline)
        resource_nodes += int(audit.explored_nodes)
        if audit.status is ResourceAuditStatus.UNKNOWN_TIMEOUT:
            unknown_resource_audit = True
            termination = "resource-audit-time-limit"
            break
        if audit.status is ResourceAuditStatus.INFEASIBLE_PROVEN:
            if not selected:
                termination = "empty-selection-resource-infeasible"
                break
            # The resource transition model is not downward closed.  Store
            # the exact selected binary pattern; master backends encode +1 on
            # the pattern and -1 on all other fixed-pool columns.
            cut = tuple(sorted(selected))
            if cut in seen_cuts:
                raise RuntimeError("resource exact-pattern loop repeated an identical proven-infeasible set")
            seen_cuts.add(cut)
            no_goods.append(cut)
            termination = "resource-cut-added"
            continue

        coverage = sum(len(cols[j]["tids"]) for j in selected)
        energy = sum(float(cols[j]["E_plan_Wh"]) for j in selected)
        incumbent_energy = sum(float(cols[j]["E_plan_Wh"]) for j in incumbent_sel)
        if coverage > incumbent_cov or (coverage == incumbent_cov and energy < incumbent_energy - 1e-9):
            incumbent_cov = int(coverage)
            incumbent_sel = selected
            incumbent_audit = audit.assignment
        if master.optimal or coverage_ub - incumbent_cov == 0:
            coverage_ub = incumbent_cov
            coverage_proven = True
            termination = ("coverage-optimum-proven" if master.optimal
                           else "coverage-bound-closed")
            break
        if coverage_ub - incumbent_cov <= int(coverage_gap_target_abs):
            termination = "coverage-gap-target-reached"
            break
        termination = "coverage-master-stopped-with-incumbent"
        if deadline_reached(deadline):
            break

    # The incumbent plan has a well-defined plan-energy value even before the
    # coverage optimum is proved.  Only the *global energy gap* remains
    # undefined until phase one closes.
    energy_inc = float(sum(float(cols[j]["E_plan_Wh"]) for j in incumbent_sel))
    energy_lb = None
    energy_proven = False
    if coverage_proven:
        energy_lb = 0.0
        while not deadline_reached(deadline):
            master = solve_binary_master(
                solver, cols, all_tids, deck_rows, no_goods, phase="energy",
                deadline=deadline, capacity_rows=capacity_rows,
                pooled_energy_cap=pooled_energy_cap, coverage_equal=incumbent_cov)
            last_backend = master.backend
            restricted_gap_pct = master.restricted_pool_gap_pct
            lb = safe_energy_lower_bound(master)
            if lb is not None:
                energy_lb = max(float(energy_lb), min(float(energy_inc), float(lb)))
            if master.x is None:
                termination = ("energy-master-infeasible" if master.infeasible_proven
                               else "energy-master-time-limit")
                break
            selected = tuple(np.flatnonzero(master.x > 0.5).tolist())
            audit = audit_resource_assignment(
                cols, selected, K, B, B_use, resource_of, quick_min, swap_min,
                quick_capacity, swap_capacity, deadline=deadline)
            resource_nodes += int(audit.explored_nodes)
            if audit.status is ResourceAuditStatus.UNKNOWN_TIMEOUT:
                unknown_resource_audit = True
                termination = "resource-audit-time-limit"
                break
            if audit.status is ResourceAuditStatus.INFEASIBLE_PROVEN:
                # Exact-pattern exclusion, not an upward-closed subset cut.
                cut = tuple(sorted(selected))
                if cut in seen_cuts:
                    raise RuntimeError("resource exact-pattern loop repeated an identical proven-infeasible set")
                seen_cuts.add(cut)
                no_goods.append(cut)
                termination = "resource-cut-added-during-energy-phase"
                continue
            energy = float(sum(float(cols[j]["E_plan_Wh"]) for j in selected))
            if energy < float(energy_inc) - 1e-9:
                energy_inc = energy
                incumbent_sel = selected
                incumbent_audit = audit.assignment
            if master.optimal:
                energy_lb = float(energy_inc)
                energy_proven = True
                termination = "lexicographic-optimum-proven"
                break
            abs_gap = max(0.0, float(energy_inc) - float(energy_lb))
            rel_gap = abs_gap / max(abs(float(energy_inc)), 1e-12)
            if abs_gap <= ENERGY_TOL_WH:
                energy_proven = True
                termination = "energy-bound-closed"
                break
            if abs_gap <= float(energy_gap_target_abs_Wh) or rel_gap <= float(energy_gap_target_rel):
                termination = "energy-gap-target-reached"
                break
            termination = "energy-master-stopped-with-incumbent"
            if deadline_reached(deadline):
                break

    return dict(
        selected=list(incumbent_sel), audit=incumbent_audit,
        coverage_incumbent=int(incumbent_cov), coverage_upper_bound=int(coverage_ub),
        coverage_optimal=bool(coverage_proven),
        energy_incumbent_Wh=(None if energy_inc is None else float(energy_inc)),
        energy_lower_bound_Wh=(None if energy_lb is None else float(energy_lb)),
        energy_optimal=bool(energy_proven), no_goods=list(no_goods),
        resource_cut_type="exact-selected-pattern",
        resource_cut_superset_assumption=False,
        resource_audit_complete=not unknown_resource_audit,
        termination_reason=termination, backend=last_backend,
        restricted_pool_gap_pct=restricted_gap_pct, resource_nodes=int(resource_nodes))


def build_route_columns(turbines, opts, p, xi_amb, T_min, deck_delta_min, max_stops,
                        weather_unc, chance_mode, budget_gamma, kappa_mode, adaptive_wind_thr,
                        pool_h_mode: str = "pareto", diagnostics_sink=None,
                        deadline: float | None = None):
    """Build the shared physical route pool and retain an auditable rejection ledger.

    Every reachable singleton is inserted explicitly before heuristic multi-stop generation.  Thus a
    feasible singleton cannot disappear because of route-generation strategy.  Pareto compression is
    applied only within the same ordered visit sequence; coverage-equivalent but physically different
    orders are not collapsed before the resource model.
    """
    horizons = RM.decision_horizons_of(xi_amb)
    orig_kappa = RM.kappa
    cols = []
    ledger = diagnostics_sink if diagnostics_sink is not None else []
    stats = dict(launch_options=0, generated_routes=0, explicit_singletons=0,
                 evaluated_route_h_pairs=0, outside_window=0,
                 feasible_before_compression=0, removed_by_pareto=0, final_shared_pool=0)
    try:
        for opt in opts:
            if deadline_reached(deadline):
                stats["termination_reason"] = "time_limit"
                break
            stats["launch_options"] += 1
            wxq = opt.wx
            if chance_mode == "drcc":
                if kappa_mode == "adaptive":
                    mode = "cantelli" if float(wxq.get("wind10", 0.0)) >= adaptive_wind_thr else "vp_unimodal"
                    RM.kappa = RM.KAPPA_MODES[mode]
                elif kappa_mode == "nominal":
                    RM.kappa = lambda e: 0.0
                    mode = "nominal"
                else:
                    RM.kappa = RM.KAPPA_MODES[kappa_mode]
                    mode = kappa_mode
            else:
                mode = chance_mode

            pool = generate_routes(turbines, opt.ship, p, wxq, xi_amb, "min_h", max_stops, "full",
                                   weather_unc=weather_unc, launch_ships=None,
                                   chance_mode=chance_mode, budget_gamma=budget_gamma)
            route_by_order = {tuple(r.turbine_ids()): r for r, _ in pool}
            for tb in turbines:
                key = (str(tb.tid),)
                if key not in route_by_order:
                    route_by_order[key] = RM.Route(rid=-1, turbines=[tb], ship=opt.ship)
                    stats["explicit_singletons"] += 1
            routes = list(route_by_order.values())
            stats["generated_routes"] += len(routes)
            seen_here = {}  # ordered sequence -> non-dominated h/energy/SOC columns

            for r in routes:
                if deadline_reached(deadline):
                    stats["termination_reason"] = "time_limit"
                    break
                ordered_key = tuple(r.turbine_ids())
                coverage_key = tuple(sorted(ordered_key))
                feasible_h = []
                for h in horizons:
                    if deadline_reached(deadline):
                        stats["termination_reason"] = "time_limit"
                        break
                    if float(opt.tau_min) + float(h) > float(T_min):
                        stats["outside_window"] += 1
                        if isinstance(ledger, list):
                            ledger.append(dict(tau=float(opt.tau_min), route_order=";".join(map(str, ordered_key)),
                                               turbines=";".join(map(str, coverage_key)), h=float(h),
                                               feasible=False, primary_reason="outside_mission_window",
                                               failure_flags={"outside_mission_window": True}, margins={}))
                        continue
                    dd = RM.route_feasible_at_h(r, float(h), p, wxq, xi_amb,
                                                weather_unc=weather_unc, chance_mode=chance_mode,
                                                budget_gamma=budget_gamma)
                    stats["evaluated_route_h_pairs"] += 1
                    if isinstance(ledger, list):
                        ledger.append(dict(
                            tau=float(opt.tau_min), route_order=";".join(map(str, ordered_key)),
                            turbines=";".join(map(str, coverage_key)), h=float(h),
                            launch_state=str(opt.ship.c_state),
                            recovery_state=str(dd.get("recovery_state", "unknown")),
                            feasible=bool(dd.get("feasible", False)),
                            primary_reason=dd.get("primary_reason", dd.get("reason")),
                            failure_flags=dict(dd.get("failure_flags", {})),
                            margins=dict(dd.get("margins", {})),
                            weather_launch_time=dd.get("weather_launch_time"),
                            weather_recovery_time=dd.get("weather_recovery_time"),
                            E_plan_Wh=dd.get("E_plan_Wh"),
                            E_soc_required_Wh=dd.get("E_soc_required_Wh"),
                            landing_wind_upper_shift_ms=dd.get("landing_wind_upper_shift_ms"),
                            wind_speed_bias_ms=dd.get("wind_speed_bias_ms"),
                            wind_speed_std_ms=dd.get("wind_speed_std_ms"),
                            route_airspeed_diagnostics=dd.get("route_airspeed_diagnostics"),
                            time_contract=dd.get("time_contract", RM.time_contract_for(p)),
                            time_contract_id=dd.get("time_contract_id", RM.time_contract_for(p)),
                            wait_is_recourse=bool(dd.get("wait_is_recourse", True)),
                            dock_risk_contract=dd.get("dock_risk_contract", RM.DOCK_RISK_CONTRACT),
                            time_flight_nom_s=dd.get("time_flight_nom_s"),
                            time_inspection_s=dd.get("time_inspection_s"),
                            time_dock_nom_s=dd.get("time_dock_nom_s"),
                            time_core_nom_s=dd.get("time_core_nom_s"),
                            time_wait_nom_s=dd.get("time_wait_nom_s"),
                            time_xi_mean_shift_s=dd.get("time_xi_mean_shift_s"),
                            time_xi_std_term_s=dd.get("time_xi_std_term_s"),
                            time_weather_mean_shift_s=dd.get("time_weather_mean_shift_s"),
                            time_weather_std_term_s=dd.get("time_weather_std_term_s"),
                            time_geometry_remainder_s=dd.get("time_geometry_remainder_s"),
                            time_geometry_correction_s=dd.get("time_geometry_correction_s"),
                            time_xi_geo_total_s=dd.get("time_xi_geo_total_s"),
                            time_weather_total_s=dd.get("time_weather_total_s"),
                            xi_mean_along_m=dd.get("xi_mean_along_m"),
                            xi_mean_cross_m=dd.get("xi_mean_cross_m"),
                            xi_std_along_m=dd.get("xi_std_along_m"),
                            xi_std_cross_m=dd.get("xi_std_cross_m"),
                            xi_geo_bound_extra_m=dd.get("xi_geo_bound_extra_m"),
                            xi_mu_e_m=dd.get("xi_mu_e_m"), xi_mu_n_m=dd.get("xi_mu_n_m"),
                            xi_sigma_ee_m2=dd.get("xi_sigma_ee_m2"),
                            xi_sigma_en_m2=dd.get("xi_sigma_en_m2"),
                            xi_sigma_nn_m2=dd.get("xi_sigma_nn_m2"),
                            xi_launch_to_recovery_state_change_rate=dd.get(
                                "xi_launch_to_recovery_state_change_rate"),
                            xi_actual_recovery_state_mode=dd.get("xi_actual_recovery_state_mode"),
                            xi_launch_speed_p50_ms=dd.get("xi_launch_speed_p50_ms"),
                            xi_launch_speed_p95_ms=dd.get("xi_launch_speed_p95_ms"),
                            eps_time_total=dd.get("eps_time_total"),
                            eps_time_xi=dd.get("eps_time_xi"),
                            eps_time_weather=dd.get("eps_time_weather"),
                            eps_time_along=dd.get("eps_time_along"),
                            eps_time_cross=dd.get("eps_time_cross"),
                            geo_risk_allocation_mode=dd.get("geo_risk_allocation_mode"),
                            geo_risk_allocation_contract=dd.get(
                                "geo_risk_allocation_contract", RM.GEO_RISK_ALLOCATION_CONTRACT),
                            d_ret0_m=dd.get("d_ret0"),
                            time_drcc_tightening_s=dd.get("time_drcc_tightening_s"),
                            time_safe_core_s=dd.get("time_safe_core_s"),
                            time_wait_safe_s=dd.get("time_wait_safe_s"),
                            nominal_time_margin_s=dd.get("nominal_time_margin_s"),
                            time_drcc_margin_s=dd.get("time_drcc_margin_s"),
                            time_feasibility_basis=dd.get("time_feasibility_basis", "core_time"),
                            speed_is_recourse=bool(dd.get("speed_is_recourse", False)),
                            return_time_budget_s=dd.get("return_time_budget_s"),
                            return_time_budget_safe_s=dd.get("return_time_budget_safe_s"),
                            nonreturn_weather_mean_shift_s=dd.get("nonreturn_weather_mean_shift_s"),
                            nonreturn_weather_std_term_s=dd.get("nonreturn_weather_std_term_s"),
                            nonreturn_weather_reserve_s=dd.get("nonreturn_weather_reserve_s"),
                            eps_time_nonreturn_weather=dd.get("eps_time_nonreturn_weather"),
                            eps_time_return_required_airspeed=dd.get("eps_time_return_required_airspeed"),
                            return_required_airspeed_nom_ms=dd.get("return_required_airspeed_nom_ms"),
                            return_required_airspeed_safe_ms=dd.get("return_required_airspeed_safe_ms"),
                            return_airspeed_margin_ms=dd.get("return_airspeed_margin_ms"),
                            return_speed_recourse_contract=dd.get("return_speed_recourse_contract"),
                            return_power_envelope_W=dd.get("return_power_envelope_W"),
                            return_energy_envelope_Wh=dd.get("return_energy_envelope_Wh"),
                            energy_recourse_contract=dd.get("energy_recourse_contract"),
                            energy_coupled_wind_gradient_Wh_per_ms=dd.get("energy_coupled_wind_gradient_Wh_per_ms"),
                            time_decomposition=dd.get("time_decomposition"),
                            kappa_used=str(mode),
                            chance_mode=str(chance_mode),
                        ))
                    time_margin = float(dd.get("time_drcc_margin_s", dd.get("margin_T", -1.0e30)))
                    route_level_feasible = bool(dd.get("feasible", False) and time_margin >= 0.0)
                    if route_level_feasible:
                        stats["feasible_before_compression"] += 1
                        ep = float(dd.get("E_plan_Wh", dd["E0"] + dd.get("E_dock_Wh", 0.0)))
                        es = float(dd.get("E_soc_required_Wh", ep))
                        feasible_h.append((float(h), ep, es, float(dd.get("E0", ep)),
                                           dd.get("gate_weather_proof"), dd))
                        if pool_h_mode == "first":
                            break
                if not feasible_h:
                    continue
                ent = seen_here.setdefault(ordered_key, [])
                for h_ok, Eplan, Esoc, E_nom, gproof, dd in feasible_h:
                    cand = dict(tau=float(opt.tau_min), ship=opt.ship, wx=dict(wxq), route=r,
                                tids=coverage_key, route_order=ordered_key, h=h_ok, E0=Eplan,
                                E_plan_Wh=Eplan, E_soc_required_Wh=Esoc, E0_nominal=E_nom,
                                kappa_used=mode, gate_proof=gproof,
                                failure_flags=dict(dd.get("failure_flags", {})),
                                margins=dict(dd.get("margins", {})),
                                time_contract=dd.get("time_contract", RM.time_contract_for(p)),
                                time_contract_id=dd.get("time_contract_id", RM.time_contract_for(p)),
                                wait_is_recourse=bool(dd.get("wait_is_recourse", True)),
                                speed_is_recourse=bool(dd.get("speed_is_recourse", False)),
                                time_feasibility_basis=dd.get("time_feasibility_basis", "core_time"),
                                return_airspeed_margin_ms=float(dd.get("return_airspeed_margin_ms", float("nan"))),
                                return_speed_recourse_contract=dd.get("return_speed_recourse_contract"),
                                dock_risk_contract=dd.get("dock_risk_contract", RM.DOCK_RISK_CONTRACT),
                                time_core_nom_s=float(dd.get("time_core_nom_s", float("nan"))),
                                time_drcc_tightening_s=float(dd.get("time_drcc_tightening_s", float("nan"))),
                                time_drcc_margin_s=float(dd.get("time_drcc_margin_s", float("nan"))),
                                launch_slot=int(round(opt.tau_min / deck_delta_min)),
                                rec_slot=int(round((opt.tau_min + h_ok) / deck_delta_min)))

                    def _dom(a, hh, ep, es):
                        return (float(a["h"]) <= hh
                                and _column_plan_energy(a) <= ep
                                and _column_soc_energy(a) <= es)
                    if any(_dom(e, h_ok, Eplan, Esoc) for e in ent):
                        stats["removed_by_pareto"] += 1
                        continue
                    kept = []
                    for e in ent:
                        dominated = (h_ok <= float(e["h"])
                                     and Eplan <= _column_plan_energy(e)
                                     and Esoc <= _column_soc_energy(e))
                        if dominated:
                            stats["removed_by_pareto"] += 1
                        else:
                            kept.append(e)
                    ent[:] = kept
                    ent.append(cand)
            for ent in seen_here.values():
                cols.extend(sorted(ent, key=lambda e: e["h"]))
    finally:
        RM.kappa = orig_kappa
    stats["final_shared_pool"] = len(cols)
    stats.setdefault("termination_reason", "complete")
    # This routine is a seed/primal heuristic: ``generate_routes`` does not prove
    # exhaustion of the ordered implicit route space.  Completion here means only
    # that the requested heuristic seed construction finished before the deadline.
    stats["seed_generation_complete"] = stats["termination_reason"] == "complete"
    stats["route_space_complete"] = False
    stats["certificate_role"] = "initial-column-heuristic-only"
    build_route_columns.last_diagnostics = list(ledger) if isinstance(ledger, list) else []
    build_route_columns.last_stage_counts = dict(stats)
    return cols


def solve_resource_master(turbines, launch_opts, p, xi_amb, K, T_min,
                          deck_delta_min=2.5, t_swap_min=4.0, max_stops=4,
                          weather_unc=None, chance_mode="drcc",
                          budget_gamma=RM.BUDGET_GAMMA_DEFAULT,
                          kappa_mode="vp_unimodal", adaptive_wind_thr=8.0,
                          ships_override=None, batteries=None, cols_override=None,
                          solver="auto", verbose=False, deck_mode="interval",
                          t_launch_min=None, pool_h_mode: str = "pareto",
                          landing_clear_min=None, quick_inspection_capacity=None,
                          swap_station_capacity=None, battery_reuse_mode=None,
                          allow_resource_only_columns: bool = False,
                          time_limit_s: float | None = None,
                          deadline: float | None = None,
                          coverage_gap_target_abs: int = 0,
                          energy_gap_target_rel: float = 0.0,
                          energy_gap_target_abs_Wh: float = 1e-6):
    """Solve the exact two-stage resource master on a finite route pool.

    The master is a set-packing model: each turbine has one row
    ``sum(a_ir*x_r) <= 1``.  Coverage is therefore exactly
    ``sum(|S_r|*x_r)`` and no independent coverage variables are used.
    Every integer candidate is checked by the exact tri-state resource DFS.
    Only ``INFEASIBLE_PROVEN`` authorizes an exact resource-pattern exclusion cut.
    """
    started = time.monotonic()
    if deadline is None and time_limit_s is not None:
        deadline = started + max(float(time_limit_s), 0.0)
    elif deadline is not None and time_limit_s is None:
        time_limit_s = max(0.0, float(deadline) - started)

    override_rejections = []
    input_turbine_ids = {getattr(t, "tid", t) for t in turbines}
    if cols_override is not None:
        raw_cols = [dict(c) for c in cols_override]
        for _c in raw_cols:
            input_turbine_ids.update(route_tids(_c))
        cols = []
        original_kappa = RM.kappa
        try:
            for idx, c in enumerate(raw_cols):
                if deadline_reached(deadline):
                    override_rejections.append(dict(index=idx, reason="global-time-limit"))
                    break
                if allow_resource_only_columns and bool(c.get("resource_only_test_column", False)):
                    try:
                        cols.append(validate_route_columns([c])[0])
                    except Exception as exc:
                        override_rejections.append(dict(index=idx, reason="invalid-resource-test-column",
                                                        error=str(exc)))
                    continue
                if "route" not in c or "h" not in c:
                    override_rejections.append(dict(index=idx, reason="missing-route-or-h"))
                    continue
                try:
                    wx_c = c.get("wx", {})
                    if chance_mode == "drcc":
                        if kappa_mode == "adaptive":
                            mode_used = ("cantelli" if float(wx_c.get("wind10", 0.0)) >= adaptive_wind_thr
                                         else "vp_unimodal")
                            RM.kappa = RM.KAPPA_MODES[mode_used]
                        elif kappa_mode == "nominal":
                            mode_used = "nominal"
                            RM.kappa = lambda e: 0.0
                        elif kappa_mode in RM.KAPPA_MODES:
                            mode_used = kappa_mode
                            RM.kappa = RM.KAPPA_MODES[kappa_mode]
                        else:
                            raise ValueError(f"unknown kappa_mode={kappa_mode!r}")
                    else:
                        mode_used = chance_mode
                    diag = RM.route_feasible_at_h(
                        c["route"], float(c["h"]), p, wx_c, xi_amb,
                        weather_unc=weather_unc, chance_mode=chance_mode,
                        budget_gamma=budget_gamma)
                    if not diag.get("feasible", False):
                        override_rejections.append(dict(
                            index=idx, reason=str(diag.get("reason", "physical-infeasible"))))
                        continue
                    c.update(
                        E0_nominal=float(diag["E0"]),
                        E_plan_Wh=float(diag["E_plan_Wh"]),
                        E_soc_required_Wh=float(diag["E_soc_required_Wh"]),
                        E_flight_Wh=float(diag.get("E_flight_Wh", diag["E0"])),
                        E_escort_Wh=float(diag.get("E_escort_Wh", 0.0)),
                        E_dock_Wh=float(diag.get("E_dock_Wh", 0.0)),
                        recovery_state=str(diag.get("recovery_state", "unknown")),
                        recovery_state_source=str(diag.get("recovery_state_source", "unknown")),
                        kappa_used=str(mode_used), physics_revalidated=True)
                    c["E0"] = c["E_plan_Wh"]
                    cols.append(validate_route_columns([c])[0])
                except Exception as exc:
                    override_rejections.append(dict(index=idx, reason="revalidation-exception",
                                                    error=f"{type(exc).__name__}: {exc}"))
        finally:
            RM.kappa = original_kappa
        pool_stage_counts = {"final_shared_pool": int(len(cols)),
                             "source": "validated-columns-override"}
        pool_rejection_ledger = []
    else:
        if deadline_reached(deadline):
            cols = []
            pool_stage_counts = {"final_shared_pool": 0, "source": "time-limit-before-route-build"}
            pool_rejection_ledger = []
        else:
            cols = build_route_columns(
                turbines, (ships_override if ships_override is not None else launch_opts),
                p, xi_amb, T_min, deck_delta_min, max_stops, weather_unc,
                chance_mode, budget_gamma, kappa_mode, adaptive_wind_thr,
                pool_h_mode=pool_h_mode, deadline=deadline)
            pool_stage_counts = dict(getattr(build_route_columns, "last_stage_counts", {}))
            pool_rejection_ledger = list(getattr(build_route_columns, "last_diagnostics", []))

    clean_cols = []
    for idx, c in enumerate(cols):
        try:
            tau, h = float(c["tau"]), float(c["h"])
            if tau < -1e-9 or h < -1e-9 or tau + h > float(T_min) + 1e-9:
                raise ValueError("route timing outside planning window")
            clean_cols.append(validate_route_columns([c])[0])
        except Exception as exc:
            override_rejections.append(dict(index=idx, reason="invalid-column-contract", error=str(exc)))
    cols = clean_cols

    B = int(batteries) if batteries is not None else 2 * int(K)
    t_launch = float(t_launch_min) if t_launch_min is not None else float(deck_delta_min)
    t_clear = (float(landing_clear_min) if landing_clear_min is not None
               else float(getattr(p, "landing_clear_min", 1.0)))
    t_quick = float(getattr(p, "quick_inspection_min", 1.0))
    C_quick = (int(quick_inspection_capacity) if quick_inspection_capacity is not None
               else int(getattr(p, "quick_inspection_capacity", 1)))
    C_swap = (int(swap_station_capacity) if swap_station_capacity is not None
              else int(getattr(p, "swap_station_capacity", 1)))
    battery_mode = str(battery_reuse_mode or getattr(p, "battery_reuse_mode", "exact_soc"))
    binding_mode = str(getattr(p, "battery_binding_mode", "horizon_fixed_uav"))
    if battery_mode not in ("exact_soc", "legacy_count"):
        raise ValueError("battery_reuse_mode must be 'exact_soc' or 'legacy_count'")
    if binding_mode != "horizon_fixed_uav":
        raise ValueError("battery_binding_mode currently supports only 'horizon_fixed_uav'")
    if int(K) <= 0 or B < 0 or C_quick <= 0 or C_swap <= 0:
        raise ValueError("UAV/check/swap capacities must be positive and batteries nonnegative")
    if min(t_launch, t_clear, t_quick, float(t_swap_min)) < 0:
        raise ValueError("launch/clear/check/swap durations must be nonnegative")

    all_tids = sorted(input_turbine_ids | {tid for c in cols for tid in c["tids"]}, key=str)
    resource_of = [_resource_intervals(c, t_launch, t_clear, t_swap_min,
                                       deck_mode, deck_delta_min) for c in cols]
    deck_rows = _multi_interval_conflict_rows([r["deck"] for r in resource_of])
    fastest_service = (min(max(t_quick, 0.0), max(float(t_swap_min), 0.0))
                       if battery_mode == "exact_soc" and B >= 2
                       else max(t_quick, 0.0))
    fastest_turn_intervals = [
        (float(r["launch_start_min"]), float(r["clear_end_min"]) + fastest_service)
        for r in resource_of]
    capacity_rows = [(row, int(K)) for row in _interval_capacity_rows(fastest_turn_intervals)]
    active_cap = min(int(K), int(B))
    if active_cap < int(K):
        capacity_rows += [(row, active_cap)
                          for row in _interval_capacity_rows([r["active"] for r in resource_of])]

    audit_cols = cols
    audit_quick = t_quick
    if battery_mode == "legacy_count":
        audit_cols = [dict(c, E_soc_required_Wh=float(p.B_use)) for c in cols]
        audit_quick = float(t_swap_min)

    result = _solve_logic_benders_anytime(
        audit_cols, all_tids, int(K), B, float(p.B_use), resource_of,
        deck_rows, audit_quick, float(t_swap_min), C_quick, C_swap,
        solver=solver, deadline=deadline, capacity_rows=capacity_rows,
        pooled_energy_cap=(float(B) * float(p.B_use) if battery_mode == "exact_soc" else None),
        coverage_gap_target_abs=coverage_gap_target_abs,
        energy_gap_target_rel=energy_gap_target_rel,
        energy_gap_target_abs_Wh=energy_gap_target_abs_Wh)

    selected = [int(j) for j in result.get("selected", [])]
    audit = result.get("audit")
    if audit is None:
        audit = dict(uav_assignment={}, battery_assignment={}, mission_service={},
                     battery_energy_used_Wh=[0.0] * B, swap_events=[],
                     quick_inspection_events=[], uav_chains=[[] for _ in range(int(K))],
                     battery_binding=[None] * B, resource_audit="not-completed")
    if battery_mode == "exact_soc" and result.get("resource_audit_complete", False):
        verify = audit_resource_assignment(
            cols, selected, int(K), B, float(p.B_use), resource_of,
            t_quick, float(t_swap_min), C_quick, C_swap, deadline=deadline)
        if verify.status is ResourceAuditStatus.FEASIBLE:
            audit = verify.assignment
        elif verify.status is ResourceAuditStatus.INFEASIBLE_PROVEN:
            raise RuntimeError("selected plan failed independent exact resource re-audit")
        else:
            result["resource_audit_complete"] = False

    successor = {}
    for j, service in audit.get("mission_service", {}).items():
        previous = service.get("predecessor")
        if previous is not None:
            successor[int(previous)] = int(j)
    chosen = []
    for j in selected:
        c = dict(cols[j])
        before = dict(audit["mission_service"][j])
        next_j = successor.get(j)
        if next_j is None:
            post_mode, post_interval = "none_after_last_mission", None
        else:
            next_service = audit["mission_service"][next_j]
            post_mode, post_interval = str(next_service["mode"]), next_service.get("interval")
        c.update(
            uav_id=int(audit["uav_assignment"][j]),
            battery_group=int(audit["battery_assignment"][j]),
            turnaround_before=before, successor_column=next_j,
            post_service_mode=post_mode, post_service_interval=post_interval,
            resource_intervals=dict(
                resource_of[j],
                quick_inspection=(tuple(post_interval)
                                  if post_mode == "quick_reuse" and post_interval else None),
                swap=(tuple(post_interval)
                      if post_mode == "battery_swap" and post_interval else None)))
        chosen.append(c)

    covered_ids = selected_turbines(cols, selected)
    duplicates = duplicate_turbines(cols, selected)
    if duplicates or len(covered_ids) != len(set(covered_ids)):
        raise RuntimeError("internal set-packing violation: duplicate turbine coverage")
    plan_energy = float(sum(float(c["E_plan_Wh"]) for c in chosen))
    reserved_energy = float(sum(float(c["E_soc_required_Wh"]) for c in chosen))
    makespan = max((float(resource_of[j]["clear_end_min"]) for j in selected), default=0.0)
    used = [float(v) for v in audit.get("battery_energy_used_Wh", [0.0] * B)]
    end_soc = [100.0 * (float(p.B_k) - used_energy) / float(p.B_k) for used_energy in used]
    swap_events = []
    quick_events = []
    for j, service in audit.get("mission_service", {}).items():
        if service.get("predecessor") is None or service.get("interval") is None:
            continue
        event = dict(predecessor_column=int(service["predecessor"]), successor_column=int(j),
                     uav_id=int(audit["uav_assignment"][j]),
                     start_min=float(service["interval"][0]), end_min=float(service["interval"][1]))
        if service["mode"] == "battery_swap":
            event["from_battery"] = int(audit["battery_assignment"][int(service["predecessor"])])
            event["to_battery"] = int(audit["battery_assignment"][j])
            swap_events.append(event)
        elif service["mode"] == "quick_reuse":
            event["battery_group"] = int(audit["battery_assignment"][j])
            quick_events.append(event)

    coverage_inc = int(result.get("coverage_incumbent", len(covered_ids)))
    coverage_ub = max(coverage_inc, int(result.get("coverage_upper_bound", len(all_tids))))
    coverage_gap_abs = coverage_ub - coverage_inc
    coverage_gap_pct = 100.0 * coverage_gap_abs / max(1, coverage_ub)
    energy_inc = result.get("energy_incumbent_Wh")
    energy_lb = result.get("energy_lower_bound_Wh")
    energy_gap_abs = None
    energy_gap_pct = None
    if bool(result.get("coverage_optimal")) and energy_inc is not None and energy_lb is not None:
        energy_gap_abs = max(0.0, float(energy_inc) - float(energy_lb))
        energy_gap_pct = 100.0 * energy_gap_abs / max(abs(float(energy_inc)), 1e-12)
    conditional_energy_gap_pct = None
    if energy_inc is not None and energy_lb is not None:
        conditional_energy_gap_pct = 100.0 * max(0.0, float(energy_inc) - float(energy_lb)) / max(abs(float(energy_inc)), 1e-12)

    runtime = time.monotonic() - started
    restricted_pool_optimal = bool(result.get("coverage_optimal") and result.get("energy_optimal"))
    return dict(
        status=("restricted_pool_optimal" if restricted_pool_optimal
                else "time-limit-or-gap-stop"),
        termination_reason=str(result.get("termination_reason")),
        runtime_s=float(runtime), time_limit_s=time_limit_s,
        coverage_incumbent=coverage_inc,
        coverage_upper_bound=coverage_ub,
        coverage_gap_abs=coverage_gap_abs,
        coverage_gap_pct=coverage_gap_pct,
        coverage_optimal=bool(result.get("coverage_optimal")),
        energy_incumbent_Wh=(None if energy_inc is None else float(energy_inc)),
        energy_lower_bound_Wh=(None if energy_lb is None else float(energy_lb)),
        energy_gap_abs_Wh=energy_gap_abs,
        energy_gap_pct=energy_gap_pct,
        global_energy_gap_pct=None,
        global_energy_gap_reason="route-space scope is assigned by solve_fleet_anytime",
        conditional_energy_gap_pct=conditional_energy_gap_pct,
        energy_optimal=bool(result.get("energy_optimal")),
        restricted_pool_lexicographic_optimal=restricted_pool_optimal,
        global_lexicographic_optimal=False,
        # Legacy field now follows the global meaning; callers that need the
        # restricted-pool result must use the explicit scoped field above.
        lexicographic_optimal=False,
        route_space_complete=False, pricing_complete=False,
        resource_audit_complete=bool(result.get("resource_audit_complete")),
        bound_scope="validated_route_pool",
        bound_source="restricted-master-mip-bound",
        restricted_pool_gap_pct=result.get("restricted_pool_gap_pct"),
        chosen=chosen, covered_turbine_ids=covered_ids,
        duplicate_turbine_visits=duplicates,
        covered=coverage_inc, coverable=len(all_tids), flights=len(chosen),
        energy_Wh=round(plan_energy, 6), battery_reserved_Wh=round(reserved_energy, 6),
        makespan_min=round(makespan, 6), solver=str(result.get("backend")),
        pool_size=len(cols), route_pool_count=len(cols),
        route_pool_status=("nonempty" if cols else "empty"),
        route_pool_stage_counts=pool_stage_counts,
        route_pool_rejection_count=len(pool_rejection_ledger),
        override_columns_input=(len(cols_override) if cols_override is not None else None),
        override_columns_rejected=len(override_rejections),
        override_rejection_details=override_rejections,
        mean_stops=(float(np.mean([len(c["tids"]) for c in chosen])) if chosen else 0.0),
        multi_stop_ratio=(float(np.mean([len(c["tids"]) >= 2 for c in chosen])) if chosen else 0.0),
        uav_assignment={int(j): int(k) for j, k in audit.get("uav_assignment", {}).items()},
        battery_assignment={int(j): int(b) for j, b in audit.get("battery_assignment", {}).items()},
        uav_chains=[[int(j) for j in chain] for chain in audit.get("uav_chains", [])],
        battery_binding=list(audit.get("battery_binding", [])),
        mission_service={int(j): dict(v) for j, v in audit.get("mission_service", {}).items()},
        swap_events=swap_events, quick_inspection_events=quick_events,
        n_swaps=len(swap_events), n_quick_reuses=len(quick_events),
        battery_energy_used_Wh=[round(v, 6) for v in used],
        battery_end_soc_pct=[round(v, 6) for v in end_soc],
        resource_cuts=len(result.get("no_goods", [])),
        resource_pattern_cuts=len(result.get("no_goods", [])),
        resource_cut_type="exact-selected-pattern",
        resource_cut_superset_assumption=False,
        resource_search_nodes=int(result.get("resource_nodes", 0)),
        restricted_master_certificate=("optimal" if result.get("coverage_optimal") and result.get("energy_optimal")
                                       else "not-proven"),
        K=int(K), batteries=B, T_min=float(T_min), deck_mode=deck_mode,
        t_launch_min=t_launch, landing_clear_min=t_clear,
        quick_inspection_min=t_quick, quick_inspection_capacity=C_quick,
        t_swap_min=float(t_swap_min), swap_station_capacity=C_swap,
        battery_reuse_mode=battery_mode, battery_binding_mode=binding_mode)


if __name__ == "__main__" and False:
    pass


# =============================================================================
# Shared set-packing master and exact entity-resource primitives
# Kept in step11 by the formal project layout contract.
# =============================================================================

# Solver/result-validation tolerances used only for external MILP/legacy-gap
# interpretation.  They are NOT part of the formal entity-resource feasible set:
# ``exact_resource_audit`` uses strict half-open time comparisons and exact
# rational sums of the binary64 SOC energies.
INT_TOL = 1e-7
ENERGY_TOL_WH = 1e-6


class ResourceAuditStatus(str, Enum):
    FEASIBLE = "FEASIBLE"
    INFEASIBLE_PROVEN = "INFEASIBLE_PROVEN"
    UNKNOWN_TIMEOUT = "UNKNOWN_TIMEOUT"


@dataclass(frozen=True)
class ResourceAuditResult:
    status: ResourceAuditStatus
    assignment: dict[str, Any] | None = None
    explored_nodes: int = 0
    memo_hits: int = 0
    # V9 diagnostic-only prune counters.  They never participate in feasibility,
    # branching, cuts, or certificates; absent callers see the historical default.
    failure_reasons: dict[str, int] | None = None


@dataclass(frozen=True)
class MasterResult:
    status: str
    x: np.ndarray | None
    objective_value: float | None
    objective_bound: float | None
    optimal: bool
    infeasible_proven: bool
    backend: str
    restricted_pool_gap_pct: float | None


def remaining_time(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return max(0.0, float(deadline) - time.monotonic())


def deadline_reached(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= float(deadline)


def route_tids(column: Mapping[str, Any]) -> tuple[Any, ...]:
    tids = tuple(column.get("tids", ()))
    if not tids and "route" in column and hasattr(column["route"], "turbine_ids"):
        tids = tuple(column["route"].turbine_ids())
    return tids


def validate_route_columns(columns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate the finite route-column contract without changing physics values."""
    out: list[dict[str, Any]] = []
    for j, raw in enumerate(columns):
        c = dict(raw)
        tids = route_tids(c)
        if not tids:
            raise ValueError(f"route column {j} covers no turbine")
        if len(tids) != len(set(tids)):
            raise ValueError(f"route column {j} repeats a turbine: {tids!r}")
        plan = float(c.get("E_plan_Wh", c.get("E0", float("nan"))))
        soc = float(c.get("E_soc_required_Wh", plan))
        if not math.isfinite(plan) or plan < 0.0:
            raise ValueError(f"route column {j} has invalid planned energy")
        if not math.isfinite(soc) or soc < plan:
            raise ValueError(f"route column {j} has SOC reserve below planned energy")
        c["tids"] = tids
        c["E_plan_Wh"] = plan
        c["E_soc_required_Wh"] = soc
        c["E0"] = plan
        out.append(c)
    return out


def selected_turbines(columns: Sequence[Mapping[str, Any]], selected: Iterable[int]) -> list[Any]:
    tids: list[Any] = []
    for j in selected:
        tids.extend(route_tids(columns[int(j)]))
    return tids


def duplicate_turbines(columns: Sequence[Mapping[str, Any]], selected: Iterable[int]) -> list[Any]:
    seen: set[Any] = set()
    dup: set[Any] = set()
    for tid in selected_turbines(columns, selected):
        if tid in seen:
            dup.add(tid)
        seen.add(tid)
    return sorted(dup, key=str)


def validate_selected_solution(
    x: Sequence[float],
    columns: Sequence[Mapping[str, Any]],
    turbine_ids: Sequence[Any],
    deck_rows: Sequence[Sequence[int]],
    no_good_cuts: Sequence[Sequence[int]],
    *,
    capacity_rows: Sequence[tuple[Sequence[int], int]] = (),
    pooled_energy_cap: float | None = None,
    coverage_equal: int | None = None,
    coverage_min: int | None = None,
    energy_upper_bound_Wh: float | None = None,
    full_cover_equal: bool = False,
    full_cover_no_good_cuts: Sequence[Sequence[int]] = (),
    tol: float = INT_TOL,
) -> np.ndarray | None:
    """Independent validator for a binary set-packing master solution."""
    xv = np.asarray(x, dtype=float).reshape(-1)
    n = len(columns)
    if xv.shape != (n,) or not np.all(np.isfinite(xv)):
        return None
    if np.any(xv < -tol) or np.any(xv > 1.0 + tol):
        return None
    if np.any(np.abs(xv - np.rint(xv)) > tol):
        return None
    xi = np.rint(xv).astype(int)

    for tid in turbine_ids:
        served = sum(int(xi[j]) for j, c in enumerate(columns) if tid in route_tids(c))
        if full_cover_equal:
            if served != 1:
                return None
        elif served > 1:
            return None
    if duplicate_turbines(columns, np.flatnonzero(xi)):
        return None
    if any(sum(int(xi[j]) for j in row) > 1 for row in deck_rows):
        return None
    for row, cap in capacity_rows:
        if sum(int(xi[j]) for j in row) > int(cap):
            return None
    if pooled_energy_cap is not None:
        used = sum(float(columns[j]["E_soc_required_Wh"]) * int(xi[j]) for j in range(n))
        if used > float(pooled_energy_cap) + tol:
            return None
    for cut in no_good_cuts:
        cut_set = {int(j) for j in cut}
        # Exact binary-pattern exclusion:
        #   sum_{j in S} x_j - sum_{j not in S} x_j <= |S|-1.
        # For binary x, the only violating vector is the exact set S.  This is
        # required because the general turnaround DFS is not downward closed.
        lhs = sum(int(xi[j]) if j in cut_set else -int(xi[j]) for j in range(n))
        if lhs > len(cut_set) - 1:
            return None
    for cut in full_cover_no_good_cuts:
        cut_set = {int(j) for j in cut}
        # Full-cover target strengthening.  When every turbine must be served
        # exactly once and every route is nonempty, a target-feasible integer
        # plan that contains all routes in S cannot contain any additional
        # route.  Hence sum_{j in S} x_j <= |S|-1 excludes the same audited
        # full-cover integer pattern but is a much stronger valid inequality.
        if sum(int(xi[j]) for j in cut_set) > len(cut_set) - 1:
            return None

    coverage = sum(len(route_tids(columns[j])) * int(xi[j]) for j in range(n))
    if coverage_equal is not None and coverage != int(coverage_equal):
        return None
    if coverage_min is not None and coverage < int(coverage_min):
        return None
    if energy_upper_bound_Wh is not None:
        energy = sum(float(columns[j]["E_plan_Wh"]) * int(xi[j]) for j in range(n))
        if energy > float(energy_upper_bound_Wh) + tol:
            return None
    return xi


def _linear_rows(
    columns: Sequence[Mapping[str, Any]],
    turbine_ids: Sequence[Any],
    deck_rows: Sequence[Sequence[int]],
    no_good_cuts: Sequence[Sequence[int]],
    *,
    capacity_rows: Sequence[tuple[Sequence[int], int]],
    pooled_energy_cap: float | None,
    coverage_equal: int | None,
    coverage_min: int | None,
    energy_upper_bound_Wh: float | None,
    full_cover_equal: bool = False,
    full_cover_no_good_cuts: Sequence[Sequence[int]] = (),
):
    rows: list[dict[int, float]] = []
    lbs: list[float] = []
    ubs: list[float] = []
    n = len(columns)
    for tid in turbine_ids:
        row = {j: 1.0 for j, c in enumerate(columns) if tid in route_tids(c)}
        if row:
            rows.append(row)
            if full_cover_equal:
                lbs.append(1.0); ubs.append(1.0)
            else:
                lbs.append(-np.inf); ubs.append(1.0)
        elif full_cover_equal:
            # Explicit impossible row 0 == 1.  This lets the backend prove
            # target infeasibility without relying on a separate precheck.
            rows.append({}); lbs.append(1.0); ubs.append(1.0)
    for rr in deck_rows:
        rows.append({int(j): 1.0 for j in rr}); lbs.append(-np.inf); ubs.append(1.0)
    for rr, cap in capacity_rows:
        rows.append({int(j): 1.0 for j in rr}); lbs.append(-np.inf); ubs.append(float(cap))
    if pooled_energy_cap is not None:
        rows.append({j: float(columns[j]["E_soc_required_Wh"]) for j in range(n)})
        lbs.append(-np.inf); ubs.append(float(pooled_energy_cap))
    for cut in no_good_cuts:
        cut_set = {int(j) for j in cut}
        rows.append({j: (1.0 if j in cut_set else -1.0) for j in range(n)})
        lbs.append(-np.inf); ubs.append(float(len(cut_set) - 1))
    for cut in full_cover_no_good_cuts:
        cut_set = {int(j) for j in cut}
        rows.append({j: 1.0 for j in cut_set})
        lbs.append(-np.inf); ubs.append(float(len(cut_set) - 1))
    cov_row = {j: float(len(route_tids(c))) for j, c in enumerate(columns)}
    if coverage_equal is not None:
        rows.append(cov_row); lbs.append(float(coverage_equal)); ubs.append(float(coverage_equal))
    elif coverage_min is not None:
        rows.append(cov_row); lbs.append(float(coverage_min)); ubs.append(np.inf)
    if energy_upper_bound_Wh is not None:
        rows.append({j: float(c["E_plan_Wh"]) for j, c in enumerate(columns)})
        lbs.append(-np.inf); ubs.append(float(energy_upper_bound_Wh))
    return rows, np.asarray(lbs, float), np.asarray(ubs, float)


def solve_binary_master_scipy(
    columns: Sequence[Mapping[str, Any]],
    turbine_ids: Sequence[Any],
    deck_rows: Sequence[Sequence[int]],
    no_good_cuts: Sequence[Sequence[int]],
    *,
    phase: str,
    deadline: float | None,
    capacity_rows: Sequence[tuple[Sequence[int], int]] = (),
    pooled_energy_cap: float | None = None,
    coverage_equal: int | None = None,
    coverage_min: int | None = None,
    energy_upper_bound_Wh: float | None = None,
    full_cover_equal: bool = False,
    full_cover_no_good_cuts: Sequence[Sequence[int]] = (),
) -> MasterResult:
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        import scipy.sparse as sp
    except Exception:
        return MasterResult("backend-unavailable", None, None, None, False, False,
                            "scipy-milp", None)
    n = len(columns)
    if n == 0:
        feasible = coverage_equal in (None, 0) and (coverage_min is None or coverage_min <= 0)
        return MasterResult("optimal" if feasible else "infeasible", np.zeros(0) if feasible else None,
                            0.0 if feasible else None, 0.0 if feasible else None,
                            feasible, not feasible, "scipy-milp", 0.0 if feasible else None)
    rem = remaining_time(deadline)
    if rem is not None and rem <= 0.0:
        return MasterResult("time-limit", None, None, None, False, False, "scipy-milp", None)
    rows, lbs, ubs = _linear_rows(
        columns, turbine_ids, deck_rows, no_good_cuts,
        capacity_rows=capacity_rows, pooled_energy_cap=pooled_energy_cap,
        coverage_equal=coverage_equal, coverage_min=coverage_min,
        energy_upper_bound_Wh=energy_upper_bound_Wh,
        full_cover_equal=bool(full_cover_equal),
        full_cover_no_good_cuts=full_cover_no_good_cuts)
    A = sp.lil_matrix((len(rows), n), dtype=float)
    for r, row in enumerate(rows):
        for j, value in row.items():
            A[r, j] = value
    constraints = [] if not rows else [LinearConstraint(A.tocsr(), lbs, ubs)]
    if phase == "coverage":
        objective = -np.asarray([len(route_tids(c)) for c in columns], dtype=float)
    elif phase == "energy":
        objective = np.asarray([float(c["E_plan_Wh"]) for c in columns], dtype=float)
    else:
        raise ValueError("phase must be 'coverage' or 'energy'")
    options: dict[str, Any] = {"mip_rel_gap": 0.0, "presolve": True}
    if rem is not None:
        options["time_limit"] = max(float(rem), 1e-9)
    try:
        res = milp(c=objective, constraints=constraints, integrality=np.ones(n),
                   bounds=Bounds(np.zeros(n), np.ones(n)), options=options)
    except Exception as exc:
        return MasterResult(f"solver-error:{type(exc).__name__}", None, None, None,
                            False, False, "scipy-milp", None)

    status_code = int(getattr(res, "status", -1))
    optimal = status_code == 0 and bool(getattr(res, "success", False))
    infeasible = status_code == 2
    raw_x = getattr(res, "x", None)
    checked = None
    if raw_x is not None:
        checked = validate_selected_solution(
            raw_x, columns, turbine_ids, deck_rows, no_good_cuts,
            capacity_rows=capacity_rows, pooled_energy_cap=pooled_energy_cap,
            coverage_equal=coverage_equal, coverage_min=coverage_min,
            energy_upper_bound_Wh=energy_upper_bound_Wh,
            full_cover_equal=bool(full_cover_equal),
            full_cover_no_good_cuts=full_cover_no_good_cuts)
    objective_value = None
    if checked is not None:
        objective_value = float(np.dot(objective, checked))
    dual_bound = getattr(res, "mip_dual_bound", None)
    objective_bound = None
    if dual_bound is not None:
        try:
            value = float(dual_bound)
            if math.isfinite(value):
                objective_bound = value
        except Exception:
            pass
    gap = getattr(res, "mip_gap", None)
    gap_pct = None
    if gap is not None:
        try:
            value = float(gap)
            if math.isfinite(value):
                gap_pct = max(0.0, 100.0 * value)
        except Exception:
            pass
    if optimal:
        status = "optimal"
    elif infeasible:
        status = "infeasible"
    elif status_code == 1:
        status = "time-or-iteration-limit"
    else:
        status = str(getattr(res, "message", "solver-stopped"))
    return MasterResult(status, checked, objective_value, objective_bound,
                        optimal, infeasible, "scipy-milp", gap_pct)


def solve_binary_master_gurobi(
    columns: Sequence[Mapping[str, Any]],
    turbine_ids: Sequence[Any],
    deck_rows: Sequence[Sequence[int]],
    no_good_cuts: Sequence[Sequence[int]],
    *,
    phase: str,
    deadline: float | None,
    capacity_rows: Sequence[tuple[Sequence[int], int]] = (),
    pooled_energy_cap: float | None = None,
    coverage_equal: int | None = None,
    coverage_min: int | None = None,
    energy_upper_bound_Wh: float | None = None,
    full_cover_equal: bool = False,
    full_cover_no_good_cuts: Sequence[Sequence[int]] = (),
) -> MasterResult:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return MasterResult("backend-unavailable", None, None, None, False, False,
                            "gurobi", None)
    n = len(columns)
    rem = remaining_time(deadline)
    if rem is not None and rem <= 0.0:
        return MasterResult("time-limit", None, None, None, False, False, "gurobi", None)
    model = gp.Model("resource_assignment_master")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 0.0
    if rem is not None:
        model.Params.TimeLimit = max(float(rem), 1e-9)
    x = model.addVars(n, vtype=GRB.BINARY, name="route")
    for tid in turbine_ids:
        expr = gp.quicksum(x[j] for j, c in enumerate(columns)
                           if tid in route_tids(c))
        model.addConstr(expr == 1 if full_cover_equal else expr <= 1)
    for row in deck_rows:
        model.addConstr(gp.quicksum(x[j] for j in row) <= 1)
    for row, cap in capacity_rows:
        model.addConstr(gp.quicksum(x[j] for j in row) <= int(cap))
    if pooled_energy_cap is not None:
        model.addConstr(gp.quicksum(float(columns[j]["E_soc_required_Wh"]) * x[j]
                                    for j in range(n)) <= float(pooled_energy_cap))
    for cut in no_good_cuts:
        cut_set = {int(j) for j in cut}
        model.addConstr(
            gp.quicksum((1.0 if j in cut_set else -1.0) * x[j] for j in range(n))
            <= len(cut_set) - 1)
    for cut in full_cover_no_good_cuts:
        cut_set = {int(j) for j in cut}
        model.addConstr(gp.quicksum(x[j] for j in cut_set) <= len(cut_set) - 1)
    coverage_expr = gp.quicksum(len(route_tids(columns[j])) * x[j] for j in range(n))
    if coverage_equal is not None:
        model.addConstr(coverage_expr == int(coverage_equal))
    elif coverage_min is not None:
        model.addConstr(coverage_expr >= int(coverage_min))
    energy_expr = gp.quicksum(float(columns[j]["E_plan_Wh"]) * x[j] for j in range(n))
    if energy_upper_bound_Wh is not None:
        model.addConstr(energy_expr <= float(energy_upper_bound_Wh))
    if phase == "coverage":
        model.setObjective(-coverage_expr, GRB.MINIMIZE)
    elif phase == "energy":
        model.setObjective(energy_expr, GRB.MINIMIZE)
    else:
        raise ValueError("phase must be 'coverage' or 'energy'")
    model.optimize()
    optimal = model.Status == GRB.OPTIMAL
    infeasible = model.Status == GRB.INFEASIBLE
    checked = None
    if model.SolCount > 0:
        checked = validate_selected_solution(
            [x[j].X for j in range(n)], columns, turbine_ids, deck_rows, no_good_cuts,
            capacity_rows=capacity_rows, pooled_energy_cap=pooled_energy_cap,
            coverage_equal=coverage_equal, coverage_min=coverage_min,
            energy_upper_bound_Wh=energy_upper_bound_Wh,
            full_cover_equal=bool(full_cover_equal),
            full_cover_no_good_cuts=full_cover_no_good_cuts)
    obj_value = None if checked is None else float(model.ObjVal)
    obj_bound = None
    try:
        if math.isfinite(float(model.ObjBound)):
            obj_bound = float(model.ObjBound)
    except Exception:
        pass
    gap_pct = None
    try:
        if model.SolCount > 0 and math.isfinite(float(model.MIPGap)):
            gap_pct = max(0.0, 100.0 * float(model.MIPGap))
    except Exception:
        pass
    return MasterResult(
        "optimal" if optimal else ("infeasible" if infeasible else "limit-or-interrupt"),
        checked, obj_value, obj_bound, optimal, infeasible, "gurobi", gap_pct)


def solve_binary_master(
    backend: str,
    *args: Any,
    **kwargs: Any,
) -> MasterResult:
    requested = str(backend).lower()
    order = ["gurobi", "scipy"] if requested == "auto" else [requested]
    last: MasterResult | None = None
    for name in order:
        if name == "gurobi":
            last = solve_binary_master_gurobi(*args, **kwargs)
        elif name == "scipy":
            last = solve_binary_master_scipy(*args, **kwargs)
        else:
            raise ValueError("solver backend must be auto, gurobi or scipy")
        if last.status != "backend-unavailable":
            return last
    assert last is not None
    return last


def safe_coverage_upper_bound(
    master: MasterResult,
    turbine_count: int,
    *,
    pricing_epsilon: float = 0.0,
    max_selected_routes: int | None = None,
) -> int | None:
    """Safe integer coverage upper bound for a minimization model of ``-C``.

    If exact pricing only proves every omitted route has reduced cost at least
    ``-pricing_epsilon``, at most ``M`` selected routes can improve the RMP
    bound by ``M*pricing_epsilon``.  Set packing gives the safe default
    ``M <= number of turbines`` because every selected route covers at least
    one distinct turbine.
    """
    if master.objective_bound is None:
        return None
    M = int(turbine_count if max_selected_routes is None else max_selected_routes)
    raw = -float(master.objective_bound) + max(0, M) * max(0.0, float(pricing_epsilon))
    return min(int(turbine_count), max(0, int(math.floor(raw + INT_TOL))))


def safe_energy_lower_bound(
    master: MasterResult,
    *,
    pricing_epsilon: float = 0.0,
    max_selected_routes: int | None = None,
) -> float | None:
    """Safe energy lower bound with the same omitted-column correction."""
    if master.objective_bound is None:
        return None
    M = 0 if max_selected_routes is None else max(0, int(max_selected_routes))
    corrected = float(master.objective_bound) - M * max(0.0, float(pricing_epsilon))
    return max(0.0, corrected - ENERGY_TOL_WH)


def exact_resource_audit(
    columns: Sequence[Mapping[str, Any]],
    selected: Sequence[int],
    *,
    uav_count: int,
    battery_count: int,
    usable_battery_energy_Wh: float,
    resource_of: Sequence[Mapping[str, Any]],
    quick_inspection_min: float,
    swap_min: float,
    quick_capacity: int,
    swap_capacity: int,
    overlap: Callable[[Sequence[float], Sequence[float]], bool],
    event_capacity_ok: Callable[[Sequence[Sequence[float]], Sequence[float], int], bool],
    deadline: float | None,
) -> ResourceAuditResult:
    """Exact UAV/battery/SOC/turnaround DFS with a global deadline and tri-state result."""
    K = int(uav_count)
    B = int(battery_count)
    ordered = tuple(sorted({int(j) for j in selected},
                           key=lambda j: (float(resource_of[j]["launch_start_min"]),
                                          float(resource_of[j]["recovery_min"]), j)))
    failure_reasons: dict[str, int] = {}

    def _note(reason: str, amount: int = 1) -> None:
        failure_reasons[str(reason)] = int(
            failure_reasons.get(str(reason), 0)) + int(amount)

    if duplicate_turbines(columns, ordered):
        _note("duplicate_turbine")
        return ResourceAuditResult(
            ResourceAuditStatus.INFEASIBLE_PROVEN, explored_nodes=0,
            failure_reasons=dict(failure_reasons))
    if not ordered:
        assignment = dict(
            uav_assignment={}, battery_assignment={}, mission_service={},
            battery_energy_used_Wh=[0.0] * max(B, 0), swap_events=[],
            quick_inspection_events=[], uav_chains=[[] for _ in range(max(K, 0))],
            battery_binding=[None] * max(B, 0), resource_audit="exact-backtracking")
        return ResourceAuditResult(
            ResourceAuditStatus.FEASIBLE, assignment, 1,
            failure_reasons=dict(failure_reasons))
    if K <= 0 or B <= 0 or int(quick_capacity) <= 0 or int(swap_capacity) <= 0:
        _note("nonpositive_resource_capacity")
        return ResourceAuditResult(
            ResourceAuditStatus.INFEASIBLE_PROVEN, explored_nodes=1,
            failure_reasons=dict(failure_reasons))
    if deadline_reached(deadline):
        return ResourceAuditResult(
            ResourceAuditStatus.UNKNOWN_TIMEOUT, explored_nodes=0,
            failure_reasons=dict(failure_reasons))

    for pos, j in enumerate(ordered):
        if deadline_reached(deadline):
            return ResourceAuditResult(ResourceAuditStatus.UNKNOWN_TIMEOUT, explored_nodes=0)
        for q in ordered[pos + 1:]:
            if deadline_reached(deadline):
                return ResourceAuditResult(ResourceAuditStatus.UNKNOWN_TIMEOUT, explored_nodes=0)
            for a in resource_of[j]["deck"]:
                for b in resource_of[q]["deck"]:
                    if deadline_reached(deadline):
                        return ResourceAuditResult(ResourceAuditStatus.UNKNOWN_TIMEOUT, explored_nodes=0)
                    if overlap(a, b):
                        _note("deck_overlap")
                        return ResourceAuditResult(
                            ResourceAuditStatus.INFEASIBLE_PROVEN,
                            explored_nodes=1,
                            failure_reasons=dict(failure_reasons))

    uav_last: list[int | None] = [None] * K
    uav_current_battery: list[int | None] = [None] * K
    uav_chains: list[list[int]] = [[] for _ in range(K)]
    battery_binding: list[int | None] = [None] * B
    # SOC accounting is exact for the binary64 input energies: convert each
    # stored float to its exact rational value and never relax capacity by a
    # positive engineering tolerance.
    usable_battery_energy_exact = Fraction.from_float(float(usable_battery_energy_Wh))
    battery_used = [Fraction(0, 1) for _ in range(B)]
    uav_assignment: dict[int, int] = {}
    battery_assignment: dict[int, int] = {}
    mission_service: dict[int, dict[str, Any]] = {}
    swap_events: list[tuple[float, float]] = []
    quick_events: list[tuple[float, float]] = []
    explored = 0
    memo_hits = 0
    # v15 exact failed-state memoization.  The key is deliberately fully
    # labeled (no unproved symmetry quotient): it contains every state component
    # that can affect future feasibility.  Only states exhaustively returning
    # False are cached; UNKNOWN_TIMEOUT is never memoized.
    failed_state_cache: set[Any] = set()

    def search(pos: int) -> bool | None:
        nonlocal explored, memo_hits
        explored += 1
        if deadline_reached(deadline):
            return None
        if pos >= len(ordered):
            return True
        state_key = (
            int(pos),
            tuple(uav_last),
            tuple(uav_current_battery),
            tuple(tuple(chain) for chain in uav_chains),
            tuple(battery_binding),
            tuple(battery_used),
            tuple(swap_events),
            tuple(quick_events),
        )
        if state_key in failed_state_cache:
            memo_hits += 1
            return False
        j = ordered[pos]
        start = float(resource_of[j]["launch_start_min"])
        start_exact = Fraction.from_float(start)
        e_need_float = float(columns[j]["E_soc_required_Wh"])
        if not math.isfinite(e_need_float) or e_need_float < 0.0:
            _note("invalid_mission_energy")
            failed_state_cache.add(state_key)
            return False
        e_need = Fraction.from_float(e_need_float)
        if e_need > usable_battery_energy_exact:
            _note("single_mission_soc_over_capacity")
            failed_state_cache.add(state_key)
            return False
        seen_uav: set[Any] = set()
        uav_order = sorted(range(K), key=lambda k: (
            uav_last[k] is not None,
            -1.0 if uav_last[k] is None else float(resource_of[uav_last[k]]["clear_end_min"]),
            len(uav_chains[k]), k))
        for k in uav_order:
            if deadline_reached(deadline):
                return None
            prev = uav_last[k]
            prev_b = uav_current_battery[k]
            signature = (
                prev is None,
                None if prev is None else float(resource_of[prev]["clear_end_min"]).hex(),
                prev_b,
                tuple(battery_used[b] for b in range(B) if battery_binding[b] == k))
            if signature in seen_uav:
                continue
            seen_uav.add(signature)
            candidates = list(range(B))
            candidates.sort(key=lambda b: (
                0 if b == prev_b else (1 if battery_binding[b] == k else 2),
                battery_used[b], b))
            seen_battery: set[Any] = set()
            for b in candidates:
                if deadline_reached(deadline):
                    return None
                owner = battery_binding[b]
                if owner not in (None, k):
                    _note("battery_bound_to_other_uav")
                    continue
                b_signature = (b == prev_b, owner is None, battery_used[b])
                if b_signature in seen_battery:
                    continue
                seen_battery.add(b_signature)
                if battery_used[b] + e_need > usable_battery_energy_exact:
                    _note("battery_soc_capacity")
                    continue
                event: tuple[float, float] | None = None
                predecessor: int | None
                if prev is None:
                    mode = "initial_preinstalled"
                    predecessor = None
                else:
                    clear = float(resource_of[prev]["clear_end_min"])
                    clear_exact = Fraction.from_float(clear)
                    predecessor = int(prev)
                    if b == prev_b:
                        mode = "quick_reuse"
                        service_exact = Fraction.from_float(
                            max(float(quick_inspection_min), 0.0))
                        ready_exact = clear_exact + service_exact
                        if start_exact < ready_exact:
                            _note("quick_reuse_ready_time")
                            continue
                        event = (clear_exact, ready_exact)
                        if not event_capacity_ok(quick_events, event, int(quick_capacity)):
                            _note("quick_capacity")
                            continue
                    else:
                        mode = "battery_swap"
                        service_exact = Fraction.from_float(max(float(swap_min), 0.0))
                        ready_exact = clear_exact + service_exact
                        if start_exact < ready_exact:
                            _note("swap_ready_time")
                            continue
                        event = (clear_exact, ready_exact)
                        if not event_capacity_ok(swap_events, event, int(swap_capacity)):
                            _note("swap_capacity")
                            continue

                old_owner = battery_binding[b]
                old_last, old_current = uav_last[k], uav_current_battery[k]
                battery_binding[b] = k
                battery_used[b] += e_need
                uav_last[k] = j
                uav_current_battery[k] = b
                uav_chains[k].append(j)
                uav_assignment[j] = k
                battery_assignment[j] = b
                mission_service[j] = dict(predecessor=predecessor, mode=mode,
                                          interval=None if event is None else tuple(event))
                if mode == "battery_swap" and event is not None:
                    swap_events.append(event)
                elif mode == "quick_reuse" and event is not None:
                    quick_events.append(event)

                result = search(pos + 1)
                if result is True:
                    return True

                if mode == "battery_swap" and event is not None:
                    swap_events.pop()
                elif mode == "quick_reuse" and event is not None:
                    quick_events.pop()
                mission_service.pop(j, None)
                battery_assignment.pop(j, None)
                uav_assignment.pop(j, None)
                uav_chains[k].pop()
                uav_last[k], uav_current_battery[k] = old_last, old_current
                battery_used[b] -= e_need
                battery_binding[b] = old_owner
                if result is None:
                    return None
        _note("assignment_dead_end")
        failed_state_cache.add(state_key)
        return False

    feasible = search(0)
    if feasible is None:
        return ResourceAuditResult(
            ResourceAuditStatus.UNKNOWN_TIMEOUT,
            explored_nodes=explored, memo_hits=memo_hits,
            failure_reasons=dict(failure_reasons))
    if not feasible:
        return ResourceAuditResult(
            ResourceAuditStatus.INFEASIBLE_PROVEN,
            explored_nodes=explored, memo_hits=memo_hits,
            failure_reasons=dict(failure_reasons))
    assignment = dict(
        uav_assignment={int(j): int(k) for j, k in uav_assignment.items()},
        battery_assignment={int(j): int(b) for j, b in battery_assignment.items()},
        mission_service={int(j): dict(v) for j, v in mission_service.items()},
        battery_energy_used_Wh=[float(v) for v in battery_used],
        swap_events=[tuple(map(float, e)) for e in swap_events],
        quick_inspection_events=[tuple(map(float, e)) for e in quick_events],
        uav_chains=[[int(j) for j in chain] for chain in uav_chains],
        battery_binding=[None if k is None else int(k) for k in battery_binding],
        resource_audit="exact-backtracking",
        resource_numeric_contract="binary64-strict-half-open-time-exact-rational-soc")
    return ResourceAuditResult(
        ResourceAuditStatus.FEASIBLE, assignment,
        explored_nodes=explored, memo_hits=memo_hits,
        failure_reasons=dict(failure_reasons))

def _resource_module_main():
    """Keep the resource module import-safe and prevent accidental legacy runs."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Resource-audit library module; formal optimization is launched from step13_experiment_model.py.")
    parser.add_argument("--show-entry", action="store_true",
                        help="print the supported formal entry")
    parser.parse_args()
    print("Formal resource audit: step11_algorithm_route_drcc.exact_resource_audit")
    print("Formal optimizer: step12_branch_price.solve_fleet_anytime")
    print("Experiment CLI: python step13_experiment_model.py --help")
    print("Historical in-file demos are research-only and are not auto-executed.")


if __name__ == "__main__":
    _resource_module_main()

