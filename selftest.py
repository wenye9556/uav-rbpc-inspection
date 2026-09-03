#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
selftest.py — 统一自测入口。

主要套件：
  core               核心数学、半径、SOC 与溯源字段；
  bp                 小实例 B&P 与完整枚举锚点对拍；
  branch_price       L1 分支定价与连续资源语义；
  l2_energy          L2 能耗证书；
  certificates       证书闭环与撤证条件；
  seed_validation    外部列与整数 LP 校验；
  solver_validation  MILP 原始解、逐时天气和对偶失败保护；
  resume_fast        检查点签名、失败任务重试与口径冲突快速门禁；
  resume             完整 E1 截断续跑（慢速）；
  counterexamples / mutations / random_oracle  独立审计。

用法：
  python selftest.py --suite core
  python selftest.py --suite certificates
  python selftest.py --suite all
"""
from __future__ import annotations
import ast

import argparse
import os
import json
import logging
import math
import sys
import time
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd

import step11_algorithm_route_drcc as FAC
import step9_model as M
import step10_model_routing as RM
import step11_algorithm_route_drcc as RA
import step12_branch_price as BP
import step13_experiment_model as S13
import step15_replay as RP
import step7_compute_xi as S7
import step8_gen_recovery as S8
EU = M  # shared utilities are merged into step9_model to preserve package layout

OK = "  ✓"


# =============================================================================
# suite: bp(原 selftest_更新_bp.py)
# =============================================================================
def suite_bp():
    turbines, wx_df, xi_amb, lat0lon0, sc_csv, src, track_csv = S13.load_all(4, allow_synth=True)
    p = M.apply_uav_profile(M.Params(), "S")
    wamb = RM.weather_ambiguity_from_series(wx_df, RM.decision_horizons_of(xi_amb), scale=1.0)
    opts, reach, kind, T_eff, wx0 = S13.build_launch_options(
        turbines, lat0lon0, None, xi_amb, wx_df, 90.0, 15.0, 8000.0,   # 夹具固定半径(测试确定性, 非实验口径)
        hs_quantile=0.5, allow_synth=True)
    print(f"实例: {kind}, T={T_eff:.0f}min, |opts|={len(opts)}, reach={len(reach)}")

    # Historical Ryan–Foster smoke: run one deck mode only.  The slot path is
    # covered by core/branch_price and could block in a native historical LP call.
    for dm in ("interval",):
        ext, st = RA.enumerate_discrete_routes(reach, opts, p, xi_amb, T_eff, 2.5, 1, wamb)
        r_ext = RA.solve_resource_master(reach, opts, p, xi_amb, 3, T_eff, t_swap_min=4.0,
                                  max_stops=1, weather_unc=wamb, cols_override=ext,
                                  solver="auto", deck_mode=dm)
        seed = RA.build_route_columns(reach, opts, p, xi_amb, T_eff, 2.5, 1, wamb,
                                 "drcc", 2.0, "vp_unimodal", 8.0)
        t0 = time.time()
        rb = BP.solve_soft_coverage_research(reach, opts, p, xi_amb, 3, T_eff, t_swap_min=4.0,
                                  max_stops=1, weather_unc=wamb, seed_cols=seed,
                                  time_limit_s=300, deck_mode=dm)
        ok = (rb["L1_status"].startswith("optimal") and rb["covered"] == rb["UB"] == r_ext["covered"])
        print(f"[{dm:8s}] ext_opt={r_ext['covered']} (池{st['n_cols']}) | "
              f"BP LB={rb['covered']} UB={rb['UB']} gap={rb['gap_pct']}% {rb['status']} "
              f"nodes={rb['nodes']} 分支(t/p/c)={rb['n_branch_turbine']}/{rb['n_branch_pair']}/{rb['n_branch_col']} "
              f"{time.time()-t0:.1f}s → {'PASS' if ok else 'FAIL'}")
        assert ok, f"deck_mode={dm} 证书不匹配!"

    # Continuous/slot semantics are checked non-trivially in branch_price.
    print("BP 证书冒烟全部通过。")


# =============================================================================
# suite: branch(原 selftest_更新_branch.py)
# =============================================================================
def suite_branch():
    """确定性分支路径回归。

    旧版在14台风机×15起飞时隙的合成E1实例上重复跑6次完整B&P，模型收紧后既可能
    退化为空池，也可能在原生HiGHS调用中长时间停滞。当前直接复用更强的
    ``suite_branch_price``：其中包含非退化小实例、自然多层风机/列分支、连续甲板冲突、
    Phase-I、限制撤证及定价调用恒等式，并对每个证书条件做断言。
    """
    print("branch 慢套件已替换为确定性、非退化的完整分支路径回归。")
    suite_branch_price()


# =============================================================================
# suite: e1(原 selftest_更新_e1.py)
# =============================================================================
def suite_e1():
    # ── 夹具(与 selftest_更新_branch 同源: 合成航迹 + 真实风机几何) ──────────────
    turbines, wx_df, _xi_loaded, lat0lon0, sc_csv, src, track_csv = S13.load_all(10, allow_synth=True)
    # 结构测试使用已知可行的小方差矩夹具；真实量级重尾/风浪判杀由 core、
    # counterexamples 和 branch_price 独立验证，避免 E1 表结构测试退化为空池。
    xi_amb = RM._demo_xi([5, 10, 15, 20, 25, 30],
                          ["直航", "转弯", "低速", "动力定位"])
    # E1 套件验证实验表/续跑结构，不在占位天气上重复验证多源风浪门；
    # 风浪门由 core/branch_price/counterexamples 独立覆盖。
    wamb = None
    opts, reach, kind, T_eff, wx0 = S13.build_launch_options(
        turbines, lat0lon0, None, xi_amb, wx_df, 150.0, 10.0, 8000.0,
        hs_quantile=0.5, allow_synth=True)
    print(f"实例: {kind}, T={T_eff:.0f}min, |opts|={len(opts)}, reach={len(reach)}")
    # This slow suite validates E1 table/resume contracts.  Replace the large
    # geometry only inside the test with one physically feasible turbine at one
    # real launch state; exact multi-stop pricing and the original geometry are
    # exercised by the dedicated BPC/oracle and core suites.  The tiny fixture
    # prevents a CSV/resume regression from becoming a full-permutation stress
    # test while still traversing the real formal solver and replay interfaces.
    opt0 = opts[0]
    tiny = M.Turbine("E1-TINY", np.array([0.0, 0.0]), 10.0, 20.0)
    tiny.local = np.asarray(opt0.ship.P_launch, float).copy()
    reach = [tiny]
    opts = [opt0]
    xi_amb = RM._demo_xi([10], ["直航", "转弯", "低速", "动力定位"])
    print(f"E1 结构夹具: |opts|={len(opts)}, reach={len(reach)}, max_stops=1")

    # 工程回归：任务窗口前历史必须保留；决策 h=25 必须有回收状态和独立回收天气。
    xi_anchor = RM._demo_xi([5, 10, 15, 20, 30],
                             ["直航", "转弯", "低速", "动力定位"])
    H_dec = RM.decision_horizons_of(xi_anchor)
    assert 25 in H_dec, "统计锚点 20/30 之间应生成决策 h=25"
    tr = M.ShipTrack(np.array([0., 60., 120., 180., 240., 900., 1800.]),
                     np.array([[0., 0.], [60., 0.], [120., 0.], [180., 0.],
                               [240., 0.], [900., 0.], [1800., 0.]]),
                     absolute_start=pd.Timestamp("2026-01-01T00:00:00Z"),
                     time_source="selftest")
    def _wx_time(t_sec):
        return dict(Hs=0.4, Tp=6.0, wave_dir=210.0, ship_heading=0.0,
                    wind10=4.0 + float(t_sec) / 600.0, wind_dir_from=230.0,
                    weather_target_time=str(tr.absolute_time(t_sec)))
    grid = RM.build_launch_grid_from_track(
        tr, [120.0], H_dec, c_state="直航", wx_of_t=_wx_time,
        predictor="cv_noleak", mission_origin_sec=120.0)
    go = grid[0]
    assert abs(go.tau_min) < 1e-12, "任务起点应保持 tau=0，而不是历史缓冲起点"
    assert np.linalg.norm(go.ship._v_ship) > 0.9, "首时隙应使用任务前 AIS 历史估速"
    assert 25 in go.ship.recovery_state_by_h, "h=25 缺回收状态预测"
    assert go.ship.weather_at_h(25)["wind10"] > go.wx["wind10"], \
        "回收天气不得复用起飞天气"
    assert go.ship.weather_at_h(25)["weather_target_time"] != go.wx["weather_target_time"]
    print("0) AIS 历史 / h=25 / 回收时刻天气合同 PASS")

    # ── 1) _stops_cap 解析: auto=⌊h_max/τ_insp⌋(默认统计层 30 ⇒ 决策 h_max=45 ⇒ 9);
    #       整数/逐档映射/缺档回退 ────────────────────────────────────────────────
    p_s = M.apply_uav_profile(M.Params(), "S")
    h_max = max(RM.decision_horizons_of(xi_amb))
    want = max(4, int(h_max // (p_s.tau_insp / 60.0)))
    assert S13._stops_cap("auto", p_s, xi_amb, 4) == want, "auto 上界计算错"
    assert S13._stops_cap("4", p_s, xi_amb, 4) == 4, "整数覆盖失效(更新 复现口径)"
    assert S13._stops_cap("S:3,L:6", p_s, xi_amb, 4) == 3, "逐档映射失效"
    assert S13._stops_cap("L:6", p_s, xi_amb, 4) == 4, "缺档未回退 fallback"
    print(f"1) _stops_cap PASS(auto={want}, h_max={h_max}, τ_insp={p_s.tau_insp/60:.0f}min)")

    # ── 2) E1_frontier: 小轴 + 低 cap 跑通; 断言结果合同/延伸/明细 ─────────
    import tempfile
    outdir = Path(tempfile.mkdtemp(prefix="selftest_e1_"))   # 更新: 临时目录, 不污染代码包
    args = Namespace(e1_uavs="S,L", fleet_ks="1,2", e1_batteries="0,1",
                     e1_b_auto="on", e1_b_cap=4, e1_sat_patience=2,
                     # This suite checks E1 table/resume structure, not large-instance
                     # pricing scalability.  Keep the fixture at an explicit low exact
                     # cap; the formal solver itself remains completely untruncated.
                     stops_cap="1", max_stops=1,
                     deck_delta_min=2.5, deck_mode="interval", dtau_min=10.0,
                     t_swap_min=None, t_launch_min=None, landing_clear_min=1.0,
                     swap_stations=1, battery_reuse_mode="exact_soc", replay_n=20,
                     allow_synth=True, track_start_min=None, uav="S",
                     knee_frac=0.95, knee_order="BK",
                     selection_metric="safe_per_inventory_kWh",
                     validation_mode="synthetic_stress", validation_samples=None,
                     xi_train_samples=None, final_test_samples=None,
                     final_weather_mode="synthetic", resume="off")
    df = S13.E1_frontier(reach, opts, M.Params(), xi_amb, wamb, outdir, args, kind, T_eff)
    assert {"stops_cap", "stops_cap_hit", "max_stops_requested",
            "max_stops_effective", "max_stops_observed", "route_pool_status",
            "route_pool_count", "optimization_status", "validation_status",
            "zero_coverage_reason"} <= set(df.columns), "E1 工程审计列缺失"
    assert (df["result_contract"] == S13.RESULT_CONTRACT).all(), "结果合同未同步"
    assert (df["max_stops"] == df["max_stops_effective"]).all()
    assert (df["stops_cap"] == df["max_stops_effective"]).all()
    assert (df["max_stops_requested"] == 1).all()
    assert (df["max_stops_observed"] <= df["max_stops_effective"]).all()
    for uk in ("S", "L"):
        usub = df[df.uav == uk]
        bmax = int(usub.batteries.max())
        assert bmax <= args.e1_b_cap, f"{uk}: B 超过 --e1-b-cap"
        # This one-turbine fixture can prove the hard coverable cap already at
        # the base endpoint B=1; formal auto-extension must stop immediately
        # rather than burn extra cells merely to satisfy the old observed-safe
        # patience rule.  If it does not stop, any extension is still bounded.
        if bmax == 1:
            tail = usub[(usub.K == usub.K.max()) & (usub.batteries == bmax)].iloc[0]
            assert int(tail.coverage_incumbent) >= int(tail.coverable_note), \
                f"{uk}: B=1 只有在 hard-coverable-cap 已证明时才允许停止延伸"
        det = outdir / f"E1_detail_Kmax_{uk}.csv"
        assert det.is_file(), f"缺逐 UAV 明细 {det.name}"
        # Formal on-demand pricing deliberately has no prebuilt route ledger;
        # its stage-count file must record that fact instead of fabricating
        # research-pool diagnostics.
        stage_file = outdir / f"E1_route_stage_counts_{uk}.csv"
        assert stage_file.is_file(), f"{uk}: 缺按需定价阶段记录"
        stage_df = pd.read_csv(stage_file)
        _stage_status = set(stage_df["status"].astype(str))
        assert _stage_status <= {
            "formal-revalidated-heuristic-warmstart",
            "formal-certified-complete-route-universe"}, _stage_status
        if "formal-certified-complete-route-universe" in _stage_status:
            assert bool(stage_df["route_universe_complete"].all())
            assert (pd.to_numeric(stage_df["route_universe_columns"], errors="coerce") >= 0).all()
            assert (stage_df["warmstart_status"].astype(str)
                    == "superseded-by-complete-universe").all()
        assert {"heuristic_seed_candidates", "heuristic_multistop_seed_candidates",
                "warmstart_status"} <= set(stage_df.columns)
        assert not (outdir / f"E1_route_diagnostics_{uk}.csv").exists()
        assert not (outdir / f"E1_failure_summary_{uk}.csv").exists()
        dd = pd.read_csv(det)
        assert {"margin_E_Wh", "margin_T_s", "binding"} <= set(dd.columns), "明细缺裕度/绑定列"
        assert set(dd["binding"]) <= {"energy", "time"}, "binding 取值异常"
    assert (outdir / "E1_selection.csv").is_file(), "E1_frontier 应同步保存选型表"
    selection = pd.read_csv(outdir / "E1_selection.csv")
    chosen = selection[selection["selected"] == True]  # noqa: E712
    # 最终协议 fail-closed：有限 synthetic 样本若无法使逐架次同时上界 ≤5%，
    # 允许没有 selected knee；不得为了表结构测试强行选择不安全配置。
    assert len(chosen) in (0, 1), "selected knee 数量异常"
    if len(chosen) == 1:
        assert (outdir / "E1_detail_Kmax.csv").is_file(), "缺 selected knee 通用明细"
        leg = pd.read_csv(outdir / "E1_detail_Kmax.csv")
        pick = chosen.iloc[0]
        assert set(leg["solution_role"]) == {"selected_knee_final_resolve"}
        assert bool(leg["final_plan_validation_holds"].all())
        assert leg["frozen_plan_fingerprint"].nunique() == 1
        assert set(leg["uav"].astype(str)) == {str(pick["uav"])}
        assert set(leg["K"].astype(int)) == {int(pick["knee_K"])}
        assert set(leg["batteries"].astype(int)) == {int(pick["knee_B"])}
    else:
        assert not (outdir / "E1_detail_Kmax.csv").exists(),             "无可靠 selected knee 时不应伪造通用明细"
    # 覆盖对 B 单调不减(资源行只会放松) —— 延伸行与基础行同池同解, 不应出现倒挂
    for uk in ("S", "L"):
        for K in (1, 2):
            s = df[(df.uav == uk) & (df.K == K)].sort_values("batteries")
            assert s["safe_served"].is_monotonic_increasing or \
                   (s["safe_served"].diff().dropna() >= 0).all(), f"{uk} K={K}: 覆盖对 B 倒挂"
    print(f"2) E1_frontier PASS({len(df)} 行; S 至 B={int(df[df.uav=='S'].batteries.max())}, "
          f"L 至 B={int(df[df.uav=='L'].batteries.max())}; cap 触发={bool(df.stops_cap_hit.any())})")

    # ── 3) E1_select(doc §2.3 两步语义): 真实跑出的 df 上出表 + 合成退化/饱和用例 ────
    sel = S13.e1_select_from_df(df, frac=0.95, order="BK", patience=2)
    assert set(sel.uav) == {"S", "L"}, "选型表档位不完整"
    for uk in ("S", "L"):
        source_safe = bool(df[(df.uav == uk) & (df.plan_holds == True)].shape[0])  # noqa: E712
        row = sel[sel.uav == uk].iloc[0]
        assert bool(pd.notna(row["knee_B"])) == source_safe
        if not source_safe:
            if int(row["plateau_safe"]) == 0:
                assert row["selection_status"] in {"empty_route_pool", "zero_positive_coverage"}
                assert pd.isna(row["sat_reached"]), "全零曲线不得标记资源饱和"
            else:
                assert row["selection_status"] == "unsafe_no_validated_candidate"
    assert {"knee_per_battery", "knee_energy_per_safe"} <= set(sel.columns), "膝点处指标缺失"
    # 退化用例: 0..3 严格线性(无饱和) ⇒ degenerate_knee 必须为 True
    lin = pd.DataFrame([dict(uav="X", K=1, batteries=b, safe_served=3 * b,
                             per_battery=(3.0 if b else None),
                             energy_per_safe=(50.0 if b else None),
                             stops_cap=9, stops_cap_hit=False, coverable_note=99,
                             plan_holds=True)
                        for b in range(4)])
    sel_lin = S13.e1_select_from_df(lin, frac=0.95, order="BK", patience=2)
    assert bool(sel_lin.iloc[0]["degenerate_knee"]) is True, "线性无饱和曲线未标 degenerate"
    # 饱和用例: 0,3,6,6,6 ⇒ sat_reached=True 且膝点=B2(6≥0.95·6)
    sat = pd.DataFrame([dict(uav="Y", K=1, batteries=b, safe_served=v,
                             per_battery=(v / b if b else None),
                             energy_per_safe=(50.0 if b else None),
                             stops_cap=9, stops_cap_hit=False, coverable_note=99,
                             plan_holds=True)
                        for b, v in enumerate((0, 3, 6, 6, 6))])
    sel_sat = S13.e1_select_from_df(sat, frac=0.95, order="BK", patience=2)
    r = sel_sat.iloc[0]
    assert bool(r["sat_reached"]) and not bool(r["degenerate_knee"]) and int(r["knee_B"]) == 2, \
        f"饱和膝点判定错: {dict(r)}"
    zero = pd.DataFrame([dict(uav="Z", K=K, batteries=B, safe_served=0,
                              per_battery=None, energy_per_safe=None,
                              stops_cap=9, stops_cap_hit=False, coverable_note=0,
                              route_pool_count=0, plan_holds=None)
                         for K in (1, 2) for B in (0, 1, 2)])
    rz = S13.e1_select_from_df(zero, frac=0.95, order="BK", patience=2).iloc[0]
    assert rz["selection_status"] == "empty_route_pool"
    assert pd.isna(rz["sat_reached"]) and bool(rz["degenerate_knee"])
    assert pd.isna(rz["knee_K"]) and pd.isna(rz["knee_B"])
    print("3) E1_select PASS(真实 df 出表; 线性→degenerate / 饱和→knee@B=2 / 全零→非饱和)")
    print("\n更新 selftest 全部 PASS。")


# =============================================================================
# suite: 更新(原 selftest_更新_fixes.py; t1..t6 函数逐字保留)
# =============================================================================
OK = "  ✓"


def t1_radius():
    print("[T1] 问题1: UAV 物理最大作业半径推导")
    h_max = 60.0
    vals = {}
    for uk in ("S", "M", "L"):
        q = M.apply_uav_profile(M.Params(), uk)
        d = M.max_flight_radius_m(q, h_max, dz_insp_m=55.0)
        vals[uk] = d
        assert d["R_max_m"] == min(d["R_energy_m"], d["R_time_m"])
        assert 3_000.0 < d["R_max_m"] < 40_000.0, f"{uk} 半径量级异常: {d}"
        print(f"{OK} {uk}: R_max={d['R_max_m']:.0f}m (能量限 {d['R_energy_m']:.0f} / "
              f"时间限 {d['R_time_m']:.0f}; E_avail={d['E_avail_Wh']:.1f}Wh)")
    assert vals["L"]["R_max_m"] > vals["S"]["R_max_m"], "L 档外包络应大于 S 档"
    # 解析器: auto 取批内外包络最大档; 显式数字直通
    xi = RM._demo_xi_realistic([5, 10, 15, 20, 30, 45], ["直航", "转弯", "低速", "动力定位"])
    tb = [M.Turbine(f"T{i}", np.array([11.6, 54.5]), 68.5, 115.0) for i in range(3)]
    a = SimpleNamespace(pair_radius="auto", e1_uavs="S,M,L", uav="L")
    r_auto, mode = S13._resolve_pair_radius(a, M.Params(), xi, tb)
    h_auto = max(RM.decision_horizons_of(xi))
    expected_auto = M.max_flight_radius_m(M.apply_uav_profile(M.Params(), "L"), h_auto, dz_insp_m=55.0)
    assert abs(r_auto - expected_auto["R_max_m"]) < 1.0 and mode.startswith("auto(L")
    a2 = SimpleNamespace(pair_radius="8000", e1_uavs="S", uav="S")
    r_fix, mode2 = S13._resolve_pair_radius(a2, M.Params(), xi, tb)
    assert r_fix == 8000.0 and mode2.startswith("explicit")
    print(f"{OK} 解析器: auto={r_auto:.0f}m({mode}); 显式 8000m 直通({mode2})")


def _cell(mu, Sxx, Syy, Sxy=0.0):
    Sig = np.array([[Sxx, Sxy], [Sxy, Syy]], float)
    return M.XiCell(h_min=30, c_state="直航", n=99999, mu=np.array(mu, float),
                    Sigma=Sig, support_radius=6 * math.sqrt(max(Sxx, Syy)),
                    p95_norm=0.0, rms_norm=0.0)


def t2_math():
    print("[T2] 问题3: geo2d 数学性质(vp 主判据口径)")
    eps, d0 = 0.05, 1500.0
    a = np.array([1.0, 0.0])                       # g=(1,0), c=1
    cell = _cell([20.0, -15.0], 300.0**2, 2500.0**2)
    orig = RM.kappa
    try:
        RM.kappa = RM.KAPPA_MODES["vp_unimodal"]   # 与 E1/E2 实验同口径(κ(0.05)=2.809)
        for b in (500.0, 900.0, 2000.0, 12000.0):
            m_lin = RM._soc_margin(a, b, cell, eps)
            m_geo = RM._soc_margin_geo2d(a, b, cell, eps, d0)
            assert m_geo <= m_lin + 1e-9, "vp 口径下 geo2d 必须不宽于线性(额外覆盖曲率)"
        print(f"{OK} 保守性(vp 口径): 对任意 b, margin_geo2d ≤ margin_linear")
        # 退化: σ⊥→0, μ⊥→0 ⇒ D_bound→L_ub ⇒ 校正额 = (κ₂(0.6ε)−κ_vp(ε))·σ_L(vp 下 >0)
        cell0 = _cell([20.0, 0.0], 300.0**2, 1e-12)
        m_lin0 = RM._soc_margin(a, 1000.0, cell0, eps)
        m_geo0 = RM._soc_margin_geo2d(a, 1000.0, cell0, eps, d0)
        gap_expect = (RM._kappa_two_sided(0.6 * eps) - RM.kappa(eps)) * 300.0
        assert gap_expect > 0 and abs((m_lin0 - m_geo0) - gap_expect) < 1e-6
        print(f"{OK} 退化(σ⊥=0): 校正额 = (κ₂(0.6ε)−κ_vp(ε))·σ_L = +{gap_expect:.1f}(解析吻合)")
    finally:
        RM.kappa = orig
    # 注: 默认 Cantelli 口径(κ=4.359)下, 沿向双侧 VP 盒(3.849)可比单侧 Cantelli 更松 ——
    # geo2d 的有效性是自含的(它界的是真实非线性距离), "不宽于线性"仅在 vp 主口径下成立。
    m_small = RM._soc_margin_geo2d(a, 5000.0, _cell([0, 0], 2000.0**2, 100.0**2), eps, 100.0)
    assert np.isfinite(m_small)
    print(f"{OK} L<0 过冲情形有限且被 |L_lo| 覆盖")


def _t3_samples(cell, n, seed):
    """矩匹配多元 t3 重尾样本(与 step15 回放同族: 协方差=Σ, 均值=μ, 尾部 t3)。"""
    rng = np.random.default_rng(seed)
    Lch = np.linalg.cholesky(cell.Sigma * (3.0 - 2.0) / 3.0)   # t3 方差=ν/(ν−2)=3 ⇒ 预除
    z = rng.standard_t(3.0, size=(n, 2))
    return cell.mu + z @ Lch.T


def t3_money():
    print("[T3] 问题3: 线性界在重尾下失守 vs geo2d 界成立(蒙特卡洛 n=200000, vp 口径)")
    eps, d0, c = 0.05, 1500.0, 1.0
    _orig_k = RM.kappa
    RM.kappa = RM.KAPPA_MODES["vp_unimodal"]   # 更新: κ₂ 已跟随口径, 展示数字 pin 主口径
    g = np.array([1.0, 0.0]); a = c * g
    cell = _cell([20.0, -15.0], 300.0**2, 2500.0**2)   # 强垂向 σ(直航大 h 病灶形态)
    xi = _t3_samples(cell, 200_000, seed=41)
    d_true = np.hypot(d0 + xi[:, 0] * (-1) * (-1) + 0*xi[:,0], 0) # placeholder replaced below
    # 真实非线性返程附加成本: extra = c·(‖v−ξ‖−d0), v = d0·(−g)?  按 step10 口径:
    # d(ξ)=‖q−(P+ξ)‖, v=q−P, g=−v/d0 ⇒ d(ξ)=‖v−ξ‖, L=d0+gᵀξ=d0−v̂ᵀξ。取 v=(−d0,0) ⇒ g=(1,0)。
    v = -d0 * g
    d_true = np.linalg.norm(v[None, :] - xi, axis=1)
    extra = c * (d_true - d0)
    # (i) 线性 VP 边界 b_lin(margin=0)在真实距离下的违反率
    b_lin = float(a @ cell.mu) + RM.kappa(eps) * math.sqrt(float(a @ cell.Sigma @ a))
    viol_lin = float(np.mean(extra > b_lin))
    print(f"{OK} 线性 VP 边界 b={b_lin:.0f}: 真实(非线性)违反率 = {viol_lin:.1%} ≫ ε={eps:.0%}"
          f"  ← 一阶低估凸距离的病灶复现")
    assert viol_lin > 4 * eps, "构造场景应显著失守以复现病灶"
    # (ii) geo2d 判定: 同一 b_lin 下应判不可行(margin<0)
    m_geo_at_blin = RM._soc_margin_geo2d(a, b_lin, cell, eps, d0)
    assert m_geo_at_blin < 0, "geo2d 应拒绝线性界下的伪可行"
    print(f"{OK} geo2d 对同一 b 的裕度 = {m_geo_at_blin:.0f} < 0 —— 正确拒绝")
    # (iii) geo2d 自身边界 b_geo(margin=0)的真实违反率 ≤ ε(界成立; VP 对 t3 应留有余量)
    D = RM._geo2d_dist_bound(cell, eps, d0, g, share_lin=0.6)
    b_geo = c * (D - d0)
    viol_geo = float(np.mean(extra > b_geo))
    print(f"{OK} geo2d 边界 b={b_geo:.0f}: 真实违反率 = {viol_geo:.2%} ≤ ε={eps:.0%}(界成立)")
    assert viol_geo <= eps + 0.003, f"geo2d 界失守: {viol_geo}"
    # (iv) 联合（含风源）实现可调用且更紧
    wu = SimpleNamespace(wind_bias=np.zeros(2), wind_cov=np.eye(2) * 0.25,
                         hs_bias=0.0, hs_std=0.05)
    m_j = RM._soc_margin_geo2d_joint(a, np.array([0.02, 0.0]), b_geo + 5.0, cell, wu, eps, d0)
    assert np.isfinite(m_j) and m_j <= 5.0 + 1e-9
    print(f"{OK} 多源联合实现可用(Bonferroni 跨源), margin={m_j:.2f}")
    RM.kappa = _orig_k


def t4_integration():
    print("[T4] 问题3: route_feasible_at_h 集成(soc_correction 开关)")
    horizons = [5, 10, 15, 20, 30, 45]
    xi = RM._demo_xi_realistic(horizons, ["直航", "转弯", "低速", "动力定位"])
    tbs = []
    for i, (x, y) in enumerate([(1200.0, 300.0), (1800.0, -200.0)]):
        t = M.Turbine(f"T{i}", np.array([11.6, 54.5]), 68.5, 115.0)
        t.local = np.array([x, y]); tbs.append(t)
    ship = RM.ShipPrediction.from_cv(np.array([0.0, 0.0]), np.array([0.2, 0.0]),
                                     horizons, "动力定位")
    r = RM.Route(rid=0, turbines=tbs, ship=ship)
    wx = dict(Hs=0.3, Tp=5.0, wave_dir=200.0, ship_heading=0.0, wind10=4.0, wind_dir=200.0)
    orig = RM.kappa
    try:
        RM.kappa = RM.KAPPA_MODES["vp_unimodal"]   # 与实验同口径
        p0 = M.apply_uav_profile(M.Params(), "L"); p0.soc_correction = "none"
        d_none = RM.route_feasible_at_h(r, 30, p0, wx, xi)
        p1 = M.apply_uav_profile(M.Params(), "L"); p1.soc_correction = "geo2d"
        d_geo = RM.route_feasible_at_h(r, 30, p1, wx, xi)
    finally:
        RM.kappa = orig
    assert d_none["soc_correction"] == "none" and d_geo["soc_correction"] == "geo2d"
    assert d_geo["margin_E"] <= d_none["margin_E"] + 1e-9
    assert d_geo["margin_T"] <= d_none["margin_T"] + 1e-9
    print(f"{OK} 裕度单调收紧(vp): mE {d_none['margin_E']:.1f}→{d_geo['margin_E']:.1f}, "
          f"mT {d_none['margin_T']:.0f}→{d_geo['margin_T']:.0f}; 口径字段随行")


def t5_select_guard():
    print("[T5] 问题4局限#4: 选型 plan_holds 守卫")
    rows = []
    for B in range(1, 6):
        for K in (1, 2):
            rows.append(dict(uav="L", K=K, batteries=B, safe_served=4 * B,
                             per_battery=4.0, energy_per_safe=100.0,
                             stops_cap=12, stops_cap_hit=False, coverable_note=90,
                             plan_holds=not (B == 5 and K == 1)))   # 膝点原本落 (B=5,K=1)=False
    df = pd.DataFrame(rows)
    sel = S13.e1_select_from_df(df, frac=0.95, order="BK")
    row = sel.iloc[0]
    assert int(row["knee_K"]) == 2 and int(row["knee_B"]) == 5 and bool(row["knee_plan_holds"])
    print(f"{OK} 守卫生效: 跳过 (B=5,K=1)[holds=False] → 选 (B=5,K=2)[holds=True]")
    df2 = df.copy(); df2.loc[df2.batteries == 5, "plan_holds"] = False
    sel2 = S13.e1_select_from_df(df2, frac=0.95, order="BK")
    assert sel2.iloc[0]["knee_K"] is None and sel2.iloc[0]["knee_B"] is None
    assert sel2.iloc[0]["selection_status"] == "unsafe_no_validated_candidate"
    pick2, warns2 = S13._pick_selection(sel2)
    assert pick2 is None and warns2
    print(f"{OK} 全不达标时 fail-closed：不冻结任何膝点")


def t6_provenance():
    print("[T6] 问题4局限#6: 溯源字段")
    a = SimpleNamespace(dtau_min=5.0, deck_delta_min=2.5, deck_mode="interval",
                        replay_n=400, allow_synth=False, track_start_min=None,
                        max_stops=8, _xi_source="真实 ξ 矩 xi_moments_caseB.csv",
                        _pair_radius_m=18711.0, _pair_radius_mode="auto(L,18711m,h_max=60)")
    p = M.apply_uav_profile(M.Params(), "L"); p.soc_correction = "geo2d"
    wx0 = dict(weather_alignment_mode="timestamp", weather_start_time="2026-01-01 00:00:00+00:00",
               weather_target_time="2026-01-01 00:05:00+00:00", weather_match_error_min=5.0,
               track_absolute_start="2026-01-01 00:05:00+00:00", track_time_source="column:time:utc")
    prov = S13._provenance(a, "真实AIS(...)", 360.0, [1, 2, 3], p, 5.0, 3.0, 12, wx0=wx0)
    for k in ("xi_source", "pair_radius_m", "pair_radius_mode", "soc_correction",
              "weather_alignment_mode", "weather_match_error_min", "max_stops_requested",
              "max_stops_effective", "stops_cap_spec"):
        assert k in prov, f"缺溯源字段 {k}"
    assert prov["soc_correction"] == "geo2d" and prov["pair_radius_mode"].startswith("auto")
    assert prov["weather_alignment_mode"] == "timestamp"
    print(f"{OK} 数据源、半径、SOC 与 AIS—天气同步口径全部入表")


def t2b_kappa_follow():
    print("[T2b] 更新: κ₂ 跟随当前判据口径(E2 对照语义保护)")
    import math as _m
    orig = RM.kappa
    try:
        RM.kappa = RM.KAPPA_MODES["vp_unimodal"]
        assert abs(RM._kappa_two_sided(0.05) - _m.sqrt(4/(9*0.05))) < 1e-9
        RM.kappa = RM.KAPPA_MODES["cantelli"]
        assert abs(RM._kappa_two_sided(0.05) - 1/_m.sqrt(0.05)) < 1e-9
        RM.kappa = RM.KAPPA_MODES["gaussian"]
        from scipy.stats import norm
        assert abs(RM._kappa_two_sided(0.05) - norm.ppf(0.975)) < 1e-6
        RM.kappa = lambda e: 0.0                    # nominal monkeypatch(step11 同法)
        assert RM._kappa_two_sided(0.05) == 0.0, "nominal 反例语义被破坏!"
        # nominal 下 geo2d 退化 = 均值点精确距离(仍无 σ 项 → 对随机性不设防, 语义保留)
        cell = _cell([20.0, -15.0], 300.0**2, 2500.0**2)
        D = RM._geo2d_dist_bound(cell, 0.05, 1500.0, np.array([1.0, 0.0]), 0.6)
        import math
        assert abs(D - math.hypot(1500.0 + 20.0, -15.0)) < 1e-6
        print(f"{OK} vp/cantelli/gaussian/nominal 四口径双侧常数各归其位; nominal 几何界=均值点距离")
    finally:
        RM.kappa = orig


def t8_autoselect():
    print("[T8] 更新: E2/A 选型自动回填(_resolve_e2_config)")
    sel = pd.DataFrame([
        dict(uav="S", knee_K=3, knee_B=22, knee_per_battery=2.18,
             knee_safe_per_inventory_kWh=5.10, degenerate_knee=False,
             sat_reached=True, knee_plan_holds=True),
        dict(uav="L", knee_K=4, knee_B=22, knee_per_battery=3.50,
             knee_safe_per_inventory_kWh=2.30, degenerate_knee=False,
             sat_reached=True, knee_plan_holds=True)])
    a = SimpleNamespace(uav="auto", k=None, batteries=None, _e1_sel_df=sel,
                        selection_metric="safe_per_inventory_kWh")
    S13._resolve_e2_config(a)
    assert (a.uav, a.k, a.batteries) == ("S", 3, 22), "auto 未取库存kWh归一化最优档"
    print(f"{OK} auto: 内存选型直通 → (S,3,22)=安全覆盖/库存kWh 最优")
    a2 = SimpleNamespace(uav="auto", k=7, batteries=9, _e1_sel_df=sel,
                         selection_metric="safe_per_inventory_kWh")   # 手给被忽略+警告
    S13._resolve_e2_config(a2)
    assert (a2.uav, a2.k, a2.batteries) == ("S", 3, 22)
    print(f"{OK} auto 下手给 --k/--batteries 被忽略(全自动语义, 带警告)")
    bad = sel.copy(); bad["degenerate_knee"] = True
    try:
        S13._resolve_e2_config(SimpleNamespace(uav="auto", k=None, batteries=None, _e1_sel_df=bad,
                                                     selection_metric="safe_per_inventory_kWh"))
        raise AssertionError("degenerate 选型被静默采用!")
    except SystemExit as e:
        assert ("方案保持未选择状态" in str(e)
                or "validation 风险门" in str(e)
                or "正式阈值配置" in str(e))
    print(f"{OK} legacy 全档 degenerate → SystemExit 拒绝自动采用；formal 离散阈值不再由该标志否决")
    m = SimpleNamespace(uav="M", k=None, batteries=None)
    S13._resolve_e2_config(m)
    assert (m.uav, m.k, m.batteries) == ("M", 3, None), "显式档位未走手动直通(K 默认 3)"
    print(f"{OK} 显式 --uav M → 手动直通, K 回默认 3")
    # 真实数据闭环: E1_frontier.csv 现算路径 → 应回填旧口径存档选型 (L,4,22)
    import shutil, tempfile
    real = Path(os.environ.get(
        "E1_FRONTIER_CSV",
        "results/model_experiments/E1_frontier/E1_frontier.csv",
    ))
    if real.is_file():
        tmp = Path(tempfile.mkdtemp(prefix="sel_"))
        (tmp / "E1_frontier").mkdir()
        shutil.copy(real, tmp / "E1_frontier" / "E1_frontier.csv")
        orig_res = S13.RESULTS
        try:
            S13.RESULTS = tmp
            r = SimpleNamespace(uav="auto", k=None, batteries=None)
            S13._resolve_e2_config(r)
            assert (r.uav, r.k, r.batteries) == ("L", 4, 22)
            print(f"{OK} 真实 CSV 现算路径闭环 → (L,4,22)(与 E1_select 一致)")
        finally:
            S13.RESULTS = orig_res
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        print("  (跳过真实 CSV 闭环: 本机无 结果文件)")



def t9_statistics_and_completion():
    print("[T9] 风险统计置信界与实验矩阵完整性")
    up = EU.binomial_upper_bound(0, 23, confidence=0.95)
    assert up is not None and 0.11 < up < 0.13, up
    lo, hi = EU.binomial_interval(0, 23, confidence=0.95)
    assert lo == 0.0 and hi > up, (lo, hi, up)
    st = EU.matrix_completion([("vp", 0.2), ("vp", 0.5)], [("vp", 0.2)])
    assert not st["complete"] and st["missing"] == [("vp", "0.5")], st
    st2 = EU.matrix_completion([("vp", 0.2)], [("vp", 0.2)])
    assert st2["complete"], st2
    # purged 三段切分 + 泄漏拒绝 + ALL 读取不再筛空
    rows = []
    for i in range(300):
        h = 5 if i % 2 == 0 else 10
        rows.append(dict(mmsi="123", source_track="track_A.csv",
                         source_track_id="/formal/track_A.csv", segment_id=0,
                         h_min=h, c_state="动力定位",
                         t0_epoch=1_700_000_000 + i * 120,
                         t1_epoch=1_700_000_000 + i * 120 + h * 60,
                         xi_e_m=float(i % 7), xi_n_m=float(i % 5)))
    raw = pd.DataFrame(rows)
    tr, va, te, meta = RP.purged_temporal_split(raw, 0.6, 0.2, purge_min=10)
    chk = RP.validate_holdout_disjointness(tr, va, te, purge_min=10)
    assert chk["disjoint"] and not chk["independence_verified"] and len(tr) and len(va) and len(te)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "samples.csv"; raw.to_csv(fp, index=False)
        assert len(RP.load_samples(fp, mmsi="ALL")) == len(raw), "ALL 不得按字面值筛空"
    try:
        RP.validate_holdout_disjointness(tr, va, va, purge_min=10)
    except ValueError:
        pass
    else:
        raise AssertionError("重复 validation/test 未被拒绝")

    # 正式真实联合接口必须同时覆盖 validation 和 test，不能只检查最终 test。
    va_real, te_real = va.copy(), te.copy()
    for d in (va_real, te_real):
        d["wind_error_e_ms"] = 0.0; d["wind_error_n_ms"] = 0.0; d["hs_error_m"] = 0.0
        d["actual_recovery_state"] = "低速"
    chk_real = RP.validate_holdout_disjointness(
        tr, va_real, te_real, purge_min=10, require_real_weather=True,
        require_real_recovery_state=True)
    assert chk_real["disjoint"] and not chk_real["independence_verified"]
    bad_va = va_real.drop(columns=["actual_recovery_state"])
    try:
        RP.validate_holdout_disjointness(
            tr, bad_va, te_real, purge_min=10, require_real_weather=True,
            require_real_recovery_state=True)
    except ValueError as exc:
        assert "validation" in str(exc) and "actual_recovery_state" in str(exc)
    else:
        raise AssertionError("validation 缺真实回收状态仍被正式接口接受")
    print(f"{OK} 0/23 上界={up:.3f}; 矩阵缺失拒绝；purged 防泄漏与validation/test真实联合通道门通过")



def t10_stage1_stage2_contracts():
    print("[T10] 阶段1/2: 终端能量唯一计量、统计支持与离散回收目标契约")
    p = M.apply_uav_profile(M.Params(), "L")
    E_to, E_land, T_to, T_land = M.to_land_energy_time(p)
    assert E_to > 0 and T_to > 0 and E_land == 0.0 and T_land == 0.0
    motion = dict(heave=0.5 * p.s_heave_max, roll=0.0, pitch=0.0)
    td, Ed = M.dock_reserve(p, motion, 0.0)
    rad = M.max_flight_radius_m(p, 60.0, s_motion=0.5)
    assert td > 0 and Ed > 0 and abs(rad["E_dock_env_Wh"] - Ed) < 1e-9
    print(f"{OK} 最终下降仅由 dock_reserve 计量；半径外包络复用同一 E_dock={Ed:.2f}Wh")

    H = [5, 10, 15, 20, 30]
    xi = RM._demo_xi(H, ["动力定位"])
    assert RM.decision_horizons_of(xi)[-1] == 30
    try:
        RM.decision_horizons_of(xi, [5, 35])
    except ValueError:
        pass
    else:
        raise AssertionError("统计区间外 h 未被拒绝")
    try:
        xi.get(12, "动力定位")
    except KeyError:
        pass
    else:
        raise AssertionError("XiAmbiguity.get 静默吸附到最近 horizon")
    try:
        xi.get_interp(10, "低速")
    except KeyError:
        pass
    else:
        raise AssertionError("XiAmbiguity 跨状态借格未被拒绝")
    mid = xi.get_interp(12.5, "动力定位")
    assert abs(float(mid.h_min) - 12.5) < 1e-12 and mid.c_state == "动力定位"

    tb = M.Turbine("T0", np.array([11.6, 54.5]), 68.5, 115.0)
    tb.local = np.array([1200.0, 0.0])
    wx = dict(Hs=0.2, Tp=5.0, wave_dir=0.0, ship_heading=0.0,
              wind10=2.0, wind_dir_from=180.0)
    ship_missing = RM.ShipPrediction.from_cv(np.zeros(2), np.zeros(2), H, "低速")
    d_missing = RM.route_feasible_at_h(RM.Route(1, [tb], ship_missing), 10, p, wx, xi)
    assert not d_missing["feasible"] and d_missing["reason"] == "missing_xi_support"

    ship_turn = RM.ShipPrediction.from_cv(np.zeros(2), np.zeros(2), H, "动力定位")
    ship_turn.recovery_state_by_h = {10: "转弯"}
    d_turn = RM.route_feasible_at_h(RM.Route(2, [tb], ship_turn), 10, p, wx, xi)
    assert not d_turn["feasible"] and d_turn["reason"] == "recovery_state_forbidden"
    ship_partial = RM.ShipPrediction.from_cv(np.zeros(2), np.zeros(2), H, "动力定位")
    ship_partial.recovery_state_by_h = {10: "动力定位"}
    d_state_missing = RM.route_feasible_at_h(RM.Route(20, [tb], ship_partial), 15, p, wx, xi)
    assert not d_state_missing["feasible"] and d_state_missing["reason"] == "missing_recovery_state_support"
    print(f"{OK} horizon 外推、跨状态/最近回收状态回退和转弯回收均 fail-closed")

    ship = RM.ShipPrediction.from_cv(np.zeros(2), np.zeros(2), H, "动力定位")
    route = RM.Route(3, [tb], ship)
    d = RM.route_feasible_at_h(route, 10, p, wx, xi, formal=True)
    assert d["recovery_target_model"] == "discrete_horizon_ship_prediction"
    assert d["terminal_sensor_error_mode"] == "out_of_scope"
    assert not any(k.startswith("acquisition_") for k in d), d.keys()
    assert not hasattr(p, "acquisition_radius_m") and not hasattr(p, "terminal_params_calibrated")
    assert "acquisition" not in RM.mission_risk_allocation(p, True)
    print(f"{OK} formal 回收目标由离散 h+预测船位+xi_h 定义；传感器级 acquisition error 不在有限模型")

def t11_stage3_data_contracts():
    print("[T11] 阶段3: 统计时长、purged 切分、样本门槛与确定性场景网格")
    assert S7.DEFAULT_HORIZONS_MIN == list(range(5, 61, 5))
    assert S8.DEFAULT_HORIZONS_MIN == list(range(5, 61, 5))
    assert S7.parse_horizons("60,5,10,10") == [5, 10, 60]
    assert S8.parse_horizons("5,15,60") == [5, 15, 60]

    # Pandas 3 可使用 datetime64[us]；epoch 秒转换必须显式归一化到 ns。
    ts_check = pd.Series(pd.array(["2025-03-01T00:00:35Z",
                                    "2025-03-01T00:01:05Z"],
                                   dtype="datetime64[us, UTC]"))
    epoch = S7.to_epoch_seconds_utc(ts_check)
    assert 1.7e9 < float(epoch[0]) < 1.8e9 and abs(float(epoch[1] - epoch[0]) - 30.0) < 1e-9
    assert pd.to_datetime(float(epoch[0]), unit="s", utc=True).year == 2025
    assert np.allclose(S8.to_seconds(ts_check), epoch)

    rows = []
    sid = 0
    base = 1_700_000_000.0
    # 每条轨迹独立时间轴，验证不是用全局分位数把不同船混切。
    for track, track_id, offset in (("track_same.csv", "/a/track_same.csv", 0.0),
                                    ("track_same.csv", "/b/track_same.csv", 2_000_000.0)):
        for i in range(240):
            t0 = base + offset + i * 600.0
            rows.append(dict(sample_id=sid, mmsi=str(track_id)[1], source_track=track,
                             source_track_id=track_id, segment_id=0,
                             predictor="cv_noleak", h_min=5, c_state="直航",
                             t0_epoch=t0, t1_epoch=t0 + 300.0,
                             xi_e_m=float(i % 5), xi_n_m=float(i % 7)))
            sid += 1
    raw = pd.DataFrame(rows)
    tr, va, te, meta = S7.purged_split_by_track(raw, [0.6, 0.2, 0.2], purge_min=10)
    assert len(meta) == 2 and set(tr.source_track_id) == {"/a/track_same.csv", "/b/track_same.csv"}
    assert set(va.source_track_id) == {"/a/track_same.csv", "/b/track_same.csv"}
    assert set(te.source_track_id) == {"/a/track_same.csv", "/b/track_same.csv"}
    for m in meta:
        assert m["train_n"] > 0 and m["validation_n"] > 0 and m["test_n"] > 0

    # 低速/DP 原格均不足 30，但预声明合并后 40；直航不足必须拒绝；转弯 35 原样通过。
    mrows = []
    for state, n in (("低速", 20), ("动力定位", 20), ("直航", 10), ("转弯", 35)):
        for i in range(n):
            mrows.append(dict(mmsi="1", source_track="track_A.csv", h_min=5, c_state=state,
                              t0_epoch=base + i * 60, xi_e_m=float(i % 3), xi_n_m=float(i % 4)))
    accepted, rejected = S7.summarize_with_contract(
        pd.DataFrame(mrows), min_cell_n=30, merge_policy="low_speed_pair", moments_source="train")
    accepted_all, _ = S7.summarize_with_contract(
        pd.DataFrame(mrows), min_cell_n=30, merge_policy="low_speed_pair", moments_source="all")
    accepted_overlap_all, _ = S7.summarize_with_contract(
        pd.DataFrame(mrows), min_cell_n=30, merge_policy="low_speed_pair",
        moments_source="train", overlap_policy="all", purge_min=0)
    acc = {(r["mmsi"], r["h_min"], r["c_state"]): r for r in accepted}
    assert acc[("1", 5, "低速")]["sample_rule"] == "merged_low_speed_pair"
    assert acc[("1", 5, "动力定位")]["n"] == 40
    assert acc[("1", 5, "转弯")]["sample_rule"] == "raw_state"
    assert any(r["mmsi"] == "1" and r["c_state"] == "直航" for r in rejected)
    assert accepted_all and not any(bool(r["valid_for_formal"]) for r in accepted_all)
    assert accepted_overlap_all and not any(bool(r["valid_for_formal"]) for r in accepted_overlap_all)
    assert np.array_equal(S8.deterministic_indices(10, 4), np.array([0, 3, 6, 9]))
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "xi.csv"
        formal_rows = []
        for h in range(5, 61, 5):
            formal_rows.append({
                "mmsi": "ALL", "h_min": h, "c_state": "直航", "n": 50,
                "mu_e_m": 0.0, "mu_n_m": 0.0,
                "sigma_ee": 4.0, "sigma_en": 0.0, "sigma_nn": 4.0,
                "max_norm_m": 8.0, "p95_norm_m": 6.0, "rms_norm_m": 3.0,
                "predictor": "cv_noleak", "predictor_contract": "cv_noleak_backward_window_epoch_seconds",
                "timestamp_epoch_contract": "utc_datetime64_ns_to_epoch_seconds",
                "moments_source": "train", "n_effective": 50, "purge_min": 60.0,
                "sample_overlap_policy": "nonoverlap",
                "sample_rule": "raw_state", "source_states": "直航",
                "state_merge_policy": "low_speed_pair",
                "min_cell_n": 30, "valid_for_formal": True,
                "t0_min_iso": "2025-01-01T00:00:00Z",
                "t0_max_iso": "2025-01-31T23:59:59Z",
            })
        pd.DataFrame(formal_rows).to_csv(fp, index=False)
        amb = M.XiAmbiguity.from_csv(fp, mmsi="ALL", formal=True)
        assert amb.horizons == list(range(5, 61, 5)) and amb.formal_validated
        assert amb.formal_horizon_grid_contract == M.XI_FORMAL_GRID_CONTRACT
        assert amb.covariance_contract == M.XI_COVARIANCE_CONTRACT

        # Formal Xi grid identity is binary64-exact: one ULP off-grid must be
        # rejected instead of round()/int() snapping back to a legal horizon.
        for bad_h in (math.nextafter(5.0, math.inf), math.nextafter(5.0, -math.inf)):
            bad_rows = [dict(r) for r in formal_rows]
            bad_rows[0]["h_min"] = bad_h
            bad_fp = Path(td) / f"offgrid_{bad_h.hex().replace('-', 'm')}.csv"
            pd.DataFrame(bad_rows).to_csv(bad_fp, index=False)
            try:
                M.XiAmbiguity.from_csv(bad_fp, mmsi="ALL", formal=True)
            except ValueError:
                pass
            else:
                raise AssertionError(f"formal Xi off-grid horizon accepted: {bad_h!r}")

        # Counts are discrete data-contract fields; fractional values must not
        # survive via later int() truncation.
        bad_rows = [dict(r) for r in formal_rows]
        bad_rows[0]["n"] = 50.5
        bad_rows[0]["n_effective"] = 50.5
        bad_fp = Path(td) / "fractional_count.csv"
        pd.DataFrame(bad_rows).to_csv(bad_fp, index=False)
        try:
            M.XiAmbiguity.from_csv(bad_fp, mmsi="ALL", formal=True)
        except ValueError:
            pass
        else:
            raise AssertionError("formal Xi fractional sample count accepted")

        # Purge must cover the largest formal horizon exactly; one ULP below
        # 60 min is still insufficient and must not be tolerance-accepted.
        bad_rows = [dict(r) for r in formal_rows]
        for r in bad_rows:
            r["purge_min"] = math.nextafter(60.0, -math.inf)
        bad_fp = Path(td) / "short_purge_ulp.csv"
        pd.DataFrame(bad_rows).to_csv(bad_fp, index=False)
        try:
            M.XiAmbiguity.from_csv(bad_fp, mmsi="ALL", formal=True)
        except ValueError:
            pass
        else:
            raise AssertionError("formal Xi one-ULP-short purge accepted")

        # Non-overlap thinning uses exact interval contact: t0==previous_t1 is
        # allowed; any t0<previous_t1, even one ULP, overlaps and is dropped.
        ov = pd.DataFrame([
            dict(mmsi="1", h_min=5, t0_epoch=0.0, t1_epoch=10.0),
            dict(mmsi="1", h_min=5, t0_epoch=math.nextafter(10.0, -math.inf), t1_epoch=20.0),
        ])
        assert len(S7.thin_nonoverlap_samples(ov)) == 1
        ov.loc[1, "t0_epoch"] = 10.0
        assert len(S7.thin_nonoverlap_samples(ov)) == 2

        # Previous validator tolerated small negative eigenvalues relative to a
        # huge matrix scale.  This indefinite covariance must fail closed.
        bad_rows = [dict(r) for r in formal_rows]
        bad_rows[0].update(sigma_ee=3.129784, sigma_en=2.985621e6, sigma_nn=3.129782e10)
        bad_fp = Path(td) / "indefinite_scale_masked.csv"
        pd.DataFrame(bad_rows).to_csv(bad_fp, index=False)
        try:
            M.XiAmbiguity.from_csv(bad_fp, mmsi="ALL", formal=True)
        except ValueError:
            pass
        else:
            raise AssertionError("formal Xi scale-masked indefinite covariance accepted")

        old_fp = Path(td) / "old.csv"
        pd.DataFrame([{
            "mmsi": "ALL", "h_min": 5, "c_state": "直航", "n": 50,
            "mu_e_m": 0, "mu_n_m": 0, "sigma_ee": 1, "sigma_en": 0, "sigma_nn": 1,
            "max_norm_m": 3, "p95_norm_m": 2, "rms_norm_m": 1
        }]).to_csv(old_fp, index=False)
        try:
            M.XiAmbiguity.from_csv(old_fp, mmsi="ALL", formal=True)
        except ValueError:
            pass
        else:
            raise AssertionError("旧 ξ CSV 缺正式数据契约仍被 formal=True 接受")
    p0 = M.apply_uav_profile(M.Params(), "S")
    xi0 = RM._demo_xi([5], ["动力定位"])
    zero = RA.solve_resource_master([], [], p0, xi0, K=1, T_min=60, batteries=0, cols_override=[])
    assert zero["covered"] == 0 and zero["batteries"] == 0 and zero["coverage_optimal"] \
        and zero["energy_optimal"] and zero["duplicate_turbine_visits"] == []
    print(f"{OK} 默认 h=5..60；逐轨迹 purged；低速/DP 预声明合并；薄格拒绝；grid 采样确定性；formal CSV/B=0 fail-closed")


def t12_stage4_resource_master():
    print("[T12] 阶段4: 实体 UAV/电池、快检、非甲板换电与精确 restricted master")
    p = M.Params()
    p.quick_inspection_min = 1.0
    p.quick_inspection_capacity = 1
    p.landing_clear_min = 1.0
    p.swap_station_capacity = 1
    p.battery_binding_mode = "horizon_fixed_uav"

    def c(tau, h, tid, e):
        return dict(tau=float(tau), h=float(h), tids=(str(tid),),
                    E0=float(e), E_plan_Wh=float(e), E_soc_required_Wh=float(e),
                    resource_only_test_column=True)

    # 同一 UAV + 同一电池：清场后 1 分钟快检即可继续，不占换电工位。
    quick_cols = [c(0, 10, "A", 100), c(14.5, 10, "B", 100)]
    rq = RA.solve_resource_master([], [], p, None, K=1, T_min=60, batteries=1,
                           cols_override=quick_cols, solver="scipy", allow_resource_only_columns=True,
                           t_launch_min=2.5, t_swap_min=4.0)
    assert rq["covered"] == 2 and rq["n_quick_reuses"] == 1 and rq["n_swaps"] == 0
    assert rq["battery_assignment"][0] == rq["battery_assignment"][1]
    assert rq["uav_assignment"][0] == rq["uav_assignment"][1]
    assert abs(rq["battery_energy_used_Wh"][0] - 200.0) < 1e-9

    # SOC 不足时必须换电；间隔不足 4 分钟时不能靠不同电池“瞬时切换”。
    short_cols = [c(0, 10, "A", 300), c(17.0, 10, "B", 300)]
    rs = RA.solve_resource_master([], [], p, None, K=1, T_min=60, batteries=2,
                           cols_override=short_cols, solver="scipy", allow_resource_only_columns=True,
                           t_launch_min=2.5, t_swap_min=4.0)
    assert rs["covered"] == 1 and rs["n_swaps"] == 0
    swap_cols = [c(0, 10, "A", 300), c(17.5, 10, "B", 300)]
    rw = RA.solve_resource_master([], [], p, None, K=1, T_min=60, batteries=2,
                           cols_override=swap_cols, solver="scipy", allow_resource_only_columns=True,
                           t_launch_min=2.5, t_swap_min=4.0)
    assert rw["covered"] == 2 and rw["n_swaps"] == 1 and rw["n_quick_reuses"] == 0
    assert rw["battery_assignment"][0] != rw["battery_assignment"][1]
    assert all(k in (None, 0) for k in rw["battery_binding"])

    # 两架 UAV 的换电事件重叠：单工位只能完成 3 个任务；双工位可完成 4 个。
    cap_cols = [c(0, 10, "A", 300), c(3, 10, "B", 300),
                c(17.5, 10, "C", 300), c(20.5, 10, "D", 300)]
    r1 = RA.solve_resource_master([], [], p, None, K=2, T_min=60, batteries=4,
                           cols_override=cap_cols, solver="scipy", allow_resource_only_columns=True,
                           t_launch_min=2.5, t_swap_min=4.0, swap_station_capacity=1)
    r2 = RA.solve_resource_master([], [], p, None, K=2, T_min=60, batteries=4,
                           cols_override=cap_cols, solver="scipy", allow_resource_only_columns=True,
                           t_launch_min=2.5, t_swap_min=4.0, swap_station_capacity=2)
    assert r1["covered"] == 3 and r1["resource_cuts"] >= 1
    assert r2["covered"] == 4 and r2["n_swaps"] == 2
    assert r2["restricted_master_certificate"] == "optimal" \
        and r2["coverage_optimal"] and r2["energy_optimal"]

    # 快检是独立容量资源：相邻着陆不冲突，但 2min 快检会重叠。
    p.quick_inspection_min = 2.0
    qcap_cols = [c(0, 10, "Q1", 100), c(3, 8, "Q2", 100),
                 c(15.5, 10, "Q3", 100), c(18.0, 10, "Q4", 100)]
    q1 = RA.solve_resource_master([], [], p, None, K=2, T_min=60, batteries=2,
                            cols_override=qcap_cols, solver="scipy", allow_resource_only_columns=True,
                            t_launch_min=2.5, t_swap_min=4.0, quick_inspection_capacity=1)
    q2 = RA.solve_resource_master([], [], p, None, K=2, T_min=60, batteries=2,
                            cols_override=qcap_cols, solver="scipy", allow_resource_only_columns=True,
                            t_launch_min=2.5, t_swap_min=4.0, quick_inspection_capacity=2)
    assert q1["covered"] == 3 and q1["resource_cuts"] >= 1
    assert q2["covered"] == 4 and q2["n_quick_reuses"] == 2
    p.quick_inspection_min = 1.0

    # 接地必须发生在 6 小时窗内；清场/周转可在窗外，但超窗接地列被过滤。
    late = RA.solve_resource_master([], [], p, None, K=1, T_min=60, batteries=1,
                             cols_override=[c(55, 10, "LATE", 100)], solver="scipy", allow_resource_only_columns=True)
    assert late["covered"] == 0 and late["pool_size"] == 0
    print(f"{OK} 快检复用/工位容量、SOC 强制换电、换电工位容量、UAV/电池身份与窗口边界全部通过")


def t13_stage5_replay_contracts():
    print("[T13] 阶段5: 离散回收目标、联合事件与统计上界")
    p = M.apply_uav_profile(M.Params(), "L")
    tb = M.Turbine("R0", np.array([11.6, 54.5]), 68.5, 115.0)
    tb.local = np.array([150.0, 0.0])
    ship = RM.ShipPrediction.from_cv(np.zeros(2), np.zeros(2), [10], "动力定位")
    ship.recovery_state_by_h = {10: "低速"}
    route = RM.Route(99, [tb], ship)
    wx = dict(Hs=0.1, Tp=5.0, wave_dir=0.0, ship_heading=0.0,
              wind10=1.0, wind_dir_from=180.0)
    base = 1_700_000_000.0
    df = pd.DataFrame([dict(mmsi="1", source_track_id="/x/track.csv", h_min=10,
                                c_state="动力定位", t0_epoch=base+i*120,
                                t1_epoch=base+i*120+600,
                                xi_e_m=1000.0, xi_n_m=0.0,
                                actual_recovery_state="低速")
                       for i in range(100)])
    rep = RP.replay_routes([(route, 10)], ship, p, wx, df)
    assert rep["n_test_total"] == 100 and rep["validation_complete"]
    assert rep["per_route"][0]["xi_state"] == "动力定位"
    assert rep["per_route"][0]["recovery_state"] == "低速"
    assert "acquisition" not in rep["required_events"]
    assert rep["upper95"] is not None and rep["protocol"] == "joint-route-replay"
    assert rep["ci_method"] == "bonferroni-simultaneous-max-per-sortie-hoeffding-azuma-conditional-risk-upper95"
    # 即使输入文件被人为加上旧 acquisition 列，也必须完全不改变当前有限模型回放。
    poisoned = df.copy(); poisoned["acq_error_e_m"] = 1.0e9; poisoned["acq_error_n_m"] = -1.0e9
    rep_poisoned = RP.replay_routes([(route, 10)], ship, p, wx, poisoned)
    assert rep_poisoned["viol_rate_any"] == rep["viol_rate_any"]
    assert "viol_rate_acquisition" not in rep_poisoned
    # 两条路线复用同一留出样本时，正式上界不能把 2n 个 route×sample 行当独立样本。
    rep_dup = RP.replay_routes([(route, 10), (route, 10)], ship, p, wx, df)
    assert rep_dup["upper95"] == rep_dup["per_route"][0]["simultaneous_upper95"]
    assert rep_dup["upper95"] == rep_dup["per_route"][1]["simultaneous_upper95"]
    assert rep_dup["ordinary_max_per_sortie_upper95"] <= rep_dup["upper95"] + 1e-12
    assert rep_dup["pooled_naive_upper95"] <= rep_dup["upper95"] + 1e-12

    _wunc = RM.WeatherUncertainty(wind_cov=np.zeros((2, 2)), hs_std=0.0,
                                  source="real-history-weather-speed-primary-coherent-noleak-residuals",
                                  formal_eligible=True)
    wamb = RM.WeatherAmbiguity(
        by_h={10: _wunc}, horizons=[10], source="real-history-weather-speed-primary-coherent-noleak-residuals",
        formal_eligible=True, predictor="weather_speed_primary_coherent_noleak",
        predictor_contract=RM.WEATHER_PREDICTOR_CONTRACTS["weather_speed_primary_coherent_noleak"],
        timestamp_epoch_contract=RM.WEATHER_TIMESTAMP_EPOCH_CONTRACT,
        truth_contract=RM.WEATHER_TRUTH_CONTRACT, weather_data_contract=RM.WEATHER_FORMAL_DATA_CONTRACT,
        sample_overlap_policy="weather_timeline_global_nonoverlap", purge_min=10.0,
        weather_source_sha256="a"*64, xi_train_source_sha256="b"*64)
    def _attach_weather_provenance(frame):
        frame = frame.copy()
        frame["wind_error_e_ms"] = 0.0; frame["wind_error_n_ms"] = 0.0; frame["hs_error_m"] = 0.0
        frame["weather_predictor"] = "weather_speed_primary_coherent_noleak"
        frame["weather_predictor_contract"] = RM.WEATHER_PREDICTOR_CONTRACTS["weather_speed_primary_coherent_noleak"]
        frame["weather_timestamp_epoch_contract"] = RM.WEATHER_TIMESTAMP_EPOCH_CONTRACT
        frame["weather_truth_contract"] = RM.WEATHER_TRUTH_CONTRACT
        frame["weather_data_contract"] = RM.WEATHER_FORMAL_DATA_CONTRACT
        frame["weather_valid_for_formal"] = True
        frame["weather_source_sha256"] = "a"*64
        frame["weather_train_source_sha256"] = "b"*64
        return frame
    real_weather = _attach_weather_provenance(df)
    _wrong_weather_source = real_weather.copy()
    _wrong_weather_source["weather_source_sha256"] = "c"*64
    try:
        RP.replay_routes([(route, 10)], ship, p, wx, _wrong_weather_source,
                         weather_unc=wamb, weather_sample_mode="real")
    except ValueError:
        pass
    else:
        raise AssertionError("real weather holdout with mismatched source SHA was accepted")

    formal = RP.replay_routes([(route, 10)], ship, p, wx, real_weather,
                              weather_unc=wamb, weather_sample_mode="real",
                              recovery_state_sample_mode="real",
                              holdout_disjointness_verified=True, confirmatory=True)
    assert formal["formal_reliability_claim_eligible"]
    # Integration regression for the step13 -> step15 boundary: step13 must pass the
    # complete WeatherAmbiguity, not a pre-resolved WeatherUncertainty cell.
    import tempfile
    xi_replay = M.XiAmbiguity({
        (10, "动力定位"): M.XiCell(10, "动力定位", 100, np.zeros(2),
                                      np.zeros((2, 2)), 0.0, 0.0, 0.0)}, [10])
    xi_replay.selected_mmsi = "1"
    xi_replay.cross_vessel_pooling = False
    with tempfile.TemporaryDirectory() as _td:
        _fp = Path(_td) / "real_joint.csv"
        _other = real_weather.iloc[[0]].copy()
        _other["mmsi"] = "2"
        pd.concat([real_weather, _other], ignore_index=True).to_csv(_fp, index=False)
        _out = S13._replay_columns(
            [dict(route=route, h=10, ship=ship, wx=wx, tids=("R0",))],
            p, xi_replay, p.eps_E, n_per=8, seed=3, wamb=wamb,
            validation_mode="real_validation", real_samples_csv=_fp,
            weather_sample_mode="real", holdout_disjointness_verified=False)
        assert _out["validation_complete"] and _out["n_replayed"] == 1 and _out["n_missing"] == 0
        assert _out["n_test_total"] == len(real_weather), (
            "step13 real replay mixed another MMSI into the selected-vessel holdout",
            _out["n_test_total"], len(real_weather))
    assert formal["evidence_scope"] == "confirmatory-purged-disjoint-real-joint-holdout-with-terminal-sensor-error-out-of-scope"
    turning = real_weather.copy(); turning["actual_recovery_state"] = "转弯"
    formal_turn = RP.replay_routes([(route, 10)], ship, p, wx, turning,
                                   weather_unc=wamb, weather_sample_mode="real",
                                   recovery_state_sample_mode="real",
                                   holdout_disjointness_verified=True, confirmatory=True)
    assert formal_turn["formal_reliability_claim_eligible"]
    assert formal_turn["viol_rate_recovery_state_gate"] == 1.0
    assert formal_turn["holds_upper95"] is False
    missing_state = real_weather.drop(columns=["actual_recovery_state"])
    formal_missing = RP.replay_routes([(route, 10)], ship, p, wx, missing_state,
                                      weather_unc=wamb, weather_sample_mode="real",
                                      recovery_state_sample_mode="auto",
                                      holdout_disjointness_verified=True, confirmatory=True)
    assert not formal_missing["formal_reliability_claim_eligible"]
    assert not formal_missing["validation_complete"]

    for budget in (0.01, 0.05, 0.10):
        pp = RP._params_for_mission_budget(M.Params(), budget, weather_on=True)
        assert RM.mission_eps_budget(pp, True) <= budget + 1e-15
        assert RM.mission_budget_compliant(pp, True)
        pp.validate_contract()
    print(f"{OK} 回收 h/xi 一体化、旧 acquisition 列无效、真实天气/回收状态和任务预算合同通过")

def t14_final_finite_solver_contracts():
    print("[T14] 最终有限模型: 按需精确定价 + 完备分支 + 资源审计证书")
    p = M.apply_uav_profile(M.Params(), "L")
    H = [10, 15, 20]
    xi = RM._demo_xi(H, ["动力定位"])
    sp = RM.ShipPrediction.from_cv(np.zeros(2), np.zeros(2), H, "动力定位")
    wx = dict(Hs=0.1, Tp=5.0, wave_dir=0.0, ship_heading=0.0,
              wind10=1.0, wind_dir_from=180.0)
    opt = RM.LaunchOption(0.0, sp, wx)
    turbines = []
    for i, x in enumerate((100.0, 250.0, 400.0)):
        tb = M.Turbine(f"G{i}", np.array([11.6, 54.5]), 68.5, 115.0)
        tb.local = np.array([x, 0.0])
        turbines.append(tb)
    exact = BP.solve_fleet_anytime(
        turbines, [opt], p, xi, K=1, T_min=60, batteries=2,
        max_stops=2, solver_mode="exact-branch-price-cut",
        pricing_mode="exact-implicit-dfs", solver="scipy", time_limit_s=30.0,
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0)
    assert exact["status"] == "lexicographic_optimal"
    assert exact["covered"] == 2 and exact["route_space_complete"] is False
    assert exact["coverage_optimal"] and exact["energy_optimal"]
    assert exact["lexicographic_optimal"] and exact["coverage_gap_abs"] == 0
    assert exact["pricing_complete"] and exact["branching_complete"]
    assert exact["farkas_pricing_complete"]
    assert exact["bound_scope"] == "global_discrete_physical_model"
    assert exact["algorithm"] == "branch-price-and-cut-with-logic-benders"
    assert exact["algorithmic_global_certificate"] is True
    assert exact["physical_model_global_certificate"] is True
    assert exact["global_certificate_available"] is True
    assert exact["route_universe_source"] == "physical-oracle"
    assert exact["route_universe_provenance_certified"] is True
    assert all("uav_id" in c and "battery_group" in c for c in exact["chosen"])

    try:
        BP.solve_fleet_anytime(
            turbines, [opt], p, xi, K=1, T_min=60, batteries=2,
            max_stops=2, solver_mode="exact-branch-price-cut",
            max_sequence_evals=1)
        raise AssertionError("exact path accepted an unsafe sequence cap")
    except ValueError as exc:
        assert "unsafe" in str(exc)

    baseline = BP.solve_fleet_anytime(
        turbines, [opt], p, xi, K=1, T_min=60, batteries=2,
        max_stops=2, solver_mode="research-baseline",
        max_sequence_evals=1, solver="scipy")
    assert baseline["global_certificate_available"] is False
    assert baseline["bound_scope"] == "validated_route_pool"
    assert baseline["pricing_complete"] is False
    print(f"{OK} 正式路径按需定价闭合；不安全扫描上限被拒绝；研究基线永不冒发全局证书")


def t15_final_experiment_protocol_contracts():
    print("[T15] 最终实验协议: formal fail-closed 与统计证书接口")
    import inspect
    sig = inspect.signature(S13._replay_columns)
    assert "holdout_disjointness_verified" in sig.parameters
    src = inspect.getsource(S13._replay_columns)
    assert "bonferroni-simultaneous-max-per-sortie" in src
    assert "formal_reliability_claim_eligible" in src
    assert "holdout_disjointness_verified=bool(holdout_disjointness_verified)" in src
    assert "weather_unc=wamb" in src, "step13 replay must carry full WeatherAmbiguity provenance into step15"
    assert "weather_unc=wu_cell" not in src and "weather_unc=wu," not in src

    import subprocess
    proc = subprocess.run(
        [sys.executable, "-m", "step13_experiment_model", "--exp", "E1_frontier",
         "--study-mode", "formal", "--allow-synth"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    assert proc.returncode != 0
    assert "formal" in proc.stdout.lower() and "allow-synth" in proc.stdout
    print(f"{OK} 正式协议禁止合成替代；独立留出标记传入逐架次联合回放")



def t16_audit_regressions():
    print("[T16] 审计回归: SOC安全列空间、κ绑定、严格统计与图表证据门")
    from unittest.mock import patch
    import tempfile
    import step16_visualize as V16
    import step17_paper_figure as V17

    # P0-1: 相同主问题时序/覆盖但SOC需求不同的访问顺序不得被能耗单维去重。
    p = M.apply_uav_profile(M.Params(), "L")
    xi = RM._demo_xi([10], ["动力定位"])
    ship = RM.ShipPrediction.from_cv(np.zeros(2), np.zeros(2), [10], "动力定位")
    opt = RM.LaunchOption(0.0, ship, dict(Hs=0.1, Tp=5.0, wave_dir=0.0,
                                         ship_heading=0.0, wind10=1.0,
                                         wind_dir_from=180.0))
    class FakeRoute:
        def __init__(self, rid, order):
            self.rid = rid; self.ship = ship; self.fixed_h = 10.0; self._order = order
        def turbine_ids(self):
            return list(self._order)
    r_low_plan_bad_soc = FakeRoute(1, ("A", "B"))
    r_high_plan_good_soc = FakeRoute(2, ("B", "A"))
    expanded = [
        (r_low_plan_bad_soc, dict(h=10.0, E_plan_Wh=100.0,
                                  E_soc_required_Wh=500.0, E0=90.0,
                                  recovery_state="动力定位",
                                  recovery_state_source="declared-test-predictor")),
        (r_high_plan_good_soc, dict(h=10.0, E_plan_Wh=120.0,
                                    E_soc_required_Wh=200.0, E0=110.0,
                                    recovery_state="动力定位",
                                    recovery_state_source="declared-test-predictor")),
    ]
    raw_columns = [
        dict(tids=("A", "B"), ordered_tids=("A", "B"), tau=0.0, h=10.0,
             E_plan_Wh=100.0, E_soc_required_Wh=500.0, E0=100.0),
        dict(tids=("A", "B"), ordered_tids=("B", "A"), tau=0.0, h=10.0,
             E_plan_Wh=120.0, E_soc_required_Wh=200.0, E0=120.0),
    ]
    cols = FAC.validate_route_columns(raw_columns)
    assert len(cols) == 2 and {c["E_soc_required_Wh"] for c in cols} == {200.0, 500.0}
    assert {tuple(c["ordered_tids"]) for c in cols} == {("A", "B"), ("B", "A")}
    assert math.isfinite(RM.KAPPA_MODES["vp_unimodal"](0.1))


    # P0-3: 直接构造但不提供逐h状态预测时必须拒绝，不得吸附起飞状态。
    tb = M.Turbine("S0", np.array([11.6, 54.5]), 68.5, 115.0); tb.local = np.array([100.0, 0.0])
    bare = RM.ShipPrediction(P_launch=np.zeros(2), pred_by_h={10: np.zeros(2)}, c_state="低速")
    d = RM.route_feasible_at_h(RM.Route(301, [tb], bare), 10, p, opt.wx, xi)
    assert not d["feasible"] and d["reason"] == "missing_recovery_state_support"

    # P0-5: duplicate key / non-PSD /非法状态合并均由formal读取器独立拒绝。
    def valid_rows():
        out=[]
        for h in range(5, 61, 5):
            out.append(dict(mmsi="ALL", h_min=h, c_state="直航", n=50,
                            mu_e_m=0.0, mu_n_m=0.0, sigma_ee=4.0,
                            sigma_en=0.0, sigma_nn=4.0, max_norm_m=8.0,
                            p95_norm_m=6.0, rms_norm_m=3.0,
                            predictor="cv_noleak", predictor_contract="cv_noleak_backward_window_epoch_seconds",
                            timestamp_epoch_contract="utc_datetime64_ns_to_epoch_seconds",
                            moments_source="train", n_effective=50, purge_min=60.0,
                            sample_overlap_policy="nonoverlap",
                            sample_rule="raw_state", source_states="直航",
                            state_merge_policy="low_speed_pair", min_cell_n=30,
                            valid_for_formal=True,
                            t0_min_iso="2025-01-01T00:00:00Z",
                            t0_max_iso="2025-01-31T23:59:59Z"))
        return out
    with tempfile.TemporaryDirectory() as td:
        td=Path(td)
        bad=valid_rows(); bad.append(dict(bad[0]))
        fp=td/"dup.csv"; pd.DataFrame(bad).to_csv(fp,index=False)
        try: M.XiAmbiguity.from_csv(fp,mmsi="ALL",formal=True)
        except ValueError: pass
        else: raise AssertionError("formal ξ duplicate key accepted")
        bad=valid_rows(); bad[0]["sigma_ee"]=-1.0
        fp=td/"psd.csv"; pd.DataFrame(bad).to_csv(fp,index=False)
        try: M.XiAmbiguity.from_csv(fp,mmsi="ALL",formal=True)
        except ValueError: pass
        else: raise AssertionError("formal ξ non-PSD covariance accepted")
        bad=valid_rows(); bad[0].update(sigma_ee=3.129784, sigma_en=2.985621e6, sigma_nn=3.129782e10)
        fp=td/"psd_scale_mask.csv"; pd.DataFrame(bad).to_csv(fp,index=False)
        try: M.XiAmbiguity.from_csv(fp,mmsi="ALL",formal=True)
        except ValueError: pass
        else: raise AssertionError("formal ξ scale-relative PSD tolerance accepted indefinite covariance")
        bad=valid_rows(); bad[0]["h_min"] = math.nextafter(5.0, math.inf)
        fp=td/"offgrid_ulp.csv"; pd.DataFrame(bad).to_csv(fp,index=False)
        try: M.XiAmbiguity.from_csv(fp,mmsi="ALL",formal=True)
        except ValueError: pass
        else: raise AssertionError("formal ξ one-ULP off-grid horizon was snapped")
        bad=valid_rows(); bad[0]["source_states"]="直航+转弯"; bad[0]["sample_rule"]="merged_low_speed_pair"
        fp=td/"merge.csv"; pd.DataFrame(bad).to_csv(fp,index=False)
        try: M.XiAmbiguity.from_csv(fp,mmsi="ALL",formal=True)
        except ValueError: pass
        else: raise AssertionError("formal ξ illegal state merge accepted")

        # P0-7: 缺失结果合同与点估计回退均不得进入正式图。
        old=td/"old.csv"; pd.DataFrame([{"method":"DRCC","emp_viol":0.01}]).to_csv(old,index=False)
        try: V17._read(old, "old-test")
        except (ValueError, SystemExit): pass
        else: raise AssertionError("paper figure accepted missing result contract")
        try: V16._require_result_contract(pd.read_csv(old), "old-test", {"result_contract"})
        except ValueError: pass
        else: raise AssertionError("visualization accepted missing result contract")

        # 正式布尔证据字段缺失或损坏时必须拒绝，不能把 NaN 当作通过。
        for vals in ([True, np.nan], [True, "unknown"]):
            try: V17._strict_bool(pd.Series(vals), "audit-test")
            except SystemExit: pass
            else: raise AssertionError("paper figure accepted missing/invalid formal boolean")

    # P0-6: 无物理路线的旧列默认拒绝；仅显式测试开关可用于资源反例。
    bad_col=dict(tau=0.0,h=10.0,tids=("X",),E0=10.0,E_plan_Wh=10.0,E_soc_required_Wh=10.0)
    rr=RA.solve_resource_master([],[],p,None,K=1,T_min=60,batteries=1,
                         cols_override=[bad_col],solver="scipy")
    assert rr["pool_size"] == 0 and rr["override_columns_rejected"] == 1
    print(f"{OK} SOC不同列不去重；κ显式绑定；缺回收状态、非法ξ、旧图表与旧列全部 fail-closed")

def t17_final_concept_regressions():
    print("[T17] 最终构想回归: SOC下界、真实转弯、资源预筛、证书签名与E2冻结")
    # 有利均值可提高DRCC余量，但不得让实体电池预留低于完整计划能耗。
    p = M.apply_uav_profile(M.Params(), "L")
    sp = RM.ShipPrediction.from_cv(np.zeros(2), np.zeros(2), [20], "动力定位")
    sp.recovery_state_by_h = {20: "低速"}
    tb = M.Turbine("SOC", np.zeros(2), 68.5, 115.0); tb.local = np.array([3000.0, 0.0])
    wx = dict(Hs=0.1, Tp=5.0, wave_dir=0.0, ship_heading=0.0,
              wind10=1.0, wind_dir_from=180.0)
    cell = M.XiCell(20, "动力定位", 100, np.array([-1000.0, 0.0]),
                    np.zeros((2, 2)), 0.0, 0.0, 0.0)
    xi = M.XiAmbiguity({(20, "动力定位"): cell}, [20])
    d = RM.route_feasible_at_h(RM.Route(1701, [tb], sp), 20, p, wx, xi)
    assert d["feasible"] and d["E_soc_required_Wh"] >= d["E_plan_Wh"] - 1e-9
    assert abs(d["E_uncertainty_buffer_Wh"]
               - (d["E_soc_required_Wh"] - d["E_plan_Wh"])) < 1e-9

    # t_swap < t_quick 时，预筛只能使用安全放松，不得删除可由备用电池换电完成的链。
    pr = M.Params(); pr.quick_inspection_min = 10.0
    def rc(tau, h, tid):
        return dict(tau=float(tau), h=float(h), tids=(tid,), E0=100.0,
                    E_plan_Wh=100.0, E_soc_required_Wh=100.0,
                    resource_only_test_column=True)
    rr = RA.solve_resource_master([], [], pr, None, K=1, T_min=60, batteries=2,
                           cols_override=[rc(0, 10, "A"), rc(12, 10, "B")],
                           t_launch_min=0.0, landing_clear_min=1.0, t_swap_min=1.0,
                           solver="scipy", allow_resource_only_columns=True)
    assert rr["covered"] == 2 and rr["n_swaps"] == 1

    # 证书模型身份必须随资源参数变化，且ξ支持半径使用正式字段。
    opt = RM.LaunchOption(0.0, sp, wx)
    cfg1 = dict(deck_delta_min=2.5, t_swap_min=4.0, t_launch_min=2.5,
                landing_clear_min=1.0, quick_inspection_min=1.0,
                quick_inspection_capacity=1, swap_station_capacity=1,
                battery_reuse_mode="exact_soc", battery_binding_mode="horizon_fixed_uav")
    cfg2 = dict(cfg1, t_swap_min=1.0, swap_station_capacity=2)
    sig1 = BP._finite_model_scope_signature([tb], [opt], p, xi, K=1, batteries=2,
                                             T_min=60, max_stops=1,
                                             kappa_mode="vp_unimodal", weather_unc=None,
                                             deck_mode="interval", pool_h_mode="pareto",
                                             resource_config=cfg1)
    sig2 = BP._finite_model_scope_signature([tb], [opt], p, xi, K=1, batteries=2,
                                             T_min=60, max_stops=1,
                                             kappa_mode="vp_unimodal", weather_unc=None,
                                             deck_mode="interval", pool_h_mode="pareto",
                                             resource_config=cfg2)
    assert sig1["sha256"] != sig2["sha256"]
    assert "support_radius" in sig1["xi_cells"][0] and "max_norm" not in sig1["xi_cells"][0]

    # E2冻结只能来自最严配置分位的validation安全候选，test不参与排名。
    e2 = pd.DataFrame([
        dict(run_status="ok", q=0.2, criterion="easy", holds=True,
             n_missing_replay=0, covered=20, safe_served=20, energy_per_safe=10, energy_Wh=200),
        dict(run_status="ok", q=0.8, criterion="unsafe", holds=False,
             n_missing_replay=0, covered=20, safe_served=20, energy_per_safe=9, energy_Wh=180),
        dict(run_status="ok", q=0.8, criterion="safe", holds=True,
             n_missing_replay=0, covered=18, safe_served=18, energy_per_safe=11, energy_Wh=198),
    ])
    pick = S13._select_e2_validation_candidate(e2, (0.2, 0.8))
    assert pick is not None and pick["criterion"] == "safe" and float(pick["q"]) == 0.8
    assert S13._select_e2_validation_candidate(e2[e2.criterion != "safe"], (0.2, 0.8)) is None

    # final test绑定的是精确路线+资源计划，而不是只绑定 criterion/q。
    fp_cols = [dict(tau=0.0, h=10.0, tids=("A",), E_plan_Wh=100.0,
                    E_soc_required_Wh=110.0, uav_id=0, battery_group=0,
                    turnaround_before={"predecessor": None, "mode": "initial", "interval": None},
                    post_service_mode="none_after_last_mission", post_service_interval=None)]
    fp1 = S13._frozen_plan_fingerprint(fp_cols)
    fp2 = S13._frozen_plan_fingerprint([dict(fp_cols[0])])
    changed = dict(fp_cols[0]); changed["battery_group"] = 1
    fp3 = S13._frozen_plan_fingerprint([changed])
    assert fp1 == fp2 and fp1 != fp3
    print(f"{OK} SOC≥Eplan；快换电链保留；资源参数绑定证书；E2冻结validation安全候选与精确计划指纹")



def t18_frozen_final_test_protocols():
    print("[T18] E1/E2 精确计划冻结与 final test 一次性消费")
    import tempfile
    out = Path(tempfile.mkdtemp(prefix="selftest_frozen_protocol_"))
    train = out / "train.csv"; validation = out / "validation.csv"; test = out / "test.csv"
    for path in (train, validation, test):
        path.write_text("x\n1\n", encoding="utf-8")
    chosen = [dict(
        tau=0.0, h=10.0, tids=("A",), E0=100.0, E_plan_Wh=100.0,
        E_soc_required_Wh=110.0, uav_id=0, battery_group=0,
        turnaround_before={"predecessor": None, "mode": "initial", "interval": None},
        post_service_mode="none_after_last_mission", post_service_interval=None)]

    def result(B=1):
        return dict(chosen=[dict(chosen[0])], covered=1, coverable=1, pool_size=1,
                    batteries=int(B), flights=1, mean_stops=1.0, multi_stop_ratio=0.0,
                    energy_Wh=100.0, makespan_min=10.0, status="lexicographic_optimal",
                    coverage_incumbent=1, coverage_upper_bound=1, coverage_gap_abs=0,
                    coverage_gap_pct=0.0, coverage_optimal=True,
                    coverage_global_certificate_available=True,
                    coverage_physical_model_certificate=True,
                    coverage_algorithmic_certificate=True,
                    energy_incumbent_Wh=100.0, energy_lower_bound_Wh=100.0,
                    energy_gap_abs_Wh=0.0, energy_gap_pct=0.0, energy_optimal=True,
                    lexicographic_optimal=True, solve_scope="lexicographic",
                    global_certificate_available=True, global_route_space_certificate=True,
                    implicit_route_space_certified=True, solver="unit")

    calls = []
    def replay(cols, p, xi, eps, **kw):
        mode = kw.get("validation_mode"); calls.append(mode)
        is_final = mode == "real_joint_final_test"
        return dict(
            safe_tids={"A"}, emp=0.0, max_col_viol=0.0, n_replayed=1, n_missing=0,
            n_test_total=100, n_viol_total=0, ci95_low=0.0, ci95_high=0.0,
            upper95=0.0, n_realized=1, n_realized_viol=0,
            allocation_budget=0.045, mission_requirement_budget=0.05,
            allocation_budget_holds=True, mission_requirement_holds=True,
            all_routes_allocation_holds=True, all_routes_mission_holds=True,
            validation_gate_contract=(
                "selection-gate-internal-allocation-budget-v14-exact-qmax-once-only;"
                "per-sortie-bonferroni-retained;"
                "event-fingerprints-audit-only;"
                "mission-0.05-reported-separately"),
            validation_event_fingerprint_count=1,
            validation_unique_event_fingerprint_count=1,
            validation_duplicate_event_groups=0,
            validation_event_group_sizes_json="[1]",
            validation_event_grouping_used_for_gate=False,
            realized_ci95_low=0.0, realized_ci95_high=0.0, realized_upper95=0.0,
            validation_type=str(mode), disjoint_xi_holdout=is_final,
            disjoint_weather_holdout=is_final, disjoint_real_holdout=is_final,
            independent_xi_holdout=False, independent_weather_holdout=False, independent_real_holdout=False,
            ci_method="bonferroni", formal_reliability_claim_eligible=is_final,
            validation_plan_fingerprint=S13._frozen_plan_fingerprint(cols),
            route_validation_records=[dict(
                tau=float(cols[0].get("tau", 0.0)), h=float(cols[0].get("h", 0.0)),
                ordered_tids=list(map(str, cols[0].get("tids", ()))),
                n_test_total=100, n_viol_any=0, viol_rate_any=0.0,
                validation_complete=True)],
            evidence_scope=("confirmatory-purged-disjoint-real-joint-holdout" if is_final else "validation-selection-only-no-formal-inference"))

    old = (RA.build_route_columns, RA.solve_resource_master, BP.solve_fleet_anytime,
           S13._replay_columns, S13._e1_detail_rows, S13.build_launch_options)
    try:
        RA.build_route_columns = lambda *a, **k: [dict(chosen[0])]
        RA.solve_resource_master = lambda *a, **k: result(k.get("batteries", 1))
        BP.solve_fleet_anytime = lambda *a, **k: result(k.get("batteries", 1))
        S13._replay_columns = replay
        S13._e1_detail_rows = lambda *a, **k: [dict(route_id=0, uav="S")]

        e1dir = out / "e1"
        a1 = Namespace(
            e1_uavs="S", fleet_ks="1", e1_batteries="1,2,3", e1_b_auto="off",
            e1_b_cap=3, e1_sat_patience=2, stops_cap="1", max_stops=1,
            deck_delta_min=0.0, deck_mode="interval", dtau_min=5.0,
            t_swap_min=1.0, t_launch_min=0.0, landing_clear_min=0.0,
            swap_stations=1, quick_inspection_capacity=1,
            battery_reuse_mode="exact_soc", replay_n=20, allow_synth=False,
            track_start_min=None, uav="S", knee_frac=0.95, knee_order="BK",
            selection_metric="safe_per_inventory_kWh", validation_mode="real_validation",
            validation_samples=validation, xi_train_samples=train, final_test_samples=test,
            final_weather_mode="real", resume="off", pool_h="pareto",
            final_solver_solver_mode="exact_enumeration", max_sequence_evals=100,
            study_mode="formal", allow_final_test_rerun=False,
            _holdout_disjointness_verified=True, _xi_source="unit",
            _pair_radius_m=1.0, _pair_radius_mode="unit", _wx0={},
            weather_alignment="timestamp", _resolved_track_csv=None)
        S13.E1_frontier([], [], M.Params(), None, None, e1dir, a1, "unit", 60.0)
        # v9 formal protocol: E1 may revalidate/freeze on validation, but it
        # must never consume the independent final test.
        assert sum(x == "real_joint_final_test" for x in calls) == 0
        assert not (e1dir / "E1_final_test.csv").exists()
        e1sel = pd.read_csv(e1dir / "E1_selection.csv")
        if "final_test_deferred_to_e2" in e1sel.columns:
            assert bool(e1sel["final_test_deferred_to_e2"].fillna(False).any())
        calls.clear()
        S13.E1_frontier([], [], M.Params(), None, None, e1dir, a1, "unit", 60.0)
        assert sum(x == "real_joint_final_test" for x in calls) == 0
        assert not (e1dir / "E1_final_test.csv").exists()

        calls.clear()
        S13.build_launch_options = lambda *a, **k: ([], [], "unit", 60.0,
                                                  {"Hs": 0.1, "wind10": 1.0})
        e2dir = out / "e2"
        a2 = Namespace(
            e2_quantiles="0.8", k=1, uav="S", t_swap_min=1.0, t_launch_min=0.0,
            stops_cap="1", max_stops=1, batteries=1, e2_criteria="vp",
            deck_delta_min=0.0, deck_mode="interval", replay_n=20, dtau_min=5.0,
            _pair_radius_m=1.0, _xi_source="unit", soc_correction="geo2d",
            recovery_predictor="cv_noleak", pool_h="pareto",
            validation_mode="real_validation", validation_samples=validation,
            study_mode="formal", quick_inspection_capacity=1, swap_stations=1,
            battery_reuse_mode="exact_soc", resume="off", window_min=60.0,
            track_start_min=None, allow_synth=False, infarm_radius=None,
            landing_clear_min=0.0, final_test_samples=test, final_weather_mode="real",
            _holdout_disjointness_verified=True, allow_incomplete_results=False,
            allow_final_test_rerun=False, xi_train_samples=train,
            _resolved_track_csv=None, weather_alignment="timestamp",
            max_sequence_evals=100, solver_mode="auto",
            _e1_formal_freeze_verified=True, _e1_formal_freeze_sha256="b"*64,
            _e1_config_source="unit-auto-e1-freeze",
            _formal_sample_hashes_verified=True,
            _formal_sample_hashes_sha256={
                "train": EU.sha256_file(train), "validation": EU.sha256_file(validation),
                "test": EU.sha256_file(test)})
        p2 = M.Params()
        xi2 = RM._demo_xi([10], ["动力定位"])
        S13.E2_robust([], None, None, xi2, None, p2, None, e2dir, a2)
        first2 = sum(x == "real_joint_final_test" for x in calls)
        e2rec = pd.read_csv(e2dir / "E2_final_test.csv").iloc[0]
        assert first2 == 1 and int(e2rec.final_test_invocations) == 1
        calls.clear()
        S13.E2_robust([], None, None, xi2, None, p2, None, e2dir, a2)
        e2rec2 = pd.read_csv(e2dir / "E2_final_test.csv").iloc[0]
        assert sum(x == "real_joint_final_test" for x in calls) == 0
        assert int(e2rec2.final_test_invocations) == 1
    finally:
        (RA.build_route_columns, RA.solve_resource_master, BP.solve_fleet_anytime,
         S13._replay_columns, S13._e1_detail_rows, S13.build_launch_options) = old
    print(f"{OK} v9: E1只在validation冻结，final test仅由E2/A最终冻结后一次性消费")


def t19_zero_pool_model_fixes():
    print("[T19] 零列池模型修复：标量风门、逐腿空速、MMSI ξ、CV回收状态与时间分解")

    # 1) 着舰标量事件不得再由二维 trace 范数半径控制。
    wu = RM.WeatherUncertainty(
        wind_cov=np.diag([4.0, 4.0]), wind_bias=np.zeros(2),
        wind_speed_std=0.10, wind_speed_bias=0.02)
    scalar_shift = RM.wind_speed_upper_shift("drcc", wu, 0.005)
    vector_radius = RM.wind_delta_radius("drcc", wu, 0.005)
    assert scalar_shift < 2.0 and vector_radius > 20.0, (scalar_shift, vector_radius)

    # 2) 逐腿空速裕度应随航向/投影变化，不能再全部固定为 v_max-v_cr-r_norm。
    p0 = M.Params()
    cov_aniso = RM.WeatherUncertainty(
        wind_cov=np.diag([1.0, 0.0025]), wind_bias=np.zeros(2),
        wind_speed_std=0.1)
    rec_e = dict(leg_index=0, is_return=False, direction=np.array([1.0, 0.0]),
                 normal=np.array([0.0, 1.0]), wind_along_ms=2.0,
                 wind_cross_signed_ms=0.0)
    rec_n = dict(leg_index=0, is_return=False, direction=np.array([0.0, 1.0]),
                 normal=np.array([-1.0, 0.0]), wind_along_ms=0.0,
                 wind_cross_signed_ms=-2.0)
    de = RM.route_airspeed_projection_check(
        {"leg_air_records": [rec_e], "speed_feasible": True,
         "max_required_airspeed_ms": p0.v_cr}, p0, cov_aniso, 0.005)
    dn = RM.route_airspeed_projection_check(
        {"leg_air_records": [rec_n], "speed_feasible": True,
         "max_required_airspeed_ms": p0.v_cr}, p0, cov_aniso, 0.005)
    assert abs(de["margin_ms"] - dn["margin_ms"]) > 1e-6, (de, dn)

    # 3) CV 预测下回收状态由预测运动推导；起飞转弯不再机械持续。
    sp = RM.ShipPrediction.from_cv(np.zeros(2), np.array([2.0, 0.0]), [5, 10], "转弯")
    assert sp.recovery_state_at(5)[0] == "直航"
    assert "predicted-motion" in sp.recovery_state_at(5)[1]

    # 4) MMSI 分层 ξ 优先具体船，并保留 ALL 收缩来源。
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        xp = Path(td) / "xi.csv"
        rows = []
        for mm, mu, n in (("ALL", 100.0, 1000), ("123", 10.0, 500)):
            rows.append(dict(mmsi=mm, h_min=5, c_state="动力定位", n=n,
                             mu_e_m=mu, mu_n_m=0.0, sigma_ee=100.0,
                             sigma_en=0.0, sigma_nn=100.0, max_norm_m=500.0,
                             p95_norm_m=200.0, rms_norm_m=120.0,
                             predictor="cv_noleak",
                             predictor_contract="cv_noleak_backward_window_epoch_seconds",
                             timestamp_epoch_contract="utc_datetime64_ns_to_epoch_seconds",
                             sample_overlap_policy="all", valid_for_formal=False))
        pd.DataFrame(rows).to_csv(xp, index=False)
        xa = M.XiAmbiguity.from_csv_hierarchical(xp, "123", shrinkage_equivalent_n=100.0)
        c = xa.get(5, "动力定位")
        assert xa.selected_mmsi == "123" and 10.0 <= c.mu[0] < 30.0, c.mu
        assert "exact-shrunk" in xa.hierarchical_sources[(5, "动力定位")]
        assert xa.predictor == "cv_noleak"
        assert xa.predictor_contract == "cv_noleak_backward_window_epoch_seconds"
        assert xa.timestamp_epoch_contract == "utc_datetime64_ns_to_epoch_seconds"

    # 5) persistence proxy 的 floor 仅作为有限样本收缩目标，不再永久硬截断每个分量。
    times = pd.date_range("2026-01-01", periods=200, freq="h", tz="UTC")
    wdf = pd.DataFrame({"time": times, "wind10": np.full(len(times), 4.0),
                        "wind_dir_from": np.full(len(times), 180.0),
                        "Hs": np.full(len(times), 0.5)})
    wa = RM.weather_ambiguity_from_series(wdf, [5])
    wc = wa.get(5)
    assert float(np.max(np.diag(wc.wind_cov))) < 0.25
    assert hasattr(wc, "wind_speed_std") and wc.source == "adjacent_reanalysis_difference_proxy"

    # 6) 时间 DRCC 必须输出可核对分解，且总收紧与最终裕度闭合。
    tb = _tb_at("T", 800.0, 0.0)
    ship = RM.ShipPrediction.from_cv(np.zeros(2), np.zeros(2), [30], "DP")
    route = RM.Route(-1, [tb], ship)
    xi = M.XiAmbiguity({(30, "DP"): M.XiCell(30, "DP", 1000, np.zeros(2),
                                               np.diag([25.0, 25.0]), 100.0, 50.0, 30.0)}, [30])
    dd = RM.route_feasible_at_h(route, 30, p0, _WX_CALM, xi,
                                weather_unc=RM.WeatherUncertainty(
                                    wind_cov=np.diag([0.01, 0.01]),
                                    wind_speed_std=0.05), chance_mode="drcc")
    tdg = dd["time_decomposition"]
    assert abs(tdg["nominal_margin_s"] - tdg["total_tightening_s"]
               - tdg["final_margin_s"]) <= 1e-6, tdg
    assert abs(tdg["total_tightening_s"] - (tdg["xi_mean_shift_s"]
               + tdg["xi_std_term_s"] + tdg["weather_mean_shift_s"]
               + tdg["weather_std_term_s"] + tdg["geometry_remainder_s"])) <= 1e-6, tdg
    assert abs(tdg["xi_geo_total_s"] - (tdg["xi_mean_shift_s"]
               + tdg["xi_std_term_s"] + tdg["geometry_remainder_s"])) <= 1e-6, tdg
    assert abs(tdg["eps_time_xi"] + tdg["eps_time_weather"] - p0.eps_T) <= 1e-12
    print(f"{OK} 标量/矢量随机变量分离、逐腿可控性、MMSI分层与时间证据链全部通过")



def t20_fixed_touchdown_wait_recourse_contract():
    """Regression tests for fixed_touchdown_wait_recourse and dock/energy coupling."""
    tol = RM.TIME_TOL_S
    # 1) waiting absorbs risk delay
    a = RM.fixed_touchdown_time_accounting(1500.0, 1200.0, 100.0)
    assert a["time_safe_core_s"] == 1300.0 and a["time_drcc_margin_s"] == 200.0
    assert a["time_wait_safe_s"] == 200.0 and not a["time_drcc_failed"]
    # 2) delay exceeds waiting
    b = RM.fixed_touchdown_time_accounting(1500.0, 1200.0, 350.0)
    assert b["time_safe_core_s"] == 1550.0 and b["time_drcc_margin_s"] == -50.0
    assert b["time_drcc_failed"]
    # 3) zero-risk consistency
    for core in (1200.0, 1500.0, 1500.1):
        z = RM.fixed_touchdown_time_accounting(1500.0, core, 0.0)
        assert z["nominal_time_failed"] == z["time_drcc_failed"]
    # 4) waiting never enters DRCC left-hand side
    assert abs(a["time_safe_core_s"] - (a["time_core_nom_s"] + a["time_drcc_tightening_s"])) <= tol
    assert abs(a["time_safe_core_s"] - (a["time_core_nom_s"] + a["time_wait_nom_s"]
                                        + a["time_drcc_tightening_s"])) > 1.0
    # 5) h monotonicity
    assert RM.fixed_touchdown_time_accounting(1600.0, 1200.0, 100.0)["time_drcc_margin_s"] \
           >= a["time_drcc_margin_s"]
    # 6/7) xi/weather variance cannot improve margin
    eps = 0.01
    g = np.array([0.2, 0.0])  # seconds/metre
    def _margin(sig_x, sig_w):
        tight = RM.kappa(eps) * math.sqrt(float(g @ np.diag([sig_x, sig_x]) @ g) + sig_w)
        return RM.fixed_touchdown_time_accounting(1500.0, 1200.0, tight)["time_drcc_margin_s"]
    assert _margin(400.0, 0.0) <= _margin(100.0, 0.0) + tol
    assert _margin(100.0, 16.0) <= _margin(100.0, 4.0) + tol
    # 8) a risk-adjusted dock belongs to core once; no separate dock risk is added.
    dock_safe = 80.0
    c = RM.fixed_touchdown_time_accounting(1500.0, 1100.0 + dock_safe, 100.0)
    assert c["time_safe_core_s"] == 1280.0
    assert RM.DOCK_RISK_CONTRACT == "risk_adjusted_dock_in_core_no_extra_dock_risk"
    # 9/10) deterministic planning/replay and fixed touchdown consistency
    plan = RM.fixed_touchdown_time_accounting(1500.0, 1300.0, 0.0)
    rep = RM.realized_fixed_touchdown_time(1500.0, 1300.0)
    assert plan["time_drcc_failed"] == rep["time_violation"]
    assert rep["scheduled_touchdown_s"] == 1500.0 and rep["realized_wait_s"] == 200.0
    # 11) nominal overrun has zero waiting
    late = RM.fixed_touchdown_time_accounting(1500.0, 1501.0, 0.0)
    assert late["nominal_time_failed"] and late["time_wait_nom_s"] == 0.0
    # 12) decomposition balance plus physical energy branch max, not independent maxima sum.
    pieces = dict(xi_mean=3.0, xi_std=7.0, weather_mean=-1.0, weather_std=5.0, remainder=2.0)
    assert sum(pieces.values()) == 16.0
    p = M.apply_uav_profile(M.Params(), "L")
    ship = RM.ShipPrediction.from_cv(np.zeros(2), np.zeros(2), [30], "DP")
    route = RM.Route(-1, [_tb_at("ENERGY_BRANCH", 500.0, 0.0)], ship)
    nom = RM.route_nominal_ET(route, 30, p, _WX_CALM, t_dock_s=30.0)
    assert abs(nom["E0"] - max(nom["E_branch_noescort_Wh"],
                                 nom["E_branch_escort_affine_Wh"])) <= 1e-8
    assert nom["E0"] <= (nom["E_branch_noescort_Wh"]
                          + max(0.0, nom["E_branch_escort_affine_Wh"]))
    print(f"{OK} fixed_touchdown_wait_recourse 的12项时间/能量回归通过")


def t21_geo2d_risk_allocation_and_xi_contract():
    """Route-wise Bonferroni allocation and formal xi metadata must be auditable."""
    orig = RM.kappa
    try:
        RM.kappa = RM.KAPPA_MODES["vp_unimodal"]
        cell = M.XiCell(30, "动力定位", 100, np.array([5.0, -3.0]),
                        np.array([[1817.0**2, 0.0], [0.0, 766.0**2]]),
                        10000.0, 3000.0, 2000.0)
        g = np.array([1.0, 0.0])
        fixed = RM._geo2d_dist_bound_details(cell, 0.01, 1800.0, g, 0.6, "fixed")
        opt = RM._geo2d_dist_bound_details(cell, 0.01, 1800.0, g, 0.6, "optimized")
        assert opt["bound_m"] <= fixed["bound_m"] + 1e-9
        assert abs(opt["eps_along"] + opt["eps_cross"] - 0.01) <= 1e-12
        # Monotonicity: more variance cannot improve the optimized certificate.
        cell_big = M.XiCell(30, "动力定位", 100, cell.mu, 4.0 * cell.Sigma,
                            20000.0, 6000.0, 4000.0)
        opt_big = RM._geo2d_dist_bound_details(cell_big, 0.01, 1800.0, g, 0.6, "optimized")
        assert opt_big["bound_m"] >= opt["bound_m"] - 1e-9
        wu = RM.WeatherUncertainty(wind_cov=np.diag([0.5, 0.2]),
                                   wind_bias=np.array([0.01, -0.02]))
        a_xi = np.array([1.0 / 15.0, 0.0]); a_w = np.array([2.0, -1.0])
        jf = RM._soc_margin_geo2d_joint_details(
            a_xi, a_w, 2000.0, cell, wu, 0.0125, 1800.0, 0.6, 0.2, "fixed")
        jo = RM._soc_margin_geo2d_joint_details(
            a_xi, a_w, 2000.0, cell, wu, 0.0125, 1800.0, 0.6, 0.2, "optimized")
        assert jo["total_tightening"] <= jf["total_tightening"] + 1e-9
        assert abs(jo["eps_weather"] + jo["eps_along"] + jo["eps_cross"] - 0.0125) <= 1e-12
        assert jo["risk_allocation_contract"] == RM.GEO_RISK_ALLOCATION_CONTRACT
    finally:
        RM.kappa = orig

    # Formal contract: purge must cover the largest horizon, not merely be positive.
    rows = []
    base = 1_700_000_000.0
    for h in (5, 10):
        for i in range(35):
            rows.append(dict(mmsi="A", h_min=h, c_state="直航",
                             t0_epoch=base + i * 1800.0,
                             xi_e_m=float(i % 3), xi_n_m=float(i % 5)))
    acc_short, _ = S7.summarize_with_contract(
        pd.DataFrame(rows), 30, "none", "train", "nonoverlap", purge_min=5.0)
    acc_full, _ = S7.summarize_with_contract(
        pd.DataFrame(rows), 30, "none", "train", "nonoverlap", purge_min=10.0)
    assert acc_short and not any(bool(r["valid_for_formal"]) for r in acc_short)
    assert acc_full and all(bool(r["valid_for_formal"]) for r in acc_full)
    assert M.Params().soc_risk_allocation == "optimized"
    # Schema resolution must not let the one-character alias 't' capture Latitude.
    cols = ["MMSI", "Latitude", "Longitude", "BaseDateTime"]
    assert S7.find_col(cols, ["t", "timestamp", "datetime"]) == "BaseDateTime"
    assert S8.find_col(cols, ["t", "timestamp", "datetime"]) == "BaseDateTime"
    # step8 and step10 must use the same predicted-motion recovery-state semantics.
    assert S8.cv_recovery_state(np.array([2.0, 0.0]), "转弯") == "直航"
    assert S8.cv_recovery_state(np.zeros(2), "直航") == "动力定位"
    print(f"{OK} geo2d 风险份额、ξ purge、AIS列解析与回收状态合同通过")


def t22_fixed_touchdown_return_speed_recourse():
    """The fixed touchdown policy may consume nominal waiting by accelerating the return leg."""
    old = RM.kappa
    try:
        RM.kappa = RM.KAPPA_MODES["vp_unimodal"]
        wx = dict(wind10=3.0, wind_dir_from=270.0, Hs=0.3, Tp=6.0,
                  wave_dir=180.0, ship_heading=0.0)
        ship = RM.ShipPrediction.from_cv(np.zeros(2), np.zeros(2), [30], "动力定位")
        route = RM.Route(-1, [_tb_at("SPEED_RECOURSE", 1800.0, 0.0)], ship)
        cell = M.XiCell(30, "动力定位", 1000, np.zeros(2),
                        np.diag([1817.655 ** 2, 766.072 ** 2]),
                        10000.0, 3000.0, 2000.0)
        xi = M.XiAmbiguity({(30, "动力定位"): cell}, [30])
        wu = RM.WeatherUncertainty(wind_cov=np.diag([0.01, 0.01]),
                                   wind_bias=np.zeros(2), wind_speed_std=0.05)

        p_wait = M.apply_uav_profile(M.Params(), "L")
        p_wait.time_recourse_mode = "wait_only"
        p_wait.speed_adjustable = False
        p_wait.soc_correction = "geo2d"
        p_wait.soc_risk_allocation = "optimized"
        d_wait = RM.route_feasible_at_h(route, 30, p_wait, wx, xi,
                                        weather_unc=wu, chance_mode="drcc")
        assert d_wait["failure_flags"]["time_drcc_failed"]

        p_speed = M.apply_uav_profile(M.Params(), "L")
        p_speed.time_recourse_mode = "wait_and_speed"
        p_speed.speed_adjustable = True
        p_speed.soc_correction = "geo2d"
        p_speed.soc_risk_allocation = "optimized"
        d_speed = RM.route_feasible_at_h(route, 30, p_speed, wx, xi,
                                         weather_unc=wu, chance_mode="drcc")
        assert d_speed["time_contract"] == RM.SPEED_RECOURSE_TIME_CONTRACT
        assert d_speed["time_feasibility_basis"] == "return_required_airspeed"
        assert d_speed["speed_is_recourse"] is True
        assert d_speed["return_time_budget_s"] > 0.0
        assert d_speed["return_required_airspeed_safe_ms"] <= p_speed.v_air_max
        assert d_speed["return_airspeed_margin_ms"] > 0.0
        assert not d_speed["failure_flags"]["time_drcc_failed"]
        assert d_speed["feasible"], d_speed
        unsupported = RM.route_feasible_at_h(route, 30, p_speed, wx, xi,
                                              weather_unc=wu, chance_mode="saa")
        assert unsupported["reason"] == "speed_recourse_chance_mode_not_certified"

        # More position uncertainty cannot improve the required-airspeed certificate.
        big = M.XiCell(30, "动力定位", 1000, np.zeros(2), 4.0 * cell.Sigma,
                       20000.0, 6000.0, 4000.0)
        c0 = RM._required_airspeed_geo2d_certificate(
            route, 30, p_speed, wx, cell, wu, d_speed["return_time_budget_s"],
            p_speed.eps_T, "optimized")
        c1 = RM._required_airspeed_geo2d_certificate(
            route, 30, p_speed, wx, big, wu, d_speed["return_time_budget_s"],
            p_speed.eps_T, "optimized")
        assert c1["safe_required_airspeed_ms"] >= c0["safe_required_airspeed_ms"] - 1e-9

        # A larger fixed return-time budget cannot worsen the required speed.
        c2 = RM._required_airspeed_geo2d_certificate(
            route, 30, p_speed, wx, cell, wu, d_speed["return_time_budget_s"] + 60.0,
            p_speed.eps_T, "optimized")
        assert c2["safe_required_airspeed_ms"] <= c0["safe_required_airspeed_ms"] + 1e-9
        assert abs(c0["eps_along"] + c0["eps_cross"] - p_speed.eps_T) <= 1e-12

        # Deterministic replay uses the same fixed-touchdown policy: return flight plus waiting
        # exactly fills h, and realised energy remains below the planning certificate.
        rep = RP._realized_speed_recourse(
            route, 30, p_speed, wx, np.zeros(2), np.zeros(2),
            d_speed["t_dock_s"], d_speed["E_dock_Wh"])
        assert not rep["time_violation"]
        assert abs(rep["realized_core_time_s"] + rep["realized_wait_s"] - 1800.0) <= 1e-5
        assert rep["E"] <= d_speed["E_soc_required_Wh"] + 1e-6
        assert rep["return_speed_recourse_contract"] == RM.SPEED_RECOURSE_CONTRACT
    finally:
        RM.kappa = old
    print(f"{OK} 固定接地等待+返程空速 recourse、风险闭合及单调性通过")


def t23_saa_contract_and_mechanism_fallback():
    """SAA samples must keep exact contracts; stale auto files may only degrade in mechanism mode."""
    import tempfile
    root = Path(tempfile.mkdtemp(prefix="saa_contract_"))
    try:
        rows = []
        for i in range(30):
            rows.append(dict(
                h_min=5.0, c_state="直航", xi_e_m=float(i), xi_n_m=float(-i),
                predictor="cv_noleak",
                predictor_contract=M.XI_PREDICTOR_CONTRACTS["cv_noleak"],
                timestamp_epoch_contract=M.XI_TIMESTAMP_EPOCH_CONTRACT,
                mmsi="ALL"))
        good = root / "good.csv"
        pd.DataFrame(rows).to_csv(good, index=False)
        assert RM.load_saa_empirical(good) == 1
        assert (5, "直航") in RM.SAA_EMPIRICAL

        stale = root / "stale.csv"
        bad = pd.DataFrame(rows)
        bad["timestamp_epoch_contract"] = "legacy-pandas-storage-units"
        bad.to_csv(stale, index=False)
        try:
            RM.load_saa_empirical(stale)
            raise AssertionError("stale SAA timestamp contract was accepted")
        except ValueError as exc:
            text = str(exc)
            assert M.XI_TIMESTAMP_EPOCH_CONTRACT in text
            assert "legacy-pandas-storage-units" in text

        offgrid = root / "offgrid.csv"
        bad_h = pd.DataFrame(rows)
        bad_h["h_min"] = math.nextafter(5.0, math.inf)
        bad_h.to_csv(offgrid, index=False)
        try:
            RM.load_saa_empirical(offgrid)
            raise AssertionError("off-grid SAA horizon was absorbed")
        except ValueError as exc:
            assert "binary64" in str(exc)

        # The user's mechanism+synthetic_stress path auto-discovers the default
        # file.  An incompatible auto file must be rejected as empirical data but
        # must not abort the documented synthetic mechanism baseline.
        mech = Namespace(study_mode="mechanism", xi_train_samples=None, saa_samples=None)
        n, usable = S13._register_saa_baseline(mech, stale, set())
        assert n == 0 and usable is False
        assert not RM.SAA_EMPIRICAL
        assert "auto SAA rejected" in RM.SAA_SOURCE

        # Explicit samples and every formal run remain fail-closed.
        explicit = Namespace(study_mode="mechanism", xi_train_samples=None, saa_samples=str(stale))
        try:
            S13._register_saa_baseline(explicit, stale, set())
            raise AssertionError("explicit incompatible SAA did not fail closed")
        except ValueError:
            pass
        formal = Namespace(study_mode="formal", xi_train_samples=None, saa_samples=None)
        try:
            S13._register_saa_baseline(formal, stale, set())
            raise AssertionError("formal incompatible SAA did not fail closed")
        except ValueError:
            pass

        missing_meta = root / "missing_meta.csv"
        pd.DataFrame(rows).drop(columns=["predictor_contract", "timestamp_epoch_contract"]).to_csv(
            missing_meta, index=False)
        try:
            S13._register_saa_baseline(formal, missing_meta, set())
            raise AssertionError("formal SAA without provenance columns did not fail closed")
        except ValueError as exc:
            assert "formal SAA" in str(exc)
    finally:
        import shutil
        shutil.rmtree(root, ignore_errors=True)
        RM.SAA_EMPIRICAL.clear()
        RM.SAA_SOURCE = "moment-gaussian(矩重建, 无经验样本文件)"
    print(f"{OK} SAA 时间戳/精确 horizon 合同与 mechanism 自动旧样本降级通过")

def suite_core():
    np.random.seed(0)
    for fn in (t1_radius, t2_math, t2b_kappa_follow, t3_money, t4_integration,
               t5_select_guard, t6_provenance, t8_autoselect, t9_statistics_and_completion,
               t10_stage1_stage2_contracts, t11_stage3_data_contracts,
               t12_stage4_resource_master, t13_stage5_replay_contracts,
               t14_final_finite_solver_contracts, t15_final_experiment_protocol_contracts,
               t16_audit_regressions, t17_final_concept_regressions,
               t18_frozen_final_test_protocols, t19_zero_pool_model_fixes,
               t20_fixed_touchdown_wait_recourse_contract,
               t21_geo2d_risk_allocation_and_xi_contract,
               t22_fixed_touchdown_return_speed_recourse,
               t23_saa_contract_and_mechanism_fallback):
        fn()
    print("suite core: 全部 PASS ✓")


SUITES = {"bp": suite_bp, "branch": suite_branch, "e1": suite_e1, "core": suite_core,
          "resume": None}   # 占位, 定义在下


# =============================================================================
# suite: resume(更新 任务4: 断点续跑 —— 截断续跑一致性 + 口径冲突拒绝)
# =============================================================================
def suite_resume():
    import shutil
    import tempfile
    print("[resume] E1 断点续跑: 首跑 → 截断后半 → 续跑 → 一致性断言")
    turbines, wx_df, _xi_loaded, lat0lon0, sc_csv, _src_loaded, track_csv = S13.load_all(10, allow_synth=True)
    # 断点续跑测试验证结果签名、跳过与补解，不重复验证风浪门。使用非退化的小方差
    # ξ夹具并关闭占位天气不确定性，避免模型收紧后整张E1表退化为空列池。
    xi_amb = RM._demo_xi([5, 10, 15, 20, 25, 30],
                          ["直航", "转弯", "低速", "动力定位"])
    src = "selftest-demo-xi-train-contract"
    wamb = None
    opts, reach, kind, T_eff, wx0 = S13.build_launch_options(
        turbines, lat0lon0, None, xi_amb, wx_df, 150.0, 10.0, 8000.0,
        hs_quantile=0.5, allow_synth=True)
    # Resume semantics do not require the large E1 geometry.  Use the same
    # physically feasible one-column formal fixture as suite_e1 so the test
    # exercises checkpoint signatures, skipped keys and detail regeneration
    # rather than repeated full-permutation pricing.
    opt0 = opts[0]
    tiny = M.Turbine("RESUME-TINY", np.array([0.0, 0.0]), 10.0, 20.0)
    tiny.local = np.asarray(opt0.ship.P_launch, float).copy()
    reach, opts = [tiny], [opt0]
    xi_amb = RM._demo_xi([10], ["直航", "转弯", "低速", "动力定位"])
    base_args = dict(e1_uavs="S", fleet_ks="1", e1_batteries="0,1",
                     e1_b_auto="on", e1_b_cap=2, e1_sat_patience=1,
                     stops_cap="1", max_stops=1,
                     deck_delta_min=2.5, deck_mode="interval", dtau_min=10.0,
                     t_swap_min=None, t_launch_min=None, landing_clear_min=1.0,
                     swap_stations=1, battery_reuse_mode="exact_soc", replay_n=10,
                     allow_synth=True, track_start_min=None, uav="S",
                     resume="on", _xi_source=src,
                     _pair_radius_m=8000.0, _pair_radius_mode="explicit(8000m)")
    out1 = Path(tempfile.mkdtemp(prefix="e1_resume_"))
    df_full = S13.E1_frontier(reach, opts, M.Params(), xi_amb, wamb, out1,
                              Namespace(**base_args), kind, T_eff)
    n_full = len(df_full)
    # 截断: 只留前一半行(模拟中断), 删明细文件(逼出补解分支)
    df_half = df_full.iloc[: n_full // 2]
    df_half.to_csv(out1 / "E1_frontier.csv", index=False, encoding="utf-8-sig")
    for f in out1.glob("E1_detail_Kmax_*.csv"):
        f.unlink()
    df_res = S13.E1_frontier(reach, opts, M.Params(), xi_amb, wamb, out1,
                             Namespace(**base_args), kind, T_eff)
    assert len(df_res) == n_full, f"续跑行数 {len(df_res)} ≠ 首跑 {n_full}"
    k = ["uav", "K", "batteries"]
    a = df_full.sort_values(k).reset_index(drop=True)
    b = df_res.sort_values(k).reset_index(drop=True)
    assert (a[k].values == b[k].values).all(), "续跑键集合与首跑不一致"
    assert (a["safe_served"].values == b["safe_served"].values).all(), \
        "续跑 safe_served 与首跑不一致(前半应逐位相同=载入, 后半=同种子重解)"
    for uk in ("S",):
        assert (out1 / f"E1_detail_Kmax_{uk}.csv").is_file(), f"续跑未补齐明细 {uk}"
    print(f"{OK} 截断续跑: {n_full//2}/{n_full} 行载入跳过, 其余补齐, 键与 safe_served 逐位一致, 明细补解")
    # 口径冲突: 改 replay_n 再跑 → 必须 SystemExit 拒绝混排
    bad = dict(base_args); bad["replay_n"] = 11
    try:
        S13.E1_frontier(reach, opts, M.Params(), xi_amb, wamb, out1,
                        Namespace(**bad), kind, T_eff)
        raise AssertionError("口径冲突未被拒绝!")
    except SystemExit as e:
        assert "replay_n" in str(e)
    print(f"{OK} 口径冲突(replay_n 10→11): SystemExit 拒绝混排, 报文点名差异字段")
    # resume=off: 整跑覆盖(行数=全量)
    off = dict(base_args); off["resume"] = "off"
    df_off = S13.E1_frontier(reach, opts, M.Params(), xi_amb, wamb, out1,
                             Namespace(**off), kind, T_eff)
    assert len(df_off) == n_full
    print(f"{OK} --resume off: 旧行为(整跑覆盖)保留")
    shutil.rmtree(out1, ignore_errors=True)
    print("suite resume: 全部 PASS ✓")


SUITES["resume"] = suite_resume


def suite_resume_fast():
    """不求解路线的检查点协议快速门禁。

    验证三个发布关键点：签名全字段一致、失败任务不会被当作已完成、
    validation_mode 等口径变化会 fail-closed 拒绝混排。
    """
    import tempfile
    print("[resume_fast] 检查点签名 + 状态过滤 + 口径冲突拒绝")
    with tempfile.TemporaryDirectory(prefix="resume_sig_") as td:
        out = Path(td)
        sig = dict(eps=0.05, result_contract=S13.RESULT_CONTRACT,
                   validation_mode="synthetic_stress",
                   validation_samples_hash="none",
                   weather_alignment_mode="timestamp",
                   weather_start_time="2026-01-01 00:00:00+00:00",
                   weather_match_error_min=0.0,
                   physical_numeric_contract=RM.FORMAL_PHYSICAL_NUMERIC_CONTRACT,
                   route_identity_contract=BP.ROUTE_IDENTITY_CONTRACT,
                   model_semantics_contract=BP.MODEL_SEMANTICS_CONTRACT,
                   resume_input_sha256="resume-input-A")
        rows = [
            dict(criterion="vp", q=0.2, run_status="ok", **sig),
            dict(criterion="box", q=0.2, run_status="failed", **sig),
        ]
        pd.DataFrame(rows).to_csv(out / "E2_robust_raw.csv", index=False,
                                  encoding="utf-8-sig")
        loaded, done = S13._resume_load(
            out, "E2_robust_raw.csv", ["criterion", "q"], sig, "on",
            completed_status_col="run_status", completed_values=("ok", "completed"))
        assert len(loaded) == 2
        assert ("vp", "0.2") in done
        assert ("box", "0.2") not in done, "失败任务必须保留为待重试键"
        try:
            bad = dict(sig, validation_mode="real_holdout")
            S13._resume_load(out, "E2_robust_raw.csv", ["criterion", "q"], bad, "on",
                             completed_status_col="run_status")
        except SystemExit as e:
            assert "validation_mode" in str(e)
        else:
            raise AssertionError("validation_mode 变化未被拒绝")

        # Numeric identity is binary64-exact: one ULP may not resume.
        try:
            bad_num = dict(sig, eps=float(np.nextafter(0.05, np.inf)))
            S13._resume_load(out, "E2_robust_raw.csv", ["criterion", "q"], bad_num, "on",
                             completed_status_col="run_status")
        except SystemExit as e:
            assert "eps" in str(e)
        else:
            raise AssertionError("resume numeric signature accepted a nextafter mismatch")

        # Result/protocol semantics are part of resume identity.  The staged
        # E1/E2 controller changed what counts as a completed formal cell, so a
        # pre-v8 checkpoint must not be silently inherited.
        try:
            old_contract = dict(sig, result_contract="fleet-anytime-result-v7-discrete-recovery-target-xi-only")
            S13._resume_load(out, "E2_robust_raw.csv", ["criterion", "q"], old_contract, "on",
                             completed_status_col="run_status")
        except SystemExit as e:
            assert "result_contract" in str(e)
        else:
            raise AssertionError("resume accepted an old experiment-result semantics contract")

        # Exact input fingerprint change must invalidate the checkpoint.
        try:
            bad_input = dict(sig, resume_input_sha256="resume-input-B")
            S13._resume_load(out, "E2_robust_raw.csv", ["criterion", "q"], bad_input, "on",
                             completed_status_col="run_status")
        except SystemExit as e:
            assert "resume_input_sha256" in str(e)
        else:
            raise AssertionError("resume accepted a changed exact input fingerprint")

        # A positive legacy certificate without exact hashes is never inherited.
        cert_row = dict(criterion="vp", q=0.3, run_status="ok",
                        global_certificate_available=True,
                        global_route_space_certificate=True,
                        implicit_route_space_certified=True, **sig)
        pd.DataFrame([cert_row]).to_csv(out / "cert_missing_hash.csv", index=False,
                                        encoding="utf-8-sig")
        try:
            S13._resume_load(out, "cert_missing_hash.csv", ["criterion", "q"], sig, "on",
                             completed_status_col="run_status")
        except SystemExit as e:
            assert "缺少" in str(e) and "model_contract_sha256" in str(e)
        else:
            raise AssertionError("resume inherited a positive certificate without exact hashes")
    print(f"{OK} 失败任务可重试；nextafter/输入指纹冲突 fail-closed；旧正证书缺 hash 拒绝；新结果合同={S13.RESULT_CONTRACT}")
    print("suite resume_fast: 全部 PASS ✓")


SUITES["resume_fast"] = suite_resume_fast


# =============================================================================
# suite: 更新(第三方审计 1–8 修复的证书语义回归; 无需任何外部文件)
# =============================================================================
def suite_certificates_full():
    """更新 证书闭环回归:
    ① 双制式(weather / ξ-only)+ 资源变体: B&P(L1+L2 证书)对拍【全枚举词典序锚】,
       覆盖数与总能耗须双双逐位相等(L2 能耗 B&P 的最强外部校验);
    ② reach_mode valid ≡ off(保真预筛的经验闭环: 排除不改变最优);
    ③ reach_mode='legacy2d' 消融 ⇒ certificate 撤销(reach_ok=False);
    ④ cg_max_iter 触限 ⇒ L1_status='cg-iter-limit-no-certificate' 且 UB=None(审计#1 降级路径);
    ⑤ enable_rf_branching 开关与 certificate.rf_ok 语义一致(审计#3)。"""
    # 使用确定性、小规模、非退化的物理夹具。旧版依赖无正式数据时的合成E1实例，
    # 在模型收紧后可能全列不可行，既无法验证L2，也会让多次HiGHS调用出现不稳定长跑。
    wx0 = dict(wind10=3.0, wind_dir_from=270.0, Hs=0.2, Tp=6.0,
               wave_dir=0.0, ship_heading=0.0)
    H = [15, 30]
    xi_amb = M.XiAmbiguity({
        (h, "DP"): M.XiCell(h, "DP", 1000, np.zeros(2), np.diag([25.0, 25.0]),
                              0.0, 0.0, 0.0) for h in H}, H)
    ship = RM.ShipPrediction.from_cv(np.zeros(2), np.array([0.1, 0.0]), H, "DP")
    ship.tau_min = 2.5; ship.wx_tau = wx0
    opts = [RM.LaunchOption(2.5, ship, wx0)]
    turbines = []
    for i, (x, y) in enumerate(((500.0, 0.0), (-450.0, 100.0), (300.0, 400.0))):
        tb = M.Turbine(f"CF{i}", np.zeros(2), 68.5, 115.0)
        tb.local = np.array([x, y]); turbines.append(tb)
    reach = turbines
    T_eff = 60.0
    wamb = RM.WeatherUncertainty(wind_cov=np.diag([0.01, 0.01]), hs_std=0.01)
    p = M.apply_uav_profile(M.Params(), "S")
    print(f"实例: deterministic-certificate-fixture, T={T_eff:.0f}min, "
          f"|opts|={len(opts)}, reach={len(reach)}")

    def _anchor_vs_bp(wu, K, Bb, uav, dm, tag):
        pp = M.apply_uav_profile(M.Params(), uav)
        ext, st = RA.enumerate_discrete_routes(reach, opts, pp, xi_amb, T_eff, 2.5, 2, wu)
        assert st["anchor_complete"], f"{tag}: 枚举锚点应完整(status={st['status']})"
        r_ext = RA.solve_resource_master(reach, opts, pp, xi_amb, K, T_eff, t_swap_min=4.0,
                                  max_stops=2, weather_unc=wu, batteries=Bb,
                                  cols_override=ext, solver="auto", deck_mode=dm)
        seed = RA.build_route_columns(reach, opts, pp, xi_amb, T_eff, 2.5, 2, wu,
                                 "drcc", 2.0, "vp_unimodal", 8.0)
        rb = BP.solve_soft_coverage_research(reach, opts, pp, xi_amb, K, T_eff, t_swap_min=4.0,
                                  max_stops=2, weather_unc=wu, batteries=Bb, seed_cols=seed,
                                  time_limit_s=420, deck_mode=dm)
        c = rb["certificate"]
        okC = rb["L1_status"].startswith("optimal") and rb["covered"] == rb["UB"] == r_ext["covered"]
        okE = abs(rb["energy_Wh"] - r_ext["energy_Wh"]) < 0.5
        okX = c["L1_certified"] and c["L2_certified"]
        print(f"[{tag}] anchor cov={r_ext['covered']} E={r_ext['energy_Wh']} | "
              f"BP cov={rb['covered']}/{rb['UB']} E={rb['energy_Wh']} L2={rb['L2_status']} "
              f"dom={rb['dominance_mode']} nodes={rb['nodes']}/L2n={rb['L2_nodes']} "
              f"L1cert={c['L1_certified']} L2cert={c['L2_certified']} "
              f"→ {'PASS' if okC and okE and okX else 'FAIL'}")
        assert okC, f"{tag}: 覆盖/UB 与锚点不一致"
        assert okE, f"{tag}: L2 能耗与锚点不一致({rb['energy_Wh']} vs {r_ext['energy_Wh']})"
        assert okX, f"{tag}: 证书未成立 {c}"
        return seed

    # ① 双制式主对拍 + 资源变体(K/B/uav/deck 变化改变最优结构)
    seed = _anchor_vs_bp(wamb, 3, None, "S", "interval", "weather")
    _anchor_vs_bp(None, 3, None, "S", "interval", "xi-only")
    _anchor_vs_bp(wamb, 1, 2, "L", "slot", "L-K1B2-slot-wu")
    _anchor_vs_bp(None, 2, 3, "S", "slot", "S-K2B3-slot-xi")

    # ② reach valid ≡ off(保真预筛不改变最优)
    ext_v, _ = RA.enumerate_discrete_routes(reach, opts, p, xi_amb, T_eff, 2.5, 2, wamb,
                                    reach_mode="valid")
    ext_o, _ = RA.enumerate_discrete_routes(reach, opts, p, xi_amb, T_eff, 2.5, 2, wamb,
                                    reach_mode="off")
    rv = RA.solve_resource_master(reach, opts, p, xi_amb, 3, T_eff, t_swap_min=4.0, max_stops=2,
                           weather_unc=wamb, cols_override=ext_v, deck_mode="interval")
    ro = RA.solve_resource_master(reach, opts, p, xi_amb, 3, T_eff, t_swap_min=4.0, max_stops=2,
                           weather_unc=wamb, cols_override=ext_o, deck_mode="interval")
    assert rv["covered"] == ro["covered"] and abs(rv["energy_Wh"] - ro["energy_Wh"]) < 0.5, \
        "reach=valid 改变了最优 —— 预筛不保真!"
    print(f"reach valid≡off: cov {rv['covered']}=={ro['covered']}, "
          f"E {rv['energy_Wh']}=={ro['energy_Wh']} → PASS")

    # ③ legacy2d 消融 ⇒ 撤销证书
    rb_leg = BP.solve_soft_coverage_research(reach, opts, p, xi_amb, 3, T_eff, t_swap_min=4.0,
                                  max_stops=2, weather_unc=wamb, seed_cols=seed,
                                  time_limit_s=300, deck_mode="interval",
                                  reach_mode="legacy2d")
    assert rb_leg["certificate"]["reach_ok"] is False and \
        not rb_leg["certificate"]["L1_certified"], "legacy2d 未撤销证书!"
    print(f"legacy2d: status={rb_leg['status']} reach_ok=False L1cert=False → PASS")

    # ④ cg_max_iter=1(必然截断)⇒ 诚实降级
    rb1 = BP.solve_soft_coverage_research(reach, opts, p, xi_amb, 2, T_eff, t_swap_min=4.0,
                               max_stops=2, weather_unc=wamb, batteries=3,
                               seed_cols=seed[:1], time_limit_s=300,
                               deck_mode="interval", cg_max_iter=1)
    assert rb1["L1_status"] == "cg-iter-limit-no-certificate" and rb1["UB"] is None \
        and rb1["gap_pct"] is None and not rb1["certificate"]["L1_certified"], \
        f"迭代触限未降级: {rb1['status']} UB={rb1['UB']}"
    print(f"cg_max_iter=1: status={rb1['status']} UB=None gap=None "
          f"covered={rb1['covered']}(incumbent 仍可报) → PASS")

    # ⑤ RF 开关(更新 M-03 语义): 覆盖型主问题下 RF 对分支不穷尽整数解, 故【启用即撤证】
    #    —— no_rf_branching = (未启用) ∧ (nb_pair==0), 不再只看是否实际发生对分支
    #    (旧口径: 启用但恰未触发对分支时仍发证, 而"未触发"只是实例巧合, 非算法保证)。
    rb2 = BP.solve_soft_coverage_research(reach, opts, p, xi_amb, 2, T_eff, t_swap_min=4.0,
                               max_stops=2, weather_unc=wamb, batteries=3, seed_cols=seed,
                               time_limit_s=300, deck_mode="interval",
                               enable_rf_branching=True)
    assert rb2["certificate"]["rf_ok"] is False, \
        f"启用 RF 却 rf_ok=True(M-03): nb_pair={rb2['n_branch_pair']}"
    assert not rb2["certificate"]["L1_certified"] and \
        "no_rf_branching" in rb2["certificate"]["certificate_reason"], \
        f"启用 RF 未撤证/未给原因: {rb2['certificate']['certificate_reason']}"
    assert rb2["certificate"]["conditions"]["L1"]["no_rf_branching"] is False
    # 未启用档: rf_ok 必须为 True(默认路径不误伤; 更新 M04 同口径)
    print(f"RF 开关(M-03): enable=True ⇒ rf_ok=False, L1cert=False, "
          f"reason 含 no_rf_branching(nb_pair={rb2['n_branch_pair']}) → PASS")

    # ⑥ 有利均值不能把SOC需求降到完整计划能耗以下。旧测试曾允许 E0>B_use 的列
    #    依赖有利风偏置变成可行，这与正式 E_soc=E_plan+U_E、U_E≥0 的电池口径冲突。
    #    新回归同时验证：带符号均值时 valid 预筛自动降级为 off，但最终物理判定仍拒绝
    #    名义已超可用SOC的列，且B&P可证明零覆盖而不是依赖名义剪枝。
    p6 = M.Params(); p6.B_k = 390.0
    p6.power_scale = 439.0 / M.P_zeng(0.0, M.Params())
    p6.safe_reserve = 0.20; p6.w_land_max = 500.0; p6.W_max = 500.0; p6.v_max = 100.0; p6.v_air_max = 100.0
    p6.airspeed_cc = "off"; p6.t_dock_base_s = 0.0; p6.dock_gamma = 0.0; p6.tau_insp = 300.0
    p6.Hs_op = 100.0; p6.s_heave_max = 100.0; p6.s_roll_max = 100.0; p6.s_pitch_max = 100.0
    h6, D6 = 60, 48750.0
    ship6 = RM.ShipPrediction.from_cv(np.zeros(2), np.array([D6 / (h6 * 60.0), 0.0]),
                                      [h6], c_state="DP")
    ship6.tau_min = 0.0
    wx6 = dict(wind10=0.0, wind_dir_from=270.0, Hs=0.0, Tp=8.0,
               wave_dir=0.0, ship_heading=0.0)
    ship6.wx_tau = wx6
    opt6 = RM.LaunchOption(0.0, ship6, wx6)
    t6 = M.Turbine("T1", np.zeros(2), 68.5, 115.0); t6.local = np.array([D6, 0.0])
    cell6 = M.XiCell(h6, "DP", 1000, np.zeros(2), np.zeros((2, 2)), 0.0, 0.0, 0.0)
    xi6 = M.XiAmbiguity({(h6, "DP"): cell6}, [h6])
    wu6 = RM.WeatherUncertainty(wind_cov=np.zeros((2, 2)), wind_bias=np.array([10.0, 0.0]),
                                hs_std=0.0, hs_bias=0.0)
    d6 = RM.route_feasible_at_h(RM.Route(-1, [t6], ship6), h6, p6, wx6, xi6,
                                weather_unc=wu6, chance_mode="drcc")
    assert (not d6["feasible"] and d6["reason"] == "energy_margin_negative"
            and d6["E0"] > p6.B_use
            and d6["E_soc_required_Wh"] >= d6["E_plan_Wh"] - 1e-9)
    rvv = RA.tau_reach(opt6, [t6], p6, h6, mode="valid", wx=wx6,
                       xi_amb=xi6, weather_unc=wu6)
    assert [x.tid for x in rvv] == ["T1"] and rvv.effective_mode == "off"
    c6v, s6v = RA.enumerate_discrete_routes([t6], [opt6], p6, xi6, T_min=60,
                                    deck_delta_min=2.5, max_stops=1,
                                    weather_unc=wu6, reach_mode="valid")
    assert len(c6v) == 0 and s6v["anchor_complete"] and s6v["reach_effective"] == "off"
    r6 = BP.solve_soft_coverage_research([t6], [opt6], p6, xi6, K=1, T_min=60,
                              deck_delta_min=2.5, t_swap_min=0.0, max_stops=1,
                              weather_unc=wu6, batteries=1, seed_cols=[],
                              time_limit_s=30, cg_max_iter=50)
    assert r6["covered"] == r6["UB"] == 0 and r6["certificate"]["L1_certified"]         and r6["certificate"]["L2_certified"]         and not r6["certificate"]["nominal_prunes_active"]
    print(f"SOC非负缓冲: E_plan={d6['E_plan_Wh']:.1f}>B_use={p6.B_use:.0f}, "
          f"有利均值仍拒绝；valid自动off；BP证明0/0 → PASS")

    # 标号预算触限必须诚实降级。使用主夹具保证存在可行列，避免零覆盖平凡闭合。
    r6b = BP.solve_soft_coverage_research(reach, opts, p, xi_amb, K=2, T_min=T_eff,
                               deck_delta_min=2.5, t_swap_min=4.0, max_stops=2,
                               weather_unc=wamb, batteries=3, seed_cols=[],
                               time_limit_s=30, cg_max_iter=50,
                               pricing_label_budget=0)
    assert r6b["L1_status"] == "pricing-label-limit-no-certificate" and r6b["UB"] is None         and r6b["gap_pct"] is None and not r6b["certificate"]["L1_certified"],         f"标号预算触限未诚实降级: {r6b['status']} UB={r6b['UB']}"
    print(f"标号预算阀: budget=0 ⇒ {r6b['status']} UB=None → PASS")

    # ⑦ 空种子 Phase-I(主夹具, ξ-only): 纯定价从空池自举须与锚点同优(杀变异#4)
    ext7, _ = RA.enumerate_discrete_routes(reach, opts, p, xi_amb, T_eff, 2.5, 2, None)
    r_ext7 = RA.solve_resource_master(reach, opts, p, xi_amb, 2, T_eff, t_swap_min=4.0,
                               max_stops=2, weather_unc=None, batteries=3,
                               cols_override=ext7, deck_mode="interval")
    r7 = BP.solve_soft_coverage_research(reach, opts, p, xi_amb, 2, T_eff, t_swap_min=4.0,
                              max_stops=2, weather_unc=None, batteries=3, seed_cols=[],
                              time_limit_s=420, deck_mode="interval")
    assert r7["L1_status"].startswith("optimal") and \
        r7["covered"] == r7["UB"] == r_ext7["covered"] and \
        abs(r7["energy_Wh"] - r_ext7["energy_Wh"]) < 0.5 and \
        r7["certificate"]["L1_certified"] and r7["certificate"]["L2_certified"], \
        f"空种子 Phase-I 失配: {r7['covered']}/{r7['UB']} vs {r_ext7['covered']}"
    print(f"空种子 Phase-I: BP {r7['covered']}/{r7['UB']} E={r7['energy_Wh']}"
          f"=={r_ext7['energy_Wh']} 证书=True → PASS")

    # ⑧ acquisition 不在有限模型；活跃 Bonferroni 总和允许保守小于 0.05，绝不超配。
    _pb = M.Params()
    expected_off = _pb.eps_E + _pb.eps_T
    expected_on = sum((_pb.eps_E, _pb.eps_T, _pb.eps_cap, _pb.eps_gate,
                       _pb.eps_air, _pb.eps_dock, _pb.eps_escort))
    assert abs(RM.mission_eps_budget(_pb, False) - expected_off) < 1e-12
    assert abs(RM.mission_eps_budget(_pb, True) - expected_on) < 1e-12
    assert expected_on < _pb.mission_failure_budget and RM.mission_budget_compliant(_pb, True)
    assert "acquisition" not in RM.mission_risk_allocation(_pb, True)
    print(f"预算上界: off={expected_off:.3f}, on={expected_on:.3f} <= mission 0.05；acquisition 不在事件集 → PASS")

    # ⑨ 物理微 oracle(独立于锚点, 杀变异 #6/#7/#8):
    #  (a) 伴飞−对接: t_dock 增加后 T_escort 等量减少，E0 按伴飞功率减少。
    rr = RM.Route(-1, [reach[0]], opts[0].ship)
    hh = int(max(RM.decision_horizons_of(xi_amb)))
    n0 = RM.route_nominal_ET(rr, hh, p, opts[0].wx, t_dock_s=0.0)
    n1 = RM.route_nominal_ET(rr, hh, p, opts[0].wx, t_dock_s=180.0)
    if n1["E_escort"] > 1e-9:    # 两次都处于伴飞制才有解析关系
        _pe = float(RM.escort_state(rr, hh, p, opts[0].wx)["power_W"])
        _dE = _pe * 180.0 / 3600.0
        assert abs((n0["E_escort"] - n1["E_escort"]) - _dE) < 1e-6, "伴飞窗未扣 t_dock(变异#6)"
        assert abs((n0["E0"] - n1["E0"]) - _dE) < 1e-6, "E0 未随 t_dock 同步变化"
        print(f"伴飞−对接 oracle: ΔE_escort={_dE:.3f}Wh 解析吻合 → PASS")
    #  (b) wx_local 腿的 wind_delta 注入: 全局风=本地风=0 时, wind_delta 仍须改变能耗
    t9 = M.Turbine("W1", np.zeros(2), 68.5, 115.0); t9.local = np.array([3000.0, 0.0])
    t9.wx_local = dict(wind10=0.0, wind_dir_from=270.0, Hs=0.3, Tp=6.0, wave_dir=0.0)
    ship9 = RM.ShipPrediction.from_cv(np.zeros(2), np.array([0.4, 0.0]),
                                      [hh], c_state="DP")
    r9 = RM.Route(-1, [t9], ship9)
    wx9 = dict(wind10=0.0, wind_dir_from=270.0, Hs=0.3, Tp=6.0, wave_dir=0.0, ship_heading=0.0)
    E_a = RM.route_energy_time(r9, hh, np.zeros(2), p, wx9)[0]
    E_b = RM.route_energy_time(r9, hh, np.zeros(2), p, wx9,
                               wind_delta=np.array([6.0, 0.0]))[0]
    assert abs(E_a - E_b) > 1e-3, "wind_delta 未进 wx_local 腿(变异#7)"
    #  (c) 门风上收解析式: w10_gate_eff == recovery_gate_wx.wind10 + r_gate(ε_gate)
    rr_g = RM.Route(-1, [reach[0]], opts[0].ship)
    dg = RM.route_feasible_at_h(rr_g, hh, p, opts[0].wx, xi_amb, weather_unc=wamb)
    wu_h = RM._resolve_weather_unc(wamb, hh)
    _rg = RM.wind_speed_upper_shift("drcc", wu_h, float(getattr(p, "eps_gate", p.eps_cap)))
    gwx = RM.recovery_gate_wx(rr_g, opts[0].wx, rr_g.ship.predicted_at(float(hh)))
    assert abs(dg["w10_gate_eff"] - (float(gwx["wind10"]) + _rg)) < 1e-6, \
        f"门风未按标量风速 r_gate 上收: {dg['w10_gate_eff']} vs {gwx['wind10']}+{_rg}"
    print(f"物理微 oracle: wx_local 注入 ΔE={abs(E_a-E_b):.2f}Wh | "
          f"w10_gate_eff={dg['w10_gate_eff']:.2f} = 名义{float(gwx['wind10']):.2f}+r_gate{_rg:.2f} → PASS")
    print("suite certificates_full: 全部 PASS ✓")


def suite_certificates():
    """快速证书门禁：正常双证书、标号预算撤证和 RF 开关撤证。"""
    p = M.apply_uav_profile(M.Params(), "L")
    t = _tb_at("C0", 2500.0, 0.0)
    opt = _mk_launch(0.0, (0.0, 0.0), _WX_CALM, horizons=(15, 30))
    xi = M.XiAmbiguity({
        (h, "DP"): M.XiCell(h, "DP", 1000, np.zeros(2),
                              np.diag([100.0, 100.0]), 0.0, 0.0, 0.0)
        for h in (15, 30)}, [15, 30])
    kw = dict(deck_delta_min=2.5, t_swap_min=1.0, t_launch_min=0.5,
              max_stops=1, weather_unc=None, batteries=1, seed_cols=[],
              time_limit_s=60, deck_mode="interval", max_nodes=30)
    r = BP.solve_soft_coverage_research([t], [opt], p, xi, 1, 40.0, **kw)
    c = r["certificate"]
    assert r["covered"] == r["UB"] == 1 and c["L1_certified"] and c["L2_certified"], r
    print("正常路径：L1/L2 双证书成立 ✓")

    r_lim = BP.solve_soft_coverage_research([t], [opt], p, xi, 1, 40.0,
                                  **{**kw, "pricing_label_budget": 0})
    assert r_lim["UB"] is None and not r_lim["certificate"]["L1_certified"] and \
        r_lim["L1_status"] == "pricing-label-limit-no-certificate", r_lim
    print("标号预算触限：UB=None 且撤证 ✓")

    r_rf = BP.solve_soft_coverage_research([t], [opt], p, xi, 1, 40.0,
                                 **{**kw, "enable_rf_branching": True})
    assert not r_rf["certificate"]["L1_certified"] and \
        not r_rf["certificate"]["conditions"]["L1"]["no_rf_branching"], r_rf
    print("启用非完备 RF 分支：按设计撤证 ✓")
    print("suite certificates: 全部 PASS ✓")


SUITES["certificates"] = suite_certificates
SUITES["certificates_full"] = suite_certificates_full


def suite_pricing_shadow():
    """M1/M2 regression: selectable mode and one-sided shadow bound."""
    base = BP._validate_anytime_public_contract(
        solver_mode="exact-branch-price-cut", pricing_mode="exact-implicit-dfs",
        kappa_mode="vp_unimodal", chance_mode="drcc", deck_mode="interval",
        battery_reuse_mode="exact_soc", solver="auto", pool_h_mode="pareto",
        time_limit_s=None, deadline=None, budget_gamma=2.0, K=1, batteries=1,
        max_stops=3, coverage_gap_target_abs=0, energy_gap_target_rel=0.0,
        energy_gap_target_abs_Wh=1e-6, pricing_batch_size=16,
        solve_scope="lexicographic")
    exp = BP._validate_anytime_public_contract(
        solver_mode="exact-branch-price-cut", pricing_mode="exact-discovery-shadow",
        kappa_mode="vp_unimodal", chance_mode="drcc", deck_mode="interval",
        battery_reuse_mode="exact_soc", solver="auto", pool_h_mode="pareto",
        time_limit_s=None, deadline=None, budget_gamma=2.0, K=1, batteries=1,
        max_stops=3, coverage_gap_target_abs=0, energy_gap_target_rel=0.0,
        energy_gap_target_abs_Wh=1e-6, pricing_batch_size=16,
        solve_scope="lexicographic")
    guided = BP._validate_anytime_public_contract(
        solver_mode="exact-branch-price-cut", pricing_mode="exact-dual-guided-shadow",
        kappa_mode="vp_unimodal", chance_mode="drcc", deck_mode="interval",
        battery_reuse_mode="exact_soc", solver="auto", pool_h_mode="pareto",
        time_limit_s=None, deadline=None, budget_gamma=2.0, K=1, batteries=1,
        max_stops=3, coverage_gap_target_abs=0, energy_gap_target_rel=0.0,
        energy_gap_target_abs_Wh=1e-6, pricing_batch_size=16,
        solve_scope="lexicographic")
    assert base["pricing_mode"] == "exact-implicit-dfs"
    assert exp["pricing_mode"] == "exact-discovery-shadow"
    layered = BP._validate_anytime_public_contract(
        solver_mode="exact-branch-price-cut",
        pricing_mode="exact-layered-guided-shadow",
        kappa_mode="vp_unimodal", chance_mode="drcc", deck_mode="interval",
        battery_reuse_mode="exact_soc", solver="auto", pool_h_mode="pareto",
        time_limit_s=None, deadline=None, budget_gamma=2.0, K=1, batteries=1,
        max_stops=3, coverage_gap_target_abs=0, energy_gap_target_rel=0.0,
        energy_gap_target_abs_Wh=1e-6, pricing_batch_size=16,
        solve_scope="lexicographic")
    batch_mode = BP._validate_anytime_public_contract(
        solver_mode="exact-branch-price-cut",
        pricing_mode="exact-layered-batch-shadow",
        kappa_mode="vp_unimodal", chance_mode="drcc", deck_mode="interval",
        battery_reuse_mode="exact_soc", solver="auto", pool_h_mode="pareto",
        time_limit_s=None, deadline=None, budget_gamma=2.0, K=1, batteries=1,
        max_stops=3, coverage_gap_target_abs=0, energy_gap_target_rel=0.0,
        energy_gap_target_abs_Wh=1e-6, pricing_batch_size=16,
        solve_scope="lexicographic")
    assert guided["pricing_mode"] == "exact-dual-guided-shadow"
    assert layered["pricing_mode"] == "exact-layered-guided-shadow"
    primal_mode = BP._validate_anytime_public_contract(
        solver_mode="exact-branch-price-cut",
        pricing_mode="exact-layered-batch-primal-shadow",
        kappa_mode="vp_unimodal", chance_mode="drcc", deck_mode="interval",
        battery_reuse_mode="exact_soc", solver="auto", pool_h_mode="pareto",
        time_limit_s=None, deadline=None, budget_gamma=2.0, K=1, batteries=1,
        max_stops=3, coverage_gap_target_abs=0, energy_gap_target_rel=0.0,
        energy_gap_target_abs_Wh=1e-6, pricing_batch_size=16,
        solve_scope="lexicographic")
    diagnostic_mode = BP._validate_anytime_public_contract(
        solver_mode="exact-branch-price-cut",
        pricing_mode="exact-layered-batch-primal-diagnostic-shadow",
        kappa_mode="vp_unimodal", chance_mode="drcc", deck_mode="interval",
        battery_reuse_mode="exact_soc", solver="auto", pool_h_mode="pareto",
        time_limit_s=None, deadline=None, budget_gamma=2.0, K=1, batteries=1,
        max_stops=3, coverage_gap_target_abs=0, energy_gap_target_rel=0.0,
        energy_gap_target_abs_Wh=1e-6, pricing_batch_size=16,
        solve_scope="lexicographic")
    assert batch_mode["pricing_mode"] == "exact-layered-batch-shadow"
    assert primal_mode["pricing_mode"] == "exact-layered-batch-primal-shadow"
    target9_mode = BP._validate_anytime_public_contract(
        solver_mode="exact-branch-price-cut",
        pricing_mode="exact-layered-batch-primal-target9-diagnostic-shadow",
        kappa_mode="vp_unimodal", chance_mode="drcc", deck_mode="interval",
        battery_reuse_mode="exact_soc", solver="auto", pool_h_mode="pareto",
        time_limit_s=None, deadline=None, budget_gamma=2.0, K=1, batteries=1,
        max_stops=3, coverage_gap_target_abs=0, energy_gap_target_rel=0.0,
        energy_gap_target_abs_Wh=1e-6, pricing_batch_size=16,
        solve_scope="lexicographic")
    assert diagnostic_mode["pricing_mode"] == (
        "exact-layered-batch-primal-diagnostic-shadow")
    assert target9_mode["pricing_mode"] == (
        "exact-layered-batch-primal-target9-diagnostic-shadow")
    target9_v10_mode = BP._validate_anytime_public_contract(
        solver_mode="exact-branch-price-cut",
        pricing_mode="exact-layered-batch-primal-target9-certificate-diagnostic-shadow",
        kappa_mode="vp_unimodal", chance_mode="drcc", deck_mode="interval",
        battery_reuse_mode="exact_soc", solver="auto", pool_h_mode="pareto",
        time_limit_s=None, deadline=None, budget_gamma=2.0, K=1, batteries=1,
        max_stops=3, coverage_gap_target_abs=0, energy_gap_target_rel=0.0,
        energy_gap_target_abs_Wh=1e-6, pricing_batch_size=16,
        solve_scope="lexicographic")
    assert target9_v10_mode["pricing_mode"] == (
        "exact-layered-batch-primal-target9-certificate-diagnostic-shadow")
    target9_v11_mode = BP._validate_anytime_public_contract(
        solver_mode="exact-branch-price-cut",
        pricing_mode="exact-layered-batch-primal-target9-battery-clique-diagnostic-shadow",
        kappa_mode="vp_unimodal", chance_mode="drcc", deck_mode="interval",
        battery_reuse_mode="exact_soc", solver="auto", pool_h_mode="pareto",
        time_limit_s=None, deadline=None, budget_gamma=2.0, K=1, batteries=1,
        max_stops=3, coverage_gap_target_abs=0, energy_gap_target_rel=0.0,
        energy_gap_target_abs_Wh=1e-6, pricing_batch_size=16,
        solve_scope="lexicographic")
    assert target9_v11_mode["pricing_mode"] == (
        "exact-layered-batch-primal-target9-battery-clique-diagnostic-shadow")

    v13_mode = BP._validate_anytime_public_contract(
        solver_mode="exact-branch-price-cut",
        pricing_mode="exact-layered-batch-primal-battery-halfcap-depth-fair-formal",
        kappa_mode="vp_unimodal", chance_mode="drcc", deck_mode="interval",
        battery_reuse_mode="exact_soc", solver="auto", pool_h_mode="pareto",
        time_limit_s=None, deadline=None, budget_gamma=2.0, K=1, batteries=1,
        max_stops=3, coverage_gap_target_abs=0, energy_gap_target_rel=0.0,
        energy_gap_target_abs_Wh=1e-6, pricing_batch_size=16,
        solve_scope="lexicographic")
    assert v13_mode["pricing_mode"] == (
        "exact-layered-batch-primal-battery-halfcap-depth-fair-formal")

    v14_mode = BP._validate_anytime_public_contract(
        solver_mode="exact-branch-price-cut",
        pricing_mode=(
            "exact-layered-batch-primal-battery-halfcap-depth-fair-neutral-formal"),
        kappa_mode="vp_unimodal", chance_mode="drcc", deck_mode="interval",
        battery_reuse_mode="exact_soc", solver="auto", pool_h_mode="pareto",
        time_limit_s=None, deadline=None, budget_gamma=2.0, K=1, batteries=1,
        max_stops=3, coverage_gap_target_abs=0, energy_gap_target_rel=0.0,
        energy_gap_target_abs_Wh=1e-6, pricing_batch_size=16,
        solve_scope="lexicographic")
    assert v14_mode["pricing_mode"] == (
        "exact-layered-batch-primal-battery-halfcap-depth-fair-neutral-formal")

    # Sequence-level launch round-robin: fairness changes only order, never set.
    rr = list(BP._round_robin_tagged_iterators([
        ("L0", [1, 2, 3]), ("L1", [10, 11]), ("L2", [20])]))
    assert rr == [
        (0, "L0", 1), (0, "L1", 10), (0, "L2", 20),
        (1, "L0", 2), (1, "L1", 11), (2, "L0", 3)]
    assert sorted((tag, value) for _, tag, value in rr) == sorted([
        ("L0", 1), ("L0", 2), ("L0", 3),
        ("L1", 10), ("L1", 11), ("L2", 20)])

    raw_order = ["A", "B", "C"]
    ranked, changed, failed = BP._stable_discovery_order(
        raw_order, lambda x: {"A": -1.0, "B": -3.0, "C": -2.0}[x])
    assert ranked == ["B", "C", "A"] and changed and not failed
    fallback, changed2, failed2 = BP._stable_discovery_order(
        raw_order, lambda x: (_ for _ in ()).throw(ValueError("probe")))
    assert fallback == raw_order and not changed2 and failed2
    assert sorted(ranked) == sorted(raw_order)

    # V5 full-domain equivalence on a tiny physical instance.  Use an energy
    # pricing objective with no dual rows, so no negative column can trigger a
    # discovery early return.  The layered discovery must therefore finish and
    # touch exactly the same (launch, ordered sequence, h) physical-cache domain
    # as the historical certification DFS.
    pdom = M.Params()
    pdom.B_k = max(float(getattr(pdom, "B_k", 1000.0)), 5000.0)
    pdom.safe_reserve = min(float(getattr(pdom, "safe_reserve", 0.2)), 0.1)
    pdom.Hs_op = max(float(getattr(pdom, "Hs_op", 1.0)), 10.0)
    pdom.W_max = max(float(getattr(pdom, "W_max", 1.0)), 100.0)
    pdom.w_land_max = max(float(getattr(pdom, "w_land_max", 1.0)), 100.0)
    da = M.Turbine("DOM-A", np.zeros(2), 68.5, 115.0)
    db = M.Turbine("DOM-B", np.zeros(2), 68.5, 115.0)
    da.local = np.array([250.0, 0.0])
    db.local = np.array([400.0, 120.0])
    dom_turbines = [da, db]
    dom_horizons = [30, 45]
    dom_wx = dict(wind10=0.0, wind_dir_from=270.0, Hs=0.1, Tp=6.0,
                  wave_dir=0.0, ship_heading=0.0)
    dom_opts = []
    for tau0, x0 in ((0.0, 0.0), (5.0, 50.0)):
        dsp = RM.ShipPrediction.from_cv(
            np.array([x0, 0.0]), np.array([0.0, 0.0]),
            dom_horizons, "DP")
        dsp.tau_min = tau0
        dsp.wx_tau = dict(dom_wx)
        dom_opts.append(RM.LaunchOption(tau0, dsp, dict(dom_wx)))
    dom_cells = {
        (h0, "DP"): M.XiCell(
            h0, "DP", 100, np.zeros(2), np.zeros((2, 2)),
            0.0, 0.0, 0.0)
        for h0 in dom_horizons
    }
    dom_xi = M.XiAmbiguity(dom_cells, dom_horizons)
    dom_node = SimpleNamespace(branch=BP.BranchState())
    dom_common = dict(
        turbines=dom_turbines, launch_opts=dom_opts, p=pdom,
        xi_amb=dom_xi, weather_unc=None, T_min=60.0, max_stops=2,
        node=dom_node, existing_signatures=set(), stage="energy",
        inequality_rows=[], equality_rows=[],
        inequality_duals=np.asarray([]), equality_duals=np.asarray([]),
        deadline=None, pricing_epsilon=1e-7, t_launch_min=0.0,
        landing_clear_min=0.0, deck_mode="interval", deck_delta_min=2.5,
        kappa_mode="vp_unimodal", chance_mode="drcc", budget_gamma=2.0,
        implicit_test_columns=None, batch_size=16,
        shadow_prefix_bounds=False)
    dom_cache_old = {}
    dom_old = BP._exact_pricing_search(
        **dom_common, physical_cache=dom_cache_old,
        search_goal="certification", discovery_column_limit=8,
        guided_discovery_order=False, layered_discovery_order=False)
    dom_cache_v5 = {}
    dom_v5 = BP._exact_pricing_search(
        **dom_common, physical_cache=dom_cache_v5,
        search_goal="discovery", discovery_column_limit=16,
        guided_discovery_order=True, layered_discovery_order=True)
    assert dom_old.complete and dom_v5.complete
    assert set(dom_cache_old) == set(dom_cache_v5)
    assert dom_old.evaluated_sequences == dom_v5.evaluated_sequences
    assert dom_v5.layered_max_depth_completed == 2

    # V13 full-domain equivalence and depth fairness.  A nonzero formal
    # battery-halfcap dual activates only the discovery visit order.  Use a
    # three-turbine extension so depth-3 exists, and compare the exact physical
    # cache domain against historical certification under the same row/dual.
    dc = M.Turbine("DOM-C", np.zeros(2), 68.5, 115.0)
    dc.local = np.array([320.0, -140.0])
    dom_fair_turbines = [da, db, dc]
    dom_fair_common = dict(
        dom_common,
        turbines=dom_fair_turbines,
        max_stops=3,
        inequality_rows=[("battery_halfcap", float(pdom.B_use))],
        inequality_duals=np.asarray([-1.0], float))
    dom_cache_v13_cert = {}
    dom_v13_cert = BP._exact_pricing_search(
        **dom_fair_common, physical_cache=dom_cache_v13_cert,
        search_goal="certification", discovery_column_limit=16,
        guided_discovery_order=False, layered_discovery_order=False,
        depth_fair_discovery_order=False)
    dom_cache_v13 = {}
    dom_v13 = BP._exact_pricing_search(
        **dom_fair_common, physical_cache=dom_cache_v13,
        search_goal="discovery", discovery_column_limit=16,
        guided_discovery_order=True, layered_discovery_order=True,
        depth_fair_discovery_order=True)
    assert dom_v13_cert.complete and dom_v13.complete
    assert dom_v13.depth_fair_requested and dom_v13.depth_fair_active
    assert dom_v13.depth_fair_halfcap_dual_abs == 1.0
    assert dom_v13.depth_fair_rounds > 0
    assert set(dom_cache_v13_cert) == set(dom_cache_v13)
    v13_prefix_order = []
    _seen_prefixes = set()
    for key in dom_cache_v13:
        _prefix_key = (key[0], key[1])
        if _prefix_key in _seen_prefixes:
            continue
        _seen_prefixes.add(_prefix_key)
        v13_prefix_order.append(len(key[1]))
    assert 3 in v13_prefix_order and 2 in v13_prefix_order
    assert v13_prefix_order.index(3) < max(
        i for i, d in enumerate(v13_prefix_order) if d == 2)

    dom_cache_v13_zero = {}
    dom_v13_zero = BP._exact_pricing_search(
        **dict(dom_fair_common, inequality_duals=np.asarray([0.0], float)),
        physical_cache=dom_cache_v13_zero,
        search_goal="discovery", discovery_column_limit=16,
        guided_discovery_order=True, layered_discovery_order=True,
        depth_fair_discovery_order=True)
    assert dom_v13_zero.complete
    assert dom_v13_zero.depth_fair_requested
    assert not dom_v13_zero.depth_fair_active
    assert dom_v13_zero.depth_fair_rounds == 0

    # V14: positive/cross-zero multi-stop columns may be returned only as an
    # explicitly incomplete discovery enrichment batch.  They must never be
    # counted as strict-negative pricing progress or establish CLOSED.
    v14_neutral_cols = [
        _bpc_test_column(("PS-A", "PS-B"), 0.0, 5.0, 6.0),
        _bpc_test_column(("PS-C", "PS-D"), 1.0, 5.0, 6.0),
        _bpc_test_column(("PS-A", "PS-C"), 2.0, 5.0, 6.0),
    ]
    v14_neutral = BP._exact_pricing_search(
        _bpc_turbines("PS-A", "PS-B", "PS-C", "PS-D"), [], M.Params(),
        _bpc_xi(), None, 60.0, 2, dom_node, set(), "coverage",
        [("battery_halfcap", 10.0)], [], np.asarray([-3.0]), np.asarray([]),
        None, 1e-7, 0.0, 0.0, "interval", 2.5,
        implicit_test_columns=v14_neutral_cols, batch_size=8,
        search_goal="discovery", discovery_column_limit=4,
        guided_discovery_order=True, layered_discovery_order=True,
        depth_fair_discovery_order=True,
        neutral_multistop_enrichment=True,
        neutral_multistop_batch_target=2,
        neutral_uncovered_tids=set())
    assert v14_neutral.neutral_multistop_enabled
    assert v14_neutral.neutral_multistop_early_return
    assert v14_neutral.termination_reason == (
        "discovery-neutral-multistop-batch-found")
    assert not v14_neutral.complete and not v14_neutral.closed
    assert v14_neutral.improving_columns_seen == 0
    assert v14_neutral.discovery_improving_columns_returned == 0
    assert v14_neutral.neutral_multistop_returned == 2
    assert v14_neutral.neutral_multistop_returned_by_depth == {2: 2}
    assert len(v14_neutral.columns) == 2
    assert all(len(BP._ordered_tids(c)) == 2 for c in v14_neutral.columns)

    # A proved-negative column always remains formal pricing progress; neutral
    # candidates cannot replace it or weaken the unchanged rc_ub<0 firewall.
    v14_negative_first = _bpc_test_column(("PS-A", "PS-B"), 3.0, 5.0, 1.0)
    v14_mixed = BP._exact_pricing_search(
        _bpc_turbines("PS-A", "PS-B", "PS-C", "PS-D"), [], M.Params(),
        _bpc_xi(), None, 60.0, 2, dom_node, set(), "coverage",
        [("battery_halfcap", 10.0)], [], np.asarray([-3.0]), np.asarray([]),
        None, 1e-7, 0.0, 0.0, "interval", 2.5,
        implicit_test_columns=[v14_negative_first] + v14_neutral_cols,
        batch_size=8, search_goal="discovery", discovery_column_limit=4,
        guided_discovery_order=True, layered_discovery_order=True,
        depth_fair_discovery_order=True,
        neutral_multistop_enrichment=True,
        neutral_multistop_batch_target=2,
        neutral_uncovered_tids=set())
    assert v14_mixed.improving_columns_seen >= 1
    assert v14_mixed.neutral_multistop_returned == 0
    assert len(v14_mixed.columns) >= 1

    wx = dict(wind10=0.0, wind_dir_from=270.0, Hs=0.2, Tp=6.0,
              wave_dir=0.0, ship_heading=0.0)
    ship = RM.ShipPrediction.from_cv(np.zeros(2), np.zeros(2), [15], "DP")
    ship.tau_min = 0.0
    ship.wx_tau = wx
    opt = RM.LaunchOption(0.0, ship, wx)
    ta = M.Turbine("PS-A", np.zeros(2), 68.5, 115.0)
    branch = BP.BranchState(required_turbines=frozenset({"PS-A"}))
    ineq = [("packing", "PS-A"), ("packing", "PS-B"),
            ("resource_pattern", frozenset())]
    eq = [("coverage", None), ("required_service", "PS-A")]
    idual = np.asarray([-0.3, -0.2, -0.1])
    edual = np.asarray([0.4, -0.25])
    lb = BP._prefix_pricing_lower_bound(
        "energy", 3, (ta,), opt, branch, ineq, eq, idual, edual)
    global_lb = BP._universal_pricing_lower_bound(
        "energy", 3, ineq, eq, idual, edual)
    assert math.isfinite(lb) and math.isfinite(global_lb)
    assert lb >= global_lb or math.isclose(
        lb, global_lb, rel_tol=0.0, abs_tol=1e-15)

    # R-BPC certified prefix pruning: promote the prefix theorem only on
    # exhaustive certification and use mathematical zero, never an epsilon.
    # The stub below deliberately preserves the theorem premise: route energy
    # is the exact inspection/climb service component plus a nonnegative term.
    _cp_p = M.Params()
    _cp_turbs = []
    for _i, _tid0 in enumerate(("CP-A", "CP-B", "CP-C", "CP-D")):
        _tb = M.Turbine(_tid0, np.array([float(_i), 0.0]), 68.5, 115.0)
        _tb.local = np.array([float(_i), 0.0])
        _cp_turbs.append(_tb)
    def _cp_service_energy(_seq):
        _e = 0.0
        for _tb in _seq:
            _dz = float(M.insp_vertical_span(_tb, _cp_p.z_cruise))
            if _cp_p.use_zeng:
                _pup = float(M.P_zeng(0.0, _cp_p) + 7.27 * 9.81 * _cp_p.v_z)
                _pinsp = float(M.P_zeng(_cp_p.v_orbit, _cp_p))
            else:
                _pup = float(_cp_p.P_climb); _pinsp = float(_cp_p.P_hov)
            _e += float(_pup * _dz / _cp_p.v_z / 3600.0
                        + _pinsp * _cp_p.tau_insp / 3600.0)
        return float(_e)
    _cp_old = BP._candidate_from_physics
    _cp_calls = {"n": 0}
    def _cp_stub(oi0, opt0, sequence0, h0, p0, xi0, w0, *args, **kwargs):
        _cp_calls["n"] += 1
        _seq = tuple(str(_t.tid) for _t in sequence0)
        _e = _cp_service_energy(sequence0) + 5.0
        return dict(ordered_tids=_seq, tids=_seq, tau=float(opt0.tau_min),
                    h=float(h0), E_plan_Wh=float(_e),
                    E_soc_required_Wh=float(_e),
                    resource_intervals={"deck": (), "active": (0.0, float(h0))})
    BP._candidate_from_physics = _cp_stub
    try:
        # No fixed-coverage dual: positive mandatory service energy closes every
        # depth-1 subtree without one whole-route physical evaluation.
        _cp_plain = BP._exact_pricing_search(
            _cp_turbs, [opt], _cp_p, _bpc_xi(), None, 60.0, 3,
            SimpleNamespace(branch=BP.BranchState()), set(), "energy",
            [], [], np.asarray([]), np.asarray([]), None, 1e-7,
            0.0, 0.0, "interval", 2.5, search_goal="certification",
            certified_prefix_pruning=False, physical_cache={}, batch_size=512)
        _cp_plain_calls = int(_cp_calls["n"])
        _cp_calls["n"] = 0
        _cp_pruned = BP._exact_pricing_search(
            _cp_turbs, [opt], _cp_p, _bpc_xi(), None, 60.0, 3,
            SimpleNamespace(branch=BP.BranchState()), set(), "energy",
            [], [], np.asarray([]), np.asarray([]), None, 1e-7,
            0.0, 0.0, "interval", 2.5, search_goal="certification",
            certified_prefix_pruning=True, physical_cache={}, batch_size=512)
        assert _cp_plain.complete and _cp_plain.closed and _cp_plain_calls > 0
        assert _cp_pruned.complete and _cp_pruned.closed
        assert _cp_pruned.certified_prefix_pruning_enabled
        assert _cp_pruned.certified_prefix_prunes == len(_cp_turbs)
        assert _cp_pruned.depth_certified_prefix_prunes == {1: len(_cp_turbs)}
        assert _cp_calls["n"] == 0 and _cp_pruned.physical_cache_misses == 0
        assert not _cp_pruned.columns and not _cp_plain.columns

        # Cardinality-aware Stage-2 oracle: over several coverage-dual signs and
        # magnitudes, pruning must return exactly the same strict-negative
        # columns as exhaustive enumeration under the service-energy premise.
        for _lam in (-200.0, -10.0, 0.0, 40.0, 70.0, 90.0, 150.0):
            _args = dict(
                turbines=_cp_turbs, launch_opts=[opt], p=_cp_p,
                xi_amb=_bpc_xi(), weather_unc=None, T_min=60.0, max_stops=3,
                node=SimpleNamespace(branch=BP.BranchState()),
                existing_signatures=set(), stage="energy",
                inequality_rows=[], equality_rows=[("coverage", None)],
                inequality_duals=np.asarray([]),
                equality_duals=np.asarray([_lam]), deadline=None,
                pricing_epsilon=1e-7, t_launch_min=0.0,
                landing_clear_min=0.0, deck_mode="interval",
                deck_delta_min=2.5, search_goal="certification", batch_size=512)
            _full = BP._exact_pricing_search(
                **_args, certified_prefix_pruning=False, physical_cache={})
            _fast = BP._exact_pricing_search(
                **_args, certified_prefix_pruning=True, physical_cache={})
            assert _full.complete and _fast.complete
            assert {BP._exact_route_signature(_c) for _c in _full.columns} == {
                BP._exact_route_signature(_c) for _c in _fast.columns}
            assert _fast.evaluated_sequences <= _full.evaluated_sequences
            if _full.closed:
                assert _fast.closed
    finally:
        BP._candidate_from_physics = _cp_old

    # Discovery remains heuristic-first and must never acquire certificate
    # pruning semantics from this flag.
    _cp_disc = BP._exact_pricing_search(
        _cp_turbs, [], _cp_p, _bpc_xi(), None, 60.0, 3,
        SimpleNamespace(branch=BP.BranchState()), set(), "energy",
        [], [], np.asarray([]), np.asarray([]), None, 1e-7,
        0.0, 0.0, "interval", 2.5, implicit_test_columns=[],
        search_goal="discovery", certified_prefix_pruning=True)
    assert not _cp_disc.certified_prefix_pruning_enabled

    # required_arc shadow interval: i at the current tail can still extend to j.
    arc_branch = BP.BranchState(required_arcs=frozenset({("PS-A", "PS-B")}))
    arc_desc = ("required_arc", ("PS-A", "PS-B"))
    assert BP._prefix_future_column_coefficient_range(
        arc_desc, 3, row_family="equality", prefix_tids=("PS-A",),
        opt=opt, branch=arc_branch) == (0.0, 1.0)
    assert BP._prefix_future_column_coefficient_range(
        arc_desc, 3, row_family="equality", prefix_tids=("PS-A", "PS-B"),
        opt=opt, branch=arc_branch) == (1.0, 1.0)
    assert BP._prefix_future_column_coefficient_range(
        arc_desc, 3, row_family="equality", prefix_tids=("PS-B",),
        opt=opt, branch=arc_branch) == (0.0, 0.0)
    assert BP._prefix_future_column_coefficient_range(
        arc_desc, 3, row_family="equality", prefix_tids=("PS-A", "PS-C"),
        opt=opt, branch=arc_branch) == (0.0, 0.0)
    assert BP._prefix_future_column_coefficient_range(
        arc_desc, 2, row_family="equality", prefix_tids=("PS-C",),
        opt=opt, branch=arc_branch) == (0.0, 0.0)

    # Formal closure uses exact mathematical zero, never -epsilon.
    assert not BP.PricingSearchResult(
        [], True, None, -1e-15, True, 0, 0, "probe").closed
    assert BP.PricingSearchResult(
        [], True, None, 0.0, True, 0, 0, "probe").closed

    # Discovery early return must be explicitly incomplete and never CLOSED.
    node = SimpleNamespace(branch=BP.BranchState())
    dcols = [_bpc_test_column(("PS-A",), 0.0, 5.0, 10.0),
             _bpc_test_column(("PS-B",), 0.0, 5.0, 11.0)]
    dr = BP._exact_pricing_search(
        _bpc_turbines("PS-A", "PS-B"), [], M.Params(), _bpc_xi(), None, 60.0,
        1, node, set(), "coverage", [], [], [], [], None, 1e-7,
        0.0, 0.0, "interval", 2.5,
        implicit_test_columns=dcols, batch_size=1,
        search_goal="discovery", shadow_prefix_bounds=True,
        discovery_column_limit=1, guided_discovery_order=True)
    assert dr.discovery_early_return and not dr.complete and not dr.closed
    assert dr.termination_reason == "discovery-improving-batch-found"
    assert len(dr.columns) == 1

    # V10: resource-pattern cut telemetry is observational only.  Construct a
    # singleton whose strict negativity depends on the outside-route (-1)
    # coefficient of one active exact-pattern cut.  The formal rc_ub<0 admission
    # is unchanged; the diagnostic must merely report the sign dependence.
    cut_col = _bpc_test_column(("PS-A",), 0.0, 5.0, 10.0)
    fake_old_sig = ("old-pattern-signature",)
    cut_dr = BP._exact_pricing_search(
        _bpc_turbines("PS-A"), [], M.Params(), _bpc_xi(), None, 60.0,
        1, node, set(), "coverage",
        [("packing", "PS-A"), ("resource_pattern", frozenset({fake_old_sig}))],
        [], np.asarray([-1.5, -2.0]), np.asarray([]), None, 1e-7,
        0.0, 0.0, "interval", 2.5,
        implicit_test_columns=[cut_col], batch_size=1,
        search_goal="discovery", discovery_column_limit=1,
        pattern_cut_diagnostics=True)
    assert len(cut_dr.columns) == 1
    assert cut_dr.pattern_cut_returned_count == 1
    assert cut_dr.pattern_cut_returned_sign_essential == 1
    assert cut_dr.pattern_cut_improving_seen_sign_essential == 1
    assert cut_dr.pattern_cut_returned_by_depth[1]["sign_essential"] == 1
    assert cut_dr.pattern_cut_returned_contribution_sum < 0.0

    # V6: four diverse strict-negative columns are returned together.  The
    # discovery result remains incomplete, so batching cannot manufacture closure.
    v6cols = [
        _bpc_test_column(("PS-A",), 0.0, 5.0, 10.0),
        _bpc_test_column(("PS-B",), 1.0, 5.0, 11.0),
        _bpc_test_column(("PS-A", "PS-B"), 2.0, 5.0, 12.0),
        _bpc_test_column(("PS-C",), 3.0, 5.0, 13.0),
        _bpc_test_column(("PS-A",), 4.0, 5.0, 14.0),
        _bpc_test_column(("PS-B",), 5.0, 5.0, 15.0),
    ]
    for idx, col in enumerate(v6cols):
        col["launch_option_index"] = idx % 3
    v6dr = BP._exact_pricing_search(
        _bpc_turbines("PS-A", "PS-B", "PS-C"), [], M.Params(),
        _bpc_xi(), None, 60.0, 2, node, set(), "coverage",
        [], [], [], [], None, 1e-7, 0.0, 0.0, "interval", 2.5,
        implicit_test_columns=v6cols, batch_size=16,
        search_goal="discovery", shadow_prefix_bounds=True,
        discovery_column_limit=4, guided_discovery_order=True,
        layered_discovery_order=True, adaptive_discovery_batch=True,
        discovery_batch_hard_cap=6,
        discovery_min_distinct_launches=2,
        discovery_min_distinct_service_sets=2)
    assert v6dr.discovery_early_return and not v6dr.complete and not v6dr.closed
    assert v6dr.termination_reason == "discovery-diverse-batch-found"
    assert len(v6dr.columns) == 4
    assert v6dr.improving_columns_seen == 4
    assert v6dr.discovery_improving_columns_returned == 4
    assert v6dr.discovery_diversity_satisfied
    assert not v6dr.discovery_hard_cap_triggered
    assert v6dr.discovery_distinct_launches >= 2
    assert v6dr.discovery_distinct_service_sets >= 2
    assert v6dr.depth_prefixes_evaluated == {1: 3, 2: 1}
    assert v6dr.depth_improving_seen == {1: 3, 2: 1}
    assert v6dr.depth_improving_returned == {1: 3, 2: 1}

    # If diversity cannot be achieved, V6 must not scan forever waiting for it:
    # at the hard cap it returns *all* already-proved negative columns.
    capcols = []
    for idx in range(6):
        col = _bpc_test_column(("PS-A",), float(idx), 5.0, 20.0 + idx)
        col["launch_option_index"] = 0
        capcols.append(col)
    capdr = BP._exact_pricing_search(
        _bpc_turbines("PS-A"), [], M.Params(), _bpc_xi(), None, 60.0,
        1, node, set(), "coverage", [], [], [], [], None, 1e-7,
        0.0, 0.0, "interval", 2.5,
        implicit_test_columns=capcols, batch_size=16,
        search_goal="discovery", shadow_prefix_bounds=True,
        discovery_column_limit=4, guided_discovery_order=True,
        layered_discovery_order=True, adaptive_discovery_batch=True,
        discovery_batch_hard_cap=6,
        discovery_min_distinct_launches=2,
        discovery_min_distinct_service_sets=2)
    assert capdr.discovery_early_return and not capdr.complete and not capdr.closed
    assert capdr.termination_reason == "discovery-batch-hard-cap-found"
    assert capdr.improving_columns_seen == 6
    assert capdr.discovery_improving_columns_returned == 6
    assert len(capdr.columns) == 6
    assert capdr.discovery_hard_cap_triggered
    assert not capdr.discovery_diversity_satisfied

    # V7 primal refresh: adding an exact-resource-feasible disjoint route must
    # improve the incumbent, while every accepted trial is audited.
    refresh_p = M.Params()
    refresh_cols = [
        _bpc_test_column(("PS-A",), 0.0, 5.0, 10.0),
        _bpc_test_column(("PS-B",), 20.0, 5.0, 11.0),
    ]
    for _c in refresh_cols:
        _c["resource_intervals"] = RA._resource_intervals(
            _c, 2.5, refresh_p.landing_clear_min, 6.0,
            deck_mode="interval", deck_delta_min=2.5)
    base_audit = BP._audit_integer_selection(
        refresh_cols, (0,), 2, 2, refresh_p, 1.0, 6.0, 1, 1, None)
    assert base_audit.status is RA.ResourceAuditStatus.FEASIBLE
    rr = BP._primal_refresh_incumbent(
        refresh_cols, (0,), base_audit, "coverage", None,
        2, 2, refresh_p, 1.0, 6.0, 1, 1, None,
        wall_budget_s=2.0, max_audits=16)
    assert rr["improved"] and rr["coverage"] == 2
    assert set(rr["selection"]) == {0, 1}
    assert rr["audit"] is not None
    assert rr["audit"].status is RA.ResourceAuditStatus.FEASIBLE
    assert rr["audit_calls"] >= 1
    assert (
        rr["augmentation_audits"]
        + rr["rebuild_audits"]
        + rr["repair_audits"] == rr["audit_calls"])
    assert (
        rr["augmentation_improvements"]
        + rr["rebuild_improvements"]
        + rr["repair_improvements"] >= 1)

    # V15 unified resource-aware exchange: an existing 2-stop archive route can
    # consolidate two selected singleton missions, freeing one half-cap battery
    # slot that is deterministically refilled by an uncovered singleton.  The
    # resulting coverage-4 incumbent is accepted only after the unchanged exact
    # resource audit; legacy augmentation/rebuild/repair are not used.
    ex_p = M.Params()
    ex_cols = [
        _bpc_test_column(("EX-A",), 0.0, 5.0, 200.0),
        _bpc_test_column(("EX-B",), 20.0, 5.0, 200.0),
        _bpc_test_column(("EX-C",), 40.0, 5.0, 200.0),
        _bpc_test_column(("EX-A", "EX-B"), 0.0, 25.0, 200.0),
        _bpc_test_column(("EX-D",), 50.0, 5.0, 200.0),
    ]
    for _c in ex_cols:
        _c["resource_intervals"] = RA._resource_intervals(
            _c, 2.5, ex_p.landing_clear_min, 6.0,
            deck_mode="interval", deck_delta_min=2.5)
    ex_base = BP._audit_integer_selection(
        ex_cols, (0, 1, 2), 2, 3, ex_p, 1.0, 6.0, 1, 1, None)
    assert ex_base.status is RA.ResourceAuditStatus.FEASIBLE
    ex_rr = BP._primal_refresh_incumbent(
        ex_cols, (0, 1, 2), ex_base, "coverage", None,
        2, 3, ex_p, 1.0, 6.0, 1, 1, None,
        wall_budget_s=2.0, max_audits=16,
        strategy="resource_exchange")
    assert ex_rr["improved"] and ex_rr["coverage"] == 4, ex_rr
    assert 3 in ex_rr["selection"] and 4 in ex_rr["selection"], ex_rr
    assert ex_rr["audit"].status is RA.ResourceAuditStatus.FEASIBLE
    assert ex_rr["exchange_candidate_routes"] >= 1
    assert ex_rr["exchange_consolidation_trials"] >= 1
    assert ex_rr["exchange_audits"] >= 1
    assert ex_rr["exchange_improvements"] >= 1
    assert ex_rr["augmentation_audits"] == 0
    assert ex_rr["rebuild_audits"] == 0
    assert ex_rr["repair_audits"] == 0
    print("V15 resource-aware exchange：2-stop consolidation + slot refill + exact audit → PASS")

    # V16 unified resource primal regression: preserve a selected 2-stop anchor
    # and fill genuinely free half-cap battery slots before attempting exchange.
    # This directly guards the real n=10 V15 regression where coverage fell to
    # five because the exchange-only strategy performed zero audits.
    rp_p = M.Params()
    rp_cols = [
        _bpc_test_column(("RP-A", "RP-B"), 0.0, 25.0, 200.0),
        _bpc_test_column(("RP-C",), 30.0, 5.0, 200.0),
        _bpc_test_column(("RP-D",), 40.0, 5.0, 200.0),
        _bpc_test_column(("RP-E",), 50.0, 5.0, 200.0),
    ]
    for _c in rp_cols:
        _c["resource_intervals"] = RA._resource_intervals(
            _c, 2.5, rp_p.landing_clear_min, 6.0,
            deck_mode="interval", deck_delta_min=2.5)
    rp_base = BP._audit_integer_selection(
        rp_cols, (0, 1), 2, 4, rp_p, 1.0, 6.0, 1, 1, None)
    assert rp_base.status is RA.ResourceAuditStatus.FEASIBLE
    rp_rr = BP._primal_refresh_incumbent(
        rp_cols, (0, 1), rp_base, "coverage", None,
        2, 4, rp_p, 1.0, 6.0, 1, 1, None,
        wall_budget_s=2.0, max_audits=16,
        strategy="resource_primal")
    assert rp_rr["improved"] and rp_rr["coverage"] == 5, rp_rr
    assert set(rp_rr["selection"]) == {0, 1, 2, 3}, rp_rr
    assert rp_rr["audit"].status is RA.ResourceAuditStatus.FEASIBLE
    assert rp_rr["augmentation_audits"] >= 2, rp_rr
    assert rp_rr["augmentation_improvements"] >= 2, rp_rr
    assert rp_rr["exchange_candidate_routes"] == 0, rp_rr
    assert rp_rr["rebuild_audits"] == 0 and rp_rr["repair_audits"] == 0
    print("V16 unified resource primal：preserve 2-stop anchor + exact-audited augmentation → PASS")

    # V17 resource-guided primal regression 1: fair ordering across uncovered
    # turbines must prevent several cheap-but-incompatible variants of one
    # turbine from consuming the entire short augmentation budget.
    rg_p = M.Params()
    rg_cols = [
        _bpc_test_column(("RG-A",), 10.0, 5.0, 200.0),
        _bpc_test_column(("RG-B",), 10.0, 5.0, 1.0),
        _bpc_test_column(("RG-B",), 11.0, 5.0, 2.0),
        _bpc_test_column(("RG-B",), 12.0, 5.0, 3.0),
        _bpc_test_column(("RG-C",), 30.0, 5.0, 200.0),
    ]
    for _c in rg_cols:
        _c["resource_intervals"] = RA._resource_intervals(
            _c, 2.5, rg_p.landing_clear_min, 6.0,
            deck_mode="interval", deck_delta_min=2.5)
    rg_base = BP._audit_integer_selection(
        rg_cols, (0,), 2, 3, rg_p, 1.0, 6.0, 1, 1, None)
    assert rg_base.status is RA.ResourceAuditStatus.FEASIBLE
    rg_cache = set()
    rg_rr = BP._primal_refresh_incumbent(
        rg_cols, (0,), rg_base, "coverage", None,
        2, 3, rg_p, 1.0, 6.0, 1, 1, None,
        wall_budget_s=2.0, max_audits=4,
        strategy="resource_guided",
        infeasible_trial_cache=rg_cache)
    assert rg_rr["improved"] and rg_rr["coverage"] == 2, rg_rr
    assert 4 in rg_rr["selection"], rg_rr
    assert rg_rr["augmentation_audits"] == 2, rg_rr
    assert rg_rr["uncovered_fair_rounds"] >= 1, rg_rr
    assert int(rg_rr["failure_reasons"].get("deck_overlap", 0)) >= 1, rg_rr

    # V17 regression 2: an exact INFEASIBLE_PROVEN trial is cached across
    # refresh calls.  The second call must skip the identical exact selection
    # instead of spending another resource-audit call on it.
    rg_bad_cols = [
        _bpc_test_column(("RG-X",), 10.0, 5.0, 200.0),
        _bpc_test_column(("RG-Y",), 10.0, 5.0, 200.0),
    ]
    for _c in rg_bad_cols:
        _c["resource_intervals"] = RA._resource_intervals(
            _c, 2.5, rg_p.landing_clear_min, 6.0,
            deck_mode="interval", deck_delta_min=2.5)
    rg_bad_base = BP._audit_integer_selection(
        rg_bad_cols, (0,), 2, 2, rg_p, 1.0, 6.0, 1, 1, None)
    assert rg_bad_base.status is RA.ResourceAuditStatus.FEASIBLE
    rg_bad_cache = set()
    rg_bad_1 = BP._primal_refresh_incumbent(
        rg_bad_cols, (0,), rg_bad_base, "coverage", None,
        2, 2, rg_p, 1.0, 6.0, 1, 1, None,
        wall_budget_s=2.0, max_audits=4,
        strategy="resource_guided",
        infeasible_trial_cache=rg_bad_cache)
    assert rg_bad_1["audit_calls"] == 1, rg_bad_1
    assert len(rg_bad_cache) == 1, rg_bad_cache
    rg_bad_2 = BP._primal_refresh_incumbent(
        rg_bad_cols, (0,), rg_bad_base, "coverage", None,
        2, 2, rg_p, 1.0, 6.0, 1, 1, None,
        wall_budget_s=2.0, max_audits=4,
        strategy="resource_guided",
        infeasible_trial_cache=rg_bad_cache)
    assert rg_bad_2["audit_calls"] == 0, rg_bad_2
    assert rg_bad_2["duplicate_trials_skipped"] >= 1, rg_bad_2
    print("V17 resource-guided primal：uncovered fairness + exact-infeasible cache + failure telemetry → PASS")

    # V18 deck-guided primal regression: use the exact same fixed half-open
    # deck-conflict relation as the resource oracle to fail-fast a conflicting
    # exact variant without spending the scarce exact-audit budget, then try a
    # deck-compatible variant of the same uncovered turbine.
    dg_p = M.Params()
    dg_cols = [
        _bpc_test_column(("DG-A",), 10.0, 5.0, 200.0),
        _bpc_test_column(("DG-B",), 10.0, 5.0, 1.0),
        _bpc_test_column(("DG-B",), 30.0, 5.0, 200.0),
    ]
    for _c in dg_cols:
        _c["resource_intervals"] = RA._resource_intervals(
            _c, 2.5, dg_p.landing_clear_min, 6.0,
            deck_mode="interval", deck_delta_min=2.5)
    dg_base = BP._audit_integer_selection(
        dg_cols, (0,), 2, 2, dg_p, 1.0, 6.0, 1, 1, None)
    assert dg_base.status is RA.ResourceAuditStatus.FEASIBLE
    dg_rr = BP._primal_refresh_incumbent(
        dg_cols, (0,), dg_base, "coverage", None,
        2, 2, dg_p, 1.0, 6.0, 1, 1, None,
        wall_budget_s=2.0, max_audits=1,
        strategy="resource_deck_guided",
        infeasible_trial_cache=set())
    assert dg_rr["improved"] and dg_rr["coverage"] == 2, dg_rr
    assert 2 in dg_rr["selection"], dg_rr
    assert dg_rr["audit_calls"] == 1, dg_rr
    assert dg_rr["deck_candidate_positive_conflict"] >= 1, dg_rr
    assert dg_rr["deck_candidate_zero_conflict"] >= 1, dg_rr
    assert dg_rr["deck_archive_conflict_edges"] >= 1, dg_rr
    assert dg_rr["deck_archive_max_degree"] >= 1, dg_rr
    assert dg_rr["deck_archive_max_component"] >= 2, dg_rr
    assert int(dg_rr["failure_reasons"].get("deck_overlap", 0)) == 0, dg_rr

    # If only a conflicting improvement candidate exists, the same exact deck
    # relation must reject it before resource DFS and therefore spend zero audit.
    dg_conflict_only = BP._primal_refresh_incumbent(
        dg_cols[:2], (0,), dg_base, "coverage", None,
        2, 2, dg_p, 1.0, 6.0, 1, 1, None,
        wall_budget_s=2.0, max_audits=1,
        strategy="resource_deck_guided",
        infeasible_trial_cache=set())
    assert not dg_conflict_only["improved"], dg_conflict_only
    assert dg_conflict_only["audit_calls"] == 0, dg_conflict_only
    assert dg_conflict_only["deck_prefilter_skips"] >= 1, dg_conflict_only
    assert dg_conflict_only["deck_conflict_pairs_sample"], dg_conflict_only
    print("V18 deck-guided resource primal：exact deck ordering/prefilter + conflict graph + one-audit compatible variant → PASS")

    # V19 heuristic-only adaptive two-stop exact-variant enrichment regression.
    # The merge helper may return a physically validated non-pricing column, but
    # it is isolated from all formal pricing-closure semantics.
    v19_p = M.Params()
    v19_turbs = _bpc_turbines("V19-A", "V19-B")
    v19_archive = [
        dict(_bpc_test_column(("V19-A",), 10.0, 5.0, 120.0),
             launch_option_index=0),
        dict(_bpc_test_column(("V19-B",), 20.0, 5.0, 125.0),
             launch_option_index=0),
    ]
    v19_existing = {BP._exact_route_signature(c) for c in v19_archive}
    v19_orig_candidate = BP._candidate_from_physics
    def _v19_fake_candidate(opt_index, opt, sequence, h, p, xi_amb, weather_unc,
                            t_launch_min, landing_clear_min, deck_mode,
                            deck_delta_min, **kwargs):
        tids = tuple(str(t.tid) for t in sequence)
        if tids != ("V19-A", "V19-B"):
            return None
        c = _bpc_test_column(tids, 12.5, float(h), 180.0)
        c["launch_option_index"] = int(opt_index)
        c["resource_intervals"] = RA._resource_intervals(
            c, 2.5, p.landing_clear_min, 6.0,
            deck_mode="interval", deck_delta_min=2.5)
        return c
    try:
        BP._candidate_from_physics = _v19_fake_candidate
        v19_merge = BP._adaptive_multistop_merge_enrichment(
            archive=v19_archive, turbines=v19_turbs,
            launch_opts=[SimpleNamespace()], p=v19_p, xi_amb=_bpc_xi(),
            weather_unc=None, T_min=60.0,
            node=SimpleNamespace(branch=BP.BranchState()),
            existing_signatures=v19_existing, incumbent_selection=(0,),
            inequality_rows=[], equality_rows=[],
            inequality_duals=np.asarray([], float),
            equality_duals=np.asarray([], float),
            deadline=None, t_launch_min=2.5,
            landing_clear_min=v19_p.landing_clear_min,
            deck_mode="interval", deck_delta_min=2.5,
            attempt_limit=8, batch_target=2, wall_budget_s=2.0)
    finally:
        BP._candidate_from_physics = v19_orig_candidate
    assert v19_merge["attempts"] >= 1, v19_merge
    assert v19_merge["physical_feasible"] >= 1, v19_merge
    assert v19_merge["new_candidates"] >= 1, v19_merge
    assert len(v19_merge["columns"]) >= 1, v19_merge
    assert tuple(BP._ordered_tids(v19_merge["columns"][0])) == (
        "V19-A", "V19-B"), v19_merge
    print("V19 adaptive multi-stop enrichment：bounded exact-variant merge + physical firewall → PASS")

    # V20 heuristic-only exact singleton timing-variant enrichment regression.
    # One uncovered turbine already has a route-local feasible singleton at a
    # deck-conflicting launch state.  V20 must discover a new deck-compatible
    # exact timing variant without changing any formal pricing semantics.
    v20_p = M.Params()
    v20_turbs = _bpc_turbines("V20-A", "V20-B")
    v20_a = dict(_bpc_test_column(("V20-A",), 10.0, 5.0, 120.0),
                   launch_option_index=0)
    v20_b_old = dict(_bpc_test_column(("V20-B",), 10.0, 5.0, 121.0),
                       launch_option_index=0)
    v20_a["resource_intervals"] = RA._resource_intervals(
        v20_a, 2.5, v20_p.landing_clear_min, 6.0,
        deck_mode="interval", deck_delta_min=2.5)
    v20_b_old["resource_intervals"] = RA._resource_intervals(
        v20_b_old, 2.5, v20_p.landing_clear_min, 6.0,
        deck_mode="interval", deck_delta_min=2.5)
    v20_archive = [v20_a, v20_b_old]
    v20_existing = {BP._exact_route_signature(c) for c in v20_archive}
    v20_opts = [
        SimpleNamespace(tau_min=10.0),
        SimpleNamespace(tau_min=30.0),
    ]
    v20_orig_candidate = BP._candidate_from_physics
    def _v20_fake_candidate(opt_index, opt, sequence, h, p, xi_amb, weather_unc,
                            t_launch_min, landing_clear_min, deck_mode,
                            deck_delta_min, **kwargs):
        tids = tuple(str(t.tid) for t in sequence)
        if tids != ("V20-B",) or int(opt_index) != 1:
            return None
        c = _bpc_test_column(tids, float(opt.tau_min), float(h), 122.0)
        c["launch_option_index"] = int(opt_index)
        c["resource_intervals"] = RA._resource_intervals(
            c, t_launch_min, landing_clear_min, 6.0,
            deck_mode=deck_mode, deck_delta_min=deck_delta_min)
        return c
    try:
        BP._candidate_from_physics = _v20_fake_candidate
        v20_rv = BP._resource_aware_singleton_variant_enrichment(
            archive=v20_archive, turbines=v20_turbs,
            launch_opts=v20_opts, p=v20_p, xi_amb=_bpc_xi(),
            weather_unc=None, T_min=60.0,
            node=SimpleNamespace(branch=BP.BranchState()),
            existing_signatures=v20_existing, incumbent_selection=(0,),
            inequality_rows=[], equality_rows=[],
            inequality_duals=np.asarray([], float),
            equality_duals=np.asarray([], float),
            deadline=None, t_launch_min=2.5,
            landing_clear_min=v20_p.landing_clear_min,
            deck_mode="interval", deck_delta_min=2.5,
            attempt_limit=4, batch_target=2, wall_budget_s=2.0,
            physical_cache={})
    finally:
        BP._candidate_from_physics = v20_orig_candidate
    assert v20_rv["attempts"] >= 1, v20_rv
    assert v20_rv["deck_compatible_specs"] >= 1, v20_rv
    assert v20_rv["physical_feasible"] >= 1, v20_rv
    assert v20_rv["new_candidates"] >= 1, v20_rv
    assert len(v20_rv["columns"]) >= 1, v20_rv
    assert tuple(BP._ordered_tids(v20_rv["columns"][0])) == ("V20-B",), v20_rv
    assert float(v20_rv["columns"][0]["tau"]) == 30.0, v20_rv
    assert len(v20_rv.get("records", [])) == len(v20_rv["columns"]), v20_rv
    assert v20_rv["records"][0]["tid"] == "V20-B", v20_rv
    assert float(v20_rv["records"][0]["tau"]) == 30.0, v20_rv
    assert "signature_repr" in v20_rv["records"][0], v20_rv
    print("V20 resource-aware exact variants：uncovered-turbine fairness + deck-compatible timing enrichment → PASS")

    # V8 fixed-archive diagnostic is exact only inside the frozen archive.
    diag_archive = []
    diag_sig = {}
    BP._add_columns(diag_archive, diag_sig, [dict(c) for c in refresh_cols])
    diag = BP._diagnose_fixed_archive_coverage(
        turbines=_bpc_turbines("PS-A", "PS-B"),
        launch_opts=[], p=refresh_p, xi_amb=_bpc_xi(),
        K=2, batteries=2, T_min=60.0, max_stops=1,
        weather_unc=None, archive=diag_archive,
        signature_to_index=diag_sig, no_good_cuts=[],
        initial_selection=(0,), initial_audit=base_audit,
        t_launch_min=2.5,
        landing_clear_min=refresh_p.landing_clear_min,
        quick_min=1.0, swap_min=6.0,
        quick_capacity=1, swap_capacity=1,
        deck_mode="interval", deck_delta_min=2.5,
        kappa_mode="vp_unimodal", chance_mode="drcc",
        budget_gamma=2.0, time_limit_s=15.0)
    assert diag["scope"] == (
        "fixed-generated-column-archive-only-not-full-route-space")
    assert diag["optimal_proven"], diag
    assert diag["exact_optimum"] == 2, diag
    assert diag["coverage_lower_bound"] == 2
    assert diag["coverage_upper_bound"] == 2
    assert len(diag["witness_selection_indices"]) >= 1, diag
    assert set(diag["witness_covered_turbines"]) == {"PS-A", "PS-B"}, diag
    assert len(diag["witness_route_signatures"]) == len(
        diag["witness_selection_indices"]), diag

    # V20.1 post-formal exact variant diagnostic is observational only.
    v201_cols = [
        _bpc_test_column(("V201-A",), 0.0, 5.0, 10.0),
        _bpc_test_column(("V201-B",), 20.0, 5.0, 11.0),
    ]
    for _c in v201_cols:
        _c["resource_intervals"] = RA._resource_intervals(
            _c, 2.5, refresh_p.landing_clear_min, 6.0,
            deck_mode="interval", deck_delta_min=2.5)
    v201_base = BP._audit_integer_selection(
        v201_cols, (0,), 2, 2, refresh_p, 1.0, 6.0, 1, 1, None)
    assert v201_base.status is RA.ResourceAuditStatus.FEASIBLE
    v201_rec = dict(
        tid="V201-B", ordered_tids=["V201-B"],
        tau=float(v201_cols[1]["tau"]), h=float(v201_cols[1]["h"]),
        signature_repr=repr(BP._exact_route_signature(v201_cols[1])),
        added_to_archive=True)
    v201_diag = BP._diagnose_resource_variant_postsolve(
        archive=v201_cols, final_selection=(0,), variant_records=[v201_rec],
        K=2, batteries=2, p=refresh_p, quick_min=1.0, swap_min=6.0,
        quick_capacity=1, swap_capacity=1, time_limit_s=2.0)
    assert v201_diag["enabled"] is True, v201_diag
    assert v201_diag["records_analyzed"] == 1, v201_diag
    assert v201_diag["direct_augmentation_audits"] == 1, v201_diag
    assert v201_diag["direct_augmentation_feasible"] == 1, v201_diag
    assert v201_diag["records"][0]["postsolve_status"] == (
        "direct-augmentation-feasible"), v201_diag
    print("V20.1 postsolve diagnostic：variant identity + exact direct augmentation audit → PASS")

    # V21 information-convergence path: an exact solve of the *currently
    # generated archive* may improve only the primal incumbent.  Its restricted
    # archive bound/certificate is never promoted to the full physical route
    # space.  Start deliberately from coverage 1 although the archive already
    # contains an exact-resource-feasible coverage-2 witness.
    v21_stage = BP._solve_branch_price_stage(
        stage="coverage",
        turbines=_bpc_turbines("PS-A", "PS-B"),
        launch_opts=[], p=refresh_p, xi_amb=_bpc_xi(),
        K=2, batteries=2, T_min=60.0, max_stops=1,
        weather_unc=None, deadline=time.monotonic() + 15.0,
        archive=[dict(c) for c in diag_archive],
        signature_to_index=dict(diag_sig), no_good_cuts=[],
        coverage_target=None, initial_selection=(0,),
        initial_audit=base_audit,
        implicit_test_columns=[dict(c) for c in diag_archive],
        quick_min=1.0, swap_min=6.0,
        quick_capacity=1, swap_capacity=1,
        formal_battery_halfcap=False,
        archive_primal_recovery=True,
        archive_primal_recovery_time_limit_s=10.0)
    assert v21_stage.archive_primal_recovery_enabled is True, v21_stage
    assert v21_stage.archive_primal_recovery_calls >= 1, v21_stage
    assert v21_stage.archive_primal_recovery_improvements >= 1, v21_stage
    assert v21_stage.archive_primal_recovery_best_coverage == 2, v21_stage
    assert v21_stage.coverage_incumbent == 2, v21_stage
    assert v21_stage.incumbent_audit.status is RA.ResourceAuditStatus.FEASIBLE
    assert v21_stage.archive_primal_recovery_records, v21_stage
    assert any(bool(_r.get("promoted"))
               for _r in v21_stage.archive_primal_recovery_records), v21_stage
    assert set(v21_stage.archive_primal_recovery_witness_covered_turbines) == {
        "PS-A", "PS-B"}, v21_stage
    assert len(v21_stage.archive_primal_recovery_witness_selection_indices) == 2
    print("V21 archive primal recovery：restricted-archive exact witness only promotes exact-audited incumbent + provenance telemetry → PASS")
    # Paper-facing canonical alias must normalize to the same proven formal
    # implementation while keeping all legacy mode strings backward compatible.
    _paper_contract = BP._validate_anytime_public_contract(
        solver_mode="exact-branch-price-cut",
        pricing_mode="r-bpc",
        kappa_mode="vp_unimodal",
        chance_mode="drcc",
        deck_mode="interval",
        battery_reuse_mode="exact_soc",
        solver="scipy-highs-rmp",
        pool_h_mode="pareto",
        time_limit_s=60.0,
        deadline=time.monotonic() + 60.0,
        budget_gamma=2.0,
        K=2,
        batteries=2,
        max_stops=4,
        coverage_gap_target_abs=0,
        energy_gap_target_rel=0.0,
        energy_gap_target_abs_Wh=0.0,
        pricing_batch_size=16,
        solve_scope="lexicographic")
    assert _paper_contract["pricing_mode"] == (
        "exact-layered-batch-primal-battery-halfcap-resource-variant-archive-recovery-formal"
    ), _paper_contract
    print("R-BPC canonical alias：paper CLI maps exactly to the proven archive-recovery formal implementation → PASS")
    assert callable(getattr(S13, "E1_lex_certify", None))
    assert S13._e1_certify_solver_kwargs(SimpleNamespace(
        time_limit_s=60.0, e1_certify_time_limit_s=60.0,
        archive_diagnostic_time_limit_s=0.0,
        archive_shadow_diagnostic_time_limit_s=0.0,
        archive_clique_diagnostic_time_limit_s=0.0,
        archive_primal_recovery="on",
        archive_primal_recovery_time_limit_s=2.0,
        fullspace_target_diagnostic_time_limit_s=0.0,
        pricing_mode="r-bpc"))["solve_scope"] == "lexicographic"
    print("E1_lex_certify：paper fixed-point entry reuses the unchanged full lexicographic R-BPC path → PASS")


    # V21 post-formal full-space target ladder is diagnostic-only.  On this tiny
    # fixture the initial frozen archive already contains target coverage 2; the
    # helper must recover that witness while mutating only its private copies.
    _v21_archive_before = list(diag_archive)
    _v21_sig_before = dict(diag_sig)
    v21_target_diag = BP._diagnose_fullspace_target_ladder(
        turbines=_bpc_turbines("PS-A", "PS-B"),
        launch_opts=[], p=refresh_p, xi_amb=_bpc_xi(),
        K=2, batteries=2, T_min=60.0, max_stops=1,
        weather_unc=None, archive=diag_archive,
        signature_to_index=diag_sig, no_good_cuts=[],
        initial_selection=(0,), initial_audit=base_audit,
        physical_cache={},
        t_launch_min=2.5,
        landing_clear_min=refresh_p.landing_clear_min,
        quick_min=1.0, swap_min=6.0,
        quick_capacity=1, swap_capacity=1,
        deck_mode="interval", deck_delta_min=2.5,
        kappa_mode="vp_unimodal", chance_mode="drcc",
        budget_gamma=2.0, time_limit_s=15.0,
        archive_primal_recovery_time_limit_s=10.0)
    assert v21_target_diag["enabled"] is True, v21_target_diag
    assert v21_target_diag["targets_attempted"] >= 1, v21_target_diag
    assert v21_target_diag["highest_feasible_target"] == 2, v21_target_diag
    assert v21_target_diag["best_coverage"] == 2, v21_target_diag
    assert len(diag_archive) == len(_v21_archive_before)
    assert diag_sig == _v21_sig_before
    print("V21 full-space target ladder diagnostic：target witness + copy-only isolation → PASS")

    # V9 target-only diagnostic must stop on an exact resource-feasible witness.
    tdiag = BP._diagnose_fixed_archive_target_min_coverage(
        turbines=_bpc_turbines("PS-A", "PS-B"),
        launch_opts=[], p=refresh_p, xi_amb=_bpc_xi(),
        K=2, batteries=2, T_min=60.0, max_stops=1,
        weather_unc=None, archive=diag_archive,
        signature_to_index=diag_sig, no_good_cuts=[],
        initial_selection=(0,), initial_audit=base_audit,
        t_launch_min=2.5,
        landing_clear_min=refresh_p.landing_clear_min,
        quick_min=1.0, swap_min=6.0,
        quick_capacity=1, swap_capacity=1,
        deck_mode="interval", deck_delta_min=2.5,
        kappa_mode="vp_unimodal", chance_mode="drcc",
        budget_gamma=2.0, target_min=2, time_limit_s=3.0)
    assert tdiag["status"] == "FEASIBLE", tdiag
    assert tdiag["feasible_proven"] and not tdiag["infeasible_proven"]
    assert tdiag["witness_coverage"] == 2

    # With only the PS-A column, target coverage 2 is exactly impossible inside
    # the frozen archive and must close without making any full-route claim.
    one_archive = [dict(diag_archive[0])]
    one_sig = {BP._exact_route_signature(one_archive[0]): 0}
    tdiag_no = BP._diagnose_fixed_archive_target_min_coverage(
        turbines=_bpc_turbines("PS-A", "PS-B"),
        launch_opts=[], p=refresh_p, xi_amb=_bpc_xi(),
        K=2, batteries=2, T_min=60.0, max_stops=1,
        weather_unc=None, archive=one_archive,
        signature_to_index=one_sig, no_good_cuts=[],
        initial_selection=(0,), initial_audit=base_audit,
        t_launch_min=2.5,
        landing_clear_min=refresh_p.landing_clear_min,
        quick_min=1.0, swap_min=6.0,
        quick_capacity=1, swap_capacity=1,
        deck_mode="interval", deck_delta_min=2.5,
        kappa_mode="vp_unimodal", chance_mode="drcc",
        budget_gamma=2.0, target_min=2, time_limit_s=3.0)
    assert tdiag_no["status"] == "INFEASIBLE_PROVEN", tdiag_no
    assert tdiag_no["infeasible_proven"] and not tdiag_no["feasible_proven"]

    # V10 post-target certificate ladder is isolated and exact on its own
    # necessary relaxations.  6+6+6 <= 2*10 passes pooled energy but requires
    # three indivisible route batteries; the first core must shadow-cover the
    # repeated second pattern without ever adding a formal cut.
    shadow_cols = []
    for i in range(3):
        shadow_cols.append(dict(
            ordered_tids=(f"S{i}",), tids=(f"S{i}",),
            E_soc_required_Wh=6.0, E_plan_Wh=6.0,
            resource_intervals=dict(
                launch_start_min=10.0 * i, clear_end_min=10.0 * i + 1.0,
                active=(10.0 * i, 10.0 * i + 1.0), deck=())))
    shadow = BP._analyze_rejected_patterns_certificate_shadow(
        archive=shadow_cols, selections=((0, 1, 2), (0, 1, 2)),
        K=3, batteries=2, usable_battery_energy_Wh=10.0,
        quick_min=1.0, swap_min=6.0, time_limit_s=2.0)
    assert shadow["analyzed_patterns"] == 2 and not shadow["timed_out"], shadow
    assert shadow["pooled_energy_infeasible_patterns"] == 0
    assert shadow["battery_binpack_infeasible_patterns"] == 2
    assert shadow["battery_min_required_counts"] == {"3": 2}, shadow
    assert shadow["battery_core_unique_count"] == 1
    assert shadow["battery_core_shadow_prior_cover_count"] == 1
    assert shadow["first_proof_layer_counts"]["battery_binpack"] == 2

    # V11: half-cap and low-energy-anchor battery clique rows are strict
    # necessary consequences of route-level battery assignment.  6+6+6 with
    # B=2,C=10 violates the half-cap row.  4+7+7 does not violate half-cap
    # (only two routes are >5), but the 4-Wh anchor plus both >6-Wh routes form
    # a 3-member pairwise-incompatible clique and therefore violate the
    # anchor row.  This is shadow-only and must not add formal cuts.
    clique_rows, clique_shadow = BP._battery_clique_diagnostic_rows_and_shadow(
        shadow_cols, ((0, 1, 2),), batteries=2,
        usable_battery_energy_Wh=10.0)
    assert clique_shadow["halfcap_rows"] == 1
    assert clique_shadow["anchor_rows"] == 0
    assert clique_shadow["rejected_halfcap_violations"] == 1
    assert clique_shadow["rejected_any_clique_violations"] == 1
    assert clique_shadow["rejected_uncovered_by_cliques"] == 0
    assert len(clique_rows) == 1
    assert all(BP._row_coefficient(c, clique_rows[0][0]) == 1.0
               for c in shadow_cols)

    anchor_cols = []
    for i, e in enumerate((4.0, 7.0, 7.0)):
        anchor_cols.append(dict(
            ordered_tids=(f"A{i}",), tids=(f"A{i}",),
            E_soc_required_Wh=e, E_plan_Wh=e,
            resource_intervals=dict(
                launch_start_min=10.0 * i, clear_end_min=10.0 * i + 1.0,
                active=(10.0 * i, 10.0 * i + 1.0), deck=())))
    anchor_rows, anchor_shadow = BP._battery_clique_diagnostic_rows_and_shadow(
        anchor_cols, ((0, 1, 2),), batteries=2,
        usable_battery_energy_Wh=10.0)
    assert anchor_shadow["rejected_halfcap_violations"] == 0
    assert anchor_shadow["rejected_anchor_only_violations"] == 1
    assert anchor_shadow["rejected_any_clique_violations"] == 1
    assert anchor_shadow["rejected_uncovered_by_cliques"] == 0
    assert anchor_shadow["anchor_rows"] == 1
    assert len(anchor_rows) == 2
    anchor_desc = [d for d, rhs in anchor_rows
                   if d[0] == "diagnostic_battery_anchor_clique"][0]
    assert [BP._row_coefficient(c, anchor_desc) for c in anchor_cols] == [
        1.0, 1.0, 1.0]

    # Diagnostic clique rows are intentionally absent from the universal
    # future-column pricing range registry.  If they ever leak into the formal
    # implicit-pricing path, certificate construction must fail closed.
    try:
        BP._future_column_coefficient_range(
            ("diagnostic_battery_halfcap", 10.0), 4,
            row_family="inequality")
        raise AssertionError("diagnostic clique row leaked into formal range registry")
    except ValueError:
        pass

    # V11 post-formal A/B rerun: target coverage 3 from three 6-Wh singleton
    # routes is impossible with B=2,C=10.  The baseline frozen-archive target
    # decision and the clique rerun must both prove NO; the latter must use only
    # diagnostic rows and cannot alter the baseline result/cuts.
    clique_p = M.Params()
    clique_p.B_k = 12.5
    clique_p.safe_reserve = 0.2  # exact binary64 model value gives B_use=10
    clique_cols = [
        _bpc_test_column((f"C{i}",), 20.0 * i, 5.0, 6.0)
        for i in range(3)]
    for _c in clique_cols:
        _c["resource_intervals"] = RA._resource_intervals(
            _c, 2.5, clique_p.landing_clear_min, 6.0,
            deck_mode="interval", deck_delta_min=2.5)
    clique_archive = []
    clique_sig = {}
    BP._add_columns(clique_archive, clique_sig, [dict(c) for c in clique_cols])
    clique_initial_audit = BP._audit_integer_selection(
        clique_archive, (0, 1), 3, 2, clique_p, 1.0, 6.0, 1, 1, None)
    assert clique_initial_audit.status is RA.ResourceAuditStatus.FEASIBLE
    clique_diag = BP._diagnose_fixed_archive_target_min_coverage(
        turbines=_bpc_turbines("C0", "C1", "C2"),
        launch_opts=[], p=clique_p, xi_amb=_bpc_xi(),
        K=3, batteries=2, T_min=60.0, max_stops=1,
        weather_unc=None, archive=clique_archive,
        signature_to_index=clique_sig, no_good_cuts=[],
        initial_selection=(0, 1), initial_audit=clique_initial_audit,
        t_launch_min=2.5, landing_clear_min=clique_p.landing_clear_min,
        quick_min=1.0, swap_min=6.0,
        quick_capacity=1, swap_capacity=1,
        deck_mode="interval", deck_delta_min=2.5,
        kappa_mode="vp_unimodal", chance_mode="drcc",
        budget_gamma=2.0, target_min=3, time_limit_s=3.0,
        certificate_shadow_time_limit_s=1.0,
        clique_rerun_time_limit_s=3.0)
    assert clique_diag["status"] == "INFEASIBLE_PROVEN", clique_diag
    assert clique_diag["battery_clique_shadow"]["rejected_halfcap_violations"] >= 1
    assert clique_diag["battery_clique_target_rerun"]["status"] == (
        "INFEASIBLE_PROVEN"), clique_diag["battery_clique_target_rerun"]
    assert clique_diag["battery_clique_target_rerun"]["clique_rows"] >= 1

    # V12 formal strengthening: the same half-cap clique is now permitted on
    # the implicit exact-pricing certificate path with a proved future-column
    # coefficient range [0,1].  Three 6-Wh singleton routes under B=2,C=10
    # must close coverage at 2 directly from the formal Master row, without a
    # resource-pattern cut.  The strict half-cap boundary uses exact binary64
    # rational semantics: E=C/2 has coefficient 0; nextafter(C/2,+inf) has 1.
    assert BP._future_column_coefficient_range(
        ("battery_halfcap", 10.0), 1,
        row_family="inequality") == (0.0, 1.0)
    _boundary = dict(clique_cols[0])
    _boundary["E_soc_required_Wh"] = 5.0
    assert BP._row_coefficient(
        _boundary, ("battery_halfcap", 10.0)) == 0.0
    _boundary["E_soc_required_Wh"] = math.nextafter(5.0, math.inf)
    assert BP._row_coefficient(
        _boundary, ("battery_halfcap", 10.0)) == 1.0
    formal_halfcap_stage = BP._solve_branch_price_stage(
        stage="coverage",
        turbines=_bpc_turbines("C0", "C1", "C2"),
        launch_opts=[], p=clique_p, xi_amb=_bpc_xi(),
        K=3, batteries=2, T_min=60.0, max_stops=1,
        weather_unc=None, deadline=time.monotonic() + 5.0,
        archive=[dict(c) for c in clique_archive],
        signature_to_index=dict(clique_sig),
        no_good_cuts=[], coverage_target=None,
        initial_selection=(0, 1), initial_audit=clique_initial_audit,
        implicit_test_columns=[dict(c) for c in clique_cols],
        quick_min=1.0, swap_min=6.0,
        quick_capacity=1, swap_capacity=1,
        formal_battery_halfcap=True)
    assert formal_halfcap_stage.optimal, formal_halfcap_stage
    assert formal_halfcap_stage.coverage_incumbent == 2, formal_halfcap_stage
    assert int(BP._safe_integer_floor(
        formal_halfcap_stage.global_bound)) == 2, formal_halfcap_stage
    assert formal_halfcap_stage.resource_cuts_added == 0, formal_halfcap_stage
    assert formal_halfcap_stage.battery_halfcap_dual_active_rmp_solves >= 1
    assert formal_halfcap_stage.battery_halfcap_dual_abs_sum > 0.0
    assert BP._future_row_range_contract_self_check(4) is True

    # Exercise the public algorithm-mode wiring as well: the private synthetic
    # fixture uses the same validated exact BPC contract but is explicitly
    # barred from physical global certification.  V12 must close 2/3 with the
    # formal row and expose the new telemetry without generating a no-good cut.
    formal_halfcap_top = BP._solve_fleet_anytime_synthetic_fixture(
        _bpc_turbines("C0", "C1", "C2"), [], clique_p, _bpc_xi(),
        3, 60.0, batteries=2, max_stops=1, time_limit_s=5.0,
        allow_resource_only_columns=True,
        implicit_test_columns=[dict(c) for c in clique_cols],
        pricing_mode="exact-layered-batch-primal-battery-halfcap-formal",
        coverage_gap_target_abs=0,
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0,
        quick_inspection_capacity=3, swap_station_capacity=3)
    assert formal_halfcap_top["battery_halfcap_formal_enabled"] is True
    assert formal_halfcap_top["coverage_incumbent"] == 2
    assert formal_halfcap_top["coverage_upper_bound"] == 2
    assert formal_halfcap_top["coverage_optimal"] is True
    assert formal_halfcap_top["battery_halfcap_archive_high_energy_routes"] == 3
    assert formal_halfcap_top["coverage_battery_halfcap_dual_active_rmp_solves"] >= 1
    assert formal_halfcap_top["resource_pattern_cuts_added"] == 0
    assert formal_halfcap_top["physical_model_global_certificate"] is False
    print("V12 formal half-cap battery clique：严格阈值/未来列区间/正式 mode wiring/coverage=2 closure → PASS")

    formal_halfcap_v13_top = BP._solve_fleet_anytime_synthetic_fixture(
        _bpc_turbines("C0", "C1", "C2"), [], clique_p, _bpc_xi(),
        3, 60.0, batteries=2, max_stops=1, time_limit_s=5.0,
        allow_resource_only_columns=True,
        implicit_test_columns=[dict(c) for c in clique_cols],
        pricing_mode=(
            "exact-layered-batch-primal-battery-halfcap-depth-fair-formal"),
        coverage_gap_target_abs=0,
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0,
        quick_inspection_capacity=3, swap_station_capacity=3)
    assert formal_halfcap_v13_top["battery_halfcap_formal_enabled"] is True
    assert formal_halfcap_v13_top["coverage_incumbent"] == 2
    assert formal_halfcap_v13_top["coverage_upper_bound"] == 2
    assert formal_halfcap_v13_top["coverage_optimal"] is True
    assert formal_halfcap_v13_top["pricing_depth_fair_requested_calls"] >= 1
    assert formal_halfcap_v13_top["pricing_depth_fair_active_calls"] >= 1
    assert formal_halfcap_v13_top["resource_pattern_cuts_added"] == 0
    assert formal_halfcap_v13_top["physical_model_global_certificate"] is False
    print("V13 depth-fair：正式 half-cap 保持、mode wiring、exact closure 与 discovery-only trigger → PASS")

    formal_halfcap_v14_top = BP._solve_fleet_anytime_synthetic_fixture(
        _bpc_turbines("C0", "C1", "C2"), [], clique_p, _bpc_xi(),
        3, 60.0, batteries=2, max_stops=1, time_limit_s=5.0,
        allow_resource_only_columns=True,
        implicit_test_columns=[dict(c) for c in clique_cols],
        pricing_mode=(
            "exact-layered-batch-primal-battery-halfcap-depth-fair-neutral-formal"),
        coverage_gap_target_abs=0,
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0,
        quick_inspection_capacity=3, swap_station_capacity=3)
    assert formal_halfcap_v14_top["battery_halfcap_formal_enabled"] is True
    assert formal_halfcap_v14_top["coverage_incumbent"] == 2
    assert formal_halfcap_v14_top["coverage_upper_bound"] == 2
    assert formal_halfcap_v14_top["coverage_optimal"] is True
    assert formal_halfcap_v14_top["resource_pattern_cuts_added"] == 0
    assert formal_halfcap_v14_top["pricing_multistop_neutral_added"] == 0
    assert formal_halfcap_v14_top["physical_model_global_certificate"] is False
    print("V14 neutral multi-stop：mode wiring / neutral firewall / exact closure 回归 → PASS")

    formal_halfcap_v15_top = BP._solve_fleet_anytime_synthetic_fixture(
        _bpc_turbines("C0", "C1", "C2"), [], clique_p, _bpc_xi(),
        3, 60.0, batteries=2, max_stops=1, time_limit_s=5.0,
        allow_resource_only_columns=True,
        implicit_test_columns=[dict(c) for c in clique_cols],
        pricing_mode=(
            "exact-layered-batch-primal-battery-halfcap-resource-exchange-formal"),
        coverage_gap_target_abs=0,
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0,
        quick_inspection_capacity=3, swap_station_capacity=3)
    assert formal_halfcap_v15_top["battery_halfcap_formal_enabled"] is True
    assert formal_halfcap_v15_top["primal_exchange_enabled"] is True
    assert formal_halfcap_v15_top["coverage_incumbent"] == 2
    assert formal_halfcap_v15_top["coverage_upper_bound"] == 2
    assert formal_halfcap_v15_top["coverage_optimal"] is True
    assert formal_halfcap_v15_top["pricing_multistop_neutral_added"] == 0
    assert formal_halfcap_v15_top["resource_pattern_cuts_added"] == 0
    assert formal_halfcap_v15_top["physical_model_global_certificate"] is False
    print("V15 architecture convergence：formal half-cap + V7 pricing + exact-audited exchange wiring → PASS")

    formal_halfcap_v16_top = BP._solve_fleet_anytime_synthetic_fixture(
        _bpc_turbines("C0", "C1", "C2"), [], clique_p, _bpc_xi(),
        3, 60.0, batteries=2, max_stops=1, time_limit_s=5.0,
        allow_resource_only_columns=True,
        implicit_test_columns=[dict(c) for c in clique_cols],
        pricing_mode=(
            "exact-layered-batch-primal-battery-halfcap-resource-primal-formal"),
        coverage_gap_target_abs=0,
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0,
        quick_inspection_capacity=3, swap_station_capacity=3)
    assert formal_halfcap_v16_top["battery_halfcap_formal_enabled"] is True
    assert formal_halfcap_v16_top["primal_exchange_enabled"] is True
    assert formal_halfcap_v16_top["coverage_incumbent"] == 2
    assert formal_halfcap_v16_top["coverage_upper_bound"] == 2
    assert formal_halfcap_v16_top["coverage_optimal"] is True
    assert formal_halfcap_v16_top["pricing_multistop_neutral_added"] == 0
    assert formal_halfcap_v16_top["resource_pattern_cuts_added"] == 0
    assert formal_halfcap_v16_top["physical_model_global_certificate"] is False
    print("V16 architecture fix：formal half-cap + V7 pricing + anchor-preserving resource primal wiring → PASS")

    formal_halfcap_v17_top = BP._solve_fleet_anytime_synthetic_fixture(
        _bpc_turbines("C0", "C1", "C2"), [], clique_p, _bpc_xi(),
        3, 60.0, batteries=2, max_stops=1, time_limit_s=5.0,
        allow_resource_only_columns=True,
        implicit_test_columns=[dict(c) for c in clique_cols],
        pricing_mode=(
            "exact-layered-batch-primal-battery-halfcap-resource-guided-formal"),
        coverage_gap_target_abs=0,
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0,
        quick_inspection_capacity=3, swap_station_capacity=3)
    assert formal_halfcap_v17_top["battery_halfcap_formal_enabled"] is True
    assert formal_halfcap_v17_top["primal_exchange_enabled"] is True
    assert formal_halfcap_v17_top["coverage_incumbent"] == 2
    assert formal_halfcap_v17_top["coverage_upper_bound"] == 2
    assert formal_halfcap_v17_top["coverage_optimal"] is True
    assert formal_halfcap_v17_top["pricing_multistop_neutral_added"] == 0
    assert formal_halfcap_v17_top["resource_pattern_cuts_added"] == 0
    assert isinstance(
        formal_halfcap_v17_top["primal_refresh_failure_reasons"], dict)
    assert formal_halfcap_v17_top["physical_model_global_certificate"] is False
    print("V17 architecture refinement：formal half-cap + V7 pricing + guided single resource primal wiring → PASS")

    formal_halfcap_v18_top = BP._solve_fleet_anytime_synthetic_fixture(
        _bpc_turbines("C0", "C1", "C2"), [], clique_p, _bpc_xi(),
        3, 60.0, batteries=2, max_stops=1, time_limit_s=5.0,
        allow_resource_only_columns=True,
        implicit_test_columns=[dict(c) for c in clique_cols],
        pricing_mode=(
            "exact-layered-batch-primal-battery-halfcap-deck-guided-formal"),
        coverage_gap_target_abs=0,
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0,
        quick_inspection_capacity=3, swap_station_capacity=3)
    assert formal_halfcap_v18_top["battery_halfcap_formal_enabled"] is True
    assert formal_halfcap_v18_top["primal_exchange_enabled"] is True
    assert formal_halfcap_v18_top["primal_deck_diagnostic_enabled"] is True
    assert formal_halfcap_v18_top["coverage_incumbent"] == 2
    assert formal_halfcap_v18_top["coverage_upper_bound"] == 2
    assert formal_halfcap_v18_top["coverage_optimal"] is True
    assert formal_halfcap_v18_top["pricing_multistop_neutral_added"] == 0
    assert formal_halfcap_v18_top["resource_pattern_cuts_added"] == 0
    assert formal_halfcap_v18_top["physical_model_global_certificate"] is False
    print("V18 architecture refinement：formal half-cap + V7 pricing + deck-guided single resource primal wiring → PASS")

    formal_halfcap_v19_top = BP._solve_fleet_anytime_synthetic_fixture(
        _bpc_turbines("C0", "C1", "C2"), [], clique_p, _bpc_xi(),
        3, 60.0, batteries=2, max_stops=1, time_limit_s=5.0,
        allow_resource_only_columns=True,
        implicit_test_columns=[dict(c) for c in clique_cols],
        pricing_mode=(
            "exact-layered-batch-primal-battery-halfcap-adaptive-multistop-formal"),
        coverage_gap_target_abs=0,
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0,
        quick_inspection_capacity=3, swap_station_capacity=3)
    assert formal_halfcap_v19_top["battery_halfcap_formal_enabled"] is True
    assert formal_halfcap_v19_top["primal_exchange_enabled"] is True
    assert formal_halfcap_v19_top["primal_deck_diagnostic_enabled"] is True
    assert formal_halfcap_v19_top["pricing_multistop_merge_enabled"] is True
    assert formal_halfcap_v19_top["coverage_incumbent"] == 2
    assert formal_halfcap_v19_top["coverage_upper_bound"] == 2
    assert formal_halfcap_v19_top["coverage_optimal"] is True
    assert formal_halfcap_v19_top["pricing_multistop_neutral_added"] == 0
    assert formal_halfcap_v19_top["resource_pattern_cuts_added"] == 0
    assert formal_halfcap_v19_top["physical_model_global_certificate"] is False
    print("V19 architecture refinement：formal half-cap + deck-guided primal + heuristic-only adaptive 2-stop enrichment wiring → PASS")

    formal_halfcap_v20_top = BP._solve_fleet_anytime_synthetic_fixture(
        _bpc_turbines("C0", "C1", "C2"), [], clique_p, _bpc_xi(),
        3, 60.0, batteries=2, max_stops=1, time_limit_s=5.0,
        allow_resource_only_columns=True,
        implicit_test_columns=[dict(c) for c in clique_cols],
        pricing_mode=(
            "exact-layered-batch-primal-battery-halfcap-resource-variant-formal"),
        coverage_gap_target_abs=0,
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0,
        quick_inspection_capacity=3, swap_station_capacity=3)
    assert formal_halfcap_v20_top["battery_halfcap_formal_enabled"] is True
    assert formal_halfcap_v20_top["primal_exchange_enabled"] is True
    assert formal_halfcap_v20_top["primal_deck_diagnostic_enabled"] is True
    assert formal_halfcap_v20_top["pricing_resource_variant_enabled"] is True
    assert formal_halfcap_v20_top["pricing_multistop_merge_enabled"] is False
    assert formal_halfcap_v20_top["coverage_incumbent"] == 2
    assert formal_halfcap_v20_top["coverage_upper_bound"] == 2
    assert formal_halfcap_v20_top["coverage_optimal"] is True
    assert formal_halfcap_v20_top["resource_pattern_cuts_added"] == 0
    assert formal_halfcap_v20_top["physical_model_global_certificate"] is False
    print("V20 architecture refinement：formal half-cap + deck-guided primal + heuristic-only resource-compatible exact variants wiring → PASS")

    formal_halfcap_v201_top = BP._solve_fleet_anytime_synthetic_fixture(
        _bpc_turbines("C0", "C1", "C2"), [], clique_p, _bpc_xi(),
        3, 60.0, batteries=2, max_stops=1, time_limit_s=5.0,
        allow_resource_only_columns=True,
        implicit_test_columns=[dict(c) for c in clique_cols],
        pricing_mode=(
            "exact-layered-batch-primal-battery-halfcap-resource-variant-diagnostic-formal"),
        coverage_gap_target_abs=0,
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0,
        quick_inspection_capacity=3, swap_station_capacity=3)
    assert formal_halfcap_v201_top["battery_halfcap_formal_enabled"] is True
    assert formal_halfcap_v201_top["primal_exchange_enabled"] is True
    assert formal_halfcap_v201_top["primal_deck_diagnostic_enabled"] is True
    assert formal_halfcap_v201_top["pricing_resource_variant_enabled"] is True
    assert formal_halfcap_v201_top["pricing_multistop_merge_enabled"] is False
    assert formal_halfcap_v201_top["coverage_incumbent"] == (
        formal_halfcap_v20_top["coverage_incumbent"])
    assert formal_halfcap_v201_top["coverage_upper_bound"] == (
        formal_halfcap_v20_top["coverage_upper_bound"])
    # Synthetic fixtures intentionally suppress post-formal physical diagnostics.
    assert formal_halfcap_v201_top["archive_diag_enabled"] is False
    assert formal_halfcap_v201_top["resource_variant_diag_enabled"] is False
    assert formal_halfcap_v201_top["physical_model_global_certificate"] is False
    print("V20.1 architecture：formal V20 semantics + post-formal diagnostic-only wiring → PASS")

    turn_cols = [
        dict(ordered_tids=("TA",), tids=("TA",), E_soc_required_Wh=1.0,
             E_plan_Wh=1.0, resource_intervals=dict(
                 launch_start_min=0.0, clear_end_min=5.0, active=(0.0, 5.0), deck=())),
        dict(ordered_tids=("TB",), tids=("TB",), E_soc_required_Wh=1.0,
             E_plan_Wh=1.0, resource_intervals=dict(
                 launch_start_min=1.0, clear_end_min=6.0, active=(1.0, 6.0), deck=()))]
    turn_shadow = BP._analyze_rejected_patterns_certificate_shadow(
        archive=turn_cols, selections=((0, 1),), K=1, batteries=2,
        usable_battery_energy_Wh=10.0, quick_min=1.0, swap_min=6.0,
        time_limit_s=2.0)
    assert turn_shadow["battery_binpack_feasible_patterns"] == 1
    assert turn_shadow["fastest_turnaround_infeasible_patterns"] == 1
    assert turn_shadow["first_proof_layer_counts"]["fastest_turnaround"] == 1

    # V9 resource-audit telemetry is observational only.  A strict deck overlap
    # must still return the historical INFEASIBLE_PROVEN status while exposing
    # the diagnostic reason count.
    conflict_cols = [
        _bpc_test_column(("PS-A",), 0.0, 5.0, 10.0),
        _bpc_test_column(("PS-B",), 0.0, 5.0, 11.0),
    ]
    for _c in conflict_cols:
        _c["resource_intervals"] = RA._resource_intervals(
            _c, 2.5, refresh_p.landing_clear_min, 6.0,
            deck_mode="interval", deck_delta_min=2.5)
    conflict_audit = BP._audit_integer_selection(
        conflict_cols, (0, 1), 2, 2, refresh_p, 1.0, 6.0, 1, 1, None)
    assert conflict_audit.status is RA.ResourceAuditStatus.INFEASIBLE_PROVEN
    assert int((conflict_audit.failure_reasons or {}).get(
        "deck_overlap", 0)) >= 1

    # A zero wall budget can only suppress the heuristic; it must return the
    # original incumbent and cannot claim an improvement.
    rr0 = BP._primal_refresh_incumbent(
        refresh_cols, (0,), base_audit, "coverage", None,
        2, 2, refresh_p, 1.0, 6.0, 1, 1, None,
        wall_budget_s=0.0, max_audits=16)
    assert not rr0["improved"]
    assert tuple(rr0["selection"]) == (0,)
    assert rr0["coverage"] == 1

    # Long E1 runs used to fail only when a result row was finally written:
    # dict(..., pricing_mode=..., **prov) duplicates a provenance keyword and
    # Python raises TypeError. Statically audit every dict(..., **prov) call.
    step13_path = Path(__file__).resolve().with_name("step13_experiment_model.py")
    step13_src = step13_path.read_text(encoding="utf-8")
    step13_ast = ast.parse(step13_src, filename=str(step13_path))
    prov_keys = set()
    for node0 in ast.walk(step13_ast):
        if isinstance(node0, ast.FunctionDef) and node0.name == "_provenance":
            for node1 in ast.walk(node0):
                if (isinstance(node1, ast.Return)
                        and isinstance(node1.value, ast.Call)
                        and isinstance(node1.value.func, ast.Name)
                        and node1.value.func.id == "dict"):
                    prov_keys.update(
                        kw.arg for kw in node1.value.keywords if kw.arg is not None)
            break
    prov_collisions = []
    for node0 in ast.walk(step13_ast):
        if not (isinstance(node0, ast.Call)
                and isinstance(node0.func, ast.Name)
                and node0.func.id == "dict"):
            continue
        if not any(
                kw.arg is None and isinstance(kw.value, ast.Name)
                and kw.value.id == "prov"
                for kw in node0.keywords):
            continue
        explicit_keys = {kw.arg for kw in node0.keywords if kw.arg is not None}
        overlap = sorted(explicit_keys & prov_keys)
        if overlap:
            prov_collisions.append((int(node0.lineno), overlap))
    assert not prov_collisions, (
        f"step13 dict(..., **prov) duplicate-key regression: {prov_collisions}")

    print("pricing shadow: mode separation / discovery firewall / one-sided bound PASS ✓")


SUITES["pricing_shadow"] = suite_pricing_shadow


# =============================================================================
# suite: 更新 —— 第三轮外部审计(致命 Big-M 假证书 + 4 项存活变异)的回归
# =============================================================================
def _r49_bigm_instance():
    """审计反例实例: 可行列 E0 ~1e7 Wh ≫ 旧硬编码 _BIGM=1e6。
    (B_k / power_scale 皆为自由参数, E0 ≤ B_use=(1−reserve)·B_k ⇒ 该参数域可达。)"""
    p = M.apply_uav_profile(M.Params(), "L")
    p.power_scale = getattr(p, "power_scale", 1.0) * 12000.0
    p.B_k = 6e7
    p.safe_reserve = 0.05
    p.w_land_max = 500.0; p.W_max = 500.0
    p.Hs_op = 100.0; p.s_heave_max = 100.0
    p.s_roll_max = 100.0; p.s_pitch_max = 100.0
    p.airspeed_cc = "off"
    p.t_dock_base_s = 0.0; p.dock_gamma = 0.0
    p.tau_insp = 300.0
    horizons = [15, 30, 60]
    turbines = []
    for tid, (x, y) in zip(("T0", "T1", "T2"),
                           ((20000.0, 0.0), (-18000.0, 4000.0), (12000.0, 13000.0))):
        t = M.Turbine(tid, np.zeros(2), 68.5, 115.0)
        t.local = np.array([x, y])
        turbines.append(t)
    wx = dict(wind10=0.0, wind_dir_from=270.0, Hs=0.3, Tp=6.0,
              wave_dir=0.0, ship_heading=0.0)
    opts = []
    for k in range(2):
        sp = RM.ShipPrediction.from_cv(np.zeros(2), np.array([0.1, 0.0]),
                                       horizons, c_state="DP")
        sp.tau_min = 15.0 * k
        sp.wx_tau = wx
        opts.append(RM.LaunchOption(15.0 * k, sp, wx))
    cells = {(h, "DP"): M.XiCell(h, "DP", 1000, np.zeros(2),
                                 np.diag([400.0, 400.0]), 0.0, 0.0, 0.0)
             for h in horizons}
    return p, turbines, opts, M.XiAmbiguity(cells, horizons), 60.0


def suite_l2_energy():
    """外部审计(第三轮)回归:
    ① 致命: L2 Phase-I 罚系数 —— 存在 E0>1e6 的可行列时, 旧 _BIGM=1e6 使人工比真实列
       便宜, 污染节点 LP 下界 ⇒ 误剪含真最优的节点却仍报 optimal(gap=0)、L2_certified=True。
       断言 B&P 的 L2 能耗【严格等于】全枚举锚点(相对容差, 不用宽固定容差);
    ② 变异 M04: RF 分支默认必须关闭;
    ③ 变异 M05: required 行 Phase-I 人工列必须存在(删之则空池节点无法自举);
    ④ 变异 M12: L2 定价须扫【全部】窗内 h(首个可行 h 非最优时仍要拿到最优);
    ⑤ 变异 M13: L2 CG 截断 ⇒ 必须撤销 L2 证书。"""
    # ---------- ① 致命: Big-M 假证书 ----------
    p, turbines, opts, xi, T_min = _r49_bigm_instance()
    K, B, ms = 1, 2, 2
    ext, st = RA.enumerate_discrete_routes(turbines, opts, p, xi, T_min, 2.5, ms, None)
    assert st["anchor_complete"], f"锚点不完整: {st}"
    r_ext = RA.solve_resource_master(turbines, opts, p, xi, K, T_min, t_swap_min=4.0,
                              max_stops=ms, weather_unc=None, batteries=B,
                              cols_override=ext, solver="auto", deck_mode="interval")
    rb = BP.solve_soft_coverage_research(turbines, opts, p, xi, K, T_min, deck_delta_min=2.5,
                              t_swap_min=4.0, max_stops=ms, weather_unc=None,
                              batteries=B, seed_cols=[], time_limit_s=300,
                              deck_mode="interval")
    maxE0 = max(float(c["E0"]) for c in ext)
    assert maxE0 > 1e6, f"反例前提失效: 需存在 E0>1e6 的可行列(实测 {maxE0:.3e})"
    assert rb["covered"] == r_ext["covered"], \
        f"L1 覆盖与锚点不符: {rb['covered']} vs {r_ext['covered']}"
    _tol = 1e-6 * max(1.0, abs(r_ext["energy_Wh"]))
    assert abs(rb["energy_Wh"] - r_ext["energy_Wh"]) <= max(_tol, 0.05), \
        (f"L2 能耗与锚点不符(Big-M 假证书): BP={rb['energy_Wh']:.1f} "
         f"vs 锚点={r_ext['energy_Wh']:.1f}, 差 {rb['energy_Wh']-r_ext['energy_Wh']:.2f} Wh")
    cert = rb["certificate"]
    assert cert.get("phase1_method") == "strict-two-phase", cert
    assert cert.get("bigm_used_for_correctness") is False, cert
    assert cert["conditions"]["L1"].get("strict_two_phase_complete") is True, cert
    assert cert["conditions"]["L2"].get("l2_strict_two_phase_complete") is True, cert
    print(f"高能耗两阶段回归: maxE0={maxE0:.3e}, Big-M 不参与正确性 | "
          f"BP E={rb['energy_Wh']:.1f} == 锚点 {r_ext['energy_Wh']:.1f} → PASS")

    # ---------- ② M04: RF 默认关闭 ----------
    import inspect as _insp
    _sig = _insp.signature(BP.solve_soft_coverage_research)
    assert _sig.parameters["enable_rf_branching"].default is False, \
        "RF 分支默认必须为 False(覆盖型主问题下 RF 不穷尽整数解)"
    assert rb["rf_branching"] == "off" and cert["rf_ok"], "默认运行不应发生 RF 对分支"
    print("M04(RF 默认关闭): 默认值=False 且 rf_ok=True → PASS")

    # ---------- ③ M05: Phase-I 人工列存在(空池 + required 仍可自举) ----------
    r5 = BP.solve_soft_coverage_research(turbines, opts, p, xi, K, T_min, deck_delta_min=2.5,
                              t_swap_min=4.0, max_stops=ms, weather_unc=None,
                              batteries=B, seed_cols=[], time_limit_s=300,
                              deck_mode="interval", _test_force_branch=True)
    assert r5["nodes"] >= 3 and r5["n_branch_turbine"] >= 1, \
        f"强制分支未生效: nodes={r5['nodes']}"
    assert r5["covered"] == rb["covered"], \
        f"分支(含 required 行 Phase-I)后最优改变: {r5['covered']} vs {rb['covered']}"
    # 白盒契约: required 行必须真的带人工松弛列(系数 −1), 否则空活跃列集下 RMP 不可行,
    # 节点会被静默剪掉(更新 #4 修的正是这一点)。删除该赋值(变异 M05)在小夹具上
    # 不改变最优值, 端到端测不出 ⇒ 直接断言 _rows 里建了人工列系数。
    import inspect as _insp0
    _rsrc = _insp0.getsource(BP.solve_soft_coverage_research)
    _rseg = _rsrc[_rsrc.find("def _rows("):]
    _rseg = _rseg[:_rseg.find("def _rmp_lp(")]
    assert "A[rrow[tid], n + m + k] = -1.0" in _rseg, \
        "required 行缺 Phase-I 人工松弛列(变异 M05): 空活跃列集下 RMP 将不可行并被静默剪枝"
    print(f"M05(Phase-I 人工列): 强制 required 分支 nodes={r5['nodes']} "
          f"覆盖不变={r5['covered']}; _rows 建有人工列系数 → PASS")

    # ---------- ④ M12: L2 须扫全部 h（完整计划能耗） ----------
    # 专用夹具: 母船朝风机方向航行 ⇒ 稍晚回收返程更短、能耗更低, 故【首个可行 h 不是
    # 能耗最优 h】。第一层按"成本升序首个可行 h"闭合是对的(覆盖层), 但 L2 能耗层必须
    # 扫全部窗内 h —— 只查首个可行 h(变异 M12)会返回次优能耗。
    p12 = M.apply_uav_profile(M.Params(), "L")
    p12.tau_insp = 300.0
    p12.P_wait = 1.0          # legacy 功率；伴飞功率按 escort_power_beta×P_wait, 使"船靠近缩短返程"主导 h 的能耗曲线
    p12.use_zeng = False
    h12 = [15, 30, 60]
    t12 = M.Turbine("T0", np.zeros(2), 68.5, 115.0)
    t12.local = np.array([12000.0, 0.0])
    wx12 = dict(wind10=0.0, wind_dir_from=270.0, Hs=0.3, Tp=6.0,
                wave_dir=0.0, ship_heading=0.0)
    sp12 = RM.ShipPrediction.from_cv(np.zeros(2), np.array([3.0, 0.0]),
                                     h12, c_state="DP")
    sp12.tau_min = 0.0
    sp12.wx_tau = wx12
    opt12 = RM.LaunchOption(0.0, sp12, wx12)
    xi12 = M.XiAmbiguity({(h, "DP"): M.XiCell(h, "DP", 1000, np.zeros(2),
                                              np.diag([400.0, 400.0]), 0.0, 0.0, 0.0)
                          for h in h12}, h12)
    _r12 = RM.Route(rid=-1, turbines=[t12], ship=sp12)
    _orig_k = RM.kappa
    RM.kappa = RM.KAPPA_MODES["vp_unimodal"]
    try:
        _feas = [(h, RM.route_feasible_at_h(_r12, int(h), p12, wx12, xi12,
                                            weather_unc=None))
                 for h in RM.decision_horizons_of(xi12)]
    finally:
        RM.kappa = _orig_k
    _ok = [(h, d.get("E_plan_Wh", d["E0"])) for h, d in _feas if d["feasible"]]
    assert len(_ok) >= 2, "M12 夹具无足够可行 h"
    _first_h, _first_E = _ok[0]
    _best_h, _best_E = min(_ok, key=lambda z: z[1])
    assert _best_h != _first_h and _best_E < _first_E - 1e-6, \
        f"M12 夹具失效: 首个可行 h 恰为最优({_ok[:3]})"
    rb12 = BP.solve_soft_coverage_research([t12], [opt12], p12, xi12, K=1, T_min=90.0,
                                deck_delta_min=2.5, t_swap_min=4.0, max_stops=1,
                                weather_unc=None, batteries=1, seed_cols=[],
                                time_limit_s=300, deck_mode="interval")
    assert rb12["covered"] == 1 and rb12["certificate"]["L2_certified"], \
        f"M12 夹具未取得 L2 证书: {rb12['covered']} {rb12['certificate']}"
    assert abs(rb12["energy_Wh"] - _best_E) <= max(1e-6 * max(1.0, _best_E), 0.05), \
        (f"L2 未取能耗最优 h(疑似只查首个可行 h): BP E={rb12['energy_Wh']:.1f} "
         f"vs 最优 {_best_E:.1f}(h={_best_h}); 首个可行 h={_first_h}(E={_first_E:.1f})")
    # 端到端还不够: 终池 h 扩列(廉价播种)会先把全部 h 灌进池, 使 L2 定价一轮即收敛 ⇒
    # "只查首个可行 h"的回退在端到端层面被掩盖。故【直接】对 price_routes 的能耗闭合
    # 断言 —— energy_weight=1 时必须返回 min-E0 的 h, 这是 L2 定价保真性的真正入口。
    _hproof = {}
    _pr = BP.price_routes([t12], sp12, p12, wx12, xi12, {0: 1e9},
                          route_cost=0.0, max_stops=1, k_near=1, max_routes=20,
                          rc_tol=-1e-6, weather_unc=None, strict_dominance=True,
                          close_cost_of_h=(lambda h, tids: 0.0),
                          dominance_mode="set", energy_weight=1.0,
                          stats_out=_hproof)
    assert _pr, "M12: 能耗定价未返回任何列"
    _pr_h, _pr_E = float(_pr[0]["h"]), float(_pr[0]["E0"])
    assert abs(_pr_E - _best_E) <= max(1e-6 * max(1.0, _best_E), 0.05), \
        (f"L2 定价未扫全部 h(变异 M12): 返回 h={_pr_h}(E={_pr_E:.1f}), "
         f"应为 h={_best_h}(E={_best_E:.1f})")
    assert _hproof.get("all_h_proof_complete") is True, _hproof
    assert _hproof.get("all_h_evaluations_observed") == _hproof.get("all_h_evaluations_expected"), _hproof
    assert rb12["certificate"].get("l2_all_h_checked") is True, rb12["certificate"]
    print(f"M12(L2 全 h 扫描): 首个可行 h={_first_h}(E={_first_E:.1f}) "
          f"≠ 最优 h={_best_h}(E={_best_E:.1f}); 端到端 E={rb12['energy_Wh']:.1f}, "
          f"定价直测 h={_pr_h}(E={_pr_E:.1f}) → PASS")

    # ---------- ⑤ M13: L2 CG 截断 ⇒ 撤证 ----------
    r13 = BP.solve_soft_coverage_research(turbines, opts, p, xi, K, T_min, deck_delta_min=2.5,
                               t_swap_min=4.0, max_stops=ms, weather_unc=None,
                               batteries=B, seed_cols=[], time_limit_s=300,
                               deck_mode="interval", pricing_label_budget=0)
    assert not r13["certificate"]["L2_certified"], \
        f"定价截断仍发 L2 证书: {r13['L2_status']} cert={r13['certificate']}"
    assert not r13["certificate"]["L1_certified"] and r13["UB"] is None, \
        f"定价截断未诚实降级: status={r13['status']} UB={r13['UB']}"
    assert r13["certificate"]["certificate_reason"], "撤证却未给出 certificate_reason"
    # 两阶段版直接检查结构化证书条件：任何截断都必须使树闭合、阶段完整性、全部 h
    # 中至少一个条件为 False，且 certificate_reason 非空。
    _l2c13 = r13["certificate"]["conditions"]["L2"]
    assert not all(_l2c13.values()), _l2c13
    assert (not _l2c13.get("l2_tree_closed", False)
            or not _l2c13.get("l2_strict_two_phase_complete", False)
            or not _l2c13.get("l2_all_h_checked", False)), _l2c13
    print(f"M13(截断撤证): status={r13['status']} L2cert=False; "
          f"结构化两阶段/全-h 条件 fail-closed → PASS")

    # ---------- ⑥ M16: L2 整数解须检查【全部】人工变量(含 required 行 s_i) ----------
    # 端到端补充: 加一台任何列都不可达的风机, 覆盖/能耗必须完全不变, 且不得被选入。
    _t_far = M.Turbine("TFAR", np.zeros(2), 68.5, 115.0)
    _t_far.local = np.array([3.0e5, 0.0])          # 300 km: 任何列都不可行
    r16 = BP.solve_soft_coverage_research(list(turbines) + [_t_far], opts, p, xi, K, T_min,
                               deck_delta_min=2.5, t_swap_min=4.0, max_stops=ms,
                               weather_unc=None, batteries=B, seed_cols=[],
                               time_limit_s=300, deck_mode="interval")
    assert r16["covered"] == rb["covered"], \
        f"不可达风机改变了覆盖数: {r16['covered']} vs {rb['covered']}"
    assert not any("TFAR" in c["tids"] for c in r16["chosen"]), \
        "选中列包含不可达风机(人工解被当作原问题解)"
    # 白盒契约: required 行的人工 s_i 只在【分支节点】才存在, 现有夹具规模下分支启发式
    # 不会选中不可达风机, 该路径无法由公开 API 稳定触发。故直接断言 _milp_int_E 检查的是
    # 【全部】人工变量切片 —— 只查 s_cnt(r.x[n+m])的回退(变异 M16)会漏掉 required 行 s_i,
    # 使违反 required 约束的增广解被当作原问题 incumbent 更新 best_E。
    _isrc = _insp.getsource(BP.solve_soft_coverage_research)
    _seg = _isrc[_isrc.find("def _milp_int_E"):]
    _seg = _seg[:_seg.find("def _price_node_E")]
    assert "n + m:n + m + n_s" in _seg.replace(" ", " "), \
        ("_milp_int_E 未检查全部人工变量(变异 M16): 应对 r.x[n+m : n+m+n_s] 求和, "
         "只查 s_cnt 会漏 required 行 s_i")
    # 更新(M-02): 阈值常量化为模块级 ART_TOL(仍为 1e-6), 契约同义
    assert "art_values" in _seg and "art_max > ART_TOL" in _seg, \
        "_milp_int_E 缺少全部人工变量残量拒绝逻辑"
    assert "hi[n + m:] = 0.0" in _seg, \
        "L2 Phase-II 未把全部人工变量固定为零"
    assert abs(BP.ART_TOL - 1e-6) < 1e-18, f"ART_TOL 被改动: {BP.ART_TOL}"
    print(f"M16(全人工检查): 不可达风机不改变 covered={r16['covered']}/E={r16['energy_Wh']:.1f}; "
          f"_milp_int_E 检查全部人工切片 [n+m : n+m+n_s] → PASS")

    # ---------- ⑥ 证书条件清单结构 ----------
    for _k in ("tree_complete", "strict_two_phase_complete", "reach_filter_proven_safe",
               "dominance_proven_safe", "no_rf_branching"):
        assert _k in cert["conditions"]["L1"], f"证书缺条件项 {_k}"
    assert cert["L1_certified"] == all(cert["conditions"]["L1"].values()), \
        "L1_certified 与条件清单不一致"
    print("证书条件清单: L1/L2 conditions 完整且与总布尔一致 → PASS")

    # ---------- ⑦ 更新 H-01(外部审计 问题#1 的模型级修复): 甲板端点未对齐 Δ ----------
    # t_launch=3.0, Δ=2.5 合法输入下 prep 端点不落格点 ⇒ 区间→格点映射漏判物理重叠(审计
    # 反例: 旧代码返回格点最优+双证书, 连续物理最优不同)。更新 的修复是【撤证】(保守);
    # 更新(H-01) 升级为【模型级修复】: 甲板逐对连续冲突行 + 占机事件行 + 返回前独立
    # 连续复检 ⇒ 未对齐 t_launch 也给出【连续物理最优】并正常发证(验收: 不应仅靠撤证
    # 通过)。本测试改为与独立连续物理词典序 oracle 逐位对拍 + 证书为真; deck_grid_exact
    # 仅余信息字段(honest=False), 证书条件换为 deck_conflict_semantics_exact。
    _tb7 = []
    for _tid, (_x, _y) in zip(("T0", "T1", "T2"),
                              ((2500.0, 0.0), (-2400.0, 300.0), (1800.0, 1500.0))):
        _t = M.Turbine(_tid, np.zeros(2), 68.5, 115.0)
        _t.local = np.array([_x, _y])
        _tb7.append(_t)
    _wx7 = dict(wind10=3.0, wind_dir_from=270.0, Hs=0.3, Tp=6.0,
                wave_dir=0.0, ship_heading=0.0)
    _op7 = []
    for _tau in (2.5, 5.0, 7.5):
        _sp = RM.ShipPrediction.from_cv(np.zeros(2), np.array([0.1, 0.0]),
                                        [15, 30], c_state="DP")
        _sp.tau_min = _tau
        _sp.wx_tau = _wx7
        _op7.append(RM.LaunchOption(_tau, _sp, _wx7))
    _xi7 = M.XiAmbiguity({(h, "DP"): M.XiCell(h, "DP", 1000, np.zeros(2),
                                             np.diag([100.0, 100.0]), 0.0, 0.0, 0.0)
                          for h in (15, 30)}, [15, 30])
    _ex7, _st7 = RA.enumerate_discrete_routes(_tb7, _op7, p, _xi7, 60.0, 2.5, 2, None)
    assert _st7["anchor_complete"], f"⑦ 锚点不完整: {_st7}"

    def _cont_oracle(cols7, K7, B7, tl7, ts7):
        """独立连续物理词典序暴力(与 BP 无共享代码路径): max 覆盖, tie-break min ΣE0;
        约束 = 甲板区间(prep/rec)逐对连续不相交 + 占机并发 ≤ K(左端点扫描) + 架次 ≤ B。"""
        import itertools as _it

        def _ivs(c):
            out = []
            a = max(c["tau"] - tl7, 0.0)
            if a < c["tau"] - 1e-9:
                out.append((a, c["tau"]))
            if ts7 > 1e-9:
                out.append((c["tau"] + c["h"], c["tau"] + c["h"] + ts7))
            return out

        best = (-1, float("inf"), None)
        for r_ in range(0, B7 + 1):
            for comb in _it.combinations(range(len(cols7)), r_):
                iv = [_ivs(cols7[j]) for j in comb]
                if any(max(a1, a2) < min(b1, b2) - 1e-9
                       for a_ in range(len(comb)) for b_ in range(a_ + 1, len(comb))
                       for a1, b1 in iv[a_] for a2, b2 in iv[b_]):
                    continue
                oc = [(max(cols7[j]["tau"] - tl7, 0.0),
                       cols7[j]["tau"] + cols7[j]["h"] + ts7) for j in comb]
                if any(sum(1 for (x0, x1) in oc if x0 - 1e-9 <= a0 < x1 - 1e-9) > K7
                       for (a0, _b0) in oc):
                    continue
                cov = len({tid for j in comb for tid in cols7[j]["tids"]})
                E = sum(float(cols7[j]["E0"]) for j in comb)
                if cov > best[0] or (cov == best[0] and E < best[1] - 1e-9):
                    best = (cov, E, comb)
        return best[0], best[1]

    for _tl, _ts in ((3.0, 4.0), (2.5, 4.0), (2.5, 3.7)):
        _cov_o, _E_o = _cont_oracle(_ex7, 2, 3, _tl, _ts)
        r7 = BP.solve_soft_coverage_research(_tb7, _op7, p, _xi7, 2, 60.0, deck_delta_min=2.5,
                                  t_swap_min=_ts, max_stops=2, weather_unc=None,
                                  batteries=3, seed_cols=[], time_limit_s=200,
                                  deck_mode="interval", t_launch_min=_tl)
        c7 = r7["certificate"]
        assert r7["covered"] == _cov_o and \
            abs(r7["energy_Wh"] - _E_o) <= max(1e-6 * max(1.0, _E_o), 0.05), \
            (f"⑦ t_launch={_tl},t_swap={_ts}: BP=({r7['covered']},{r7['energy_Wh']:.1f}) "
             f"≠ 连续 oracle=({_cov_o},{_E_o:.1f})")
        assert c7["L1_certified"] and c7["L2_certified"], \
            f"⑦ 修复后应正常发证(不靠撤证通过): t_launch={_tl} reason={c7['certificate_reason']}"
        assert c7["deck_conflict_semantics_exact"] and \
            c7["conditions"]["L1"]["physical_plan_verified"], \
            f"⑦ 连续语义/物理复检条件失败: {c7['deck_pair_stats']}"
        # 选中列独立逐对复核(第三方口径, 不读 BP 内部字段)
        for _i in range(len(r7["chosen"])):
            for _j in range(_i + 1, len(r7["chosen"])):
                ci, cj = r7["chosen"][_i], r7["chosen"][_j]
                for a1, b1 in ((max(ci["tau"] - _tl, 0.0), ci["tau"]),
                               (ci["tau"] + ci["h"], ci["tau"] + ci["h"] + _ts)):
                    for a2, b2 in ((max(cj["tau"] - _tl, 0.0), cj["tau"]),
                                   (cj["tau"] + cj["h"], cj["tau"] + cj["h"] + _ts)):
                        assert not (max(a1, a2) < min(b1, b2) - 1e-9
                                    and b1 > a1 + 1e-9 and b2 > a2 + 1e-9), \
                            f"⑦ 选中列甲板物理重叠: t_launch={_tl}"
        if _tl == 3.0:
            assert c7["deck_grid_exact"] is False, "⑦ 未如实报告端点未对齐(信息字段)"
            assert r7["deck_pair_stats"]["row_added"] > 0, \
                "⑦ 未对齐档应有格点漏判的逐对补充行(row_added>0)"
        else:
            assert c7["deck_grid_exact"] is True and \
                r7["deck_pair_stats"]["row_added"] == 0, \
                "⑦ 对齐档不应产生逐对补充行(全部由格点行蕴含)"
    print("问题#1→H-01(甲板连续语义): t_launch=3.0(未对齐) ⇒ 连续物理最优+正常发证"
          "(逐对行生效); 2.5/t_swap=3.7 不误伤 → PASS")
    print("suite l2_energy: 全部 PASS ✓")


SUITES["l2_energy"] = suite_l2_energy


# =============================================================================
# 更新 修复回归: C-01 / H-01 / H-02 / H-03 / M-01 / M-02 / M-03 全链验收。
# =============================================================================
def suite_branch_price():
    p = M.apply_uav_profile(M.Params(), "L")
    wxc = dict(wind10=3.0, wind_dir_from=270.0, Hs=0.3, Tp=6.0,
               wave_dir=0.0, ship_heading=0.0)
    wxs = dict(wind10=25.0, wind_dir_from=270.0, Hs=4.5, Tp=9.0,
               wave_dir=0.0, ship_heading=0.0)

    def _sp(tau, vel, wx, hs=(15, 30)):
        s = RM.ShipPrediction.from_cv(np.zeros(2), np.asarray(vel, float),
                                      list(hs), c_state="DP")
        s.tau_min = float(tau)
        s.wx_tau = wx
        return RM.LaunchOption(float(tau), s, wx)

    def _tb(tid, x, y, wx=None):
        t = M.Turbine(tid, np.zeros(2), 68.5, 115.0)
        t.local = np.array([float(x), float(y)])
        if wx is not None:
            t.wx_local = dict(wx)
        return t

    def _xi(hs=(15, 30), var=100.0):
        return M.XiAmbiguity({(h, "DP"): M.XiCell(h, "DP", 1000, np.zeros(2),
                                                  np.diag([var, var]), 0.0, 0.0, 0.0)
                              for h in hs}, list(hs))

    # ① C-01 端到端: 风暴局地场风机必须被全候选最坏情况判杀(cov=1), 且双证书;
    #    对照: 摘除风暴场(同几何)应 cov=2 —— 判杀确因天气切换而非几何。
    tbs = [_tb("T0", 2500.0, 0.0), _tb("T1", -2400.0, 300.0, wx=wxs)]
    r1 = BP.solve_soft_coverage_research(tbs, [_sp(2.5, (0.1, 0.0), wxc)], p, _xi(), 2, 60.0,
                              deck_delta_min=2.5, t_swap_min=4.0, max_stops=2,
                              weather_unc=None, batteries=2, seed_cols=[],
                              time_limit_s=200, deck_mode="interval")
    c1 = r1["certificate"]
    _cT1 = any("T1" in c["tids"] for c in r1["chosen"])
    assert r1["covered"] == 1 and not _cT1 and c1["L1_certified"] and \
        c1["L2_certified"] and c1["gate_weather_switch_proven_safe"] and \
        c1["gate_proof_missing"] == 0, \
        f"C-01 风暴判杀失败: cov={r1['covered']} covT1={_cT1} {c1['certificate_reason']}"
    tb0 = [_tb("T0", 2500.0, 0.0), _tb("T1", -2400.0, 300.0)]
    r1b = BP.solve_soft_coverage_research(tb0, [_sp(2.5, (0.1, 0.0), wxc)], p, _xi(), 2, 60.0,
                               deck_delta_min=2.5, t_swap_min=4.0, max_stops=2,
                               weather_unc=None, batteries=2, seed_cols=[],
                               time_limit_s=200, deck_mode="interval")
    assert r1b["covered"] == 2 and r1b["certificate"]["L1_certified"], \
        "C-01 对照失败: 摘除风暴场应恢复 cov=2"
    print(f"① C-01 风暴场判杀 cov=1(对照摘除后 2), 双证书, gate 证据 "
          f"{c1['gate_proof_missing']} 缺失 ✓")

    # ② 更新 P0-2: 恶意外部种子缺 gate 证据且 h 不在决策域。
    #    新口径不把外部种子直接灌入列池，而是规范化/重验证；该列应被拒绝，
    #    后续定价生成的真实列自带 gate proof，证书不应被恶意缺字段污染。
    ext0, _ = RA.enumerate_discrete_routes(tb0, [_sp(2.5, (0.1, 0.0), wxc)], p, _xi(),
                                   60.0, 2.5, 1, None)
    rogue = dict(ext0[0])
    rogue["h"] = 17.0
    rogue["E0"] = 1.0e5
    rogue["route"] = None
    rogue.pop("gate_weather_proof", None)
    rogue.pop("gate_proof", None)
    r2 = BP.solve_soft_coverage_research(tbs, [_sp(2.5, (0.1, 0.0), wxc)], p, _xi(), 2, 60.0,
                              deck_delta_min=2.5, t_swap_min=4.0, max_stops=2,
                              weather_unc=None, batteries=2, seed_cols=[rogue],
                              time_limit_s=200, deck_mode="interval")
    c2 = r2["certificate"]
    sv2 = r2["seed_validation"]
    assert (sv2["accepted_count"] == 0 and sv2["rejected_count"] == 1
            and sv2["rejection_reasons"].get("h-outside-decision-domain") == 1
            and c2["seed_columns_revalidated"] and c2["gate_proof_missing"] == 0
            and c2["gate_weather_switch_proven_safe"]), (
                f"C-01/seed 重验证失败: seed={sv2} cert={c2['certificate_reason']}")
    print("② C-01 + P0-2: 缺证/越域种子被拒绝，真实定价列 gate proof 完整 ✓")

    # ③ H-01 绑定实例(τ∈{2.5,5}, t_launch=3.0): prep [0,2.5)∩[2,5)=[2,2.5)
    #    连续重叠但无公共 Δ 格点 —— 逐对行必须绑定 ⇒ cov=1 且双证书;
    #    E-01: 该实例自然触发两类分支(nb_t≥1 ∧ nb_c≥1) 与深度>1 的树。
    tb3 = [_tb("T0", 2500.0, 0.0), _tb("T1", -2400.0, 300.0),
           _tb("T2", 1800.0, 1500.0)]
    op3 = [_sp(2.5, (0.1, 0.0), wxc), _sp(5.0, (0.1, 0.0), wxc)]
    r3 = BP.solve_soft_coverage_research(tb3, op3, p, _xi(), 2, 60.0, deck_delta_min=2.5,
                              t_swap_min=4.0, max_stops=1, weather_unc=None,
                              batteries=2, seed_cols=[], time_limit_s=200,
                              deck_mode="interval", t_launch_min=3.0)
    c3 = r3["certificate"]
    ps3 = r3["deck_pair_stats"]
    assert r3["covered"] == 1 and c3["L1_certified"] and c3["L2_certified"] and \
        c3["deck_conflict_semantics_exact"] and ps3["row_added"] > 0, \
        (f"H-01 绑定失败: cov={r3['covered']} rows={ps3} "
         f"{c3['certificate_reason']}")
    assert r3["n_branch_turbine"] >= 1 and r3["n_branch_col"] >= 1 and \
        r3["max_depth"] > 1 and r3["nodes"] > 3, \
        (f"E-01 分支覆盖不足: nb_t={r3['n_branch_turbine']} nb_c={r3['n_branch_col']} "
         f"depth={r3['max_depth']} nodes={r3['nodes']}")
    print(f"③ H-01 逐对行绑定 cov=1, 行数={ps3['row_added']}, "
          f"E-01 nb_t={r3['n_branch_turbine']} nb_c={r3['n_branch_col']} "
          f"depth={r3['max_depth']} nodes={r3['nodes']}, "
          f"E-02 phase1 激活={r3['phase1_stats']['l1_activated']} "
          f"补列={r3['phase1_stats']['l1_resolved']} "
          f"不可行判定={r3['phase1_stats']['l1_infeasible']} ✓")
    _p1 = r3["phase1_stats"]
    assert _p1["l1_activated"] >= 1 and \
        _p1["l1_infeasible"] >= 1 and \
        _p1["l1_activated"] == _p1["l1_resolved"] + _p1["l1_infeasible"], \
        f"E-02 phase1 计数不一致: {_p1}"


    # ④ M-01: time_limit_s=0 ⇒ L1_status='time-limit-no-certificate', UB=None,
    #    且旗标独立(hit_time ∧ ¬hit_node); node_limit=0 ⇒ 对偶情形。
    r4 = BP.solve_soft_coverage_research(tb0, [_sp(2.5, (0.1, 0.0), wxc)], p, _xi(), 1, 60.0,
                              deck_delta_min=2.5, t_swap_min=4.0, max_stops=1,
                              weather_unc=None, batteries=1, seed_cols=[],
                              time_limit_s=0, deck_mode="interval")
    cl4 = r4["certificate"]["conditions"]["L1"]
    assert r4["L1_status"] == "time-limit-no-certificate" and r4["UB"] is None and \
        r4["hit_time_limit"] and not r4["hit_node_limit"] and \
        cl4["no_timeout"] is False and cl4["no_node_limit"] is True
    r4b = BP.solve_soft_coverage_research(tb0, [_sp(2.5, (0.1, 0.0), wxc)], p, _xi(), 1, 60.0,
                               deck_delta_min=2.5, t_swap_min=4.0, max_stops=1,
                               weather_unc=None, batteries=1, seed_cols=[],
                               time_limit_s=200, max_nodes=0, deck_mode="interval")
    cl4b = r4b["certificate"]["conditions"]["L1"]
    assert r4b["L1_status"] == "node-limit-no-certificate" and \
        r4b["hit_node_limit"] and not r4b["hit_time_limit"] and \
        cl4b["no_node_limit"] is False and cl4b["no_timeout"] is True
    print("④ M-01 时间/节点限旗标独立、状态串拆分、UB=None ✓")

    # ⑤ M-02: l2_mode='expand' 回退路径 ⇒ L2_scope 明示无定价证书, 人工审计
    #    fail-closed(l2_no_artificial_residue=False), L2 不发证。
    r5 = BP.solve_soft_coverage_research(tb3, op3, p, _xi(), 2, 60.0, deck_delta_min=2.5,
                              t_swap_min=4.0, max_stops=1, weather_unc=None,
                              batteries=2, seed_cols=[], time_limit_s=200,
                              deck_mode="interval", t_launch_min=3.0,
                              l2_mode="expand")
    c5 = r5["certificate"]
    assert r5["L2_scope"] == "expanded-pool(no-L2-pricing-certificate)" and \
        not c5["L2_certified"] and \
        c5["conditions"]["L2"]["l2_no_artificial_residue"] is False, \
        f"M-02 扩池 fail-closed 失败: {r5['L2_scope']} {c5['certificate_reason']}"
    print(f"⑤ M-02 扩池回退: scope 明示 + 人工审计 fail-closed + L2 不发证 ✓")

    # ⑥ H-02 运行时证明: 定价计数恒等式(calls==expected>0)与偏置档早剪自动
    #    禁用(mean_relax_free=False ⇒ nominal_prunes_active=False)下仍发证。
    wu = RM.WeatherUncertainty(wind_cov=np.zeros((2, 2)),
                               wind_bias=np.array([2.0, 0.0]), hs_std=0.0, hs_bias=0.0)
    r6 = BP.solve_soft_coverage_research(tb0, [_sp(2.5, (0.1, 0.0), wxc)], p, _xi(), 2, 60.0,
                              deck_delta_min=2.5, t_swap_min=4.0, max_stops=1,
                              weather_unc=wu, batteries=2, seed_cols=[],
                              time_limit_s=200, deck_mode="interval")
    c6 = r6["certificate"]
    assert not RM.mean_relax_free(_xi(), wu) and \
        c6["nominal_prunes_active"] is False and c6["L1_certified"] and \
        c6["pricing_calls"] == c6["pricing_calls_expected"] > 0 and \
        c6["conditions"]["L1"]["energy_pruning_proven_safe"] is True, \
        (f"H-02 失败: calls={c6['pricing_calls']}/{c6['pricing_calls_expected']} "
         f"nom={c6['nominal_prunes_active']} {c6['certificate_reason']}")
    print(f"⑥ H-02 定价恒等式 {c6['pricing_calls']}/{c6['pricing_calls_expected']}, "
          f"偏置档早剪禁用仍发证 ✓")

    print("suite branch_price: 全部 PASS ✓")


SUITES["branch_price"] = suite_branch_price


# =============================================================================
# 更新 反例 / 独立审计 / 变异 —— 按交付约束全部收编于本文件(不新增 .py 文件):
#   suite counterexamples : C-01 两点切换反例 + H-01 Δ-格点漏判反例(含旧口径独立重建);
#   suite random_oracle   : 随机小实例 BP vs【独立实现】连续物理词典序 oracle 对拍;
#   suite mutations       : MUT-01..111 行为级变异逐一"必须被杀"(临时目录打补丁 + 子进程驱动)。
# =============================================================================

# ---- 连续物理几何助手(独立于 step12 内部实现; oracle/反例/驱动共用) ----
def _c_deck_ivs(tau, h, tl, ts):
    out = []
    a = max(tau - tl, 0.0)
    if a < tau - 1e-9:
        out.append((a, tau))
    if ts > 1e-9:
        out.append((tau + h, tau + h + ts))
    return out


def _c_ovl(iv1, iv2):
    return any(max(a1, a2) < min(b1, b2) - 1e-9
               for a1, b1 in iv1 for a2, b2 in iv2)


def _cont_lex_oracle(cols, B, K, tl, ts):
    """连续物理词典序暴力(独立实现): 甲板逐对连续不相交 + 占机左端点扫描 ≤K +
    架次 ≤B; 先 max 覆盖后 min 能耗。返回 (cov, E)。"""
    import itertools as _it
    best = (-1, float("inf"))
    for r in range(0, B + 1):
        for comb in _it.combinations(range(len(cols)), r):
            iv = [_c_deck_ivs(cols[j]["tau"], cols[j]["h"], tl, ts) for j in comb]
            if any(_c_ovl(iv[a], iv[b])
                   for a in range(len(comb)) for b in range(a + 1, len(comb))):
                continue
            oc = [(max(cols[j]["tau"] - tl, 0.0),
                   cols[j]["tau"] + cols[j]["h"] + ts) for j in comb]
            if any(sum(1 for (x0, x1) in oc if x0 - 1e-9 <= a0 < x1 - 1e-9) > K
                   for (a0, _b0) in oc):
                continue
            cov = len({t for j in comb for t in cols[j]["tids"]})
            En = sum(float(cols[j]["E0"]) for j in comb)
            if cov > best[0] or (cov == best[0] and En < best[1] - 1e-9):
                best = (cov, En)
    return best


def _plan_cont_ok(chosen, B, K, tl, ts):
    """独立复核一个方案(BP 返回的 chosen)是否连续物理可执行。"""
    if len(chosen) > B:
        return False
    iv = [_c_deck_ivs(c["tau"], c["h"], tl, ts) for c in chosen]
    if any(_c_ovl(iv[a], iv[b])
           for a in range(len(chosen)) for b in range(a + 1, len(chosen))):
        return False
    oc = [(max(c["tau"] - tl, 0.0), c["tau"] + c["h"] + ts) for c in chosen]
    return not any(sum(1 for (x0, x1) in oc if x0 - 1e-9 <= a0 < x1 - 1e-9) > K
                   for (a0, _b0) in oc)


def _mk_launch(tau, vel, wx, horizons=(15, 30)):
    sp = RM.ShipPrediction.from_cv(np.zeros(2), np.asarray(vel, float),
                                   list(horizons), c_state="DP")
    sp.tau_min = float(tau)
    sp.wx_tau = wx
    return RM.LaunchOption(float(tau), sp, wx)


_WX_CALM = dict(wind10=3.0, wind_dir_from=270.0, Hs=0.3, Tp=6.0,
                wave_dir=0.0, ship_heading=0.0)
_WX_STORM = dict(wind10=25.0, wind_dir_from=270.0, Hs=4.5, Tp=9.0,
                 wave_dir=0.0, ship_heading=0.0)


def _xi_diag(h_list=(15, 30), var=100.0):
    return M.XiAmbiguity({(h, "DP"): M.XiCell(h, "DP", 1000, np.zeros(2),
                                              np.diag([var, var]), 0.0, 0.0, 0.0)
                          for h in h_list}, list(h_list))


def _tb_at(tid, x, y, storm=False):
    t = M.Turbine(tid, np.zeros(2), 68.5, 115.0)
    t.local = np.array([float(x), float(y)])
    if storm:
        t.wx_local = dict(_WX_STORM)
    return t


# ---------- 变异驱动共用夹具(全部经公开 API; 亦被 clean 校验直跑) ----------
def _fx_gate_storm(**kw):
    tb = [_tb_at("T0", 2500.0, 0.0), _tb_at("T1", -2400.0, 300.0, storm=True)]
    return BP.solve_soft_coverage_research(tb, [_mk_launch(2.5, (0.1, 0.0), _WX_CALM)],
                                M.apply_uav_profile(M.Params(), "L"), _xi_diag(),
                                2, 60.0, deck_delta_min=2.5, t_swap_min=4.0,
                                max_stops=2, weather_unc=None, batteries=2,
                                seed_cols=[], time_limit_s=200,
                                deck_mode="interval", **kw)


def _fx_binding(**kw):
    tb = [_tb_at("T0", 2500.0, 0.0), _tb_at("T1", -2400.0, 300.0),
          _tb_at("T2", 1800.0, 1500.0)]
    ops = [_mk_launch(2.5, (0.1, 0.0), _WX_CALM),
           _mk_launch(5.0, (0.1, 0.0), _WX_CALM)]
    return BP.solve_soft_coverage_research(tb, ops, M.apply_uav_profile(M.Params(), "L"),
                                _xi_diag(), 2, 60.0, deck_delta_min=2.5,
                                t_swap_min=4.0, max_stops=1, weather_unc=None,
                                batteries=2, seed_cols=[], time_limit_s=200,
                                deck_mode="interval", t_launch_min=3.0, **kw)


def _fx_occ_event(**kw):
    """占机事件行绑定实例: τ∈{0,16}, t_launch=3, t_swap=0, K=1 ⇒
    occ_A=[0,15) 与 occ_B=[13,16+h) 在 [13,15) 连续重叠且【无公共 Δ 格点】
    (12.5<13, 15∉[13,15)); prep_B=[13,16) 独占甲板(A 无甲板区间, 无逐对冲突)。
    连续最优 = 1 台; 仅格点占机行会放行双发(=2, 错) —— 事件行在此绑定。"""
    tb = [_tb_at("T0", 2500.0, 0.0), _tb_at("T1", -2400.0, 300.0)]
    ops = [_mk_launch(0.0, (0.1, 0.0), _WX_CALM),
           _mk_launch(16.0, (0.1, 0.0), _WX_CALM)]
    return BP.solve_soft_coverage_research(tb, ops, M.apply_uav_profile(M.Params(), "L"),
                                _xi_diag(), 1, 60.0, deck_delta_min=2.5,
                                t_swap_min=0.0, max_stops=1, weather_unc=None,
                                batteries=2, seed_cols=[], time_limit_s=200,
                                deck_mode="interval", t_launch_min=3.0, **kw)


def _fx_bias(**kw):
    """更新 ⑥ 同源夹具: 顺风偏置下名义 E0>B_use 仍 DRCC 可行 ——
    valid 预筛须自动降级、空种子定价必须自己找到该列(更新 审计 6.1)。"""
    p6 = M.Params(); p6.B_k = 390.0
    p6.power_scale = 439.0 / M.P_zeng(0.0, M.Params())
    p6.safe_reserve = 0.20; p6.w_land_max = 500.0; p6.W_max = 500.0; p6.v_max = 100.0; p6.v_air_max = 100.0
    p6.airspeed_cc = "off"; p6.t_dock_base_s = 0.0; p6.dock_gamma = 0.0
    p6.tau_insp = 300.0
    p6.Hs_op = 100.0; p6.s_heave_max = 100.0
    p6.s_roll_max = 100.0; p6.s_pitch_max = 100.0
    h6, D6 = 60, 48750.0
    wx6 = dict(wind10=0.0, wind_dir_from=270.0, Hs=0.0, Tp=8.0,
               wave_dir=0.0, ship_heading=0.0)
    opt6 = _mk_launch(0.0, (D6 / (h6 * 60.0), 0.0), wx6, horizons=(h6,))
    t6 = _tb_at("T1", D6, 0.0)
    xi6 = M.XiAmbiguity({(h6, "DP"): M.XiCell(h6, "DP", 1000, np.zeros(2),
                                              np.zeros((2, 2)), 0.0, 0.0, 0.0)}, [h6])
    wu6 = RM.WeatherUncertainty(wind_cov=np.zeros((2, 2)),
                                wind_bias=np.array([10.0, 0.0]),
                                hs_std=0.0, hs_bias=0.0)
    return BP.solve_soft_coverage_research([t6], [opt6], p6, xi6, 1, 60, deck_delta_min=2.5,
                                t_swap_min=0.0, max_stops=1, weather_unc=wu6,
                                batteries=1, seed_cols=[], time_limit_s=120,
                                cg_max_iter=50, **kw)


def _fx_tiny(**kw):
    tb = [_tb_at("T0", 2500.0, 0.0)]
    base = dict(deck_delta_min=2.5, t_swap_min=4.0, max_stops=1, weather_unc=None,
                batteries=1, seed_cols=[], time_limit_s=120, deck_mode="interval")
    base.update(kw)
    return BP.solve_soft_coverage_research(tb, [_mk_launch(2.5, (0.1, 0.0), _WX_CALM)],
                                M.apply_uav_profile(M.Params(), "L"), _xi_diag(),
                                1, 60.0, **base)


# ---------- 变异驱动(未变异代码上必须 PASS; 变异后必须失败 = 被杀) ----------
def _drv_gate_worst():
    # C-01 分界线几何: 预测回收点的最近场是【温和】场, 但候选集含风暴场
    # (T0/T1 最近邻分界上偏 T0 一侧 1mm)。只查预测最近点/∃开 的变体在此放行
    # T1 列 ⇒ cov=2; 正确的全候选最坏情况 ⇒ cov=1。
    p = M.apply_uav_profile(M.Params(), "L")
    T0 = _tb_at("T0", 2500.0, 0.0)
    T1 = _tb_at("T1", -2400.0, 300.0, storm=True)
    axis = (T0.local - T1.local) / np.linalg.norm(T0.local - T1.local)
    P_pred = (T0.local + T1.local) / 2.0 + 1e-3 * axis
    opt = _mk_launch(2.5, tuple(P_pred / (30 * 60.0)), _WX_CALM)
    # 直接契约: 分界两站路线在 h=30 必须被【全候选最坏情况】判不可行, 且证据
    # 显示恰好 2 个候选全查(单候选/∃开 变体在此当场暴露)。
    d = RM.route_feasible_at_h(RM.Route(-1, [T0, T1], opt.ship), 30, p,
                               _WX_CALM, _xi_diag(), weather_unc=None)
    gp = d["gate_weather_proof"]
    assert not d["feasible"] and d["gate"] == 0 and gp["candidate_count"] == 2 \
        and gp["all_candidates_checked"], \
        f"分界路线须由 2 候选最坏情况判杀: {d['feasible']} {gp}"
    r = BP.solve_soft_coverage_research([T0, T1], [opt], p, _xi_diag(), 2, 60.0,
                             deck_delta_min=2.5, t_swap_min=4.0, max_stops=2,
                             weather_unc=None, batteries=2, seed_cols=[],
                             time_limit_s=200, deck_mode="interval")
    c = r["certificate"]
    cov_T1 = any("T1" in col["tids"] for col in r["chosen"])
    assert r["covered"] == 1 and not cov_T1, \
        f"分界线几何须由全候选最坏情况判杀 T1: cov={r['covered']} covT1={cov_T1}"
    assert c["L1_certified"] and c["L2_certified"] and \
        c["gate_weather_switch_proven_safe"] and c["gate_proof_missing"] == 0


def _drv_bind_cert():
    r = _fx_binding()
    assert r["covered"] == 1, f"连续语义最优应为 1: {r['covered']}"
    c = r["certificate"]
    assert c["L1_certified"] and c["L2_certified"] and \
        c["deck_conflict_semantics_exact"], c["certificate_reason"]


def _drv_bind_oracle():
    # 连续物理真值 = 1(同 τ 共格点 + 跨 τ prep [0,2.5)∩[2,5)=[2,2.5) 连续重叠
    # ⇒ 任意两列冲突); 附独立于 step12 的方案连续复核兜底。
    r = _fx_binding()
    assert r["covered"] == 1 and _plan_cont_ok(r["chosen"], 2, 2, 3.0, 4.0), \
        f"BP 偏离连续物理真值/方案物理不可执行: cov={r['covered']}"


def _drv_occ_event():
    r = _fx_occ_event()
    assert r["covered"] == 1, f"占机连续语义最优应为 1: {r['covered']}"
    assert r["certificate"]["L1_certified"], r["certificate"]["certificate_reason"]
    assert _plan_cont_ok(r["chosen"], 2, 1, 3.0, 0.0)


def _drv_bias_reach():
    # 正式SOC语义下有利均值不能让 E_plan>B_use 的列变可行，但带符号均值仍必须
    # 令 reach=valid 降级为 off，避免把方向性均值误当作可证明的名义剪枝条件。
    r = _fx_bias()
    c = r["certificate"]
    assert r["covered"] == r["UB"] == 0 and c["L1_certified"] and c["L2_certified"], \
        (f"偏置SOC反例: 应证明0/0, got {r['covered']}/{r['UB']} "
         f"{c['certificate_reason']}")
    assert not c["nominal_prunes_active"] and c.get("reach_modes_seen") == ["off"] and \
        c["conditions"]["L1"]["energy_pruning_proven_safe"] is True and \
        c["conditions"]["L1"]["reach_filter_proven_safe"] is True


def _drv_soc_dedup():
    """Ordered routes with equal covered sets must remain distinct SOC columns."""
    cols = FAC.validate_route_columns([
        dict(tids=("A", "B"), ordered_tids=("A", "B"), tau=0.0, h=10.0,
             E_plan_Wh=100.0, E_soc_required_Wh=500.0, E0=100.0),
        dict(tids=("A", "B"), ordered_tids=("B", "A"), tau=0.0, h=10.0,
             E_plan_Wh=120.0, E_soc_required_Wh=200.0, E0=120.0),
    ])
    assert len(cols) == 2 and {tuple(c["ordered_tids"]) for c in cols} == \
        {("A", "B"), ("B", "A")} and \
        {float(c["E_soc_required_Wh"]) for c in cols} == {200.0, 500.0}


def _drv_pricing_id():
    r = _fx_tiny()
    c = r["certificate"]
    assert c["L1_certified"] and \
        c["pricing_calls"] == c["pricing_calls_expected"] > 0, \
        f"定价计数恒等式: {c['pricing_calls']}/{c['pricing_calls_expected']}"


def _drv_m01():
    r = _fx_tiny(time_limit_s=0)
    cl = r["certificate"]["conditions"]["L1"]
    assert r["L1_status"] == "time-limit-no-certificate" and r["UB"] is None and \
        r["hit_time_limit"] and not r["hit_node_limit"] and \
        cl["no_timeout"] is False and cl["no_node_limit"] is True, \
        f"M-01 时间限旗标: {r['status']} {cl}"


def _drv_m02():
    r = _fx_binding(l2_mode="expand")
    assert r["L2_scope"] == "expanded-pool(no-L2-pricing-certificate)"
    assert r["certificate"]["conditions"]["L2"]["l2_no_artificial_residue"] is False, \
        "扩池回退的 L2 人工审计必须如实 fail-closed(audit_complete=False)"


def _drv_m03():
    r = _fx_tiny(enable_rf_branching=True)
    c = r["certificate"]
    assert c["conditions"]["L1"]["no_rf_branching"] is False and \
        not c["L1_certified"] and "no_rf_branching" in c["certificate_reason"], \
        f"RF 启用即须撤证: {c['certificate_reason']}"


def _drv_h03():
    p = M.Params()
    alloc = RM.mission_risk_allocation(p, True)
    b_on = RM.mission_eps_budget(p, True)
    assert "acquisition" not in alloc, f"当前有限模型不得重新引入 acquisition 事件: {alloc}"
    assert abs(b_on - 0.045) < 1e-12, f"默认活跃事件预算应为 0.045: got {b_on}"
    assert b_on <= p.mission_failure_budget, (b_on, p.mission_failure_budget)
    assert RM.mission_budget_compliant(p, True)


def _drv_terminal_scope():
    p = M.Params()
    p.validate_contract(formal=True)
    alloc = RM.mission_risk_allocation(p, True)
    assert "acquisition" not in alloc, alloc
    assert not hasattr(p, "eps_acq") and not hasattr(p, "acquisition_radius_m")
    assert getattr(p, "recovery_target_model", None) == "discrete_horizon_ship_prediction"
    assert getattr(p, "terminal_sensor_error_mode", None) == "out_of_scope"

def _drv_p0_milp():
    import scipy.optimize as _spo
    p = M.apply_uav_profile(M.Params(), "L")
    p.tau_insp = 300.0; p.P_wait = 1.0; p.use_zeng = False
    H = [15, 30, 60]
    t = _tb_at("T0", 12000.0, 0.0)
    wx = dict(wind10=0.0, wind_dir_from=270.0, Hs=0.3, Tp=6.0,
              wave_dir=0.0, ship_heading=0.0)
    sp = RM.ShipPrediction.from_cv(np.zeros(2), np.array([3.0, 0.0]), H, c_state="DP")
    sp.tau_min = 0.0; sp.wx_tau = wx
    opt = RM.LaunchOption(0.0, sp, wx)
    xi = M.XiAmbiguity({(h, "DP"): M.XiCell(h, "DP", 1000, np.zeros(2),
                                                    np.diag([400.0, 400.0]), 0.0, 0.0, 0.0)
                         for h in H}, H)
    route = RM.Route(-1, [t], sp)
    _orig_k = RM.kappa; RM.kappa = RM.KAPPA_MODES["vp_unimodal"]
    try:
        best_E = min(float(d["E_plan_Wh"]) for h in H
                     for d in [RM.route_feasible_at_h(route, h, p, wx, xi)] if d["feasible"])
    finally:
        RM.kappa = _orig_k
    class _Fail:
        success = False; status = 1; message = "forced"; x = None
    orig = _spo.milp; _spo.milp = lambda *a, **k: _Fail()
    try:
        r = BP.solve_soft_coverage_research([t], [opt], p, xi, 1, 90.0, max_stops=1,
                                 batteries=1, seed_cols=[], time_limit_s=100,
                                 deck_mode="interval")
    finally:
        _spo.milp = orig
    assert r["covered"] == 1 and abs(r["energy_Wh"] - best_E) <= 0.05 and \
        r["certificate"]["L1_certified"] and r["certificate"]["L2_certified"], r


def _drv_p0_seed():
    p = M.apply_uav_profile(M.Params(), "L")
    H = [15, 30]
    wx = dict(_WX_CALM)
    t = _tb_at("N", 2500.0, 0.0)
    sp = RM.ShipPrediction.from_cv(np.zeros(2), np.zeros(2), H, c_state="DP")
    sp.tau_min = 0.0; sp.wx_tau = wx
    opt = RM.LaunchOption(0.0, sp, wx)
    xi = M.XiAmbiguity({(h, "DP"): M.XiCell(h, "DP", 1000, np.zeros(2),
                                                    np.diag([100.0, 100.0]), 0.0, 0.0, 0.0)
                         for h in H}, H)
    seed = [dict(tau=0.0, ship=sp, wx=wx, tids=("N",), h=30.0, E0=0.0,
                 route=RM.Route(-1, [t], sp))]
    r = BP.solve_soft_coverage_research([t], [opt], p, xi, 1, 40.0, max_stops=1,
                             batteries=1, seed_cols=seed, time_limit_s=100,
                             deck_mode="interval")
    assert r["covered"] == 1 and r["energy_Wh"] > 1.0 and \
        r["seed_validation"]["energy_overwritten_count"] == 1 and \
        r["certificate"]["L1_certified"] and r["certificate"]["L2_certified"], r


def _drv_p0new_milp():
    """求解器返回 success=True 但原始向量违反主问题约束时，独立验证器必须拒绝。"""
    import types
    obj = np.array([0.0, -1.0])
    A = np.array([[-1.0, 1.0]])       # y <= x
    b = np.array([0.0])
    lo = np.zeros(2); hi = np.ones(2); integ = np.ones(2)
    lying = types.SimpleNamespace(success=True, status=0,
                                  x=np.array([0.0, 1.0]), fun=-1.0)
    z, value, reason = BP._validate_milp_primal(lying, obj, A, b, lo, hi, integ)
    assert z is None and value is None and reason == "constraint_violation", \
        (z, value, reason)


def _drv_lex_wx():
    """更新 P2-01: lex Stage-2 能耗评估必须用列自己的 per-τ 天气(_wx_of_route),
    与定价/池展开一致。构造: 全局 wx=平静; 唯一可行船带 wind10=9 的 wx_tau(远船不可达)
    ⇒ 若 Stage-2 用全局平静天气, E_LP 会被低估且 h 可行性缓存被污染, L2 证书失真。"""
    p = M.apply_uav_profile(M.Params(), "L")
    t = _tb_at("T0", 2500.0, 0.0)
    x2 = _xi_diag()
    wxw = dict(_WX_CALM)
    wxw["wind10"] = 9.0
    sp_far = RM.ShipPrediction.from_cv(np.array([5.0e5, 0.0]), np.zeros(2),
                                       [15, 30], c_state="DP")
    sp_far.tau_min = 0.0
    sp_far.wx_tau = dict(_WX_CALM)
    sp_w = RM.ShipPrediction.from_cv(np.zeros(2), np.zeros(2), [15, 30], c_state="DP")
    sp_w.tau_min = 5.0
    sp_w.wx_tau = wxw
    res = BP.lex_column_generation([t], sp_far, p, _WX_CALM, x2, max_stops=1,
                                   launch_ships=[sp_far, sp_w], verbose=False)
    # 直接按 per-τ 天气(wind9)重算该序列的最低可行能耗, 作为对拍锚
    r1 = RM.Route(-1, [t], sp_w)
    _es = [RM.route_feasible_at_h(r1, int(h), p, wxw, x2)
           for h in RM.decision_horizons_of(x2)]
    best_w = min(float(d["E_plan_Wh"]) for d in _es if d["feasible"])
    assert res["certified_L1"] and res["certified_L1_L2"], \
        (res["certified_L1"], res["certified_L1_L2"], res["stage2_converged"],
         res["L2_lp_tight"], res["energy_LP_lb"], res["total_energy_Wh"])
    assert res["energy_LP_lb"] is not None and \
        abs(res["total_energy_Wh"] - res["energy_LP_lb"]) \
        <= 1e-3 * max(1.0, res["total_energy_Wh"]), \
        (res["total_energy_Wh"], res["energy_LP_lb"])
    assert abs(res["total_energy_Wh"] - best_w) <= 0.05, \
        (res["total_energy_Wh"], best_w)


def _drv_xi_formal_input():
    """Formal Xi input must reject off-grid horizons and any indefinite covariance."""
    import tempfile
    rows = []
    for h in range(5, 61, 5):
        rows.append(dict(
            mmsi="ALL", h_min=float(h), c_state="直航", n=50, n_effective=50,
            min_cell_n=30, mu_e_m=0.0, mu_n_m=0.0, sigma_ee=4.0, sigma_en=0.0,
            sigma_nn=4.0, max_norm_m=8.0, p95_norm_m=6.0, rms_norm_m=3.0,
            predictor="cv_noleak", predictor_contract=M.XI_PREDICTOR_CONTRACTS["cv_noleak"],
            timestamp_epoch_contract=M.XI_TIMESTAMP_EPOCH_CONTRACT, moments_source="train",
            valid_for_formal=True, purge_min=60.0, sample_overlap_policy="nonoverlap",
            sample_rule="raw_state", source_states="直航", state_merge_policy="low_speed_pair",
            t0_min_iso="2025-01-01T00:00:00Z", t0_max_iso="2025-01-31T23:59:59Z"))
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        bad = [dict(r) for r in rows]
        bad[0]["h_min"] = math.nextafter(5.0, math.inf)
        fp = td / "offgrid.csv"; pd.DataFrame(bad).to_csv(fp, index=False)
        try:
            M.XiAmbiguity.from_csv(fp, mmsi="ALL", formal=True)
        except ValueError:
            pass
        else:
            raise AssertionError("formal Xi off-grid horizon accepted")

        bad = [dict(r) for r in rows]
        for r in bad:
            r["purge_min"] = math.nextafter(60.0, -math.inf)
        fp = td / "short_purge.csv"; pd.DataFrame(bad).to_csv(fp, index=False)
        try:
            M.XiAmbiguity.from_csv(fp, mmsi="ALL", formal=True)
        except ValueError:
            pass
        else:
            raise AssertionError("formal Xi one-ULP-short purge accepted")

        ov = pd.DataFrame([
            dict(mmsi="1", h_min=5, t0_epoch=0.0, t1_epoch=10.0),
            dict(mmsi="1", h_min=5, t0_epoch=math.nextafter(10.0, -math.inf), t1_epoch=20.0),
        ])
        if len(S7.thin_nonoverlap_samples(ov)) != 1:
            raise AssertionError("formal Xi overlapping sample was retained")

        bad = [dict(r) for r in rows]
        bad[0].update(sigma_ee=3.129784, sigma_en=2.985621e6, sigma_nn=3.129782e10)
        fp = td / "indef.csv"; pd.DataFrame(bad).to_csv(fp, index=False)
        try:
            M.XiAmbiguity.from_csv(fp, mmsi="ALL", formal=True)
        except ValueError:
            pass
        else:
            raise AssertionError("formal Xi scale-masked indefinite covariance accepted")

        saa_rows = [dict(
            h_min=5.0, c_state="直航", xi_e_m=float(i), xi_n_m=float(-i),
            predictor="cv_noleak",
            predictor_contract=M.XI_PREDICTOR_CONTRACTS["cv_noleak"],
            timestamp_epoch_contract=M.XI_TIMESTAMP_EPOCH_CONTRACT, mmsi="ALL")
            for i in range(30)]
        stale = td / "saa_stale.csv"
        sdf = pd.DataFrame(saa_rows)
        sdf["timestamp_epoch_contract"] = "legacy-pandas-storage-units"
        sdf.to_csv(stale, index=False)
        try:
            RM.load_saa_empirical(stale)
        except ValueError:
            pass
        else:
            raise AssertionError("stale SAA timestamp contract accepted")
        mech = Namespace(study_mode="mechanism", xi_train_samples=None, saa_samples=None)
        n, usable = S13._register_saa_baseline(mech, stale, set())
        if n != 0 or usable or RM.SAA_EMPIRICAL:
            raise AssertionError("mechanism auto stale SAA did not degrade safely")

        off = td / "saa_offgrid.csv"
        sdf = pd.DataFrame(saa_rows)
        sdf["h_min"] = math.nextafter(5.0, math.inf)
        sdf.to_csv(off, index=False)
        try:
            RM.load_saa_empirical(off)
        except ValueError:
            pass
        else:
            raise AssertionError("off-grid SAA horizon accepted")

        # Formal train-sample reconstruction must preserve already validated Xi
        # provenance; dropping it to "unknown" is a false rejection in step13.
        sample_rows = []
        for k, (xe, xn) in enumerate(((1.0, 0.0), (0.0, 1.0))):
            sample_rows.append(dict(
                mmsi="219018788", source_track="track_219018788.csv",
                source_track_id="fixture-track", h_min=5.0, c_state="直航",
                t0_epoch=1000.0 + 600.0*k, t1_epoch=1300.0 + 600.0*k,
                xi_e_m=xe, xi_n_m=xn, split="train", predictor="cv_noleak",
                predictor_contract=M.XI_PREDICTOR_CONTRACTS["cv_noleak"],
                timestamp_epoch_contract=M.XI_TIMESTAMP_EPOCH_CONTRACT,
                sample_overlap_policy="nonoverlap", purge_min=30.0,
                moments_source="train", valid_for_formal=True))
        sf = td / "formal_train_samples.csv"
        pd.DataFrame(sample_rows).to_csv(sf, index=False)
        sdf = RP.load_samples(sf, mmsi="219018788", formal=True, expected_split="train")
        xia = RP.ambiguity_from_samples(sdf, [5], ["直航"], min_cell_n=2, formal=True)
        if xia.predictor != "cv_noleak":
            raise AssertionError("formal train-sample ambiguity dropped predictor provenance")
        if xia.predictor_contract != M.XI_PREDICTOR_CONTRACTS["cv_noleak"]:
            raise AssertionError("formal train-sample ambiguity dropped predictor contract")
        if xia.timestamp_epoch_contract != M.XI_TIMESTAMP_EPOCH_CONTRACT:
            raise AssertionError("formal train-sample ambiguity dropped epoch contract")


def _drv_weather_formal():
    """Real-weather no-leak predictor and formal moments fail-closed gates."""
    wt = np.array([0.0, 3600.0, 7200.0])
    a = np.array([0.0, 1.0, 100.0]); b = np.array([0.0, 1.0, -100.0])
    p1, _ = S7._forecast_backward_linear(3600.0, 3900.0, wt, a, 5400.0)
    p2, _ = S7._forecast_backward_linear(3600.0, 3900.0, wt, b, 5400.0)
    assert float(p1) == float(p2), (p1, p2)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # Regression for pandas storage-resolution drift: historical weather and Xi
        # must live on the same POSIX-second scale (~1.7e9 in 2025), not 1.7e6.
        _td0 = Path(td)
        _hist = _td0 / "historical_weather.csv"
        pd.DataFrame({
            "time": ["2025-03-01 00:00:00", "2025-03-01 01:00:00",
                     "2025-03-01 02:00:00", "2025-03-01 03:00:00",
                     "2025-03-01 04:00:00", "2025-03-01 05:00:00"],
            "wind10_ms": [4.,5.,6.,7.,8.,9.],
            "wind_dir_from_deg": [0.,10.,20.,30.,40.,50.],
            "Hs_m": [0.2,0.25,0.3,0.35,0.4,0.45],
        }).to_csv(_hist, index=False)
        _, _epoch, _wvec, _wspd, _hs = S7._load_weather(_hist)
        assert _epoch[0] == 1740787200.0, _epoch[:2]
        _xi_one = pd.DataFrame([{
            "h_min": 5.0, "t0_epoch": 1740792600.0, "t1_epoch": 1740792900.0,
            "purge_min": 30.0
        }])
        _res, _drop = S7._make_residuals(_xi_one, _epoch, _wvec, _wspd, _hs, 5400.0, "a"*64, "b"*64)
        assert len(_res) == 1 and _drop == {"no_history":0,"no_truth":0,"nonfinite":0}, (_res, _drop)
        # Weather moments must be sampled on the weather timeline itself and globally
        # non-overlapping for each horizon, independent of vessel/AIS row density.
        _dense_t = np.arange(0.0, 3600.0 + 60.0, 60.0)
        _dense_w = np.column_stack([0.001*_dense_t, -0.0005*_dense_t])
        _dense_sp = np.linalg.norm(_dense_w, axis=1)
        _dense_hs = 0.2 + 1e-5*_dense_t
        _tl, _tls = S7._make_weather_timeline_train_residuals(
            _dense_t, _dense_w, _dense_sp, _dense_hs, [5.0], 120.0, 3300.0, 30.0, 5400.0)
        assert len(_tl) >= 2, _tls
        _ts = _tl.sort_values("t0_epoch")
        assert bool((_ts["t0_epoch"].to_numpy()[1:] >= _ts["t1_epoch"].to_numpy()[:-1]).all()), _ts[["t0_epoch","t1_epoch"]]
        assert set(_tl["sample_overlap_policy"].astype(str)) == {"weather_timeline_global_nonoverlap"}
        fp = Path(td) / "w.csv"
        rows = []
        for h in (5,10,15,20,25,30):
            rows.append(dict(h_min=float(h), n=20, wind_bias_e_ms=0.0, wind_bias_n_ms=0.0,
                wind_sigma_ee=1.0, wind_sigma_en=0.0, wind_sigma_nn=1.0,
                wind_speed_bias_ms=0.0, wind_speed_std_ms=1.0, hs_bias_m=0.0, hs_std_m=0.1,
                predictor="weather_speed_primary_coherent_noleak", predictor_contract=RM.WEATHER_PREDICTOR_CONTRACTS["weather_speed_primary_coherent_noleak"],
                timestamp_epoch_contract=RM.WEATHER_TIMESTAMP_EPOCH_CONTRACT, truth_contract=RM.WEATHER_TRUTH_CONTRACT,
                weather_data_contract=RM.WEATHER_FORMAL_DATA_CONTRACT, moments_source="train",
                sample_overlap_policy="weather_timeline_global_nonoverlap", purge_min=30.0, valid_for_formal=True,
                weather_source_sha256="a"*64, xi_train_source_sha256="b"*64))
        pd.DataFrame(rows).to_csv(fp,index=False)
        assert RM.weather_ambiguity_from_moments_csv(fp,[5,10,15,20,25,30],formal=True).formal_eligible
        bad=pd.DataFrame(rows); bad.loc[0,"h_min"]=np.nextafter(5.0,np.inf); bad.to_csv(fp,index=False)
        try: RM.weather_ambiguity_from_moments_csv(fp,[5,10,15,20,25,30],formal=True)
        except ValueError: pass
        else: raise AssertionError("off-grid weather accepted")
        bad=pd.DataFrame(rows); bad.loc[0,"wind_sigma_ee"]=0.0; bad.loc[0,"wind_sigma_en"]=1.0; bad.loc[0,"wind_sigma_nn"]=0.0; bad.to_csv(fp,index=False)
        try: RM.weather_ambiguity_from_moments_csv(fp,[5,10,15,20,25,30],formal=True)
        except ValueError: pass
        else: raise AssertionError("indefinite weather covariance accepted")


def _drv_formal_exact_contract():
    """Fast formal invariants used to kill false-certificate mutations."""
    # Strict physical comparison: one ULP above a limit is infeasible.
    assert RM._strict_finite_leq(np.nextafter(10.0, np.inf), 10.0) is False

    # Raw formal Xi covariance must be rejected, never silently projected.
    bad_cell = M.XiCell(5, "DP", 10, np.zeros(2),
                        np.array([[0.0, 1.0], [1.0, 0.0]]), 0.0, 0.0, 0.0)
    try:
        bad_xi = M.XiAmbiguity({(5, "DP"): bad_cell}, [5])
        M.validate_xi_ambiguity_math(bad_xi)
    except ValueError:
        pass
    else:
        raise AssertionError("indefinite raw formal Xi covariance was projected/accepted")

    # Resource audit is tri-state on timeout and exact-rational on SOC.
    e_bad = np.nextafter(1.0, np.inf)
    cols = FAC.validate_route_columns([dict(
        tids=("A",), tau=0.0, h=1.0, E_plan_Wh=1.0,
        E_soc_required_Wh=float(e_bad))])
    rmap = {0: dict(launch_start_min=0.0, recovery_min=1.0,
                    clear_end_min=1.0, deck=[])}
    soc = RA.audit_resource_assignment(cols, (0,), 1, 1, 1.0, rmap,
                                       0.0, 0.0, 1, 1, deadline=None)
    assert soc.status is FAC.ResourceAuditStatus.INFEASIBLE_PROVEN
    unk = RA.audit_resource_assignment(cols, (0,), 1, 1, 2.0, rmap,
                                       0.0, 0.0, 1, 1,
                                       deadline=time.monotonic() - 1.0)
    assert unk.status is FAC.ResourceAuditStatus.UNKNOWN_TIMEOUT

    p = M.Params()
    cA = BP._normalize_exact_column(_bpc_test_column(("A",), 0, 5, 1), p=p,
                                    t_launch_min=0.0, landing_clear_min=0.0,
                                    deck_mode="interval", deck_delta_min=2.5)
    cB = BP._normalize_exact_column(_bpc_test_column(("B",), 10, 5, 1), p=p,
                                    t_launch_min=0.0, landing_clear_min=0.0,
                                    deck_mode="interval", deck_delta_min=2.5)
    cut = frozenset({BP._exact_route_signature(cA)})
    assert BP._row_coefficient(cA, ("resource_pattern", cut)) == 1.0
    assert BP._row_coefficient(cB, ("resource_pattern", cut)) == -1.0
    assert BP._column_allowed_at_node(
        cB, BP.BranchState(forbidden_turbines=frozenset({"B"}))) is False

    # For min c'x with u<=0, rc=c-uA.  Here c=-1,u=-2,A=1 => rc=+1.
    _, rc_lo, rc_hi = BP._column_reduced_cost_interval(
        cA, "coverage", [("packing", "A")], [], [-2.0], [])
    assert rc_lo > 0.0 and rc_hi > 0.0

    # Incomplete pricing is never closure, regardless of a convenient bound.
    inc = BP.PricingSearchResult([], False, None, 0.0, True, 0, 0, "fixture")
    assert inc.closed is False

    # Phase-I must include M_n min(0,delta); omitting it would turn 0.5 into a
    # false positive infeasibility proof when delta=-1 and M_n=1.
    phase = BP.RestrictedMasterResult(
        "optimal", np.zeros(0), 0.5, 0.5, np.zeros(0), np.zeros(0), [], [], [],
        np.zeros((0, 0)), np.zeros(0), np.zeros((0, 0)), np.zeros(0), np.zeros(0),
        phase_one_value=0.5)
    pr = BP.PricingSearchResult([], False, None, -1.0, True, 0, 0, "fixture")
    proved, full_lb = BP._phase_one_infeasibility_proven(phase, pr, 1, BP.ART_TOL)
    assert proved is False and full_lb < 0.0

    # Certificate provenance: public exact path rejects a supplied synthetic
    # route universe, while the private algorithmic fixture can solve it only
    # with the physical/global certificate disabled.
    injected = [_bpc_test_column(("A",), 0.0, 5.0, 1.0)]
    try:
        BP.solve_fleet_anytime(_bpc_turbines("A"), [], M.Params(), _bpc_xi(),
                               1, 60.0, batteries=1, max_stops=1,
                               time_limit_s=2.0, implicit_test_columns=injected)
    except ValueError:
        pass
    else:
        raise AssertionError("public exact path accepted synthetic route universe")
    syn = BP._solve_fleet_anytime_synthetic_fixture(
        _bpc_turbines("A"), [], M.Params(), _bpc_xi(), 1, 60.0,
        batteries=1, max_stops=1, time_limit_s=5.0,
        allow_resource_only_columns=True, implicit_test_columns=injected,
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0)
    assert syn["algorithmic_global_certificate"] is True
    assert syn["physical_model_global_certificate"] is False
    assert syn["global_certificate_available"] is False

    # Canonical route identity cannot change objective/resource semantics.
    dup_lo = BP._normalize_exact_column(_bpc_test_column(("A",), 0, 5, 0.5), p=p,
                                        t_launch_min=0.0, landing_clear_min=0.0,
                                        deck_mode="interval", deck_delta_min=2.5)
    ar = [cA]; sm = {BP._exact_route_signature(cA): 0}
    try:
        BP._add_columns(ar, sm, [dup_lo])
    except RuntimeError:
        pass
    else:
        raise AssertionError("same route signature changed formal energy semantics")

    # Unknown future master rows cannot be silently dropped from universal rc bounds.
    try:
        BP._universal_pricing_lower_bound(
            "coverage", 2, [("future_signed_cut", None)], [], [-1.0], [])
    except ValueError:
        pass
    else:
        raise AssertionError("unknown future row silently received a pricing certificate")

    # Required arcs are aggregate equalities, not route-admissibility filters.
    cC = BP._normalize_exact_column(_bpc_test_column(("C",), 20, 5, 1), p=p,
                                    t_launch_min=0.0, landing_clear_min=0.0,
                                    deck_mode="interval", deck_delta_min=2.5)
    req_arc = BP.BranchState(required_arcs=frozenset({("A", "B")}))
    assert BP._column_allowed_at_node(cC, req_arc) is True

    # Node route-mass bound excludes forbidden-service turbines.
    mass = BP.BranchState(forbidden_turbines=frozenset({"B", "D"}))
    assert BP._node_allowed_turbine_bound(("A", "B", "C", "D"), mass) == 2

    # Formal real-weather validation must preserve the complete WeatherAmbiguity
    # provenance envelope across the step13 -> step15 replay boundary.
    import inspect
    _replay_src = inspect.getsource(S13._replay_columns)
    assert "weather_unc=wamb" in _replay_src
    assert "weather_unc=wu_cell" not in _replay_src and "weather_unc=wu," not in _replay_src

    # [THM-LEX] final physical certificate is a literal fail-closed conjunction.
    guard = dict(algorithmic_global_certificate=True,
                 route_universe_provenance_certified=True,
                 mode="exact-branch-price-cut",
                 route_semantics_invariance_certified=True,
                 future_column_row_ranges_certified=True,
                 binary64_model_contract_enforced=True,
                 formal_proof_contract_enforced=True)
    assert BP._physical_certificate_guard(**guard) is True
    for key in ("route_universe_provenance_certified",
                "route_semantics_invariance_certified",
                "future_column_row_ranges_certified",
                "binary64_model_contract_enforced",
                "formal_proof_contract_enforced"):
        bad = dict(guard); bad[key] = False
        assert BP._physical_certificate_guard(**bad) is False, key



def _drv_e1_staged_contract():
    """Fast mutation driver for coverage-only certificate scope and E1 sandwich logic."""
    cov_only = BP.solve_fleet_anytime(
        _bpc_turbines("A"), [], M.Params(), _bpc_xi(), 1, 60.0,
        batteries=1, max_stops=1, time_limit_s=2.0, solve_scope="coverage-only",
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0)
    assert cov_only["coverage_global_certificate_available"] is True
    assert cov_only["coverage_physical_model_certificate"] is True
    assert cov_only["energy_optimal"] is False
    assert cov_only["lexicographic_optimal"] is False
    assert cov_only["algorithmic_global_certificate"] is False
    assert cov_only["global_certificate_available"] is False

    rows = []
    for K in (1, 2):
        for B in range(5):
            cov = min(K, B, 2)
            rows.append(dict(
                uav="S", K=K, batteries=B, safe_served=cov,
                coverage_incumbent=cov, coverage_upper_bound=cov, covered=cov,
                coverable_note=3, plan_holds=(True if cov else None),
                coverage_global_certificate_available=True, study_mode="formal",
                global_certificate_available=(True if (K, B) == (2, 2) else False),
                global_route_space_certificate=(True if (K, B) == (2, 2) else False),
                implicit_route_space_certified=(True if (K, B) == (2, 2) else False),
                inventory_energy_kWh=0.2 * B,
                safe_per_inventory_kWh=(None if B == 0 else cov / (0.2 * B)),
                energy_per_safe=(10.0 if cov else None),
                per_battery=(None if B == 0 else cov / B),
                max_stops_requested=4, stops_cap_spec="4", max_stops_effective=4,
                stops_cap=4, max_stops_observed=1, stops_cap_hit=False))
    df = pd.DataFrame(rows)
    ok = S13.e1_select_from_df(df, frac=0.95, order="BK", patience=2).iloc[0]
    assert ok["plateau_coverage_certified"] == True
    df.loc[(df.K == 2) & (df.batteries == 4), "coverage_upper_bound"] = 3
    bad = S13.e1_select_from_df(df, frac=0.95, order="BK", patience=2).iloc[0]
    assert bad["selection_status"] == "uncertified_coverage_plateau"

    assert S13._e1_plateau_long_refinement_allowed("off") is False
    assert S13._e1_plateau_long_refinement_allowed("on") is True
    assert S13._e1_bound_strictly_improved((1, 5), (1, 5)) is False
    assert S13._e1_bound_strictly_improved((1, 5), (2, 5)) is True
    env = pd.DataFrame([
        dict(K=1, batteries=1, coverage_incumbent=1, coverage_upper_bound=8, coverable_note=8),
        dict(K=2, batteries=2, coverage_incumbent=2, coverage_upper_bound=8, coverable_note=8),
        dict(K=3, batteries=4, coverage_incumbent=2, coverage_upper_bound=2, coverable_note=8),
    ])
    assert S13._e1_monotone_coverage_interval(env, 1, 1) == (1, 2)
    assert S13._e1_monotone_coverage_interval(env, 2, 2) == (2, 2)

    e2 = pd.DataFrame([dict(criterion="vp", q=0.8, run_status="ok", holds=True,
                            n_missing_replay=0, covered=3, safe_served=3,
                            energy_per_safe=10.0, energy_Wh=30.0, study_mode="formal",
                            global_certificate_available=False, global_route_space_certificate=False,
                            implicit_route_space_certified=False)])
    assert S13._select_e2_validation_candidate(e2, (0.2, 0.5, 0.8)) is None

def _drv_v9_single_vessel_protocol():
    """P0 v9: one vessel only, no formal pooled fallback, one final-test consumer."""
    from types import SimpleNamespace as _NS
    assert not S13._e1_final_test_consumption_allowed(
        _NS(study_mode="formal", final_test_samples="holdout.csv"))
    assert S13._e1_final_test_consumption_allowed(
        _NS(study_mode="mechanism", final_test_samples="holdout.csv"))
    assert not S13._e1_final_test_consumption_allowed(
        _NS(study_mode="mechanism", final_test_samples=None))
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rows = []
        for mmsi, offset in (("219018788", 0.0), ("999999999", 100.0)):
            for k in range(2):
                rows.append(dict(
                    mmsi=mmsi, source_track="track_"+mmsi+".csv",
                    source_track_id="fixture-"+mmsi, h_min=5.0, c_state="直航",
                    t0_epoch=1000.0 + offset + 600.0*k,
                    t1_epoch=1300.0 + offset + 600.0*k,
                    xi_e_m=float(k), xi_n_m=float(-k), split="train",
                    predictor="cv_noleak",
                    predictor_contract=M.XI_PREDICTOR_CONTRACTS["cv_noleak"],
                    timestamp_epoch_contract=M.XI_TIMESTAMP_EPOCH_CONTRACT,
                    sample_overlap_policy="nonoverlap", purge_min=30.0,
                    moments_source="train", valid_for_formal=True))
        fp = td / "mixed_train.csv"
        pd.DataFrame(rows).to_csv(fp, index=False)

        one = RP.load_samples(
            fp, mmsi="219018788", formal=True, expected_split="train")
        xia = RP.ambiguity_from_samples(
            one, [5], ["直航"], min_cell_n=2, formal=True)
        assert xia.selected_mmsi == "219018788"
        assert xia.cross_vessel_pooling is False

        mixed = RP.load_samples(fp, mmsi="ALL", formal=True, expected_split="train")
        try:
            RP.ambiguity_from_samples(
                mixed, [5], ["直航"], min_cell_n=2, formal=True)
        except ValueError:
            pass
        else:
            raise AssertionError("formal ambiguity accepted cross-vessel mmsi=ALL")

        saa_rows = []
        for mmsi, n in (("219018788", 1), ("999999999", 4)):
            for k in range(n):
                saa_rows.append(dict(
                    mmsi=mmsi, h_min=5.0, c_state="直航",
                    xi_e_m=float(k), xi_n_m=0.0, predictor="cv_noleak",
                    predictor_contract=M.XI_PREDICTOR_CONTRACTS["cv_noleak"],
                    timestamp_epoch_contract=M.XI_TIMESTAMP_EPOCH_CONTRACT))
        sf = td / "saa.csv"
        pd.DataFrame(saa_rows).to_csv(sf, index=False)
        n = RM.load_saa_empirical(
            sf, mmsi="219018788", min_n=2,
            require_current_contract=True, allow_pooled_fallback=False)
        assert n == 0 and not RM.SAA_EMPIRICAL
        assert RM.SAA_ALLOW_MOMENT_FALLBACK is False

    # Behavior-level replay boundary: another MMSI in the same holdout file
    # must not change the selected-vessel replay sample count.
    t13_stage5_replay_contracts()
    t18_frozen_final_test_protocols()


def _drv_v10_target_knee_protocol():
    """v10 target decision, refined monotone knee, conservative validation gate."""
    from types import SimpleNamespace as _NS
    good = _NS(open_nodes=0, resource_audit_complete=True,
               branching_complete=True, pricing_bound_available=True,
               farkas_pricing_complete=True)
    assert BP._target_infeasibility_algorithmic_proven(good, False)
    for field in ("resource_audit_complete", "branching_complete",
                  "pricing_bound_available", "farkas_pricing_complete"):
        bad = _NS(**vars(good)); setattr(bad, field, False)
        assert not BP._target_infeasibility_algorithmic_proven(bad, False), field
    bad = _NS(**vars(good)); bad.open_nodes = 1
    assert not BP._target_infeasibility_algorithmic_proven(bad, False)
    assert not BP._target_infeasibility_algorithmic_proven(good, True)

    p = M.apply_uav_profile(M.Params(), "L")
    p.time_recourse_mode = "wait_and_speed"; p.speed_adjustable = True
    p.validate_contract(formal=False)
    tb = [_tb_at("TGT", 100.0, 0.0)]
    opt = [_mk_launch(2.5, (0.0, 0.0), _WX_CALM, horizons=(15, 30))]
    xi = _xi_diag((15, 30), var=1.0)
    yes = BP.solve_fleet_anytime(
        tb, opt, p, xi, 1, 60.0, batteries=1, max_stops=1,
        time_limit_s=8.0, solve_scope="coverage-target", coverage_target=1,
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0)
    assert yes["target_decision"] == "FEASIBLE" and yes["target_feasible_proven"], yes

    rows = []
    for K in (1, 2, 3):
        for B in range(9):
            cap = {1: 3, 2: 6, 3: 8}[K]
            cov = min(B, cap)
            rows.append(dict(
                uav="Q", K=K, batteries=B, safe_served=max(0, cov-1),
                per_battery=(None if B == 0 else max(0, cov-1)/B),
                coverage_incumbent=cov, coverage_upper_bound=8,
                covered=cov, coverable_note=8, plan_holds=False,
                global_certificate_available=False,
                global_route_space_certificate=False,
                implicit_route_space_certified=False,
                coverage_global_certificate_available=(K == 3 and B == 8),
                study_mode="formal", max_stops_requested=4, stops_cap_spec="4",
                max_stops_effective=4, stops_cap=4, max_stops_observed=1,
                stops_cap_hit=False))
    df = pd.DataFrame(rows)
    s0 = S13.e1_select_from_df(df, frac=0.95, order="BK", patience=2).iloc[0]
    assert S13._e1_target_blockers(df, "Q", s0, "BK") == [(3, 7, 8, "B-predecessor")]
    no = dict(target_decision="INFEASIBLE", target_decision_certified=True,
              target_feasible_proven=False, target_infeasible_proven=True,
              target_certificate_type="full-space-phase1-bpc-infeasibility")
    S13._apply_target_decision_to_frontier(df, "Q", 3, 7, 8, no)
    assert S13._e1_raw_coverage_interval_record(
        df[(df.K == 2) & (df.batteries == 8)].iloc[0]) == (6, 8)
    s1 = S13.e1_select_from_df(df, frac=0.95, order="BK", patience=2).iloc[0]
    assert S13._e1_target_blockers(df, "Q", s1, "BK") == [(2, 8, 8, "K-predecessor")]
    S13._apply_target_decision_to_frontier(df, "Q", 2, 8, 8, no)
    s2 = S13.e1_select_from_df(df, frac=0.95, order="BK", patience=2).iloc[0]
    assert s2["selection_status"] == "needs_lexicographic_knee_certification", dict(s2)
    assert (int(s2["knee_K"]), int(s2["knee_B"])) == (3, 8)

    assert S13._formal_validation_selection_gate(
        {"allocation_budget_holds": False,
         "mission_requirement_holds": True}) is False

    # v9 -> v10 migration accepts old rigorous bounds only on identical
    # physical/data provenance; proof/result contract is not silently upgraded.
    import tempfile as _tmp
    with _tmp.TemporaryDirectory() as _td:
        _td = Path(_td)
        train = _td / "train.csv"; val = _td / "val.csv"; test = _td / "test.csv"
        for fp, marker in ((train, "a"), (val, "b"), (test, "c")):
            fp.write_text(marker, encoding="utf-8")
        a = _NS(_resolved_xi_mmsi="219018788",
                validation_samples=val, xi_train_samples=train,
                final_test_samples=test)
        v9 = pd.DataFrame([dict(
            uav="S", K=3, batteries=8, study_mode="formal",
            result_contract="fleet-anytime-result-v9-single-vessel-final-test-freeze",
            physical_numeric_contract=RM.FORMAL_PHYSICAL_NUMERIC_CONTRACT,
            route_identity_contract=BP.ROUTE_IDENTITY_CONTRACT,
            model_semantics_contract=BP.MODEL_SEMANTICS_CONTRACT,
            resume_input_sha256="same-physical-instance",
            xi_mmsi="219018788",
            validation_samples_hash=EU.sha256_file(val),
            xi_train_samples_hash=EU.sha256_file(train),
            final_test_samples_hash=EU.sha256_file(test))])
        for resume_sha in ("same-physical-instance", "different-physical-instance"):
            try:
                S13._validate_e1_frontier_for_target_refine(v9, a, resume_sha)
            except SystemExit:
                pass
            else:
                raise AssertionError("hardened formal target-refine accepted a legacy v9 frontier")


def _drv_v11_complete_universe():
    """v11 [THM-CU]: complete physical universe is exact acceleration, never a model shortcut."""
    p = M.apply_uav_profile(M.Params(), "L")
    p.time_recourse_mode = "wait_and_speed"; p.speed_adjustable = True
    p.validate_contract(formal=False)
    tb = [_tb_at("CU", 100.0, 0.0)]
    opt = [_mk_launch(2.5, (0.0, 0.0), _WX_CALM, horizons=(15, 30))]
    xi = _xi_diag((15, 30), var=1.0)
    cu = BP.build_certified_route_universe(
        tb, opt, p, xi, 60.0, max_stops=1, time_limit_s=8.0,
        t_launch_min=2.5, landing_clear_min=1.0,
        deck_mode="interval", deck_delta_min=2.5)
    assert cu.complete and cu.columns
    ok, why = BP._validate_certified_route_universe(
        cu, tb, opt, p, xi, 60.0, max_stops=1, weather_unc=None,
        kappa_mode="vp_unimodal", chance_mode="drcc", budget_gamma=2.0,
        t_launch_min=2.5, landing_clear_min=1.0,
        deck_mode="interval", deck_delta_min=2.5)
    assert ok, why
    yes = BP.solve_fleet_anytime(
        tb, opt, p, xi, 1, 60.0, batteries=1, max_stops=1,
        time_limit_s=8.0, solve_scope="coverage-target", coverage_target=1,
        certified_route_universe=cu,
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0)
    assert yes["target_decision"] == "FEASIBLE" and yes["target_decision_certified"]
    assert yes["pricing_calls"] == 0 and yes["exact_pricing_calls"] == 0
    assert yes["route_space_complete"] and yes["route_space_materialized"]

    cu0 = BP.build_certified_route_universe(
        tb, [], p, xi, 60.0, max_stops=1, time_limit_s=3.0,
        t_launch_min=2.5, landing_clear_min=1.0,
        deck_mode="interval", deck_delta_min=2.5)
    no = BP.solve_fleet_anytime(
        tb, [], p, xi, 1, 60.0, batteries=1, max_stops=1,
        time_limit_s=5.0, solve_scope="coverage-target", coverage_target=1,
        certified_route_universe=cu0,
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0)
    assert no["target_decision"] == "INFEASIBLE" and no["target_decision_certified"]
    assert no["phase_one_solves"] == 0 and no["pricing_calls"] == 0
    # v16 strengthens the same certified-complete-universe NO: an empty
    # structural route universe is already infeasible in THM-GBR, before the
    # pattern-level v15 closure.  Keep the old THM-CU regression but require the
    # stronger exact certificate path rather than pinning a historical backend.
    assert no["target_certificate_type"] == (
        "complete-materialized-universe-fullcover-global-battery-relaxation-infeasibility")
    assert no["target_master_backend"] == "exact-global-battery-mask-dp"
    assert no["target_global_battery_relaxation_status"] == "INFEASIBLE_PROVEN"

    # Partial targets deliberately stay on the general exact BPC path.  This
    # keeps THM-CU coverage of complete-universe node closure/pricing bypass,
    # while THM-FCT remains a strict full-cover-only specialization.
    tb2 = [_tb_at("CUA", 100.0, 0.0), _tb_at("CUB", 150.0, 0.0)]
    cu2 = BP.build_certified_route_universe(
        tb2, opt, p, xi, 60.0, max_stops=1, time_limit_s=8.0,
        t_launch_min=2.5, landing_clear_min=1.0,
        deck_mode="interval", deck_delta_min=2.5)
    partial = BP.solve_fleet_anytime(
        tb2, opt, p, xi, 1, 60.0, batteries=1, max_stops=1,
        time_limit_s=8.0, solve_scope="coverage-target", coverage_target=1,
        certified_route_universe=cu2,
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0)
    assert partial["target_decision"] == "FEASIBLE" and partial["target_decision_certified"]
    assert partial.get("target_master_backend") is None
    assert partial["pricing_calls"] == 0 and partial["exact_pricing_calls"] == 0

    cu2_empty = BP.build_certified_route_universe(
        tb2, [], p, xi, 60.0, max_stops=1, time_limit_s=3.0,
        t_launch_min=2.5, landing_clear_min=1.0,
        deck_mode="interval", deck_delta_min=2.5)
    partial_no = BP.solve_fleet_anytime(
        tb2, [], p, xi, 1, 60.0, batteries=1, max_stops=1,
        time_limit_s=5.0, solve_scope="coverage-target", coverage_target=1,
        certified_route_universe=cu2_empty,
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0)
    assert partial_no["target_decision"] == "INFEASIBLE" and partial_no["target_decision_certified"]
    assert partial_no["phase_one_solves"] == 0 and partial_no["pricing_calls"] == 0

    # Incomplete / stale / tampered acceleration evidence must fail closed.
    bad_incomplete = BP.CertifiedRouteUniverse(
        columns=cu.columns, complete=False, context_sha256=cu.context_sha256,
        columns_sha256=cu.columns_sha256, builder_contract=cu.builder_contract,
        stats=dict(cu.stats))
    try:
        BP.solve_fleet_anytime(
            tb, opt, p, xi, 1, 60.0, batteries=1, max_stops=1,
            time_limit_s=3.0, solve_scope="coverage-target", coverage_target=1,
            certified_route_universe=bad_incomplete)
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete route universe accepted")

    bad_hash = BP.CertifiedRouteUniverse(
        columns=cu.columns, complete=True, context_sha256=cu.context_sha256,
        columns_sha256="0"*64, builder_contract=cu.builder_contract,
        stats=dict(cu.stats))
    try:
        BP.solve_fleet_anytime(
            tb, opt, p, xi, 1, 60.0, batteries=1, max_stops=1,
            time_limit_s=3.0, solve_scope="coverage-target", coverage_target=1,
            certified_route_universe=bad_hash)
    except ValueError:
        pass
    else:
        raise AssertionError("tampered route universe hash accepted")

    p2 = M.apply_uav_profile(M.Params(), "L")
    p2.B_k = float(p2.B_k) + 1.0
    p2.time_recourse_mode = "wait_and_speed"; p2.speed_adjustable = True
    try:
        BP.solve_fleet_anytime(
            tb, opt, p2, xi, 1, 60.0, batteries=1, max_stops=1,
            time_limit_s=3.0, solve_scope="coverage-target", coverage_target=1,
            certified_route_universe=cu)
    except ValueError:
        pass
    else:
        raise AssertionError("stale route universe context accepted")

    # v11 knee detail persistence must accept the list returned by
    # _e1_detail_rows; this path was previously masked because no real knee had
    # closed far enough to execute it.
    import tempfile as _cu_tmp
    with _cu_tmp.TemporaryDirectory() as _td:
        _out = Path(_td)
        _saved = S13._save_e1_knee_detail(
            [dict(route_id=0, uav="L", turbines="TGT", stops=1)],
            _out, "L", 3, 5)
        assert _saved is not None and Path(_saved).is_file()
        _dd = pd.read_csv(_saved)
        assert set(_dd["solution_role"]) == {"certified_resource_knee_full_lex"}
        assert set(_dd["K"].astype(int)) == {3}
        assert set(_dd["batteries"].astype(int)) == {5}


def _drv_v12_resource_closure():
    """v12 [THM-FCT]: discrete threshold + direct full-cover resource closure."""
    # A four-route full-cover pattern passes the v15 direct-master necessary
    # relaxations (active min(K,B), fastest-turnaround intervals, and battery
    # bin packing) but is still resource-infeasible: two UAVs must perform
    # overlapping quick-inspection events while quick_capacity=1 and the two
    # batteries are already permanently bound one per UAV.  This keeps the
    # historical THM-FCT test focused on the exact resource strong cut rather
    # than on a newly stronger static relaxation.
    p = M.Params()
    cA = BP._normalize_exact_column(
        _bpc_test_column(("A",), 0.0, 5.0, 1.0), p=p,
        t_launch_min=0.0, landing_clear_min=0.0,
        deck_mode="interval", deck_delta_min=2.5)
    cB = BP._normalize_exact_column(
        _bpc_test_column(("B",), 0.2, 5.0, 1.0), p=p,
        t_launch_min=0.0, landing_clear_min=0.0,
        deck_mode="interval", deck_delta_min=2.5)
    cC = BP._normalize_exact_column(
        _bpc_test_column(("C",), 7.0, 5.0, 1.0), p=p,
        t_launch_min=0.0, landing_clear_min=0.0,
        deck_mode="interval", deck_delta_min=2.5)
    cD = BP._normalize_exact_column(
        _bpc_test_column(("D",), 7.2, 5.0, 1.0), p=p,
        t_launch_min=0.0, landing_clear_min=0.0,
        deck_mode="interval", deck_delta_min=2.5)
    archive = [cA, cB, cC, cD]
    no = BP._solve_complete_universe_fullcover_target(
        archive=archive, all_tids=("A", "B", "C", "D"), K=2, batteries=2, p=p,
        deadline=time.monotonic() + 4.0, deck_times=(), active_times=(),
        pooled_energy_cap=float(p.B_use) * 2.0, quick_min=2.0, swap_min=10.0,
        quick_capacity=1, swap_capacity=1)
    assert no.termination_reason == "fullcover-target-master-infeasible-proven", no
    assert no.open_nodes == 0 and no.fullcover_strong_cuts == 1, no
    assert no.target_master_solves == 2 and no.resource_audit_calls == 1, no
    assert no.direct_target_backend in {"scipy-milp+exact-fullcover-dfs", "gurobi+exact-fullcover-dfs"}, no

    # The production stage dispatcher must actually route a complete-universe
    # full-cover target through THM-FCT rather than silently falling back to the
    # old custom branch-price tree.
    sig_full = {BP._exact_route_signature(c): j for j, c in enumerate(archive)}
    dispatched = BP._solve_branch_price_stage(
        stage="energy", turbines=_bpc_turbines("A", "B", "C", "D"), launch_opts=[],
        p=p, xi_amb=_bpc_xi(), K=2, batteries=2, T_min=60.0, max_stops=1,
        weather_unc=None, deadline=time.monotonic() + 4.0,
        archive=archive, signature_to_index=sig_full, no_good_cuts=[],
        coverage_target=4, initial_selection=(), initial_audit=None,
        coverage_gap_target_abs=0, energy_gap_target_rel=0.0,
        energy_gap_target_abs_Wh=0.0, t_launch_min=0.0,
        landing_clear_min=0.0, quick_min=2.0, swap_min=10.0,
        quick_capacity=1, swap_capacity=1, deck_mode="interval",
        deck_delta_min=2.5, kappa_mode="vp_unimodal", chance_mode="drcc",
        budget_gamma=2.0, implicit_test_columns=None, pricing_batch_size=16,
        root_branch=None, physical_cache={}, decision_only=True,
        complete_universe_mode=True)
    assert dispatched.termination_reason == "fullcover-target-master-infeasible-proven"
    assert dispatched.direct_target_backend in {"scipy-milp+exact-fullcover-dfs", "gurobi+exact-fullcover-dfs"}
    assert dispatched.fullcover_strong_cuts == 1

    # THM-FCT is *strictly* full-cover-only.  Partial target T=1 over two
    # turbines must remain on the general complete-universe exact BPC path.
    cols = [
        BP._normalize_exact_column(
            _bpc_test_column(("A",), 0.0, 5.0, 1.0), p=p,
            t_launch_min=0.0, landing_clear_min=0.0,
            deck_mode="interval", deck_delta_min=2.5),
        BP._normalize_exact_column(
            _bpc_test_column(("B",), 10.0, 5.0, 1.0), p=p,
            t_launch_min=0.0, landing_clear_min=0.0,
            deck_mode="interval", deck_delta_min=2.5),
    ]
    sig = {BP._exact_route_signature(c): j for j, c in enumerate(cols)}
    partial = BP._solve_branch_price_stage(
        stage="energy", turbines=_bpc_turbines("A", "B"), launch_opts=[],
        p=p, xi_amb=_bpc_xi(), K=1, batteries=2, T_min=60.0, max_stops=1,
        weather_unc=None, deadline=time.monotonic() + 4.0,
        archive=cols, signature_to_index=sig, no_good_cuts=[],
        coverage_target=1, initial_selection=(), initial_audit=None,
        coverage_gap_target_abs=0, energy_gap_target_rel=0.0,
        energy_gap_target_abs_Wh=0.0, t_launch_min=0.0,
        landing_clear_min=0.0, quick_min=1.0, swap_min=6.0,
        quick_capacity=1, swap_capacity=1, deck_mode="interval",
        deck_delta_min=2.5, kappa_mode="vp_unimodal", chance_mode="drcc",
        budget_gamma=2.0, implicit_test_columns=None, pricing_batch_size=16,
        root_branch=None, physical_cache={}, decision_only=True,
        complete_universe_mode=True)
    assert partial.coverage_incumbent == 1
    assert partial.direct_target_backend is None

    # Formal rho=.95 on an 8-turbine certified plateau rounds to T=8 by
    # definition.  That is a valid discrete threshold, not a geometric-knee
    # degeneracy, and an otherwise eligible formal point must remain selectable.
    rows = []
    for K in (1, 2, 3):
        for B in range(9):
            cap = {1: 3, 2: 6, 3: 8}[K]
            cov = min(B, cap)
            rows.append(dict(
                uav="Q", K=K, batteries=B, safe_served=max(0, cov-1),
                per_battery=(None if B == 0 else max(0, cov-1)/B),
                coverage_incumbent=cov, coverage_upper_bound=8,
                covered=cov, coverable_note=8, plan_holds=False,
                global_certificate_available=False,
                global_route_space_certificate=False,
                implicit_route_space_certified=False,
                coverage_global_certificate_available=(K == 3 and B == 8),
                study_mode="formal", max_stops_requested=4, stops_cap_spec="4",
                max_stops_effective=4, stops_cap=4, max_stops_observed=1,
                stops_cap_hit=False))
    df = pd.DataFrame(rows)
    no_tgt = dict(target_decision="INFEASIBLE", target_decision_certified=True,
                  target_feasible_proven=False, target_infeasible_proven=True,
                  target_certificate_type="unit-target-no")
    S13._apply_target_decision_to_frontier(df, "Q", 3, 7, 8, no_tgt)
    S13._apply_target_decision_to_frontier(df, "Q", 2, 8, 8, no_tgt)
    s = S13.e1_select_from_df(df, frac=0.95, order="BK", patience=2).iloc[0]
    assert int(s["coverage_threshold"]) == 8
    assert bool(s["threshold_equals_plateau"])
    assert bool(s["threshold_rounded_to_full_coverage"])
    assert not bool(s["degenerate_knee"])
    eligible = s.copy()
    eligible["selection_status"] = "eligible"
    eligible["knee_plan_holds"] = True
    eligible["knee_resource_minimality_certified"] = True
    eligible["knee_global_certificate_available"] = True
    eligible["knee_coverage_certificate_available"] = True
    eligible["resource_threshold_point_valid"] = True
    eligible["knee_safe_per_inventory_kWh"] = 1.0
    eligible["knee_energy_per_safe"] = 1.0
    eligible["formal_selection_contract"] = "discrete-coverage-threshold-min-resource-plus-lex-v3"
    eligible["source_result_contract"] = S13.RESULT_CONTRACT
    eligible["source_formal_experiment_scheduler_contract"] = S13.FORMAL_EXPERIMENT_SCHEDULER_CONTRACT
    eligible["source_result_certificate_contract"] = BP.RESULT_CERTIFICATE_CONTRACT
    eligible["source_formal_proof_contract"] = BP.FORMAL_PROOF_CONTRACT
    eligible["degenerate_knee"] = True  # legacy flag must not veto formal threshold
    pick, warns = S13._pick_selection(pd.DataFrame([eligible]))
    assert pick is not None and not warns
    tampered = eligible.copy()
    tampered["knee_global_certificate_available"] = False
    pick_bad, _ = S13._pick_selection(pd.DataFrame([tampered]))
    assert pick_bad is None

    # Once hard-coverable-cap fixes P/T, generic max-coverage long refinement
    # is forbidden; exact predecessor target closure is the only formal blocker.
    assert not S13._formal_resource_knee_generic_refinement_allowed(
        {"selection_status": "uncertified_resource_knee",
         "saturation_proof": "hard-coverable-cap"})
    assert S13._formal_resource_knee_generic_refinement_allowed(
        {"selection_status": "uncertified_resource_knee",
         "saturation_proof": "monotone-sandwich:B=4->8"})

    # Duplicate empirical event fingerprints are audit evidence only.  They do
    # not relax the predeclared per-sortie allocation gate.
    assert S13._formal_validation_selection_gate({
        "allocation_budget_holds": False,
        "mission_requirement_holds": True,
        "validation_unique_event_fingerprint_count": 1,
        "validation_event_fingerprint_count": 8,
        "validation_duplicate_event_groups": 1,
        "validation_event_grouping_used_for_gate": False,
    }) is False


def _drv_dual_fail():
    """更新 P2-03: 划分 LP 的对偶提取失败必须 fail-closed(返回 None 元组),
    不得静默零对偶导致 Stage-2/3 假收敛。"""
    import types
    import scipy.optimize as _spo
    orig = _spo.linprog

    def _no_duals(*a, **k):
        return types.SimpleNamespace(success=True, fun=1.0, status=0,
                                     x=np.array([1.0, 0.0]))
    _spo.linprog = _no_duals
    try:
        out = BP._energy_partition_lp([(1.0, {"T"})], ["T"], 1)
        out2 = BP._robust_partition_lp([(0.5, 1.0, {"T"})], ["T"], 1, 1.0)
    finally:
        _spo.linprog = orig
    assert out == (None, None, None), out
    assert out2 == (None, None, None, None, None), out2


def _drv_v13_hardening():
    """Audit regressions added after the independent v12 P0 review."""
    # H-82: a floating backend may lie about infeasibility; the exact full-cover
    # verifier must recover the obvious witness instead of signing target-NO.
    p = M.Params()
    c = BP._normalize_exact_column(
        _bpc_test_column(("A",), 0.0, 5.0, 1.0), p=p,
        t_launch_min=0.0, landing_clear_min=0.0,
        deck_mode="interval", deck_delta_min=2.5)
    old_master = RA.solve_binary_master
    try:
        RA.solve_binary_master = lambda *a, **k: RA.MasterResult(
            "infeasible", None, None, None, False, True, "fake-milp", None)
        recovered = BP._solve_complete_universe_fullcover_target(
            archive=[c], all_tids=("A",), K=1, batteries=1, p=p,
            deadline=time.monotonic() + 2.0, deck_times=(), active_times=(),
            pooled_energy_cap=float(p.B_use), quick_min=1.0, swap_min=6.0,
            quick_capacity=1, swap_capacity=1)
    finally:
        RA.solve_binary_master = old_master
    assert recovered.coverage_incumbent == 1, recovered
    assert recovered.termination_reason == "target-feasible-witness", recovered
    assert "exact-fullcover-dfs-recovered" in str(recovered.direct_target_backend)

    # H-91: an interrupted independent exact-master verifier is UNKNOWN, never NO.
    old_master = RA.solve_binary_master
    old_exact_master = BP._exact_fullcover_master_feasibility
    try:
        RA.solve_binary_master = lambda *a, **k: RA.MasterResult(
            "infeasible", None, None, None, False, True, "fake-milp", None)
        BP._exact_fullcover_master_feasibility = lambda **k: (
            "UNKNOWN_TIMEOUT", tuple(), 7)
        unresolved = BP._solve_complete_universe_fullcover_target(
            archive=[c], all_tids=("A",), K=1, batteries=1, p=p,
            deadline=time.monotonic() + 2.0, deck_times=(), active_times=(),
            pooled_energy_cap=float(p.B_use), quick_min=1.0, swap_min=6.0,
            quick_capacity=1, swap_capacity=1)
    finally:
        RA.solve_binary_master = old_master
        BP._exact_fullcover_master_feasibility = old_exact_master
    assert unresolved.open_nodes == 1, unresolved
    assert unresolved.pricing_bound_available is False, unresolved
    assert unresolved.branching_complete is False, unresolved
    assert unresolved.termination_reason == (
        "fullcover-target-exact-master-verification-time-limit"), unresolved

    # H-83: old result contracts are diagnostics only, even if all other fields
    # are maliciously made to look current.
    args = SimpleNamespace(
        _resolved_xi_mmsi="219018788", validation_samples=None,
        xi_train_samples=None, final_test_samples=None)
    old_df = pd.DataFrame([dict(
        study_mode="formal",
        result_contract="fleet-anytime-result-v11-complete-universe-target-closure",
        physical_numeric_contract=RM.FORMAL_PHYSICAL_NUMERIC_CONTRACT,
        route_identity_contract=BP.ROUTE_IDENTITY_CONTRACT,
        model_semantics_contract=BP.MODEL_SEMANTICS_CONTRACT,
        formal_experiment_scheduler_contract=S13.FORMAL_EXPERIMENT_SCHEDULER_CONTRACT,
        resume_input_sha256="current-instance", xi_mmsi="219018788",
        validation_samples_hash="none", xi_train_samples_hash="none",
        final_test_samples_hash="none")])
    try:
        S13._validate_e1_frontier_for_target_refine(old_df, args, "current-instance")
    except SystemExit:
        pass
    else:
        raise AssertionError("old v11 frontier was accepted as current formal proof")

    # H-84: exact frozen-plan identity must distinguish adjacent binary64 values.
    base = dict(tau=0.0, h=5.0, tids=("A",), ordered_tids=("A",),
                E_plan_Wh=100.0, E_soc_required_Wh=100.0,
                resource_intervals={"deck": (0.0, 1.0), "active": (0.0, 5.0)},
                uav_id=0, battery_group=0, turnaround_before=None,
                post_service_mode="none_after_last_mission", post_service_interval=None)
    alt = dict(base)
    alt["E_plan_Wh"] = math.nextafter(100.0, math.inf)
    alt["E_soc_required_Wh"] = math.nextafter(100.0, math.inf)
    assert float(base["E_plan_Wh"]).hex() != float(alt["E_plan_Wh"]).hex()
    assert S13._frozen_plan_fingerprint([base]) != S13._frozen_plan_fingerprint([alt])

    # H-85: manual formal configuration is diagnostic and cannot authorize a
    # confirmatory final test; a structured auto freeze can.
    manual = SimpleNamespace(uav="S", k=1, batteries=1, study_mode="formal")
    S13._resolve_e2_config(manual)
    assert not S13._formal_e2_final_test_authorized(manual)
    frozen = SimpleNamespace(study_mode="formal", _e1_formal_freeze_verified=True,
                             _e1_formal_freeze_sha256="a" * 64,
                             _e2_matrix_completion_verified=True,
                             _formal_sample_hashes_verified=True,
                             allow_incomplete_results=False, allow_final_test_rerun=False)
    assert S13._formal_e2_final_test_authorized(frozen)

    # Same-version but different-instance E1 selection is also rejected.
    formal_sel = pd.DataFrame([dict(
        uav="S", knee_K=1, knee_B=1, knee_safe_per_inventory_kWh=1.0,
        knee_energy_per_safe=1.0, knee_plan_holds=True, selection_status="eligible",
        knee_resource_minimality_certified=True, knee_global_certificate_available=True,
        knee_coverage_certificate_available=True, resource_threshold_point_valid=True,
        degenerate_knee=False,
        formal_selection_contract="discrete-coverage-threshold-min-resource-plus-lex-v3",
        source_result_contract=S13.RESULT_CONTRACT,
        source_formal_experiment_scheduler_contract=S13.FORMAL_EXPERIMENT_SCHEDULER_CONTRACT,
        source_result_certificate_contract=BP.RESULT_CERTIFICATE_CONTRACT,
        source_formal_proof_contract=BP.FORMAL_PROOF_CONTRACT,
        source_resume_input_sha256="old-instance")])
    mismatch = SimpleNamespace(
        uav="auto", k=None, batteries=None, study_mode="formal",
        _e1_sel_df=formal_sel, selection_metric="safe_per_inventory_kWh",
        _expected_e1_resume_input_sha256="new-instance")
    try:
        S13._resolve_e2_config(mismatch)
    except SystemExit:
        pass
    else:
        raise AssertionError("same-version E1 selection from another instance was frozen")

    # H-86: formal threshold is literal ceil(binary64 rho * P), no epsilon nudge.
    rows = []
    for K in (1, 2, 3):
        for B in range(9):
            cap = {1: 3, 2: 6, 3: 8}[K]
            cov = min(B, cap)
            rows.append(dict(
                uav="Q", K=K, batteries=B, safe_served=cov,
                per_battery=(None if B == 0 else cov / B),
                coverage_incumbent=cov, coverage_upper_bound=8,
                covered=cov, coverable_note=8, plan_holds=False,
                global_certificate_available=False,
                global_route_space_certificate=False, implicit_route_space_certified=False,
                coverage_global_certificate_available=(K == 3 and B == 8),
                study_mode="formal", max_stops_requested=4, stops_cap_spec="4",
                max_stops_effective=4, stops_cap=4, max_stops_observed=1,
                stops_cap_hit=False))
    rho = math.nextafter(0.5, 1.0)
    sel = S13.e1_select_from_df(pd.DataFrame(rows), frac=rho, order="BK", patience=2)
    assert int(sel.iloc[0]["coverage_threshold"]) == 5, sel

    # H-89: THM-FCT proof-code concordance must bind the independent exact
    # master verifier; otherwise the v5 theorem metadata omits a proof-critical step.
    fct = dict(BP.FORMAL_PROOF_CODE_ANCHORS)["THM-FCT"]
    assert "_exact_fullcover_master_feasibility" in fct, fct

    # H-88: gate-critical UCBs retain full binary64 precision; display rounding
    # must never turn a value just above the allocation budget into a pass.
    ucb = math.nextafter(0.045, math.inf)
    kept = S13._formal_statistic_value(ucb)
    assert kept == ucb and kept > 0.045
    assert round(kept, 6) == 0.045  # demonstrates the historical false-pass boundary

    # H-87: purge is a strict formal inequality; 10-5e-10 min does not satisfy
    # a 10-minute horizon merely because it is within a numerical tolerance.
    def _part(sample_id, t0):
        return pd.DataFrame([dict(sample_id=sample_id, source_track_id="ship",
                                  t0_epoch=float(t0), h_min=10.0,
                                  t1_epoch=float(t0) + 600.0)])
    try:
        RP.validate_holdout_disjointness(
            _part("tr", 0.0), _part("va", 2000.0), _part("te", 4000.0),
            purge_min=10.0 - 5e-10)
    except ValueError as exc:
        assert "purge_min" in str(exc)
    else:
        raise AssertionError("sub-horizon purge passed via positive tolerance")


def _drv_v14_protocol():
    """Independent multi-role audit regressions added after the v13 re-review."""
    # H-92: harshest-q is an exact predeclared binary64 cell, never an isclose band.
    base = dict(
        run_status="ok", holds=True, n_missing_replay=0, study_mode="formal",
        e1_formal_freeze_verified=True, e1_formal_freeze_sha256="a" * 64,
        coverage_optimal=True, energy_optimal=True, lexicographic_optimal=True,
        global_certificate_available=True, coverage_global_certificate_available=True,
        algorithmic_global_certificate=True, physical_model_global_certificate=True,
        route_universe_provenance_certified=True, flights=1, energy_Wh=1.0,
        energy_per_safe=1.0, plan_fingerprint="b" * 64, emp_viol_upper95=0.01)
    low = dict(base, criterion="lowq", q=0.799995, safe_served=8, covered=8)
    exact = dict(base, criterion="exactq", q=0.8, safe_served=7, covered=7)
    picked = S13._select_e2_validation_candidate(
        pd.DataFrame([low, exact]), [0.799995, 0.8])
    assert picked is not None and str(picked["criterion"]) == "exactq", picked
    assert S13._binary64_equal(float(picked["q"]), 0.8)
    assert not S13._binary64_equal(0.799995, 0.8)

    # H-93/H-94: diagnostic escape hatches are impossible in formal mode.
    for kw, marker in ((dict(allow_incomplete_results=True, allow_final_test_rerun=False),
                         "allow-incomplete-results"),
                        (dict(allow_incomplete_results=False, allow_final_test_rerun=True),
                         "allow-final-test-rerun")):
        a = SimpleNamespace(study_mode="formal", **kw)
        try:
            S13._validate_formal_protocol_overrides(a)
        except SystemExit as exc:
            assert marker in str(exc)
        else:
            raise AssertionError(f"formal diagnostic override escaped: {marker}")
    S13._validate_formal_protocol_overrides(SimpleNamespace(
        study_mode="mechanism", allow_incomplete_results=True, allow_final_test_rerun=True))

    # H-95: even a valid E1 freeze cannot consume test before the full E2 matrix closes.
    frozen = SimpleNamespace(study_mode="formal", _e1_formal_freeze_verified=True,
                             _e1_formal_freeze_sha256="c" * 64,
                             _e2_matrix_completion_verified=False,
                             _formal_sample_hashes_verified=True,
                             allow_incomplete_results=False, allow_final_test_rerun=False)
    assert not S13._formal_e2_final_test_authorized(frozen)
    frozen._e2_matrix_completion_verified = True
    assert S13._formal_e2_final_test_authorized(frozen)

    # H-96: the pre-freeze test loader must never materialize Xi/weather/recovery outcomes.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "test.csv"
        row = dict(
            mmsi="219018788", h_min=5, c_state="slow", t0_epoch=1000.0,
            t1_epoch=1300.0, source_track_id="track-sha", sample_id="te-1",
            predictor="cv_noleak", predictor_contract=M.XI_PREDICTOR_CONTRACTS["cv_noleak"],
            timestamp_epoch_contract=M.XI_TIMESTAMP_EPOCH_CONTRACT,
            sample_overlap_policy="nonoverlap", purge_min=30.0, moments_source="train",
            valid_for_formal=True, split="test",
            xi_e_m="SECRET_XI_E", xi_n_m="SECRET_XI_N",
            wind_error_e_ms="SECRET_WE", wind_error_n_ms="SECRET_WN", hs_error_m="SECRET_HS",
            weather_predictor="weather_speed_primary_coherent_noleak",
            weather_predictor_contract=RM.WEATHER_PREDICTOR_CONTRACTS["weather_speed_primary_coherent_noleak"],
            weather_timestamp_epoch_contract=RM.WEATHER_TIMESTAMP_EPOCH_CONTRACT,
            weather_truth_contract=RM.WEATHER_TRUTH_CONTRACT,
            weather_data_contract=RM.WEATHER_FORMAL_DATA_CONTRACT,
            weather_valid_for_formal=True, weather_source_sha256="d" * 64,
            weather_train_source_sha256="e" * 64,
            actual_recovery_state="SECRET_STATE")
        pd.DataFrame([row]).to_csv(fp, index=False)
        md = S13._load_final_test_metadata_pre_freeze(fp, mmsi="219018788")
        assert len(md) == 1, md
        forbidden = {"xi_e_m", "xi_n_m", "wind_error_e_ms", "wind_error_n_ms",
                     "hs_error_m", "actual_recovery_state"}
        assert forbidden.isdisjoint(md.columns), md.columns
        assert forbidden.issubset(set(md.attrs.get("available_columns", ())))
        assert md.attrs.get("outcomes_materialized") is False
        tr = pd.DataFrame([dict(source_track_id="track-sha", sample_id="tr-1",
                                    t0_epoch=0.0, t1_epoch=300.0, h_min=5.0)])
        va = pd.DataFrame([dict(source_track_id="track-sha", sample_id="va-1",
                                    t0_epoch=1000.0, t1_epoch=1300.0, h_min=5.0,
                                    wind_error_e_ms=0.0, wind_error_n_ms=0.0, hs_error_m=0.0,
                                    actual_recovery_state="slow")])
        md["t0_epoch"] = 2000.0; md["t1_epoch"] = 2300.0
        RP.validate_holdout_disjointness(
            tr, va, md, purge_min=5.0,
            require_real_weather=True, require_real_recovery_state=True)

    # H-97: q-grid inputs themselves are finite, bounded and exact-unique.
    assert S13._parse_e2_quantiles("0.2,0.5,0.8") == (0.2, 0.5, 0.8)
    for bad in ("", "0.2,nan,0.8", "0.2,1.1", "0.8,0.8"):
        try:
            S13._parse_e2_quantiles(bad)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"invalid E2 quantile grid accepted: {bad!r}")

    # H-98: post-validation file replacement invalidates the hash-bound no-leak proof.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        trf, vaf, tef = (td / "train.csv", td / "validation.csv", td / "test.csv")
        for fp, txt in ((trf, "train"), (vaf, "validation"), (tef, "test")):
            fp.write_text(txt, encoding="utf-8")
        a = SimpleNamespace(study_mode="formal", xi_train_samples=trf,
                            validation_samples=vaf, final_test_samples=tef)
        S13._bind_formal_sample_hashes(a)
        S13._verify_formal_sample_hashes_unchanged(a)
        tef.write_text("test-tampered", encoding="utf-8")
        try:
            S13._verify_formal_sample_hashes_unchanged(a)
        except SystemExit as exc:
            assert "changed" in str(exc)
        else:
            raise AssertionError("tampered final-test bytes retained old independence authorization")



def _drv_v15_resource_closure():
    """v15 persistent full-cover resource-closure exactness regressions."""
    import tempfile

    # Historical v15 behavior remains a regression obligation under newer
    # proof/result contracts; do not require the current contract string to say v15.
    assert hasattr(BP, "_exact_battery_binpack_status")
    assert hasattr(BP, "_minimal_battery_conflict_core")
    assert "v17" in S13.RESULT_CONTRACT
    fct = dict(BP.FORMAL_PROOF_CODE_ANCHORS)["THM-FCT"]
    for name in ("_exact_battery_binpack_status", "_minimal_battery_conflict_core",
                 "_load_fullcover_closure_checkpoint",
                 "_save_fullcover_closure_checkpoint"):
        assert name in fct, (name, fct)
    import inspect
    e1src = inspect.getsource(S13.E1_knee_refine)
    assert "target_closure_checkpoint_path=" in e1src
    assert 'target_closure_resume=(str(getattr(args, "resume", "on")).lower() == "on")' in e1src

    def _col(tid, tau, energy=4.0, launch_start=None, clear_end=None):
        tau = float(tau)
        a = tau if launch_start is None else float(launch_start)
        b = tau + 1.0 if clear_end is None else float(clear_end)
        return dict(
            ordered_tids=(tid,), tids=(tid,), tau=tau, h=1.0,
            E_plan_Wh=float(energy), E_soc_required_Wh=float(energy),
            resource_intervals=dict(
                deck=(), active=(a, b), launch_start_min=a, launch_min=tau,
                recovery_min=tau + 1.0, clear_end_min=b))

    p = SimpleNamespace(B_use=10.0)

    # H-99: reusable proof state is bound to immutable binary64 route/resource semantics.
    a0 = [_col("A", 0.0, 4.0), _col("B", 2.0, 4.0)]
    a1 = [_col("A", 0.0, math.nextafter(4.0, math.inf)), _col("B", 2.0, 4.0)]
    c0 = BP._fullcover_closure_context_sha256(
        a0, ("A", "B"), 1, 2, p, 20.0, 1.0, 1.0, 1, 1)
    c1 = BP._fullcover_closure_context_sha256(
        a1, ("A", "B"), 1, 2, p, 20.0, 1.0, 1.0, 1, 1)
    assert c0 != c1
    ca = BP._fullcover_closure_context_sha256(
        a0, ("A", "B"), 1, 2, p, 20.0, 1.0, 1.0, 1, 1,
        algorithm_sha256="a" * 64)
    cb = BP._fullcover_closure_context_sha256(
        a0, ("A", "B"), 1, 2, p, 20.0, 1.0, 1.0, 1, 1,
        algorithm_sha256="b" * 64)
    assert ca != cb

    # H-100/H-101: checkpoint contract+payload hash+context are fail-closed.
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "closure.json"
        BP._save_fullcover_closure_checkpoint(
            fp, context_sha256=c0,
            cuts=[((0, 1), "full-pattern-resource-dfs")])
        loaded = BP._load_fullcover_closure_checkpoint(
            fp, context_sha256=c0, archive_len=2, resume=True)
        assert loaded == [((0, 1), "full-pattern-resource-dfs")], loaded
        try:
            BP._load_fullcover_closure_checkpoint(
                fp, context_sha256=c1, archive_len=2, resume=True)
        except RuntimeError as exc:
            assert "context mismatch" in str(exc)
        else:
            raise AssertionError("cross-instance target closure checkpoint was accepted")
        raw = json.loads(fp.read_text(encoding="utf-8"))
        raw["cuts"][0]["kind"] = "battery-binpack-core"
        fp.write_text(json.dumps(raw), encoding="utf-8")
        try:
            BP._load_fullcover_closure_checkpoint(
                fp, context_sha256=c0, archive_len=2, resume=True)
        except RuntimeError as exc:
            assert "payload hash mismatch" in str(exc)
        else:
            raise AssertionError("tampered closure checkpoint retained authorization")

    # H-102: exact battery packing catches conflicts invisible to pooled energy.
    bcols = [dict(E_soc_required_Wh=6.0) for _ in range(3)]
    st, _nodes = BP._exact_battery_binpack_status(
        bcols, (0, 1, 2), 2, 10.0, None)
    assert st == "INFEASIBLE_PROVEN"
    st, core, _nodes = BP._minimal_battery_conflict_core(
        bcols, (0, 1, 2), 2, 10.0, None)
    assert st == "INFEASIBLE_PROVEN" and core == (0, 1, 2)
    fcols = [dict(E_soc_required_Wh=v) for v in (6.0, 4.0, 6.0)]
    st, _nodes = BP._exact_battery_binpack_status(
        fcols, (0, 1, 2), 2, 10.0, None)
    assert st == "FEASIBLE"

    # H-103/H-104: direct full-cover master inherits min(K,B) active capacity
    # and fastest-turnaround UAV interval capacity from the generic resource model.
    rows_cols = [
        _col("A", 1.0, launch_start=0.0, clear_end=2.0),
        _col("B", 3.0, launch_start=1.5, clear_end=3.0),
    ]
    _d, caps = BP._fullcover_target_master_rows(
        rows_cols, ("A", "B"), (), (0.0, 1.5), 2, 1, 1.0, 2.0)
    assert any(set(row) == {0, 1} and cap == 1 for row, cap in caps), caps
    _d, caps = BP._fullcover_target_master_rows(
        rows_cols, ("A", "B"), (), (), 1, 2, 1.0, 2.0)
    assert any(set(row) == {0, 1} and cap == 1 for row, cap in caps), caps

    # H-105/H-106: one exact resource-infeasible full-cover pattern is persisted;
    # a same-context resume may close the target without repeating that resource DFS.
    orig_master = RA.solve_binary_master
    orig_audit = BP._audit_integer_selection
    try:
        calls = {"n": 0}
        def fake_master(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return RA.MasterResult(
                    "optimal", np.array([1.0, 1.0]), 8.0, 8.0,
                    True, False, "fake-v15", 0.0)
            return RA.MasterResult(
                "infeasible", None, None, None,
                False, True, "fake-v15", None)
        RA.solve_binary_master = fake_master
        BP._audit_integer_selection = lambda *a, **k: RA.ResourceAuditResult(
            RA.ResourceAuditStatus.INFEASIBLE_PROVEN, None, 7)
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td) / "target.json"
            r1 = BP._solve_complete_universe_fullcover_target(
                archive=a0, all_tids=("A", "B"), K=1, batteries=2, p=p,
                deadline=None, deck_times=(), active_times=(),
                pooled_energy_cap=20.0, quick_min=1.0, swap_min=1.0,
                quick_capacity=1, swap_capacity=1,
                target_closure_checkpoint_path=fp,
                target_closure_resume=False)
            assert r1.open_nodes == 0 and r1.fullcover_strong_cuts == 1
            assert r1.resource_audit_nodes == 7 and fp.is_file()
            persisted = json.loads(fp.read_text(encoding="utf-8"))
            assert persisted["cuts"] == [
                {"indices": [0, 1], "kind": "full-pattern-resource-dfs"}], persisted
            calls["n"] = 100
            r2 = BP._solve_complete_universe_fullcover_target(
                archive=a0, all_tids=("A", "B"), K=1, batteries=2, p=p,
                deadline=None, deck_times=(), active_times=(),
                pooled_energy_cap=20.0, quick_min=1.0, swap_min=1.0,
                quick_capacity=1, swap_capacity=1,
                target_closure_checkpoint_path=fp,
                target_closure_resume=True)
            assert r2.open_nodes == 0 and r2.fullcover_cuts_loaded == 1
            assert r2.resource_audit_calls == 0, r2
    finally:
        RA.solve_binary_master = orig_master
        BP._audit_integer_selection = orig_audit

    # H-107: UNKNOWN in the new battery relaxation is fail-closed and never cut.
    orig_master = RA.solve_binary_master
    orig_core = BP._minimal_battery_conflict_core
    try:
        RA.solve_binary_master = lambda *a, **k: RA.MasterResult(
            "optimal", np.array([1.0, 1.0]), 8.0, 8.0,
            True, False, "fake-v15", 0.0)
        BP._minimal_battery_conflict_core = lambda *a, **k: (
            "UNKNOWN_TIMEOUT", tuple(), 3)
        r = BP._solve_complete_universe_fullcover_target(
            archive=a0, all_tids=("A", "B"), K=1, batteries=2, p=p,
            deadline=None, deck_times=(), active_times=(),
            pooled_energy_cap=20.0, quick_min=1.0, swap_min=1.0,
            quick_capacity=1, swap_capacity=1)
        assert r.termination_reason == "battery-relaxation-time-limit"
        assert r.open_nodes == 1 and not r.resource_audit_complete
        assert r.fullcover_strong_cuts == 0
    finally:
        RA.solve_binary_master = orig_master
        BP._minimal_battery_conflict_core = orig_core

    # H-108: --resume off resets a stale same-path ledger before any witness exits.
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "fresh.json"
        BP._save_fullcover_closure_checkpoint(
            fp, context_sha256=c0,
            cuts=[((0, 1), "full-pattern-resource-dfs")])
        aud = RA.ResourceAuditResult(RA.ResourceAuditStatus.FEASIBLE, {}, 1)
        r = BP._solve_complete_universe_fullcover_target(
            archive=a0, all_tids=("A", "B"), K=1, batteries=2, p=p,
            deadline=None, deck_times=(), active_times=(),
            pooled_energy_cap=20.0, quick_min=1.0, swap_min=1.0,
            quick_capacity=1, swap_capacity=1,
            initial_selection=(0, 1), initial_audit=aud,
            target_closure_checkpoint_path=fp, target_closure_resume=False)
        assert r.incumbent_selection == (0, 1)
        payload = json.loads(fp.read_text(encoding="utf-8"))
        assert payload.get("cuts") == [], payload



def _drv_v16_global_battery_relaxation():
    """v16 global full-cover battery-relaxation proof regressions."""
    assert "global-battery-relaxation" in BP.RESULT_CERTIFICATE_CONTRACT
    assert "global-battery-relaxation" in BP.FORMAL_PROOF_CONTRACT
    assert "v17" in S13.RESULT_CONTRACT
    assert "global-battery" in S13.FORMAL_EXPERIMENT_SCHEDULER_CONTRACT
    anchors = dict(BP.FORMAL_PROOF_CODE_ANCHORS)
    assert "THM-GBR" in anchors
    import inspect
    e1src = inspect.getsource(S13.E1_knee_refine)
    for field in ("target_global_battery_relaxation_status",
                  "target_global_battery_min_required",
                  "target_global_battery_dp_states",
                  "target_global_battery_one_pack_masks"):
        assert field in e1src, field
    for name in ("_exact_global_fullcover_battery_relaxation",
                 "_solve_complete_universe_fullcover_target",
                 "_target_infeasibility_algorithmic_proven"):
        assert name in anchors["THM-GBR"], (name, anchors["THM-GBR"])

    def _col(tids, energy, *, plan_energy=None, tau=0.0):
        tids = tuple(tids) if not isinstance(tids, str) else (tids,)
        return dict(
            ordered_tids=tids, tids=tids, tau=float(tau), h=1.0,
            E_plan_Wh=float(energy if plan_energy is None else plan_energy),
            E_soc_required_Wh=float(energy),
            resource_intervals=dict(
                deck=(), active=(float(tau), float(tau) + 1.0),
                launch_start_min=float(tau), launch_min=float(tau),
                recovery_min=float(tau) + 1.0, clear_end_min=float(tau) + 1.0))

    tids8 = tuple("ABCDEFGH")
    # H-109: S/M-like case. Every singleton costs > C/2, so exact relaxed
    # minimum is eight batteries and B=7 is a rigorous NO.
    cols = [_col(t, 6.0, plan_energy=1.0, tau=2.0*i) for i, t in enumerate(tids8)]
    st, mb, states, bundles, wit = BP._exact_global_fullcover_battery_relaxation(
        cols, tids8, 7, 10.0, None)
    assert st == "INFEASIBLE_PROVEN" and mb == 8, (st, mb, states, bundles, wit)
    assert states > 0 and bundles >= 8 and len(wit) == 8

    # H-110: L-like structure. A is heavy and cannot pair; the other seven
    # singleton jobs pair two-per-battery. Multi-stop alternatives all share F,
    # so the relaxed exact minimum remains five batteries.
    lcols = [_col("A", 6.1)]
    lcols += [_col(t, 4.0) for t in "BCDEFGH"]
    lcols += [_col(("E", "F"), 6.1), _col(("F", "H"), 6.1)]
    st4, mb4, _n4, _b4, _w4 = BP._exact_global_fullcover_battery_relaxation(
        lcols, tids8, 4, 10.0, None)
    st5, mb5, _n5, _b5, _w5 = BP._exact_global_fullcover_battery_relaxation(
        lcols, tids8, 5, 10.0, None)
    assert st4 == "INFEASIBLE_PROVEN" and mb4 == 5
    assert st5 == "FEASIBLE_RELAXATION" and mb5 == 5

    # H-111: exact binary64 boundary, with no +epsilon widening.
    exact_pair = [_col("A", 5.0), _col("B", 5.0)]
    st, mb, *_ = BP._exact_global_fullcover_battery_relaxation(
        exact_pair, ("A", "B"), 1, 10.0, None)
    assert st == "FEASIBLE_RELAXATION" and mb == 1
    above_pair = [_col("A", 5.0), _col("B", math.nextafter(5.0, math.inf))]
    st, mb, *_ = BP._exact_global_fullcover_battery_relaxation(
        above_pair, ("A", "B"), 1, 10.0, None)
    assert st == "INFEASIBLE_PROVEN" and mb == 2

    # H-112: route masks must be an exact partition; overlapping cheap routes
    # may not be combined to fake full coverage.
    overlap = [_col(("A", "B"), 1.0), _col(("B", "C"), 1.0)]
    st, mb, *_ = BP._exact_global_fullcover_battery_relaxation(
        overlap, ("A", "B", "C"), 2, 10.0, None)
    assert st == "INFEASIBLE_PROVEN" and mb is None

    # H-113: timeout/unknown never signs a NO.
    st, mb, *_ = BP._exact_global_fullcover_battery_relaxation(
        cols, tids8, 7, 10.0, time.monotonic() - 1.0)
    assert st == "UNKNOWN_TIMEOUT" and mb is None
    p = SimpleNamespace(B_use=10.0)
    r_unknown = BP._solve_complete_universe_fullcover_target(
        archive=cols, all_tids=tids8, K=3, batteries=7, p=p,
        deadline=time.monotonic() - 1.0, deck_times=(), active_times=(),
        pooled_energy_cap=70.0, quick_min=1.0, swap_min=1.0,
        quick_capacity=1, swap_capacity=1)
    assert r_unknown.termination_reason == "global-battery-relaxation-time-limit"
    assert r_unknown.open_nodes == 1 and not r_unknown.pricing_bound_available
    assert not BP._target_infeasibility_algorithmic_proven(r_unknown, False)

    # H-114: a global relaxed NO bypasses the floating master entirely and
    # produces a fully closed algorithmic certificate state.
    orig_master = RA.solve_binary_master
    try:
        def _must_not_run(*a, **k):
            raise AssertionError("floating master called despite global battery NO")
        RA.solve_binary_master = _must_not_run
        p = SimpleNamespace(B_use=10.0)
        r = BP._solve_complete_universe_fullcover_target(
            archive=cols, all_tids=tids8, K=3, batteries=7, p=p,
            deadline=None, deck_times=(), active_times=(),
            pooled_energy_cap=70.0, quick_min=1.0, swap_min=1.0,
            quick_capacity=1, swap_capacity=1)
        assert r.termination_reason == (
            "fullcover-target-global-battery-relaxation-infeasible-proven")
        assert r.open_nodes == 0 and r.target_master_solves == 0
        assert r.direct_target_backend == "exact-global-battery-mask-dp"
        assert r.global_battery_min_required == 8
        assert r.global_battery_relaxation_status == "INFEASIBLE_PROVEN"
        assert BP._target_infeasibility_algorithmic_proven(r, False)
    finally:
        RA.solve_binary_master = orig_master

    # H-115: FEASIBLE_RELAXATION is only a lower-bound statement. It must
    # continue to the real master/resource chain and cannot become target YES.
    small = [_col("A", 4.0, tau=0.0), _col("B", 4.0, tau=2.0)]
    orig_master = RA.solve_binary_master
    orig_audit = BP._audit_integer_selection
    calls = {"master": 0}
    try:
        def _master(*a, **k):
            calls["master"] += 1
            return RA.MasterResult(
                "optimal", np.array([1.0, 1.0]), 8.0, 8.0,
                True, False, "fake-v16", 0.0)
        RA.solve_binary_master = _master
        BP._audit_integer_selection = lambda *a, **k: RA.ResourceAuditResult(
            RA.ResourceAuditStatus.UNKNOWN_TIMEOUT, None, 3)
        p = SimpleNamespace(B_use=10.0)
        r = BP._solve_complete_universe_fullcover_target(
            archive=small, all_tids=("A", "B"), K=1, batteries=1, p=p,
            deadline=None, deck_times=(), active_times=(),
            pooled_energy_cap=10.0, quick_min=1.0, swap_min=1.0,
            quick_capacity=1, swap_capacity=1)
        assert calls["master"] == 1
        assert r.global_battery_relaxation_status == "FEASIBLE_RELAXATION"
        assert r.global_battery_min_required == 1
        assert r.termination_reason == "resource-audit-time-limit"
        assert r.open_nodes == 1 and not BP._target_infeasibility_algorithmic_proven(r, False)
    finally:
        RA.solve_binary_master = orig_master
        BP._audit_integer_selection = orig_audit

    # H-116: independent tiny-instance exhaustive oracle.  This does not
    # reuse the production mask-DP recurrence: enumerate route subsets that are
    # exact covers, then solve each selected energy multiset by an independent
    # integer-capacity packing recursion.  It guards the theorem implementation,
    # not merely its hand-crafted examples.
    import itertools, random
    rng = random.Random(160816)
    def _brute_min_batteries(cols0, tids0, cap_int):
        tidpos = {t: i for i, t in enumerate(tids0)}
        recs = []
        for cc in cols0:
            mm = 0
            for tt in cc["ordered_tids"]:
                mm |= 1 << tidpos[tt]
            recs.append((mm, int(cc["E_soc_required_Wh"])))
        full0 = (1 << len(tids0)) - 1
        best = None
        for take_bits in range(1 << len(recs)):
            cov = 0
            es = []
            ok = True
            for j, (mm, ee) in enumerate(recs):
                if not ((take_bits >> j) & 1):
                    continue
                if cov & mm:
                    ok = False
                    break
                cov |= mm
                es.append(ee)
            if not ok or cov != full0:
                continue
            es.sort(reverse=True)
            bins = []
            local_best = [len(es) + 1]
            def pack(k):
                if len(bins) >= local_best[0]:
                    return
                if k == len(es):
                    local_best[0] = len(bins)
                    return
                e0 = es[k]
                seen_loads = set()
                for bi in range(len(bins)):
                    load = bins[bi]
                    if load in seen_loads:
                        continue
                    seen_loads.add(load)
                    if load + e0 <= cap_int:
                        bins[bi] += e0
                        pack(k + 1)
                        bins[bi] -= e0
                if e0 <= cap_int:
                    bins.append(e0)
                    pack(k + 1)
                    bins.pop()
            pack(0)
            if local_best[0] <= len(es):
                best = local_best[0] if best is None else min(best, local_best[0])
        return best

    for case in range(12):
        n0 = rng.randint(2, 5)
        tids0 = tuple(chr(ord("A") + i) for i in range(n0))
        # Always include singleton routes so an exact cover exists; add a few
        # random multi-stop alternatives. Energies/capacity are integers, hence
        # their binary64 representation is exact and the independent oracle is
        # comparing the same mathematical object.
        cols0 = [_col(t, rng.randint(2, 9)) for t in tids0]
        masks_seen = {1 << i for i in range(n0)}
        for _ in range(rng.randint(0, 4)):
            k0 = rng.randint(2, min(3, n0))
            idxs = tuple(sorted(rng.sample(range(n0), k0)))
            mm0 = sum(1 << i for i in idxs)
            if mm0 in masks_seen:
                continue
            masks_seen.add(mm0)
            cols0.append(_col(tuple(tids0[i] for i in idxs), rng.randint(2, 12)))
        cap0 = rng.randint(6, 14)
        brute = _brute_min_batteries(cols0, tids0, cap0)
        st0, mb0, *_ = BP._exact_global_fullcover_battery_relaxation(
            cols0, tids0, n0 + 1, float(cap0), None)
        if brute is None:
            assert st0 == "INFEASIBLE_PROVEN" and mb0 is None, (case, brute, st0, mb0)
        else:
            assert mb0 == brute and st0 == "FEASIBLE_RELAXATION", (
                case, brute, st0, mb0)

    # H-117: malformed route semantics fail closed.
    bad = [_col(("A", "A"), 1.0)]
    try:
        BP._exact_global_fullcover_battery_relaxation(
            bad, ("A",), 1, 10.0, None)
    except RuntimeError as exc:
        assert "non-elementary" in str(exc)
    else:
        raise AssertionError("global battery relaxation accepted duplicate turbine visit")

    # H-118: size guard is an accelerator guard, never a NO certificate.
    tids13 = tuple(f"T{i}" for i in range(13))
    big = [_col(t, 1.0) for t in tids13]
    st, mb, *_ = BP._exact_global_fullcover_battery_relaxation(
        big, tids13, 13, 10.0, None, max_turbines=12)
    assert st == "SKIPPED_SIZE" and mb is None

def suite_hardening():
    _drv_v13_hardening()
    _drv_v14_protocol()
    _drv_v15_resource_closure()
    _drv_v16_global_battery_relaxation()
    print("suite hardening: v12-v17 independent-audit/P0/resource-closure regressions PASS ✓")


SUITES["hardening"] = suite_hardening


_MUT_DRIVERS = dict(gate_worst=_drv_gate_worst, bind_cert=_drv_bind_cert,
                    bind_oracle=_drv_bind_oracle, occ_event=_drv_occ_event,
                    bias_reach=_drv_bias_reach, soc_dedup=_drv_soc_dedup,
                    pricing_id=_drv_pricing_id,
                    m01=_drv_m01, m02=_drv_m02, m03=_drv_m03, h03=_drv_h03,
                    p0_milp=_drv_p0_milp, p0_seed=_drv_p0_seed,
                    p0new_milp=_drv_p0new_milp, lex_wx=_drv_lex_wx,
                    dual_fail=_drv_dual_fail, xi_formal=_drv_xi_formal_input,
                    weather_formal=_drv_weather_formal,
                    terminal_scope=_drv_terminal_scope,
                    formal_exact=_drv_formal_exact_contract,
                    e1_staged=_drv_e1_staged_contract,
                    v9_protocol=_drv_v9_single_vessel_protocol,
                    v10_target=_drv_v10_target_knee_protocol,
                    v11_universe=_drv_v11_complete_universe,
                    v12_resource=_drv_v12_resource_closure,
                    v13_hardening=_drv_v13_hardening,
                    v14_protocol=_drv_v14_protocol,
                    v15_resource=_drv_v15_resource_closure,
                    v16_battery=_drv_v16_global_battery_relaxation)

# ---------- 行为级变异清单: (编号说明, [(文件, 原文, 变体), ...], 驱动) ----------
_S7, _S9, _S10, _S11, _S12, _S13, _S15 = ("step7_compute_xi.py", "step9_model.py",
                                          "step10_model_routing.py", "step11_algorithm_route_drcc.py",
                                          "step12_branch_price.py", "step13_experiment_model.py",
                                          "step15_replay.py")
_P_PAIR = ("            if deck_conf_adj:", "            if False and deck_conf_adj:")
_P_AUDIT = ("                elif j2 in deck_conf_adj.get(i, ()):",
            "                elif True or j2 in deck_conf_adj.get(i, ()):")
MUTATIONS = [
    ("MUT-01 C01 门场只查预测最近点(丢候选集)", [(_S10,
      '    cand_wx = gate_weather_candidates(route, wx_recovery)\n'
      '    eps_gate = float(getattr(p, "eps_gate", p.eps_cap))',
      '    cand_wx = [dict(gate_wx)]\n'
      '    eps_gate = float(getattr(p, "eps_gate", p.eps_cap))')], "gate_worst"),
    ("MUT-02 C01 门判据 ∀开 → ∃开", [(_S10,
      'gate_all_open = all(', 'gate_all_open = any(')], "gate_worst"),
    ("MUT-03 H01 删甲板逐对连续行", [(_S12, *_P_PAIR)], "bind_cert"),
    ("MUT-04 H01 删占机事件行", [(_S12,
      "            if occ_event_times:",
      "            if False and occ_event_times:")], "occ_event"),
    ("MUT-05 H01 删逐对冲突登记", [(_S12,
      "            _register_col_conflicts(len(cols) - 1)",
      "            pass  # mutant: 不登记")], "bind_cert"),
    ("MUT-06 H01 删逐对行+伪造重扫审计", [(_S12, *_P_PAIR), (_S12, *_P_AUDIT)],
     "bind_cert"),
    ("MUT-07 H01 删行+伪审计+伪物理复检(三重旁路)", [(_S12, *_P_PAIR),
     (_S12, *_P_AUDIT),
     (_S12, "        phys_ok, phys_reason = _plan_physical_check(chosen)",
      "        phys_ok, phys_reason = True, None")], "bind_oracle"),
    ("MUT-08 H02 tau_reach 带偏置不降级", [(_S11,
      '    if mode == "valid" and not _mrf:', "    if False:")], "bias_reach"),
    ("MUT-09 路线合同恢复按无序风机集的SOC不安全去重", [(_S11,
      '        out.append(c)\n    return out',
      '        out.append(c)\n'
      '    _best = {}\n'
      '    for _c in out:\n'
      '        _k = (round(float(_c.get("tau", 0.0)), 9), tuple(sorted(_c["tids"])), round(float(_c.get("h", 0.0)), 9))\n'
      '        if _k not in _best or float(_c["E_plan_Wh"]) < float(_best[_k]["E_plan_Wh"]):\n'
      '            _best[_k] = _c\n'
      '    return list(_best.values())  # mutant: 丢失访问顺序和SOC需求差异')], "soc_dedup"),
    ("MUT-10 H02 定价证明不登记", [(_S12,
      "                pricing_stats.append(_pst)     "
      "# 更新(H-02): 先登记后调用, 异常路径也留痕",
      "                pass  # mutant: 不登记")], "pricing_id"),
    ("MUT-11 M01 时间限旗标误报为节点限", [(_S12,
      '                hit_time_limit = True\n'
      '                status = "time-limit-no-certificate"',
      '                hit_node_limit = True\n'
      '                status = "node-limit-no-certificate"')], "m01"),
    ("MUT-12 M02 扩池路径伪造人工审计", [(_S12,
      "            _no_aud = dict(max_lp_artificial=None, max_incumbent_artificial=None,\n"
      "                           artificial_nodes_seen=0, incumbents_accepted=0,\n"
      "                           incumbents_rejected_artificial=0,\n"
      "                           all_accepted_incumbents_artificial_free=False,\n"
      "                           artificial_audit_complete=False)",
      "            _no_aud = dict(max_lp_artificial=0.0, max_incumbent_artificial=0.0,\n"
      "                           artificial_nodes_seen=0, incumbents_accepted=0,\n"
      "                           incumbents_rejected_artificial=0,\n"
      "                           all_accepted_incumbents_artificial_free=True,\n"
      "                           artificial_audit_complete=True,\n"
      "                           phase2_artificials_fixed_zero=True)")], "m02"),
    ("MUT-13 M03 忽略 RF 开关(退回 nb_p 口径)", [(_S12,
      "        no_rf_branching = bool((not enable_rf_branching) and nb_p == 0)",
      "        no_rf_branching = bool(nb_p == 0)")], "m03"),
    ("MUT-14 H03 联合预算去掉 eps_dock", [(_S10,
      '            "dock_reserve": float(getattr(p, "eps_dock", p.eps_cap)),',
      '            "dock_reserve": 0.0,  # mutant: 去 eps_dock')], "h03"),
    ("MUT-15 P0-1 恢复整数 LP 后依赖失败 MILP 并静默闭点", [(_S12,
      '            integral_sel = _integral_lp_selection(lp)\n'
      '            if integral_sel is not None:',
      '            integral_sel = _integral_lp_selection(lp)\n'
      '            if False and integral_sel is not None:'), (_S12,
      '            if not fr:\n'
      '                # 到达此处说明“非整数”判定与分支候选不一致；绝不能关闭节点。\n'
      '                milp_runtime["integral_lp_validation_failures"] += 1\n'
      '                no_certificate = True\n'
      '                all_nodes_converged = False\n'
      '                status = "fractionality-audit-fail-no-certificate"\n'
      '                stack.clear()\n'
      '                break',
      '            if not fr:\n'
      '                continue  # mutant: MILP 失败后静默关闭整数 LP 节点')], "p0_milp"),
    ("MUT-16 P0-2 信任外部 seed 的 E0", [(_S12,
      '                E_can = _plan_energy(dd)',
      '                E_can = float(c.get("E0", _plan_energy(dd)))  # mutant: 信任外部 E0')],
     "p0_seed"),
    ("MUT-17 P0-NEW 跳过 MILP 主问题行验证", [(_S12,
      '        if np.any(lhs > rhs + feasibility_tol):\n'
      '            return None, None, "constraint_violation"',
      '        if False and np.any(lhs > rhs + feasibility_tol):  # mutant\n'
      '            return None, None, "constraint_violation"')], "p0new_milp"),
    ("MUT-18 P2-01 lex Stage-2 回退全局天气(绕过列天气合同)", [(_S12,
      '        wx_col = _wx_of_route(r, wx)',
      '        wx_col = wx  # mutant: 回退全局天气'), (_S10,
      '        if hq == 0.0 and isinstance(getattr(self, "wx_tau", None), dict):\n'
      '            return dict(self.wx_tau)',
      '        if hq == 0.0 and fallback is not None:\n'
      '            return dict(fallback)  # mutant: 允许全局天气覆盖列天气')],
     "lex_wx"),
    ("MUT-19 P2-03 对偶提取失败静默归零", [
     (_S12,
      '    if checked is None:\n'
      '        return None, None, None\n'
      '    x, fun, _, marg, dual_lb = checked',
      '    if checked is None:\n'
      '        return 0.0, {t: 0.0 for t in turbine_ids}, 0.0  # mutant\n'
      '    x, fun, _, marg, dual_lb = checked'),
     (_S12,
      '    if checked is None:\n'
      '        return None, None, None, None, None\n'
      '    x, fun, imarg, emarg, dual_lb = checked',
      '    if checked is None:\n'
      '        return 0.0, {t: 0.0 for t in turbine_ids}, 0.0, 0.0, 0  # mutant\n'
      '    x, fun, imarg, emarg, dual_lb = checked')],
     "dual_fail"),
    ("MUT-20 Xi formal horizon 恢复 round/int 吸附", [(_S9,
      '            raw_h = [float(x) for x in num["h_min"].to_numpy(float)]',
      '            raw_h = [float(int(round(float(x)))) for x in num["h_min"].to_numpy(float)]  # mutant: off-grid snap')],
     "xi_formal"),
    ("MUT-21 Xi formal PSD 恢复尺度相对容差", [(_S9,
      '    return (Fraction.from_float(see_f) * Fraction.from_float(snn_f)\n'
      '            >= Fraction.from_float(sen_f) * Fraction.from_float(sen_f))',
      '    _eig = np.linalg.eigvalsh(np.array([[see_f, sen_f], [sen_f, snn_f]], float))\n'
      '    return float(_eig.min()) >= -1e-8 * max(1.0, abs(see_f), abs(sen_f), abs(snn_f))  # mutant')],
     "xi_formal"),
    ("MUT-22 Xi formal purge 恢复 +1e-12 容差", [(_S9,
      '            if not bool((purge_vals >= max_h_file).all()):',
      '            if not bool((purge_vals + 1e-12 >= max_h_file).all()):  # mutant')],
     "xi_formal"),
    ("MUT-23 Xi nonoverlap 恢复 +1e-9 时间扩域", [(_S7,
      '            if t0 >= last_end:',
      '            if t0 + 1e-9 >= last_end:  # mutant')],
     "xi_formal"),
    ("MUT-24 SAA 时间戳合同失配不再拒绝", [(_S10,
      '        if found != [M.XI_TIMESTAMP_EPOCH_CONTRACT]:\n            raise ValueError(',
      '        if False and found != [M.XI_TIMESTAMP_EPOCH_CONTRACT]:\n            raise ValueError(')],
     "xi_formal"),
    ("MUT-25 SAA horizon 恢复 round 吸附", [(_S10,
      '    h_raw = pd.to_numeric(full["h_min"], errors="coerce").to_numpy(float)',
      '    full["h_min"] = pd.to_numeric(full["h_min"], errors="coerce").round()  # mutant\n'
      '    h_raw = full["h_min"].to_numpy(float)')],
     "xi_formal"),
    ("MUT-26 mechanism 自动旧 SAA 重新 fail-fast", [(_S13,
      '        if str(getattr(args, "study_mode", "formal")) == "formal" or saa_explicit:',
      '        if True:  # mutant: auto stale SAA also aborts mechanism')],
     "xi_formal"),
    ("MUT-27 天气 no-leak predictor 偷看 t1 后记录", [(_S7,
      '    j = int(np.searchsorted(times, t0, side="right")) - 1',
      '    j = min(int(np.searchsorted(times, t0, side="right")), len(times)-1)  # mutant: future leakage')],
     "weather_formal"),
    ("MUT-28 weather horizon 恢复 round 吸附", [(_S10,
      '    raw_h = [float(x) for x in num["h_min"].to_numpy(float)]',
      '    raw_h = [float(round(float(x))) for x in num["h_min"].to_numpy(float)]  # mutant')],
     "weather_formal"),
    ("MUT-29 weather covariance 跳过 exact PSD", [(_S10,
      '    return bool(see >= 0 and snn >= 0 and see * snn - sen * sen >= 0)',
      '    return True  # mutant: accept indefinite weather covariance')],
     "weather_formal"),
    ("MUT-30 weather epoch 秒尺度错误缩小1000倍", [(_S7,
      '    sec = to_epoch_seconds_utc(ts)',
      '    sec = to_epoch_seconds_utc(ts) / 1000.0  # mutant: emulate pandas us/ns storage-unit bug')],
     "weather_formal"),
    ("MUT-31 weather moments 取消全局 nonoverlap", [(_S7,
      '            if t0 < last_t1:',
      '            if False and t0 < last_t1:  # mutant: allow overlapping weather-moment intervals')],
     "weather_formal"),
    ("MUT-32 回收风险重新引入无实测 acquisition 事件", [(_S10,
      '    out = {\n        "energy": float(p.eps_E),\n        "time": float(p.eps_T),\n    }',
      '    out = {\n        "energy": float(p.eps_E),\n        "time": float(p.eps_T),\n        "acquisition": 0.005,  # mutant: reintroduce unsupported terminal sensor event\n    }')],
     "terminal_scope"),
    ("MUT-33 formal 空速比较重新加入正 tolerance", [(_S10,
      '    return bool(math.isfinite(value) and math.isfinite(limit) and value <= limit)',
      '    return bool(math.isfinite(value) and math.isfinite(limit) and value <= limit + 1e-9)  # mutant')],
     "formal_exact"),
    ("MUT-34 resource UNKNOWN_TIMEOUT 错当 INFEASIBLE_PROVEN", [(_S11,
      '    if deadline_reached(deadline):\n        return ResourceAuditResult(ResourceAuditStatus.UNKNOWN_TIMEOUT, explored_nodes=0)',
      '    if deadline_reached(deadline):\n        return ResourceAuditResult(ResourceAuditStatus.INFEASIBLE_PROVEN, explored_nodes=0)  # mutant')],
     "formal_exact"),
    ("MUT-35 incomplete pricing 错当 closed", [(_S12,
      '        if not self.complete or not self.bound_available or self.reduced_value_bound is None:\n            return False',
      '        if not self.bound_available or self.reduced_value_bound is None:  # mutant: ignore completeness\n            return False')],
     "formal_exact"),
    ("MUT-36 exact-pattern future column 系数改为 0", [(_S12,
      '        return 1.0 if _exact_route_signature(column) in key else -1.0',
      '        return 1.0 if _exact_route_signature(column) in key else 0.0  # mutant')],
     "formal_exact"),
    ("MUT-37 pricing 遗漏 forbidden-service branch filter", [(_S12,
      '    if tids & branch.forbidden_turbines:\n        return False',
      '    if False and tids & branch.forbidden_turbines:  # mutant\n        return False')],
     "formal_exact"),
    ("MUT-38 reduced-cost inequality dual 符号翻转", [(_S12,
      '        # contribution is -(d*coeff)\n        clo, chi = -phi, -plo',
      '        # mutant: wrong sign\n        clo, chi = plo, phi')],
     "formal_exact"),
    ("MUT-39 Phase-I 去掉遗漏列 M_n*delta 修正", [(_S12,
      '    correction = 0.0 if math.isinf(delta) and delta > 0 else min(0.0, delta)\n    try:\n        if correction == 0.0:\n            return math.nextafter(lb, -math.inf)',
      '    correction = 0.0  # mutant: omit omitted-column correction\n    try:\n        if correction == 0.0:\n            return math.nextafter(lb, -math.inf)')],
     "formal_exact"),
    ("MUT-40 resource SOC 恢复 float tolerance", [(_S11,
      '        if e_need > usable_battery_energy_exact:\n            failed_state_cache.add(state_key)\n            return False',
      '        if float(e_need) > float(usable_battery_energy_exact) + 1e-6:  # mutant\n            failed_state_cache.add(state_key)\n            return False'), (_S11,
      '                if battery_used[b] + e_need > usable_battery_energy_exact:\n                    continue',
      '                if float(battery_used[b] + e_need) > float(usable_battery_energy_exact) + 1e-6:  # mutant\n                    continue')],
     "formal_exact"),
    ("MUT-41 formal Xi raw PSD 不再拒绝而投影", [(_S9,
      '        if not _binary64_psd_cov2(float(Sigma[0, 0]), float(Sigma[0, 1]), float(Sigma[1, 1])):\n            raise ValueError(f"Xi cell {key!r} 的 covariance 不是 binary64-as-real PSD。")',
      '        if not _binary64_psd_cov2(float(Sigma[0, 0]), float(Sigma[0, 1]), float(Sigma[1, 1])):\n            Sigma = _canonicalize_binary64_psd_cov2(Sigma)  # mutant: project raw formal input')],
     "formal_exact"),
    ("MUT-42 formal public API 重新允许 synthetic route universe", [(_S12,
      '    if synthetic_fixture and not bool(_internal_synthetic_route_universe):',
      '    if False and synthetic_fixture and not bool(_internal_synthetic_route_universe):  # mutant')],
     "formal_exact"),
    ("MUT-43 physical global certificate guard 忽略 route-universe provenance", [(_S12,
      '        algorithmic_global_certificate\n        and route_universe_provenance_certified\n        and mode == "exact-branch-price-cut"',
      '        algorithmic_global_certificate\n        and True  # mutant: ignore route-universe provenance\n        and mode == "exact-branch-price-cut"')],
     "formal_exact"),
    ("MUT-44 same-signature 不同 formal semantics 不再 fail-closed", [(_S12,
      '            if _column_semantics_fp(old) != _column_semantics_fp(c):\n                raise RuntimeError(\n                    "same canonical route signature has different formal semantics")',
      '            if False and _column_semantics_fp(old) != _column_semantics_fp(c):  # mutant\n                raise RuntimeError(\n                    "same canonical route signature has different formal semantics")')],
     "formal_exact"),
    ("MUT-45 unknown future master row 被静默当非负", [(_S12,
      '        if kind not in ranges:\n            raise ValueError(\n                f"no certified future-column coefficient range for inequality row {kind!r}")\n        return ranges[kind]',
      '        if kind not in ranges:\n            return (0.0, 1.0)  # mutant: silently assume nonnegative\n        return ranges[kind]')],
     "formal_exact"),
    ("MUT-46 required arc 被错误变成 route filter", [(_S12,
      '    if frozenset(arcs) & branch.forbidden_arcs:\n        return False\n    sig = _exact_route_signature(column)',
      '    if frozenset(arcs) & branch.forbidden_arcs:\n        return False\n    if branch.required_arcs and not branch.required_arcs.issubset(frozenset(arcs)):\n        return False  # mutant: required aggregate equality used as route filter\n    sig = _exact_route_signature(column)')],
     "formal_exact"),
    ("MUT-47 node route-mass bound 忽略 forbidden service", [(_S12,
      'def _node_allowed_turbine_bound(all_tids, branch):\n    return len(set(all_tids) - set(branch.forbidden_turbines))',
      'def _node_allowed_turbine_bound(all_tids, branch):\n    return len(set(all_tids))  # mutant: lose node-level M_n tightening')],
     "formal_exact"),
    ("MUT-48 physical certificate guard 忽略 proof-contract concordance", [(_S12,
      '        and binary64_model_contract_enforced\n        and formal_proof_contract_enforced)',
      '        and binary64_model_contract_enforced\n        and True)  # mutant: ignore proof contract')],
     "formal_exact"),
    ("MUT-49 physical certificate guard 忽略 future-row range contract", [(_S12,
      '        and route_semantics_invariance_certified\n        and future_column_row_ranges_certified\n        and binary64_model_contract_enforced',
      '        and route_semantics_invariance_certified\n        and True  # mutant: ignore row-range proof\n        and binary64_model_contract_enforced')],
     "formal_exact"),
    ("MUT-50 formal train-sample ambiguity 丢失 predictor provenance", [(_S15,
      '    obj.predictor = str(meta.get("predictor", _single_text_value(d, "predictor")))',
      '    obj.predictor = "unknown"  # mutant: drop already validated provenance')],
     "xi_formal"),
    ("MUT-51 formal real-weather replay 提前拆格丢失 WeatherAmbiguity provenance", [(_S13,
      '                               weather_unc=wamb, weather_dist="t3",',
      '                               weather_unc=wu_cell, weather_dist="t3",  # mutant: lose formal envelope')],
     "formal_exact"),
    ("MUT-52 coverage-only 错把 Stage-1 certificate 升级为 lexicographic global", [(_S12,
      '    algorithmic_global_certificate = bool(lex_opt)',
      '    algorithmic_global_certificate = bool(coverage_optimal)  # mutant: promote Stage-1 only')],
     "e1_staged"),
    ("MUT-53 formal E1 plateau 无视 coverage-bound sandwich equality", [(_S13,
      '                    if start_iv is not None and start_iv[0] == end_iv[1]:',
      '                    if start_iv is not None:  # mutant: certify despite loose endpoint UB')],
     "e1_staged"),
    ("MUT-54 formal E2 final-test 接受未认证 time-limit incumbent", [(_S13,
      '        ok = ok & cert.astype(bool)',
      '        ok = ok  # mutant: ignore complete lexicographic certificate')],
     "e1_staged"),
    ("MUT-55 formal fixed-B grid 仍允许反复长预算 plateau refinement", [(_S13,
      '    return str(e1_b_auto).lower() == "on"',
      '    return True  # mutant: ignore explicit fixed-grid request')],
     "e1_staged"),
    ("MUT-56 coverage certification 无 bound 信息增益仍被当作进展", [(_S13,
      '    return bool(\n        before is not None and after is not None\n        and (int(after[0]) > int(before[0]) or int(after[1]) < int(before[1])))',
      '    return True  # mutant: repeat zero-information long solves')],
     "e1_staged"),
    ("MUT-57 E1 二维单调 upper envelope 被禁用", [(_S13,
      '        if kk >= K and bb >= B:\n            ups.append(int(q[1]))',
      '        if False and kk >= K and bb >= B:  # mutant: lose northeast upper envelope\n            ups.append(int(q[1]))')],
     "e1_staged"),
    ("MUT-58 formal Xi 重新允许 mmsi=ALL 跨船统计", [(_S15,
      '        if len(concrete_mmsi) != 1:\n            raise ValueError(',
      '        if False and len(concrete_mmsi) != 1:  # mutant: permit cross-vessel mixture\n            raise ValueError('), (_S15,
      '        if requested.upper() == "ALL" or requested != target_mmsi:\n            raise ValueError(',
      '        if False:  # mutant: ignore requested ALL/mismatch\n            raise ValueError(')],
     "v9_protocol"),
    ("MUT-59 formal real replay 忽略 selected_mmsi 并混入 ALL 船", [(_S13,
      '            Path(real_samples_csv), mmsi=_replay_mmsi,',
      '            Path(real_samples_csv), mmsi="ALL",  # mutant: cross-vessel replay')],
     "v9_protocol"),
    ("MUT-60 formal SAA 稀疏格重新 pooled 跨船回退", [(_S10,
      '            if str(mmsi).upper() != "ALL" and not bool(allow_pooled_fallback):\n                continue',
      '            if False:  # mutant: allow pooled cross-vessel fallback\n                continue')],
     "v9_protocol"),
    ("MUT-61 formal E1 重新提前消费 final test", [(_S13,
      '        str(getattr(args, "study_mode", "mechanism")).lower() != "formal"\n        and getattr(args, "final_test_samples", None) is not None)',
      '        str(getattr(args, "study_mode", "mechanism")).lower() == "formal"  # mutant: E1 consumes test\n        and getattr(args, "final_test_samples", None) is not None)')],
     "v9_protocol"),
    ("MUT-62 coverage-target master 丢失固定覆盖等式", [(_S12,
      '                archive, all_tids, node, stage, coverage_target,\n',
      '                archive, all_tids, node, stage, None,  # mutant: drop target equality\n')],
     "v10_target"),
    ("MUT-63 target NO 证书忽略 Farkas pricing 完备性", [(_S12,
      '        and bool(stage_result.farkas_pricing_complete))',
      '        and True)  # mutant: ignore Farkas completeness')],
     "v10_target"),
    ("MUT-64 target INFEASIBLE 不再把 rigorous UB 收紧到 T-1", [(_S13,
      '        new_lb, new_ub = lb, min(ub, int(target) - 1)',
      '        new_lb, new_ub = lb, ub  # mutant: discard target NO information')],
     "v10_target"),
    ("MUT-65 targeted refinement 的 NaN 列不回退原 rigorous LB", [(_S13,
      '                  if raw_lb_ref is None or pd.isna(raw_lb_ref) else raw_lb_ref)',
      '                  if raw_lb_ref is None else raw_lb_ref)  # mutant: NaN erases evidence')],
     "v10_target"),
    ("MUT-66 validation selection gate 偷换成 mission 5% 而非内部 allocation", [(_S13,
      '    return replay_result.get("allocation_budget_holds")',
      '    return replay_result.get("mission_requirement_holds")  # mutant: silently loosen gate')],
     "v10_target"),
    ("MUT-67 target NO 证书忽略未闭合 open nodes", [(_S12,
      '        and int(stage_result.open_nodes) == 0',
      '        and True  # mutant: ignore open target nodes')],
     "v10_target"),
    ("MUT-68 完整路线宇宙错误接受 incomplete", [(_S12,
      '    if not universe.complete:\n        return False, "universe-incomplete"',
      '    if False:\n        return False, "universe-incomplete"')],
     "v11_universe"),
    ("MUT-69 完整路线宇宙忽略实例上下文哈希", [(_S12,
      '    if str(universe.context_sha256) != str(expected):\n        return False, "universe-context-mismatch"',
      '    if False:\n        return False, "universe-context-mismatch"')],
     "v11_universe"),
    ("MUT-70 完整路线宇宙忽略列语义哈希", [(_S12,
      '        if _route_universe_columns_sha256(universe.columns) != universe.columns_sha256:\n            return False, "universe-columns-hash-mismatch"',
      '        if False:\n            return False, "universe-columns-hash-mismatch"')],
     "v11_universe"),
    ("MUT-71 完整宇宙节点不可行仍退回 Phase-I", [(_S12,
      '            if master.status == "infeasible":\n                if complete_universe_mode:',
      '            if master.status == "infeasible":\n                if False:  # mutant: ignore full-universe closure')],
     "v11_universe"),
    ("MUT-72 完整宇宙仍重复隐式定价", [(_S12,
      '            if complete_universe_mode:\n                pricing = PricingSearchResult(',
      '            if False:  # mutant: force repeated implicit pricing\n                pricing = PricingSearchResult(')],
     "v11_universe"),
    ("MUT-73 knee 明细重新把 list 当 DataFrame.empty", [(_S13,
      '    if not detail_rows:\n        return None',
      '    if not detail_rows.empty:  # mutant: historical hidden list/DataFrame bug\n        pass\n    else:\n        return None')],
     "v11_universe"),
    ("MUT-74 full-cover target 不走 THM-FCT 直接闭合", [(_S12,
      '    if (bool(complete_universe_mode) and bool(decision_only)\n'
      '            and stage == "energy"\n'
      '            and int(coverage_target or -1) == len(all_tids)):',
      '    if (False and bool(complete_universe_mode) and bool(decision_only)\n'
      '            and stage == "energy"\n'
      '            and int(coverage_target or -1) == len(all_tids)):')],
     "v12_resource"),
    ("MUT-75 THM-FCT 错误扩展到 partial target", [(_S12,
      '            and int(coverage_target or -1) == len(all_tids)):',
      '            and int(coverage_target or -1) <= len(all_tids)):  # mutant: partial target')],
     "v12_resource"),
    ("MUT-76 full-cover 资源冲突退化回弱 exact-pattern cut", [(_S12,
      '        strong_cuts.append(cut)',
      '        exact_index_cuts.append(cut)  # mutant: discard strong full-cover cut')],
     "v12_resource"),
    ("MUT-77 formal 95% 离散满覆盖阈值重新标成 degenerate", [(_S13,
      '                degenerate_knee=False,',
      '                degenerate_knee=True,  # mutant: geometric-knee veto leaks into formal threshold')],
     "v12_resource"),
    ("MUT-78 hard-coverable-cap 后仍允许 generic long coverage refinement", [(_S13,
      '    return str(selection_row.get("saturation_proof", "")) != "hard-coverable-cap"',
      '    return True  # mutant: repeat no-gain generic coverage solves after hard cap')],
     "v12_resource"),
    ("MUT-79 validation 相同经验事件指纹偷偷放宽正式门", [(_S13,
      '    return replay_result.get("allocation_budget_holds")',
      '    return (replay_result.get("allocation_budget_holds") or\n'
      '            bool(replay_result.get("validation_duplicate_event_groups")))  # mutant: empirical grouping relaxes gate')],
     "v12_resource"),
    ("MUT-80 full-cover binary master 丢失覆盖等式", [(_S12,
      '            coverage_equal=target,',
      '            coverage_equal=None,  # mutant: drop total full-cover equality'),
     (_S12,
      '            full_cover_equal=True,',
      '            full_cover_equal=False,  # mutant: drop per-turbine full-cover equalities')],
     "v12_resource"),
    ("MUT-81 formal freeze 只信 eligible 标签不复核结构化证书", [(_S13,
      '        formal_ok = formal_mask & status_eligible & formal_proof_ok',
      '        formal_ok = formal_mask & status_eligible  # mutant: trust summary label only')],
     "v12_resource"),
    ("MUT-82 浮点 MILP infeasible 未经 exact master 复核直接签 NO", [(_S12,
      '                if exact_status == "INFEASIBLE_PROVEN":',
      '                if bool(master.infeasible_proven):  # mutant: trust floating infeasible')],
     "v13_hardening"),
    ("MUT-83 formal target refine 重新接受 v11 frontier", [(_S13,
      '    allowed = {RESULT_CONTRACT}',
      '    allowed = {RESULT_CONTRACT, "fleet-anytime-result-v11-complete-universe-target-closure"}')],
     "v13_hardening"),
    ("MUT-84 frozen plan fingerprint 恢复 9 位小数碰撞", [(_S13,
      '    raw = repr(BP._state_fp(tuple(rows)))',
      '    raw = repr([round(float(c.get("E_plan_Wh", c.get("E0", 0.0))), 9) for c in chosen or []])')],
     "v13_hardening"),
    ("MUT-85 formal final-test gate 忽略 E1 freeze", [(_S13,
      '    return bool(getattr(args, "_e1_formal_freeze_verified", False)\n                and getattr(args, "_e2_matrix_completion_verified", False)\n                and getattr(args, "_formal_sample_hashes_verified", False)\n                and not bool(getattr(args, "allow_incomplete_results", False))\n                and not bool(getattr(args, "allow_final_test_rerun", False))\n                and len(sha) == 64 and all(ch in "0123456789abcdef" for ch in sha))',
      '    return True  # mutant: manual formal config may consume final test')],
     "v13_hardening"),
    ("MUT-86 formal threshold 恢复 -1e-12 epsilon nudge", [(_S13,
      'int(math.ceil(float(frac) * int(plateau_cov)))',
      'int(math.ceil(float(frac) * int(plateau_cov) - 1e-12))')],
     "v13_hardening"),
    ("MUT-87 holdout purge 恢复 +1e-9 泄漏容差", [(_S15,
      '    if float(purge_min) < max_h:',
      '    if float(purge_min) + 1e-9 < max_h:')],
     "v13_hardening"),
    ("MUT-88 正式 simultaneous UCB 重新舍入 6 位后再过门", [(_S13,
      '    return None if value is None else float(value)',
      '    return None if value is None else round(float(value), 6)  # mutant: gate statistic rounded')],
     "v13_hardening"),
    ("MUT-89 THM-FCT proof-code anchor 漏掉 exact master verifier", [(_S12,
      '                 "_exact_fullcover_master_feasibility",\n',
      '')],
     "v13_hardening"),
    ("MUT-90 same-version E1 freeze 忽略 current-instance SHA", [(_S13,
      '        if not expected_e1 or got_e1 != expected_e1:',
      '        if False and (not expected_e1 or got_e1 != expected_e1):  # mutant: stale same-version freeze')],
     "v13_hardening"),
    ("MUT-91 exact master UNKNOWN_TIMEOUT 错当 INFEASIBLE_PROVEN", [(_S12,
      '                if exact_status == "INFEASIBLE_PROVEN":',
      '                if exact_status in ("INFEASIBLE_PROVEN", "UNKNOWN_TIMEOUT"):  # mutant: timeout signs NO')],
     "v13_hardening"),
    ("MUT-92 q_max 用 isclose 近似带而非 exact binary64 cell", [(_S13,
      '& pd.to_numeric(sub.get("q"), errors="coerce").map(lambda q: _binary64_equal(q, q_target))',
      '& np.isclose(pd.to_numeric(sub.get("q"), errors="coerce"), q_target)  # mutant: nearby lower q eligible')],
     "v14_protocol"),
    ("MUT-93 formal 重新允许 incomplete E2 matrix override", [(_S13,
      '    if bool(getattr(args, "allow_incomplete_results", False)):',
      '    if False and bool(getattr(args, "allow_incomplete_results", False)):  # mutant: formal escape hatch')],
     "v14_protocol"),
    ("MUT-94 formal 重新允许 final-test rerun override", [(_S13,
      '    if bool(getattr(args, "allow_final_test_rerun", False)):',
      '    if False and bool(getattr(args, "allow_final_test_rerun", False)):  # mutant: repeat confirmatory test')],
     "v14_protocol"),
    ("MUT-95 final test authorization 忽略 E2 matrix completion", [(_S13,
      '                and getattr(args, "_e2_matrix_completion_verified", False)\n',
      '')],
     "v14_protocol"),
    ("MUT-96 freeze 前重新 materialize final-test outcomes", [(_S13,
      '    md = RP.load_sample_metadata(path, mmsi=mmsi, formal=True, expected_split="test")',
      '    md = RP.load_samples(path, mmsi=mmsi, formal=True, expected_split="test")  # mutant: pre-freeze outcome load')],
     "v14_protocol"),
    ("MUT-97 E2 quantile grid 重新接受 NaN/out-of-range", [(_S13,
      '    if any((not math.isfinite(q)) or q < 0.0 or q > 1.0 for q in qs):',
      '    if False and any((not math.isfinite(q)) or q < 0.0 or q > 1.0 for q in qs):  # mutant: invalid grid')],
     "v14_protocol"),
    ("MUT-98 provenance 检查后样本文件被替换仍允许 final test", [(_S13,
      '        if got != want:',
      '        if False and got != want:  # mutant: ignore post-freeze sample tampering')],
     "v14_protocol"),
    ("MUT-99 v15 closure context 不再绑定 binary64 route semantics", [(_S12,
      '        tuple(_state_fp(_column_semantics_fp(c)) for c in archive),',
      '        tuple(_state_fp(_exact_route_signature(c)) for c in archive),  # mutant: energy/resource semantics omitted')],
     "v15_resource"),
    ("MUT-100 v15 跨实例 target checkpoint 被接受", [(_S12,
      '    if payload.get("context_sha256") != str(context_sha256):',
      '    if False and payload.get("context_sha256") != str(context_sha256):  # mutant: cross-instance reuse')],
     "v15_resource"),
    ("MUT-101 v15 target checkpoint payload 篡改不再 fail-closed", [(_S12,
      '    if got_sha != want_sha:',
      '    if False and got_sha != want_sha:  # mutant: ignore payload tampering')],
     "v15_resource"),
    ("MUT-102 v15 resume 丢弃已经证明的 resource cuts", [(_S12,
      '    return out\n\n\ndef _save_fullcover_closure_checkpoint',
      '    return []  # mutant: proven closure progress silently discarded\n\n\ndef _save_fullcover_closure_checkpoint')],
     "v15_resource"),
    ("MUT-103 battery bin-packing 只看 pooled energy 后错误判可行", [(_S12,
      '    items.sort(key=lambda z: (-z[0], z[1]))',
      '    return "FEASIBLE", 1  # mutant: skip exact bin packing\n    items.sort(key=lambda z: (-z[0], z[1]))')],
     "v15_resource"),
    ("MUT-104 battery relaxation UNKNOWN 错当作可切 infeasible core", [(_S12,
      '        if bp_status == "INFEASIBLE_PROVEN":',
      '        if bp_status in ("INFEASIBLE_PROVEN", "UNKNOWN_TIMEOUT"):  # mutant: timeout creates cut')],
     "v15_resource"),
    ("MUT-105 direct full-cover master 丢失 min(K,B) active cap", [(_S12,
      '    active_cap = min(int(K), int(batteries))',
      '    active_cap = int(K)  # mutant: forget battery concurrency lower bound')],
     "v15_resource"),
    ("MUT-106 direct full-cover master 丢失 fastest-turnaround interval cuts", [(_S12,
      '    for row in RA._interval_capacity_rows(fastest_turn_intervals):',
      '    for row in ():  # mutant: omit exact UAV turnaround relaxation')],
     "v15_resource"),
    ("MUT-107 --resume off 不再清空旧 target-closure ledger", [(_S12,
      '    if target_closure_checkpoint_path is not None and not bool(target_closure_resume):',
      '    if False and target_closure_checkpoint_path is not None and not bool(target_closure_resume):  # mutant: stale cuts survive fresh run')],
     "v15_resource"),
    ("MUT-108 resource-infeasible full pattern 错误缩成未证明 subset cut", [(_S12,
      '        cut = tuple(sorted(selection))',
      '        cut = tuple(sorted(selection[:-1]))  # mutant: unjustified subset cut')],
     "v15_resource"),
    ("MUT-109 THM-FCT proof-code anchors 漏掉 persistent checkpoint loader", [(_S12,
      '                 "_load_fullcover_closure_checkpoint",\n',
      '')],
     "v15_resource"),
    ("MUT-110 E1 scheduler 不把 --resume on 传给 persistent target closure", [(_S13,
      '                    target_closure_resume=(str(getattr(args, "resume", "on")).lower() == "on"),',
      '                    target_closure_resume=False,  # mutant: every target rerun restarts closure')],
     "v15_resource"),
    ("MUT-111 v15 target-closure ledger 不再绑定 algorithm fingerprint", [(_S12,
      '        str(algorithm_sha256 or "missing-algorithm-sha256"),',
      '        "algorithm-fingerprint-ignored",  # mutant: cross-algorithm proof state reuse')],
     "v15_resource"),
    ("MUT-112 v16 global battery relaxation 错用 E_plan 而非 SOC energy", [(_S12,
      '        e_f = float(c["E_soc_required_Wh"])\n        if not math.isfinite(e_f) or e_f < 0.0:',
      '        e_f = float(c["E_plan_Wh"])  # mutant: wrong energy semantics\n        if not math.isfinite(e_f) or e_f < 0.0:')],
     "v16_battery"),
    ("MUT-113 v16 battery lower bound 把 B*=B 错判成 NO", [(_S12,
      '    status = ("INFEASIBLE_PROVEN"\n              if min_required > B else "FEASIBLE_RELAXATION")',
      '    status = ("INFEASIBLE_PROVEN"\n              if min_required >= B else "FEASIBLE_RELAXATION")  # mutant')],
     "v16_battery"),
    ("MUT-114 v16 global battery UNKNOWN 错当 NO certificate", [(_S12,
      '    if global_battery_status == "INFEASIBLE_PROVEN":',
      '    if global_battery_status in {"INFEASIBLE_PROVEN", "UNKNOWN_TIMEOUT"}:  # mutant')],
     "v16_battery"),
    ("MUT-115 v16 coverage-mask DP 允许重叠 routes 拼成 full cover", [(_S12,
      '            if mask & rm:\n                continue\n            nm = mask | rm',
      '            if False and mask & rm:  # mutant: overlapping route masks allowed\n                continue\n            nm = mask | rm')],
     "v16_battery"),
    ("MUT-116 THM-GBR proof-code anchor 漏掉 global relaxation", [(_S12,
      '    ("THM-GBR", ("_exact_global_fullcover_battery_relaxation",',
      '    ("THM-GBR", ("_missing_global_battery_anchor",  # mutant')],
     "v16_battery"),
    ("MUT-117 v16 跳过 universe-level global battery relaxation", [(_S12,
      '     _global_battery_witness) = _exact_global_fullcover_battery_relaxation(\n        archive, all_tids, batteries, float(p.B_use), deadline)',
      '     _global_battery_witness) = ("SKIPPED_SIZE", None, 0, 0, tuple())  # mutant')],
     "v16_battery"),
    ("MUT-118 v16 binary64 battery capacity 比较重新加入正 tolerance", [(_S12,
      '        one_pack[mask] = e is not None and e <= cap',
      '        one_pack[mask] = e is not None and float(e) <= float(cap) + 1e-9  # mutant')],
     "v16_battery"),
    ("MUT-119 E1 target certificate 不记录 v16 global battery minimum", [(_S13,
      '                    target_global_battery_min_required=result.get(\n                        "target_global_battery_min_required"),',
      '')],
     "v16_battery"),
]


def _mut_workspace(dst):
    import shutil
    here = os.path.dirname(os.path.abspath(__file__))
    for f in os.listdir(here):
        if f.endswith(".py"):
            shutil.copy(os.path.join(here, f), os.path.join(dst, f))


def _mut_apply(dst, patches):
    for fn, old, new in patches:
        path = os.path.join(dst, fn)
        src = open(path, encoding="utf-8").read()
        assert src.count(old) == 1, \
            f"变异锚点漂移({fn}): 命中 {src.count(old)} 次: {old[:60]!r}"
        open(path, "w", encoding="utf-8").write(src.replace(old, new))


def _mut_run(dst, key, timeout_s=120):
    import subprocess
    try:
        r = subprocess.run([sys.executable, os.path.join(dst, "selftest.py"),
                            "--mutation-driver", key],
                           timeout=timeout_s, capture_output=True, text=True)
        return r.returncode, (r.stderr or "").strip().splitlines()[-3:]
    except subprocess.TimeoutExpired:
        return -9, ["TIMEOUT"]


def suite_mutations():
    """行为级 mutation。

    各 clean 驱动由对应功能 suite 独立验证；本 suite 只运行 mutant，避免
    clean+mutant 两轮并发 SciPy/BLAS 子进程造成资源竞争和假超时。每个 mutant
    仍在独立临时目录中执行，并保留单项 120s fail-safe。
    """
    import tempfile
    import concurrent.futures
    t0 = time.time()

    def _one_mut(item):
        mid, patches, key = item
        with tempfile.TemporaryDirectory() as td:
            _mut_workspace(td)
            _mut_apply(td, patches)
            rc, tail = _mut_run(td, key, timeout_s=120)
            return mid, key, rc, tail

    killed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(6, len(MUTATIONS))) as ex:
        for fut in concurrent.futures.as_completed([ex.submit(_one_mut, x) for x in MUTATIONS]):
            mid, key, rc, tail = fut.result()
            assert rc != 0, f"变异存活(未被杀): {mid} 驱动={key} tail={tail}"
            killed += 1
            print(f"  {mid} → 被杀 ✓ (驱动 {key})")
    print(f"suite mutations: {killed}/{len(MUTATIONS)} 项变异全部被杀 "
          f"({time.time() - t0:.0f}s) ✓")


SUITES["mutations"] = suite_mutations


def suite_counterexamples():
    """更新 反例(按交付约束收编于本文件; 不变量式验收)。"""
    _cx_gate_switch()
    _cx_deck_grid()
    print("suite counterexamples: C-01 / H-01 反例全部 PASS ✓")


def _cx_gate_switch():
    """C-01: gate 天气场离散切换无概率预算 ⇒ 假证书。两点分布构造:
    预测回收点落 T0/T1 最近邻分界线上偏 T0 一侧 1mm; ξ 取秩-1 Σ=vvᵀ(v 沿轴,
    |v|=60m); 两点分布 ξ∈{±v}(各 1/2)与 (μ=0,Σ) 逐位匹配, 两原子各落分界两侧
    ⇒ 旧口径(只查【预测】最近场)判门开, 实际门失败概率=0.5 ≫ ε_gate 且无预算。
    修复后: 全候选最坏情况 ⇒ 含 T1 列不可行, BP 覆盖=1, 双证书, 证据齐备。
    不变量: ¬(覆盖含 T1 ∧ 任一证书真)。"""
    p = M.apply_uav_profile(M.Params(), "L")
    T0 = _tb_at("T0", 2500.0, 0.0)
    T1 = _tb_at("T1", -2400.0, 300.0, storm=True)
    axis = (T0.local - T1.local) / np.linalg.norm(T0.local - T1.local)
    P_pred = (T0.local + T1.local) / 2.0 + 1e-3 * axis
    v = 60.0 * axis
    assert np.linalg.norm(P_pred + v - T0.local) < np.linalg.norm(P_pred + v - T1.local) \
        and np.linalg.norm(P_pred - v - T1.local) < np.linalg.norm(P_pred - v - T0.local), \
        "两点构造失效: 应各落最近邻分界两侧"
    H = 30
    opt = _mk_launch(2.5, tuple(P_pred / (H * 60.0)), _WX_CALM)
    Sigma = np.outer(v, v) + 1e-9 * np.eye(2)
    xi = M.XiAmbiguity({(h, "DP"): M.XiCell(h, "DP", 1000, np.zeros(2), Sigma,
                                            0.0, 0.0, 0.0)
                        for h in (15, 30)}, [15, 30])
    # (a) 旧口径演示: 预测最近场(温和)门开; 风暴候选门闭 ⇒ 切换失败概率 0.5
    route01 = RM.Route(-1, [T0, T1], opt.ship)
    gwx = RM.recovery_gate_wx(route01, _WX_CALM, opt.ship.predicted_at(float(H)))
    L_pred, _ = M.landing_gate(float(gwx["Hs"]), gwx.get("Tp", 6.0),
                               gwx.get("wave_dir", 0.0), gwx.get("ship_heading", 0.0),
                               float(gwx["wind10"]), p)
    L_storm, _ = M.landing_gate(_WX_STORM["Hs"], _WX_STORM["Tp"],
                                _WX_STORM["wave_dir"], _WX_STORM["ship_heading"],
                                _WX_STORM["wind10"], p)
    assert L_pred == 1 and L_storm == 0, "反例前提失效: 预测场应开门而风暴候选关门"
    # (b) 修复后逐候选最坏情况: 含 T1 的路线不可行, 证据记录 2 候选全查
    d = RM.route_feasible_at_h(route01, H, p, _WX_CALM, xi, weather_unc=None)
    gp = d["gate_weather_proof"]
    assert not d["feasible"] and d["gate"] == 0 and gp["candidate_count"] == 2 \
        and gp["all_candidates_checked"], "含风暴候选的路线未被最坏情况判杀"
    # (c) 端到端 BP + 不变量
    rb = BP.solve_soft_coverage_research([T0, T1], [opt], p, xi, 2, 60.0, deck_delta_min=2.5,
                              t_swap_min=4.0, max_stops=2, weather_unc=None,
                              batteries=2, seed_cols=[], time_limit_s=200,
                              deck_mode="interval")
    c = rb["certificate"]
    cov_T1 = any("T1" in col["tids"] for col in rb["chosen"])
    assert not (cov_T1 and (c["L1_certified"] or c["L2_certified"])), \
        "不变量违反: 覆盖含风暴场风机却仍持证"
    assert rb["covered"] == 1 and not cov_T1 and c["L1_certified"] and \
        c["L2_certified"] and c["gate_weather_switch_proven_safe"] and \
        c["gate_proof_missing"] == 0, "修复档应 cov=1(排除 T1)+双证书+证据齐备"
    print(f"C-01 反例: 两点分布切换概率 0.5 ≫ ε_gate="
          f"{float(getattr(p, 'eps_gate', p.eps_cap)):.2f}(旧口径无预算) → "
          f"全候选最坏情况判杀, BP cov=1 + 双证书 → PASS")


def _cx_deck_grid():
    """H-01: Δ-格点漏判连续物理重叠。绑定实例(τ∈{2.5,5}, t_launch=3.0):
    prep [0,2.5) 与 [2,5) 在 [2,2.5) 物理重叠却无公共 Δ 格点。三方对拍:
    (a) 旧口径独立重建(仅格点行的两阶段词典序 MILP, 本函数内独立实现)
        ⇒ 覆盖 2 且所选列连续甲板重叠(物理不可执行);
    (b) 修复后 BP ⇒ 覆盖 1, 双证书, 逐对行生效;
    (c) 独立连续物理 oracle ⇒ 覆盖 1, 与 BP 逐位一致。
    不变量: ¬(BP 方案连续甲板重叠 ∧ 任一证书真); 且 cov_old > cov_new(绑定性)。"""
    from scipy.optimize import milp, LinearConstraint, Bounds
    TL, TS, DELTA, K, BAT, T_MIN = 3.0, 4.0, 2.5, 2, 2, 60.0
    p = M.apply_uav_profile(M.Params(), "L")
    tb = [_tb_at("T0", 2500.0, 0.0), _tb_at("T1", -2400.0, 300.0)]
    ops = [_mk_launch(2.5, (0.1, 0.0), _WX_CALM),
           _mk_launch(5.0, (0.1, 0.0), _WX_CALM)]
    xi = _xi_diag()
    ext, st = RA.enumerate_discrete_routes(tb, ops, p, xi, T_MIN, DELTA, 1, None)
    assert st["anchor_complete"], f"锚点不完整: {st}"

    def _grid_pts(a, b, n_tgrid):
        i0 = int(math.ceil((a - 1e-9) / DELTA))
        i1 = int(math.floor((b - 1e-9) / DELTA))
        return {i for i in range(max(i0, 0), min(i1 + 1, n_tgrid))
                if a - 1e-9 <= i * DELTA < b - 1e-9}

    def _legacy_grid_only(cols):
        n = len(cols)
        n_tgrid = int(math.floor((T_MIN + TS) / DELTA)) + 1
        deck_of, occ_of = [], []
        for c in cols:
            dset = set()
            for (a, b) in _c_deck_ivs(c["tau"], c["h"], TL, TS):
                dset |= _grid_pts(a, b, n_tgrid)
            deck_of.append(dset)
            occ_of.append(_grid_pts(max(c["tau"] - TL, 0.0),
                                    c["tau"] + c["h"] + TS, n_tgrid))
        tids = sorted({t for c in cols for t in c["tids"]})
        ti = {t: k for k, t in enumerate(tids)}
        m = len(tids)
        rows, rhs = [], []
        for t in tids:                               # y_t − Σ_{c∋t} x_c ≤ 0
            r = np.zeros(n + m)
            r[n + ti[t]] = 1.0
            for j, c in enumerate(cols):
                if t in c["tids"]:
                    r[j] = -1.0
            rows.append(r); rhs.append(0.0)
        r = np.zeros(n + m); r[:n] = 1.0             # 电池 ≤ B
        rows.append(r); rhs.append(float(BAT))
        for g in range(n_tgrid):                     # 占机 ≤ K(仅格点)
            r = np.zeros(n + m)
            for j in range(n):
                if g in occ_of[j]:
                    r[j] = 1.0
            if r.any():
                rows.append(r); rhs.append(float(K))
        for g in range(n_tgrid):                     # 甲板容量 1(仅格点)
            r = np.zeros(n + m)
            for j in range(n):
                if g in deck_of[j]:
                    r[j] = 1.0
            if r[:n].sum() >= 2:
                rows.append(r); rhs.append(1.0)
        A = np.vstack(rows); b = np.asarray(rhs)
        integ = np.ones(n + m)
        bnd = Bounds(np.zeros(n + m), np.ones(n + m))
        obj1 = np.concatenate([np.zeros(n), -np.ones(m)])
        r1 = milp(obj1, constraints=LinearConstraint(A, -np.inf, b),
                  integrality=integ, bounds=bnd)
        cov = int(round(-r1.fun))
        E = np.array([float(c["E0"]) for c in cols])
        A2 = np.vstack([A, obj1])                    # −Σy ≤ −cov ⇔ 覆盖 ≥ cov
        b2 = np.concatenate([b, [-float(cov)]])
        r2 = milp(np.concatenate([E, np.zeros(m)]),
                  constraints=LinearConstraint(A2, -np.inf, b2),
                  integrality=integ, bounds=bnd)
        sel = [j for j in range(n) if r2.x[j] > 0.5]
        return cov, float(sum(E[j] for j in sel)), sel

    cov_old, E_old, sel_old = _legacy_grid_only(ext)
    overlap_old = any(
        _c_ovl(_c_deck_ivs(ext[i]["tau"], ext[i]["h"], TL, TS),
               _c_deck_ivs(ext[j]["tau"], ext[j]["h"], TL, TS))
        for a, i in enumerate(sel_old) for j in sel_old[a + 1:])
    assert cov_old == 2 and overlap_old, \
        "反例前提失效: 旧格点语义应允许物理重叠的双发(覆盖 2)"
    cov_o, E_o = _cont_lex_oracle(ext, BAT, K, TL, TS)
    rb = BP.solve_soft_coverage_research(tb, ops, p, xi, K, T_MIN, deck_delta_min=DELTA,
                              t_swap_min=TS, max_stops=1, weather_unc=None,
                              batteries=BAT, seed_cols=[], time_limit_s=200,
                              deck_mode="interval", t_launch_min=TL)
    c = rb["certificate"]
    overlap_new = not _plan_cont_ok(rb["chosen"], BAT, K, TL, TS)
    assert not (overlap_new and (c["L1_certified"] or c["L2_certified"])), \
        "不变量违反: 物理重叠方案持有证书"
    assert not overlap_new and rb["covered"] == cov_o and \
        abs(rb["energy_Wh"] - E_o) <= max(1e-6 * max(1.0, E_o), 0.05) and \
        c["L1_certified"] and c["L2_certified"] and \
        c["deck_conflict_semantics_exact"] and cov_old > cov_o, \
        (f"H-01 反例断言失败: BP=({rb['covered']},{rb['energy_Wh']}) "
         f"oracle=({cov_o},{E_o}) old={cov_old} reason={c['certificate_reason']}")
    print(f"H-01 反例: 旧格点重建 覆盖 {cov_old}(E={E_old:.1f}, 连续甲板重叠) → "
          f"连续语义 覆盖 {cov_o}(E={E_o:.1f}, 双证书, 逐对行 "
          f"{rb['deck_pair_stats']['row_added']} 条) → PASS")


SUITES["counterexamples"] = suite_counterexamples



def _independent_event_capacity_ok(events, candidate, capacity):
    """Independent strict half-open interval-capacity checker."""
    if float(candidate[1]) <= float(candidate[0]):
        return True
    all_events = list(events) + [tuple(candidate)]
    starts = sorted(float(a) for a, b in all_events if float(b) > float(a))
    return all(sum(float(a) <= t < float(b)
                   for a, b in all_events) <= int(capacity) for t in starts)


def _resource_oracle_no_symmetry(columns, resource_of, K, B, B_use,
                                 quick_min, swap_min, quick_capacity, swap_capacity):
    """Independent exhaustive resource oracle with no UAV/battery symmetry pruning.

    It deliberately enumerates every admissible UAV and battery identity.  The
    implementation shares no search-state signature or dominance rule with the
    production DFS; only the public task/resource contract is reproduced.
    """
    ordered = tuple(sorted(range(len(columns)),
                           key=lambda j: (float(resource_of[j]["launch_start_min"]),
                                          float(resource_of[j]["recovery_min"]), j)))
    seen_tids = set()
    for j in ordered:
        for tid in tuple(columns[j].get("ordered_tids", columns[j].get("tids", ()))):
            if tid in seen_tids:
                return False
            seen_tids.add(tid)
    if not ordered:
        return True
    if K <= 0 or B <= 0 or quick_capacity <= 0 or swap_capacity <= 0:
        return False

    def ov(a, b):
        return max(float(a[0]), float(b[0])) < min(float(a[1]), float(b[1]))

    for pos, j in enumerate(ordered):
        for q in ordered[pos + 1:]:
            if any(ov(a, b) for a in resource_of[j]["deck"] for b in resource_of[q]["deck"]):
                return False

    from fractions import Fraction as _ResourceFraction
    uav_last = [None] * int(K)
    uav_current_battery = [None] * int(K)
    battery_binding = [None] * int(B)
    battery_used = [_ResourceFraction(0, 1) for _ in range(int(B))]
    B_use_exact = _ResourceFraction.from_float(float(B_use))
    quick_events = []
    swap_events = []

    def search(pos):
        if pos == len(ordered):
            return True
        j = ordered[pos]
        start = float(resource_of[j]["launch_start_min"])
        e_need_float = float(columns[j]["E_soc_required_Wh"])
        if not math.isfinite(e_need_float) or e_need_float < 0.0:
            return False
        e_need = _ResourceFraction.from_float(e_need_float)
        if e_need > B_use_exact:
            return False
        for k in range(int(K)):                         # no UAV symmetry pruning
            prev = uav_last[k]
            prev_b = uav_current_battery[k]
            for b in range(int(B)):                     # no battery symmetry pruning
                owner = battery_binding[b]
                if owner not in (None, k):
                    continue
                if battery_used[b] + e_need > B_use_exact:
                    continue
                event = None
                event_list = None
                if prev is not None:
                    clear = float(resource_of[prev]["clear_end_min"])
                    if b == prev_b:
                        ready = clear + max(float(quick_min), 0.0)
                        if start < ready:
                            continue
                        event = (clear, ready); event_list = quick_events
                        if not _independent_event_capacity_ok(
                                quick_events, event, int(quick_capacity)):
                            continue
                    else:
                        ready = clear + max(float(swap_min), 0.0)
                        if start < ready:
                            continue
                        event = (clear, ready); event_list = swap_events
                        if not _independent_event_capacity_ok(
                                swap_events, event, int(swap_capacity)):
                            continue

                old_owner = battery_binding[b]
                old_last = uav_last[k]
                old_current = uav_current_battery[k]
                battery_binding[b] = k
                battery_used[b] += e_need
                uav_last[k] = j
                uav_current_battery[k] = b
                if event is not None:
                    event_list.append(event)
                if search(pos + 1):
                    return True
                if event is not None:
                    event_list.pop()
                uav_current_battery[k] = old_current
                uav_last[k] = old_last
                battery_used[b] -= e_need
                battery_binding[b] = old_owner
        return False

    return search(0)


def suite_random_oracle():
    """Three independent randomized checks.

    1. Six physical continuous-time instances retain the historical research-BP
       cross-check.
    2. Four production-physics instances compare formal ``solve_fleet_anytime``
       (without injected test columns) against complete discrete-route enumeration
       plus exhaustive subset/entity-resource audit.
    3. At least 100 abstract finite-route instances compare the formal exact BPC
       against a complete subset/resource oracle.
    4. At least 10,000 entity-resource instances compare the production DFS with
       an independent exhaustive oracle that performs no symmetry pruning.
    """
    n_physical = int(os.environ.get("SELFTEST_ORACLE_PHYSICAL_N", "6"))
    n_formal_physical = int(os.environ.get("SELFTEST_FORMAL_PHYSICAL_BPC_N", "4"))
    n_bpc = int(os.environ.get("SELFTEST_BPC_ORACLE_N", "100"))
    n_resource = int(os.environ.get("SELFTEST_RESOURCE_ORACLE_N", "10000"))
    seed = int(os.environ.get("SELFTEST_ORACLE_SEED", "0"))
    if n_formal_physical < 4 or n_bpc < 100 or n_resource < 10000:
        raise AssertionError(
            "certification run requires >=4 formal physical BPC, >=100 abstract BPC "
            "and >=10000 resource instances")
    rng = np.random.default_rng(seed)
    t0 = time.time()

    # A. Physical continuous-time oracle retained as a separate engineering check.
    for it in range(n_physical):
        ntb = int(rng.integers(2, 4))
        tb = []
        for i in range(ntb):
            ang = rng.uniform(0, 2 * np.pi)
            rr = rng.uniform(1500, 3200)
            tb.append(_tb_at(f"T{i}", rr * np.cos(ang), rr * np.sin(ang)))
        taus = ((2.5,), (2.5, 5.0), (2.5, 7.5))[it % 3]
        ops = [_mk_launch(tau, (0.1, 0.0), _WX_CALM) for tau in taus]
        tl = float(rng.choice([2.5, 3.0]))
        ts = float(rng.choice([4.0, 3.7]))
        K = int(rng.integers(1, 3)); B = int(rng.integers(1, 4))
        ms = int(rng.integers(1, 3))
        p = M.apply_uav_profile(M.Params(), "L")
        xi = _xi_diag()
        ext, st = RA.enumerate_discrete_routes(tb, ops, p, xi, 60.0, 2.5, ms, None)
        assert st["anchor_complete"], f"[{it}] 锚点不完整: {st}"
        cov_o, E_o = _cont_lex_oracle(ext, B, K, tl, ts)
        r = BP.solve_soft_coverage_research(
            tb, ops, p, xi, K, 60.0, deck_delta_min=2.5,
            t_swap_min=ts, max_stops=ms, weather_unc=None,
            batteries=B, seed_cols=[], time_limit_s=200,
            deck_mode="interval", t_launch_min=tl)
        c = r["certificate"]
        assert r["covered"] == cov_o and abs(r["energy_Wh"] - E_o) <= max(
            1e-6 * max(1.0, E_o), 0.05), (it, r["covered"], r["energy_Wh"], cov_o, E_o)
        assert c["L1_certified"] and c["L2_certified"]
        assert _plan_cont_ok(r["chosen"], B, K, tl, ts)
    print(f"  物理连续 oracle: {n_physical} 实例全部一致 ✓")

    # B. Production physics path: no implicit_test_columns and no injected route
    #    pool.  Complete enumeration is test-only and supplies the independent
    #    truth set for exhaustive subset/resource auditing.
    formal_cases = [
        ([(600.0, 0.0), (-700.0, 0.0)], 1, 2, 1),
        ([(600.0, 0.0), (0.0, 700.0)], 1, 2, 2),
        ([(500.0, 0.0), (-500.0, 0.0), (0.0, 600.0)], 2, 3, 1),
        ([(500.0, 0.0), (-500.0, 0.0), (0.0, 600.0)], 2, 3, 2),
    ]
    for it in range(n_formal_physical):
        points, K, B, ms = formal_cases[it % len(formal_cases)]
        shift = 20.0 * (it // len(formal_cases))
        tb = [_tb_at(f"FP{it}_{j}", x + shift, y)
              for j, (x, y) in enumerate(points)]
        horizons = (15, 30)
        ops = [_mk_launch(2.5, (0.0, 0.0), _WX_CALM, horizons=horizons)]
        p = M.apply_uav_profile(M.Params(), "L")
        xi = _xi_diag(horizons, var=100.0)
        ext, stats = RA.enumerate_discrete_routes(
            tb, ops, p, xi, 60.0, 2.5, ms, None,
            kappa_mode="vp_unimodal", max_evals=None,
            reach_mode="valid", deadline=None)
        assert stats["route_space_complete"] and stats["anchor_complete"], stats
        ids = tuple(str(t.tid) for t in tb)
        cov_o, E_o, _, _ = _bpc_oracle(
            ext, ids, K=K, batteries=B, p=p,
            quick=p.quick_inspection_min, swap=6.0)
        result = BP.solve_fleet_anytime(
            tb, ops, p, xi, K, 60.0,
            batteries=B, max_stops=ms, time_limit_s=15.0,
            solver_mode="exact-branch-price-cut",
            pricing_mode="exact-implicit-dfs", t_swap_min=6.0,
            t_launch_min=2.5, quick_inspection_capacity=1,
            swap_station_capacity=1, energy_gap_target_abs_Wh=0.0,
            energy_gap_target_rel=0.0)
        assert result["coverage_incumbent"] == cov_o, (it, result, cov_o, E_o)
        assert abs(float(result["energy_incumbent_Wh"]) - float(E_o)) <= 1e-6, (it, result, cov_o, E_o)
        assert result["lexicographic_optimal"] is True
        assert result["global_certificate_available"] is True
        assert result["route_space_materialized"] is False
        assert result["exact_pricing_calls"] >= 1
        assert result["duplicate_turbine_visits"] == []
    print(f"  正式物理 BPC vs 完整离散路线/资源 oracle: {n_formal_physical} 实例全部一致 ✓")

    # C. Algorithmic synthetic finite-route BPC vs complete abstract oracle.
    aggregate_cuts = 0
    aggregate_nodes = 0
    for it in range(n_bpc):
        nt = int(rng.integers(2, 7))
        ids = tuple(f"R{it}_T{i}" for i in range(nt))
        K = int(rng.integers(1, 4)); B = int(rng.integers(1, 5))
        target_n = int(rng.integers(nt, min(11, 3 * nt + 1)))
        keys = set(); columns = []
        attempts = 0
        while len(columns) < target_n and attempts < 500:
            attempts += 1
            m = int(rng.integers(1, min(3, nt) + 1))
            seq = tuple(str(x) for x in rng.choice(ids, size=m, replace=False).tolist())
            tau = float(rng.choice([2.5, 5.0, 7.5, 10.0, 12.5]))
            h = float(rng.choice([5.0, 10.0, 15.0]))
            key = (tau, seq, h)
            if key in keys:
                continue
            keys.add(key)
            columns.append(_bpc_test_column(seq, tau, h, float(rng.integers(5, 100))))
        cov_o, E_o, _, _ = _bpc_oracle(columns, ids, K=K, batteries=B)
        result = BP._solve_fleet_anytime_synthetic_fixture(
            _bpc_turbines(*ids), [], M.Params(), _bpc_xi(), K, 90.0,
            batteries=B, max_stops=max(len(c["tids"]) for c in columns),
            time_limit_s=15.0, allow_resource_only_columns=True,
            implicit_test_columns=columns, energy_gap_target_abs_Wh=0.0,
            energy_gap_target_rel=0.0)
        assert result["coverage_incumbent"] == cov_o, (it, result, cov_o, E_o)
        assert abs(float(result["energy_incumbent_Wh"]) - float(E_o)) <= 1e-6, (it, result, cov_o, E_o)
        assert result["coverage_gap_abs"] == 0
        assert result["energy_gap_abs_Wh"] is not None and result["energy_gap_abs_Wh"] <= 1e-6
        assert result["lexicographic_optimal"] is True
        assert result["algorithmic_global_certificate"] is True
        assert result["global_certificate_available"] is False
        assert result["physical_model_global_certificate"] is False
        assert result["implicit_route_space_certified"] is False
        assert result["route_universe_source"] == "synthetic-test-fixture"
        assert result["route_universe_provenance_certified"] is False
        assert result["duplicate_turbine_visits"] == []
        aggregate_cuts += int(result["resource_pattern_cuts_added"])
        aggregate_nodes += int(result["processed_nodes"])
    print(f"  算法级 synthetic finite-route BPC oracle: {n_bpc} 实例全部一致; "
          f"processed_nodes={aggregate_nodes}, cuts={aggregate_cuts}；物理证书保持关闭 ✓")

    # D. Production resource DFS vs independent no-symmetry exhaustive oracle.
    feasible_count = infeasible_count = 0
    size_seen = set()
    for it in range(n_resource):
        n = 1 + (it % 7); size_seen.add(n)
        K = 1 + int(rng.integers(0, 3)); B = 1 + int(rng.integers(0, 4))
        B_use = 100.0
        quick = float(rng.integers(0, 4)); swap = float(rng.integers(0, 6))
        qcap = 1 + int(rng.integers(0, 2)); scap = 1 + int(rng.integers(0, 2))
        case = it % 5
        if case in (0, 1):
            starts = np.sort(rng.integers(0, 8, size=n)).astype(float)
        else:
            starts = np.cumsum(rng.integers(4, 9, size=n)).astype(float)
        columns = []
        resources = []
        for j, start in enumerate(starts):
            duration = float(rng.integers(1, 5))
            recovery = float(start + max(0.5, duration - 0.5))
            clear = float(start + duration)
            energy = (125.0 if case == 0 and j == 0 else float(rng.integers(8, 56)))
            columns.append(dict(tids=(f"Q{it}_{j}",), ordered_tids=(f"Q{it}_{j}",),
                                E_plan_Wh=min(energy, B_use), E_soc_required_Wh=energy,
                                tau=start, h=max(0.5, recovery - start)))
            deck = ()
            if case == 1 and j % 3 == 0:
                deck = ((start, start + 0.25), (recovery, clear))
            resources.append(dict(deck=deck, active=(start, clear),
                                  launch_start_min=start, launch_min=start,
                                  recovery_min=recovery, clear_end_min=clear))
        expected = _resource_oracle_no_symmetry(
            columns, resources, K, B, B_use, quick, swap, qcap, scap)
        audit = RA.audit_resource_assignment(
            columns, tuple(range(n)), K, B, B_use, resources,
            quick, swap, qcap, scap, deadline=None)
        actual = audit.status is FAC.ResourceAuditStatus.FEASIBLE
        assert audit.status is not FAC.ResourceAuditStatus.UNKNOWN_TIMEOUT
        assert actual == expected, (it, n, K, B, quick, swap, qcap, scap, audit.status)
        feasible_count += int(actual); infeasible_count += int(not actual)
    assert size_seen == set(range(1, 8))
    assert feasible_count > 0 and infeasible_count > 0
    print(f"  资源 DFS 无对称剪枝 oracle: {n_resource} 实例一致 "
          f"(feasible={feasible_count}, infeasible={infeasible_count}) ✓")
    print(f"suite random_oracle: 全部 PASS ({time.time() - t0:.1f}s) ✓")


SUITES["random_oracle"] = suite_random_oracle


def suite_seed_validation():
    """更新 P0 回归：整数 LP 不依赖 MILP；外部 seed 列全部按当前模型重验证。"""
    import scipy.optimize as _spo

    # P0-1: 复用“首个可行 h 非最低能耗”夹具。把 scipy.milp 强制设为失败；
    # 根 LP 已整数时，求解器必须直接重建/验证 LP incumbent，并给出真实 L1/L2 最优值。
    p = M.apply_uav_profile(M.Params(), "L")
    p.tau_insp = 300.0
    p.P_wait = 1.0
    p.use_zeng = False
    H = [15, 30, 60]
    t = _tb_at("T0", 12000.0, 0.0)
    wx = dict(wind10=0.0, wind_dir_from=270.0, Hs=0.3, Tp=6.0,
              wave_dir=0.0, ship_heading=0.0)
    sp = RM.ShipPrediction.from_cv(np.zeros(2), np.array([3.0, 0.0]), H, c_state="DP")
    sp.tau_min = 0.0; sp.wx_tau = wx
    opt = RM.LaunchOption(0.0, sp, wx)
    xi = M.XiAmbiguity({(h, "DP"): M.XiCell(h, "DP", 1000, np.zeros(2),
                                                    np.diag([400.0, 400.0]), 0.0, 0.0, 0.0)
                         for h in H}, H)
    route = RM.Route(-1, [t], sp)
    _orig_k = RM.kappa
    RM.kappa = RM.KAPPA_MODES["vp_unimodal"]
    try:
        feas = [(h, RM.route_feasible_at_h(route, h, p, wx, xi)) for h in H]
    finally:
        RM.kappa = _orig_k
    best_E = min(float(d["E_plan_Wh"]) for _h, d in feas if d["feasible"])

    class _FailedMilp:
        success = False
        status = 1
        message = "更新 forced MILP failure"
        x = None

    orig_milp = _spo.milp
    _spo.milp = lambda *a, **k: _FailedMilp()
    try:
        r = BP.solve_soft_coverage_research([t], [opt], p, xi, K=1, T_min=90.0,
                                 deck_delta_min=2.5, t_swap_min=4.0, max_stops=1,
                                 weather_unc=None, batteries=1, seed_cols=[],
                                 time_limit_s=200, deck_mode="interval")
    finally:
        _spo.milp = orig_milp
    assert r["covered"] == 1 and abs(r["energy_Wh"] - best_E) <= 0.05, r
    assert r["certificate"]["L1_certified"] and r["certificate"]["L2_certified"],         r["certificate"]
    assert r["milp_runtime"]["l1_integral_lp_incumbents"] >= 1 and         r["milp_runtime"]["l2_integral_lp_incumbents"] >= 1 and         r["milp_runtime"]["integral_lp_validation_failures"] == 0, r["milp_runtime"]
    print(f"P0-1: MILP 不可用时由整数 LP 直接构造并验证 incumbent，覆盖=1, E={r['energy_Wh']:.1f} → PASS")

    # P0-2a: 当前风机在 10^9 m 外不可达；seed 内伪造同 tid 的近距离 turbine 与 E0=0。
    # 规范化必须映射回当前 turbine，重算后拒绝，绝不能覆盖 1。
    p2 = M.apply_uav_profile(M.Params(), "L")
    H2 = [15, 30]
    far = _tb_at("FAR", 1.0e9, 0.0)
    sp2 = RM.ShipPrediction.from_cv(np.zeros(2), np.zeros(2), H2, c_state="DP")
    sp2.tau_min = 0.0; sp2.wx_tau = wx
    opt2 = RM.LaunchOption(0.0, sp2, wx)
    xi2 = M.XiAmbiguity({(h, "DP"): M.XiCell(h, "DP", 1000, np.zeros(2),
                                                     np.diag([100.0, 100.0]), 0.0, 0.0, 0.0)
                          for h in H2}, H2)
    fake_near = _tb_at("FAR", 1.0, 0.0)
    fake_seed = [dict(tau=0.0, ship=sp2, wx=wx, tids=("FAR",), h=15.0,
                      E0=0.0, route=RM.Route(-1, [fake_near], sp2))]
    r_far = BP.solve_soft_coverage_research([far], [opt2], p2, xi2, K=1, T_min=40.0,
                                 max_stops=1, batteries=1, seed_cols=fake_seed,
                                 time_limit_s=100, deck_mode="interval")
    sv = r_far["seed_validation"]
    assert r_far["covered"] == 0 and r_far["energy_Wh"] == 0.0, r_far
    assert sv["accepted_count"] == 0 and sv["rejected_count"] == 1 and         sv["rejection_reasons"].get("physical-infeasible") == 1, sv
    assert r_far["certificate"]["seed_columns_revalidated"] is True, r_far["certificate"]
    print("P0-2a: 伪造近距离 route/E0=0 的不可达 seed 被重验证并拒绝 → PASS")

    # P0-2b: 航路真实可行但输入 E0=0。接受列时必须覆盖 E0；最终 chosen 再逐列物理复算。
    near = _tb_at("N", 2500.0, 0.0)
    real_route = RM.Route(-1, [near], sp2)
    tampered = [dict(tau=0.0, ship=sp2, wx=wx, tids=("N",), h=30.0,
                     E0=0.0, route=real_route)]
    r_near = BP.solve_soft_coverage_research([near], [opt2], p2, xi2, K=1, T_min=40.0,
                                  max_stops=1, batteries=1, seed_cols=tampered,
                                  time_limit_s=100, deck_mode="interval")
    sv2 = r_near["seed_validation"]
    assert r_near["covered"] == 1 and r_near["energy_Wh"] > 1.0, r_near
    assert sv2["accepted_count"] == 1 and sv2["energy_overwritten_count"] == 1, sv2
    for c in r_near["chosen"]:
        dd = RM.route_feasible_at_h(c["route"], int(round(c["h"])), p2, c["wx"], xi2)
        assert dd["feasible"] and abs(float(c["E0"]) - float(dd["E_plan_Wh"])) <= 1e-6, (c, dd)
    assert r_near["certificate"]["L1_certified"] and r_near["certificate"]["L2_certified"],         r_near["certificate"]
    print(f"P0-2b: 可行 seed 的伪 E0=0 被覆盖，最终 E={r_near['energy_Wh']:.1f} → PASS")

    # P0-3: 即使总时间为 0，已验证 seed restricted-master incumbent 也必须保留。
    r_timeout = BP.solve_soft_coverage_research([near], [opt2], p2, xi2, K=1, T_min=40.0,
                                     max_stops=1, batteries=1, seed_cols=tampered,
                                     time_limit_s=0, deck_mode="interval")
    assert r_timeout["covered"] == 1 and r_timeout["incumbent_preserved"], r_timeout
    assert r_timeout["incumbent_source"] == "seed_restricted_master", r_timeout["incumbent_source"]
    assert r_timeout["L1_status"] == "time-limit-no-certificate" and \
        r_timeout["status"].startswith("feasible-unproven"), r_timeout["status"]
    print("P0-3: time_limit=0 仍返回 seed 覆盖1，不再退化为空解 → PASS")

    print("suite seed_validation: seed 重验证、能耗覆盖与超时 incumbent 保护全部 PASS ✓")


SUITES["seed_validation"] = suite_seed_validation


def suite_solver_validation():
    """更新 回归: ① 分数节点腐败 milp"成功"解(维度错/NaN/违约伪解)被独立行验证
    拒绝, 值与真 milp 基线一致 + 双证书 + 失败计数; ② lex Stage-2 per-τ 天气一致;
    ③ 划分 LP 对偶提取 fail-closed; ④ seed 天气解析与定价一致(wx_tau 优先);
    ⑤ 外部 eval_cache 投毒不再被采信。"""
    import scipy.optimize as _spo

    # ---- ① P0-NEW: 默认认证路径不依赖启发式 MILP，独立验证器拒绝伪成功解 ----
    p = M.apply_uav_profile(M.Params(), "L")
    tbs = [_tb_at("A", 2500.0, 0.0), _tb_at("B", 0.0, 2500.0),
           _tb_at("C", -2500.0, 0.0)]
    opts = [_mk_launch(tt, (0.0, 0.0), _WX_CALM, horizons=(15,))
            for tt in (0.1, 0.2, 0.3)]
    x1 = M.XiAmbiguity({(15, "DP"): M.XiCell(15, "DP", 1000, np.zeros(2),
                                             np.diag([100.0, 100.0]), 0.0, 0.0, 0.0)},
                       [15])
    kw = dict(deck_delta_min=2.5, t_swap_min=1.0, t_launch_min=0.05, max_stops=1,
              weather_unc=None, batteries=3, seed_cols=[], time_limit_s=200,
              deck_mode="interval")
    base = BP.solve_soft_coverage_research(tbs, opts, p, x1, 3, 40.0, **kw)
    assert base["certificate"]["L1_certified"] and base["certificate"]["L2_certified"], \
        base["certificate"]
    assert base["milp_runtime"]["l1_calls"] == 0 and base["milp_runtime"]["l2_calls"] == 0, \
        base["milp_runtime"]
    print(f"① 默认认证路径: cov={base['covered']} E={base['energy_Wh']}，"
          "不调用启发式 MILP 且双证书成立 ✓")

    import types
    obj = np.array([0.0, -1.0])
    A_test = np.array([[-1.0, 1.0]])       # y <= x
    b_test = np.array([0.0])
    lo_test = np.zeros(2); hi_test = np.ones(2); int_test = np.ones(2)
    bad = {
        "wrongdim": np.zeros(1),
        "nan": np.array([np.nan, 0.0]),
        "lying": np.array([0.0, 1.0]),
    }
    for kind, xbad in bad.items():
        rr = types.SimpleNamespace(success=True, status=0, x=xbad, fun=None)
        z, value, reason = BP._validate_milp_primal(
            rr, obj, A_test, b_test, lo_test, hi_test, int_test)
        assert z is None and value is None and reason, (kind, z, value, reason)
        print(f"① P0-NEW[{kind}]: 伪 success 解被独立验证拒绝({reason}) → PASS")

    # ---- ② P2-01: lex Stage-2 per-τ 天气一致 ----
    _drv_lex_wx()
    print("② P2-01: lex Stage-2 用 per-τ 天气, E_LP=chosen 总能耗=wind9 直接重算 → PASS")

    # ---- ③ P2-03: 对偶提取 fail-closed ----
    _drv_dual_fail()
    print("③ P2-03: 划分 LP 对偶缺失 ⇒ (None,…) fail-closed → PASS")

    # ---- ④ P2-02: seed 天气解析与定价一致(wx_tau 优先) ----
    sp4 = RM.ShipPrediction.from_cv(np.zeros(2), np.zeros(2), [15, 30], c_state="DP")
    sp4.tau_min = 0.0
    sp4.wx_tau = dict(_WX_CALM)                       # 定价将用 calm
    opt4 = RM.LaunchOption(0.0, sp4, dict(_WX_STORM))  # 调用方却给 storm 的 opt.wx
    tN = _tb_at("N", 2500.0, 0.0)
    x4 = _xi_diag()
    seed4 = [dict(tau=0.0, ship=sp4, wx=dict(_WX_STORM), tids=("N",), h=30.0,
                  E0=0.0, route=RM.Route(-1, [tN], sp4))]
    r4 = BP.solve_soft_coverage_research([tN], [opt4], p, x4, 1, 40.0, max_stops=1,
                              batteries=1, seed_cols=seed4, time_limit_s=100,
                              deck_mode="interval")
    sv4 = r4["seed_validation"]
    assert sv4["accepted_count"] == 1 and r4["covered"] == 1 and \
        r4["energy_Wh"] > 1.0 and r4["certificate"]["L1_certified"], \
        (sv4, r4["covered"], r4["energy_Wh"])
    print("④ P2-02: opt.wx=storm 而 wx_tau=calm 时, seed 按定价口径(calm)判定并接受 → PASS")

    # ---- ⑤ P2-04: 外部 eval_cache 投毒不再被采信 ----
    sp5 = RM.ShipPrediction.from_cv(np.zeros(2), np.zeros(2), [15, 30], c_state="DP")
    sp5.tau_min = 0.0
    sp5.wx_tau = dict(_WX_CALM)
    t5 = _tb_at("T0", 2500.0, 0.0)
    x5 = _xi_diag()
    base5 = BP.lex_column_generation([t5], sp5, p, _WX_CALM, x5, max_stops=1,
                                     verbose=False)
    poison = {(id(sp5), ("T0",), int(h)): dict(feasible=True, E0=0.0, T0=0.0,
                                               h=int(h), M_omega=0.0)
              for h in RM.decision_horizons_of(x5)}
    r5 = BP.lex_column_generation([t5], sp5, p, _WX_CALM, x5, max_stops=1,
                                  verbose=False, eval_cache=poison)
    assert base5["certified_L1_L2"] is True and \
        r5["certified_L1_L2"] == base5["certified_L1_L2"] and \
        abs(r5["total_energy_Wh"] - base5["total_energy_Wh"]) <= 1e-6 and \
        r5["total_energy_Wh"] > 1.0 and \
        abs(float(r5["energy_LP_lb"]) - float(base5["energy_LP_lb"])) <= 1e-6, \
        (base5["total_energy_Wh"], r5["total_energy_Wh"],
         base5["energy_LP_lb"], r5["energy_LP_lb"], r5["certified_L1_L2"])
    print("⑤ P2-04: 伪 feasible/E0=0 的外部 eval_cache 被忽略, 结果与基线逐位一致 → PASS")

    print("suite solver_validation: P0-NEW + P2-01/02/03/04 全部关闭 ✓")


SUITES["solver_validation"] = suite_solver_validation



# =============================================================================
# suite: exact_bpc — 正式按需 Branch-Price-and-Cut + Logic-Based Benders
# =============================================================================
def _bpc_test_column(tids, tau, h, energy, *, signature=None):
    c = dict(
        tids=tuple(str(t) for t in tids), ordered_tids=tuple(str(t) for t in tids),
        tau=float(tau), h=float(h), E_plan_Wh=float(energy),
        E_soc_required_Wh=float(energy), E0=float(energy),
        resource_only_test_column=True)
    if signature is not None:
        c["route_signature"] = tuple(signature)
    return c


def _bpc_turbines(*ids):
    return [SimpleNamespace(tid=str(t), local=np.array([float(i), 0.0]))
            for i, t in enumerate(ids)]


def _bpc_xi():
    return RM._demo_xi_realistic([5, 10, 15, 20, 30], ["直航"])


def _bpc_oracle(columns, turbine_ids, *, K=1, batteries=2, p=None,
                quick=1.0, swap=6.0):
    """Tiny-test-only full subset oracle; never used by formal experiment code."""
    import itertools as _it
    p = M.Params() if p is None else p
    clean = FAC.validate_route_columns(columns)
    resources = [RA._resource_intervals(c, 2.5, p.landing_clear_min, swap,
                                        "interval", 2.5) for c in clean]
    best = (-1, float("inf"), tuple(), None)
    for bits in _it.product((0, 1), repeat=len(clean)):
        selection = tuple(j for j, bit in enumerate(bits) if bit)
        covered = FAC.selected_turbines(clean, selection)
        if len(covered) != len(set(covered)):
            continue
        audit = RA.audit_resource_assignment(
            clean, selection, K, batteries, p.B_use, resources,
            quick, swap, 1, 1, deadline=None)
        if audit.status is not FAC.ResourceAuditStatus.FEASIBLE:
            continue
        cov = len(covered)
        energy = sum(float(clean[j]["E_plan_Wh"]) for j in selection)
        if cov > best[0] or (cov == best[0] and energy < best[1] - 1e-9):
            best = (cov, energy, selection, audit)
    return best


def _bpc_solve(ids, columns, *, limit=10.0, K=1, batteries=2,
               coverage_gap_target_abs=0):
    return BP._solve_fleet_anytime_synthetic_fixture(
        _bpc_turbines(*ids), [], M.Params(), _bpc_xi(), K, 90.0,
        batteries=batteries, max_stops=max((len(c["tids"]) for c in columns), default=1),
        time_limit_s=limit, allow_resource_only_columns=True,
        implicit_test_columns=columns,
        coverage_gap_target_abs=coverage_gap_target_abs,
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0)


def suite_exact_bpc():
    import itertools as _it
    import tempfile as _tempfile

    # P0) Formal physical certificate provenance: a caller-supplied synthetic
    # route universe must never be accepted by the public exact API.  The same
    # route set remains available only through a private algorithmic fixture,
    # whose certificate scope is deliberately non-physical.
    injected = [_bpc_test_column(("A",), 2.5, 5.0, 1.0)]
    try:
        BP.solve_fleet_anytime(
            _bpc_turbines("A"), [], M.Params(), _bpc_xi(), 1, 60.0,
            batteries=1, max_stops=1, time_limit_s=2.0,
            implicit_test_columns=injected)
    except ValueError as exc:
        assert "implicit_test_columns" in str(exc) and "formal physical" in str(exc)
    else:
        raise AssertionError("public formal API accepted a synthetic route universe")
    try:
        BP.solve_fleet_anytime(
            _bpc_turbines("A"), [], M.Params(), _bpc_xi(), 1, 60.0,
            batteries=1, max_stops=1, time_limit_s=2.0,
            allow_resource_only_columns=True)
    except ValueError as exc:
        assert "allow_resource_only_columns" in str(exc) and "formal physical" in str(exc)
    else:
        raise AssertionError("public formal API accepted resource-only column bypass")
    synthetic_scope = BP._solve_fleet_anytime_synthetic_fixture(
        _bpc_turbines("A"), [], M.Params(), _bpc_xi(), 1, 60.0,
        batteries=1, max_stops=1, time_limit_s=5.0,
        allow_resource_only_columns=True, implicit_test_columns=injected,
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0)
    assert synthetic_scope["coverage_incumbent"] == 1
    assert synthetic_scope["lexicographic_optimal"] is True
    assert synthetic_scope["algorithmic_global_certificate"] is True
    assert synthetic_scope["physical_model_global_certificate"] is False
    assert synthetic_scope["global_certificate_available"] is False
    assert synthetic_scope["global_route_space_certificate"] is False
    assert synthetic_scope["route_universe_source"] == "synthetic-test-fixture"
    assert synthetic_scope["route_universe_provenance_certified"] is False
    assert synthetic_scope["pricing_uses_implicit_full_permutation_search"] is False
    physical_empty = BP.solve_fleet_anytime(
        _bpc_turbines("A"), [], M.Params(), _bpc_xi(), 1, 60.0,
        batteries=1, max_stops=1, time_limit_s=5.0,
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0)
    assert physical_empty["coverage_incumbent"] == 0
    assert physical_empty["lexicographic_optimal"] is True
    assert physical_empty["physical_model_global_certificate"] is True
    assert physical_empty["global_certificate_available"] is True
    assert physical_empty["route_universe_source"] == "physical-oracle"
    assert physical_empty["route_universe_provenance_certified"] is True

    coverage_only = BP.solve_fleet_anytime(
        _bpc_turbines("A"), [], M.Params(), _bpc_xi(), 1, 60.0,
        batteries=1, max_stops=1, time_limit_s=5.0,
        solve_scope="coverage-only",
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0)
    assert coverage_only["coverage_optimal"] is True
    assert coverage_only["coverage_global_certificate_available"] is True
    assert coverage_only["coverage_physical_model_certificate"] is True
    assert coverage_only["solve_scope"] == "coverage-only"
    assert coverage_only["energy_optimal"] is False
    assert coverage_only["lexicographic_optimal"] is False
    assert coverage_only["global_certificate_available"] is False
    assert coverage_only["status"] == "coverage_optimal_scope_complete"
    print("⓪c coverage-only 仅签发 Stage-1 physical certificate，不冒充 lexicographic global certificate → PASS")

    # v10 [THM-TGT] fixed-target exact decision: physical YES witness and
    # full-space NO certificate are distinct from lexicographic optimality.
    target_no = BP.solve_fleet_anytime(
        _bpc_turbines("A"), [], M.Params(), _bpc_xi(), 1, 60.0,
        batteries=1, max_stops=1, time_limit_s=5.0,
        solve_scope="coverage-target", coverage_target=1,
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0)
    assert target_no["target_decision"] == "INFEASIBLE", target_no
    assert target_no["target_decision_certified"] is True
    assert target_no["target_infeasible_proven"] is True
    assert target_no["target_coverage_upper_bound"] == 0
    assert target_no["target_feasible_witness_found"] is False
    assert target_no["global_certificate_available"] is False

    _tgt_p = M.apply_uav_profile(M.Params(), "L")
    _tgt_p.time_recourse_mode = "wait_and_speed"
    _tgt_p.speed_adjustable = True
    _tgt_p.validate_contract(formal=False)
    _tgt_tb = [_tb_at("TGT", 100.0, 0.0)]
    _tgt_opt = [_mk_launch(2.5, (0.0, 0.0), _WX_CALM, horizons=(15, 30))]
    _tgt_xi = _xi_diag((15, 30), var=1.0)
    target_yes = BP.solve_fleet_anytime(
        _tgt_tb, _tgt_opt, _tgt_p, _tgt_xi, 1, 60.0,
        batteries=1, max_stops=1, time_limit_s=8.0,
        solve_scope="coverage-target", coverage_target=1,
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0)
    assert target_yes["target_decision"] == "FEASIBLE", target_yes
    assert target_yes["target_decision_certified"] is True
    assert target_yes["target_feasible_proven"] is True
    assert target_yes["target_feasible_witness_found"] is True
    assert target_yes["target_witness_coverage"] == 1
    assert len(target_yes["covered_turbine_ids"]) == 1
    assert target_yes["global_certificate_available"] is False

    # v11 [THM-CU] materialized complete physical route universe: same exact
    # target decision, zero omitted-column pricing calls, and strict context/hash
    # validation.
    _cu = BP.build_certified_route_universe(
        _tgt_tb, _tgt_opt, _tgt_p, _tgt_xi, 60.0,
        max_stops=1, weather_unc=None, kappa_mode="vp_unimodal",
        chance_mode="drcc", budget_gamma=2.0,
        t_launch_min=2.5, landing_clear_min=1.0,
        deck_mode="interval", deck_delta_min=2.5, time_limit_s=8.0)
    assert _cu.complete and len(_cu.columns) >= 1, _cu.stats
    _ok, _why = BP._validate_certified_route_universe(
        _cu, _tgt_tb, _tgt_opt, _tgt_p, _tgt_xi, 60.0,
        max_stops=1, weather_unc=None, kappa_mode="vp_unimodal",
        chance_mode="drcc", budget_gamma=2.0, t_launch_min=2.5,
        landing_clear_min=1.0, deck_mode="interval", deck_delta_min=2.5)
    assert _ok, _why
    target_yes_cu = BP.solve_fleet_anytime(
        _tgt_tb, _tgt_opt, _tgt_p, _tgt_xi, 1, 60.0,
        batteries=1, max_stops=1, time_limit_s=8.0,
        solve_scope="coverage-target", coverage_target=1,
        certified_route_universe=_cu,
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0)
    assert target_yes_cu["target_decision"] == "FEASIBLE", target_yes_cu
    assert target_yes_cu["target_decision_certified"] is True
    assert target_yes_cu["route_space_complete"] is True
    assert target_yes_cu["route_space_materialized"] is True
    assert target_yes_cu["route_universe_source"] == "materialized-complete-physical-oracle"
    assert target_yes_cu["pricing_calls"] == 0
    assert target_yes_cu["exact_pricing_calls"] == 0
    assert target_yes_cu["complete_route_universe_columns_sha256"] == _cu.columns_sha256

    # The acceleration must preserve the ordinary lexicographic optimum, not
    # merely the fixed-target feasibility answer.
    _lex_implicit = BP.solve_fleet_anytime(
        _tgt_tb, _tgt_opt, _tgt_p, _tgt_xi, 1, 60.0,
        batteries=1, max_stops=1, time_limit_s=8.0,
        solve_scope="lexicographic",
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0)
    _lex_cu = BP.solve_fleet_anytime(
        _tgt_tb, _tgt_opt, _tgt_p, _tgt_xi, 1, 60.0,
        batteries=1, max_stops=1, time_limit_s=8.0,
        solve_scope="lexicographic", certified_route_universe=_cu,
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0)
    assert _lex_implicit["global_certificate_available"] is True
    assert _lex_cu["global_certificate_available"] is True
    assert int(_lex_cu["covered"]) == int(_lex_implicit["covered"])
    assert float(_lex_cu["energy_Wh"]).hex() == float(_lex_implicit["energy_Wh"]).hex()
    assert _lex_cu["pricing_calls"] == 0
    assert _lex_cu["route_space_complete"] is True
    assert _lex_cu["route_space_materialized"] is True

    _bad_p = M.apply_uav_profile(M.Params(), "L")
    _bad_p.B_k = float(_bad_p.B_k) + 1.0
    _bad_p.time_recourse_mode = "wait_and_speed"; _bad_p.speed_adjustable = True
    try:
        BP.solve_fleet_anytime(
            _tgt_tb, _tgt_opt, _bad_p, _tgt_xi, 1, 60.0,
            batteries=1, max_stops=1, time_limit_s=3.0,
            solve_scope="coverage-target", coverage_target=1,
            certified_route_universe=_cu,
            energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("changed physical params reused stale certified route universe")
    print("⓪c3 THM-CU 完整路线宇宙与 implicit target 决策一致，零遗漏列定价，上下文变化 fail-closed → PASS")

    from types import SimpleNamespace as _TgtNS
    _proof_ok = _TgtNS(open_nodes=0, resource_audit_complete=True,
                       branching_complete=True, pricing_bound_available=True,
                       farkas_pricing_complete=True)
    assert BP._target_infeasibility_algorithmic_proven(_proof_ok, False)
    for _field in ("resource_audit_complete", "branching_complete",
                   "pricing_bound_available", "farkas_pricing_complete"):
        _bad = _TgtNS(**vars(_proof_ok)); setattr(_bad, _field, False)
        assert not BP._target_infeasibility_algorithmic_proven(_bad, False), _field
    _open = _TgtNS(**vars(_proof_ok)); _open.open_nodes = 1
    assert not BP._target_infeasibility_algorithmic_proven(_open, False)
    assert not BP._target_infeasibility_algorithmic_proven(_proof_ok, True)
    assert not BP._target_infeasibility_algorithmic_proven(None, False)
    print("⓪c2 coverage-target YES=witness / NO=closed Phase-I+BPC proof，缺任一义务 fail-closed → PASS")

    # Formal E1 selection uses coverage lower/upper bounds and resource
    # monotonicity, never observed validation-safe monotonicity.  Kmax has
    # C*(B)=0,1,2,2,2; the B=2..4 plateau is proved by LB(B=2)=UB(B=4)=2.
    e1_rows = []
    for K in (1, 2):
        for B in range(5):
            cov = min(K, B, 2)
            e1_rows.append(dict(
                uav="S", K=K, batteries=B, safe_served=cov,
                per_battery=(None if B == 0 else cov / B),
                coverage_incumbent=cov, coverage_upper_bound=cov,
                covered=cov, coverable_note=3, plan_holds=(True if cov else None),
                global_certificate_available=(True if (K, B) == (2, 2) else False),
                global_route_space_certificate=(True if (K, B) == (2, 2) else False),
                implicit_route_space_certified=(True if (K, B) == (2, 2) else False),
                result_certificate_contract=(BP.RESULT_CERTIFICATE_CONTRACT
                                             if (K, B) == (2, 2) else None),
                formal_proof_contract=(BP.FORMAL_PROOF_CONTRACT
                                       if (K, B) == (2, 2) else None),
                proof_contract_sha256=(BP.FORMAL_PROOF_CONTRACT_SHA256
                                       if (K, B) == (2, 2) else None),
                coverage_global_certificate_available=True,
                study_mode="formal", inventory_energy_kWh=(0.2 * B),
                safe_per_inventory_kWh=(None if B == 0 else cov / (0.2 * B)),
                energy_per_safe=(10.0 if cov else None), max_stops_requested=4,
                stops_cap_spec="4", max_stops_effective=4, stops_cap=4,
                max_stops_observed=1, stops_cap_hit=False))
    e1_sel = S13.e1_select_from_df(pd.DataFrame(e1_rows), frac=0.95, order="BK", patience=2)
    er = e1_sel.iloc[0]
    assert er["plateau_coverage_certified"] == True and int(er["plateau_coverage"]) == 2
    assert er["knee_resource_minimality_certified"] == True
    assert (int(er["knee_K"]), int(er["knee_B"])) == (2, 2)
    assert er["selection_status"] == "eligible" and er["degenerate_knee"] == False
    e1_unc = pd.DataFrame(e1_rows)
    e1_unc.loc[(e1_unc.K == 2) & (e1_unc.batteries == 4), "coverage_upper_bound"] = 3
    er_unc = S13.e1_select_from_df(e1_unc, frac=0.95, order="BK", patience=2).iloc[0]
    assert er_unc["selection_status"] == "uncertified_coverage_plateau"
    e1_need_lex = pd.DataFrame(e1_rows)
    for c in ("global_certificate_available", "global_route_space_certificate", "implicit_route_space_certified"):
        e1_need_lex.loc[(e1_need_lex.K == 2) & (e1_need_lex.batteries == 2), c] = False
    er_need = S13.e1_select_from_df(e1_need_lex, frac=0.95, order="BK", patience=2).iloc[0]
    assert er_need["selection_status"] == "needs_lexicographic_knee_certification"
    print("⓪d formal E1 平台 sandwich、资源单调 knee 最小性与 knee-only lex certification 门禁 → PASS")

    # v10 current hard-cap pattern: two immediate target-8 predecessor NO
    # certificates close the BK knee at (3,8) without rescanning the grid.
    _trows = []
    for _K in (1, 2, 3):
        for _B in range(9):
            _cap = {1: 3, 2: 6, 3: 8}[_K]
            _cov = min(_B, _cap)
            _trows.append(dict(
                uav="Q", K=_K, batteries=_B, safe_served=max(0, _cov-1),
                per_battery=(None if _B == 0 else max(0, _cov-1)/_B),
                coverage_incumbent=_cov, coverage_upper_bound=8,
                covered=_cov, coverable_note=8, plan_holds=False,
                global_certificate_available=False,
                global_route_space_certificate=False,
                implicit_route_space_certified=False,
                coverage_global_certificate_available=(_K == 3 and _B == 8),
                study_mode="formal", inventory_energy_kWh=0.2*_B,
                safe_per_inventory_kWh=None, energy_per_safe=None,
                max_stops_requested=4, stops_cap_spec="4",
                max_stops_effective=4, stops_cap=4,
                max_stops_observed=1, stops_cap_hit=False))
    _tdf = pd.DataFrame(_trows)
    _ts0 = S13.e1_select_from_df(_tdf, frac=0.95, order="BK", patience=2).iloc[0]
    assert _ts0["selection_status"] == "uncertified_resource_knee", dict(_ts0)
    assert S13._e1_target_blockers(_tdf, "Q", _ts0, "BK") == [(3, 7, 8, "B-predecessor")]
    _no = dict(target_decision="INFEASIBLE", target_decision_certified=True,
               target_feasible_proven=False, target_infeasible_proven=True,
               target_certificate_type="full-space-phase1-bpc-infeasibility")
    S13._apply_target_decision_to_frontier(_tdf, "Q", 3, 7, 8, _no)
    assert S13._e1_raw_coverage_interval_record(
        _tdf[(_tdf.K == 2) & (_tdf.batteries == 8)].iloc[0]) == (6, 8)
    _ts1 = S13.e1_select_from_df(_tdf, frac=0.95, order="BK", patience=2).iloc[0]
    assert S13._e1_target_blockers(_tdf, "Q", _ts1, "BK") == [(2, 8, 8, "K-predecessor")]
    S13._apply_target_decision_to_frontier(_tdf, "Q", 2, 8, 8, _no)
    _ts2 = S13.e1_select_from_df(_tdf, frac=0.95, order="BK", patience=2).iloc[0]
    assert _ts2["selection_status"] == "needs_lexicographic_knee_certification", dict(_ts2)
    assert bool(_ts2["knee_resource_minimality_certified"])
    assert (int(_ts2["knee_K"]), int(_ts2["knee_B"])) == (3, 8)
    assert S13._e1_target_blockers(_tdf, "Q", _ts2, "BK") == []
    print("⓪e v10 target-8 predecessor NO 证书关闭资源 knee，NaN refined 列保留旧 rigorous bounds → PASS")

    e2_args = Namespace(study_mode="formal", solver_mode="exact-branch-price-cut",
                        time_limit_s=1800.0, coverage_gap_target_abs=0,
                        energy_gap_target_abs_wh=0.0, energy_gap_target_rel=0.0,
                        pricing_mode="exact-implicit-dfs", e2_discovery_time_limit_s=120.0,
                        e2_certify_time_limit_s=None)
    e2_lo = S13._e2_solver_kwargs(e2_args, 0.2, (0.2, 0.5, 0.8))
    e2_hi = S13._e2_solver_kwargs(e2_args, 0.8, (0.2, 0.5, 0.8))
    assert e2_lo["time_limit_s"] == 120.0 and e2_hi["time_limit_s"] == 1800.0
    assert e2_lo["solve_scope"] == e2_hi["solve_scope"] == "lexicographic"
    e2_df = pd.DataFrame([
        dict(criterion="vp", q=0.8, run_status="ok", holds=True, n_missing_replay=0,
             covered=3, safe_served=3, energy_per_safe=10.0, energy_Wh=30.0,
             study_mode="formal", e1_formal_freeze_verified=True,
             e1_formal_freeze_sha256="a"*64, global_certificate_available=False,
             global_route_space_certificate=False, implicit_route_space_certified=False),
        dict(criterion="cantelli", q=0.8, run_status="ok", holds=True, n_missing_replay=0,
             covered=2, safe_served=2, energy_per_safe=11.0, energy_Wh=22.0,
             study_mode="formal", e1_formal_freeze_verified=True,
             e1_formal_freeze_sha256="a"*64, global_certificate_available=True,
             global_route_space_certificate=True, implicit_route_space_certified=True),
    ])
    e2_pick = S13._select_e2_validation_candidate(e2_df, (0.2, 0.5, 0.8))
    assert e2_pick is not None and e2_pick["criterion"] == "cantelli"
    e2_df.loc[:, ["global_certificate_available", "global_route_space_certificate",
                  "implicit_route_space_certified"]] = False
    assert S13._select_e2_validation_candidate(e2_df, (0.2, 0.5, 0.8)) is None
    print("⓪e formal E2 非最严 q 短预算；最严 q final-test 候选必须完整 lexicographic 证书 → PASS")

    # The paper/code proof contract is machine-readable.  Every advertised
    # theorem ID has concrete code anchors, and the final physical certificate
    # guard is the literal conjunction documented by [THM-LEX].
    assert tuple(k for k, _ in BP.FORMAL_PROOF_CODE_ANCHORS) == BP.FORMAL_PROOF_OBLIGATIONS
    for theorem_id, anchors in BP.FORMAL_PROOF_CODE_ANCHORS:
        assert theorem_id in BP.FORMAL_PROOF_OBLIGATIONS
        assert anchors
        for name in anchors:
            assert hasattr(BP, name), (theorem_id, name)
    guard_true = dict(
        algorithmic_global_certificate=True,
        route_universe_provenance_certified=True,
        mode="exact-branch-price-cut",
        route_semantics_invariance_certified=True,
        future_column_row_ranges_certified=True,
        binary64_model_contract_enforced=True,
        formal_proof_contract_enforced=True)
    assert BP._physical_certificate_guard(**guard_true) is True
    for key in ("algorithmic_global_certificate",
                "route_universe_provenance_certified",
                "route_semantics_invariance_certified",
                "future_column_row_ranges_certified",
                "binary64_model_contract_enforced",
                "formal_proof_contract_enforced"):
        bad = dict(guard_true); bad[key] = False
        assert BP._physical_certificate_guard(**bad) is False, key
    bad = dict(guard_true); bad["mode"] = "research-baseline"
    assert BP._physical_certificate_guard(**bad) is False

    # [THM-LRC] does not require exact stationarity of the binary64 multiplier:
    # any y<=0 (positive raw y is projected to zero) plus a complete box infimum
    # is a valid weak-Lagrangian lower bound.  Check against this one-variable LP.
    c = np.array([-1.0]); lo = np.array([0.0]); hi = np.array([1.0])
    Au = np.array([[1.0]]); bu = np.array([1.0])
    Ae = np.zeros((0, 1)); be = np.zeros(0)
    for raw_y in (-100.0, -0.25, 0.5):
        lb = BP._lagrangian_dual_lower_bound(
            c, lo, hi, Au, bu, Ae, be, np.array([raw_y]), np.zeros(0))
        assert lb is not None and float(lb) <= -1.0, (raw_y, lb)

    print("⓪ synthetic route universe 与 formal physical certificate 已硬隔离；proof-code guard 与 Lagrangian 弱对偶同构 → PASS")

    # 0b) A formal Xi ambiguity rebuilt from train samples must preserve the
    # predictor/timestamp provenance that load_samples already validated.  This
    # protects step13 from turning a valid cv_noleak sample file into
    # predictor='unknown'.  Missing/mismatched provenance must still fail closed.
    with _tempfile.TemporaryDirectory() as _xd:
        _xd = Path(_xd)
        rows = []
        for k, (xe, xn) in enumerate(((1.0, 0.0), (0.0, 1.0))):
            rows.append(dict(
                mmsi="219018788", source_track="track_219018788.csv",
                source_track_id="fixture-track", h_min=5.0, c_state="直航",
                t0_epoch=1000.0 + 600.0 * k, t1_epoch=1300.0 + 600.0 * k,
                xi_e_m=xe, xi_n_m=xn, split="train",
                predictor="cv_noleak",
                predictor_contract=M.XI_PREDICTOR_CONTRACTS["cv_noleak"],
                timestamp_epoch_contract=M.XI_TIMESTAMP_EPOCH_CONTRACT,
                sample_overlap_policy="nonoverlap", purge_min=30.0,
                moments_source="train", valid_for_formal=True))
        good_path = _xd / "xi_samples_caseB.csv"
        pd.DataFrame(rows).to_csv(good_path, index=False)
        formal_df = RP.load_samples(
            good_path, mmsi="219018788", formal=True, expected_split="train")
        rebuilt = RP.ambiguity_from_samples(
            formal_df, [5], ["直航"], min_cell_n=2, formal=True)
        assert rebuilt.predictor == "cv_noleak"
        assert rebuilt.predictor_contract == M.XI_PREDICTOR_CONTRACTS["cv_noleak"]
        assert rebuilt.timestamp_epoch_contract == M.XI_TIMESTAMP_EPOCH_CONTRACT
        assert rebuilt.sample_overlap_policy == "nonoverlap"
        assert rebuilt.moments_source == "train"
        assert rebuilt.valid_for_formal_data is True
        assert rebuilt.formal_validated is True
        assert rebuilt.selected_mmsi == "219018788"
        assert rebuilt.cross_vessel_pooling is False

        # Formal sample ambiguity must reject cross-vessel pooling even when the
        # source file itself has valid predictor/timestamp provenance.
        mixed_rows = rows + [dict(rows[0], mmsi="999999999", t0_epoch=9000.0, t1_epoch=9300.0)]
        mixed_path = _xd / "mixed_mmsi.csv"
        pd.DataFrame(mixed_rows).to_csv(mixed_path, index=False)
        mixed_df = RP.load_samples(mixed_path, mmsi="ALL", formal=True, expected_split="train")
        try:
            RP.ambiguity_from_samples(mixed_df, [5], ["直航"], min_cell_n=2, formal=True)
        except ValueError as exc:
            assert "一个具体 MMSI" in str(exc) or "跨船" in str(exc)
        else:
            raise AssertionError("formal Xi ambiguity accepted cross-vessel mmsi=ALL pooling")

        missing_path = _xd / "missing_predictor.csv"
        pd.DataFrame(rows).drop(columns=["predictor"]).to_csv(missing_path, index=False)
        try:
            RP.load_samples(missing_path, formal=True, expected_split="train")
        except ValueError as exc:
            assert "missing contract columns" in str(exc)
        else:
            raise AssertionError("formal Xi sample loader manufactured missing predictor provenance")

        bad_rows = [dict(r) for r in rows]
        bad_rows[0]["predictor"] = "unknown"
        bad_path = _xd / "bad_predictor.csv"
        pd.DataFrame(bad_rows).to_csv(bad_path, index=False)
        try:
            RP.load_samples(bad_path, formal=True, expected_split="train")
        except ValueError as exc:
            assert "predictor" in str(exc)
        else:
            raise AssertionError("formal Xi sample loader accepted mixed predictor provenance")
    print("⓪b formal train-sample Xi 重建保留 predictor/epoch provenance；缺失或混合合同 fail-closed → PASS")

    # 1) Formal path must not call either complete-route enumeration function.
    columns = [
        _bpc_test_column(("A",), 2.5, 20, 10),
        _bpc_test_column(("B",), 2.5, 20, 11),
        _bpc_test_column(("C",), 30.0, 5, 8),
        _bpc_test_column(("A", "B"), 2.5, 20, 16),
    ]
    assert S13._formal_ondemand_pricing(Namespace(
        study_mode="formal", solver_mode="exact-branch-price-cut")) is True
    assert S13._formal_ondemand_pricing(Namespace(
        study_mode="mechanism", solver_mode="exact-branch-price-cut")) is False
    old_ra_enum = RA.enumerate_discrete_routes
    old_bp_enum = BP.enumerate_discrete_route_columns
    try:
        def _forbidden_enumeration(*_a, **_k):
            raise AssertionError("formal exact BPC called complete route enumeration")
        RA.enumerate_discrete_routes = _forbidden_enumeration
        BP.enumerate_discrete_route_columns = _forbidden_enumeration
        result = _bpc_solve(("A", "B", "C"), columns, limit=15.0)
    finally:
        RA.enumerate_discrete_routes = old_ra_enum
        BP.enumerate_discrete_route_columns = old_bp_enum

    oracle_cov, oracle_energy, _, _ = _bpc_oracle(columns, ("A", "B", "C"))
    assert result["coverage_incumbent"] == oracle_cov == 3, result
    assert abs(float(result["energy_incumbent_Wh"]) - oracle_energy) <= 1e-6, result
    assert result["coverage_gap_abs"] == 0 and result["energy_gap_abs_Wh"] <= 1e-6
    assert result["lexicographic_optimal"] is True
    assert result["generated_columns"] >= 1, result
    assert result["exact_pricing_calls"] >= 1 and result["pricing_complete"] is True
    assert result["resource_cuts_added"] >= 1, result
    assert len(result["covered_turbine_ids"]) == len(set(result["covered_turbine_ids"]))
    assert result["duplicate_turbine_visits"] == []
    print("① 小规模 oracle + 按需列生成 + 资源割 + 零 Gap → PASS")

    # 1b) Exact-pricing oracle comparison.  For two consecutive pricing rounds,
    #      compare the implicit DFS hook against an independently written brute
    #      reduced-cost formula over the complete small route universe.
    p_price = M.Params()
    price_cols = [BP._normalize_exact_column(
        c, p=p_price, t_launch_min=0.0, landing_clear_min=0.0,
        deck_mode="interval", deck_delta_min=2.5) for c in columns]
    rows = [("packing", tid) for tid in ("A", "B", "C")]
    dual = np.array([-0.2, -1.2, -0.4])
    node_price = BP.BranchPriceNode(0, 0, BP.BranchState(), 3.0, "fixture")
    existing = set()
    for _round in range(2):
        priced = BP._exact_pricing_search(
            _bpc_turbines("A", "B", "C"), [], p_price, _bpc_xi(), None, 90.0,
            2, node_price, existing, "coverage", rows, [], dual, np.zeros(0),
            None, BP.PRICING_EPS, 0.0, 0.0, "interval", 2.5,
            implicit_test_columns=columns, batch_size=16)
        brute = []
        for c in price_cols:
            sig = BP._exact_route_signature(c)
            if sig in existing:
                continue
            tids = set(BP._ordered_tids(c))
            rc = -float(len(tids))
            for tid, u in zip(("A", "B", "C"), dual):
                if tid in tids:
                    rc -= float(u)
            brute.append((rc, repr(sig), sig))
        brute.sort()
        assert brute and priced.complete and priced.bound_available
        assert abs(float(priced.best_reduced_value) - float(brute[0][0])) <= 1e-12, \
            (priced.best_reduced_value, brute[0])
        assert float(priced.reduced_value_bound) <= float(brute[0][0])
        existing.add(brute[0][2])
    print("①b exact DFS pricing 与独立 brute-force reduced-cost oracle 连续两轮一致 → PASS")

    # 2) Ordered routes are distinct and the lower-energy order wins stage two.
    order_columns = [
        _bpc_test_column(("A",), 2.5, 20, 20),
        _bpc_test_column(("B",), 2.5, 20, 20),
        _bpc_test_column(("A", "B"), 2.5, 20, 15),
        _bpc_test_column(("B", "A"), 2.5, 20, 12),
    ]
    ab = BP._normalize_exact_column(order_columns[2], p=M.Params(),
                                    t_launch_min=2.5, landing_clear_min=1.0,
                                    deck_mode="interval", deck_delta_min=2.5)
    ba = BP._normalize_exact_column(order_columns[3], p=M.Params(),
                                    t_launch_min=2.5, landing_clear_min=1.0,
                                    deck_mode="interval", deck_delta_min=2.5)
    assert BP._exact_route_signature(ab) != BP._exact_route_signature(ba)
    order_result = _bpc_solve(("A", "B"), order_columns, limit=10.0)
    assert order_result["coverage_incumbent"] == 2
    assert abs(order_result["energy_incumbent_Wh"] - 12.0) <= 1e-6
    assert tuple(order_result["chosen"][0]["ordered_tids"]) == ("B", "A")
    print("② A→B 与 B→A 保留独立身份，能耗层选择正确顺序 → PASS")

    # 3) Duplicate turbine semantics agree across validator/master/result.
    duplicate_columns = [
        _bpc_test_column(("A",), 2.5, 5, 1),
        _bpc_test_column(("A",), 20.0, 5, 1),
    ]
    dup_result = _bpc_solve(("A",), duplicate_columns, limit=8.0)
    assert dup_result["coverage_incumbent"] == 1
    assert dup_result["duplicate_turbine_visits"] == []
    try:
        BP._normalize_exact_column(_bpc_test_column(("A", "A"), 2.5, 5, 1),
                                   p=M.Params(), t_launch_min=2.5,
                                   landing_clear_min=1.0, deck_mode="interval",
                                   deck_delta_min=2.5)
    except ValueError:
        pass
    else:
        raise AssertionError("single-column repeated turbine was accepted")
    print("③ 风机互斥 set-packing 与单列唯一性 → PASS")

    # 4) Farkas-equivalent Phase-I pricing restores a branch node missing columns.
    p = M.Params()
    implicit = [_bpc_test_column(("A",), 2.5, 5, 3)]
    archive, sig = [], {}
    stage = BP._solve_branch_price_stage(
        stage="coverage", turbines=_bpc_turbines("A"), launch_opts=[], p=p,
        xi_amb=_bpc_xi(), K=1, batteries=1, T_min=60.0, max_stops=1,
        weather_unc=None, deadline=time.monotonic() + 8.0,
        archive=archive, signature_to_index=sig, no_good_cuts=[],
        coverage_target=None, implicit_test_columns=implicit,
        root_branch=BP.BranchState(required_turbines=frozenset({"A"})))
    assert stage.coverage_incumbent == 1 and stage.optimal
    assert stage.farkas_pricing_complete and stage.generated_columns == 1
    print("④ Phase-I/Farkas 定价恢复缺列节点，不误剪枝 → PASS")

    # 4b) A reduced cost inside the ordinary 1e-6 pricing tolerance is
    #      NOT enough to prove Phase-I infeasibility.  The full elastic lower
    #      bound must remain strictly positive after the |I|*delta correction.
    phase_fixture = BP.RestrictedMasterResult(
        "optimal", np.zeros(0), 0.0, 4.0e-7, np.zeros(0), np.array([3.0e-7]),
        [], [], [("required_service", "A")], np.zeros((0, 0)), np.zeros(0),
        np.zeros((1, 0)), np.ones(1), np.zeros(0), phase_one_value=4.0e-7)
    tiny_negative = BP.PricingSearchResult(
        [], True, -3.0e-7, -3.0e-7, True, 1, 1, "exact-pricing-closed")
    proved, p1_lb = BP._phase_one_infeasibility_proven(
        phase_fixture, tiny_negative, 2, BP.ART_TOL)
    assert proved is False and p1_lb < 0.0, (proved, p1_lb)

    # Farkas pricing itself must retain a genuinely negative column even when
    # |rc| < ordinary PRICING_EPS; otherwise the solver could stall before
    # restoring a feasible restricted master.
    farkas_price = BP._exact_pricing_search(
        _bpc_turbines("A"), [], p, _bpc_xi(), None, 60.0, 1,
        BP.BranchPriceNode(0, 0, BP.BranchState(), 1.0, "fixture"), set(),
        "farkas", [], [("required_service", "A")], np.zeros(0),
        np.array([3.0e-7]), time.monotonic() + 5.0, BP.PRICING_EPS,
        2.5, 1.0, "interval", 2.5, implicit_test_columns=implicit, batch_size=4)
    assert farkas_price.complete is True
    assert farkas_price.best_reduced_value is not None
    assert -BP.PRICING_EPS < farkas_price.best_reduced_value < 0.0
    assert len(farkas_price.columns) == 1, farkas_price

    # Integration guard: complete pricing with the tiny negative bound but no
    # addable column must leave the node open/fail-closed, never prune it as
    # "infeasible".  This reproduces the audited tolerance counterexample.
    old_rmp = BP._solve_restricted_master
    old_p1 = BP._solve_elastic_phase_one
    old_price = BP._exact_pricing_search
    try:
        infeasible_master = BP.RestrictedMasterResult(
            "infeasible", None, None, None, None, None, [], [],
            [("required_service", "A")], np.zeros((0, 0)), np.zeros(0),
            np.zeros((1, 0)), np.ones(1), np.zeros(0))
        BP._solve_restricted_master = lambda *a, **k: infeasible_master
        BP._solve_elastic_phase_one = lambda *a, **k: phase_fixture
        BP._exact_pricing_search = lambda *a, **k: tiny_negative
        unresolved = BP._solve_branch_price_stage(
            stage="coverage", turbines=_bpc_turbines("A", "B"), launch_opts=[], p=p,
            xi_amb=_bpc_xi(), K=1, batteries=1, T_min=60.0, max_stops=1,
            weather_unc=None, deadline=time.monotonic() + 5.0, archive=[],
            signature_to_index={}, no_good_cuts=[], coverage_target=None,
            implicit_test_columns=[])
    finally:
        BP._solve_restricted_master = old_rmp
        BP._solve_elastic_phase_one = old_p1
        BP._exact_pricing_search = old_price
    assert unresolved.optimal is False
    assert unresolved.open_nodes == 1
    assert unresolved.farkas_pricing_complete is False
    assert unresolved.termination_reason == "farkas-full-space-infeasibility-unproven"
    print("④b Phase-I 容差反例：只按完整空间人工目标安全下界剪枝 → PASS")

    # 4c) If ordinary RMP says infeasible but elastic Phase-I returns only a
    #      tiny positive artificial value at the feasibility tolerance, retrying
    #      the identical master can loop forever with deadline=None.  This is a
    #      numerical ambiguity, never a proof of infeasibility.
    tiny_phase = BP.RestrictedMasterResult(
        "optimal", np.zeros(0), 1.0e-9, 0.0, np.zeros(0), np.zeros(1),
        [], [], [("required_service", "A")], np.zeros((0, 0)), np.zeros(0),
        np.zeros((1, 0)), np.ones(1), np.zeros(0), phase_one_value=1.0e-9)
    old_rmp = BP._solve_restricted_master
    old_p1 = BP._solve_elastic_phase_one
    old_price = BP._exact_pricing_search
    try:
        BP._solve_restricted_master = lambda *a, **k: infeasible_master
        BP._solve_elastic_phase_one = lambda *a, **k: tiny_phase
        def _pricing_must_not_run(*a, **k):
            raise AssertionError("pricing must not run on unresolved phase-I numeric ambiguity")
        BP._exact_pricing_search = _pricing_must_not_run
        phase_amb = BP._solve_branch_price_stage(
            stage="coverage", turbines=_bpc_turbines("A"), launch_opts=[], p=p,
            xi_amb=_bpc_xi(), K=1, batteries=1, T_min=60.0, max_stops=1,
            weather_unc=None, deadline=None, archive=[], signature_to_index={},
            no_good_cuts=[], coverage_target=None, implicit_test_columns=[])
    finally:
        BP._solve_restricted_master = old_rmp
        BP._solve_elastic_phase_one = old_p1
        BP._exact_pricing_search = old_price
    assert phase_amb.optimal is False and phase_amb.open_nodes == 1
    assert phase_amb.termination_reason == "phase-one-numeric-feasibility-ambiguity"
    assert phase_amb.farkas_pricing_complete is False
    print("④c RMP/Phase-I 数值边界冲突 fail-closed；deadline=None 也不重复相同主问题 → PASS")

    # 5) Pricing-timeout bound correction and no-bound fallback.
    dummy_master = BP.RestrictedMasterResult(
        "optimal", np.zeros(0), -3.2, -3.2, np.zeros(0), np.zeros(0),
        [], [], [], np.zeros((0, 0)), np.zeros(0), np.zeros((0, 0)),
        np.zeros(0), np.zeros(0))
    node = BP.BranchPriceNode(0, 0, BP.BranchState(), 10.0, "fixture")
    pricing = BP.PricingSearchResult([], False, None, -0.05, True, 0, 0, "timeout")
    ub, src = BP._safe_node_bound_from_pricing(
        dummy_master, pricing, "coverage", 4, tuple(str(i) for i in range(10)), node)
    assert ub == 3 and src == "rmp-lagrangian-plus-pricing-bound", (ub, src)
    dummy_master.dual_lower_bound = 100.0
    lb, src = BP._safe_node_bound_from_pricing(
        dummy_master, pricing, "energy", 4, tuple(str(i) for i in range(10)), node)
    assert abs(lb - 99.8) <= 1e-9 and src == "rmp-lagrangian-plus-pricing-bound"
    no_bound = BP.PricingSearchResult([], False, None, None, False, 0, 0, "no-bound")
    ub2, src2 = BP._safe_node_bound_from_pricing(
        dummy_master, no_bound, "coverage", 4, ("A", "B", "C"), node)
    assert ub2 == 3 and src2 == "trivial-node-allowed-turbine-bound"
    print("⑤ 定价超时安全修正与无 bound 平凡回退 → PASS")

    # 5b) Interrupted exact pricing may safely close a node only through the
    #      rigorous full-space bound.  If that bound does not dominate the
    #      incumbent, the node must remain open.
    old_price = BP._exact_pricing_search
    try:
        BP._exact_pricing_search = lambda *a, **k: BP.PricingSearchResult(
            [], False, None, 0.0, True, 0, 0, "synthetic-pricing-interrupt")
        safely_closed = BP._solve_branch_price_stage(
            stage="coverage", turbines=_bpc_turbines("A"), launch_opts=[], p=p,
            xi_amb=_bpc_xi(), K=1, batteries=1, T_min=60.0, max_stops=1,
            weather_unc=None, deadline=None, archive=[], signature_to_index={},
            no_good_cuts=[], coverage_target=None, implicit_test_columns=[])
        BP._exact_pricing_search = lambda *a, **k: BP.PricingSearchResult(
            [], False, None, -1.0, True, 0, 0, "synthetic-pricing-interrupt")
        stays_open = BP._solve_branch_price_stage(
            stage="coverage", turbines=_bpc_turbines("A"), launch_opts=[], p=p,
            xi_amb=_bpc_xi(), K=1, batteries=1, T_min=60.0, max_stops=1,
            weather_unc=None, deadline=None, archive=[], signature_to_index={},
            no_good_cuts=[], coverage_target=None, implicit_test_columns=[])
    finally:
        BP._exact_pricing_search = old_price
    assert safely_closed.optimal is True and safely_closed.open_nodes == 0
    assert safely_closed.pricing_complete is False and safely_closed.pricing_bound_available is True
    assert stays_open.optimal is False and stays_open.open_nodes == 1
    assert stays_open.termination_reason == "synthetic-pricing-interrupt"
    print("⑤b pricing 中断仅由严格 full-space bound 安全闭点；界不足时保持 open → PASS")

    # 5c) Solver-near-integer vectors are discovery output, not integer
    #      certificates.  Recheck the rounded 0/1 pattern exactly over the
    #      binary64 matrix/RHS payload before accepting it.
    coeff = np.nextafter(1.0, np.inf)
    near_master = BP.RestrictedMasterResult(
        "optimal", np.array([1.0 - 5e-8]), 0.0, 0.0, np.zeros(1), np.zeros(0),
        [0], [("pooled_energy", None)], [], np.array([[coeff]]), np.array([1.0]),
        np.zeros((0, 1)), np.zeros(0), np.zeros(1))
    assert BP._is_integral(near_master.x)
    assert BP._exact_binary_master_feasible(near_master, np.rint(near_master.x)) is False
    exact_master = BP.RestrictedMasterResult(
        "optimal", np.array([1.0]), 0.0, 0.0, np.zeros(1), np.zeros(1),
        [0], [("packing", "A")], [("required_service", "A")],
        np.array([[1.0]]), np.array([1.0]), np.array([[1.0]]), np.array([1.0]),
        np.zeros(1))
    assert BP._exact_binary_master_feasible(exact_master, np.array([1.0])) is True
    print("⑤c rounded 0/1 RMP 模式使用 Fraction(binary64) 无容差复核后才可成为 incumbent → PASS")

    # 6) Completeness/disjointness of service, arc and route branching.
    branch_cols = [
        BP._normalize_exact_column(_bpc_test_column(("A",), 0, 5, 1), p=p,
                                   t_launch_min=2.5, landing_clear_min=1,
                                   deck_mode="interval", deck_delta_min=2.5),
        BP._normalize_exact_column(_bpc_test_column(("B",), 10, 5, 1), p=p,
                                   t_launch_min=2.5, landing_clear_min=1,
                                   deck_mode="interval", deck_delta_min=2.5),
        BP._normalize_exact_column(_bpc_test_column(("A", "B"), 20, 5, 1), p=p,
                                   t_launch_min=2.5, landing_clear_min=1,
                                   deck_mode="interval", deck_delta_min=2.5),
    ]
    feasible = set()
    for bits in _it.product((0, 1), repeat=3):
        chosen = tuple(i for i, b in enumerate(bits) if b)
        tids = [t for j in chosen for t in BP._ordered_tids(branch_cols[j])]
        if len(tids) == len(set(tids)):
            feasible.add(chosen)
    def split(parent, predicate):
        left = {s for s in parent if not predicate(s)}
        right = {s for s in parent if predicate(s)}
        assert left.isdisjoint(right) and left | right == parent
    split(feasible, lambda s: any("A" in BP._ordered_tids(branch_cols[j]) for j in s))
    split(feasible, lambda s: any(("A", "B") in BP._route_arcs(branch_cols[j]) for j in s))
    split(feasible, lambda s: 2 in s)
    print("⑥ 服务/弧/路线变量二分支完备且互斥 → PASS")

    # 7) Resource DFS is tri-state but not downward closed.  Reproduce the
    #    audited counterexample and verify that exact-pattern cuts exclude only
    #    S, while allowing a feasible strict superset T.
    launch = [1, 5, 7, 12, 19, 20]
    clear = [5, 6, 10, 13, 20, 23]
    energy = [4, 4, 3, 4, 6, 5]
    clean = FAC.validate_route_columns([
        dict(tids=(str(j),), tau=float(launch[j]), h=float(clear[j] - launch[j]),
             E_plan_Wh=float(energy[j]), E_soc_required_Wh=float(energy[j]))
        for j in range(6)
    ])
    resource_map = {
        j: dict(launch_start_min=float(launch[j]), recovery_min=float(clear[j]),
                clear_end_min=float(clear[j]), deck=[])
        for j in range(6)
    }
    S = (0, 1, 3, 4, 5)
    T = (0, 1, 2, 3, 4, 5)
    infeasible_audit = RA.audit_resource_assignment(
        clean, S, 2, 3, 9.0, resource_map, 2, 2, 1, 2, deadline=None)
    feasible_superset = RA.audit_resource_assignment(
        clean, T, 2, 3, 9.0, resource_map, 2, 2, 1, 2, deadline=None)
    timeout_audit = RA.audit_resource_assignment(
        clean, (0,), 2, 3, 9.0, resource_map, 2, 2, 1, 2,
        deadline=time.monotonic() - 1)
    assert infeasible_audit.status is FAC.ResourceAuditStatus.INFEASIBLE_PROVEN
    assert feasible_superset.status is FAC.ResourceAuditStatus.FEASIBLE
    assert timeout_audit.status is FAC.ResourceAuditStatus.UNKNOWN_TIMEOUT

    # Fixed-pool exact pattern: +1 on S, -1 outside S.
    xS = np.zeros(6); xS[list(S)] = 1
    xT = np.ones(6)
    xSub = np.zeros(6); xSub[[0, 1, 3, 4]] = 1
    tids = [str(j) for j in range(6)]
    assert FAC.validate_selected_solution(xS, clean, tids, [], [S]) is None
    assert FAC.validate_selected_solution(xT, clean, tids, [], [S]) is not None
    assert FAC.validate_selected_solution(xSub, clean, tids, [], [S]) is not None
    fixed_master = FAC.solve_binary_master_scipy(
        clean, tids, [], [S], phase="coverage", deadline=None)
    assert fixed_master.optimal and tuple(np.flatnonzero(fixed_master.x > 0.5)) == T

    # Column-generation row has the same semantics for current and future columns.
    normalized = [BP._normalize_exact_column(
        c, p=None, t_launch_min=0.0, landing_clear_min=0.0,
        deck_mode="interval", deck_delta_min=2.5) for c in clean]
    sig_cut = frozenset(BP._exact_route_signature(normalized[j]) for j in S)
    coeff = [BP._row_coefficient(c, ("resource_pattern", sig_cut))
             for c in normalized]
    assert sum(coeff[j] for j in S) > len(S) - 1
    assert sum(coeff[j] for j in T) <= len(S) - 1
    for bits in _it.product((0, 1), repeat=6):
        lhs = sum(coeff[j] * bits[j] for j in range(6))
        selected_bits = tuple(j for j, bit in enumerate(bits) if bit)
        assert (lhs > len(S) - 1 + 1e-12) == (selected_bits == S)
    # Interrupted-pricing fallback must include the possible -1 cut coefficient.
    universal_lb = BP._universal_pricing_lower_bound(
        "coverage", 1, [("resource_pattern", sig_cut)], [], [-2.0], [])
    assert universal_lb <= -3.0
    assert BP._column_reduced_cost(
        normalized[2], "farkas", [("resource_pattern", sig_cut)], [],
        [-2.0], []) < 0.0

    # A current RMP can be infeasible because the exact pattern is excluded,
    # while an implicit route outside S restores feasibility.  Phase-I pricing
    # must see the future route's -1 coefficient and generate it.
    p_pattern = M.Params()
    singleton_raw = [
        _bpc_test_column((tid,), 2.5 + 10.0 * idx, 2.0, 1.0)
        for idx, tid in enumerate(("A", "B", "C", "D", "E"))
    ]
    pattern_archive = [BP._normalize_exact_column(
        c, p=p_pattern, t_launch_min=0.0, landing_clear_min=0.0,
        deck_mode="interval", deck_delta_min=2.5) for c in singleton_raw]
    pattern_sig = {BP._exact_route_signature(c): j
                   for j, c in enumerate(pattern_archive)}
    pattern_cut = frozenset(pattern_sig)
    restored = BP._solve_branch_price_stage(
        stage="coverage", turbines=_bpc_turbines("A", "B", "C", "D", "E"),
        launch_opts=[], p=p_pattern, xi_amb=_bpc_xi(), K=5, batteries=5,
        T_min=90.0, max_stops=2, weather_unc=None,
        deadline=time.monotonic() + 8.0, archive=pattern_archive,
        signature_to_index=pattern_sig, no_good_cuts=[pattern_cut],
        coverage_target=None,
        implicit_test_columns=[_bpc_test_column(("A", "B"), 2.5, 2.0, 2.0)],
        root_branch=BP.BranchState(required_turbines=frozenset({"A", "B", "C", "D", "E"})),
        t_launch_min=0.0, landing_clear_min=0.0)
    assert restored.optimal and restored.coverage_incumbent == 5
    assert restored.farkas_pricing_complete and restored.generated_columns == 1
    assert any(BP._ordered_tids(c) == ("A", "B") for c in pattern_archive)
    print("⑦ 资源 DFS 非向下封闭反例复现；精确整数模式割只排除 S，且 Phase-I 可生成 -1 系数新列恢复 RMP → PASS")

    # 7b) Future-column master-row coefficient ranges are an explicit
    # certificate registry.  Unknown signed rows fail closed instead of being
    # silently omitted from a universal reduced-cost bound.
    assert BP._future_column_coefficient_range(
        ("resource_pattern", frozenset()), 3, row_family="inequality") == (-1.0, 1.0)
    assert BP._future_column_coefficient_range(
        ("coverage", None), 3, row_family="equality") == (1.0, 3.0)
    for desc, family in [(("future_signed_cut", None), "inequality"),
                         (("future_required_kind", None), "equality")]:
        try:
            BP._future_column_coefficient_range(desc, 3, row_family=family)
        except ValueError:
            pass
        else:
            raise AssertionError("unregistered future master row did not fail closed")
    try:
        BP._universal_pricing_lower_bound(
            "coverage", 3, [("future_signed_cut", None)], [],
            np.array([-1.0]), np.zeros(0))
    except ValueError:
        pass
    else:
        raise AssertionError("universal pricing bound silently ignored an unknown row")

    # Node-level route-mass bound: only node-allowed turbines can carry route
    # mass because every route is nonempty and forbidden service is a route filter.
    mass_branch = BP.BranchState(forbidden_turbines=frozenset({"B", "D"}))
    assert BP._node_allowed_turbine_bound(("A", "B", "C", "D"), mass_branch) == 2

    # Required arc is an aggregate master equality, never a route filter.  A
    # disjoint C route must remain admissible and be generated alongside AB.
    p_arc = M.Params()
    raw_ab = _bpc_test_column(("A", "B"), 2.5, 5.0, 2.0)
    raw_c = _bpc_test_column(("C",), 20.0, 5.0, 1.0)
    n_ab = BP._normalize_exact_column(raw_ab, p=p_arc, t_launch_min=0.0,
                                      landing_clear_min=0.0,
                                      deck_mode="interval", deck_delta_min=2.5)
    arc_archive = [n_ab]
    arc_sig = {BP._exact_route_signature(n_ab): 0}
    arc_branch = BP.BranchState(required_arcs=frozenset({("A", "B")}))
    n_c_probe = BP._normalize_exact_column(raw_c, p=p_arc, t_launch_min=0.0,
                                           landing_clear_min=0.0,
                                           deck_mode="interval", deck_delta_min=2.5)
    assert BP._column_allowed_at_node(n_c_probe, arc_branch) is True
    arc_stage = BP._solve_branch_price_stage(
        stage="coverage", turbines=_bpc_turbines("A", "B", "C"), launch_opts=[],
        p=p_arc, xi_amb=_bpc_xi(), K=2, batteries=2, T_min=90.0,
        max_stops=2, weather_unc=None, deadline=time.monotonic() + 8.0,
        archive=arc_archive, signature_to_index=arc_sig, no_good_cuts=[],
        coverage_target=None, implicit_test_columns=[raw_ab, raw_c],
        root_branch=arc_branch, t_launch_min=0.0, landing_clear_min=0.0)
    assert arc_stage.optimal and arc_stage.coverage_incumbent == 3, arc_stage
    assert any(BP._ordered_tids(c) == ("C",) for c in arc_archive)
    print("⑦b future-row range fail-closed、节点 M_n、required-arc aggregate equality 集成反例 → PASS")

    # 8) Time-limit anytime contract and monotone deterministic bounds.
    limits = [0.0, 0.05, 12.0]
    timed = [_bpc_solve(("A", "B", "C"), columns, limit=t) for t in limits]
    inc = [r["coverage_incumbent"] for r in timed]
    ubv = [r["coverage_upper_bound"] for r in timed]
    assert all(inc[i] >= inc[i - 1] for i in range(1, len(inc))), inc
    assert all(ubv[i] <= ubv[i - 1] for i in range(1, len(ubv))), ubv
    assert timed[0]["lexicographic_optimal"] is False
    assert timed[0]["coverage_upper_bound"] >= timed[0]["coverage_incumbent"]
    assert timed[0]["resource_cuts_added"] == 0
    assert timed[0]["status"] == "time_limit_feasible", timed[0]
    assert timed[0]["empty_plan_allowed"] is True
    assert timed[0]["energy_incumbent_Wh"] == 0.0
    assert timed[0]["energy_gap_abs_Wh"] is None
    assert timed[0]["global_energy_gap_reason"] == "coverage optimum not proven"
    assert timed[0]["pricing_bound_available"] is False
    assert timed[0]["implicit_route_space_bound_valid"] is True
    assert timed[0]["runtime_s"] <= 1.0, timed[0]["runtime_s"]
    assert timed[1]["runtime_s"] <= 1.5, timed[1]["runtime_s"]
    print("⑧ 统一墙钟时间、anytime 空方案 incumbent/bound 与 Gap 单调性 → PASS")

    # 9) Empty-plan semantics: if the implicit finite route space has no feasible
    #    nonempty column, coverage zero and energy zero are a proved lexicographic optimum.
    empty_result = _bpc_solve(("A", "B"), [], limit=8.0)
    assert empty_result["status"] == "lexicographic_optimal", empty_result
    assert empty_result["coverage_incumbent"] == 0
    assert empty_result["coverage_upper_bound"] == 0
    assert empty_result["coverage_gap_abs"] == 0
    assert empty_result["coverage_optimal"] is True
    assert empty_result["energy_incumbent_Wh"] == 0.0
    assert empty_result["energy_lower_bound_Wh"] == 0.0
    assert empty_result["energy_gap_abs_Wh"] == 0.0
    assert empty_result["energy_optimal"] is True
    assert empty_result["lexicographic_optimal"] is True
    assert empty_result["empty_plan_allowed"] is True
    assert empty_result["empty_plan_is_incumbent"] is True
    assert empty_result["implicit_route_space_bound_valid"] is True
    assert empty_result["global_energy_gap_reason"] is None
    assert empty_result["chosen"] == [] and empty_result["covered_turbine_ids"] == []
    print("⑨ 无非空可行列时，空方案覆盖0/能耗0被严格证明为词典序最优 → PASS")

    # 10) Canonical route semantics are immutable.  Exact duplicates may be
    # deduplicated, but the same signature with different objective/resource
    # payload is a model inconsistency, not a dominance opportunity.
    p_dup = M.Params()
    dup_a = BP._normalize_exact_column(
        _bpc_test_column(("A", "B"), 2.5, 20, 10.0), p=p_dup,
        t_launch_min=0.0, landing_clear_min=0.0,
        deck_mode="interval", deck_delta_min=2.5)
    dup_same = BP._normalize_exact_column(
        _bpc_test_column(("A", "B"), 2.5, 20, 10.0), p=p_dup,
        t_launch_min=0.0, landing_clear_min=0.0,
        deck_mode="interval", deck_delta_min=2.5)
    dup_lower = BP._normalize_exact_column(
        _bpc_test_column(("A", "B"), 2.5, 20, 9.999999999), p=p_dup,
        t_launch_min=0.0, landing_clear_min=0.0,
        deck_mode="interval", deck_delta_min=2.5)
    ar = []; sm = {}
    assert BP._add_columns(ar, sm, [dup_a]) == 1
    assert BP._add_columns(ar, sm, [dup_same]) == 0
    try:
        BP._add_columns(ar, sm, [dup_lower])
    except RuntimeError as exc:
        assert "different formal semantics" in str(exc)
    else:
        raise AssertionError("same route signature changed its formal energy semantics")
    dup_interval = dict(dup_a)
    dup_interval["resource_intervals"] = dict(dup_a["resource_intervals"])
    dup_interval["resource_intervals"]["active"] = (
        dup_interval["resource_intervals"]["active"][0],
        np.nextafter(dup_interval["resource_intervals"]["active"][1], np.inf))
    try:
        BP._add_columns(ar, sm, [dup_interval])
    except RuntimeError:
        pass
    else:
        raise AssertionError("same route signature changed its resource interval semantics")

    # Route identity and cache keys are binary64-exact: two adjacent floating
    # states must not collide, and caller-supplied route_signature metadata is
    # ignored by the formal normalizer.
    tau0 = 1.0
    tau1 = np.nextafter(tau0, math.inf)
    c_sig0 = _bpc_test_column(("A",), tau0, 5.0, 1.0, signature=("forged",))
    c_sig1 = _bpc_test_column(("A",), tau1, 5.0, 1.0, signature=("forged",))
    n_sig0 = BP._normalize_exact_column(
        c_sig0, p=M.Params(), t_launch_min=0.0, landing_clear_min=0.0,
        deck_mode="interval", deck_delta_min=2.5)
    n_sig1 = BP._normalize_exact_column(
        c_sig1, p=M.Params(), t_launch_min=0.0, landing_clear_min=0.0,
        deck_mode="interval", deck_delta_min=2.5)
    assert BP._exact_route_signature(n_sig0) != BP._exact_route_signature(n_sig1)
    assert BP._state_fp(tau0) != BP._state_fp(tau1)
    try:
        BP._normalize_exact_column(
            _bpc_test_column(("A",), 0.0, 5.0, -5e-10),
            p=M.Params(), t_launch_min=0.0, landing_clear_min=0.0,
            deck_mode="interval", deck_delta_min=2.5)
    except ValueError:
        pass
    else:
        raise AssertionError("negative planned energy was tolerance-clamped into the exact model")
    print("⑩ canonical route signature 绑定不可变 binary64 objective/resource 语义；差异表示 fail-closed → PASS")

    # 11) Exact mode rejects unsafe truncation controls and unknown misspelled options.
    for bad_name in ("label_budget", "node_budget", "pricing_column_limit"):
        try:
            BP.solve_fleet_anytime(
                _bpc_turbines("A"), [], M.Params(), _bpc_xi(), 1, 60.0,
                batteries=1, max_stops=1, time_limit_s=1.0,
                solver_mode="exact-branch-price-cut", **{bad_name: 1})
        except ValueError as exc:
            assert bad_name in str(exc)
        else:
            raise AssertionError(f"unsafe exact option {bad_name} was accepted")
    try:
        BP.solve_fleet_anytime(
            _bpc_turbines("A"), [], M.Params(), _bpc_xi(), 1, 60.0,
            batteries=1, max_stops=1, time_limit_s=1.0,
            solver_mode="exact-branch-price-cut", pricing_bacth_size=2)
    except TypeError as exc:
        assert "pricing_bacth_size" in str(exc)
    else:
        raise AssertionError("unknown exact solver option was silently ignored")
    print("⑪ 证书路径拒绝不安全截断和未知参数 → PASS")

    # 11b) The public exact API must bind every model mode before creating
    #      the deadline.  Unknown/ignored values and nonfinite budgets fail
    #      closed, and invalid kappa requests cannot inherit process-global RM.kappa.
    original_public_kappa = RM.kappa
    try:
        RM.kappa = RM.KAPPA_MODES["cantelli"]
        bad_contracts = [
            dict(kappa_mode="nonsense"),
            dict(chance_mode="garbage"),
            dict(deck_mode="garbage"),
            dict(battery_reuse_mode="legacy_count"),
            dict(pool_h_mode="first"),
            dict(solver_mode="garbage"),
            dict(pricing_mode="exact-labeling"),
            dict(solver="gurobi"),
            dict(time_limit_s=float("nan")),
            dict(budget_gamma=float("inf")),
            dict(K=1.5),
            dict(K=0),
            dict(batteries=-1),
            dict(quick_inspection_capacity=1.5),
            dict(quick_inspection_capacity=0),
            dict(swap_station_capacity=0),
        ]
        for bad in bad_contracts:
            kwargs = dict(
                batteries=1, max_stops=1, time_limit_s=1.0,
                solver_mode="exact-branch-price-cut",
                pricing_mode="exact-implicit-dfs",
                )
            kwargs.update(bad)
            # Avoid duplicate positional/keyword K when exercising count validation.
            K_bad = kwargs.pop("K", 1)
            try:
                BP.solve_fleet_anytime(
                    _bpc_turbines("A"), [], M.Params(), _bpc_xi(), K_bad, 60.0,
                    **kwargs)
            except (ValueError, TypeError):
                pass
            else:
                raise AssertionError(f"invalid public exact contract was accepted: {bad}")
            assert RM.kappa is RM.KAPPA_MODES["cantelli"], bad
    finally:
        RM.kappa = original_public_kappa
    print("⑪b 非法风险/甲板/电池/池模式与 NaN/非整数参数在公开入口 fail-closed → PASS")

    # 11c) Every legal kappa mode must be fully determined by the request, not
    #      by a residual module-global RM.kappa.  Exercise both the physical
    #      route predicate and the public formal BPC under four hostile globals.
    p_risk = M.apply_uav_profile(M.Params(), "L")
    p_risk.time_recourse_mode = "wait_and_speed"
    p_risk.speed_adjustable = True
    p_risk.validate_contract(formal=False)
    assert p_risk.speed_adjustable is True
    tb_risk = [_tb_at("KR", 600.0, 0.0)]
    op_risk = [_mk_launch(2.5, (0.0, 0.0), _WX_CALM, horizons=(15, 30))]
    xi_risk = _xi_diag((15, 30), var=100.0)
    route_risk = RM.Route(-1, tb_risk, op_risk[0].ship)
    residuals = [
        lambda _e: 0.0,
        RM.KAPPA_MODES["vp_unimodal"],
        RM.KAPPA_MODES["cantelli"],
        RM.KAPPA_MODES["gaussian"],
    ]
    original_public_kappa = RM.kappa
    try:
        for mode in ("nominal", "cantelli", "vp_unimodal", "gaussian"):
            physical = []
            solves = []
            for hostile in residuals:
                RM.kappa = hostile
                dd = RM.route_feasible_at_h(
                    route_risk, 15, p_risk, op_risk[0].wx, xi_risk,
                    chance_mode="drcc", risk_policy=RM.risk_policy_for_mode(mode))
                physical.append((
                    bool(dd.get("feasible", False)),
                    round(float(dd.get("margin_E", float("nan"))), 9),
                    round(float(dd.get("margin_T", float("nan"))), 9),
                    round(float(dd.get("E_soc_required_Wh", float("nan"))), 9),
                ))
                rr = BP.solve_fleet_anytime(
                    tb_risk, op_risk, p_risk, xi_risk, 1, 60.0,
                    batteries=2, max_stops=1, time_limit_s=4.0,
                    kappa_mode=mode, solver_mode="exact-branch-price-cut",
                    pricing_mode="exact-implicit-dfs")
                solves.append((
                    int(rr["coverage_incumbent"]), int(rr["coverage_upper_bound"]),
                    round(float(rr["energy_incumbent_Wh"]), 9), rr["status"],
                    rr["model_contract_sha256"], rr["risk_policy_contract"],
                ))
            assert len(set(physical)) == 1, (mode, physical)
            assert len(set(solves)) == 1, (mode, solves)
    finally:
        RM.kappa = original_public_kappa
    print("⑪c 合法 κ 模式的物理判定与正式 BPC 对残留全局 RM.kappa 完全不变 → PASS")

    # 11d) Catastrophic cancellation at large dual scales must not invalidate
    #      the omitted-column lower bound.  Reproduce a case where the legacy
    #      point-estimate-minus-1e-9 guard lies ABOVE the exact-real rc, then
    #      verify the outward binary64 interval and complete-pricing bound enclose it.
    from fractions import Fraction as _Fraction
    tids_num = tuple(f"N{i}" for i in range(8))
    col_num = _bpc_test_column(tids_num, 0.0, 10.0, 352.1595567288451)
    col_num["E_soc_required_Wh"] = 357577.06583647744
    d_num = -73170000.0
    v_num = ((float(col_num["E_plan_Wh"])
              + (-d_num) * float(col_num["E_soc_required_Wh"])) / 8.0)
    rc_est, rc_lb, rc_ub = BP._column_reduced_cost_interval(
        col_num, "energy", [("pooled_energy", None)], [("coverage", None)],
        np.array([d_num]), np.array([v_num]))
    _F = _Fraction.from_float
    rc_exact = float(
        _F(float(col_num["E_plan_Wh"]))
        - _F(d_num) * _F(float(col_num["E_soc_required_Wh"]))
        - _F(v_num) * 8)
    assert rc_lb <= rc_exact <= rc_ub, (rc_est, rc_lb, rc_exact, rc_ub)
    assert rc_est - 1e-9 > rc_exact, (rc_est, rc_exact)  # old fixed guard is unsafe here
    p_num = M.Params(B_k=1.0e9)
    priced_num = BP._exact_pricing_search(
        _bpc_turbines(*tids_num), [], p_num, _bpc_xi(), None, 60.0, 8,
        BP.BranchPriceNode(0, 0, BP.BranchState(), 0.0, "numeric-fixture"), set(),
        "energy", [("pooled_energy", None)], [("coverage", None)],
        np.array([d_num]), np.array([v_num]), time.monotonic() + 5.0,
        BP.PRICING_EPS, 2.5, 1.0, "interval", 2.5,
        implicit_test_columns=[col_num], batch_size=4)
    assert priced_num.complete and priced_num.bound_available
    assert float(priced_num.reduced_value_bound) <= rc_exact
    assert priced_num.best_reduced_value is not None

    # The same directed-rounding contract must extend through the RMP
    # Lagrangian bound and the L_RMP + M*delta composition.
    lag_lb = BP._lagrangian_dual_lower_bound(
        np.array([float(col_num["E_plan_Wh"])]), np.array([0.0]), np.array([1.0]),
        np.array([[float(col_num["E_soc_required_Wh"])]]), np.array([0.0]),
        np.array([[8.0]]), np.array([0.0]), np.array([d_num]), np.array([v_num]))
    assert lag_lb is not None and float(lag_lb) <= min(0.0, rc_exact), (lag_lb, rc_exact)

    from types import SimpleNamespace as _NS
    base_num = 3270489238450.902
    delta_num = -0.00130374222322459
    phase_lb = BP._phase_one_full_space_lower_bound(
        _NS(dual_lower_bound=base_num),
        _NS(bound_available=True, reduced_value_bound=delta_num), 8)
    phase_exact = float(_F(base_num) + _F(8.0) * _F(delta_num))
    assert phase_lb is not None and float(phase_lb) <= phase_exact, (phase_lb, phase_exact)
    print("⑪d 大尺度相消约化成本、RMP 拉格朗日界与 L+Mδ 组合均采用 binary64 向外下界 → PASS")

    # 11e) Params fields that select unsupported physical/resource semantics
    #      cannot reach a certified exact solve.
    for field_name, bad_value in (
        ("soc_correction", "garbage"),
        ("battery_energy_mode", "nominal_plan"),
        ("recovery_target_model", "garbage"),
        ("terminal_sensor_error_mode", "garbage"),
        ("escort_mode", "garbage"),
        ("use_zeng", "False"),
        ("B_k", float("nan")),
        ("B_k", -1.0),
        ("v_cr", 0.0),
        ("v_max", -2.0),
        ("zeng_Utip", 0.0),
        ("zeng_v0", 0.0),
        ("zeng_A", 0.0),
        ("zeng_rho", 0.0),
        ("z0", 0.0),
        ("W_max", -1.0),
        ("tau_insp", -5.0),
        ("quick_inspection_capacity", 1.5),
        ("swap_station_capacity", 1.5),
        ("soc_share_lin", 2.0),
        ("soc_share_wind", -0.5),
    ):
        p_bad = M.Params()
        setattr(p_bad, field_name, bad_value)
        try:
            BP.solve_fleet_anytime(
                _bpc_turbines("A"), [], p_bad, _bpc_xi(), 1, 60.0,
                batteries=1, max_stops=1, time_limit_s=1.0,
                solver_mode="exact-branch-price-cut")
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsupported Params.{field_name} was certified")
    # Cross-field physical contracts and pure validation: no hidden mutation.
    p_rel = M.Params()
    p_rel.v_air_min = p_rel.v_air_max + 1.0
    try:
        p_rel.validate_contract()
    except ValueError:
        pass
    else:
        raise AssertionError("invalid airspeed envelope was accepted")
    p_rel = M.Params()
    p_rel.time_recourse_mode = "wait_and_speed"
    before_speed = p_rel.speed_adjustable
    try:
        p_rel.validate_contract()
    except ValueError:
        pass
    else:
        raise AssertionError("inconsistent speed recourse mirror was silently normalized")
    assert p_rel.speed_adjustable is before_speed is False
    p_rel.speed_adjustable = True
    p_rel.validate_contract()
    assert p_rel.speed_adjustable is True

    # 11f) Numeric ambiguity must make finite RMP progress instead of using a
    #      point estimate to re-solve the identical master forever.
    col_amb = _bpc_test_column(("A",), 0.0, 10.0, 0.0)
    sig_amb = BP._exact_route_signature(col_amb)
    d_lo = -1.0e10
    d_hi = np.nextafter(1.0e10, math.inf)
    amb_est, amb_lb, amb_ub = BP._column_reduced_cost_interval(
        col_amb, "energy", [],
        [("required_service", "A"), ("required_route", sig_amb)],
        np.array([]), np.array([d_lo, d_hi]))
    assert amb_est < -BP.PRICING_EPS
    assert amb_lb < -BP.PRICING_EPS <= amb_ub
    priced_amb = BP._exact_pricing_search(
        _bpc_turbines("A"), [], M.Params(B_k=1.0e6), _bpc_xi(), None, 60.0, 1,
        BP.BranchPriceNode(0, 0, BP.BranchState(), 0.0, "ambiguity-fixture"), set(),
        "energy", [],
        [("required_service", "A"), ("required_route", sig_amb)],
        np.array([]), np.array([d_lo, d_hi]), time.monotonic() + 5.0,
        BP.PRICING_EPS, 2.5, 1.0, "interval", 2.5,
        implicit_test_columns=[col_amb], batch_size=4)
    assert priced_amb.complete
    assert priced_amb.columns and BP._exact_route_signature(priced_amb.columns[0]) == sig_amb
    assert priced_amb.termination_reason == "exact-pricing-numeric-ambiguity-progress"
    print("⑪f 数值模糊约化成本通过中性列增补取得有限进展，不再用点估计重复相同 RMP → PASS")

    # 11g) Certificate pricing must use the mathematical zero threshold, not
    #       the ordinary 1e-6 search tolerance.  This is the exact adversarial
    #       counterexample that previously returned 1.0000005 Wh as a false
    #       global optimum although a 1.0 Wh column existed with rc=-5e-7.
    tiny_energy_cols = [
        _bpc_test_column(("A",), 0.0, 5.0, 0.3333335),
        _bpc_test_column(("B",), 10.0, 5.0, 0.3333335),
        _bpc_test_column(("C",), 20.0, 5.0, 0.3333335),
        _bpc_test_column(("A", "B", "C"), 30.0, 5.0, 1.0),
    ]
    p_tiny = M.Params(B_k=1.0e6)
    norm_singletons = [
        BP._normalize_exact_column(
            c, p=p_tiny, t_launch_min=2.5, landing_clear_min=1.0,
            deck_mode="interval", deck_delta_min=2.5)
        for c in tiny_energy_cols[:3]
    ]
    archive_tiny = list(norm_singletons)
    sig_tiny = {BP._exact_route_signature(c): j for j, c in enumerate(archive_tiny)}
    node_tiny = BP.BranchPriceNode(0, 0, BP.BranchState(), 0.0, "tiny-negative-fixture")
    master_tiny = BP._solve_restricted_master(
        archive_tiny, ("A", "B", "C"), node_tiny, "energy", 3,
        [], [], 1, 3.0 * float(p_tiny.B_use), [], time.monotonic() + 5.0)
    assert master_tiny.status == "optimal" and master_tiny.x is not None
    priced_tiny = BP._exact_pricing_search(
        _bpc_turbines("A", "B", "C"), [], p_tiny, _bpc_xi(), None, 60.0, 3,
        node_tiny, set(sig_tiny), "energy", master_tiny.inequality_rows, master_tiny.equality_rows,
        master_tiny.inequality_duals, master_tiny.equality_duals, time.monotonic() + 5.0,
        BP.PRICING_EPS, 2.5, 1.0, "interval", 2.5,
        implicit_test_columns=tiny_energy_cols, batch_size=8)
    assert priced_tiny.complete and priced_tiny.columns, priced_tiny
    assert priced_tiny.best_reduced_value is not None
    assert -BP.PRICING_EPS < priced_tiny.best_reduced_value < 0.0, priced_tiny
    assert any(BP._ordered_tids(c) == ("A", "B", "C") for c in priced_tiny.columns)

    stage_tiny_archive = list(norm_singletons)
    stage_tiny = BP._solve_branch_price_stage(
        stage="energy", turbines=_bpc_turbines("A", "B", "C"), launch_opts=[],
        p=p_tiny, xi_amb=_bpc_xi(), K=1, batteries=3, T_min=60.0, max_stops=3,
        weather_unc=None, deadline=time.monotonic() + 10.0,
        archive=stage_tiny_archive,
        signature_to_index={BP._exact_route_signature(c): j
                            for j, c in enumerate(norm_singletons)},
        no_good_cuts=[], coverage_target=3,
        implicit_test_columns=tiny_energy_cols,
        initial_selection=(0, 1, 2),
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0)
    assert stage_tiny.generated_columns >= 1, stage_tiny
    assert stage_tiny.open_nodes == 0 and stage_tiny.optimal, stage_tiny
    assert abs(float(stage_tiny.incumbent_value) - 1.0) <= 1e-15, stage_tiny
    assert abs(float(stage_tiny.global_bound) - 1.0) <= 1e-15, stage_tiny
    assert stage_tiny.pricing_complete is True and stage_tiny.pricing_search_complete is True
    assert len(stage_tiny.incumbent_selection) == 1
    assert tuple(BP._ordered_tids(
        stage_tiny_archive[stage_tiny.incumbent_selection[0]])) == ("A", "B", "C")
    print("⑪g rc∈(-1e-6,0) 的真实改善列按严格零阈值加入；整数 RMP 不再错误 fathom → PASS")

    # 11h) A positive user gap tolerance is a stopping rule, never an exact
    #       optimality proof.  The root LP of this triangle has a tiny positive
    #       integrality gap, and branching leaves two open nodes when the
    #       1e-6 user target is reached.
    pair_e = 0.9999998 / 1.5
    gap_cols_raw = [
        _bpc_test_column(("A", "B"), 0.0, 5.0, pair_e),
        _bpc_test_column(("B", "C"), 10.0, 5.0, pair_e),
        _bpc_test_column(("A", "C"), 20.0, 5.0, pair_e),
        _bpc_test_column(("A", "B", "C"), 30.0, 5.0, 1.0000005),
    ]
    gap_cols = [
        BP._normalize_exact_column(
            c, p=p_tiny, t_launch_min=2.5, landing_clear_min=1.0,
            deck_mode="interval", deck_delta_min=2.5)
        for c in gap_cols_raw
    ]
    gap_stage = BP._solve_branch_price_stage(
        stage="energy", turbines=_bpc_turbines("A", "B", "C"), launch_opts=[],
        p=p_tiny, xi_amb=_bpc_xi(), K=1, batteries=3, T_min=60.0, max_stops=3,
        weather_unc=None, deadline=time.monotonic() + 10.0,
        archive=list(gap_cols),
        signature_to_index={BP._exact_route_signature(c): j
                            for j, c in enumerate(gap_cols)},
        no_good_cuts=[], coverage_target=3,
        implicit_test_columns=gap_cols_raw,
        initial_selection=(3,),
        energy_gap_target_abs_Wh=1.0e-6, energy_gap_target_rel=0.0)
    assert gap_stage.termination_reason == "energy-gap-target-reached", gap_stage
    assert gap_stage.open_nodes > 0, gap_stage
    assert float(gap_stage.global_bound) < float(gap_stage.incumbent_upper_bound), gap_stage
    assert gap_stage.optimal is False, gap_stage
    assert gap_stage.pricing_complete is False
    assert gap_stage.pricing_search_complete is False
    print("⑪h 正 Gap 达到用户停止目标时只返回 gap-target，不再提升为 exact optimal → PASS")

    # With a zero target the same instance must continue through the complete
    # branch tree and prove the actual integer optimum.
    gap_stage_exact = BP._solve_branch_price_stage(
        stage="energy", turbines=_bpc_turbines("A", "B", "C"), launch_opts=[],
        p=p_tiny, xi_amb=_bpc_xi(), K=1, batteries=3, T_min=60.0, max_stops=3,
        weather_unc=None, deadline=time.monotonic() + 10.0,
        archive=list(gap_cols),
        signature_to_index={BP._exact_route_signature(c): j
                            for j, c in enumerate(gap_cols)},
        no_good_cuts=[], coverage_target=3,
        implicit_test_columns=gap_cols_raw,
        initial_selection=(3,),
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0)
    assert gap_stage_exact.open_nodes == 0 and gap_stage_exact.optimal, gap_stage_exact
    assert gap_stage_exact.termination_reason == "stage-optimum-proven", gap_stage_exact
    assert abs(float(gap_stage_exact.incumbent_value) - 1.0000005) <= 1e-15
    assert float(gap_stage_exact.global_bound) <= float(gap_stage_exact.incumbent_upper_bound)
    print("⑪h-2 同一实例在零 Gap 目标下继续完整分支并证明真实整数最优 → PASS")

    # 11i) A floating LP solver can return an integer RMP point that is only
    #      tolerance-optimal.  If the rigorous LB still permits improvement,
    #      exact BPC must branch on an unfixed x_r rather than stop with an open
    #      integer node.  This deterministic 4-turbine case has a 9e-8 better
    #      integer combination than the first tolerance-optimal RMP point.
    micro_rows = [
        (("A",), 1.0),
        (("B",), 0.9999998),
        (("C",), 0.9999999),
        (("D",), 1.00000001),
        (("A", "B"), 1.9999991),
        (("A", "C"), 2.0),
        (("A", "D"), 2.00000001),
        (("B", "C"), 1.999999999998),
        (("B", "D"), 2.0000000002),
        (("C", "D"), 1.999999999999),
        (("A", "B", "C"), 2.9999991),
        (("A", "B", "D"), 3.000000002),
        (("A", "C", "D"), 3.00000001),
        (("B", "C", "D"), 3.0),
        (("A", "B", "C", "D"), 4.0),
    ]
    micro_cols = [
        _bpc_test_column(ts, 20.0 * j, 5.0, e)
        for j, (ts, e) in enumerate(micro_rows)
    ]
    micro = BP._solve_fleet_anytime_synthetic_fixture(
        _bpc_turbines("A", "B", "C", "D"), [], M.Params(B_k=1.0e8),
        _bpc_xi(), 10, 400.0, batteries=20, max_stops=4,
        time_limit_s=15.0, allow_resource_only_columns=True,
        implicit_test_columns=micro_cols, coverage_gap_target_abs=0,
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0,
        quick_inspection_capacity=10, swap_station_capacity=10)
    micro_exact = sum(
        (_Fraction.from_float(float(c["E_plan_Wh"])) for c in micro["chosen"]),
        _Fraction(0))
    expected_micro = (
        _Fraction.from_float(0.9999999)
        + _Fraction.from_float(1.00000001)
        + _Fraction.from_float(1.9999991))
    assert micro["lexicographic_optimal"] is True, micro
    assert micro_exact == expected_micro, (micro_exact, expected_micro, micro)
    assert int(micro["processed_nodes"]) >= 4, micro
    assert (_Fraction.from_float(float(micro["energy_lower_bound_Wh"]))
            <= micro_exact
            <= _Fraction.from_float(float(micro["energy_incumbent_Wh"]))), micro
    print("⑪i 整数 RMP 数值歧义采用完备 x_r=0/1 兜底分支；严格 LB/UB 包络真最优 → PASS")

    p_hash_a = M.Params(); p_hash_b = M.Params(); p_hash_b.B_k += 1.0
    hash_a = BP.solve_fleet_anytime(
        _bpc_turbines("A"), [], p_hash_a, _bpc_xi(), 1, 60.0, batteries=1,
        max_stops=1, time_limit_s=1.0)["model_contract_sha256"]
    hash_b = BP.solve_fleet_anytime(
        _bpc_turbines("A"), [], p_hash_b, _bpc_xi(), 1, 60.0, batteries=1,
        max_stops=1, time_limit_s=1.0)["model_contract_sha256"]
    assert hash_a != hash_b

    def _contract_hash_probe(T_min_probe=60.0, **extra):
        kw = dict(
            batteries=1, max_stops=1, time_limit_s=0.0,
            solver_mode="exact-branch-price-cut",
            pricing_mode="exact-implicit-dfs",
            )
        kw.update(extra)
        return BP.solve_fleet_anytime(
            _bpc_turbines("A"), [], M.Params(), _bpc_xi(), 1, T_min_probe, **kw)

    hash_base = _contract_hash_probe()
    for label, probe in (
        ("T_min", _contract_hash_probe(T_min_probe=61.0)),
        ("max_stops", _contract_hash_probe(max_stops=2)),
        ("budget_gamma", _contract_hash_probe(budget_gamma=3.0)),
        ("deck_delta_min", _contract_hash_probe(deck_delta_min=3.0)),
    ):
        assert probe["model_contract_sha256"] != hash_base["model_contract_sha256"], label
    gap_probe = _contract_hash_probe(energy_gap_target_abs_Wh=2e-6)
    assert gap_probe["model_contract_sha256"] == hash_base["model_contract_sha256"]
    assert gap_probe["algorithm_contract_sha256"] != hash_base["algorithm_contract_sha256"]
    time_probe = _contract_hash_probe(time_limit_s=1e-5)
    assert time_probe["model_contract_sha256"] == hash_base["model_contract_sha256"]
    assert time_probe["algorithm_contract_sha256"] == hash_base["algorithm_contract_sha256"]
    print("⑪e 模型模式/物理定义域 fail-closed；模型哈希覆盖离散化参数，搜索 Gap 目标只进入算法哈希 → PASS")

    # 11j) Strict resource semantics: no positive tolerance may enlarge the
    # certified integer feasible set in time or SOC dimensions.
    _strict_cols = [
        dict(tids=("A",), ordered_tids=("A",), tau=0.0, h=5.0,
             E_plan_Wh=1.0, E_soc_required_Wh=1.0),
        dict(tids=("B",), ordered_tids=("B",), tau=4.9999999995, h=5.0,
             E_plan_Wh=1.0, E_soc_required_Wh=1.0),
    ]
    _strict_res = [
        dict(deck=(), active=(0.0, 5.0), launch_start_min=0.0,
             launch_min=0.0, recovery_min=5.0, clear_end_min=5.0),
        dict(deck=(), active=(4.9999999995, 9.9999999995),
             launch_start_min=4.9999999995, launch_min=4.9999999995,
             recovery_min=9.9999999995, clear_end_min=9.9999999995),
    ]
    assert RA._halfopen_overlap((0.0, 5.0), (4.9999999995, 9.0))
    assert not RA._halfopen_overlap((0.0, 5.0), (5.0, 9.0))
    _strict_audit = RA.audit_resource_assignment(
        _strict_cols, (0, 1), 1, 2, 100.0, _strict_res,
        0.0, 0.0, 1, 1, deadline=None)
    assert _strict_audit.status is FAC.ResourceAuditStatus.INFEASIBLE_PROVEN, _strict_audit

    _soc_cols = [
        dict(tids=("A",), ordered_tids=("A",), tau=0.0, h=1.0,
             E_plan_Wh=1.0, E_soc_required_Wh=50.00000025),
        dict(tids=("B",), ordered_tids=("B",), tau=2.0, h=1.0,
             E_plan_Wh=1.0, E_soc_required_Wh=50.00000025),
    ]
    _soc_res = [
        dict(deck=(), active=(0.0, 1.0), launch_start_min=0.0,
             launch_min=0.0, recovery_min=1.0, clear_end_min=1.0),
        dict(deck=(), active=(2.0, 3.0), launch_start_min=2.0,
             launch_min=2.0, recovery_min=3.0, clear_end_min=3.0),
    ]
    _soc_audit = RA.audit_resource_assignment(
        _soc_cols, (0, 1), 1, 1, 100.0, _soc_res,
        0.0, 0.0, 1, 1, deadline=None)
    assert _soc_audit.status is FAC.ResourceAuditStatus.INFEASIBLE_PROVEN, _soc_audit
    print("⑪j 资源 exact audit 使用严格半开时间与 Fraction SOC；5e-10 overlap / 5e-7Wh 超容均判不可行 → PASS")

    # 11k) Full formal physical BPC regression for the third-party P0:
    # two singleton routes overlap by 5e-10 min with one UAV, max_stops=1.
    _p_res = M.Params()
    _p_res.tau_insp = 0.0
    _p_res.landing_clear_min = 0.0
    _p_res.quick_inspection_min = 0.0
    _p_res.use_zeng = False
    _wx_res = dict(wind10=0.0, wind_dir_from=270.0, Hs=0.0, Tp=6.0,
                   wave_dir=0.0, ship_heading=0.0)
    _H_res = [5.0]
    def _ship_at(_tau):
        _sp = RM.ShipPrediction.from_cv(
            np.zeros(2), np.zeros(2), _H_res, c_state="DP")
        _sp.tau_min = float(_tau)
        _sp.wx_tau = dict(_wx_res)
        return _sp
    _sp0 = _ship_at(0.0)
    _sp1 = _ship_at(4.9999999995)
    _opts_res = [
        RM.LaunchOption(0.0, _sp0, dict(_wx_res)),
        RM.LaunchOption(4.9999999995, _sp1, dict(_wx_res)),
    ]
    _xi_res = M.XiAmbiguity(
        {(5.0, "DP"): M.XiCell(
            5.0, "DP", 1000, np.zeros(2), np.zeros((2, 2)),
            0.0, 0.0, 0.0)}, _H_res)
    _deck_ts, _active_ts = BP._possible_resource_row_times(
        _opts_res, _xi_res, 10.0, 0.0, "interval", 2.5)
    _tiny_t = 4.9999999995
    assert _tiny_t in _active_ts, (_deck_ts, _active_ts)
    _r0 = dict(ordered_tids=("A",), E_soc_required_Wh=1.0,
               resource_intervals={"deck": (), "active": (0.0, 5.0)})
    _r1 = dict(ordered_tids=("B",), E_soc_required_Wh=1.0,
               resource_intervals={"deck": (), "active": (_tiny_t, _tiny_t + 5.0)})
    assert BP._row_coefficient(_r0, ("active", _tiny_t)) == 1.0
    assert BP._row_coefficient(_r1, ("active", _tiny_t)) == 1.0
    _ta = M.Turbine("A", np.zeros(2), 68.5, 115.0); _ta.local = np.zeros(2)
    _tb = M.Turbine("B", np.zeros(2), 68.5, 115.0); _tb.local = np.zeros(2)
    _phys_res = BP.solve_fleet_anytime(
        [_ta, _tb], _opts_res, _p_res, _xi_res, K=1, T_min=10.0,
        max_stops=1, batteries=2, t_launch_min=0.0,
        landing_clear_min=0.0, t_swap_min=0.0,
        quick_inspection_capacity=1, swap_station_capacity=1,
        time_limit_s=20.0, coverage_gap_target_abs=0,
        energy_gap_target_abs_Wh=0.0, energy_gap_target_rel=0.0,
        kappa_mode="nominal", chance_mode="drcc")
    assert _phys_res["coverage_incumbent"] == _phys_res["coverage_upper_bound"] == 1, _phys_res
    assert _phys_res["lexicographic_optimal"] is True, _phys_res
    assert _phys_res["global_certificate_available"] is True, _phys_res
    assert _phys_res["resource_numeric_contract"] == (
        "binary64-strict-half-open-time-exact-rational-soc")
    print("⑪k 正式物理两-launch 5e-10 overlap 反例：K=1,max_stops=1 的严格覆盖最优=1 → PASS")

    # 11l) The public model fingerprint binds binary64-exact instance data.
    _ht0 = _bpc_turbines("A")
    _ht1 = _bpc_turbines("A")
    _ht0[0].local = np.array([1.0, 0.0])
    _ht1[0].local = np.array([np.nextafter(1.0, np.inf), 0.0])
    _h0 = BP.solve_fleet_anytime(
        _ht0, [], M.Params(), _bpc_xi(), 1, 60.0, batteries=1,
        max_stops=1, time_limit_s=0.0)
    _h1 = BP.solve_fleet_anytime(
        _ht1, [], M.Params(), _bpc_xi(), 1, 60.0, batteries=1,
        max_stops=1, time_limit_s=0.0)
    assert _h0["model_contract_sha256"] != _h1["model_contract_sha256"], (_h0, _h1)
    assert _h0["instance_contract_sha256"] != _h1["instance_contract_sha256"]
    assert _h0["parameter_contract_sha256"] == _h1["parameter_contract_sha256"]
    assert _h0["model_contract_scope"] == "full-finite-model-including-instance-data-binary64-exact"
    print("⑪l 模型证书哈希绑定 binary64-exact turbine/launch/weather/Xi 实例；nextafter 坐标不再碰撞 → PASS")

    # 11m) Strict physical boundary regression for the third-party false certificate:
    #      v_required = v_air_max + 5e-10 must be rejected all the way through
    #      complete implicit pricing, while equality remains feasible.
    _p_air = M.Params()
    _p_air.time_recourse_mode = "wait_and_speed"
    _p_air.speed_adjustable = True
    _p_air.v_air_max = 23.0; _p_air.v_max = 30.0; _p_air.v_cr = 15.0
    _p_air.v_z = 1.0e9; _p_air.z0 = 1.0; _p_air.z_cruise = 2.0
    _p_air.tau_insp = 0.0; _p_air.t_dock_base_s = 0.0; _p_air.dock_gamma = 0.0
    _p_air.use_zeng = False; _p_air.B_k = 1.0e9; _p_air.safe_reserve = 0.1
    _p_air.Hs_op = 1.0e6; _p_air.s_heave_max = 1.0e6
    _p_air.s_roll_max = 1.0e6; _p_air.s_pitch_max = 1.0e6
    _p_air.W_max = 1.0e6; _p_air.w_land_max = 1.0e6
    _wx_air = dict(wind10=0.0, wind_dir_from=270.0, Hs=0.0, Tp=6.0,
                   wave_dir=0.0, ship_heading=0.0)
    _sp_air = RM.ShipPrediction(
        P_launch=np.zeros(2), pred_by_h={10: np.array([1.0, 0.0])}, c_state="DP",
        recovery_state_by_h={10: "DP"},
        recovery_state_source_by_h={10: "declared-noleak-state-predictor"})
    _sp_air.tau_min = 0.0; _sp_air.wx_tau = dict(_wx_air)
    _tb_air = M.Turbine("AIR", np.zeros(2), 1.0, 1.0); _tb_air.local = np.zeros(2)
    _route_air = RM.Route(-1, [_tb_air], _sp_air)
    _xi_air = M.XiAmbiguity({
        (10.0, "DP"): M.XiCell(10.0, "DP", 1000, np.zeros(2),
                                  np.zeros((2, 2)), 0.0, 0.0, 0.0)}, [10.0])
    _probe_air = RM.route_feasible_at_h(
        _route_air, 10, _p_air, _wx_air, _xi_air, chance_mode="drcc",
        risk_policy=RM.risk_policy_for_mode("nominal"))
    _ret_budget = float(_probe_air["return_time_budget_s"])

    def _air_case(required_speed):
        _sp_air.pred_by_h = {10: np.array([float(required_speed) * _ret_budget, 0.0])}
        _diag = RM.route_feasible_at_h(
            _route_air, 10, _p_air, _wx_air, _xi_air, chance_mode="drcc",
            risk_policy=RM.risk_policy_for_mode("nominal"))
        _opt = RM.LaunchOption(0.0, _sp_air, dict(_wx_air))
        _res = BP.solve_fleet_anytime(
            [_tb_air], [_opt], _p_air, _xi_air, K=1, T_min=10.0, batteries=1,
            max_stops=1, t_launch_min=0.0, landing_clear_min=0.0, t_swap_min=0.0,
            quick_inspection_capacity=1, swap_station_capacity=1, time_limit_s=20.0,
            coverage_gap_target_abs=0, energy_gap_target_abs_Wh=0.0,
            energy_gap_target_rel=0.0, kappa_mode="nominal", chance_mode="drcc")
        return _diag, _res

    _eq_diag, _eq_res = _air_case(_p_air.v_air_max)
    assert _eq_diag["feasible"] is True, _eq_diag
    assert _eq_res["coverage_incumbent"] == _eq_res["coverage_upper_bound"] == 1, _eq_res
    assert _eq_res["global_certificate_available"] is True, _eq_res
    _over_diag, _over_res = _air_case(_p_air.v_air_max + 5e-10)
    assert _over_diag["return_required_airspeed_safe_ms"] > _p_air.v_air_max, _over_diag
    assert _over_diag["feasible"] is False, _over_diag
    assert _over_res["coverage_incumbent"] == _over_res["coverage_upper_bound"] == 0, _over_res
    assert _over_res["global_certificate_available"] is True, _over_res
    assert _over_res["physical_numeric_contract"] == RM.FORMAL_PHYSICAL_NUMERIC_CONTRACT
    print("⑪m 空速严格边界端到端：vmax 可行，vmax+5e-10 不可行且完整 BPC 正确证明 coverage*=0 → PASS")

    # 11n) Numeric-contract boundaries: fixed touchdown is strict, tiny eps is
    #      never clipped, and route/weather identity distinguishes one ULP.
    _touch_eq = RM.fixed_touchdown_time_accounting(5.0, 5.0, 0.0)
    _touch_over = RM.fixed_touchdown_time_accounting(5.0, np.nextafter(5.0, np.inf), 0.0)
    assert _touch_eq["nominal_time_failed"] is False
    assert _touch_over["nominal_time_failed"] is True
    _eps_tiny = 1e-14
    assert RM.kappa_cantelli(_eps_tiny) == math.sqrt((1.0 - _eps_tiny) / _eps_tiny)
    assert RM.kappa_cantelli(_eps_tiny) > RM.kappa_cantelli(1e-6)
    try:
        RM.kappa_cantelli(0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("eps=0 was silently clipped instead of rejected")
    _wx_a = {"wind10": 1.0, "Hs": 0.0}
    _wx_b = {"wind10": np.nextafter(1.0, np.inf), "Hs": 0.0}
    assert BP._wx_fp(_wx_a) != BP._wx_fp(_wx_b)
    _sig_a = BP._exact_route_signature(dict(ordered_tids=("A",), tau=0.0, h=10.0, wx=_wx_a))
    _sig_b = BP._exact_route_signature(dict(ordered_tids=("A",), tau=0.0, h=10.0, wx=_wx_b))
    assert _sig_a != _sig_b
    print("⑪n fixed-touchdown/epsilon/天气 ULP 身份均严格，无 tolerance/clipping dead zone → PASS")

    # 11o) The public certificate API must validate Xi mathematics even when
    # callers bypass the CSV loader and construct XiAmbiguity in memory.
    _bad_sigma = np.array([[3.129784, 2.985621e6],
                           [2.985621e6, 3.129782e10]], dtype=float)
    _bad_xi_direct = M.XiAmbiguity(
        {(5.0, "DP"): M.XiCell(5.0, "DP", 50, np.zeros(2), _bad_sigma,
                                  1.0, 1.0, 1.0)}, [5.0])
    try:
        BP.solve_fleet_anytime([], [], M.Params(), _bad_xi_direct, 1, 5.0,
                               batteries=1, max_stops=1, time_limit_s=0.0,
                               )
    except ValueError:
        pass
    else:
        raise AssertionError("formal API accepted direct non-PSD XiAmbiguity")
    _asym = np.array([[1.0, 0.1], [np.nextafter(0.1, np.inf), 1.0]], dtype=float)
    _bad_xi_asym = M.XiAmbiguity(
        {(5.0, "DP"): M.XiCell(5.0, "DP", 50, np.zeros(2), _asym,
                                  1.0, 1.0, 1.0)}, [5.0])
    try:
        BP.solve_fleet_anytime([], [], M.Params(), _bad_xi_asym, 1, 5.0,
                               batteries=1, max_stops=1, time_limit_s=0.0,
                               )
    except ValueError:
        pass
    else:
        raise AssertionError("formal API accepted asymmetric direct Xi covariance")
    print("⑪o formal API 二次验证 Xi 数学合同；绕过 CSV 的 non-PSD/非对称 Xi 仍 fail-closed → PASS")

    # 11p) Real-history weather residuals: forecast must use t<=t0 only, while
    # truth may consume the future historical realization.  Formal moments accept
    # the selected 5..30 subset, reject off-grid h and reject non-PSD covariance.
    _wt = np.array([0.0, 3600.0, 7200.0])
    _wv1 = np.array([0.0, 1.0, 100.0])
    _wv2 = np.array([0.0, 1.0, -100.0])
    _pf1, _ = S7._forecast_backward_linear(3600.0, 3900.0, _wt, _wv1, 5400.0)
    _pf2, _ = S7._forecast_backward_linear(3600.0, 3900.0, _wt, _wv2, 5400.0)
    assert float(_pf1) == float(_pf2), (_pf1, _pf2)
    assert float(S7._truth_interp(3900.0, _wt, _wv1, 5400.0)) != float(
        S7._truth_interp(3900.0, _wt, _wv2, 5400.0))
    import tempfile as _weather_tempfile
    with _weather_tempfile.TemporaryDirectory() as _wd:
        _wd = Path(_wd); _wp = _wd / "weather_moments.csv"
        _rows = []
        for _h in (5, 10, 15, 20, 25, 30):
            _rows.append(dict(
                h_min=float(_h), n=50, wind_bias_e_ms=0.0, wind_bias_n_ms=0.0,
                wind_sigma_ee=1.0, wind_sigma_en=0.0, wind_sigma_nn=1.0,
                wind_speed_bias_ms=0.0, wind_speed_std_ms=1.0, hs_bias_m=0.0, hs_std_m=0.1,
                predictor="weather_speed_primary_coherent_noleak",
                predictor_contract=RM.WEATHER_PREDICTOR_CONTRACTS["weather_speed_primary_coherent_noleak"],
                timestamp_epoch_contract=RM.WEATHER_TIMESTAMP_EPOCH_CONTRACT,
                truth_contract=RM.WEATHER_TRUTH_CONTRACT, weather_data_contract=RM.WEATHER_FORMAL_DATA_CONTRACT,
                moments_source="train", sample_overlap_policy="weather_timeline_global_nonoverlap", purge_min=30.0,
                valid_for_formal=True, weather_source_sha256="a"*64, xi_train_source_sha256="b"*64))
        pd.DataFrame(_rows).to_csv(_wp, index=False)
        _wa = RM.weather_ambiguity_from_moments_csv(_wp, [5,10,15,20,25,30], formal=True)
        assert _wa.formal_eligible and _wa.horizons == [5,10,15,20,25,30]
        _bad = pd.DataFrame(_rows); _bad.loc[0, "h_min"] = np.nextafter(5.0, np.inf)
        _bad.to_csv(_wp, index=False)
        try: RM.weather_ambiguity_from_moments_csv(_wp, [5,10,15,20,25,30], formal=True)
        except ValueError: pass
        else: raise AssertionError("formal weather loader accepted off-grid horizon")
        _bad = pd.DataFrame(_rows); _bad.loc[0, "wind_sigma_ee"] = 0.0; _bad.loc[0, "wind_sigma_en"] = 1.0
        _bad.loc[0, "wind_sigma_nn"] = 0.0; _bad.to_csv(_wp, index=False)
        try: RM.weather_ambiguity_from_moments_csv(_wp, [5,10,15,20,25,30], formal=True)
        except ValueError: pass
        else: raise AssertionError("formal weather loader accepted non-PSD covariance")
    print("⑪p 真实历史天气 no-leak 预测、5..30 子集和 formal weather moments fail-closed → PASS")

    # 12) Every real-file path and every Python path matches the uploaded ZIP baseline.
    expected_files = sorted([
        "README_FOR_AI.md", "doc_algorithm.md", "doc_data.md", "doc_experiments.md",
        "doc_model.md", "doc_params.md", "doc_process.md", "doc_proof.md",
        "doc_related_work.md", "selftest.py", "step10_model_routing.py",
        "step11_algorithm_route_drcc.py", "step12_branch_price.py",
        "step13_experiment_model.py", "step14_experiment_algorithm.py",
        "step15_replay.py", "step16_visualize.py", "step17_paper_figure.py", "step18_diagnose_multistop.py", "step19_diagnose_formal_bottlenecks.py", "step20_preflight_final.py",
        "step1_fetch_ais.py", "step2_match_windfarms.py", "step3_fetch_turbines.py",
        "step4_fetch_wind_era5.py", "step5_fetch_wave_cmems.py",
        "step6_export_tracks.py", "step7_compute_xi.py", "step8_gen_recovery.py",
        "step9_model.py",
    ])
    expected_python = sorted(p for p in expected_files if p.endswith(".py"))
    root = Path(__file__).resolve().parent
    ignored_parts = {"__pycache__", ".pytest_cache"}
    # Source-package topology is a top-level-file contract. Operational worktrees
    # legitimately contain data/, tracks/, weather/, results/, etc.; recursing into
    # those directories would incorrectly classify runtime inputs/outputs as source
    # package files. Keep the strict 30/30 whitelist for direct root files only.
    actual_files = sorted(
        path.name
        for path in root.iterdir()
        if path.is_file() and path.suffix != ".pyc")
    actual_python = sorted(p for p in actual_files if p.endswith(".py"))
    assert actual_files == expected_files, (actual_files, expected_files)
    assert actual_python == expected_python, (actual_python, expected_python)
    print("⑫ 当前交付拓扑全部文件集合（30/30）和 Python 集合（21/21）逐路径相等 → PASS")

    # 12b) Direct module execution must not enter the historical Big-M/research demos.
    import subprocess as _subprocess
    import sys as _sys
    for _module, _marker in (
        ("step12_branch_price.py", "Formal exact solver:"),
        ("step11_algorithm_route_drcc.py", "Formal resource audit:"),
    ):
        _cp = _subprocess.run(
            [_sys.executable, str(root / _module), "--show-entry"],
            cwd=root, text=True, stdout=_subprocess.PIPE, stderr=_subprocess.STDOUT,
            timeout=30)
        assert _cp.returncode == 0, (_module, _cp.stdout[-2000:])
        assert _marker in _cp.stdout, (_module, _cp.stdout[-2000:])
        assert "自检完成。完整 B&P" not in _cp.stdout
    print("⑫b Step11/Step12 直接执行已隔离历史研究主程序，不会误跑旧 Big-M 路径 → PASS")

    # 13) External seed columns are recomputed from current physics; route_order
    #     takes precedence over a sorted legacy tids field and supplied energy is ignored.
    p_seed = M.apply_uav_profile(M.Params(), "L")
    seed_turbines = [_tb_at("A", 1200.0, 0.0), _tb_at("B", 0.0, 1000.0)]
    seed_opt = _mk_launch(2.5, (0.1, 0.0), _WX_CALM, horizons=(15, 30, 45))
    seed_xi = _xi_diag((15, 30, 45), var=100.0)
    tampered_seed = dict(
        tids=("A", "B"), route_order=("B", "A"), tau=2.5, h=30,
        launch_option_index=0, ship=seed_opt.ship, wx=seed_opt.wx,
        E0=0.0, E_plan_Wh=0.0, E_soc_required_Wh=0.0,
        feasible=True)
    original_seed_kappa = RM.kappa
    try:
        RM.kappa = RM.KAPPA_MODES["vp_unimodal"]
        physical_ba = BP._candidate_from_physics(
            0, seed_opt, (seed_turbines[1], seed_turbines[0]), 30,
            p_seed, seed_xi, None, 2.5, 1.0, "interval", 2.5)
        assert physical_ba is not None
        # Deliberately leave the module in the conflicting Cantelli mode.
        RM.kappa = RM.KAPPA_MODES["cantelli"]
        rebuilt = BP._revalidate_seed_column(
            tampered_seed, seed_turbines, [seed_opt], p_seed, seed_xi, None,
            60.0, 2.5, 1.0, "interval", 2.5,
            "vp_unimodal", "drcc", 2.0)
        assert BP._ordered_tids(rebuilt) == ("B", "A")
        assert abs(rebuilt["E_plan_Wh"] - physical_ba["E_plan_Wh"]) <= 1e-9
        assert rebuilt["E_plan_Wh"] > 1.0
        assert RM.kappa is RM.KAPPA_MODES["cantelli"]
        near_grid_seed = dict(tampered_seed)
        near_grid_seed["tau"] = float(seed_opt.tau_min) + 5e-10
        near_grid_seed["h"] = 30.0 + 5e-10
        try:
            BP._revalidate_seed_column(
                near_grid_seed, seed_turbines, [seed_opt], p_seed, seed_xi, None,
                60.0, 2.5, 1.0, "interval", 2.5,
                "vp_unimodal", "drcc", 2.0)
        except ValueError:
            pass
        else:
            raise AssertionError("off-grid seed was tolerance-snapped into the formal route space")
    finally:
        RM.kappa = original_seed_kappa
    seed_result = BP.solve_fleet_anytime(
        seed_turbines, [seed_opt], p_seed, seed_xi, 1, 60.0,
        batteries=2, max_stops=2, time_limit_s=15.0,
        solver_mode="exact-branch-price-cut", pricing_mode="exact-implicit-dfs",
        seed_cols=[tampered_seed])
    assert seed_result["seed_validation"]["accepted_count"] == 1, seed_result
    assert seed_result["seed_validation"]["kappa_mode"] == "vp_unimodal"
    assert seed_result["seed_columns_revalidated"] is True
    assert seed_result["duplicate_turbine_visits"] == []
    # One-shot generators must be materialized once.  The historical bug called
    # list(seed_cols) for metadata and then iterated it again, silently dropping
    # every seed from generator inputs.
    generator_seed_result = BP.solve_fleet_anytime(
        seed_turbines, [seed_opt], p_seed, seed_xi, 1, 60.0,
        batteries=2, max_stops=2, time_limit_s=15.0,
        solver_mode="exact-branch-price-cut", pricing_mode="exact-implicit-dfs",
        seed_cols=(c for c in [tampered_seed]),
        seed_iterator_nonblocking=True)
    gsv = generator_seed_result["seed_validation"]
    assert gsv["input_count"] == gsv["consumed_count"] == gsv["materialized_count"] == 1
    assert gsv["accepted_count"] == gsv["validated_count"] == 1
    assert gsv["materialization_timed_out"] is False
    assert gsv["validation_complete"] is True

    class _CountingSeedIterator:
        def __init__(self, value, sleep_s=0.0):
            self.value = value
            self.sleep_s = float(sleep_s)
            self.calls = 0
            self.done = False
        def __iter__(self):
            return self
        def __next__(self):
            self.calls += 1
            if self.sleep_s:
                time.sleep(self.sleep_s)
            if self.done:
                raise StopIteration
            self.done = True
            return self.value

    zero_iter = _CountingSeedIterator(tampered_seed)
    zero_result = BP.solve_fleet_anytime(
        seed_turbines, [seed_opt], p_seed, seed_xi, 1, 60.0,
        batteries=2, max_stops=2, time_limit_s=0.0,
        solver_mode="exact-branch-price-cut", pricing_mode="exact-implicit-dfs",
        seed_cols=zero_iter, seed_iterator_nonblocking=True)
    zsv = zero_result["seed_validation"]
    assert zero_iter.calls == 0
    assert zsv["consumed_count"] == zsv["materialized_count"] == 0
    assert zsv["materialization_timed_out"] is True
    assert zsv["validation_complete"] is False
    assert zero_result["seed_columns_revalidated"] is False

    slow_iter = _CountingSeedIterator(tampered_seed, sleep_s=0.20)
    slow_started = time.monotonic()
    slow_result = BP.solve_fleet_anytime(
        seed_turbines, [seed_opt], p_seed, seed_xi, 1, 60.0,
        batteries=2, max_stops=2, time_limit_s=0.05,
        solver_mode="exact-branch-price-cut", pricing_mode="exact-implicit-dfs",
        seed_cols=slow_iter)
    slow_elapsed = time.monotonic() - slow_started
    ssv = slow_result["seed_validation"]
    assert slow_iter.calls == 0, "finite-deadline default must not enter arbitrary blocking next()"
    assert slow_elapsed < 0.15, slow_elapsed
    assert ssv["materialization_skipped_unbounded_iterator"] is True
    assert ssv["materialization_complete"] is False
    assert ssv["validation_complete"] is False
    assert slow_result["seed_columns_revalidated"] is False
    print("⑬ seed 一次物化；next 前检查零时限；阻塞迭代器在有限 deadline 下 fail-closed → PASS")

    # 13b) Physical route evaluation receives the same absolute deadline.
    # A deliberately non-cooperative replacement can overrun by one black-box
    # call, but the horizon loop must stop immediately afterwards instead of
    # evaluating every recovery horizon under a fresh/unchecked budget.
    old_route_eval = RM.route_feasible_at_h
    slow_calls = {"n": 0}
    def _slow_noncooperative_route(*args, **kwargs):
        slow_calls["n"] += 1
        time.sleep(0.08)
        return {"feasible": False}
    try:
        RM.route_feasible_at_h = _slow_noncooperative_route
        phys_started = time.monotonic()
        phys_timeout = BP.solve_fleet_anytime(
            seed_turbines[:1], [seed_opt], p_seed, seed_xi, 1, 60.0,
            batteries=2, max_stops=1, time_limit_s=0.02,
            solver_mode="exact-branch-price-cut", pricing_mode="exact-implicit-dfs")
        phys_elapsed = time.monotonic() - phys_started
    finally:
        RM.route_feasible_at_h = old_route_eval
    assert slow_calls["n"] == 1, slow_calls
    assert phys_elapsed < 0.25, phys_elapsed
    assert phys_timeout["termination_reason"] == "global-time-limit-before-root-rmp"
    assert phys_timeout["global_certificate_available"] is False
    assert phys_timeout["wall_clock_deadline_enforcement"] == "cooperative"
    assert phys_timeout["blackbox_hard_interrupt_available"] is False
    print("⑬b 物理链共享 deadline；非合作黑盒最多产生一次调用级超时偏差 → PASS")

    # 14) Abnormal exits cannot fall through to a false gap-target status.
    assert BP._classify_anytime_status(
        lexicographic_optimal=False, coverage_optimal=False,
        coverage_gap_abs=0, coverage_gap_target_abs=0,
        energy_gap_abs=None, energy_gap_pct=None,
        energy_gap_target_abs_Wh=1e-6, energy_gap_target_rel=0.0,
        termination_reason="farkas-phase-time-limit-or-invalid",
        deadline_hit=False) == "solver_error"
    assert BP._classify_anytime_status(
        lexicographic_optimal=False, coverage_optimal=False,
        coverage_gap_abs=2, coverage_gap_target_abs=0,
        energy_gap_abs=None, energy_gap_pct=None,
        energy_gap_target_abs_Wh=1e-6, energy_gap_target_rel=0.0,
        termination_reason="rmp-time_limit",
        deadline_hit=False) == "time_limit_feasible"
    assert BP._classify_anytime_status(
        lexicographic_optimal=False, coverage_optimal=False,
        coverage_gap_abs=1, coverage_gap_target_abs=1,
        energy_gap_abs=None, energy_gap_pct=None,
        energy_gap_target_abs_Wh=1e-6, energy_gap_target_rel=0.0,
        termination_reason="coverage-gap-target-reached",
        deadline_hit=False) == "gap_target_reached"
    assert BP._classify_anytime_status(
        lexicographic_optimal=False, coverage_optimal=True,
        coverage_gap_abs=0, coverage_gap_target_abs=0,
        energy_gap_abs=0.5, energy_gap_pct=0.25,
        energy_gap_target_abs_Wh=1.0, energy_gap_target_rel=0.0,
        termination_reason="energy-gap-target-reached",
        deadline_hit=False) == "coverage_optimal_energy_gap_target_reached"
    print("⑭ 终止状态只由实际最优、Gap 或明确时间条件触发，无异常兜底误报 → PASS")

    # 15) The continuous speed-power envelope is analytical and conservative.
    dense_v = np.linspace(0.0, 23.0, 20001)
    for profile in ("S", "M", "L"):
        p_power = M.apply_uav_profile(M.Params(), profile)
        ub = RM._max_power_on_interval(p_power, 0.0, p_power.v_air_max)
        dense_max = max(M.leg_power(p_power, float(v)) for v in dense_v
                        if float(v) <= float(p_power.v_air_max) + 1e-12)
        endpoint_max = max(M.leg_power(p_power, 0.0),
                           M.leg_power(p_power, p_power.v_air_max))
        assert ub + 1e-10 >= dense_max, (profile, ub, dense_max)
        assert 0.0 <= ub - endpoint_max <= 1e-8 * max(1.0, endpoint_max)
    p_power = M.Params()
    p_power.zeng_P0 *= 3.7; p_power.zeng_Pi *= 0.4
    p_power.zeng_d0 *= 2.2; p_power.power_scale = 1.9
    ub = RM._max_power_on_interval(p_power, 1.7, 21.3, n=3)
    dense_max = max(M.leg_power(p_power, float(v))
                    for v in np.linspace(1.7, 21.3, 30001))
    endpoint_max = max(M.leg_power(p_power, 1.7), M.leg_power(p_power, 21.3))
    assert ub + 1e-10 >= dense_max, (ub, dense_max)
    assert 0.0 <= ub - endpoint_max <= 1e-8 * max(1.0, endpoint_max)
    p_legacy = M.Params(); p_legacy.use_zeng = False
    assert RM._max_power_on_interval(p_legacy, 2.0, 17.0) >= M.leg_power(p_legacy, 17.0)
    p_bad = M.Params(); p_bad.power_scale = -1.0
    try:
        RM._max_power_on_interval(p_bad, 0.0, 23.0)
    except ValueError:
        pass
    else:
        raise AssertionError("nonphysical parameters obtained an uncertified power envelope")
    print("⑮ 连续速度功率采用解析端点最大值证明并对参数变体保持保守，非物理参数 fail-closed → PASS")

    # 16) Result contract and finite-discrete physical certificate scope.
    formal_result = _phys_res
    required = {
        "status", "termination_reason", "runtime_s", "time_limit_s",
        "algorithm", "pricing_method", "branching_complete",
        "farkas_pricing_complete", "coverage_incumbent",
        "coverage_upper_bound", "coverage_gap_abs", "coverage_gap_pct",
        "coverage_optimal", "energy_incumbent_Wh", "energy_lower_bound_Wh",
        "energy_gap_abs_Wh", "energy_gap_pct", "energy_optimal",
        "conditional_energy_gap_pct", "global_energy_gap_reason",
        "lexicographic_optimal", "pricing_complete", "pricing_bound_available",
        "resource_audit_complete", "bound_scope", "bound_source", "open_nodes",
        "processed_nodes", "branch_nodes", "branch_decisions",
        "branch_children_created", "rmp_solves", "phase_one_solves",
        "generated_columns", "columns_accepted", "pricing_calls",
        "exact_pricing_calls", "exact_certification_calls",
        "pricing_candidates", "pricing_nodes", "heuristic_columns",
        "resource_audit_calls", "resource_cuts_added",
        "resource_pattern_cuts_added", "resource_cut_type",
        "resource_cut_superset_assumption", "global_certificate_available",
        "global_route_space_certificate", "implicit_route_space_certified",
        "algorithmic_global_certificate", "algorithmic_route_space_certified",
        "route_universe_source", "route_universe_provenance_certified",
        "physical_model_global_certificate", "route_semantics_invariance_certified",
        "future_column_row_ranges_certified", "binary64_model_contract_enforced",
        "formal_proof_contract_enforced", "formal_proof_contract",
        "formal_proof_obligations", "formal_proof_code_anchors",
        "proof_contract_sha256", "result_certificate_contract",
        "route_semantics_contract", "future_column_row_range_contract", "chosen",
        "covered_turbine_ids", "duplicate_turbine_visits",
    }
    assert required <= set(formal_result), sorted(required - set(formal_result))
    assert formal_result["algorithm"] == "branch-price-and-cut-with-logic-benders"
    assert formal_result["bound_scope"] == "global_discrete_physical_model"
    assert formal_result["finite_discrete_model_only"] is True
    assert formal_result["pricing_non_enumerative"] is False
    assert formal_result["pricing_uses_implicit_full_permutation_search"] is True
    assert formal_result["pricing_dominance"] == "identity-only"
    assert formal_result["resource_cut_type"] == "exact-selected-pattern"
    assert formal_result["resource_cut_superset_assumption"] is False
    assert formal_result["resource_pattern_cuts_added"] == formal_result["resource_cuts_added"]
    assert formal_result["branch_nodes"] == formal_result["processed_nodes"]
    assert formal_result["columns_accepted"] == formal_result["generated_columns"]
    assert formal_result["exact_certification_calls"] == formal_result["exact_pricing_calls"]
    assert formal_result["heuristic_columns"] == 0
    assert formal_result["multi_column_generation"] is True and formal_result["route_pool_reuse"] is True
    assert formal_result["global_certificate_available"] is formal_result["global_route_space_certificate"]
    assert formal_result["implicit_route_space_certified"] is formal_result["global_certificate_available"]
    assert formal_result["algorithmic_global_certificate"] is True
    assert formal_result["physical_model_global_certificate"] is True
    assert formal_result["route_universe_source"] == "physical-oracle"
    assert formal_result["route_universe_provenance_certified"] is True
    assert formal_result["result_certificate_contract"] == BP.RESULT_CERTIFICATE_CONTRACT
    assert formal_result["route_semantics_contract"] == BP.ROUTE_SEMANTICS_CONTRACT
    assert formal_result["future_column_row_range_contract"] == BP.FUTURE_COLUMN_ROW_RANGE_CONTRACT
    assert formal_result["route_semantics_invariance_certified"] is True
    assert formal_result["future_column_row_ranges_certified"] is True
    assert formal_result["binary64_model_contract_enforced"] is True
    assert formal_result["formal_proof_contract_enforced"] is True
    assert formal_result["formal_proof_contract"] == BP.FORMAL_PROOF_CONTRACT
    assert tuple(formal_result["formal_proof_obligations"]) == BP.FORMAL_PROOF_OBLIGATIONS
    assert formal_result["proof_contract_sha256"] == BP.FORMAL_PROOF_CONTRACT_SHA256
    assert [(k, tuple(v)) for k, v in formal_result["formal_proof_code_anchors"]] == list(BP.FORMAL_PROOF_CODE_ANCHORS)
    print("⑯ 正式结果合同、proof-code concordance、统一证书字段和隐式全排列定价披露 → PASS")

    # 17) The advertised exact-MIP option is fail-closed until the black-box
    #     physics/DRCC has a proved equivalent algebraic pricing formulation.
    mip_result = BP.solve_fleet_anytime(
        _bpc_turbines("A"), [], M.Params(), _bpc_xi(), 1, 60.0,
        batteries=1, max_stops=1, time_limit_s=1.0,
        solver_mode="exact-branch-price-cut", pricing_mode="exact-mip")
    assert mip_result["status"] == "solver_error", mip_result
    assert mip_result["pricing_bound_available"] is False
    assert mip_result["implicit_route_space_bound_valid"] is True
    assert mip_result["lexicographic_optimal"] is False
    assert "not-implemented" in mip_result["termination_reason"]
    print("⑰ exact-mip 无严格等价编码时 fail-closed，不冒发证书 → PASS")

    # 18) Plot-field migration regression: generate the existing E1/E2 and
    #     paper algorithm figures from tiny synthetic result tables.  This is
    #     a rendering/interface test only, never a formal business result.
    import tempfile as _tempfile
    import warnings as _warnings
    import step16_visualize as _V16
    import step17_paper_figure as _V17
    with _tempfile.TemporaryDirectory() as _td:
        _td = Path(_td)
        e1_rows = []
        for K1 in (1, 2, 3):
            for B1 in (1, 2, 3, 4):
                safe = min(K1 + B1, 5)
                e1_rows.append(dict(
                    result_contract=S13.RESULT_CONTRACT, uav="S", uav_label="fixture",
                    K=K1, batteries=B1, safe_served=safe, plan_holds=True,
                    per_battery=safe / B1, energy_per_safe=10.0,
                    inventory_energy_kWh=0.5 * B1,
                    safe_per_inventory_kWh=safe / (0.5 * B1),
                    coverage_gap_abs=0, energy_gap_pct=0.0,
                    bound_scope="global_discrete_physical_model"))
        e1_df = pd.DataFrame(e1_rows)
        e1_csv = _td / "e1.csv"
        e1_df.to_csv(e1_csv, index=False, encoding="utf-8-sig")
        e2_rows = []
        for q in (0.2, 0.8):
            for criterion in ("vp", "nominal"):
                e2_rows.append(dict(
                    result_contract=S13.RESULT_CONTRACT, criterion=criterion, q=q,
                    safe_served=(4 if criterion == "vp" else 3), covered=5,
                    max_col_viol=0.02, emp_viol_upper95=0.03,
                    formal_reliability_claim_eligible=True,
                    evidence_scope="confirmatory-purged-disjoint-real-joint-holdout",
                    component_eps=0.05, mission_eps_budget=0.10, holds=True,
                    Hs0=0.2 + q, run_status="ok", coverage_gap_abs=0,
                    energy_gap_pct=0.0, bound_scope="global_discrete_physical_model"))
        e2_df = pd.DataFrame(e2_rows)
        e2_csv = _td / "e2.csv"
        e2_df.to_csv(e2_csv, index=False, encoding="utf-8-sig")
        a1_df = pd.DataFrame([
            dict(result_contract=S13.RESULT_CONTRACT, n_turbines=n1, method=method,
                 covered=(n1 if method == "exact_branch_price_cut" else n1 - 1),
                 gap_to_best=(0 if method == "exact_branch_price_cut" else 1),
                 coverage_gap_abs=(0 if method == "exact_branch_price_cut" else np.nan),
                 energy_gap_pct=(0.0 if method == "exact_branch_price_cut" else np.nan),
                 bound_scope=("global_discrete_physical_model" if method == "exact_branch_price_cut"
                              else "validated_route_pool"))
            for n1 in (4, 6)
            for method in ("research_greedy", "research_restricted_pool",
                           "exact_branch_price_cut")])
        a2_df = pd.DataFrame([dict(
            result_contract=S13.RESULT_CONTRACT, n_turbines=4, dtau_min=5,
            ext_pool=100, t_gurobi_total_s=2.0, t_ours_s=1.0, ours_pool=20,
            runtime_ratio=2.0, exact_coverage_gap_abs=0,
            exact_energy_gap_pct=0.0, exact_bound_scope="global_discrete_physical_model")])
        with _warnings.catch_warnings():
            _warnings.filterwarnings("ignore", message="Glyph .* missing from font")
            _V16.plot_e1(e1_csv, _td / "e1.png", dpi=50)
            _V16.plot_e2(e2_csv, _td / "e2.png", dpi=50)
            _V17.figure1_algorithm(a1_df, a2_df, _td, 50)
        for image in (_td / "e1.png", _td / "e2.png", _td / "figure1.png"):
            assert image.is_file() and image.stat().st_size > 1000, image
    print("⑱ 绘图字段迁移与原业务图入口回归 → PASS")

    # 19) Formal AIS launch-state classification must execute the straight/turn
    #     branch instead of failing before optimization.
    class _StraightTrack:
        t = np.array([0.0, 30.0, 60.0])
        def pos(self, t):
            return np.array([2.0 * float(t), 0.0])
    assert S13._classify_state_noleak(_StraightTrack(), 60.0) == "直航"
    print("⑲ 正式 AIS 直航/转弯分类器可执行且无变量解包错误 → PASS")

    # 20) One canonical certificate source; any conflicting/missing migration is
    #     fail-closed and recorded instead of generating contradictory evidence.
    conflict_true_false = {
        "global_certificate_available": True,
        "global_route_space_certificate": False,
        "implicit_route_space_certified": True,
    }
    legacy_only = {"global_route_space_certificate": True}
    assert S13._global_certificate_flag(conflict_true_false) is False
    assert S13._implicit_route_space_certificate(conflict_true_false) is False
    assert S13._certificate_field_conflict(conflict_true_false) is True
    assert S13._global_certificate_flag(legacy_only) is False
    assert S13._certificate_field_conflict(legacy_only) is True
    normalized = S13._canonical_certificate_fields(conflict_true_false)
    assert normalized == dict(
        global_certificate_available=False,
        global_route_space_certificate=False,
        implicit_route_space_certified=False,
        certificate_field_conflict=True,
        certificate_field_invalid=False)
    import step14_experiment_algorithm as _S14
    csv_fields = _S14._global_gap_fields({
        "global_certificate_available": False,
        "global_route_space_certificate": True,
        "implicit_route_space_certified": True})
    assert csv_fields["certificate_field_conflict"] is True
    assert not any(csv_fields[k] for k in (
        "global_certificate_available", "global_route_space_certificate",
        "implicit_route_space_certified"))
    for bad_cert in (float("nan"), "False", "0", None, object()):
        bad_fields = S13._canonical_certificate_fields({
            "global_certificate_available": bad_cert})
        assert bad_fields["certificate_field_invalid"] is True, bad_fields
        assert bad_fields["global_certificate_available"] is False
        assert bad_fields["global_route_space_certificate"] is False
        assert bad_fields["implicit_route_space_certified"] is False
    string_false = S13._canonical_certificate_fields({
        "global_certificate_available": "False",
        "global_route_space_certificate": "False",
        "implicit_route_space_certified": "False"})
    assert string_false["certificate_field_invalid"] is True
    assert string_false["global_certificate_available"] is False
    assert result["global_certificate_available"] is result["global_route_space_certificate"]
    assert result["global_certificate_available"] is result["implicit_route_space_certified"]
    print("⑳ 三证书字段由唯一 canonical 来源归一化；冲突输入 fail-closed → PASS")

    # 21) An intentionally empty prebuilt list in formal BPC is not an empty
    #     route space, and nonzero coverage never receives a zero-coverage reason.
    rs, rc = S13._route_pool_metadata(
        True, [], {"generated_column_archive_size": 7, "pool_size": 7})
    assert rs == "on_demand_implicit_route_space" and rc == 7
    assert S13._zero_coverage_reason(
        True, [], {"covered": 2, "coverage_optimal": True,
                   "coverage_upper_bound": 2}, 1) is None
    formal_zero = pd.DataFrame([dict(
        uav="Z", K=1, batteries=1, safe_served=0, per_battery=None,
        route_pool_status="on_demand_implicit_route_space", route_pool_count=0,
        plan_holds=None)])
    zsel = S13.e1_select_from_df(formal_zero).iloc[0]
    assert zsel["selection_status"] == "zero_positive_coverage", dict(zsel)
    research_zero = formal_zero.copy()
    research_zero["route_pool_status"] = "empty_all_routes_infeasible"
    assert S13.e1_select_from_df(research_zero).iloc[0]["selection_status"] == "empty_route_pool"
    print("㉑ 按需隐式路线空间元数据不再误报为空池/全部路线不可行 → PASS")

    # 22) Research-only greedy fallback must return its computed uncovered list.
    greedy = RA._greedy_master(
        [("A",)], {"A": [0]}, [1.0], ["preexisting"],
        energy=[1.0], robust=[0.0], slot_of=[0])
    assert greedy["chosen"] == [0]
    assert greedy["uncovered"] == ["preexisting"]
    print("㉒ 研究贪心入口 uncovered 返回值无 NameError → PASS")

    # 23) Exercise the GIF construction path through the first rendered frame.
    #     Saving is replaced by a tiny in-memory writer; all plotting code,
    #     including lane-palette lookup, still executes.
    import step16_visualize as _GIF
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib import animation as _animation
    with _tempfile.TemporaryDirectory() as _gd:
        _gd = Path(_gd)
        detail = _gd / "detail.csv"
        pd.DataFrame([dict(turbines="A")]).to_csv(detail, index=False)
        out_gif = _gd / "fixture.gif"
        turbine = SimpleNamespace(tid="A", local=np.array([100.0, 0.0]))
        class _Track:
            t = np.array([0.0, 60.0])
            P = np.array([[0.0, 0.0], [100.0, 0.0]])
            def duration_sec(self): return 60.0
            def pos(self, t): return np.array([min(max(float(t), 0.0), 60.0) / 60.0 * 100.0, 0.0])
            def vel(self, t): return np.array([100.0 / 60.0, 0.0])
        flight = dict(
            wp=[(0.0, np.array([0.0, 0.0])), (60.0, np.array([100.0, 0.0]))],
            hold=[("A", 20.0, 30.0)], tau=0.0, end=60.0, tids=["A"],
            path=np.array([[0.0, 0.0], [100.0, 0.0]]), uav_id=0,
            battery_group=0, post_service_mode="none_after_last_mission",
            service_start=None, service_end=60.0)
        class _FakeAnimation:
            def __init__(self, fig, update, frames, blit=False):
                update(0)
            def save(self, out, writer=None, dpi=None):
                Path(out).write_bytes(b"GIF89a")
        class _FakeWriter:
            def __init__(self, fps): self.fps = fps
        old_load = _GIF.S13.load_all
        old_track = _GIF._get_track
        old_schedule = _GIF._flight_schedule
        old_font = _GIF._setup_font
        old_anim = _animation.FuncAnimation
        old_writer = _animation.PillowWriter
        try:
            _GIF.S13.load_all = lambda *a, **k: (
                [turbine], pd.DataFrame(), SimpleNamespace(), (0.0, 0.0),
                None, "fixture", None)
            _GIF._get_track = lambda *a, **k: (_Track(), "fixture", False)
            _GIF._flight_schedule = lambda *a, **k: dict(flight)
            _GIF._setup_font = lambda: False
            _animation.FuncAnimation = _FakeAnimation
            _animation.PillowWriter = _FakeWriter
            _GIF.gif_main(Namespace(
                n_turbines=1, farm="fixture", pair_radius=1000.0,
                track_mmsi=None, track_csv=None, window_min=1.0,
                probe=False, detail=detail, out=out_gif, fps=1,
                step_min=1.0, dpi=30))
        finally:
            _GIF.S13.load_all = old_load
            _GIF._get_track = old_track
            _GIF._flight_schedule = old_schedule
            _GIF._setup_font = old_font
            _animation.FuncAnimation = old_anim
            _animation.PillowWriter = old_writer
        assert len(_GIF.PALETTE) >= 2 and out_gif.read_bytes().startswith(b"GIF89a")
    print("㉓ GIF 动画入口完成首帧与配色访问，无 PALETTE NameError → PASS")

    print("suite exact_bpc: 全部 PASS ✓")


SUITES["exact_bpc"] = suite_exact_bpc


# =============================================================================
# 套件编组：子进程隔离、逐套件时限与全量墙钟预算。
#   旧口径 `--suite all` 串行同进程跑全部套件(含 >6min 的 branch/resume), 无时限,
#   卡死即挂起。现改为:
#   - 缺省 `all` = 10 个快速编组(basic/cg_pricing/l1_branch/l2_energy/certificates/
#     counterexamples/mutations/random_oracle/p0_regressions/exact_optimizer), 每组【子进程】运行并各设上限;
#     任一组超时/失败 ⇒ 显式 FAIL(退出码非 0), 不静默挂起; 全量墙钟预算 300s,
#     超出即失败(E-03 验收)。
#   - 慢组(`--suite slow` = branch/e1/resume/certificates_full)显式点名才跑, 同样
#     子进程 + 时限。单套件名仍可直接指定(本进程内运行, 供调试/慢跑)。
#   - 反例/变异/随机 oracle 审计按交付约束收编为本文件套件(不新增 .py 文件)。
GROUPS = {
    "basic":            ["core", "resume_fast"],
    "cg_pricing":       ["bp"],
    "l1_branch":        ["branch_price"],
    "l2_energy":        ["l2_energy"],
    "certificates":     ["certificates"],
    "counterexamples":  ["counterexamples"],
    "mutations":        ["mutations"],
    "random_oracle":    ["random_oracle"],
    "p0_regressions":    ["seed_validation", "solver_validation", "hardening"],
    "exact_optimizer":   ["exact_bpc"],
}

# =============================================================================
# suite: v17.1 formal alignment (coherent weather + Step14 parity + E2 recourse)
# =============================================================================
def suite_v17_alignment():
    import tempfile
    import inspect
    import step14_experiment_algorithm as S14

    # V17.1.5: inline E1_frontier full-lex certification and E1_knee_refine
    # must persist the same proof-contract provenance required by final freeze.
    _cp = S13._e1_certificate_provenance_fields(dict(
        result_certificate_contract=BP.RESULT_CERTIFICATE_CONTRACT,
        formal_proof_contract=BP.FORMAL_PROOF_CONTRACT,
        proof_contract_sha256=BP.FORMAL_PROOF_CONTRACT_SHA256))
    assert _cp["result_certificate_contract"] == BP.RESULT_CERTIFICATE_CONTRACT
    assert _cp["formal_proof_contract"] == BP.FORMAL_PROOF_CONTRACT
    assert _cp["proof_contract_sha256"] == BP.FORMAL_PROOF_CONTRACT_SHA256
    assert S13._e1_global_certificate_with_provenance(dict(
        global_certificate_available=True,
        result_certificate_contract=BP.RESULT_CERTIFICATE_CONTRACT,
        formal_proof_contract=BP.FORMAL_PROOF_CONTRACT,
        proof_contract_sha256=BP.FORMAL_PROOF_CONTRACT_SHA256))
    assert not S13._e1_global_certificate_with_provenance(dict(
        global_certificate_available=True))
    _cell_src = inspect.getsource(S13.E1_frontier)
    assert "**_e1_certificate_provenance_fields(r)" in _cell_src
    _sel_src = inspect.getsource(S13.e1_select_from_df)
    assert "_e1_global_certificate_with_provenance(knee.to_dict())" in _sel_src

    # 1) Coherent weather predictor: positive scalar speed sets magnitude; vector sets direction.
    times = np.array([0.0, 3600.0, 7200.0])
    wind_vec = np.array([[1.0, 0.0], [2.0, 1.0], [9.0, 9.0]])
    wind_speed = np.array([2.0, 3.0, 99.0])
    hs = np.array([0.5, 0.6, 9.0])
    fc, reason = S7._forecast_weather_coherent(
        3600.0, 7200.0, times, wind_vec, wind_speed, hs, 5400.0)
    assert reason is None and fc is not None
    assert fc["wind_speed"] >= 0.0
    assert abs(np.linalg.norm(fc["wind_vec"]) - fc["wind_speed"]) <= 1e-12

    # Negative scalar extrapolation is projected to zero as part of the declared predictor.
    speed_neg = np.array([1.0, 0.1, 99.0])
    fc0, reason0 = S7._forecast_weather_coherent(
        3600.0, 7200.0, times, wind_vec, speed_neg, hs, 5400.0)
    assert reason0 is None and fc0 is not None
    assert fc0["raw_wind_speed_forecast"] < 0.0
    assert fc0["wind_speed"] == 0.0
    assert np.array_equal(fc0["wind_vec"], np.zeros(2))

    # Negative Hs never receives a silent clip in formal preprocessing.
    hs_bad = np.array([0.2, 0.05, 9.0])
    fch, reasonh = S7._forecast_weather_coherent(
        3600.0, 7200.0, times, wind_vec, wind_speed, hs_bad, 5400.0)
    assert fch is None and reasonh == "negative_hs"

    # 2) Old weather moments contract must be rejected after the predictor change.
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        wp = td / "weather_old.csv"
        rows = []
        for h in (5, 10, 15, 20, 25, 30):
            rows.append(dict(
                h_min=float(h), n=50, wind_bias_e_ms=0.0, wind_bias_n_ms=0.0,
                wind_sigma_ee=1.0, wind_sigma_en=0.0, wind_sigma_nn=1.0,
                wind_speed_bias_ms=0.0, wind_speed_std_ms=1.0,
                hs_bias_m=0.0, hs_std_m=0.1,
                predictor="weather_linear_noleak",
                predictor_contract="weather_backward_linear_hourly_epoch_seconds_v1",
                timestamp_epoch_contract=RM.WEATHER_TIMESTAMP_EPOCH_CONTRACT,
                truth_contract=RM.WEATHER_TRUTH_CONTRACT,
                weather_data_contract="real-history-noleak-weather-residuals-global-weather-nonoverlap-v2",
                moments_source="train", sample_overlap_policy="weather_timeline_global_nonoverlap",
                purge_min=30.0, valid_for_formal=True,
                weather_source_sha256="a"*64, xi_train_source_sha256="b"*64))
        pd.DataFrame(rows).to_csv(wp, index=False)
        try:
            RM.weather_ambiguity_from_moments_csv(
                wp, [5,10,15,20,25,30], formal=True)
        except ValueError:
            pass
        else:
            raise AssertionError("old incoherent weather contract was accepted")

        # 3) Shared formal Xi train helper filters the selected vessel and never pools ALL.
        xp = td / "xi_train.csv"
        xr = []
        for mmsi, off in (("219018788", 0.0), ("219028973", 100.0)):
            for k in range(30):
                t0 = 1000.0 + 600.0 * k + off
                xr.append(dict(
                    mmsi=mmsi, h_min=5, c_state="动力定位",
                    t0_epoch=t0, t1_epoch=t0 + 300.0,
                    xi_e_m=float(k % 3), xi_n_m=float((k + 1) % 4),
                    predictor="cv_noleak",
                    predictor_contract=M.XI_PREDICTOR_CONTRACTS["cv_noleak"],
                    timestamp_epoch_contract=M.XI_TIMESTAMP_EPOCH_CONTRACT,
                    sample_overlap_policy="nonoverlap", purge_min=30.0,
                    moments_source="train", valid_for_formal=True, split="train"))
        pd.DataFrame(xr).to_csv(xp, index=False)
        _df, xa = S13._xi_ambiguity_from_train_samples(
            xp, "219018788", formal=True)
        assert set(_df["mmsi"].astype(str)) == {"219018788"}
        assert xa.selected_mmsi == "219018788"
        assert not xa.cross_vessel_pooling

    # Step14 wiring must consume the shared formal helper and formal launch builder.
    s14src = inspect.getsource(S14._setup)
    assert "_xi_ambiguity_from_train_samples" in s14src
    assert "formal=formal" in s14src
    assert "_assert_weather_xi_train_binding" in s14src
    assert "_record_formal_instance_provenance" in s14src

    # Formal A1/A2 is fail-closed against Step13 parity-breaking CLI overrides.
    base = Namespace(study_mode="formal", weather_drcc="on", recovery_predictor="cv_noleak",
                     pool_h="pareto", soc_correction="geo2d", deck_mode="interval")
    S14._validate_formal_base_parity(base)
    for field, bad_value in (("weather_drcc", "off"), ("recovery_predictor", "true_track"),
                             ("pool_h", "first"), ("soc_correction", "none"),
                             ("deck_mode", "slot")):
        bad = Namespace(**vars(base)); setattr(bad, field, bad_value)
        try:
            S14._validate_formal_base_parity(bad)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"formal Step14 accepted parity-breaking {field}={bad_value}")

    # Private formal-instance attributes are omitted by Namespace JSON serialization,
    # so the formal manifest helper must explicitly preserve them.
    aud = Namespace(_formal_instance_mmsi="219018788",
                    _formal_instance_track_sha256="a" * 64,
                    _formal_instance_xi_train_sha256="b" * 64,
                    _formal_instance_weather_moments_sha256="c" * 64,
                    _formal_instance_weather_predictor_contract="coherent-v2",
                    _formal_instance_launch_formal=True)
    extra = S13._formal_instance_manifest_extra(aud)
    assert extra["_formal_instance_mmsi"] == "219018788"
    assert extra["_formal_instance_launch_formal"] is True
    assert "_formal_instance_mmsi" not in M._jsonable(aud)

    # Explicit custom --track-csv + --track-mmsi is checked against an MMSI column.
    with tempfile.TemporaryDirectory() as td:
        cp = Path(td) / "custom_ship.csv"
        pd.DataFrame({"mmsi": [219018788, 219018788], "x": [1, 2]}).to_csv(cp, index=False)
        assert S13._infer_concrete_track_mmsi(cp, "219018788", formal=True) == "219018788"
        try:
            S13._infer_concrete_track_mmsi(cp, "219028973", formal=True)
        except SystemExit:
            pass
        else:
            raise AssertionError("formal custom track accepted explicit MMSI conflicting with file data")

    # 4) E2 never compares unsupported speed-recourse methods as zero-coverage baselines.
    ns = Namespace(e2_criteria="recourse_compatible", time_recourse="wait_and_speed")
    got = S13._resolve_e2_criteria_for_recourse(ns)
    assert got == S13._E2_DRCC_RECOURSE_CRITERIA
    ns_all = Namespace(e2_criteria="all", time_recourse="wait_and_speed")
    try:
        S13._resolve_e2_criteria_for_recourse(ns_all)
    except SystemExit:
        pass
    else:
        raise AssertionError("wait_and_speed accepted unsupported SAA/box/budget criteria")
    ns_wait = Namespace(e2_criteria="all", time_recourse="wait_only")
    assert len(S13._resolve_e2_criteria_for_recourse(ns_wait)) == len(S13._E2_CRITERIA)

    print("suite v17_alignment: coherent weather / formal Step14 / E2 recourse PASS ✓")


SUITES["v17_alignment"] = suite_v17_alignment


FAST_GROUPS = ["basic", "cg_pricing", "l1_branch", "l2_energy", "certificates",
               "counterexamples", "mutations", "random_oracle", "p0_regressions",
               "exact_optimizer"]
SLOW_SUITES = ["branch", "e1", "resume", "certificates_full"]
SUITE_TIMEOUT_S = {"core": 60, "bp": 90, "branch_price": 90, "l2_energy": 90,
                   "certificates": 60, "certificates_full": 600, "counterexamples": 90, "mutations": 220,
                   "random_oracle": 130, "seed_validation": 90, "solver_validation": 180,
                   "resume_fast": 30, "branch": 900, "e1": 900, "resume": 900,
                   "exact_bpc": 120, "hardening": 60}
GLOBAL_BUDGET_S = 300.0


def _configure_console_error_policy():
    """Keep legacy Windows console encodings from crashing passing test suites.

    Only stream text error handling changes: unsupported glyphs are escaped.
    Assertions, child return codes, timeouts and solver/test semantics are untouched.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(errors="backslashreplace")
            except (ValueError, OSError):
                pass


def _console_safe_text(value):
    """Render child-test diagnostics without letting console code pages crash the runner.

    The child process itself runs under a deterministic UTF-8 contract.  The parent
    process may still have a legacy Windows stdout encoding (for example GBK when
    PowerShell pipes/redirections are involved).  Preserve all ASCII/Chinese text
    that the parent can encode and escape only unsupported glyphs such as circled
    digits/checkmarks.  This function affects diagnostics only, never test logic or
    return codes.
    """
    text = str(value)
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(enc, errors="strict")
        return text
    except (UnicodeEncodeError, LookupError):
        try:
            return text.encode(enc, errors="backslashreplace").decode(enc, errors="strict")
        except Exception:
            return text.encode("ascii", errors="backslashreplace").decode("ascii")


def _run_child(cmd, timeout_s, label):
    import subprocess
    # Windows hardening: grouped suites run with stdout/stderr captured by PIPE.
    # On Windows this can make a child Python fall back to the system ANSI/GBK
    # code page even when the interactive parent console is UTF-8. The tests
    # deliberately print Unicode audit markers (e.g. ✓, ⓪, ㉓), so a locale-only
    # child can fail after all assertions have already passed. Force a
    # deterministic UTF-8 child contract and decode captured streams as UTF-8.
    child_env = os.environ.copy()
    child_env["PYTHONUTF8"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONLEGACYWINDOWSSTDIO"] = "0"
    t0 = time.time()
    try:
        r = subprocess.run(
            cmd,
            timeout=timeout_s,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
        )
    except subprocess.TimeoutExpired:
        print(_console_safe_text(f"  [{label}] TIMEOUT(> {timeout_s}s) → FAIL"))
        return False, timeout_s
    dt = time.time() - t0
    tail = "\n".join((r.stdout or "").strip().splitlines()[-3:])
    if r.returncode != 0:
        errtail = "\n".join((r.stderr or "").strip().splitlines()[-8:])
        print(_console_safe_text(f"  [{label}] FAIL(rc={r.returncode}, {dt:.0f}s)\n{tail}\n{errtail}"))
        return False, dt
    print(_console_safe_text(f"  [{label}] PASS({dt:.0f}s): {tail.splitlines()[-1] if tail else ''}"))
    return True, dt

def main():
    _configure_console_error_policy()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite",
                    choices=sorted(set(["all", "slow"] + list(SUITES) + list(GROUPS))),
                    default="all")
    ap.add_argument("--mutation-driver", default=None,
                    help="(内部) 在当前目录代码上运行指定变异驱动后退出; "
                         "'clean_all' = 依次运行全部驱动")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))

    if args.mutation_driver:                       # 变异驱动分派(suite mutations 子进程)
        keys = (sorted(_MUT_DRIVERS) if args.mutation_driver == "clean_all"
                else [args.mutation_driver])
        for k in keys:
            _MUT_DRIVERS[k]()
            print(f"driver {k}: PASS")
        return

    if args.suite in SUITES:                       # 单套件: 本进程直跑(调试/慢跑)
        print(f"\n========== suite: {args.suite} ==========")
        SUITES[args.suite]()
        print("\nselftest: 所选套件全部 PASS ✓")
        return

    if args.suite in GROUPS:
        groups = [args.suite]
    elif args.suite == "slow":
        groups = None
    else:
        groups = FAST_GROUPS

    t_all = time.time()
    ok_all = True
    if groups is None:                             # slow: 逐慢套件子进程
        for s in SLOW_SUITES:
            print(f"\n===== slow suite: {s} (cap {SUITE_TIMEOUT_S[s]}s) =====")
            ok, _ = _run_child([sys.executable, os.path.join(here, "selftest.py"),
                                "--suite", s], SUITE_TIMEOUT_S[s], s)
            ok_all &= ok
    else:
        for g in groups:
            print(f"\n===== group: {g} =====")
            for s in GROUPS[g]:
                elapsed = time.time() - t_all
                remaining = GLOBAL_BUDGET_S - elapsed
                if remaining <= 0.0:
                    print(f"selftest: 超出全量预算 {GLOBAL_BUDGET_S:.0f}s → FAIL")
                    sys.exit(2)
                # The per-suite timeout must also respect the *remaining* global
                # wall-clock budget.  Checking only before subprocess.run() lets
                # the last child overrun GLOBAL_BUDGET_S by its whole local cap.
                child_timeout = min(float(SUITE_TIMEOUT_S[s]), max(0.1, remaining))
                ok, _ = _run_child([sys.executable, os.path.join(here, "selftest.py"),
                                    "--suite", s], child_timeout, s)
                ok_all &= ok
    dt = time.time() - t_all
    if not ok_all:
        print(f"\nselftest: 存在失败/超时组(总用时 {dt:.0f}s) → FAIL")
        sys.exit(1)
    if groups is not None and dt > GLOBAL_BUDGET_S:
        print(f"\nselftest: 全部组通过但总用时 {dt:.0f}s > "
              f"预算 {GLOBAL_BUDGET_S:.0f}s → FAIL")
        sys.exit(2)
    print(f"\nselftest: 所选套件全部 PASS ✓(总用时 {dt:.0f}s)")


if __name__ == "__main__":
    main()
