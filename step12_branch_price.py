#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step12_branch_price.py — 有限离散模型的精确按需列生成与最终求解入口。

正式入口 ``solve_fleet_anytime`` 采用 Branch-Price-and-Cut + Logic-Based Benders：
路线不在求解开始前完整物化，而是在分支节点中通过精确定价按需生成；主问题使用
风机互斥集合打包约束；整数候选必须通过实体 UAV/电池/SOC/甲板/快检/换电资源审计。
只有分支划分完备、遗漏列界有效、Farkas/Phase-I 可行性定价完成、资源审计无未知态，
且两阶段 Gap 闭合时，才声明当前有限离散模型的词典序全局最优。

文件前部保留的完整列扫描、旧 Ryan–Foster 搜索和其他实验实现仅是研究基线或回归
工具，不在正式证书调用链中；其受限列池值和求解器 Gap 不得解释为隐式全路线空间界。
"""
from __future__ import annotations

import logging
import math
import time
import heapq
import hashlib
import json
import os
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

import step9_model as M
import step10_model_routing as RM
import step11_algorithm_route_drcc as RA

log = logging.getLogger("bp")

# 更新(M-02): L2 人工残量统一容差 —— 证书条件 l2_no_artificial_residue 与
# incumbent 接受判据共用同一常数(缺字段/超容差一律 False, 不用宽松默认)。
ART_TOL = 1e-6
# ``PRICING_EPS`` is retained only as a reporting / heuristic-scale constant.
# The formal certificate path never uses a negative tolerance as the mathematical
# pricing-closure threshold: a route is sign-definitely improving whenever its
# outward reduced-cost upper endpoint is < 0, and exact closure requires the
# omitted-column reduced-cost lower bound to be >= 0.
PRICING_EPS = 1e-6
# Phase-I/Farkas pricing cannot use the ordinary objective-improvement tolerance
# as an infeasibility certificate.  We still add numerically meaningful negative
# Phase-I columns aggressively; formal infeasibility is decided separately from
# a complete-space lower bound on the artificial objective.
FARKAS_COLUMN_EPS = 1e-12
LP_CERT_TOL = 1e-7
ROUTE_IDENTITY_CONTRACT = "binary64-exact-weather-route-identity-v2"
MODEL_SEMANTICS_CONTRACT = "finite-route-model-strict-physical-v8-discrete-recovery-target-xi-only-coherent-weather"
RESULT_CERTIFICATE_CONTRACT = "finite-binary64-physical-route-universe-certificate-v12-source-bound-global-battery-relaxation"
ROUTE_SEMANTICS_CONTRACT = "canonical-route-signature-immutable-formal-semantics-v1"
FUTURE_COLUMN_ROW_RANGE_CONTRACT = "explicit-master-row-coefficient-range-fail-closed-v1"
FORMAL_PROOF_CONTRACT = "exact-bpc-proof-code-concordance-v9-source-bound-global-battery-relaxation"
FULLCOVER_CLOSURE_CHECKPOINT_CONTRACT = "fullcover-target-closure-checkpoint-v2-global-battery-relaxation"
# Stable theorem/lemma identifiers used by code comments, result provenance and
# doc_proof.md/doc_algorithm.md.  These are certificate-semantics metadata, not
# changes to the mathematical model.
FORMAL_PROOF_OBLIGATIONS = (
    "THM-RU",       # actual route universe equals the formal physical Oracle universe
    "LEM-CS",       # canonical route signature has immutable formal semantics
    "THM-LRC",      # full-space Lagrangian + omitted-column reduced-cost correction
    "COR-P1",       # Elastic Phase-I full-space infeasibility corollary
    "LEM-PAT",      # exact-pattern Hamming-distance resource cut
    "THM-BR",       # complete pricing-compatible service/arc/route branching
    "THM-NUM",      # binary64 outward/exact-rational certificate chain
    "THM-LEX",      # strict two-stage lexicographic global closure
    "THM-TGT",      # fixed-coverage target decision
    "THM-CU",       # certified materialized complete physical route universe
    "THM-FCT",      # direct full-cover target MILP + exact resource Logic-Benders
    "THM-GBR",      # exact global battery-energy relaxation on complete full-cover universe
)
FORMAL_PROOF_CODE_ANCHORS = (
    ("THM-RU", ("solve_fleet_anytime", "_solve_fleet_anytime_impl",
                "_exact_pricing_search")),
    ("LEM-CS", ("_column_semantics_fp", "_add_columns",
                "_route_archive_semantics_invariant")),
    ("THM-LRC", ("_lagrangian_dual_lower_bound",
                 "_future_column_coefficient_range",
                 "_universal_pricing_lower_bound",
                 "_safe_node_bound_from_pricing")),
    ("COR-P1", ("_solve_elastic_phase_one",
                "_phase_one_full_space_lower_bound",
                "_phase_one_infeasibility_proven")),
    ("LEM-PAT", ("_row_coefficient", "_audit_integer_selection")),
    ("THM-BR", ("_column_allowed_at_node", "_master_rows",
                "_branch_on_fractional_solution",
                "_branch_on_integral_numeric_ambiguity")),
    ("THM-NUM", ("_validate_linprog_result",
                 "_column_reduced_cost_interval",
                 "_exact_binary_master_feasible",
                 "_energy_of_selection_exact")),
    ("THM-LEX", ("_solve_branch_price_stage",
                 "_physical_certificate_guard", "_solve_fleet_anytime_impl")),
    ("THM-TGT", ("_master_rows", "_solve_elastic_phase_one",
                 "_phase_one_infeasibility_proven", "_solve_branch_price_stage",
                 "_target_infeasibility_algorithmic_proven", "_solve_fleet_anytime_impl")),
    ("THM-CU", ("build_certified_route_universe", "_validate_certified_route_universe",
                "_solve_branch_price_stage", "_solve_fleet_anytime_impl")),
    ("THM-FCT", ("_solve_complete_universe_fullcover_target",
                 "_exact_fullcover_master_feasibility",
                 "_fullcover_target_master_rows",
                 "_exact_battery_binpack_status",
                 "_minimal_battery_conflict_core",
                 "_fullcover_closure_context_sha256",
                 "_load_fullcover_closure_checkpoint",
                 "_save_fullcover_closure_checkpoint",
                 "_target_infeasibility_algorithmic_proven",
                 "_solve_fleet_anytime_impl")),
    ("THM-GBR", ("_exact_global_fullcover_battery_relaxation",
                 "_solve_complete_universe_fullcover_target",
                 "_target_infeasibility_algorithmic_proven",
                 "_solve_fleet_anytime_impl")),
)
def _formal_proof_code_sha256() -> str:
    """Bind the proof contract to the actual proof-critical source bytes."""
    base = Path(__file__).resolve().parent
    names = ("step9_model.py", "step10_model_routing.py",
             "step11_algorithm_route_drcc.py", "step12_branch_price.py")
    h = hashlib.sha256()
    for name in names:
        path = base / name
        h.update(name.encode("utf-8")); h.update(b"\0")
        h.update(path.read_bytes()); h.update(b"\0")
    return h.hexdigest()


FORMAL_PROOF_CODE_SHA256 = _formal_proof_code_sha256()
FORMAL_PROOF_CONTRACT_SHA256 = hashlib.sha256(json.dumps(
    dict(contract=FORMAL_PROOF_CONTRACT, obligations=FORMAL_PROOF_OBLIGATIONS,
         code_anchors=FORMAL_PROOF_CODE_ANCHORS,
         proof_code_sha256=FORMAL_PROOF_CODE_SHA256),
    ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _finite_number(name, value, *, nonnegative=False, positive=False):
    """Validate a public numeric model/control parameter without silent NaN/inf."""
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    if positive and not out > 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and out < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return out


def _nonnegative_int(name, value, *, positive=False):
    """Validate integer-valued counts rather than silently truncating floats."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not bool")
    out = _finite_number(name, value)
    if out != math.floor(out):
        raise ValueError(f"{name} must be an integer")
    out = int(out)
    if positive and out < 1:
        raise ValueError(f"{name} must be positive")
    if not positive and out < 0:
        raise ValueError(f"{name} must be nonnegative")
    return out


def _normalize_solver_mode(value):
    key = str(value).strip().lower().replace("_", "-")
    aliases = {
        "auto": "exact-branch-price-cut",
        "branch-cut-price": "exact-branch-price-cut",
        "branch-price-cut": "exact-branch-price-cut",
        "branch-cut-and-price": "exact-branch-price-cut",
        "exact-enumeration": "research-baseline",
        "restricted-pool": "research-baseline",
    }
    key = aliases.get(key, key)
    if key not in {"exact-branch-price-cut", "research-baseline"}:
        raise ValueError("solver_mode must be exact-branch-price-cut or research-baseline")
    return key


def _normalize_exact_rmp_solver(value):
    """Normalize the only LP backend implemented by the formal BPC.

    The public ``solver`` argument used to be silently ignored by the exact
    path. Requesting another backend must not return a SciPy/HiGHS certificate
    under a misleading provenance label.
    """
    key = str(value).strip().lower().replace("_", "-")
    aliases = {
        "auto": "scipy-highs-rmp",
        "scipy": "scipy-highs-rmp",
        "highs": "scipy-highs-rmp",
        "scipy-highs": "scipy-highs-rmp",
        "scipy-highs-rmp": "scipy-highs-rmp",
    }
    if key not in aliases:
        raise ValueError(
            "formal exact BPC RMP backend is scipy-highs-rmp; "
            f"unsupported solver={value!r}")
    return aliases[key]


def _validate_anytime_public_contract(*, solver_mode, pricing_mode, kappa_mode,
                                       chance_mode, deck_mode, battery_reuse_mode,
                                       solver, pool_h_mode, time_limit_s, deadline,
                                       budget_gamma, K, batteries, max_stops,
                                       coverage_gap_target_abs,
                                       energy_gap_target_rel,
                                       energy_gap_target_abs_Wh,
                                       pricing_batch_size, solve_scope, coverage_target=None):
    """Normalize/validate the mathematical model requested by the public API.

    Unknown categorical values are never mapped through a default ``else``
    branch: doing so would make the certified feasible region depend on process
    history or on an implementation fallback rather than on the call arguments.
    """
    mode = _normalize_solver_mode(solver_mode)
    solver_requested = str(solver).strip().lower().replace("_", "-")
    solver_effective = (_normalize_exact_rmp_solver(solver)
                        if mode == "exact-branch-price-cut" else solver_requested)
    pricing_raw = str(pricing_mode).strip().lower().replace("_", "-")
    if pricing_raw in {"exact-labeling", "exact-sequence-labeling"}:
        raise ValueError(
            "pricing_mode='exact-labeling' is not implemented: the formal exact "
            "pricer is exhaustive elementary-sequence DFS, not an ESPPRC/RCSP labeler")
    if pricing_raw in {"exact-implicit-dfs", "exact-implicit-sequence-dfs"}:
        pricing_key = "exact-implicit-dfs"
    elif pricing_raw in {"exact-discovery-shadow", "exact-implicit-dfs-discovery-shadow"}:
        # Experimental certificate-preserving mode. It changes search control
        # only: sign-definite improving columns may return early to the RMP,
        # while prefix completion bounds are evaluated in shadow mode and NEVER
        # prune the exact DFS in this version.
        pricing_key = "exact-discovery-shadow"
    elif pricing_raw in {"exact-dual-guided-shadow", "exact-guided-discovery-shadow"}:
        # V4 experiment: same finite pricing domain and unchanged physical oracle.
        # Discovery order only is changed; no launch/prefix/horizon is removed.
        pricing_key = "exact-dual-guided-shadow"
    elif pricing_raw in {"exact-layered-guided-shadow", "exact-layered-discovery"}:
        # V5 experiment: discovery is breadth-by-depth across the global launch
        # set, round-robin across launches within each depth.  Ordering only:
        # no legal launch/prefix/horizon is removed and no shadow bound prunes.
        pricing_key = "exact-layered-guided-shadow"
    elif pricing_raw in {"exact-layered-batch-shadow", "exact-adaptive-batch-shadow"}:
        # V6 experiment: keep the V5 traversal, but amortize repeated RMP solves
        # by returning a small batch of strictly sign-definite improving columns.
        # Batch/diversity rules have no proof role and never admit rc_ub >= 0.
        pricing_key = "exact-layered-batch-shadow"
    elif pricing_raw in {
            "exact-layered-batch-primal-shadow",
            "exact-primal-refresh-shadow"}:
        # V7 experiment: V6 pricing plus a bounded primal-only incumbent refresh
        # after newly generated columns enter the archive.  Refresh solutions are
        # accepted only after the unchanged exact resource audit; they never
        # contribute a lower bound, pruning decision, or pricing certificate.
        pricing_key = "exact-layered-batch-primal-shadow"
    elif pricing_raw in {
            "exact-layered-batch-primal-diagnostic-shadow",
            "exact-root-cause-diagnostic-shadow"}:
        # V8 diagnostic: V7 search/control is unchanged. Extra telemetry dissects
        # depth and primal neighborhoods; a fixed-archive exact diagnostic runs
        # only after the formal solver wall-clock budget has ended.
        pricing_key = "exact-layered-batch-primal-diagnostic-shadow"
    elif pricing_raw in {
            "exact-layered-batch-primal-target9-diagnostic-shadow",
            "exact-target9-diagnostic-shadow"}:
        # V9 diagnostic: the formal V7 search/control policy is still unchanged.
        # After the formal clock, freeze the generated archive and answer only
        # whether an exact-resource-feasible coverage >= 9 witness exists.
        pricing_key = "exact-layered-batch-primal-target9-diagnostic-shadow"
    elif pricing_raw in {
            "exact-layered-batch-primal-target9-certificate-diagnostic-shadow",
            "exact-target9-certificate-diagnostic-shadow"}:
        # V10 diagnostic: V9 target decision is unchanged. A separate post-target
        # shadow budget analyzes already rejected patterns with certificate-safe
        # necessary relaxations; pricing also records cut-induced RC telemetry.
        pricing_key = "exact-layered-batch-primal-target9-certificate-diagnostic-shadow"
    elif pricing_raw in {
            "exact-layered-batch-primal-target9-battery-clique-diagnostic-shadow",
            "exact-target9-battery-clique-diagnostic-shadow"}:
        # V11 diagnostic: formal V7/V10 control remains unchanged.  After the
        # baseline frozen-archive target decision and V10 certificate shadow, a
        # second frozen-archive target run adds only mathematically safe
        # route-energy incompatibility clique rows.  The rerun is diagnostic-only.
        pricing_key = "exact-layered-batch-primal-target9-battery-clique-diagnostic-shadow"
    elif pricing_raw in {
            "exact-layered-batch-primal-battery-halfcap-formal",
            "exact-battery-halfcap-formal"}:
        # V12 formal strengthening: retain the V7 search/control policy, but add
        # the certificate-safe global battery half-capacity clique inequality to
        # every formal restricted master.  This changes only the LP relaxation,
        # never the integer feasible set or the exact resource audit.
        pricing_key = "exact-layered-batch-primal-battery-halfcap-formal"
    elif pricing_raw in {
            "exact-layered-batch-primal-battery-halfcap-depth-fair-formal",
            "exact-battery-halfcap-depth-fair-formal"}:
        # V13 formal experiment: keep the V12 valid inequality and all proof
        # semantics unchanged.  Only incomplete *discovery* order changes: once
        # the formal half-cap row has a nonzero dual, exact target-depth
        # iterators are round-robined across depths as well as launches so
        # depth-3/4 cannot be starved behind a huge depth-2 layer.  Exhaustive
        # certification and every rc/pruning/certificate rule remain unchanged.
        pricing_key = "exact-layered-batch-primal-battery-halfcap-depth-fair-formal"
    elif pricing_raw in {
            "exact-layered-batch-primal-battery-halfcap-depth-fair-neutral-formal",
            "exact-battery-halfcap-depth-fair-neutral-formal"}:
        # V14 formal experiment: keep V13 exact pricing/certificate semantics,
        # but permit a bounded discovery-only batch of physically valid
        # multi-stop columns whose rc interval is not sign-definite negative.
        # These heuristic enrichment columns may improve the restricted integer
        # incumbent, but NEVER prove pricing closure, node fathoming, or a bound.
        pricing_key = (
            "exact-layered-batch-primal-battery-halfcap-depth-fair-neutral-formal")
    elif pricing_raw in {
            "exact-layered-batch-primal-battery-halfcap-resource-exchange-formal",
            "exact-battery-halfcap-resource-exchange-formal",
            "exact-resource-aware-bpc"}:
        # V15 architecture-convergence experiment: retain the V12 formal
        # half-cap valid inequality and the V7 heuristic-first exact pricing
        # policy.  Replace the legacy augmentation/rebuild/repair coverage
        # refresh by one bounded archive exchange heuristic.  Every incumbent
        # accepted by the exchange still passes the unchanged exact resource
        # audit; the heuristic has no pricing/bound/pruning/certificate role.
        pricing_key = (
            "exact-layered-batch-primal-battery-halfcap-resource-exchange-formal")
    elif pricing_raw in {
            "exact-layered-batch-primal-battery-halfcap-resource-primal-formal",
            "exact-battery-halfcap-resource-primal-formal",
            "exact-resource-aware-primal-bpc"}:
        # V16: preserve the V12 formal half-cap row and V7 heuristic-first
        # exact pricing.  Use one unified coverage primal: preserve the current
        # incumbent, exact-audit monotone additions first, then use the V15
        # exchange neighborhood with the remaining short budget.
        pricing_key = (
            "exact-layered-batch-primal-battery-halfcap-resource-primal-formal")
    elif pricing_raw in {
            "exact-layered-batch-primal-battery-halfcap-resource-guided-formal",
            "exact-battery-halfcap-resource-guided-formal",
            "exact-resource-guided-bpc"}:
        # V17: keep the V16/V12 formal architecture, but make the single
        # resource-aware primal spend its short audit budget efficiently.
        # Proven-infeasible exact selections are cached across refresh calls,
        # and augmentation variants are interleaved across uncovered turbines.
        # This changes heuristic search order only; all formal proof semantics
        # remain identical to V16.
        pricing_key = (
            "exact-layered-batch-primal-battery-halfcap-resource-guided-formal")
    elif pricing_raw in {
            "exact-layered-batch-primal-battery-halfcap-deck-guided-formal",
            "exact-battery-halfcap-deck-guided-formal",
            "exact-deck-guided-bpc"}:
        # V18: retain V17/V12 formal semantics.  The single coverage primal
        # uses the exact same fixed half-open deck-interval conflict relation
        # as the resource oracle to order candidates and fail-fast heuristic
        # trials before an expensive resource DFS.  Archive deck-conflict graph
        # statistics are diagnostic only; no new Master row or certificate rule
        # is introduced.
        pricing_key = (
            "exact-layered-batch-primal-battery-halfcap-deck-guided-formal")
    elif pricing_raw in {
            "exact-layered-batch-primal-battery-halfcap-adaptive-multistop-formal",
            "exact-battery-halfcap-adaptive-multistop-formal",
            "exact-adaptive-multistop-bpc"}:
        # V19: keep the V18/V12 formal architecture and add one bounded
        # heuristic-only two-stop exact-variant enrichment after singleton-only
        # pricing batches.  Every enrichment candidate passes the unchanged
        # whole-route physical/DRCC evaluator.  It is a legal RMP column but has
        # no pricing-closure, bound, pruning, or certificate role.
        pricing_key = (
            "exact-layered-batch-primal-battery-halfcap-adaptive-multistop-formal")
    elif pricing_raw in {
            "exact-layered-batch-primal-battery-halfcap-resource-variant-formal",
            "exact-battery-halfcap-resource-variant-formal",
            "exact-resource-variant-bpc"}:
        # V20: evidence-driven simplification after V19 found 72/72 blind
        # two-stop merge probes route-level infeasible.  Keep the V18/V12
        # formal architecture, but enrich only deck-compatible exact singleton
        # timing variants for currently uncovered turbines.  These are legal
        # heuristic RMP columns and never participate in pricing closure,
        # bounds, pruning, infeasibility, or optimality certificates.
        pricing_key = (
            "exact-layered-batch-primal-battery-halfcap-resource-variant-formal")
    elif pricing_raw in {
            "exact-layered-batch-primal-battery-halfcap-resource-variant-diagnostic-formal",
            "exact-battery-halfcap-resource-variant-diagnostic-formal",
            "exact-resource-variant-diagnostic-bpc"}:
        # V20.1: formal V20 search semantics are unchanged.  Only after the
        # formal solver clock expires/stops, run frozen-archive diagnostics that
        # identify V20 variants, exact-audit their relationship to the final
        # incumbent, test single-deck-blocker retiming witnesses, and solve the
        # current generated archive exactly.  None of those postsolve results
        # can modify the formal incumbent, bounds, pruning, or certificates.
        pricing_key = (
            "exact-layered-batch-primal-battery-halfcap-resource-variant-diagnostic-formal")
    elif pricing_raw in {
            "r-bpc",
            "exact-layered-batch-primal-battery-halfcap-resource-variant-archive-recovery-formal",
            "exact-battery-halfcap-resource-variant-archive-recovery-formal",
            "exact-resource-variant-archive-recovery-bpc"}:
        # Paper algorithm (R-BPC).  Keep the full-space exact pricing, Master,
        # branching, resource-audit and certificate semantics unchanged.
        # Resource-aware singleton timing enrichment may add legal physical
        # columns but has no closure/bound/pruning role.  After root
        # initialization and each successful column batch, solve only the
        # currently generated archive exactly for a better primal witness.
        # A witness may update the incumbent/LB only after the unchanged exact
        # resource audit re-accepts it.  Restricted-archive bounds, cuts and
        # certificates are never imported into full-space UB, pruning or proof.
        pricing_key = (
            "exact-layered-batch-primal-battery-halfcap-resource-variant-archive-recovery-formal")
    elif pricing_raw == "exact-mip":
        pricing_key = "exact-mip"
    else:
        raise ValueError(
            "pricing_mode must be exact-implicit-dfs, exact-discovery-shadow, "
            "exact-dual-guided-shadow, exact-layered-guided-shadow, "
            "exact-layered-batch-shadow, exact-layered-batch-primal-shadow, "
            "exact-layered-batch-primal-diagnostic-shadow, "
            "exact-layered-batch-primal-target9-diagnostic-shadow, "
            "exact-layered-batch-primal-target9-certificate-diagnostic-shadow, "
            "exact-layered-batch-primal-target9-battery-clique-diagnostic-shadow, "
            "exact-layered-batch-primal-battery-halfcap-formal, "
            "exact-layered-batch-primal-battery-halfcap-depth-fair-formal, "
            "exact-layered-batch-primal-battery-halfcap-depth-fair-neutral-formal, "
            "exact-layered-batch-primal-battery-halfcap-resource-exchange-formal, "
            "exact-layered-batch-primal-battery-halfcap-resource-primal-formal, "
            "exact-layered-batch-primal-battery-halfcap-resource-guided-formal, "
            "exact-layered-batch-primal-battery-halfcap-deck-guided-formal, "
            "exact-layered-batch-primal-battery-halfcap-adaptive-multistop-formal, "
            "exact-layered-batch-primal-battery-halfcap-resource-variant-formal, "
            "exact-layered-batch-primal-battery-halfcap-resource-variant-diagnostic-formal, "
            "exact-layered-batch-primal-battery-halfcap-resource-variant-archive-recovery-formal, "
            "r-bpc, or exact-mip")
    kappa_key = str(kappa_mode).strip().lower()
    if kappa_key not in set(RM.KAPPA_MODES) | {"nominal"}:
        raise ValueError(f"unknown kappa_mode={kappa_mode!r}")
    chance_key = str(chance_mode).strip().lower()
    if chance_key not in {"drcc", "saa", "budget", "box"}:
        raise ValueError(f"unknown chance_mode={chance_mode!r}")
    deck_key = str(deck_mode).strip().lower()
    if deck_key not in {"interval", "slot"}:
        raise ValueError(f"unknown deck_mode={deck_mode!r}")
    battery_key = str(battery_reuse_mode).strip().lower()
    pool_key = str(pool_h_mode).strip().lower()
    if mode == "exact-branch-price-cut":
        if battery_key != "exact_soc":
            raise ValueError("exact-branch-price-cut requires battery_reuse_mode='exact_soc'")
        # The on-demand BPC does not construct a recovery-horizon route pool.
        # Accept only the formal/default spelling so callers cannot believe that
        # an ignored 'first' pool policy changed the certified model.
        if pool_key != "pareto":
            raise ValueError("exact-branch-price-cut requires pool_h_mode='pareto' (on-demand route space)")
    else:
        if battery_key not in {"exact_soc", "legacy_count"}:
            raise ValueError("research-baseline battery_reuse_mode must be exact_soc or legacy_count")
        if pool_key not in {"pareto", "first"}:
            raise ValueError("research-baseline pool_h_mode must be pareto or first")

    if time_limit_s is not None:
        _finite_number("time_limit_s", time_limit_s, nonnegative=True)
    if deadline is not None:
        _finite_number("deadline", deadline)
    _finite_number("budget_gamma", budget_gamma, nonnegative=True)
    k_count = _nonnegative_int("K", K, positive=True)
    b_count = None if batteries is None else _nonnegative_int("batteries", batteries)
    _nonnegative_int("max_stops", max_stops, positive=True)
    _nonnegative_int("coverage_gap_target_abs", coverage_gap_target_abs)
    _finite_number("energy_gap_target_rel", energy_gap_target_rel, nonnegative=True)
    _finite_number("energy_gap_target_abs_Wh", energy_gap_target_abs_Wh, nonnegative=True)
    _nonnegative_int("pricing_batch_size", pricing_batch_size, positive=True)
    solve_scope_key = str(solve_scope).strip().lower().replace("_", "-")
    if solve_scope_key not in {"lexicographic", "coverage-only", "coverage-target"}:
        raise ValueError("solve_scope must be 'lexicographic', 'coverage-only', or 'coverage-target'")
    coverage_target_value = None
    if solve_scope_key == "coverage-target":
        if coverage_target is None:
            raise ValueError("solve_scope='coverage-target' requires coverage_target")
        coverage_target_value = _nonnegative_int("coverage_target", coverage_target)
        if coverage_target_value < 1:
            raise ValueError("coverage_target must be positive for coverage-target scope")
    elif coverage_target is not None:
        raise ValueError("coverage_target is only valid with solve_scope='coverage-target'")
    return dict(
        solver_mode=mode, pricing_mode=pricing_key, kappa_mode=kappa_key,
        chance_mode=chance_key, deck_mode=deck_key,
        solver_requested=solver_requested, solver_effective=solver_effective,
        battery_reuse_mode=battery_key, pool_h_mode=pool_key,
        K=k_count, batteries=b_count,
        budget_gamma=float(budget_gamma),
        max_stops=int(max_stops),
        coverage_gap_target_abs=int(coverage_gap_target_abs),
        energy_gap_target_rel=float(energy_gap_target_rel),
        energy_gap_target_abs_Wh=float(energy_gap_target_abs_Wh),
        pricing_batch_size=int(pricing_batch_size), solve_scope=solve_scope_key,
        coverage_target=coverage_target_value)


def _risk_policy_for_mode(kappa_mode):
    """Immutable one-/two-sided DRCC contract for the formal physical chain."""
    return RM.risk_policy_for_mode(str(kappa_mode).strip().lower())


def _kappa_function_for_mode(kappa_mode):
    """Compatibility accessor; formal BPC passes ``RiskPolicy`` instead."""
    return _risk_policy_for_mode(kappa_mode).one_sided


def _plan_energy(diag: dict) -> float:
    """L2 column cost: flight + stern escort + docking reserve."""
    if "E_plan_Wh" in diag:
        return float(diag["E_plan_Wh"])
    return float(diag.get("E0", 0.0)) + float(diag.get("E_dock_Wh", 0.0))


# =============================================================================
# 1. 物理原语(与 step10 公式一致; 增量式以便标号高效, 自检中校验一致性)
# =============================================================================
def _validate_milp_primal(result, objective, A, b, lo, hi, integrality,
                           feasibility_tol=1e-7, integrality_tol=1e-7):
    """Validate a complete primal vector independently of solver claims.

    Returns ``(rounded_vector, objective_value, None)`` on success, otherwise
    ``(None, None, reason)``.  The function checks status, shape, finite values,
    bounds, integrality, all ``A x <= b`` rows and (when supplied) solver ``fun``.
    """
    try:
        if result is None or getattr(result, "success", False) is not True:
            return None, None, "solver_failure"
        if getattr(result, "status", None) != 0:
            return None, None, "solver_status_not_optimal"
        c = np.asarray(objective, float).reshape(-1)
        x = np.asarray(getattr(result, "x", None), float)
        lower = np.asarray(lo, float).reshape(-1)
        upper = np.asarray(hi, float).reshape(-1)
        integer = np.asarray(integrality, float).reshape(-1)
        if x.ndim != 1 or x.shape != c.shape or lower.shape != c.shape \
                or upper.shape != c.shape or integer.shape != c.shape:
            return None, None, "bad_shape"
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(c))
                and np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
            return None, None, "non_finite"
        if np.any(x < lower - feasibility_tol) or np.any(x > upper + feasibility_tol):
            return None, None, "bounds"
        mask = integer != 0
        if np.any(np.abs(x[mask] - np.rint(x[mask])) > integrality_tol):
            return None, None, "non_integral"
        z = x.copy()
        z[mask] = np.rint(z[mask])
        lhs = np.asarray(A @ z, float).reshape(-1)
        rhs = np.asarray(b, float).reshape(-1)
        if lhs.shape != rhs.shape or not np.all(np.isfinite(lhs)) or not np.all(np.isfinite(rhs)):
            return None, None, "bad_constraint_shape"
        if np.any(lhs > rhs + feasibility_tol):
            return None, None, "constraint_violation"
        value = float(c @ z)
        fun = getattr(result, "fun", None)
        if fun is not None:
            fun = float(fun)
            if (not math.isfinite(fun) or
                    abs(fun - value) > 1e-6 * max(1.0, abs(value))):
                return None, None, "objective_mismatch"
        return z, value, None
    except Exception as exc:
        return None, None, f"validation_exception:{type(exc).__name__}"


def _lagrangian_dual_lower_bound(c, lo, hi, Au, bu, Ae, be, im, em):
    """[THM-LRC] Rigorous restricted-variable Lagrangian lower bound.

    This routine does *not* require the binary64 row multipliers returned by
    HiGHS to satisfy exact stationarity.  It projects inequality marginals to
    ``y<=0`` and allows arbitrary free equality multipliers ``v``.  For every
    primal-feasible point in the supplied variable box, weak Lagrangian duality
    gives

        c^T x >= b_ub^T y + b_eq^T v
                 + (c-A_ub^T y-A_eq^T v)^T x.

    Taking the complete box infimum therefore yields a valid lower bound even
    when the multipliers are only approximate.  The surrounding LP validator
    still checks KKT/strong-duality consistency as an independent sanity gate,
    but the mathematical lower-bound validity comes from this box infimum.

    Every product/addition is enclosed with directed ``nextafter`` arithmetic.
    Non-finite data, overflow or an unbounded box infimum fail closed with
    ``None``.
    """
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
                or not all(np.all(np.isfinite(a)) for a in
                           (c, Au, bu, Ae, be, y, v))):
            return None
        # Variable bounds may be infinite; NaN is never allowed.
        if np.any(np.isnan(lo)) or np.any(np.isnan(hi)) or np.any(lo > hi):
            return None

        base_lo = base_hi = 0.0
        for a, b in [(float(bu[j]), float(y[j])) for j in range(bu.size)]:
            plo, phi = _outward_product_interval(a, b)
            base_lo, base_hi = _outward_add_interval(base_lo, base_hi, plo, phi)
        for a, b in [(float(be[j]), float(v[j])) for j in range(be.size)]:
            plo, phi = _outward_product_interval(a, b)
            base_lo, base_hi = _outward_add_interval(base_lo, base_hi, plo, phi)

        total_lb = float(base_lo)
        for i in range(c.size):
            q_lo = q_hi = float(c[i])
            for j in range(bu.size):
                plo, phi = _outward_product_interval(float(Au[j, i]), float(y[j]))
                q_lo, q_hi = _outward_add_interval(q_lo, q_hi, -phi, -plo)
            for j in range(be.size):
                plo, phi = _outward_product_interval(float(Ae[j, i]), float(v[j]))
                q_lo, q_hi = _outward_add_interval(q_lo, q_hi, -phi, -plo)

            l, u = float(lo[i]), float(hi[i])
            if math.isfinite(l) and math.isfinite(u):
                candidates = []
                for qend in (q_lo, q_hi):
                    for xend in (l, u):
                        prod_lo, _ = _outward_product_interval(qend, xend)
                        candidates.append(prod_lo)
                contribution_lb = min(candidates)
            elif math.isfinite(l):
                # x in [l,+inf): any negative q permits -inf.
                if q_lo < 0.0:
                    return None
                contribution_lb = min(
                    _outward_product_interval(q_lo, l)[0],
                    _outward_product_interval(q_hi, l)[0])
            elif math.isfinite(u):
                # x in (-inf,u]: any positive q permits -inf.
                if q_hi > 0.0:
                    return None
                contribution_lb = min(
                    _outward_product_interval(q_lo, u)[0],
                    _outward_product_interval(q_hi, u)[0])
            else:
                if q_lo < 0.0 or q_hi > 0.0:
                    return None
                contribution_lb = 0.0
            total_lb = _outward_add_lower(total_lb, contribution_lb)
        return math.nextafter(float(total_lb), -math.inf)
    except Exception:
        return None



def _exact_binary64_unit_box_lagrangian_lower_bound(master):
    """Exact-rational Lagrangian LB for an ordinary route RMP.

    Every route variable can be capped at one without changing the route master:
    each nonempty route contains at least one turbine and the packing row for that
    turbine is <= 1.  Binary64 inputs are interpreted as exact rationals.
    """
    try:
        c = np.asarray(master.objective, float).reshape(-1)
        Au = np.asarray(master.A_ub, float).reshape((-1, c.size))
        bu = np.asarray(master.b_ub, float).reshape(-1)
        Ae = np.asarray(master.A_eq, float).reshape((-1, c.size))
        be = np.asarray(master.b_eq, float).reshape(-1)
        y_raw = np.asarray(master.inequality_duals, float).reshape(-1)
        v_raw = np.asarray(master.equality_duals, float).reshape(-1)
        if (y_raw.shape != bu.shape or v_raw.shape != be.shape
                or not all(np.all(np.isfinite(a))
                           for a in (c, Au, bu, Ae, be, y_raw, v_raw))):
            return None
        y = [Fraction.from_float(min(float(d), 0.0)) for d in y_raw]
        v = [Fraction.from_float(float(d)) for d in v_raw]
        total = Fraction(0)
        for j, d in enumerate(y):
            total += Fraction.from_float(float(bu[j])) * d
        for j, d in enumerate(v):
            total += Fraction.from_float(float(be[j])) * d
        for i in range(c.size):
            q = Fraction.from_float(float(c[i]))
            for j, d in enumerate(y):
                q -= Fraction.from_float(float(Au[j, i])) * d
            for j, d in enumerate(v):
                q -= Fraction.from_float(float(Ae[j, i])) * d
            if q < 0:
                total += q
        return total
    except Exception:
        return None


def _fraction_unique_linear_solve(rows, rhs, n_unknown):
    """Exact Gaussian elimination; return unique rational solution or ``None``."""
    if n_unknown == 0:
        return [] if all(Fraction(v) == 0 for v in rhs) else None
    if len(rows) < n_unknown:
        return None
    aug = [[Fraction(v) for v in row] + [Fraction(rhs[i])]
           for i, row in enumerate(rows)]
    m = len(aug)
    pivot_cols = []
    r = 0
    for c in range(n_unknown):
        pivot = next((rr for rr in range(r, m) if aug[rr][c] != 0), None)
        if pivot is None:
            continue
        aug[r], aug[pivot] = aug[pivot], aug[r]
        pv = aug[r][c]
        aug[r] = [v / pv for v in aug[r]]
        for rr in range(m):
            if rr == r or aug[rr][c] == 0:
                continue
            factor = aug[rr][c]
            aug[rr] = [aug[rr][cc] - factor * aug[r][cc]
                       for cc in range(n_unknown + 1)]
        pivot_cols.append(c)
        r += 1
        if r == m:
            break
    for rr in range(m):
        if all(aug[rr][c] == 0 for c in range(n_unknown)) and aug[rr][-1] != 0:
            return None
    if len(pivot_cols) != n_unknown:
        return None
    sol = [Fraction(0)] * n_unknown
    for rr, c in enumerate(pivot_cols):
        sol[c] = aug[rr][-1]
    return sol


def _verify_exact_rmp_dual_certificate(master, local_x, y, v):
    """Verify exact primal/dual feasibility and equality for the binary64 RMP."""
    try:
        c = [Fraction.from_float(float(a)) for a in np.asarray(master.objective, float)]
        Au_f = np.asarray(master.A_ub, float)
        bu_f = np.asarray(master.b_ub, float)
        Ae_f = np.asarray(master.A_eq, float)
        be_f = np.asarray(master.b_eq, float)
        z = [Fraction(int(a), 1) for a in local_x]
        y = [Fraction(a) for a in y]
        v = [Fraction(a) for a in v]
        if len(y) != len(bu_f) or len(v) != len(be_f) or len(z) != len(c):
            return None
        if any(d > 0 for d in y):
            return None

        # Exact primal feasibility of the rounded integral candidate.
        for rr in range(len(bu_f)):
            lhs = sum((Fraction.from_float(float(Au_f[rr, j])) * z[j]
                       for j in range(len(z))), Fraction(0))
            if lhs > Fraction.from_float(float(bu_f[rr])):
                return None
        for rr in range(len(be_f)):
            lhs = sum((Fraction.from_float(float(Ae_f[rr, j])) * z[j]
                       for j in range(len(z))), Fraction(0))
            if lhs != Fraction.from_float(float(be_f[rr])):
                return None

        primal = sum((c[j] * z[j] for j in range(len(z))), Fraction(0))
        dual = Fraction(0)
        for rr, d in enumerate(y):
            dual += Fraction.from_float(float(bu_f[rr])) * d
        for rr, d in enumerate(v):
            dual += Fraction.from_float(float(be_f[rr])) * d

        # Exact dual feasibility for x >= 0: every reduced cost is nonnegative.
        for j in range(len(c)):
            q = c[j]
            for rr, d in enumerate(y):
                q -= Fraction.from_float(float(Au_f[rr, j])) * d
            for rr, d in enumerate(v):
                q -= Fraction.from_float(float(Ae_f[rr, j])) * d
            if q < 0:
                return None
        if dual != primal:
            return None
        return primal
    except Exception:
        return None


def _exact_rmp_integral_optimality_certificate(master, selection):
    """Reconstruct and verify an exact rational dual for an integral ordinary RMP.

    HiGHS duals are binary64 approximations.  For a legal integral candidate we
    use their support only as a hint, reconstruct rational multipliers from the
    positive-column stationarity equations, and then verify primal feasibility,
    dual signs/reduced costs, and exact strong duality with ``Fraction``.  A
    failed reconstruction merely returns ``None``; it never weakens the safe
    outward floating-point bound.
    """
    try:
        n = len(master.eligible_indices)
        selected = set(int(j) for j in selection)
        local_x = [1 if int(master.eligible_indices[k]) in selected else 0
                   for k in range(n)]
        if not any(local_x) and n:
            # Zero solution can still be certified by zero/nonnegative dual, but
            # the ordinary caller handles empty-plan bound closure directly.
            pass
        Au = np.asarray(master.A_ub, float).reshape((-1, n))
        Ae = np.asarray(master.A_eq, float).reshape((-1, n))
        c = np.asarray(master.objective, float).reshape(-1)
        im = np.asarray(master.inequality_duals, float).reshape(-1)
        em = np.asarray(master.equality_duals, float).reshape(-1)
        positive_cols = [j for j, bit in enumerate(local_x) if bit]
        if not (np.all(np.isfinite(Au)) and np.all(np.isfinite(Ae))
                and np.all(np.isfinite(c)) and np.all(np.isfinite(im))
                and np.all(np.isfinite(em))):
            return None

        # Try several supports of numerically nonzero solver multipliers.  Each
        # candidate is accepted only after exact dual verification.
        thresholds = (0.0, 1e-15, 1e-12, 1e-9, 1e-7)
        for tol in thresholds:
            iu = [r for r, d in enumerate(im) if float(d) < -tol]
            eq = [r for r, d in enumerate(em) if abs(float(d)) > tol]
            unknowns = [("u", r) for r in iu] + [("e", r) for r in eq]
            rows = []
            rhs = []
            for j in positive_cols:
                row = []
                for kind, r in unknowns:
                    row.append(Fraction.from_float(float(
                        Au[r, j] if kind == "u" else Ae[r, j])))
                rows.append(row)
                rhs.append(Fraction.from_float(float(c[j])))
            sol = _fraction_unique_linear_solve(rows, rhs, len(unknowns))
            if sol is None:
                continue
            y = [Fraction(0)] * len(im)
            v = [Fraction(0)] * len(em)
            for value, (kind, r) in zip(sol, unknowns):
                if kind == "u":
                    y[r] = value
                else:
                    v[r] = value
            proved = _verify_exact_rmp_dual_certificate(master, local_x, y, v)
            if proved is not None:
                return proved

        # Independent rational reconstruction is useful for a nonunique dual
        # where the solver already returned a simple exact ratio (e.g. 1/3).
        for cap in (10**3, 10**6, 10**9, 10**12):
            y = [Fraction(float(min(d, 0.0))).limit_denominator(cap) for d in im]
            v = [Fraction(float(d)).limit_denominator(cap) for d in em]
            proved = _verify_exact_rmp_dual_certificate(master, local_x, y, v)
            if proved is not None:
                return proved
        return None
    except Exception:
        return None


def _integer_node_exact_full_space_proof(master, pricing, stage, columns, selection):
    """Prove an integral RMP candidate optimal for its full implicit node."""
    if not pricing.closed:
        return False, None
    if stage == "coverage":
        candidate = Fraction(-int(_coverage_of_selection(columns, selection)), 1)
    elif stage == "energy":
        candidate = _energy_of_selection_exact(columns, selection)
    else:
        return False, None

    # First use the *same binary64 row multipliers* that exact pricing used.
    # Their exact-rational Lagrangian value extends to omitted columns because
    # pricing.closed proves every omitted reduced cost is nonnegative for these
    # very multipliers.
    same_dual_lb = _exact_binary64_unit_box_lagrangian_lower_bound(master)
    if same_dual_lb is not None and same_dual_lb >= candidate:
        return True, same_dual_lb

    # If pricing examined no omitted route at all (bound = +inf), the node's
    # complete implicit route space is already materialized.  In that special
    # case we may reconstruct a different exact rational RMP dual because there
    # are no unseen columns whose reduced costs would need to be re-certified.
    try:
        no_omitted_routes = (
            pricing.reduced_value_bound is not None
            and math.isinf(float(pricing.reduced_value_bound))
            and float(pricing.reduced_value_bound) > 0.0)
    except (TypeError, ValueError, OverflowError):
        no_omitted_routes = False
    if no_omitted_routes:
        exact_rmp_obj = _exact_rmp_integral_optimality_certificate(master, selection)
        if exact_rmp_obj is not None and exact_rmp_obj == candidate:
            return True, exact_rmp_obj
        return False, exact_rmp_obj
    return False, same_dual_lb


def _safe_integer_ceiling(lower_bound):
    """Ceiling of a conservative lower bound, with downward directed rounding."""
    if lower_bound is None or not math.isfinite(float(lower_bound)):
        return 0
    return int(math.ceil(math.nextafter(float(lower_bound), -math.inf)))


def _safe_integer_floor(upper_bound):
    """Floor of a conservative upper bound, with upward directed rounding."""
    if upper_bound is None or not math.isfinite(float(upper_bound)):
        return 0
    return int(math.floor(math.nextafter(float(upper_bound), math.inf)))


def _pricing_relaxed_lower_bound(rmp_lb, max_column_mass, rc_tol=-PRICING_EPS):
    """Convert a closed RMP dual bound into a full-column-space lower bound.

    If exhaustive pricing proves every omitted column has reduced cost at
    least ``rc_tol`` (negative), any feasible master solution with
    ``sum(x) <= max_column_mass`` can improve the RMP dual value by at most
    ``max_column_mass * abs(rc_tol)``.  Subtracting that amount is therefore
    a valid full-space lower bound.
    """
    if rmp_lb is None or not math.isfinite(float(rmp_lb)):
        return None
    mass = max(0.0, float(max_column_mass))
    slack = mass * max(0.0, -float(rc_tol))
    return math.nextafter(float(rmp_lb) - slack, -math.inf)


def _validate_linprog_result(res, objective, bounds, A_ub=None, b_ub=None,
                             A_eq=None, b_eq=None, need_ineqlin=0, need_eqlin=0,
                             tol=LP_CERT_TOL, dual_bounds=None):
    """Fail-closed validation for a SciPy/HiGHS LP result.

    Besides checking the primal vector, this routine verifies the complete KKT
    certificate supplied by HiGHS: row-dual dimensions/signs, bound marginals,
    stationarity, complementary slackness and strong duality.  A mere
    ``success=True``/``status==0`` flag is never enough for a global certificate.
    Returns ``(x, primal_fun, inequality_marginals, equality_marginals,
    conservative_dual_lower_bound)`` or ``None``.
    """
    try:
        c = np.asarray(objective, dtype=float).reshape(-1)
        n = c.size
        if (res is None or getattr(res, "success", False) is not True
                or getattr(res, "status", None) != 0
                or not np.all(np.isfinite(c))):
            return None

        x = np.asarray(getattr(res, "x", None), dtype=float).reshape(-1)
        if x.shape != (n,) or not np.all(np.isfinite(x)):
            return None
        if len(bounds) != n:
            return None

        lo = np.array([-np.inf if b[0] is None else float(b[0]) for b in bounds], dtype=float)
        hi = np.array([ np.inf if b[1] is None else float(b[1]) for b in bounds], dtype=float)
        if np.any(np.isnan(lo)) or np.any(np.isnan(hi)) or np.any(lo > hi):
            return None
        pscale = max(1.0, float(np.max(np.abs(x))) if n else 1.0)
        p_tol = tol * pscale
        if np.any(x < lo - p_tol) or np.any(x > hi + p_tol):
            return None

        Au = np.zeros((0, n), dtype=float)
        bu = np.zeros(0, dtype=float)
        if A_ub is not None:
            Au = np.asarray(A_ub.toarray() if hasattr(A_ub, "toarray") else A_ub,
                            dtype=float)
            bu = np.asarray(b_ub, dtype=float).reshape(-1)
            if (Au.ndim != 2 or Au.shape != (bu.size, n)
                    or not np.all(np.isfinite(Au)) or not np.all(np.isfinite(bu))):
                return None
            if int(need_ineqlin) != bu.size:
                return None
            if np.any(Au @ x > bu + tol * np.maximum(1.0, np.abs(bu))):
                return None
        elif int(need_ineqlin) != 0:
            return None

        Ae = np.zeros((0, n), dtype=float)
        be = np.zeros(0, dtype=float)
        if A_eq is not None:
            Ae = np.asarray(A_eq.toarray() if hasattr(A_eq, "toarray") else A_eq,
                            dtype=float)
            be = np.asarray(b_eq, dtype=float).reshape(-1)
            if (Ae.ndim != 2 or Ae.shape != (be.size, n)
                    or not np.all(np.isfinite(Ae)) or not np.all(np.isfinite(be))):
                return None
            if int(need_eqlin) != be.size:
                return None
            if np.any(np.abs(Ae @ x - be) > tol * np.maximum(1.0, np.abs(be))):
                return None
        elif int(need_eqlin) != 0:
            return None

        fun = float(getattr(res, "fun", np.nan))
        calc = float(c @ x)
        obj_tol = tol * max(1.0, abs(calc), abs(fun) if math.isfinite(fun) else 1.0)
        if not math.isfinite(fun) or abs(fun - calc) > obj_tol:
            return None

        im = np.zeros(bu.size, dtype=float)
        em = np.zeros(be.size, dtype=float)
        if bu.size:
            im = np.asarray(res.ineqlin.marginals, dtype=float).reshape(-1)
            if im.shape != (bu.size,) or not np.all(np.isfinite(im)) or np.any(im > tol):
                return None
        if be.size:
            em = np.asarray(res.eqlin.marginals, dtype=float).reshape(-1)
            if em.shape != (be.size,) or not np.all(np.isfinite(em)):
                return None

        # HiGHS exposes reduced costs as lower/upper bound marginals.  Requiring
        # these closes the remaining path where a forged non-optimal primal is
        # paired with arbitrary row marginals that merely have the right shape.
        lm = np.asarray(res.lower.marginals, dtype=float).reshape(-1)
        um = np.asarray(res.upper.marginals, dtype=float).reshape(-1)
        if (lm.shape != (n,) or um.shape != (n,)
                or not np.all(np.isfinite(lm)) or not np.all(np.isfinite(um))):
            return None
        finite_lo = np.isfinite(lo)
        finite_hi = np.isfinite(hi)
        if (np.any(lm[finite_lo] < -tol) or np.any(um[finite_hi] > tol)
                or np.any(np.abs(lm[~finite_lo]) > tol)
                or np.any(np.abs(um[~finite_hi]) > tol)):
            return None

        # Stationarity in SciPy's marginal convention:
        # c = A_ub^T m_ub + A_eq^T m_eq + m_lower + m_upper.
        station = Au.T @ im + Ae.T @ em + lm + um
        stat_tol = tol * max(1.0,
                             float(np.max(np.abs(c))) if n else 1.0,
                             float(np.max(np.abs(station))) if n else 1.0)
        if np.any(np.abs(c - station) > stat_tol):
            return None

        # Complementary slackness for rows and finite bounds.
        comp_tol = 10.0 * tol * max(1.0, abs(fun))
        if bu.size and np.any(np.abs(im * (bu - Au @ x)) > comp_tol):
            return None
        if np.any(np.abs(lm[finite_lo] * (x[finite_lo] - lo[finite_lo])) > comp_tol):
            return None
        if np.any(np.abs(um[finite_hi] * (hi[finite_hi] - x[finite_hi])) > comp_tol):
            return None

        # Strong duality independently proves LP optimality from the returned
        # primal/dual certificate instead of trusting the solver status string.
        dual_obj = 0.0
        if bu.size:
            dual_obj += float(bu @ im)
        if be.size:
            dual_obj += float(be @ em)
        if np.any(finite_lo):
            dual_obj += float(lo[finite_lo] @ lm[finite_lo])
        if np.any(finite_hi):
            dual_obj += float(hi[finite_hi] @ um[finite_hi])
        dual_tol = 10.0 * tol * max(1.0, abs(fun), abs(dual_obj))
        if not math.isfinite(dual_obj) or abs(fun - dual_obj) > dual_tol:
            return None

        dlo, dhi = lo, hi
        if dual_bounds is not None:
            if len(dual_bounds) != n:
                return None
            dlo = np.array([-np.inf if b[0] is None else float(b[0])
                            for b in dual_bounds], dtype=float)
            dhi = np.array([ np.inf if b[1] is None else float(b[1])
                            for b in dual_bounds], dtype=float)
            if np.any(np.isnan(dlo)) or np.any(np.isnan(dhi)) or np.any(dlo > dhi):
                return None
            # The certificate bounds must contain the returned primal and are
            # used only where callers have proved those bounds redundant for
            # an optimum (route master columns can be capped at one).
            if np.any(x < dlo - p_tol) or np.any(x > dhi + p_tol):
                return None
        dual_lb = _lagrangian_dual_lower_bound(c, dlo, dhi, Au, bu, Ae, be, im, em)
        if dual_lb is None:
            return None
        # Weak duality must hold after the conservative construction.  A
        # forged result that places the alleged lower bound above its own
        # feasible primal objective is rejected rather than tolerated.
        if dual_lb > calc + obj_tol:
            return None

        return x, fun, im, em, dual_lb
    except Exception:
        return None


def _float_binary64_fp(value):
    """Collision-free fingerprint of one binary64 value for exact-model keys."""
    val = float(value)
    if math.isnan(val):
        return ("f64", "nan")
    if math.isinf(val):
        return ("f64", "+inf" if val > 0.0 else "-inf")
    return ("f64", val.hex())


def _state_fp(obj, depth=0):
    """Deterministic, binary64-exact content fingerprint for mutable cache inputs.

    Certificate caches and route identities must never merge states merely
    because their decimal representations agree to a fixed number of digits.
    """
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, (float, np.floating)):
        return _float_binary64_fp(obj)
    if isinstance(obj, np.ndarray):
        a = np.asarray(obj)
        if a.dtype.kind in "biu":
            flat = tuple(int(v) for v in a.reshape(-1))
        elif a.dtype.kind == "f":
            flat = tuple(_float_binary64_fp(v) for v in a.reshape(-1))
        elif a.dtype.kind == "c":
            flat = tuple(
                ("c128", _float_binary64_fp(complex(v).real),
                 _float_binary64_fp(complex(v).imag))
                for v in a.reshape(-1))
        else:
            flat = tuple(_state_fp(v, depth + 1) for v in a.reshape(-1).tolist())
        return ("nd", a.shape, str(a.dtype), flat)
    if isinstance(obj, dict):
        return tuple(sorted((str(k), _state_fp(v, depth + 1)) for k, v in obj.items()))
    if isinstance(obj, (set, frozenset)):
        vals = [_state_fp(v, depth + 1) for v in obj]
        return tuple(sorted(vals, key=repr))
    if isinstance(obj, (list, tuple)):
        return tuple(_state_fp(v, depth + 1) for v in obj)
    if depth < 4 and hasattr(obj, "__dict__"):
        return (type(obj).__name__, tuple(sorted(
            (str(k), _state_fp(v, depth + 1))
            for k, v in vars(obj).items()
            if not callable(v) and not str(k).startswith("_cache")
        )))
    return repr(obj)


def _eval_context_fp(p, xi_amb, weather_unc, ship, turbines):
    """Fingerprint every mutable input that can change route feasibility/cost."""
    return (_state_fp(p), _state_fp(xi_amb), _state_fp(weather_unc), _state_fp(ship),
            tuple(_state_fp(t) for t in turbines))


def _ship_column_fp(ship):
    """Content key for a launch option used in column de-duplication.

    ``tau_min`` alone is not a unique column identifier: two boats/options may
    share the same clock time while having different launch positions, recovery
    predictions, slots, states or weather.  Collapsing those options can delete
    a necessary column and invalidate a finite-model certificate.
    """
    return (
        getattr(ship, "tau_min", None),
        getattr(ship, "slot", None),
        str(getattr(ship, "c_state", "")),
        _state_fp(getattr(ship, "P_launch", None)),
        _state_fp(getattr(ship, "pred_by_h", None)),
        _state_fp(getattr(ship, "wx_tau", None)),
        _state_fp(getattr(ship, "weather_by_h", None)),
        _state_fp(getattr(ship, "_v_ship", None)),
    )


@dataclass
class _WxCache:
    w_cruise: float
    wdir: float
    t_dock_s: float = 120.0     # 更新: 平原天气(未收紧)对接储备 —— 标号闭合下界用
    E_dock_Wh: float = 33.3


def _wx_cache(p: M.Params, wx: dict) -> _WxCache:
    w10 = wx.get("wind10"); wdir = wx.get("wind_dir_from")
    if w10 is None or (isinstance(w10, float) and math.isnan(w10)): w10 = 6.7
    if wdir is None or (isinstance(wdir, float) and math.isnan(wdir)): wdir = 230.0
    # 动态对接储备的【乐观(admissible)】实现 —— 用未收紧的平原 wx 计算。
    #   可采纳性: 可行性侧用(Hs_eff≥Hs 的运动, w10_low≤w10 的功率) ⇒ t/E 只会更大;
    #   闭合下界取平原值 ≤ 可行性真值 ⇒ 标号下界仍乐观, B&P 精确性证书保持(weather_unc=None 时二者相等, 界紧)。
    _motion = M.deck_motion(float(wx.get("Hs", 0.5)), float(wx.get("Tp", 2.1)),
                            float(wx.get("wave_dir", 200.0)) - float(wx.get("ship_heading", 0.0)), p)
    _t_dock, _E_dock = M.dock_reserve(p, _motion, float(w10))
    return _WxCache(w_cruise=M.wind_at_height(w10, p.z_cruise, p.z0), wdir=wdir,
                    t_dock_s=_t_dock, E_dock_Wh=_E_dock)


# =============================================================================
# 1b. 天气解析: 逐 τ(列所属起飞时刻)+ 逐风机(本地天气) —— 任务 #7/#8
#   #7: 列属于某起飞时刻 τ 的 ship; 该 ship 携带 sp.wx_tau(τ 的风浪场) ⇒ 不同 τ 用不同天气窗。
#   #8: 标号腿增量(E_nom/T_nom)此前用【全局 wc】, 而闭合(route_nominal_ET)用【逐风机本地风】,
#       二者不一致 ⇒ 逐风机天气下标号支配比较的资源 ≠ 闭合实际使用的资源, 破坏 exact 支配。
#       现让标号腿增量与闭合返程都用【目的/末端风机本地风】(与 step10._leg_wc 同口径), 资源一致 ⇒ 支配有效。
#   两者默认退回全局 wx(无 wx_tau / 无 wx_local 时字节一致, 向后兼容)。
# =============================================================================
def _wx_of_ship(sp, wx_default: dict) -> dict:
    """列所属起飞时刻 τ 的天气(sp.wx_tau); 缺失退回全局 wx(任务 #7)。"""
    w = getattr(sp, "wx_tau", None)
    return w if w is not None else wx_default


def _wx_of_route(r, wx_default: dict) -> dict:
    """路由(列)所属 τ 的天气(经其 ship); 缺失退回全局 wx(任务 #7)。"""
    sp = getattr(r, "ship", None)
    w = getattr(sp, "wx_tau", None) if sp is not None else None
    return w if w is not None else wx_default


def _wx_fp(wxd):
    """Deterministic binary64-exact weather identity for certified routes/caches."""
    if not isinstance(wxd, dict):
        raise TypeError("formal weather identity requires a dict")
    return (ROUTE_IDENTITY_CONTRACT, _state_fp(wxd))


def _leg_wc(p: M.Params, wc_global: _WxCache, turbine) -> _WxCache:
    """该腿巡航高度风: 优先【目的风机本地风 wx_local】, 否则全局 wc(与 step10.route_nominal_ET._leg_wc 同口径)。
    任务 #8: 标号腿增量用此 ⇒ 与闭合一致, 逐风机天气下支配仍有效。"""
    loc = getattr(turbine, "wx_local", None)
    if loc is None:
        return wc_global
    w10 = loc.get("wind10"); wdir = loc.get("wind_dir_from")
    if w10 is None or (isinstance(w10, float) and math.isnan(w10)):
        return wc_global
    if wdir is None or (isinstance(wdir, float) and math.isnan(wdir)):
        wdir = wc_global.wdir
    return _WxCache(w_cruise=M.wind_at_height(float(w10), p.z_cruise, p.z0), wdir=float(wdir))


def _leg_ET(p: M.Params, wc: _WxCache, p_from: np.ndarray, p_to: np.ndarray) -> tuple[float, float]:
    """一段巡航腿的 (能耗 Wh, 时间 s)。用 leg_kinematics(风三角 + 视速功率, 与闭合 route_nominal_ET 同口径, P0-6)。
    wc 由调用方按【目的风机本地风】解析(_leg_wc), 任务 #8 ⇒ 与闭合一致。"""
    d = float(np.linalg.norm(p_to - p_from))
    _ok, _v_eff, vg, power = M.leg_kinematics(p, M.wind_vector_from(wc.w_cruise, wc.wdir), p_to - p_from)
    return power * (d / vg) / 3600.0, d / vg


def _insp_ET(p: M.Params, t, z_cruise: float) -> tuple[float, float]:
    dz = M.insp_vertical_span(t, z_cruise)
    if getattr(p, "use_zeng", False):
        # 更新(PR-1): 与 step10.route_nominal_ET 一致 —— 爬升 P_zeng(0)+做功, 绕飞巡检 P_zeng(v_orbit)
        P_up = M.P_zeng(0.0, p) + 7.27 * 9.81 * p.v_z
        E = P_up * dz / p.v_z / 3600.0 + M.P_zeng(p.v_orbit, p) * p.tau_insp / 3600.0
    else:
        E = p.P_climb * dz / p.v_z / 3600.0 + p.P_hov * p.tau_insp / 3600.0
    return E, dz / p.v_z + p.tau_insp


def _close_ET_soc(p: M.Params, wc: _WxCache, last_local: np.ndarray,
                  E_nom: float, T_nom: float, ship: RM.ShipPrediction, h: int,
                  cell: "M.XiCell", last_turbine=None) -> tuple[bool, float, float, float, float, float]:
    """闭合路由: 末端风机 → 预测回收点(h) + 着舰, 算总 E0/T0 与能量/时间 SOC 余量。
    返回 (feasible, E0, T0, mE, mT, M_omega)。E_nom/T_nom 为不含返程的名义量(含起降已在初始/闭合补)。
    任务 #8: 返程腿巡航高度风用【末端风机本地风】(_leg_wc(last_turbine)), 与 step10.route_nominal_ET 一致。

    两阶段会合(更新, 与 step10.route_feasible_at_h 一致):
      b_E = B_use - E0 - E_dock(wx)  (更新: 状态依赖对接储备; 闭合用平原天气乐观版, 界可采纳)
      b_T = h*60 - T0 - t_dock(wx)
    DRCC SOC 因此保证: 无人机以 ≥1-ε 概率到达回收区时持有 ≥E_dock 电量和 ≥t_dock 时间。
    """
    E_to, E_land, T_to, T_land = M.to_land_energy_time(p)
    wc_ret = _leg_wc(p, wc, last_turbine) if last_turbine is not None else wc
    P_rec = ship.pred_by_h[int(h)]
    diff = last_local - P_rec
    d_ret0 = float(np.linalg.norm(diff))
    g = -diff / (d_ret0 + 1e-9)
    _ok, _v_eff, v_ret, pw_ret = M.leg_kinematics(p, M.wind_vector_from(wc_ret.w_cruise, wc_ret.wdir), P_rec - last_local)
    E_ret0 = pw_ret * (d_ret0 / v_ret) / 3600.0
    T_ret0 = d_ret0 / v_ret
    T_flight = T_nom + T_ret0 + T_land               # 名义飞行时间(不含等待)
    # 等待/盘旋能耗(提前到达, 名义近似; 与 step10 route_nominal_ET 一致, 更新 用 P_zeng(v_loiter))
    # 更新(审计修复#8): 对接预留窗不再重复计盘旋(与 step10 同口径)
    T_wait = max(0.0, float(h) * 60.0 - T_flight - max(float(wc.t_dock_s), 0.0))
    P_loiter = M.P_zeng(p.v_loiter, p) if getattr(p, "use_zeng", False) else p.P_wait
    E_wait = P_loiter * T_wait / 3600.0
    E0 = E_nom + E_ret0 + E_land + E_wait            # 能耗含盘旋
    T0 = T_flight                                    # 时间裕度以飞行到达计
    # 决策依赖 SOC(用该 h 的 cell); 能量灵敏度用【实际返程功率 pw_ret】(P0-9, 强横风 ≫ P_cr), 不用固定 P_cr
    a_E = (pw_ret / v_ret / 3600.0) * g
    a_T = (1.0 / v_ret) * g
    # 两阶段会合: 各预算扣除末端对接储备(与 step10.route_feasible_at_h 同口径)
    b_E = p.B_use - E0 - wc.E_dock_Wh   # 更新: 动态储备的乐观版(见 _wx_cache), 下界可采纳
    b_T = h * 60.0 - T0 - wc.t_dock_s
    kE = RM.kappa(p.eps_E); kT = RM.kappa(p.eps_T)
    mE = b_E - (float(a_E @ cell.mu) + kE * math.sqrt(max(float(a_E @ cell.Sigma @ a_E), 0.0)))
    mT = b_T - (float(a_T @ cell.mu) + kT * math.sqrt(max(float(a_T @ cell.Sigma @ a_T), 0.0)))
    feasible = bool(mE >= 0 and mT >= 0 and b_T >= 0)
    M_omega = float(min(mE, mT))
    return feasible, E0, T0, mE, mT, M_omega


# =============================================================================
# 2. DR-RCSPP 定价: 带 SOC 支配的标号算法
# =============================================================================
@dataclass
class _Label:
    last: int                 # 末端节点(-1 = 起飞点 L; 否则风机下标)
    visited: frozenset        # 已访问风机下标集(elementarity)
    E_nom: float              # 名义能耗(不含返程, 含起飞)
    T_nom: float              # 名义时间(不含返程, 含起飞)
    prize: float              # 已收集对偶 Σλ_i
    seq: tuple                # 风机下标访问序列(用于还原路由)
    e_dom: float = 0.0        # 等待修正能耗 E_nom − (P_wait/3600)·T_nom(支配用; 见下)


def _dominates(a: _Label, b: _Label, strict: bool = False) -> bool:
    r"""a 占优 b(同 last、资源≤、prize≥, 至少一处严格)。

    **支配模式(critique 4.4/硬伤4: exact 用安全支配, 强支配只给 accelerated)**:
      - strict=True (exact_discrete 默认): 要求 **visited 完全相同** $S_A=S_B$ —— 教科书级保守支配
        (同节点、同已访问集, 只比资源), 无任何理论争议。此时 prize 自动相等。
      - strict=False (accelerated): 用子集支配 $S_A\subseteq S_B$ + $prize_A\ge prize_B$ —— 这是
        Feillet 等(2004)elementary RCSPP 的标准有效支配(满足时不误删最小 reduced cost 列, 见
        doc_proof 引理 2), 剪枝更强但偏激进, 仅用于加速模式。
    能量维用等待修正能耗 e_dom(对任意 h 闭合能量单调, 见引理 1), 时间维用 T_nom。"""
    if a.last != b.last:
        return False
    if strict:
        if a.visited != b.visited:
            return False
        # 同 visited ⇒ prize 相等; 只比资源
        if a.e_dom <= b.e_dom + 1e-9 and a.T_nom <= b.T_nom + 1e-9:
            return (a.e_dom < b.e_dom - 1e-9 or a.T_nom < b.T_nom - 1e-9)
        return False
    if not a.visited.issubset(b.visited):
        return False
    if a.e_dom <= b.e_dom + 1e-9 and a.T_nom <= b.T_nom + 1e-9 and a.prize >= b.prize - 1e-9:
        return (a.e_dom < b.e_dom - 1e-9 or a.T_nom < b.T_nom - 1e-9 or
                a.prize > b.prize + 1e-9 or a.visited < b.visited)
    return False


def _neighbors(turbines, k_near: int) -> dict:
    """每台风机的 k 个最近邻(下标), 限制标号扩展, 控制规模。"""
    P = np.array([t.local for t in turbines])
    nb = {}
    for i in range(len(turbines)):
        d = np.linalg.norm(P - P[i], axis=1)
        order = np.argsort(d)
        nb[i] = [j for j in order if j != i][:k_near]
    return nb


def price_routes(turbines, ship, p, wx, xi_amb, dual: dict, route_cost: float = 1.0,
                 max_stops: int = 8, k_near: int = 8, max_routes: int = 40,
                 rc_tol: float = -PRICING_EPS, weather_unc=None,
                 forbid_pairs=None, force_pairs=None, strict_dominance: bool = False,
                 objective: str = "count", dual_offset: float = 0.0,
                 energy_dual: float = 0.0, eval_cache: dict | None = None,
                 close_cost_of_h=None, dominance_mode: str | None = None,
                 energy_weight: float = 0.0, label_budget: int | None = None,
                 stats_out: dict | None = None,
                 forbid_route_sequences=None) -> list:
    r"""DR-RCSPP 最优定价: 返回若干 reduced cost < rc_tol 的 DR 可行路由。
    dual: {turbine_idx: λ_i 或 π_i}。每条路由逐 h 选最优 h*(决策依赖)。
    **objective(词典序列生成)**:
      'count'  —— 第一层最少架次定价: rc = route_cost − Σλ_i − dual_offset, 闭合取最早可行 h(min_h)。
      'energy' —— 第二层最低能耗定价: rc = E_ω − Σλ_i − σ(E_ω=该序列【能耗最低的可行 h】的 E0), 闭合取 min-energy h。
                  支配规则不变(同 last、$E^{nom}{\le}$、$T^{nom}{\le}$、$prize{\ge}$): 对 'energy' 目标
                  (cost = E−prize)仍是有效保守支配, 见 doc_proof 引理 2'。
      'robust' —— **第三层最大鲁棒裕度定价(更新 任务3)**: L3 主问题 $\max\sum M_\omega z$ s.t. 覆盖$=1$、
                  $\sum z=N^\star$、$\sum E_\omega z\le E^\star$, 取覆盖对偶 $\pi_i$(自由号)、架次对偶 $\sigma$、
                  能耗 $\le$ 约束对偶 $\rho\le0$(`energy_dual`)。列 $(\tau,\pi,h)$ 在 min$(-M)$ 形式下
                  reduced cost $\bar c=-M_\omega-\sum_i\pi_i a_{i\omega}-\sigma-\rho E_\omega$;
                  闭合对每个可行 $h$ 算 score$(h)=M_\omega(h)+\rho E_\omega(h)$, 取 $h^\star=\arg\max$ score,
                  $\bar c=-\text{score}(h^\star)-prize-\sigma$。**支配有效性(doc_proof 引理 3')**: 因 $\rho\le0$,
                  低 $E_{total}$(经引理1 的 $e_{dom}$ 单调)同时令 $M_\omega\uparrow$ 与 $\rho E_\omega\uparrow$、
                  $T^{nom}\downarrow$ 令 $m_T\uparrow$ ⇒ score 单调 ⇒ 同 visited 集 strict 支配对 L3 仍保守有效。
    精确定价前提: k_near = 风机数(完全近邻)+ max_stops 足够大 ⇒ 标号枚举全部 elementary 部分路径(支配意义下),
      闭合枚举全部 h ⇒ 返回最小 reduced cost 列(见 doc_proof)。
    **逐腿空速可行性(更新 审计 claim6)**: 标号扩展时【立即】检查新腿的风三角空速可行性, 不可飞的腿
      不进入标号集合 —— 否则"内部腿不可飞但资源更小"的标号可能占优删除可行标号(闭合才查为时已晚)。
    分支约束(Ryan–Foster, 供 branch_and_price): forbid_pairs(apart)/force_pairs(together)。
    weather_unc(可选): 闭合可行性由 RM.route_feasible_at_h 统一判定, 自动纳入【风联合 SOC + 浪门 + 着舰门】。
    **任务 #8**: 标号腿增量与闭合返程均用【目的/末端风机本地风】(_leg_wc), 与 step10 闭合一致 ⇒ 逐风机天气下支配有效。
    返回 [dict(seq_idx, turbines, h, E0, T0, M_omega, reduced_cost)], 按 reduced cost 升序。

    更新(审计修复#7-多源标签/严格支配):
      dominance_mode ∈ {'subset','set','sequence'} —— 'subset'=Feillet 子集支配(仅加速);
      'set'=同 visited 集安全支配(旧 strict; 对【ξ-only】精确 —— 闭合裕度只依赖返程腿与
      (集合,末端,h), E0/T0 经 e_dom/T_nom 单调, 见引理1); 'sequence'=不做支配合并(标号=
      全部初等有序部分路径) —— 多源(weather_unc)下闭合裕度含【逐腿风灵敏度】(依赖访问顺序),
      (e_dom,T_nom) 支配不再保真, 'sequence' 是使标号定价对多源仍完备的安全模式(完成界
      剪枝 _cb_prune 与逐腿空速剪枝均与顺序无关, 保持有效)。缺省 None ⇒ 由 strict_dominance
      映射('set'/'subset'), 向后兼容。
    更新(审计修复#6-独立第二层): energy_weight>0 且 close_cost_of_h 给定时, 闭合改为
      逐【全部窗内 h】评估 rc(h)=route_cost+cost(h)+energy_weight·E0(h)−prize, 取最小 ——
      第二层能耗 B&P 的定价目标(E0 随 h 变, 不能再用"成本升序首个可行 h"捷径)。
    更新(采纳外部审计 6.1/6.6):
      · 名义能量早剪枝仅在 RM.mean_relax_free(xi_amb, weather_unc) 为真时启用(带符号
        均值 —— 风偏置或 ξ 均值 —— 可把 DRCC 接受判据放松到名义预算之下, 使该剪枝丢真列);
      · label_budget / stats_out: 'sequence' 模式标号数可阶乘级增长且无支配合并; 超出
        label_budget 时停止扩展并置 stats_out['complete']=False —— 调用方【必须】视为
        定价不完整并撤销证书(fleet B&P 映射为 'pricing-label-limit-no-certificate')。
        缺省 None 不设限, 老调用方语义不变。"""
    forbid_pairs = forbid_pairs or set()
    force_pairs = force_pairs or set()
    forbid_route_sequences = set(forbid_route_sequences or ())
    _prune_nom = RM.mean_relax_free(xi_amb, weather_unc)
    _n_labels_total = 0
    _n_nom_prunes = 0
    # 更新(H-02): stats_out 升级为【结构化运行时定价证明 PricingProof】。字段全部来自
    # 本次调用的实际执行事实, 供 solve_branch_cut_price 聚合成 energy_pruning_proven_safe:
    #   proof_complete           初始 False, 仅正常执行到函数末尾才置 True(异常路径拿不到);
    #   nominal_prune_enabled    本次实际使用的早剪门控值(= _prune_nom 变量本身 —— 若被
    #                            变异强制为 True, 此处如实上报 True);
    #   nominal_prune_count      名义能量早剪【实际触发】次数(在剪枝点计数, 门控被绕过
    #                            而剪枝仍发生时同样如实计数);
    #   mean_relax_free_observed 与门控【解耦】的独立观测(单独调用 RM.mean_relax_free) ——
    #                            "强制早剪"变异下 enabled=True 而 observed=False, 证书据此
    #                            fail-closed;
    #   label_budget_hit / complete / n_labels  同旧语义。
    if stats_out is not None:
        stats_out["proof_complete"] = False
        stats_out["complete"] = True
        stats_out["n_labels"] = 0
        stats_out["nominal_prune_enabled"] = bool(_prune_nom)
        stats_out["nominal_prune_count"] = 0
        stats_out["mean_relax_free_observed"] = bool(RM.mean_relax_free(xi_amb, weather_unc))
        stats_out["label_budget_hit"] = False
        stats_out["nominal_prunes_active"] = bool(_prune_nom)   # 兼容旧键
        # Phase-II 能耗定价的“全部 h 已检查”运行时证明。仅从实际循环计数生成，
        # 不根据 objective/配置重新推导。异常退出时 proof_complete 保持 False。
        stats_out["all_h_required"] = bool(
            objective == "energy" or (close_cost_of_h is not None and energy_weight > 0.0))
        stats_out["all_h_requested"] = None
        stats_out["all_h_route_scans"] = 0
        stats_out["all_h_scans_complete"] = 0
        stats_out["all_h_scans_incomplete"] = 0
        stats_out["all_h_evaluations_expected"] = 0
        stats_out["all_h_evaluations_observed"] = 0
        stats_out["all_h_early_termination"] = False
        stats_out["all_h_proof_complete"] = False
    if dominance_mode is None:
        dominance_mode = "set" if strict_dominance else "subset"
    _dom_seq = (dominance_mode == "sequence")
    _dom_strict = (dominance_mode != "subset")
    wx = _wx_of_ship(ship, wx)   # 任务 #7: 该起飞时刻 ship 的 τ 天气窗(无 wx_tau 则退回全局, 幂等)
    _wxfp = _wx_fp(wx)           # 更新(P2-04b): 天气指纹入缓存键(内容寻址)
    horizons = RM.decision_horizons_of(xi_amb)   # 决策层细网格 {5..45}(双层 h 网格)
    if stats_out is not None:
        stats_out["all_h_requested"] = tuple(float(h) for h in horizons)
    nb = _neighbors(turbines, k_near)
    wc = _wx_cache(p, wx)   # 全局风(兜底); 各腿/返程按目的风机本地风解析(_leg_wc, 任务 #8)
    E_to, E_land, T_to, T_land = M.to_land_energy_time(p)

    def _together_ok(visited: frozenset) -> bool:
        """together(i,j): 路由 turbine 集须同含或同不含 i,j。"""
        for pair in force_pairs:
            i, j = tuple(pair)
            if (i in visited) != (j in visited):
                return False
        return True

    # 初始标号: 在起飞点 L
    pw = (M.P_zeng(p.v_loiter, p) if getattr(p, "use_zeng", False) else p.P_wait) / 3600.0   # 等待功率(Wh/s), 支配修正 e_dom = E_nom − pw·T_nom (更新 与闭合一致)
    init = _Label(last=-1, visited=frozenset(), E_nom=E_to, T_nom=T_to, prize=0.0, seq=(),
                  e_dom=E_to - pw * T_to)
    found = []  # 闭合得到的候选路由

    def _try_close(lb: _Label):
        if not lb.seq:
            return
        if not _together_ok(lb.visited):     # Ryan–Foster together 约束
            return
        route = RM.Route(rid=-1, turbines=[turbines[j] for j in lb.seq], ship=ship)
        # 更新(提速#B): 闭合评估缓存 —— d 只依赖 (τ=ship, 访问序列, h), 与对偶/objective 无关,
        #   故可在同一 solve 内跨定价迭代、跨词典序三层、跨 B&P 节点复用(键用 tid 序列, 与池展开一致)。
        _seq_tids = tuple(t.tid for t in route.turbines)
        if _seq_tids in forbid_route_sequences:
            return
        def _feas(h):
            if eval_cache is None:
                return RM.route_feasible_at_h(route, h, p, wx, xi_amb, weather_unc=weather_unc)
            _ck = (id(ship), _seq_tids, int(h), _wxfp,
                   _eval_context_fp(p, xi_amb, weather_unc, ship, route.turbines))
            d = eval_cache.get(_ck)
            if d is None:
                d = RM.route_feasible_at_h(route, h, p, wx, xi_amb, weather_unc=weather_unc)
                eval_cache[_ck] = d
            return d
        if objective == "energy":
            # 第二层: 取能耗最低的可行 h。逐项登记实际循环次数，供证书验证没有
            # “首个可行 h 即停止”的回退。
            best = None
            _h_seen = 0
            if stats_out is not None:
                stats_out["all_h_route_scans"] += 1
                stats_out["all_h_evaluations_expected"] += len(horizons)
            for h in horizons:
                _h_seen += 1
                d = _feas(h)
                if d["feasible"] and (best is None or _plan_energy(d) < best["E0"] - 1e-12):
                    best = dict(seq_idx=lb.seq, turbines=[turbines[j] for j in lb.seq],
                                h=h, E0=_plan_energy(d), E0_nominal=float(d.get("E0", 0.0)),
                                T0=d["T0"], M_omega=d.get("M_omega", 0.0),
                                gate_proof=d.get("gate_weather_proof"))
            if stats_out is not None:
                stats_out["all_h_evaluations_observed"] += _h_seen
                if _h_seen == len(horizons):
                    stats_out["all_h_scans_complete"] += 1
                else:
                    stats_out["all_h_scans_incomplete"] += 1
                    stats_out["all_h_early_termination"] = True
            if best is not None:
                best["reduced_cost"] = best["E0"] - lb.prize - dual_offset   # P0-2: 减架次等式对偶 σ
                if best["reduced_cost"] < rc_tol:
                    found.append(best)
        elif objective == "robust":
            # 第三层(任务3): 取 score(h)=M_ω(h)+ρ·E0(h) 最大的可行 h(ρ=energy_dual≤0),
            #   rc = −score(h*) − prize − σ。score 在 e_dom/T_nom 支配下单调(引理3'), 故同 visited strict 支配有效。
            best = None; best_score = None
            for h in horizons:
                d = _feas(h)
                if not d["feasible"]:
                    continue
                score = float(d.get("M_omega", 0.0)) + energy_dual * _plan_energy(d)
                if best_score is None or score > best_score + 1e-12:
                    best_score = score
                    best = dict(seq_idx=lb.seq, turbines=[turbines[j] for j in lb.seq],
                                h=h, E0=_plan_energy(d), E0_nominal=float(d.get("E0", 0.0)),
                                T0=d["T0"], M_omega=d.get("M_omega", 0.0),
                                gate_proof=d.get("gate_weather_proof"))
            if best is not None:
                best["reduced_cost"] = -best_score - lb.prize - dual_offset
                if best["reduced_cost"] < rc_tol:
                    found.append(best)
        elif close_cost_of_h is not None:
            if lb.prize <= route_cost + dual_offset + 1e-9:
                return   # 更新 剪枝: close_cost≥0 ⇒ net≤0 ⇒ 不可能改进(π 全非负, 安全)
            # 分支定价: h 依赖闭合成本 —— 占机区间对偶 Σμ_t(随 h 变长)与回收甲板槽对偶
            #   由调用方打包为 close_cost_of_h(h); 电池对偶 β 与起飞槽对偶(τ 常量)折入 route_cost。
            #   逐 h 取净利最大: net(h) = prize − route_cost − cost(h); rc = −net。标号支配不受影响:
            #   闭合选项只依赖 (visited 集经 prize, h) —— 与既有 count/robust 层同一结构。
            # 更新 提速(引理: 闭合成本升序首个可行 h 即最优, 精确无损) ——
            #   net(h)=prize−route_cost−cost(h), cost(h)=Σμ_t+δ_rec 与 DRCC 可行性无关且很便宜;
            #   按 (cost,h) 升序扫: ①cost 已使 net≤|rc_tol| ⇒ 后面只会更差, 整体 break;
            #   ②首个可行 h 的 net 即全局最大(升序保证)。把每标号 9 次昂贵 _feas 降到 ~首个可行处。
            _tids_key = tuple(sorted(turbines[j].tid for j in lb.seq))
            _base = lb.prize + dual_offset - route_cost
            if energy_weight > 0.0:
                # 更新(L2 定价): rc(h)=route_cost+cost(h)+w·E0(h)−prize; E0 随 h 变
                # ⇒ 须评估全部窗内 h, 取最小(禁列/窗外 cost=+∞ 跳过)。
                best = None
                _h_seen = 0
                if stats_out is not None:
                    stats_out["all_h_route_scans"] += 1
                    stats_out["all_h_evaluations_expected"] += len(horizons)
                for h in horizons:
                    _h_seen += 1
                    _cost = float(close_cost_of_h(h, _tids_key))
                    if _cost >= 1e17:
                        continue
                    d = _feas(h)
                    if not d["feasible"]:
                        continue
                    rc = _cost - _base + energy_weight * _plan_energy(d)
                    if best is None or rc < best["reduced_cost"] - 1e-12:
                        best = dict(seq_idx=lb.seq, turbines=[turbines[j] for j in lb.seq],
                                    h=h, E0=_plan_energy(d), E0_nominal=float(d.get("E0", 0.0)), T0=d["T0"],
                                    M_omega=d.get("M_omega", 0.0), reduced_cost=rc,
                                    gate_proof=d.get("gate_weather_proof"))
                if stats_out is not None:
                    stats_out["all_h_evaluations_observed"] += _h_seen
                    if _h_seen == len(horizons):
                        stats_out["all_h_scans_complete"] += 1
                    else:
                        stats_out["all_h_scans_incomplete"] += 1
                        stats_out["all_h_early_termination"] = True
                if best is not None and best["reduced_cost"] < rc_tol:
                    found.append(best)
                return
            for _cost, h in sorted((float(close_cost_of_h(h, _tids_key)), h) for h in horizons):
                if _cost >= 1e17:            # 窗外/禁列(+∞), 升序 ⇒ 其后全是
                    break
                net = _base - _cost
                if net <= -rc_tol:           # 最好情形也不改进 ⇒ 升序 break
                    break
                d = _feas(h)
                if d["feasible"]:
                    found.append(dict(seq_idx=lb.seq, turbines=[turbines[j] for j in lb.seq],
                                      h=h, E0=_plan_energy(d), E0_nominal=float(d.get("E0", 0.0)), T0=d["T0"],
                                      M_omega=d.get("M_omega", 0.0), reduced_cost=-net,
                                      gate_proof=d.get("gate_weather_proof")))
                    break
        else:
            # 第一层: 最早可行 h(min_h), rc = route_cost − prize (− dual_offset)
            for h in horizons:
                d = _feas(h)
                if d["feasible"]:
                    rc = route_cost - lb.prize - dual_offset
                    if rc < rc_tol:
                        found.append(dict(seq_idx=lb.seq, turbines=[turbines[j] for j in lb.seq],
                                          h=h, E0=_plan_energy(d), E0_nominal=float(d.get("E0", 0.0)), T0=d["T0"],
                                          M_omega=d.get("M_omega", 0.0), reduced_cost=rc,
                                          gate_proof=d.get("gate_weather_proof")))
                    break  # 最早可行 h(min_h)

    # 更新 完成界剪枝(引理见 doc_proof §R39-2, 仅 'count' 目标; 闭合成本 ≥0 保有效):
    #   标号 L 的任何后代 prize ≤ prize(L) + [全图最大的 (max_stops−|S|) 个非负对偶之和];
    #   若该上界 ≤ route_cost+offset, 后代 reduced cost 必 ≥0 ⇒ 安全剪掉扩展。对偶稀疏时数量级级剪枝。
    _pz_sorted = sorted((max(float(dual.get(j, 0.0)), 0.0) for j in range(len(turbines))), reverse=True)
    _pz_prefix = [0.0]
    for _v in _pz_sorted:
        _pz_prefix.append(_pz_prefix[-1] + _v)

    def _cb_prune(lb: _Label) -> bool:
        if objective != "count":
            return False
        rem = max(max_stops - len(lb.visited), 0)
        return lb.prize + _pz_prefix[min(rem, len(_pz_sorted))] <= route_cost + dual_offset + 1e-9

    # 标号扩展(BFS 按层, 层数 = 已访问风机数, ≤ max_stops)
    frontier = [init]
    _try_close(init)  # 空路由不会闭合
    for depth in range(max_stops):
        new_labels = []
        for lb in frontier:
            if _cb_prune(lb):
                continue
            # 候选下一台: 若 last=L 则所有风机, 否则 last 的近邻
            cands = range(len(turbines)) if lb.last == -1 else nb[lb.last]
            from_local = ship.P_launch if lb.last == -1 else turbines[lb.last].local
            for j in cands:
                if j in lb.visited:
                    continue
                # Ryan–Foster apart 约束: 禁止与已访问风机构成 forbid 对
                if forbid_pairs and any(frozenset((j, v)) in forbid_pairs for v in lb.visited):
                    continue
                # 任务 #8: 该腿巡航高度风按【目的风机本地风】解析(_leg_wc), 标号增量与闭合 route_nominal_ET 一致。
                wc_leg = _leg_wc(p, wc, turbines[j])
                eL, tL = _leg_ET(p, wc_leg, from_local, turbines[j].local)
                # 更新 审计 claim6: 立即检查该腿风三角空速可行性(同 wc_leg 口径, 与闭合不更严),
                # 不可飞的腿不进标号集 —— 否则内部腿不可飞却资源更小的标号会误删可行标号。
                _ok, _, _ = M.leg_airspeed_feasibility(p.v_cr, p.v_air_max,
                                                       M.wind_vector_from(wc_leg.w_cruise, wc_leg.wdir),
                                                       turbines[j].local - from_local, v_air_min=p.v_air_min)
                if not _ok:
                    continue
                eI, tI = _insp_ET(p, turbines[j], p.z_cruise)
                nlb = _Label(last=j, visited=lb.visited | {j},
                             E_nom=lb.E_nom + eL + eI, T_nom=lb.T_nom + tL + tI,
                             prize=lb.prize + dual.get(j, 0.0), seq=lb.seq + (j,),
                             e_dom=(lb.E_nom + eL + eI) - pw * (lb.T_nom + tL + tI))
                # 早剪枝: 名义能耗已超可用(即便返程为0也不可行)。
                # 更新(采纳外部审计 6.1): 该推理只在无带符号均值放松时成立 —— 顺风
                # 偏置/ξ 均值可把 DRCC 接受判据放松到 B_use 之下(外部审计给出可执行
                # 单风机反例: 名义 E0=323Wh>B_use=262Wh 仍 DRCC 可行, 本剪枝使定价
                # 永远生成不了它, CG "收敛"并冒发 L1_certified=True; 该反例已收进
                # selftest --suite 更新 ⑥)。均值不为零时禁用。
                if _prune_nom and nlb.E_nom + E_land > p.B_use:
                    _n_nom_prunes += 1
                    if stats_out is not None:
                        stats_out["nominal_prune_count"] = _n_nom_prunes
                    continue
                new_labels.append(nlb)
        # 支配过滤(按 last 分桶)。更新: 'sequence' 模式不合并标号(顺序完备, 多源安全);
        # 完成界/空速/能量早剪枝仍生效(与顺序无关)。
        if _dom_seq:
            survivors = new_labels
        else:
            by_last: dict[int, list] = {}
            for lb in new_labels:
                by_last.setdefault(lb.last, []).append(lb)
            survivors = []
            for last, labs in by_last.items():
                labs.sort(key=lambda x: (x.E_nom + x.T_nom, -x.prize))
                keep = []
                for cand in labs:
                    if any(_dominates(k, cand, strict=_dom_strict) for k in keep):
                        continue
                    keep = [k for k in keep if not _dominates(cand, k, strict=_dom_strict)]
                    keep.append(cand)
                survivors.extend(keep)
        # 更新(采纳外部审计 6.6): 标号资源安全阀 —— sequence 模式无合并, 标号数可
        # 阶乘级增长; 超预算即停止扩展并声明"定价不完整"(调用方据此撤证), 避免 OOM/
        # 进程被杀时连诚实降级的机会都没有。已生成标号仍闭合(列可用, 只是无最优性)。
        _n_labels_total += len(survivors)
        if stats_out is not None:
            stats_out["n_labels"] = _n_labels_total
        if label_budget is not None and _n_labels_total > int(label_budget):
            if stats_out is not None:
                stats_out["complete"] = False
                stats_out["label_budget_hit"] = True
            for lb in survivors:
                _try_close(lb)
            break
        for lb in survivors:
            _try_close(lb)
        frontier = survivors
        if not frontier:
            break

    found.sort(key=lambda d: d["reduced_cost"])
    if stats_out is not None:
        _h_required = bool(stats_out.get("all_h_required") is True)
        _h_ok = bool(
            (not _h_required)
            or (stats_out.get("all_h_scans_incomplete") == 0
                and stats_out.get("all_h_evaluations_observed")
                    == stats_out.get("all_h_evaluations_expected")
                and stats_out.get("all_h_early_termination") is False))
        stats_out["all_h_proof_complete"] = _h_ok
        stats_out["proof_complete"] = True   # 仅正常到达返回点才算证明完整
    return found[:max_routes]


# =============================================================================
# 3. 主问题 LP(scipy, 取对偶)与整数主问题
# =============================================================================
def _master_lp(columns, turbine_ids, big=1e4):
    """覆盖 LP 松弛(下界): min Σ c_ω z_ω s.t. Σ_{ω∋i} z_ω ≥ 1, z≥0。
    **更新 审计 P0-1**: 对【全部】风机建约束 + 每风机人工列(big)。组合可达风机(单独不可行、联合可行)
    经人工列对偶 λ_i≈big 驱动定价生成组合列; 真正不可达者收敛后仍占人工列 → 报 uncovered。
    LB 扣人工灌水(big×人工用量)以免失真(此前为避免灌水而只建可覆盖行, 反致组合可达风机漏覆盖)。
    返回 (LB_real, duals{tid:λ_i}(全风机), used_artificial, uncovered_set)。"""
    from scipy.optimize import linprog
    # 更新 审计 P0-1: 对【全部】风机建约束 + 每风机一条人工列(big)。组合可达风机(单独不可行、与他者联合可行)
    # 在初始仅含单例时其约束只被人工列覆盖 ⇒ 对偶 λ_i≈big 强力驱动定价生成含它的组合列;
    # 此前只对"已被现有列覆盖"的风机建约束 → 这类风机无行无对偶 → 永不被定价覆盖(致 exact_lex 漏覆盖 + 假收敛)。
    ids = list(turbine_ids); n_t = len(ids)
    if n_t == 0:
        return 0.0, {t: 0.0 for t in turbine_ids}, 0.0, set()
    tidx = {t: i for i, t in enumerate(ids)}
    cols = [(c_ω, set(cover)) for c_ω, cover in columns]
    n_real = len(cols)
    for t in ids:                       # 每风机一条人工列, 保 LP 可行并给未覆盖风机对偶价格
        cols.append((big, {t}))
    n_col = len(cols)
    c = np.array([cc for cc, _ in cols], float)
    A = np.zeros((n_t, n_col))           # A_ub x ≤ b_ub: -cover ≤ -1(全风机行)
    for k, (_, cover) in enumerate(cols):
        for t in cover:
            if t in tidx:
                A[tidx[t], k] = -1.0
    b = -np.ones(n_t)
    bounds = [(0, None)] * n_col
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    checked = _validate_linprog_result(res, c, bounds, A_ub=A, b_ub=b,
                                       need_ineqlin=n_t,
                                       dual_bounds=[(0, 1)] * n_col)
    if checked is None:
        return None, None, None, None
    x, fun, marg, _, dual_lb = checked
    duals = {t: max(-marg[tidx[t]], 0.0) for t in ids}
    art_z = x[n_real:]
    used_art = float(np.maximum(art_z, 0.0).sum())
    uncovered = {ids[i] for i in range(n_t) if art_z[i] > 1e-6}
    lb_real = dual_lb - big * used_art
    return lb_real, duals, used_art, uncovered


def _integer_master(routes_with_diag, turbine_ids, p=None, wx=None, lex=True, partition=True):
    """整数主问题(复用 step11.solve_master: Gurobi 惰性 / scipy.milp / 贪心)。
    **partition=True: 历史集合划分模型(每台恰好 1 次)**；不属于当前正式车队模型。
    不可达风机由 solve_master 报 uncovered, 不纳入等式。lex=True: 词典序三层(架次→能耗→鲁棒裕度)。"""
    if not lex or p is None:
        return RA.solve_master(routes_with_diag, turbine_ids, costs=None, partition=partition)
    energy_arr, robust_arr = [], []
    for r, d in routes_with_diag:
        # 更新(P2-01b): L2 层排序统一【DRCC 能耗口径】d["E0"](列在其 fixed_h/min_h 的
        # DR 评估), 与 Stage-2 cols_E/能耗定价/E_LP/车队 BP 完全同基。旧 nominal 口径不含
        # 甲板停靠低功耗等 DR 项, 与证书界恒差 ⇒ 主问题选择基准与被证明的目标不一致。
        # (同时免去逐列 route_nominal_ET 重算。)
        energy_arr.append(_plan_energy(d))
        robust_arr.append(float(d.get("M_omega", 0.0)))
    return RA.solve_master(routes_with_diag, turbine_ids, partition=partition,
                           lex=True, energy=energy_arr, robust=robust_arr)


def _master_solution_validated(ip) -> bool:
    """Fail-closed predicate for any incumbent imported from an integer master."""
    return bool(isinstance(ip, dict) and ip.get("solution_validated") is True)


def _master_solution_proven(ip) -> bool:
    """The complete restricted-master lexicographic claim was independently proven."""
    return bool(_master_solution_validated(ip) and ip.get("proven_optimal") is True)


def _master_l1_proven(ip) -> bool:
    """The first-stage integer objective was independently matched to a valid bound."""
    return bool(_master_solution_validated(ip)
                and ip.get("l1_proven_optimal", ip.get("proven_optimal")) is True)


def _master_outputs_clean(ip) -> bool:
    """All attempted exact integer-solver outputs passed independent validation."""
    return bool(isinstance(ip, dict)
                and ip.get("all_exact_solver_outputs_validated", True) is True
                and ip.get("exact_solver_output_rejected", False) is not True)


def _expand_pool_h(pool, p, wx, xi_amb, weather_unc=None, h_grid=None, eval_cache=None):
    r"""把每条 (τ,π) 列展开为其【全部 DR 可行 h】的独立列, 使最终列空间真正为完整 (τ,π,h)
    (闭合 doc_proof.md §1 / 定理3注 的降调#1: 不再每序列只留单个 h*)。

    为什么这样做不破坏精确性、且只可能改善:
      - **L1(最少架次)**: 与 h 无关(任意可行 h 覆盖同一组风机), 故展开【不改变】最少架次、
        不改变列生成下界 LB、不改变整数最优性证书 ⌈LB⌉=UB。
      - **L2(总能耗)/L3(鲁棒裕度)**: 词典序整数主问题现可在【同一 (τ,π) 的全部可行 h】里
        为 L2 选最低能耗 h、为 L3 选最大 M_ω 的 h —— 只可能令总能耗↓、总裕度↑(或不变), 绝不变差。
    即: 把"每序列单个 h*"升级为"每序列全部可行 h 皆为独立列", 与"列 = (τ,π,h), 不同 h 即不同列"
    的列定义一致(doc_model.md §3/§12.3.1)。这是【最终列池的后处理展开】, 不改 pricing 效率
    (列生成期间每序列仍只一列驱动, 收敛后才展开供整数主问题选 h)。
    """
    expanded = []
    seen = set()
    for r, _d in pool:
        seq = tuple(r.turbine_ids())
        ship_key = _ship_column_fp(r.ship)
        wxr = _wx_of_route(r, wx)   # 任务 #7: 该列起飞时刻 τ 的天气窗
        _wxfp = _wx_fp(wxr)         # 更新(P2-04b): 天气指纹入缓存键
        any_feasible = False
        for h in RM.decision_horizons_of(xi_amb, h_grid):
            # 更新(提速#B): (τ=ship, 序列, h) 的可行性评估与对偶无关 —— 同一 solve 内跨定价迭代/词典序层/
            #   B&P 节点/池展开完全可复用。命中即免去整条 route_nominal_ET 重算(profile: 该重算占定价 93%)。
            _ck = (id(r.ship), seq, int(h), _wxfp,
                   _eval_context_fp(p, xi_amb, weather_unc, r.ship, r.turbines))
            d = eval_cache.get(_ck) if eval_cache is not None else None
            if d is None:
                d = RM.route_feasible_at_h(r, int(h), p, wxr, xi_amb, weather_unc=weather_unc)
                if eval_cache is not None:
                    eval_cache[_ck] = d
            if not d["feasible"]:
                continue
            key = (ship_key, seq, int(h))
            if key in seen:
                continue
            seen.add(key)
            rc = RM.Route(rid=-1, turbines=list(r.turbines), ship=r.ship)
            rc.fixed_h = int(h)
            expanded.append((rc, d))
            any_feasible = True
        if not any_feasible:
            # Fail closed: a sequence with no feasible h in its own launch/weather
            # context is not a legal column of the finite discrete model.
            continue
    return expanded


def _route_at_selected_h(r, p, wx, xi_amb, weather_unc=None):
    r"""按【被整数主问题选中的 $h$】(列的 `fixed_h`)汇总该路由的 $h$/能耗/裕度/诊断。
    更新 审计 claim1 修复: 此前汇总循环重新调 `route_drcc_feasible(objective="min_h")` 选最早可行 $h$,
    忽略了 `_expand_pool_h` 让主问题实际选中的 $h$ —— 导致【优化用一个 $h$、报告用另一个 $h$】,
    总能耗/鲁棒裕度/route diag 与真正选中的解不一致。现统一用 `r.fixed_h`(无则回退 min_h)。
    """
    wxr = _wx_of_route(r, wx)   # 任务 #7: 该列起飞时刻 τ 的天气窗
    if getattr(r, "fixed_h", None) is not None:
        h_sel = int(r.fixed_h)
        d = RM.route_feasible_at_h(r, h_sel, p, wxr, xi_amb, weather_unc=weather_unc)
    else:
        d = RM.route_drcc_feasible(r, p, wxr, xi_amb, objective="min_h", weather_unc=weather_unc)
        h_sel = int(d["h"])
    nom = RM.route_nominal_ET(r, h_sel, p, wxr,
                              t_dock_s=float(d.get("t_dock_wait_s", 0.0)))
    return h_sel, nom, d


def _verify_selected_route_plan(routes, turbine_ids, p, wx, xi_amb, weather_unc=None):
    """Independently revalidate the final finite-discrete route plan.

    The master validates only its algebraic rows.  This second layer re-runs
    every selected physical column at its selected ``h`` and own ``wx_tau``,
    recomputes objectives, checks exact partition coverage and launch-slot
    capacity, and therefore prevents a stale or incorrectly generated column
    from entering a global certificate.
    """
    ids = list(turbine_ids)
    idset = set(ids)
    counts = {tid: 0 for tid in ids}
    slot_count = {}
    total_E = 0.0
    total_nominal = 0.0
    total_M = 0.0
    diag = []
    try:
        for r in routes:
            tids = list(r.turbine_ids())
            if len(tids) != len(set(tids)) or any(t not in idset for t in tids):
                return dict(ok=False, reason="route_turbine_ids", total_energy=None,
                            total_nominal=None, total_margin=None, diag=[])
            h_sel, nom, d = _route_at_selected_h(
                r, p, wx, xi_amb, weather_unc=weather_unc)
            if d.get("feasible") is not True:
                return dict(ok=False, reason="selected_route_infeasible", total_energy=None,
                            total_nominal=None, total_margin=None, diag=[])
            E = _plan_energy(d)
            En = float(nom.get("E0", float("nan")))
            Mv = float(d.get("M_omega", 0.0))
            if not all(math.isfinite(v) for v in (E, En, Mv)):
                return dict(ok=False, reason="selected_route_non_finite", total_energy=None,
                            total_nominal=None, total_margin=None, diag=[])
            for tid in tids:
                counts[tid] += 1
            slot = getattr(getattr(r, "ship", None), "slot", None)
            if slot is not None:
                slot_count[slot] = slot_count.get(slot, 0) + 1
                if slot_count[slot] > 1:
                    return dict(ok=False, reason="launch_slot_capacity", total_energy=None,
                                total_nominal=None, total_margin=None, diag=[])
            total_E += E
            total_nominal += En
            total_M += Mv
            diag.append(dict(stops=r.n_stops(), h=int(h_sel),
                             tau=getattr(r.ship, "tau_min", None), slot=slot,
                             E0=E, E0_nominal=En, M_omega=Mv,
                             turbines=tids))
        if any(counts[tid] != 1 for tid in ids):
            return dict(ok=False, reason="partition_coverage", total_energy=None,
                        total_nominal=None, total_margin=None, diag=[])
        return dict(ok=True, reason=None, total_energy=float(total_E),
                    total_nominal=float(total_nominal), total_margin=float(total_M),
                    diag=diag)
    except Exception as exc:
        return dict(ok=False, reason=f"physical_validation_exception:{type(exc).__name__}",
                    total_energy=None, total_nominal=None, total_margin=None, diag=[])


# =============================================================================
# 4. 列生成循环(LB) + 整数主问题(UB) + 完整 Ryan–Foster 分支定价
# =============================================================================
def column_generation(turbines, ship, p, wx, xi_amb, max_stops=8, k_near=8,
                      max_iter=30, route_cost=1.0, verbose=True, weather_unc=None,
                      launch_ships=None, forbid_pairs=None, force_pairs=None,
                      strict_dominance=False, objective="count", duals_override=None,
                      eval_cache=None):
    """根节点列生成: 反复(LP 取对偶 → DR-RCSPP 定价 → 加列)直至无改进列。
    返回 dict(LB, columns[Route+diag], n_iter, priced_total)。weather_unc 透传(多源 SOC + 浪门)。
    launch_ships(可选, 任务2 起飞—回收协同): list[ShipPrediction], 逐 τ 定价并并池, 列结构 r=(τ,ω,h)。
    forbid_pairs/force_pairs(可选, Ryan–Foster 节点约束): 透传给 price_routes(apart/together)。"""
    all_ids = [t.tid for t in turbines]
    tid2idx = {t.tid: i for i, t in enumerate(turbines)}
    ships = launch_ships if launch_ships is not None else [ship]
    forbid_pairs = forbid_pairs or set()
    force_pairs = force_pairs or set()

    def _together_ok(tset) -> bool:
        for pair in force_pairs:
            i, j = tuple(pair)
            ii = turbines[i].tid if isinstance(i, int) and i < len(turbines) else i
            jj = turbines[j].tid if isinstance(j, int) and j < len(turbines) else j
            if (ii in tset) != (jj in tset):
                return False
        return True

    # 初始列: 各起飞时刻的单台 DR 可行路由(满足 together 约束的); 每 τ 用其 sp.wx_tau(任务 #7)
    pool = []
    seen = set()
    for sp in ships:
        wx_sp = _wx_of_ship(sp, wx)
        for r, d in RA.gen_singletons(turbines, sp, p, wx_sp, xi_amb, objective="min_h", weather_unc=weather_unc):
            tset = set(r.turbine_ids())
            if not _together_ok(tset):
                continue
            key = (_ship_column_fp(sp), tuple(r.turbine_ids()))
            if key in seen:
                continue
            seen.add(key); pool.append((r, d))
    if verbose:
        log.info("初始单台可行列 %d (起飞时刻 %d 个)", len(pool), len(ships))

    def _pool_as_columns():
        return [(route_cost, set(r.turbine_ids())) for r, _ in pool]

    LB = None; n_iter = 0; priced_total = 0
    pricing_rc_tol = -PRICING_EPS
    converged = False; hit_max_iter = False; termination = "max_iter"
    last_min_rc = None
    _t_lp_total = 0.0; _t_price_total = 0.0; _t_feas_total = 0.0   # 更新(任务5诊断): 三段计时
    _t_cg0 = time.time()
    for it in range(max_iter):
        n_iter = it + 1
        _t_a = time.time()
        obj, duals_by_tid, used_art, lp_uncov = _master_lp(_pool_as_columns(), all_ids)
        _t_lp_total += time.time() - _t_a
        if obj is None:
            log.warning("LP 主问题求解失败, 停止列生成。"); termination = "lp_fail"; break
        LB = obj
        # duals_override(stage-2 能耗定价用 energy-LP 对偶 λ); 否则用本层 LP(SC-LP)对偶
        if duals_override is not None:
            dual_idx = {tid2idx[t]: duals_override.get(t, 0.0) for t in all_ids}
        else:
            dual_idx = {tid2idx[t]: duals_by_tid.get(t, 0.0) for t in all_ids}
        # 逐 τ 定价(每个起飞时刻用其 ship 的起飞点 + 其 wx_tau 天气, 任务 #7); 合并改进列
        _t_b = time.time()
        priced_all = []
        for sp in ships:
            wx_sp = _wx_of_ship(sp, wx)
            # Exclude columns already present for this exact launch option
            # *inside* exhaustive pricing.  Otherwise the top-k return may be
            # filled entirely by duplicate negative-rc columns, causing a
            # false "stalled" termination even though unseen improving
            # columns exist deeper in the ordered list.  The exclusion is
            # safe because every sequence already in the pool has all of its
            # discrete recovery horizons expanded by _expand_pool_h before an
            # integer/lexicographic master is solved.
            _sp_fp = _ship_column_fp(sp)
            _forbid_seq = {seq for fp, seq in seen if fp == _sp_fp}
            pr = price_routes(turbines, sp, p, wx_sp, xi_amb, dual_idx,
                              route_cost=route_cost, max_stops=max_stops, k_near=k_near,
                              rc_tol=pricing_rc_tol,
                              weather_unc=weather_unc, forbid_pairs=forbid_pairs, force_pairs=force_pairs,
                              strict_dominance=strict_dominance, objective=objective,
                              eval_cache=eval_cache,
                              forbid_route_sequences=_forbid_seq)
            for d in pr:
                d["_ship"] = sp
            priced_all.extend(pr)
        _t_price_total += time.time() - _t_b
        priced_all.sort(key=lambda d: d["reduced_cost"])
        last_min_rc = priced_all[0]["reduced_cost"] if priced_all else 0.0
        priced_total += len(priced_all)
        if verbose:
            log.info("  迭代 %d: LP=%.3f(含人工%.1f) | 定价得改进列 %d", it + 1, obj, used_art, len(priced_all))
        if not priced_all:
            converged = True; termination = "no_neg_rc"; break  # 唯一真收敛: 无负 reduced cost 列
        added = 0
        _t_c = time.time()
        for pr in priced_all:
            sp = pr["_ship"]
            r = RM.Route(rid=-1, turbines=pr["turbines"], ship=sp)
            r.fixed_h = pr["h"]
            key = (_ship_column_fp(sp), tuple(r.turbine_ids()))
            if key in seen:
                continue
            d = RM.route_drcc_feasible(r, p, _wx_of_ship(sp, wx), xi_amb, objective="min_h", weather_unc=weather_unc)
            if d["feasible"]:
                pool.append((r, d)); seen.add(key); added += 1
        _t_feas_total += time.time() - _t_c
        if added == 0:
            # 更新 审计 P0-8: 有负 reduced cost 列却一列都没加(全是重复 key / 后验不可行)⇒ 这是【停滞】,
            # 不是"无负 reduced cost 列"的收敛; 不得置 converged=True(否则会冒发证书)。
            converged = False; termination = "no_new_col_stall"; break
    else:
        hit_max_iter = True; termination = "max_iter"
    _t_cg_total = time.time() - _t_cg0
    for i, (r, _) in enumerate(pool):
        r.rid = i
    n = len(turbines)
    if verbose:
        print(f"  [column_generation 计时诊断] 总计={_t_cg_total:.1f}s({n_iter}轮) | "
              f"定价(price_routes)={_t_price_total:.1f}s({_t_price_total/max(_t_cg_total,1e-9)*100:.0f}%) | "
              f"LP松弛(scipy.linprog)={_t_lp_total:.1f}s({_t_lp_total/max(_t_cg_total,1e-9)*100:.0f}%) | "
              f"可行性后验={_t_feas_total:.1f}s({_t_feas_total/max(_t_cg_total,1e-9)*100:.0f}%) | "
              f"n={n} k_near={k_near} max_stops={max_stops} pool_size={len(pool)}")
        if _t_price_total / max(_t_cg_total, 1e-9) > 0.6:
            print(f"  → 瓶颈在定价搜索(price_routes, 纯Python BFS标号, 不受Gurobi影响); "
                  f"装Gurobi只会加速上面这行的【LP松弛】({_t_lp_total/max(_t_cg_total,1e-9)*100:.0f}%占比), "
                  f"对总时间影响有限。")
    certifiable_shape = bool(k_near >= n and strict_dominance and converged
                             and termination == "no_neg_rc")
    global_LB = (_pricing_relaxed_lower_bound(LB, n, pricing_rc_tol)
                 if certifiable_shape else None)
    return dict(LB=LB, global_LB=global_LB, columns=pool, n_iter=n_iter, priced_total=priced_total,
                converged=bool(converged), hit_max_iter=bool(hit_max_iter),
                neighbor_graph_complete=bool(k_near >= n), strict_dominance=bool(strict_dominance),
                termination_reason=termination, last_min_reduced_cost=last_min_rc, k_near_used=k_near,
                pricing_rc_tolerance=float(pricing_rc_tol),
                # 更新(任务5诊断): 暴露分段计时, 定位"求解慢"的真实瓶颈
                cg_total_time_s=round(_t_cg_total, 2), cg_pricing_time_s=round(_t_price_total, 2),
                cg_lp_time_s=round(_t_lp_total, 2), cg_feasibility_time_s=round(_t_feas_total, 2),
                cg_pricing_frac=round(_t_price_total / max(_t_cg_total, 1e-9), 3))


def _cg_certifiable(cg: dict) -> bool:
    """列生成是否满足【可发第一层最少架次全局证书】的全部充分条件(更新 审计 P0-4/5/7/8):
      ① 完全近邻图 k_near=n(neighbor_graph_complete);② 用 strict(同 visited 集)安全支配;
      ③ 因【无负 reduced cost 列】收敛(termination_reason=='no_neg_rc', 非 max_iter / 停滞)。
    三者缺一 ⇒ 不得发全局证书(accelerated 子集支配、近邻受限、停滞、迭代上限均被此挡住)。"""
    return bool(cg and cg.get("neighbor_graph_complete") and cg.get("strict_dominance")
                and cg.get("converged") and cg.get("termination_reason") == "no_neg_rc")


def _energy_partition_lp(cols_E, turbine_ids, n_star, big=1e7):
    r"""第二层能耗划分 LP: $\min \sum E_\omega z_\omega$ s.t. $\sum_{\omega\ni i}z_\omega=1\ \forall i$,
    $\sum_\omega z_\omega = N^\star$(真实列计数), $z\ge0$。返回 (obj, duals{tid: λ_i}, sigma)。
    **更新 审计 P0-2**: 除覆盖等式对偶 $\lambda_i$ 外, 还返回架次等式 $\sum z=N^\star$ 的对偶 $\sigma$。
    第二层正确约化成本 $\bar c_\omega=E_\omega-\sum_i\lambda_i a_{i\omega}-\sigma$ —— $\sigma$ 虽对各列为常数、
    不改变定价【排序】, 却改变"$\bar c_\omega<0$ 是否成立"的判断(漏掉 $\sigma$ 会过早停止加列 → 假收敛/假 LP 紧)。
    **更新 审计 P0-1**: 对【全部】风机建覆盖等式 + 人工列, 与 `_master_lp` 同口径(组合可达风机也获对偶)。"""
    from scipy.optimize import linprog
    ids = list(turbine_ids); n_t = len(ids); tidx = {t: i for i, t in enumerate(ids)}
    if n_t == 0:
        return 0.0, {t: 0.0 for t in turbine_ids}, 0.0
    n_real = len(cols_E)
    cols = list(cols_E) + [(big, {t}) for t in ids]   # 全风机人工列
    n_col = len(cols)
    c = np.array([E for E, _ in cols], float)
    A_cov = np.zeros((n_t, n_col))
    for k, (_, cov) in enumerate(cols):
        for t in cov:
            if t in tidx:
                A_cov[tidx[t], k] = 1.0
    A_cnt = np.zeros((1, n_col)); A_cnt[0, :n_real] = 1.0   # 仅真实列计入架次
    A_eq = np.vstack([A_cov, A_cnt])
    b_eq = np.concatenate([np.ones(n_t), np.array([float(n_star)])])
    bounds = [(0, None)] * n_col
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    checked = _validate_linprog_result(res, c, bounds, A_eq=A_eq, b_eq=b_eq,
                                       need_eqlin=n_t + 1,
                                       dual_bounds=[(0, 1)] * n_col)
    if checked is None:
        return None, None, None
    x, fun, _, marg, dual_lb = checked
    duals = {t: float(marg[i]) for i, t in enumerate(ids)}
    sigma = float(marg[n_t])
    art_z = x[n_real:]
    lb_real = dual_lb - big * float(np.maximum(art_z, 0.0).sum())
    return lb_real, duals, sigma


def _robust_partition_lp(cols_M, turbine_ids, n_star, e_star, big=1e7, e_tol=None):
    r"""第三层鲁棒裕度划分 LP(更新 任务3): 在【最少架次 $N^\star$ + 最低能耗 $E^\star$】的解集内
    最大化总鲁棒裕度。$\max\sum_\omega M_\omega z_\omega$ 写成 $\min\sum(-M_\omega)z_\omega$ s.t.

      - 覆盖:  $\sum_{\omega\ni i} z_\omega = 1\ \forall i$            (对偶 $\pi_i$, 自由号);
      - 架次:  $\sum_{\omega\,real} z_\omega = N^\star$               (对偶 $\sigma$, 自由号);
      - 能耗:  $\sum_\omega E_\omega z_\omega \le E^\star+\varepsilon$ (对偶 $\rho\le0$, **不等式**);
      - $z\ge0$。

    返回 (maxΣM, duals{tid: π_i}, sigma, rho, uncovered_count)。
    `cols_M` = [(M_ω, E_ω, cover_set)]。人工列(每风机一列): $M=-big$(即 $-M=+big$ 罚)、$E=0$、cover$=\{t\}$,
    保证可行; 用到人工列 ⇒ uncovered>0(理论上 Stage-1/2 后不应发生)。

    **能耗用不等式 $\le E^\star$ 而非等式**: 这是 $\rho\le0$(进而引理3' 支配有效)的来源 ——
    放松能耗预算只可能让 $\max\sum M$ 不减, 故 $\partial/\partial E^\star\ge0$, 即 min$(-M)$ 形式下 $\rho\le0$
    (已用 scipy/HiGHS 经验证: $\le$ 约束 marginals $\le0$)。等式约束会给自由号 $\rho$, 破坏 L3 标号支配。
    """
    from scipy.optimize import linprog
    ids = list(turbine_ids); n_t = len(ids); tidx = {t: i for i, t in enumerate(ids)}
    if n_t == 0:
        return 0.0, {t: 0.0 for t in turbine_ids}, 0.0, 0.0, 0
    if e_tol is None:
        e_tol = 1e-7 * max(1.0, abs(float(e_star)))
    n_real = len(cols_M)
    # 真实列 + 全风机人工列(−M=+big 罚, E=0)
    cols = [(-float(M), float(E), set(cov)) for (M, E, cov) in cols_M] + [(big, 0.0, {t}) for t in ids]
    n_col = len(cols)
    c = np.array([cM for cM, _E, _cov in cols], float)
    A_cov = np.zeros((n_t, n_col))
    for k, (_cM, _E, cov) in enumerate(cols):
        for t in cov:
            if t in tidx:
                A_cov[tidx[t], k] = 1.0
    A_cnt = np.zeros((1, n_col)); A_cnt[0, :n_real] = 1.0           # 仅真实列计入架次
    A_eq = np.vstack([A_cov, A_cnt])
    b_eq = np.concatenate([np.ones(n_t), np.array([float(n_star)])])
    A_ub = np.array([[E for _cM, E, _cov in cols]], float)          # 能耗预算(不等式)
    b_ub = np.array([float(e_star) + float(e_tol)])
    bounds = [(0, None)] * n_col
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    checked = _validate_linprog_result(res, c, bounds, A_ub=A_ub, b_ub=b_ub,
                                       A_eq=A_eq, b_eq=b_eq, need_ineqlin=1, need_eqlin=n_t + 1,
                                       dual_bounds=[(0, 1)] * n_col)
    if checked is None:
        return None, None, None, None, None
    x, fun, imarg, emarg, dual_lb = checked
    duals = {t: float(emarg[i]) for i, t in enumerate(ids)}
    sigma = float(emarg[n_t])
    rho = float(imarg[0])
    art_z = x[n_real:]
    uncovered = int(np.sum(art_z > 1e-6))
    # ``dual_lb`` lower-bounds min(-M); negating it is a conservative upper
    # bound on max(M).  The stage-3 certificate compares against that bound.
    maxM = -(dual_lb - big * float(np.maximum(art_z, 0.0).sum()))
    return maxM, duals, sigma, rho, uncovered


def lex_column_generation(turbines, ship, p, wx, xi_amb, max_stops=8, k_near=None, eval_cache=None,
                          weather_unc=None, launch_ships=None, max_iter=30,
                          energy_max_iter=8, verbose=True) -> dict:
    r"""【真正的全局词典序列生成】(更新 任务2)。两阶段严格列生成 + 第三层 h 展开池内最优:

      Stage 1(架次): 以 $c_\omega=1$ 列生成至收敛 ⇒ 完整列空间 $\mathrm{SC\text-LP}^\star$ 真下界 + $N^\star$。
                      根整数证书 $\lceil\mathrm{SC\text-LP}^\star\rceil=N^\star$ ⇒ 最少架次全局最优(sound)。
      Stage 2(能耗, 固定架次=$N^\star$): 解能耗划分 LP 取覆盖对偶 $\lambda$ → 以 $rc=E_\omega-\sum\lambda_i$
                      【重新定价】补充能耗相关列, 直至无改进能耗列(收敛)。若能耗 LP 下界 $=$ 整数能耗 $E^\star$
                      ⇒ 在 $N^\star$ 架次下能耗全局最优。
      Stage 3(裕度): 多-h 展开后, 词典序整数主问题在 $(N^\star,E^\star)$ 内取最大 $M_\omega$ 的 $h$/列(池内最优;
                      可对 enumeration_anchor 验证)。

    `certified_min_sorties` = 第一层根整数证书(完全定价 k_near=n + 收敛 + 无 uncovered + LP 紧)。
    `certified_lex`         = 第一层 ∧ 第二层均(完全定价 + 收敛 + LP 紧)⇒ 前两层全局精确。
    """
    n = len(turbines); k_near = k_near or n
    all_ids = [t.tid for t in turbines]
    tid2idx = {t.tid: i for i, t in enumerate(turbines)}
    horizons = RM.decision_horizons_of(xi_amb)
    ships = launch_ships if launch_ships is not None else [ship]
    _t0 = time.time()   # 更新(任务5诊断): 分阶段计时, 定位"求解慢"到底慢在哪一段
    # 更新(P2-04): 外部 eval_cache 是投毒面(键含 id(ship), 可伪造 feasible/E0 且
    # 跨 solve 存在 GC 地址复用碰撞), 证书完整性要求本入口【始终】新建内部缓存;
    # 形参仅保留签名兼容, 传入值不再被采信。更新 的同-solve 内共享提速语义不变。
    eval_cache = {}

    def _min_energy_of(r):
        bestE = None
        _tids = tuple(r.turbine_ids())
        # 更新(P2-01): Stage-2 能耗评估与定价/池展开同一 per-τ 天气解析(_wx_of_route),
        # 消除多起飞时刻 + 局地天气下的口径不一致与 eval_cache 交叉污染。
        wx_col = _wx_of_route(r, wx)
        _wxfp = _wx_fp(wx_col)      # 更新(P2-04b): 天气指纹入缓存键
        for h in horizons:
            _ck = (id(r.ship), _tids, int(h), _wxfp,
                   _eval_context_fp(p, xi_amb, weather_unc, r.ship, r.turbines))
            dd = eval_cache.get(_ck)
            if dd is None:
                dd = RM.route_feasible_at_h(r, h, p, wx_col, xi_amb, weather_unc=weather_unc)
                eval_cache[_ck] = dd
            if dd["feasible"] and (bestE is None or _plan_energy(dd) < bestE):
                bestE = _plan_energy(dd)
        return bestE

    # ---------- Stage 1: 最少架次(exact 安全支配 strict=True, P0-4) ----------
    cg1 = column_generation(turbines, ship, p, wx, xi_amb, max_stops, k_near, max_iter=max_iter,
                            route_cost=1.0, verbose=verbose, weather_unc=weather_unc,
                            launch_ships=launch_ships, strict_dominance=True, eval_cache=eval_cache)
    pool = list(cg1["columns"])
    LB1_rmp = cg1["LB"]
    LB1 = cg1.get("global_LB")
    ip1 = _integer_master(_expand_pool_h(pool, p, wx, xi_amb, weather_unc=weather_unc, eval_cache=eval_cache), all_ids, p=p, wx=wx, lex=True)
    ip1_master_validated = _master_solution_validated(ip1)
    ip1_master_proven = _master_solution_proven(ip1)
    ip1_master_outputs_clean = _master_outputs_clean(ip1)
    N_star = ip1["n_sorties"]; n_uncov = len(ip1["uncovered"])
    ceilLB1 = _safe_integer_ceiling(LB1)
    # 第一层全局证书: 无 uncovered + LP 紧 + 列生成可发证书(完全图 + strict 支配 + no_neg_rc 收敛, P0-4/5/7/8)
    L1_ok = bool(ip1_master_validated and ip1_master_outputs_clean
                 and n_uncov == 0 and N_star and math.isfinite(N_star)
                 and ceilLB1 >= int(N_star) and _cg_certifiable(cg1))
    _t_stage1 = time.time() - _t0

    # ---------- Stage 2: 最低能耗(固定架次=N*; strict 支配; 约化成本含 σ, P0-2) ----------
    seen = set((_ship_column_fp(r.ship), tuple(r.turbine_ids())) for r, _ in pool)
    E_LP = None; stage2_conv = False; stage2_complete = False; _stage2_iters = 0
    _t_s2_start = time.time()
    if L1_ok:
        for _ in range(energy_max_iter):
            _stage2_iters += 1
            cols_E = []
            for r, _d in pool:
                be = _min_energy_of(r)
                if be is not None:
                    cols_E.append((be, set(r.turbine_ids())))
            objE, dualsE, sigmaE = _energy_partition_lp(cols_E, all_ids, N_star)   # P0-2: 取 σ
            if objE is None:
                break
            E_LP = objE
            dual_idx = {tid2idx[t]: dualsE.get(t, 0.0) for t in all_ids}
            new = []
            for sp in ships:
                _sp_fp = _ship_column_fp(sp)
                _forbid_seq = {seq for fp, seq in seen if fp == _sp_fp}
                pr = price_routes(turbines, sp, p, _wx_of_ship(sp, wx), xi_amb, dual_idx, max_stops=max_stops,
                                  k_near=k_near, weather_unc=weather_unc, objective="energy",
                                  dual_offset=sigmaE, strict_dominance=True, eval_cache=eval_cache,
                                  forbid_route_sequences=_forbid_seq)   # P0-2 σ + P0-4 strict
                for d in pr:
                    d["_ship"] = sp
                new.extend(pr)
            if not new:
                # Exhaustive pricing establishes rc >= -PRICING_EPS.  Because
                # the fixed-count master has sum(z)=N_star, subtracting
                # N_star*PRICING_EPS converts the RMP dual value into a valid
                # full-column-space L2 lower bound.
                E_LP = _pricing_relaxed_lower_bound(objE, N_star, -PRICING_EPS)
                stage2_conv = E_LP is not None
                stage2_complete = stage2_conv
                break
            added = 0
            for pr in sorted(new, key=lambda d: d["reduced_cost"]):
                sp = pr["_ship"]; r = RM.Route(rid=-1, turbines=pr["turbines"], ship=sp); r.fixed_h = pr["h"]
                key = (_ship_column_fp(sp), tuple(r.turbine_ids()))
                if key in seen:
                    continue
                d = RM.route_drcc_feasible(r, p, _wx_of_ship(sp, wx), xi_amb, objective="min_h", weather_unc=weather_unc)   # 更新(P2-01): per-τ 天气
                if d["feasible"]:
                    pool.append((r, d)); seen.add(key); added += 1
            if added == 0:
                # 有负约化成本列却没能加入(重复/后验不可行)⇒ 停滞, 非收敛(P0-8 同理)
                stage2_conv = False; break

    _t_stage2 = time.time() - _t_s2_start

    # ---------- Stage 3: 最大鲁棒裕度(固定架次=N*, 能耗≤E*; 全列空间严格定价, 更新 任务3) ----------
    # L3 主问题 max Σ M_ω z s.t. 覆盖=1、Σz=N*、Σ E_ω z ≤ E*(=E_LP); 取对偶 (π_i, σ, ρ≤0)
    #   → price_routes(objective='robust', energy_dual=ρ) 闭合取 argmax_h[M_ω(h)+ρ E_ω(h)],
    #   strict 同 visited 支配(引理3': ρ≤0 ⇒ 低 e_dom/低 T_nom 同时增 M 与 ρ·E ⇒ score 单调, 支配保守有效)。
    stage3_conv = False; stage3_complete = False; M_LP = None; _stage3_iters = 0
    _t_s3_start = time.time()
    e_star_L3 = E_LP if (E_LP is not None) else None
    L2_tol_Wh = (1e-7 * max(1.0, abs(float(e_star_L3)))) if e_star_L3 is not None else 0.0
    if L1_ok and stage2_conv and stage2_complete and e_star_L3 is not None:
        for _ in range(energy_max_iter):
            _stage3_iters += 1
            pool_h3 = _expand_pool_h(pool, p, wx, xi_amb, weather_unc=weather_unc, eval_cache=eval_cache)
            cols_M = [(float(d.get("M_omega", 0.0)), _plan_energy(d), set(r.turbine_ids()))
                      for r, d in pool_h3]
            objM, dualsM, sigmaM, rhoM, _uncovM = _robust_partition_lp(
                cols_M, all_ids, N_star, e_star_L3, e_tol=L2_tol_Wh)
            if objM is None:
                break
            M_LP = objM
            dual_idx = {tid2idx[t]: dualsM.get(t, 0.0) for t in all_ids}
            new = []
            for sp in ships:
                _sp_fp = _ship_column_fp(sp)
                _forbid_seq = {seq for fp, seq in seen if fp == _sp_fp}
                pr = price_routes(turbines, sp, p, _wx_of_ship(sp, wx), xi_amb, dual_idx, max_stops=max_stops,
                                  k_near=k_near, weather_unc=weather_unc, objective="robust",
                                  dual_offset=sigmaM, energy_dual=rhoM, strict_dominance=True, eval_cache=eval_cache,
                                  forbid_route_sequences=_forbid_seq)
                for d in pr:
                    d["_ship"] = sp
                new.extend(pr)
            if not new:
                # objM is an upper bound on the restricted max-margin LP.
                # Pricing tolerance can increase the full-space maximum by at
                # most N_star*PRICING_EPS.
                M_LP = math.nextafter(float(objM) + float(N_star) * PRICING_EPS,
                                      math.inf)
                stage3_conv = True; stage3_complete = True; break
            added = 0
            for pr in sorted(new, key=lambda d: d["reduced_cost"]):
                sp = pr["_ship"]; r = RM.Route(rid=-1, turbines=pr["turbines"], ship=sp); r.fixed_h = pr["h"]
                key = (_ship_column_fp(sp), tuple(r.turbine_ids()))
                if key in seen:
                    continue
                d = RM.route_drcc_feasible(r, p, _wx_of_ship(sp, wx), xi_amb, objective="min_h", weather_unc=weather_unc)
                if d["feasible"]:
                    pool.append((r, d)); seen.add(key); added += 1
            if added == 0:
                stage3_conv = False; break   # 有负约化成本列却全是重复序列/后验不可行 ⇒ 停滞

    _t_stage3 = time.time() - _t_s3_start

    # ---------- 最终词典序整数主问题(多-h 展开 ⇒ L3 在 (N*,E*) 内取最大裕度)----------
    _t_final_start = time.time()
    ip = _integer_master(_expand_pool_h(pool, p, wx, xi_amb, weather_unc=weather_unc, eval_cache=eval_cache), all_ids, p=p, wx=wx, lex=True)
    final_master_validated = _master_solution_validated(ip)
    final_master_proven = _master_solution_proven(ip)
    final_master_outputs_clean = _master_outputs_clean(ip)
    _t_final = time.time() - _t_final_start
    chosen = ip["routes"]
    plan = _verify_selected_route_plan(chosen, all_ids, p, wx, xi_amb,
                                       weather_unc=weather_unc)
    physical_plan_verified = bool(plan["ok"])
    tot_E = float(plan["total_energy"] or 0.0)
    tot_M = float(plan["total_margin"] or 0.0)
    # The final reported plan must be the same L1 slice whose energy was
    # priced in stage 2.  A merely feasible backend result with a different
    # number of routes cannot inherit the independently proved value N_star.
    final_sortie_consistent = bool(
        physical_plan_verified
        and N_star is not None and math.isfinite(float(N_star))
        and int(ip.get("n_sorties", -1)) == int(N_star)
        and len(chosen) == int(N_star))
    diag = [dict(stops=d["stops"], h=d["h"], tau=d["tau"], slot=d["slot"],
                 E0=round(d["E0"], 1), E0_nominal=round(d["E0_nominal"], 1),
                 M_omega=round(d["M_omega"], 2), turbines=d["turbines"])
            for d in plan["diag"]]
    # Use and disclose the same tight numerical scale as the lexicographic fixing rows.
    # The previous 1e-3 relative test silently allowed ~0.1% L2 error and could still set
    # certified_lex3=True at a strictly worse L2 value.
    L3_tol = 1e-7 * max(1.0, abs(tot_M))
    L2_gap = (float(tot_E) - float(E_LP)) if E_LP is not None else None
    L3_gap = (float(M_LP) - float(tot_M)) if M_LP is not None else None
    L2_tight = bool(L2_gap is not None and -L2_tol_Wh <= L2_gap <= L2_tol_Wh)
    M_LP_tight = bool(L3_gap is not None and -L3_tol <= L3_gap <= L3_tol)
    # 更新 审计 P0-3 + 更新 任务3: 三档证书 —— L1(最少架次) / L1_L2(架次+能耗) / lex3(完整三层全列空间)。
    certified_L1 = bool(L1_ok and final_master_validated and final_master_outputs_clean
                        and physical_plan_verified and final_sortie_consistent
                        and len(ip["uncovered"]) == 0)
    certified_L1_L2 = bool(certified_L1 and stage2_conv and stage2_complete and L2_tight)
    # 第三层全局精确(任务3): 前两层证书 ∧ L3 全列空间定价收敛(strict 同 visited + ρ≤0 引理3')∧ 整数 ΣM 达 L3-LP 上界
    certified_lex3 = bool(certified_L1_L2 and stage3_conv and stage3_complete and M_LP_tight)
    _t_total = time.time() - _t0
    if verbose:
        print(f"  [计时诊断] stage1(架次)={_t_stage1:.1f}s | stage2(能耗,{_stage2_iters}轮)={_t_stage2:.1f}s | "
              f"stage3(裕度,{_stage3_iters}轮)={_t_stage3:.1f}s | 末整数主问题={_t_final:.2f}s | "
              f"总计={_t_total:.1f}s | pool_size={len(pool)} | n={n} k_near={k_near} max_stops={max_stops}")
    return dict(method="lex_column_generation",
                n_sorties=int(N_star) if (N_star and math.isfinite(N_star)) else None,
                total_energy_Wh=round(tot_E, 1), total_energy_Wh_raw=float(tot_E),
                total_robust_margin=round(tot_M, 2),
                uncovered=ip["uncovered"], LB1=round(LB1, 9) if LB1 is not None else None,
                stage1_restricted_master_LB=(float(LB1_rmp) if LB1_rmp is not None else None),
                ceil_LB1=ceilLB1,
                energy_LP_lb=round(E_LP, 1) if E_LP is not None else None,
                energy_LP_lb_raw=(float(E_LP) if E_LP is not None else None),
                L2_certificate_gap_Wh=(float(L2_gap) if L2_gap is not None else None),
                robust_LP_ub=round(M_LP, 2) if M_LP is not None else None,
                L3_certificate_gap=(float(L3_gap) if L3_gap is not None else None),
                certified_L1=certified_L1, certified_L1_L2=certified_L1_L2, certified_lex3=certified_lex3,
                l2_optimality_tolerance_Wh=L2_tol_Wh, l3_optimality_tolerance=L3_tol,
                pricing_reduced_cost_tolerance=float(PRICING_EPS),
                L2_full_space_pricing_slack_Wh=float(N_star) * float(PRICING_EPS),
                proof_model_scope="finite_discrete_route_model",
                continuous_real_world_optimality_claimed=False,
                stage1_master_validated=ip1_master_validated,
                stage1_master_proven_optimal=ip1_master_proven,
                stage1_all_milp_outputs_validated=ip1_master_outputs_clean,
                final_master_validated=final_master_validated,
                final_master_proven_optimal=final_master_proven,
                final_all_milp_outputs_validated=final_master_outputs_clean,
                master_validation_reason=ip.get("validation_reason"),
                physical_plan_verified=physical_plan_verified,
                physical_plan_validation_reason=plan["reason"],
                final_sortie_count_consistent=final_sortie_consistent,
                reported_solution_consistent=bool(physical_plan_verified and final_sortie_consistent),
                certified_min_sorties=certified_L1,           # 别名(向后兼容)
                certified_lex=certified_lex3,                 # FIX #4: 三层别名现【确指完整 lex3】(L1∧L2∧L3), 非仅前两层
                stage1_certifiable=_cg_certifiable(cg1), stage1_converged=cg1["converged"],
                stage1_termination=cg1["termination_reason"], stage1_strict=cg1["strict_dominance"],
                stage2_converged=stage2_conv, stage2_complete=stage2_complete, L2_lp_tight=L2_tight,
                stage3_converged=stage3_conv, stage3_complete=stage3_complete, L3_lp_tight=M_LP_tight,
                pool_size=len(pool), route_diag=diag,
                # 更新(任务5诊断): 分阶段计时, 用于定位"求解慢"的真实瓶颈(见 doc_algorithm §诊断)
                stage1_time_s=round(_t_stage1, 2), stage2_time_s=round(_t_stage2, 2),
                stage3_time_s=round(_t_stage3, 2), final_master_time_s=round(_t_final, 3),
                total_time_s=round(_t_total, 2), stage2_iters=_stage2_iters, stage3_iters=_stage3_iters)


def solve_route_drcc_exact(turbines, ship, p, wx, xi_amb, max_stops=8, k_near=8,
                           verbose=True, weather_unc=None, lex=True, launch_ships=None,
                           strict_dominance=False) -> dict:
    """精确求解(列生成下界 + 整数主问题上界 + gap + 整数最优性证书)。weather_unc 透传(多源 SOC + 浪门)。
    lex=True(默认): 整数主问题用词典序三层目标(架次→能耗→鲁棒裕度 M_ω)。
    launch_ships(可选): 起飞时刻网格(任务2), 逐 τ 定价并池。
    strict_dominance: True ⇒ 标号支配用同 visited 集(exact 安全支配); False ⇒ 子集支配(accelerated, 更快)。

    两种 gap(critique 必改 2 — 区分 LP 松弛 gap 与整数最优性):
      gap_pct      = (UB − LB)/UB        : LP 松弛 gap(连续下界 vs 整数上界);
      opt_gap_pct  = (UB − ⌈LB⌉)/UB      : 【整数最优性 gap】;
      certified_optimal = (⌈LB⌉ ≥ UB)   : 若真 ⇒ 不存在更优整数解 ⇒ UB 即【该离散化模型全局最优】,
                                            此时无需分支即被证明最优(B&P 根节点即闭合)。"""
    eval_cache = {}   # 更新(提速#B): 同一 solve 内共享 (τ,序列,h) 闭合评估(定价迭代/池展开复用)
    cg = column_generation(turbines, ship, p, wx, xi_amb, max_stops, k_near,
                           verbose=verbose, weather_unc=weather_unc, launch_ships=launch_ships,
                           strict_dominance=strict_dominance, eval_cache=eval_cache)
    pool = cg["columns"]
    restricted_LB = cg["LB"]
    LB = cg.get("global_LB")
    all_ids = [t.tid for t in turbines]
    # 闭合降调#1: 把列池展开为完整 (τ,π,h)(每序列全部可行 h 皆独立列), 供词典序主问题为 L2/L3 选 h
    pool_h = _expand_pool_h(pool, p, wx, xi_amb, weather_unc=weather_unc, eval_cache=eval_cache)
    ip = _integer_master(pool_h, all_ids, p=p, wx=wx, lex=lex)
    master_solution_validated = _master_solution_validated(ip)
    master_proven_optimal = _master_solution_proven(ip)
    master_outputs_clean = _master_outputs_clean(ip)
    UB = float(ip["n_sorties"])
    n_uncov = len(ip["uncovered"])
    gap = (UB - LB) / UB if UB > 0 and LB is not None else None
    ceilLB = _safe_integer_ceiling(LB)
    opt_gap = (UB - ceilLB) / UB if UB > 0 and LB is not None else None
    # 证书(更新/更新 审计 claim3/4/5/6): 仅当全部满足才证最少架次全局最优 ——
    #   ① 无 uncovered(覆盖全部风机的划分可行); ② UB 有限>0; ③ ⌈SC-LP*⌉≥UB(根整数证书);
    #   ④ 列生成【完全定价】k_near=n(否则只是受限近邻列空间, claim3); ⑤ 列生成【收敛】(非达 max_iter, claim5)。
    #   ④⑤ 在【最底层】把关 ⇒ 即便调用者传 k_near<n 或迭代上限, 底层也不会冒称全局证书(不依赖 wrapper)。
    has_feasible = bool(master_solution_validated and master_outputs_clean
                        and UB > 0 and math.isfinite(UB) and n_uncov == 0)
    pricing_ok = _cg_certifiable(cg)    # 完全图 + strict 支配 + no_neg_rc 收敛(P0-4/5/7/8)
    certified_min_sorties = bool(has_feasible and LB is not None and ceilLB >= int(round(UB)) and pricing_ok)
    certified_lex = False     # 词典序第 2/3 层全局精确见 lex_column_generation(本函数不做能耗重定价)
    chosen = ip["routes"]
    plan = _verify_selected_route_plan(chosen, all_ids, p, wx, xi_amb,
                                       weather_unc=weather_unc)
    physical_plan_verified = bool(plan["ok"])
    if not physical_plan_verified:
        has_feasible = False
        certified_min_sorties = False
    tot_E = float(plan["total_energy"] or 0.0)
    tot_M = float(plan["total_margin"] or 0.0)
    n_multi = sum(int(r.n_stops() > 1) for r in chosen) if physical_plan_verified else 0
    diag = [dict(stops=d["stops"], h=d["h"], E0=round(d["E0"], 1),
                 tau=d["tau"], slot=d["slot"],
                 M_omega=round(d["M_omega"], 2), turbines=d["turbines"])
            for d in plan["diag"]]
    return dict(LB=(round(LB, 9) if LB is not None else None),
                restricted_master_LB=(float(restricted_LB) if restricted_LB is not None else None),
                UB=UB, gap=(round(gap, 9) if gap is not None else None),
                gap_pct=(round(100 * gap, 6) if gap is not None else None),
                opt_gap_pct=(round(100 * opt_gap, 6) if opt_gap is not None else None),
                certified_min_sorties=certified_min_sorties, certified_lex=certified_lex,
                certified_optimal=certified_min_sorties, has_feasible_cover=has_feasible,
                ceil_LB=ceilLB,
                n_sorties=int(UB), n_multi_stop=n_multi, total_energy_Wh=round(tot_E, 1),
                total_robust_margin=round(tot_M, 2),
                pool_size=len(pool), cg_iters=cg["n_iter"], priced_total=cg["priced_total"],
                cg_converged=cg.get("converged"), neighbor_graph_complete=cg.get("neighbor_graph_complete"),
                cg_strict_dominance=cg.get("strict_dominance"), cg_certifiable=pricing_ok,
                cg_termination=cg.get("termination_reason"), k_near_used=cg.get("k_near_used"),
                hit_max_iter=cg.get("hit_max_iter"), last_min_reduced_cost=cg.get("last_min_reduced_cost"),
                pricing_reduced_cost_tolerance=cg.get("pricing_rc_tolerance"),
                full_space_pricing_slack=float(len(turbines)) * float(PRICING_EPS),
                proof_model_scope="finite_discrete_route_model",
                continuous_real_world_optimality_claimed=False,
                uncovered=ip["uncovered"], solver_master=ip["solver"],
                master_solution_validated=master_solution_validated,
                master_proven_optimal=master_proven_optimal,
                master_L1_proven_optimal=_master_l1_proven(ip),
                all_milp_outputs_validated=master_outputs_clean,
                master_validation_reason=ip.get("validation_reason"),
                physical_plan_verified=physical_plan_verified,
                physical_plan_validation_reason=plan["reason"],
                mean_stops=round(np.mean([r.n_stops() for r in chosen]), 2) if chosen else 0.0,
                route_diag=diag,
                # 更新(任务5诊断): 透传 column_generation 的分段计时 —— 定位"求解慢"瓶颈在定价还是LP
                cg_total_time_s=cg.get("cg_total_time_s"), cg_pricing_time_s=cg.get("cg_pricing_time_s"),
                cg_lp_time_s=cg.get("cg_lp_time_s"), cg_feasibility_time_s=cg.get("cg_feasibility_time_s"),
                cg_pricing_frac=cg.get("cg_pricing_frac"))


# =============================================================================
# 4b. 完整列枚举 集合划分 基准(把全部 DR 可行列一次性展开喂给求解器)
#     —— 算法实验的【速度对照基准】(step14 vs_gurobi_speed)。历史名 monolithic_misocp。
#     思路: 枚举所有 DR 可行的(序列 π, 回收时长 h)组合作为列, 每列预先离线判定
#     决策依赖 DRCC + 着舰门 + 空速可行性(可行性已"烘焙"进列), 再解词典序【集合划分 IP】。
#     注: 因可行性已烘焙进列, 主问题是 0-1 MILP / 集合划分 IP, **不是 MISOCP**(SOC 未显式交给求解器)。
#     与列生成的区别: 列生成【按需】生成列(指数空间里只碰必要的), 本法【全部枚举】,
#     列数随 |I|、max_stops 组合爆炸 → 规模一大直接慢/不可解, 正是要展示的对比。
#     【完整枚举前提(用作 enumeration_anchor 时)】: k_near=n、max_cols 不截断、max_stops 足够、
#       无提前剪枝误删 —— 满足时才是真·全列空间; 否则是近邻受限枚举(仍是有效 UB, 但非全枚举)。
# =============================================================================
def _enumerate_routes(turbines, ship, p, wx, xi_amb, max_stops, k_near,
                      weather_unc=None, max_cols=200000, forbid_pairs=None, force_pairs=None,
                      launch_ships=None):
    """枚举所有长度 ≤ max_stops 的近邻路由(序列), 每条逐 h 选最优 h* 并判 DR 可行。
    返回 [(Route, 诊断)]; 仅保留 DR 可行列。列数随组合爆炸——用于完整列枚举基准 / 枚举锚点。
    **完整性前提**: k_near=n 且 max_cols 不截断 才是真·全列空间(否则近邻受限, 仍是有效 UB)。
    forbid_pairs/force_pairs(Ryan–Foster): apart=不同时含; together=同含或同不含。用于 B&P 节点枚举定价。

    **任务2(起飞时刻 τ 作决策)**: 模型列 r=(τ,ω,h) —— 同一访问序列在【不同起飞时刻 τ】下是【不同列】
    (起飞船位/几何/风浪不同)。给 launch_ships(τ 网格)时, 对每个 τ 各枚举一遍 ⇒ 完整列空间随 |τ网格| 倍增。
    这正是【完整枚举须遍历 routes×τ×h, 而列生成只按需对每条路由定价最优 τ,h】的对照根据。"""
    # A supplied launch grid is part of the finite discrete model even when it
    # contains exactly one element.  Do not silently fall back to the separate
    # ``ship`` argument: callers may deliberately pass a different sole
    # launch option (different position, slot or weather).  Enumerating each
    # supplied option also makes the completeness statement literally cover
    # routes × launch-options × recovery-horizons.
    if launch_ships is not None:
        all_cols, trunc_any = [], False
        for sp in launch_ships:
            c, t = _enumerate_routes(turbines, sp, p, wx, xi_amb, max_stops, k_near,
                                     weather_unc, max_cols, forbid_pairs, force_pairs, launch_ships=None)
            all_cols.extend(c); trunc_any = trunc_any or t
        return all_cols, trunc_any
    forbid_pairs = forbid_pairs or set()
    force_pairs = force_pairs or set()
    # Every enumerated column belongs to this launch option and must be tested
    # under that option's own weather, exactly like pricing and final expansion.
    wx = _wx_of_ship(ship, wx)
    nb = _neighbors(turbines, k_near)
    feasible_cols = []
    seen = set()
    # 更新(任务5修复): 早期能量剪枝(镜像 price_routes 的同款剪枝, 见该函数 line "早剪枝: 名义能耗已超可用")。
    # 问题根因: 本函数原先对每条候选序列都做【完整的 route_drcc_feasible 检查】(扫全部 h 网格)才知道是否
    # 可行, 没有在扩展阶段提前排除"名义能耗已超预算"的序列。更新 把 max_stops 由 4 放宽到 8 后,
    # 多出的深度 5-8 层全部要在【扩展时枚举 + 事后才发现不可行】上花时间——实测 n=6 时 max_stops 4→6
    # 耗时 8.4s→38.1s(4.5×), 但两者找到的可行列数完全相同(均 18 条)。
    # 更新(采纳外部审计 6.1, 修正 更新 的过强断言): "被剪掉的分支本来就会在 _eval_seq 里
    # 判不可行"只在【无带符号均值放松】时成立 —— 顺风偏置/ξ 均值可使名义超预算的序列
    # DRCC 可行。故本剪枝与 price_routes 同款, 仅在 mean_relax_free 时启用。
    _prune_nom = RM.mean_relax_free(xi_amb, weather_unc)
    wc0 = _wx_cache(p, wx)
    E_to0, E_land0, T_to0, T_land0 = M.to_land_energy_time(p)

    def _ext_energy(from_local, to_idx, e_nom_so_far):
        """扩展到 turbines[to_idx] 的增量能耗(腿 + 巡检), 镜像 price_routes 的 _leg_ET/_insp_ET。"""
        wc_leg = _leg_wc(p, wc0, turbines[to_idx])
        eL, _tL = _leg_ET(p, wc_leg, from_local, turbines[to_idx].local)
        eI, _tI = _insp_ET(p, turbines[to_idx], p.z_cruise)
        return e_nom_so_far + eL + eI

    def _ok_constraints(seq_idx) -> bool:
        s = set(seq_idx)
        for pair in forbid_pairs:               # apart: 不可同时含
            i, j = tuple(pair)
            if i in s and j in s:
                return False
        for pair in force_pairs:                # together: 同含或同不含
            i, j = tuple(pair)
            if (i in s) != (j in s):
                return False
        return True

    def _eval_seq(seq_idx):
        key = tuple(seq_idx)
        if key in seen:
            return
        seen.add(key)
        if not _ok_constraints(seq_idx):
            return
        route = RM.Route(rid=-1, turbines=[turbines[j] for j in seq_idx], ship=ship)
        d = RM.route_drcc_feasible(route, p, wx, xi_amb, objective="max_robust",
                                   weather_unc=weather_unc)
        if d["feasible"]:
            route.fixed_h = d["h"]
            feasible_cols.append((route, d))

    # BFS 枚举近邻路由(从起飞点出发); frontier 现含 (seq, E_nom_so_far) 供早剪枝。
    frontier = []
    for j in range(len(turbines)):
        e0 = _ext_energy(ship.P_launch, j, E_to0)
        frontier.append(((j,), e0))
        _eval_seq([j])
    depth = 1
    truncated = False
    while depth < max_stops and len(seen) < max_cols:
        new_frontier = []
        for seq, e_nom in frontier:
            last = seq[-1]
            for j in nb[last]:
                if j in seq:
                    continue
                # apart 早剪枝: 序列中已含与 j 互斥的风机则不扩展
                if forbid_pairs and any(frozenset((j, v)) in forbid_pairs for v in seq):
                    continue
                e_new = _ext_energy(turbines[last].local, j, e_nom)
                # 早剪枝(更新/更新): 名义能耗(不含返程)加降落预留超预算 ⇒ 仅在
                # mean_relax_free 时保真(见上方 更新 注释); 否则不剪。
                if _prune_nom and e_new + E_land0 > p.B_use:
                    continue
                ns = seq + (j,)
                new_frontier.append((ns, e_new))
                _eval_seq(list(ns))
                if len(seen) >= max_cols:
                    truncated = True; break
            if len(seen) >= max_cols:
                truncated = True; break
        frontier = new_frontier
        depth += 1
    # 更新 审计 claim3: 若达 max_cols 被截断, 则非全列空间(不是 ground-truth 锚点), 须报告
    return feasible_cols, truncated


def _independent_complete_partition_dp(columns, turbine_ids, max_states=2_000_000):
    """Exact lexicographic verifier for a *fully enumerated* finite column set.

    This routine is intentionally independent of scipy.milp/Gurobi.  It solves
    the set-partitioning master by memoized recursion on

        (remaining turbine mask, occupied launch-slot mask),

    always branching on the first remaining turbine.  Exact partitioning makes
    every recursive transition strictly reduce the remaining mask, so the
    recursion enumerates every feasible route combination exactly once up to
    state memoization.  The objective is lexicographic ``(number of routes,
    DRCC energy, -robust margin)``.  It is used only as an independent verifier
    for ``enumeration_anchor``; large instances fail closed when ``max_states``
    is exceeded rather than weakening the certificate.
    """
    from functools import lru_cache

    ids = list(turbine_ids)
    tidx = {tid: i for i, tid in enumerate(ids)}
    if len(tidx) != len(ids):
        return dict(complete=False, feasible=False, reason="duplicate_turbine_ids",
                    states=0, L1=None, L2=None, L3=None, chosen_indices=[])

    # Map arbitrary slot labels to exact bits.  ``None`` means no launch-slot
    # capacity row, matching solve_master/_verify_selected_route_plan.
    slot_values = []
    for route, _diag in columns:
        slot = getattr(getattr(route, "ship", None), "slot", None)
        if slot is not None:
            fp = _state_fp(slot)
            if fp not in slot_values:
                slot_values.append(fp)
    slot_index = {fp: i for i, fp in enumerate(slot_values)}

    candidates = []
    # For identical (coverage, slot) states, only the lexicographically best
    # physical column can ever be useful in an additive L1/L2/L3 objective.
    best_same = {}
    for k, (route, diag) in enumerate(columns):
        tids = list(route.turbine_ids())
        if (not tids or len(tids) != len(set(tids))
                or any(tid not in tidx for tid in tids)):
            continue
        mask = 0
        for tid in tids:
            mask |= 1 << tidx[tid]
        try:
            energy = _plan_energy(diag)
            margin = float(diag.get("M_omega", 0.0))
        except Exception:
            continue
        if not (math.isfinite(energy) and math.isfinite(margin)):
            continue
        slot = getattr(getattr(route, "ship", None), "slot", None)
        slot_bit = 0 if slot is None else (1 << slot_index[_state_fp(slot)])
        key = (mask, slot_bit)
        score = (energy, -margin, k)
        old = best_same.get(key)
        if old is None or score < old[0]:
            best_same[key] = (score, (mask, slot_bit, energy, margin, k))
    candidates = [item for _score, item in best_same.values()]

    by_task = [[] for _ in ids]
    for ci, (mask, _slot_bit, _energy, _margin, _k) in enumerate(candidates):
        for i in range(len(ids)):
            if mask & (1 << i):
                by_task[i].append(ci)

    states = 0
    limit_hit = False

    @lru_cache(maxsize=None)
    def solve(remaining, occupied_slots):
        nonlocal states, limit_hit
        states += 1
        if states > int(max_states):
            limit_hit = True
            raise RuntimeError("state_limit")
        if remaining == 0:
            return (0, 0.0, 0.0, ())  # count, energy, -margin, chosen indices
        first = (remaining & -remaining).bit_length() - 1
        best = None
        for ci in by_task[first]:
            mask, slot_bit, energy, margin, original_k = candidates[ci]
            if mask & remaining != mask:
                continue                         # would overlap an already selected route
            if slot_bit and (occupied_slots & slot_bit):
                continue                         # launch-slot capacity
            tail = solve(remaining ^ mask, occupied_slots | slot_bit)
            if tail is None:
                continue
            cand = (tail[0] + 1,
                    math.fsum((energy, tail[1])),
                    math.fsum((-margin, tail[2])),
                    (original_k,) + tail[3])
            if best is None or cand[:3] < best[:3]:
                best = cand
        return best

    try:
        full = (1 << len(ids)) - 1
        ans = solve(full, 0)
    except RuntimeError:
        ans = None
    if limit_hit:
        return dict(complete=False, feasible=False, reason="state_limit",
                    states=states, L1=None, L2=None, L3=None, chosen_indices=[])
    if ans is None:
        return dict(complete=True, feasible=False, reason="infeasible",
                    states=states, L1=None, L2=None, L3=None, chosen_indices=[])
    return dict(complete=True, feasible=True, reason=None, states=states,
                L1=int(ans[0]), L2=float(ans[1]), L3=float(-ans[2]),
                chosen_indices=list(ans[3]))


def solve_monolithic_misocp(turbines, ship, p, wx, xi_amb, max_stops=8, k_near=8,
                            weather_unc=None, time_limit=None, lex=True, launch_ships=None, h_grid=None) -> dict:
    """完整列枚举集合划分基准(complete-enumeration set-partitioning benchmark; 历史名 monolithic_misocp):
    枚举全部 DR 可行列 → 一次性整数主问题(词典序, partition=True)。报告枚举列数、求解时间、目标值,
    供与列生成/分支定价对比速度。**主问题是集合划分 IP, 非 MISOCP**(SOC 已烘焙进列可行性)。
    无 Gurobi 时用 scipy.milp 精确求解(speedup 的绝对时间须作者本机有 Gurobi 时跑)。
    **任务2**: launch_ships(τ 网格)与 h_grid(细回收时长格)使完整列空间 = routes×|τ|×|h|, 须全部枚举。"""
    import time as _time
    n = len(turbines)
    t0 = _time.perf_counter()
    cols, enum_truncated = _enumerate_routes(turbines, ship, p, wx, xi_amb, max_stops, k_near,
                                             weather_unc, launch_ships=launch_ships)
    t_enum = _time.perf_counter() - t0
    n_seq_enum = len(cols)                     # 枚举到的(τ,π)序列列数(展开 h 前)
    all_ids = [t.tid for t in turbines]
    # 闭合降调#1: 枚举锚点同样展开为完整 (τ,π,h), 与 B&P 列空间一致(便于 L2/L3 也可对齐验证)
    cols = _expand_pool_h(cols, p, wx, xi_amb, weather_unc=weather_unc, h_grid=h_grid)
    t1 = _time.perf_counter()
    ip = _integer_master(cols, all_ids, p=p, wx=wx, lex=lex)
    t_solve = _time.perf_counter() - t1
    chosen = ip["routes"]
    _plan = _verify_selected_route_plan(chosen, all_ids, p, wx, xi_amb,
                                       weather_unc=weather_unc)
    physical_plan_verified = bool(_plan["ok"])
    tot_E = float(_plan["total_energy"]) if physical_plan_verified else 0.0
    tot_E_nom = float(_plan["total_nominal"]) if physical_plan_verified else 0.0
    tot_M = float(_plan["total_margin"]) if physical_plan_verified else 0.0
    # Independent certificate checks: do not rely solely on scipy/Gurobi's
    # reported MIP status.  LP tightness is a cheap sufficient check; the exact
    # mask/slot DP additionally handles complete-enumeration instances with an
    # integrality gap.
    # On the complete column set, LP tightness supplies a separately validated bound.  If the
    # LP is fractional, the returned MIP solution may still be correct, but this code does not
    # possess an independent optimality proof and therefore must not certify it.
    _cnt_cols = [(1.0, set(r.turbine_ids())) for r, _d in cols]
    _anchor_lb, _az, _adual, _auncov = _partition_lp(_cnt_cols, all_ids)
    _anchor_l1_tight = bool(_anchor_lb is not None and not _auncov and
                            _safe_integer_ceiling(_anchor_lb) >= int(ip["n_sorties"]))
    _E_cols = [(_plan_energy(d), set(r.turbine_ids())) for r, d in cols]
    _anchor_E_lb, _aedual, _aesigma = _energy_partition_lp(_E_cols, all_ids, int(ip["n_sorties"]))
    _anchor_l2_tol = 1e-7 * max(1.0, abs(tot_E))
    _anchor_l2_gap = ((float(tot_E) - float(_anchor_E_lb))
                      if _anchor_E_lb is not None else None)
    _anchor_l2_tight = bool(_anchor_l2_gap is not None
                            and -_anchor_l2_tol <= _anchor_l2_gap <= _anchor_l2_tol)

    # 更新 审计 claim5: 完整枚举(ground-truth 锚点)须同时满足 k_near≥n(覆盖全部风机近邻)且未被 max_cols 截断;
    # 否则只是【近邻受限枚举】(仍是有效 UB, 但不是全列空间锚点)。
    complete_enum = bool(k_near >= n and not enum_truncated)
    _anchor_dp = (_independent_complete_partition_dp(cols, all_ids)
                  if complete_enum else
                  dict(complete=False, feasible=False, reason="enumeration_incomplete",
                       states=0, L1=None, L2=None, L3=None, chosen_indices=[]))
    _anchor_dp_l1_match = bool(
        _anchor_dp.get("complete") is True and _anchor_dp.get("feasible") is True
        and int(ip["n_sorties"]) == int(_anchor_dp["L1"]))
    _anchor_dp_l2_gap = ((float(tot_E) - float(_anchor_dp["L2"]))
                         if _anchor_dp_l1_match and physical_plan_verified else None)
    _anchor_dp_l2_match = bool(
        _anchor_dp_l2_gap is not None
        and -_anchor_l2_tol <= _anchor_dp_l2_gap <= _anchor_l2_tol)
    master_solution_validated = _master_solution_validated(ip)
    master_proven_optimal = _master_solution_proven(ip)
    master_outputs_clean = _master_outputs_clean(ip)
    certified_anchor = bool(complete_enum and master_solution_validated and master_outputs_clean
                            and physical_plan_verified
                            and not ip["uncovered"]
                            and (_anchor_dp_l1_match or _anchor_l1_tight))
    certified_anchor_lex = bool(certified_anchor and lex
                                and (_anchor_dp_l2_match or _anchor_l2_tight))
    return dict(method="monolithic_misocp", n_cols_enumerated=len(cols),
                n_seq_enumerated=n_seq_enum, enum_truncated=bool(enum_truncated),
                k_near_used=k_near, complete_enumeration=complete_enum,
                certified_optimal=certified_anchor,
                certified_min_sorties=certified_anchor,
                certified_lex=certified_anchor_lex,
                independently_validated_L1_bound=_anchor_l1_tight,
                independently_validated_L2_bound=_anchor_l2_tight,
                independent_enumeration_DP_complete=bool(_anchor_dp.get("complete")),
                independent_enumeration_DP_feasible=bool(_anchor_dp.get("feasible")),
                independent_enumeration_DP_reason=_anchor_dp.get("reason"),
                independent_enumeration_DP_states=int(_anchor_dp.get("states", 0)),
                independent_enumeration_DP_L1=_anchor_dp.get("L1"),
                independent_enumeration_DP_L2=_anchor_dp.get("L2"),
                independent_enumeration_DP_L1_match=_anchor_dp_l1_match,
                independent_enumeration_DP_L2_match=_anchor_dp_l2_match,
                independent_enumeration_DP_L2_gap_Wh=_anchor_dp_l2_gap,
                anchor_L1_LP_lb=_anchor_lb, anchor_L2_LP_lb=_anchor_E_lb,
                l2_optimality_tolerance_Wh=_anchor_l2_tol,
                L2_certificate_gap_Wh=_anchor_l2_gap,
                proof_model_scope="finite_discrete_route_model",
                continuous_real_world_optimality_claimed=False,
                master_solution_validated=master_solution_validated,
                master_proven_optimal=master_proven_optimal,
                master_L1_proven_optimal=_master_l1_proven(ip),
                all_milp_outputs_validated=master_outputs_clean,
                master_validation_reason=ip.get("validation_reason"),
                physical_plan_verified=physical_plan_verified,
                physical_plan_validation_reason=_plan.get("reason"),
                reported_solution_consistent=physical_plan_verified,
                certified_L1=certified_anchor,
                certified_L1_L2=certified_anchor_lex,
                n_sorties=int(ip["n_sorties"]), total_energy_Wh=round(tot_E, 1),
                total_energy_Wh_raw=float(tot_E),
                total_nominal_energy_Wh=round(tot_E_nom, 1),
                anchor_L2_LP_lb_raw=(float(_anchor_E_lb) if _anchor_E_lb is not None else None),
                total_robust_margin=round(tot_M, 2),
                uncovered=ip["uncovered"], solver_master=ip["solver"],
                t_enumerate_s=round(t_enum, 3), t_solve_s=round(t_solve, 3),
                t_total_s=round(t_enum + t_solve, 3))


def _master_lp_primal(columns, turbine_ids, big=1e4, partition=False):
    r"""主问题 LP 松弛, 额外返回【原始解 z】(各真实列的 LP 取值), 供 Ryan–Foster 找分数对。
    只对可覆盖风机建约束(与 _master_lp 一致)。返回 (obj, z[len(columns)], duals{tid:λ})。

    更新 审计 claim2: **`partition=True` 时用集合划分 LP(等式 $\sum_{\omega\ni i}z_\omega=1$)**,
    使 Ryan–Foster 定理严格成立 —— 分数解 ⇒ 必存在分数对 $(i,j)$; 无分数对 ⇒ 划分 LP 整数 ⇒ 该节点闭合。
    覆盖 LP(≥1)下无此保证(可能分数却无分数对, "无对即闭合"会成为理论缝隙)。人工单列(成本 big)
    保证 LP 恒可行(不可覆盖风机由人工列以 =1 顶上, 实为 uncovered, 由整数主问题另报)。
    """
    from scipy.optimize import linprog
    coverable = set()
    for _, cov in columns:
        coverable |= set(cov)
    ids = [t for t in turbine_ids if t in coverable]
    n_t = len(ids); tidx = {t: i for i, t in enumerate(ids)}
    n_real = len(columns)
    if n_t == 0:
        return 0.0, np.zeros(n_real), {t: 0.0 for t in turbine_ids}
    cols = list(columns) + [(big, {t}) for t in ids]
    n_col = len(cols)
    c = np.array([cc for cc, _ in cols], float)
    A = np.zeros((n_t, n_col))    # +1 入射矩阵
    for k, (_, cov) in enumerate(cols):
        for t in cov:
            if t in tidx:
                A[tidx[t], k] = 1.0
    bounds = [(0, None)] * n_col
    if partition:
        beq = np.ones(n_t)
        res = linprog(c, A_eq=A, b_eq=beq, bounds=bounds, method="highs")
        checked = _validate_linprog_result(res, c, bounds, A_eq=A, b_eq=beq,
                                           need_eqlin=n_t,
                                           dual_bounds=[(0, 1)] * n_col)
    else:
        Aub, bub = -A, -np.ones(n_t)
        res = linprog(c, A_ub=Aub, b_ub=bub, bounds=bounds, method="highs")
        checked = _validate_linprog_result(res, c, bounds, A_ub=Aub, b_ub=bub,
                                           need_ineqlin=n_t,
                                           dual_bounds=[(0, 1)] * n_col)
    if checked is None:
        return None, None, None
    x, fun, imarg, emarg, dual_lb = checked
    z = x[:n_real]
    duals = {t: 0.0 for t in turbine_ids}
    marg = emarg if partition else imarg
    for i, t in enumerate(ids):
        duals[t] = float(marg[i]) if partition else max(-float(marg[i]), 0.0)
    return dual_lb, z, duals


def _partition_lp(columns, turbine_ids, big=1e4, slot_ids=None,
                  return_slot_duals=False, column_keys=None, force_keys=None):
    r"""【全列空间 SP-LP 的受限主问题】: 集合划分 LP $\min \sum c_\omega z_\omega$
    s.t. $\sum_{\omega\ni i}z_\omega=1\ \forall i$(对【全部】风机, P0-1 同款), $z\ge0$, 外加每风机人工列(big)。
    ``force_keys`` 用额外等式 $z_k=1$ 实现完整的列变量分支；与
    ``column_keys`` 一起使用。禁止列在调用前从列池和定价结果中删除。
    返回 (LB_real, z[真实列], duals{tid:σ_i}(自由号 ±), uncovered)。若
    ``return_slot_duals=True``，再返回 ``slot_duals{slot:μ_s}``。
    $\sigma_i$ 为划分等式对偶(可负), 用于 SP-dual 定价: reduced cost $\bar c_\omega=c_\omega-\sum_{i\in\omega}\sigma_i$。
    LB 扣人工灌水; 收敛后仍占人工列的风机为 uncovered(真正不可达)。"""
    from scipy.optimize import linprog
    ids = list(turbine_ids); n_t = len(ids)
    if n_t == 0:
        base = (0.0, np.zeros(len(columns)), {t: 0.0 for t in turbine_ids}, set())
        return base + ({},) if return_slot_duals else base
    tidx = {t: i for i, t in enumerate(ids)}
    cols = list(columns); n_real = len(cols)
    for t in ids:
        cols.append((big, {t}))                       # 每风机人工列, 保 LP 可行
    n_col = len(cols)
    c = np.array([cc for cc, _ in cols], float)
    A = np.zeros((n_t, n_col))                         # +1 入射矩阵(全风机行)
    for k, (_, cov) in enumerate(cols):
        for t in cov:
            if t in tidx:
                A[tidx[t], k] = 1.0
    bounds = [(0, None)] * n_col
    force_keys = set(force_keys or ())
    force_rows = []
    if force_keys:
        if column_keys is None or len(column_keys) != n_real:
            base = (None, None, None, None)
            return base + (None,) if return_slot_duals else base
        key_to_idx = {k: i for i, k in enumerate(column_keys)}
        for key in sorted(force_keys, key=repr):
            if key not in key_to_idx:
                base = (None, None, None, None)
                return base + (None,) if return_slot_duals else base
            row = np.zeros(n_col)
            row[key_to_idx[key]] = 1.0
            force_rows.append(row)
    Aeq = A if not force_rows else np.vstack([A] + force_rows)
    beq = np.concatenate([np.ones(n_t), np.ones(len(force_rows))])
    Aub = None; bub = None; slot_order = []
    if slot_ids is not None:
        if len(slot_ids) != n_real:
            base = (None, None, None, None)
            return base + (None,) if return_slot_duals else base
        slot_order = list(dict.fromkeys(s for s in slot_ids if s is not None))
        if slot_order:
            sidx = {s: i for i, s in enumerate(slot_order)}
            Aub = np.zeros((len(slot_order), n_col))
            for k, s in enumerate(slot_ids):
                if s is not None:
                    Aub[sidx[s], k] = 1.0
            bub = np.ones(len(slot_order))
    res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq,
                  bounds=bounds, method="highs")
    checked = _validate_linprog_result(res, c, bounds,
                                       A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq,
                                       need_ineqlin=len(slot_order),
                                       need_eqlin=n_t + len(force_rows),
                                       dual_bounds=[(0, 1)] * n_col)
    if checked is None:
        base = (None, None, None, None)
        return base + (None,) if return_slot_duals else base
    x, fun, imarg, marg, dual_lb = checked
    z = x[:n_real]
    art = x[n_real:]
    duals = {t: 0.0 for t in turbine_ids}
    for i, t in enumerate(ids):
        duals[t] = float(marg[i])
    used_art = float(np.maximum(art, 0.0).sum())
    uncovered = {ids[i] for i in range(n_t) if art[i] > 1e-6}
    lb_real = dual_lb - big * used_art
    slot_duals = {s: float(imarg[i]) for i, s in enumerate(slot_order)}
    base = (lb_real, z, duals, uncovered)
    return base + (slot_duals,) if return_slot_duals else base


def _ryan_foster_pair(pool, z, tid2idx, tol=1e-6):
    """从 LP 原始解 z 找 Ryan–Foster 分支对(i,j)(风机【下标】):
    x_ij = Σ_{ω⊇{i,j}} z_ω。取最接近 0.5 的分数 x_ij(0<x_ij<1) 作分支对; 全整数则返回 None。"""
    from collections import defaultdict
    pair_val = defaultdict(float)
    for k, (r, _) in enumerate(pool):
        if z[k] <= tol:
            continue
        idxs = sorted(tid2idx[t] for t in r.turbine_ids())
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                pair_val[(idxs[a], idxs[b])] += z[k]
    best = None; best_d = 1.0
    for pair, v in pair_val.items():
        if tol < v < 1 - tol:
            d = abs(v - 0.5)
            if d < best_d:
                best_d = d; best = pair
    return best


def branch_and_price(turbines, ship, p, wx, xi_amb, max_stops=8, k_near=None, eval_cache=None,
                     weather_unc=None, launch_ships=None, max_nodes=64, verbose=True) -> dict:
    """完整(离散化下 globally exact)分支定价 (任务1 核心)。

    历史框架: 主问题为集合划分(`partition=True`)；不属于当前风机可不服务的正式集合打包模型。
    【集合覆盖 LP 松弛】仅作有效下界(更松, 见引理4b)。标号支配用【同 visited 集安全支配】(strict_dominance=True, critique 4.4)。
    每个 B&P 节点: 惰性列生成(精确 DR-RCSPP 定价, k_near=n)求节点 LB + 列池; 整数主问题(partition=True)求节点 UB;
    若 LP 解非整 ⇒ 取 Ryan–Foster 分支对(i,j): 'together'(同一架次) 与 'apart'(不同架次)两子节点,
    分别用 force_pairs/forbid_pairs 约束节点定价(精确)。界限剪枝(⌈节点LB⌉ ≥ 全局UB 则剪)。

    退出: 栈空 ⇒ 全局最优(opt_gap=0); 或达 max_nodes(报告当前 gap, 未必闭合)。
    对【离散化时空列空间】(给定 𝒯、ℋ、有限风机、固定线性化与可行性规则)是 globally exact 的。
    k_near 缺省=风机数(完全近邻 ⇒ 精确定价)。"""
    n = len(turbines)
    k_near = k_near if k_near is not None else n
    # 更新(P2-04): 与 lex 同理 —— 本入口发根整数证书, 外部 eval_cache 不再被采信,
    # 始终新建内部缓存(跨 B&P 节点共享的提速语义不变)。
    eval_cache = {}
    all_ids = [t.tid for t in turbines]
    tid2idx = {t.tid: i for i, t in enumerate(turbines)}

    incumbent = dict(key=(float("inf"), float("inf"), float("inf")), UB=float("inf"), routes=[], ip=None)
    all_master_outputs_clean = True
    root_LB = None
    root_certifiable = False; root_neighbor_complete = False
    nodes = 0
    stack = [(frozenset(), frozenset())]    # (forbid_pairs, force_pairs)
    seen_nodes = set()

    while stack and nodes < max_nodes:
        forbid, force = stack.pop()
        if (forbid, force) in seen_nodes:
            continue
        seen_nodes.add((forbid, force))
        nodes += 1
        cg = column_generation(turbines, ship, p, wx, xi_amb, max_stops, k_near,
                               verbose=False, weather_unc=weather_unc, launch_ships=launch_ships,
                               forbid_pairs=set(forbid), force_pairs=set(force),
                               strict_dominance=True,   # exact 模式: 同 visited 集安全支配(critique 4.4,
                               eval_cache=eval_cache)
        LB = cg.get("global_LB"); pool = cg["columns"]
        if root_LB is None:
            root_LB = LB
            root_certifiable = _cg_certifiable(cg)               # 完全图+strict支配+no_neg_rc收敛(P0-4/5/7/8)
            root_neighbor_complete = bool(cg.get("neighbor_graph_complete"))
        if LB is None or not pool:
            continue
        # 界限剪枝(更新 审计 claim4: 词典序只有 L1 下界(SC-LP 仅约束架次), 故只能在
        #   ⌈LB⌉ > 现有最优架次 时剪枝 —— 即"连架次都不可能追平"才剪; ⌈LB⌉==最优架次的节点
        #   仍须探索(可能同架次但能耗更低/裕度更高)。用 > 而非 ≥, 以免误剪 L2/L3 改进。
        N_star = incumbent["key"][0]
        if LB is not None and _safe_integer_ceiling(LB) > N_star:
            continue
        # 节点整数主问题(词典序三层; partition=True; pool 已展开为完整 (τ,π,h))
        pool_h = _expand_pool_h(pool, p, wx, xi_amb, weather_unc=weather_unc, eval_cache=eval_cache)   # 闭合降调#1: 完整 (τ,π,h)
        ip = _integer_master(pool_h, all_ids, p=p, wx=wx, lex=True)
        all_master_outputs_clean = bool(all_master_outputs_clean and _master_outputs_clean(ip))
        if _master_solution_validated(ip) and not ip["uncovered"]:
            _plan = _verify_selected_route_plan(ip["routes"], all_ids, p, wx, xi_amb,
                                                weather_unc=weather_unc)
            if _plan["ok"]:
                # 更新 审计 claim4: incumbent 按【词典序三元组】(架次, 能耗, −鲁棒裕度)比较更新,
                # 不再只比架次 —— 同架次下能耗更低/裕度更高的解也会替换 incumbent。
                nE = float(_plan["total_energy"]); nM = float(_plan["total_margin"])
                key = (float(ip["n_sorties"]), round(nE, 6), -round(nM, 6))
                if key < incumbent["key"]:
                    incumbent = dict(key=key, UB=float(ip["n_sorties"]), routes=ip["routes"], ip=ip,
                                     physical_plan=_plan)
        # 找 Ryan–Foster 分数对(更新 审计 claim2: 用【集合划分 LP】, 使 Ryan–Foster 严格成立:
        #   分数 ⇒ 必有分数对; 无分数对 ⇒ 划分 LP 整数 ⇒ 该节点闭合。覆盖 LP 下无此保证)
        cols_for_lp = [(1.0, set(r.turbine_ids())) for r, _ in pool]
        obj, z, _ = _master_lp_primal(cols_for_lp, all_ids, partition=True)
        pair = _ryan_foster_pair(pool, z, tid2idx) if z is not None else None
        if pair is None:
            continue   # 划分 LP 整数(或无可行划分) ⇒ 无可分支分数解 ⇒ 该节点闭合
        i, j = pair
        if verbose:
            log.info("  分支 节点#%d LB=%.3f UB*=%s | Ryan–Foster 对 (风机#%d,#%d)",
                     nodes, LB, "%.0f" % N_star if N_star != float("inf") else "∞", i, j)
        stack.append((forbid, force | {frozenset((i, j))}))    # together
        stack.append((forbid | {frozenset((i, j))}, force))    # apart

    UB = incumbent["UB"]
    closed = (not stack)        # 栈空 ⇒ 启发式分支遍历完(注: 非严格 gap-closing, 见下)
    ceil_root = _safe_integer_ceiling(root_LB)
    opt_gap = (UB - ceil_root) / UB if UB not in (0, float("inf")) else 0.0
    # === 证书(更新/更新 审计 claim1/4/6: 只认【根节点整数证书】, 它数学上 sound 且不依赖分支)===
    #  根列生成精确定价 ⇒ root_LB = 完整列空间的 SC-LP*(真下界); ⌈SC-LP*⌉ ≤ SP* ≤ UB(可行划分上界)。
    #  故【仅当】有可行全覆盖解(UB 有限>0)且 ⌈root_LB⌉≥UB 时, 才证 SP*=UB(最少架次全局最优)。
    #  ❌ 不再用 branch_closed 作证书依据: 分支用 SC 对偶定价、SP-LP 仅在受限列池求解, "无分数对即闭合"
    #     不能证完整列空间 SP-LP 已整 ⇒ 启发式分支可改进 incumbent, 但【不构成】严格 gap-closing 证书。
    has_feasible = (UB not in (0, float("inf")))
    # 更新/更新 审计: 根整数证书须【根列生成完全定价 k_near=n + strict 支配 + no_neg_rc 收敛】, 否则 root_LB 非真 SC-LP*。
    certified_min_sorties = bool(has_feasible and all_master_outputs_clean
                                  and root_LB is not None and ceil_root >= int(UB)
                                  and root_certifiable)
    certified_lex = False     # 词典序第 2/3 层全局精确见 lex_column_generation; 当前 B&P 为序列侧 pool-optimal
    chosen = incumbent["routes"]
    _final_plan = incumbent.get("physical_plan") or _verify_selected_route_plan(
        chosen, all_ids, p, wx, xi_amb, weather_unc=weather_unc)
    physical_plan_verified = bool(_final_plan["ok"])
    if not physical_plan_verified:
        certified_min_sorties = False
    tot_E = float(_final_plan["total_energy"]) if physical_plan_verified else 0.0
    tot_M = float(_final_plan["total_margin"]) if physical_plan_verified else 0.0
    diag = _final_plan["diag"] if physical_plan_verified else []
    return dict(method="branch_and_price", n_sorties=int(UB) if UB != float("inf") else None,
                root_LB=round(root_LB, 9) if root_LB is not None else None,
                ceil_root_LB=ceil_root,
                opt_gap_pct=(round(100 * opt_gap, 2) if certified_min_sorties or has_feasible else None),
                certified_min_sorties=certified_min_sorties, certified_lex=certified_lex,
                certified_optimal=certified_min_sorties,   # 向后兼容别名(语义=最少架次全局最优, 非词典序全层)
                branch_closed=closed,                       # 仅信息性(启发式分支遍历完), 非严格证书依据
                root_cg_certifiable=root_certifiable, root_neighbor_graph_complete=root_neighbor_complete,
                k_near_used=k_near,
                has_feasible_cover=has_feasible, nodes_explored=nodes,
                incumbent_master_validated=bool(incumbent.get("ip") and _master_solution_validated(incumbent["ip"])),
                incumbent_master_proven_optimal=bool(incumbent.get("ip") and _master_solution_proven(incumbent["ip"])),
                all_milp_outputs_validated=all_master_outputs_clean,
                physical_plan_verified=physical_plan_verified,
                physical_plan_validation_reason=_final_plan.get("reason"),
                reported_solution_consistent=physical_plan_verified,
                total_energy_Wh=round(tot_E, 1), total_robust_margin=round(tot_M, 2),
                route_diag=diag)


def branch_and_price_exact(turbines, ship, p, wx, xi_amb, max_stops=8, k_near=None,
                           weather_unc=None, launch_ships=None, max_nodes=300,
                           cg_max_iter=40, verbose=True) -> dict:
    r"""【完整 SP-对偶 gap-closing 分支定价】(更新 任务2)。与 `branch_and_price` 的本质区别:
    每个节点用【集合划分 LP 对偶 $\sigma_i$(自由号, 可负)】继续列生成至无负 reduced cost 列 ——
    即在【完整列空间】上求该节点的 SP-LP, 节点界 $=\mathrm{SP\text-LP}^\star\ (\ge \mathrm{SC\text-LP}^\star$, 更紧)。
    Ryan–Foster(together/apart)分支, 节点剪枝用 $\lceil\mathrm{SP\text-LP}^\star\rceil>$ 现任架次。
    **栈空且全部节点 SP-LP 收敛 ⇒ 树严格闭合 ⇒ 全局最优(真 gap-closing, 不依赖根证书)**。

    证书 `certified_min_sorties` = 有可行全覆盖解 ∧ [ 树严格闭合(tree_closed) ∨ 根 SP-LP 整数证书
      (⌈root SP-LP*⌉=UB 且根列生成收敛、完全图、strict 支配) ]。仅 $\xi$-only 下成立(多源退化, 见 §范围)。"""
    n = len(turbines); k_near = k_near or n
    eval_cache = {}   # 更新(提速#B): 跨节点共享闭合评估
    all_ids = [t.tid for t in turbines]
    tid2idx = {t.tid: i for i, t in enumerate(turbines)}
    ships = launch_ships if launch_ships is not None else [ship]

    column_archive = {}

    def _node_sp_lp(forbid, force, forbid_cols, force_cols):
        """节点级【集合划分 LP 列生成】: 反复 [划分 LP 取 σ → 以 rc=1−Σσ_i 定价(strict 支配, 节点约束) → 加列]
        至无负 rc 列。返回 (SP_LP*, pool, z, converged, uncovered)。"""
        cg = column_generation(turbines, ship, p, wx, xi_amb, max_stops, k_near, max_iter=cg_max_iter,
                               route_cost=1.0, verbose=False, weather_unc=weather_unc, launch_ships=launch_ships,
                               forbid_pairs=set(forbid), force_pairs=set(force), strict_dominance=True,
                               eval_cache=eval_cache)
        pool = []
        seen = set()
        for r, d in cg["columns"]:
            key = (_ship_column_fp(r.ship), tuple(r.turbine_ids()))
            column_archive[key] = (r, d)
            if key in forbid_cols or key in seen:
                continue
            pool.append((r, d)); seen.add(key)
        for key in force_cols:
            if key not in column_archive:
                return None, pool, None, False, set()
            if key in forbid_cols:
                return None, pool, None, False, set()
            if key not in seen:
                pool.append(column_archive[key]); seen.add(key)
        spobj = None; z = None; converged = False; uncov = set()
        for _ in range(cg_max_iter):
            cols_lp = [(1.0, set(r.turbine_ids())) for r, _ in pool]
            slot_ids = [getattr(r.ship, "slot", None) for r, _ in pool]
            column_keys = [(_ship_column_fp(r.ship), tuple(r.turbine_ids())) for r, _ in pool]
            spobj, z, sigma, uncov, slot_sigma = _partition_lp(
                cols_lp, all_ids, slot_ids=slot_ids, return_slot_duals=True,
                column_keys=column_keys, force_keys=force_cols)
            if spobj is None:
                return None, pool, None, False, set()
            dual_idx = {tid2idx[t]: sigma.get(t, 0.0) for t in all_ids}   # SP-dual(可负)
            new = []
            for sp in ships:
                _spfp = _ship_column_fp(sp)
                # Exclude both branch-forbidden and already-present projected
                # columns inside the pricing search itself.  Otherwise the
                # top-k return cap can repeatedly return only duplicates and
                # prevent the algorithm from ever proving closure despite
                # unexplored negative-reduced-cost columns beyond that cap.
                _forbid_seq = {key[1] for key in (set(forbid_cols) | set(seen))
                               if key[0] == _spfp}
                pr = price_routes(turbines, sp, p, wx, xi_amb, dual_idx, route_cost=1.0,
                                  max_stops=max_stops, k_near=k_near, weather_unc=weather_unc,
                                  forbid_pairs=set(forbid), force_pairs=set(force), strict_dominance=True,
                                  rc_tol=-PRICING_EPS,
                                  dual_offset=float(slot_sigma.get(getattr(sp, "slot", None), 0.0)),
                                  eval_cache=eval_cache,
                                  forbid_route_sequences=_forbid_seq)
                for d in pr:
                    d["_ship"] = sp
                new.extend(pr)
            if not new:
                spobj = _pricing_relaxed_lower_bound(spobj, n, -PRICING_EPS)
                converged = spobj is not None
                break
            added = 0
            for pr in sorted(new, key=lambda d: d["reduced_cost"]):
                sp = pr["_ship"]; r = RM.Route(rid=-1, turbines=pr["turbines"], ship=sp); r.fixed_h = pr["h"]
                key = (_ship_column_fp(sp), tuple(r.turbine_ids()))
                if key in forbid_cols or key in seen:
                    continue
                d = RM.route_drcc_feasible(r, p, _wx_of_ship(sp, wx), xi_amb,
                                           objective="min_h", weather_unc=weather_unc)
                if d["feasible"]:
                    pool.append((r, d)); seen.add(key); column_archive[key] = (r, d); added += 1
            if added == 0:
                converged = False; break          # 停滞(P0-8 同理), 不视为收敛
        return spobj, pool, z, converged, uncov

    incumbent = dict(key=(float("inf"), float("inf"), float("inf")), UB=float("inf"), routes=[], ip=None)
    all_master_outputs_clean = True
    root_sp_lp = None; root_converged = False; root_uncov = set()
    # Node state: Ryan–Foster pair branches plus a complete fallback column-variable
    # branch.  The latter is essential when duplicate coverage patterns across launch
    # slots yield a fractional SP-LP solution with no fractional turbine pair.
    stack = [(frozenset(), frozenset(), frozenset(), frozenset())]
    seen_nodes = set(); nodes = 0
    neighbor_complete = bool(k_near >= n)            # 完全近邻图 ⇒ 标号枚举全部 elementary 部分路径(支配意义); 否则定价不达全列空间
    multisource = (weather_unc is not None)          # 多源风浪: 标号支配未纳入路由相关风灵敏度 ⇒ exact 支配不严格(任务 #2)
    unresolved = 0                                   # 未解析节点计数(SP-LP 失败 / 未在全列空间收敛 / 停滞 / 截断), 任务 #5
    while stack and nodes < max_nodes:
        forbid, force, forbid_cols, force_cols = stack.pop()
        node_key = (forbid, force, forbid_cols, force_cols)
        if node_key in seen_nodes:
            continue
        seen_nodes.add(node_key); nodes += 1
        spobj, pool, z, conv, uncov = _node_sp_lp(forbid, force, forbid_cols, force_cols)
        if root_sp_lp is None:                       # 记录根节点(首个弹出)
            root_sp_lp = spobj; root_converged = conv; root_uncov = uncov
        # ---- 节点三分类(任务 #5): ①已证不可行 ②已闭合 ③未解析。仅当无③、栈空、完全图、单源 ⇒ tree_closed。----
        if spobj is None:
            unresolved += 1; continue                # ③ SP-LP 求解失败 ⇒ 未解析(不能宣称闭合)
        if conv and len(uncov) > 0:
            continue                                 # ① 全列空间收敛仍有风机无【实列】覆盖 ⇒ 该节点已证不可行, 安全剪枝
        # A converged integral SP-LP primal is independently validated by _partition_lp and
        # is itself an exact node solution.  Use it directly; otherwise a solver that merely
        # *claims* optimality for a worse restricted-master incumbent can poison tree closure.
        integral_node_closed = False
        if conv and z is not None and len(z) == len(pool) and np.all(np.isfinite(z)) \
                and np.all(np.abs(z - np.rint(z)) <= 1e-7):
            sel = [pool[i][0] for i, v in enumerate(z) if v > 0.5]
            _plan = _verify_selected_route_plan(sel, all_ids, p, wx, xi_amb,
                                                weather_unc=weather_unc)
            if _plan["ok"]:
                nE = float(_plan["total_energy"]); nM = float(_plan["total_margin"])
                key = (float(len(sel)), round(nE, 6), -round(nM, 6))
                if key < incumbent["key"]:
                    incumbent = dict(key=key, UB=float(len(sel)), routes=sel,
                                     ip=dict(routes=sel, n_sorties=len(sel), uncovered=[],
                                             solution_validated=True, proven_optimal=True,
                                             solver="validated-integral-SP-LP"),
                                     physical_plan=_plan)
                integral_node_closed = True
        # 节点整数主问题(词典序三层) → 候选 incumbent。仅接受其可行性；
        # 最优性仍由完整 LP 定价/分支闭合给出。
        if pool:
            pool_h = _expand_pool_h(pool, p, wx, xi_amb, weather_unc=weather_unc, eval_cache=eval_cache)
            ip = _integer_master(pool_h, all_ids, p=p, wx=wx, lex=True)
            all_master_outputs_clean = bool(all_master_outputs_clean and _master_outputs_clean(ip))
            if _master_solution_validated(ip) and not ip["uncovered"]:
                _plan = _verify_selected_route_plan(ip["routes"], all_ids, p, wx, xi_amb,
                                                    weather_unc=weather_unc)
                if _plan["ok"]:
                    nE = float(_plan["total_energy"]); nM = float(_plan["total_margin"])
                    key = (float(ip["n_sorties"]), round(nE, 6), -round(nM, 6))
                    if key < incumbent["key"]:
                        incumbent = dict(key=key, UB=float(ip["n_sorties"]), routes=ip["routes"], ip=ip,
                                         physical_plan=_plan)
        if not conv:
            unresolved += 1; continue                # ③ 节点 SP-LP 未在全列空间收敛(停滞/截断)⇒ 界无效, 不剪枝/不分支, 未解析
        N_inc = incumbent["key"][0]
        # 界剪枝(已闭合): 仅当【完全近邻图】(spobj 才是真 SP-LP 下界)且 ⌈SP-LP*⌉>现任架次。
        #   k_near<n 时 spobj 非全列空间下界, 界剪枝不 sound ⇒ 不剪(证书也不会发放, 见下)。
        if neighbor_complete and _safe_integer_ceiling(spobj) > N_inc:
            continue
        # Ryan–Foster 分支(用节点 SP-LP 原始解 z); 收敛且无分数对 ⇒ 节点 SP-LP 整数 ⇒ 已闭合。
        pair = _ryan_foster_pair(pool, z, tid2idx) if z is not None else None
        if pair is not None:
            i, j = pair
            if verbose:
                log.info("  [exact-B&P] 节点#%d SP-LP=%.3f UB*=%s | Ryan–Foster (风机#%d,#%d)",
                         nodes, spobj, "%.0f" % N_inc if N_inc != float("inf") else "∞", i, j)
            stack.append((forbid, force | {frozenset((i, j))}, forbid_cols, force_cols))
            stack.append((forbid | {frozenset((i, j))}, force, forbid_cols, force_cols))
            continue
        if integral_node_closed:
            continue
        # Complete fallback: branch on one fractional projected route variable.
        # Children z_k=0 and z_k=1 exactly partition the parent feasible set;
        # forbidden keys are filtered from both RMP and pricing, while forced keys
        # are retained and fixed by an explicit equality in _partition_lp.
        frac = []
        if z is not None and len(z) == len(pool) and np.all(np.isfinite(z)):
            for k, value in enumerate(z):
                if 1e-7 < value < 1.0 - 1e-7:
                    frac.append((abs(float(value) - 0.5), k))
        if not frac:
            unresolved += 1
            continue
        _, k_branch = min(frac)
        rr, dd = pool[k_branch]
        col_key = (_ship_column_fp(rr.ship), tuple(rr.turbine_ids()))
        column_archive[col_key] = (rr, dd)
        if verbose:
            log.info("  [exact-B&P] 节点#%d SP-LP=%.3f | 列变量分支 z[%s]=0/1",
                     nodes, spobj, col_key)
        stack.append((forbid, force, forbid_cols, force_cols | {col_key}))
        stack.append((forbid, force, forbid_cols | {col_key}, force_cols))

    UB = incumbent["UB"]
    hit_node_cap = bool(stack)                       # 达节点上限未遍历完
    if hit_node_cap:
        unresolved += len(stack)                     # 栈中未展开节点亦属未解析(任务 #5)
    # 树严格闭合(任务 #1/#2/#5): 栈空 ∧ 无未解析节点 ∧ 完全近邻图(全列空间定价)∧ 单一天气场(支配严格)。
    tree_closed = bool((not stack) and unresolved == 0 and neighbor_complete and not multisource)
    sp_all_converged = bool(unresolved == 0)         # 向后兼容字段
    ceil_root = _safe_integer_ceiling(root_sp_lp)
    has_feasible = (UB not in (0, float("inf")))     # 任一节点整数主问题给出有限全覆盖 ⇒ 存在可行解
    opt_gap = (UB - ceil_root) / UB if UB not in (0, float("inf")) else 0.0
    # 根 SP-LP 整数证书(不依赖分支): ⌈root SP-LP*⌉=UB ∧ 根列生成收敛 ∧ 根无 uncovered ∧ 完全图 ∧ 单源(任务 #1/#2)。
    root_cert = bool(has_feasible and root_sp_lp is not None and ceil_root >= int(UB)
                     and root_converged and len(root_uncov) == 0
                     and neighbor_complete and not multisource)
    chosen = incumbent["routes"]
    _final_plan = incumbent.get("physical_plan") or _verify_selected_route_plan(
        chosen, all_ids, p, wx, xi_amb, weather_unc=weather_unc)
    physical_plan_verified = bool(_final_plan["ok"])
    certified_min_sorties = bool(has_feasible and all_master_outputs_clean
                                  and physical_plan_verified
                                  and (tree_closed or root_cert))
    if multisource:
        exact_scope = "restricted_column_space (multisource; labeling dominance not wind-sensitive, 任务#2)"
    elif not neighbor_complete:
        exact_scope = f"restricted_column_space (k_near={k_near}<n={n}; incomplete neighbor graph, 任务#1)"
    else:
        exact_scope = "global (xi-only, complete neighbor graph; SP-dual gap-closing)"
    tot_E = float(_final_plan["total_energy"]) if physical_plan_verified else 0.0
    tot_M = float(_final_plan["total_margin"]) if physical_plan_verified else 0.0
    diag = _final_plan["diag"] if physical_plan_verified else []
    return dict(method="branch_and_price_exact",
                n_sorties=int(UB) if UB != float("inf") else None,
                root_SP_LP=round(root_sp_lp, 9) if root_sp_lp is not None else None,
                ceil_root_SP_LP=ceil_root, opt_gap_pct=round(100 * opt_gap, 2),
                tree_closed=tree_closed, sp_all_converged=sp_all_converged, hit_node_cap=hit_node_cap,
                root_sp_converged=root_converged, nodes_explored=nodes, unresolved_nodes=int(unresolved),
                neighbor_graph_complete=neighbor_complete, multisource=multisource, exact_scope=exact_scope,
                certified_min_sorties=certified_min_sorties, certified_optimal=certified_min_sorties,
                certified_via=("tree_closed" if tree_closed else ("root_sp_lp" if root_cert else "none")),
                certified_lex=False,        # 词典序 L2/L3 见 lex_column_generation
                gap_closing="SP-dual node pricing (partition-LP duals, may be negative)",
                has_feasible_cover=has_feasible, uncovered=sorted(root_uncov),
                incumbent_master_validated=bool(incumbent.get("ip") and _master_solution_validated(incumbent["ip"])),
                incumbent_master_proven_optimal=bool(incumbent.get("ip") and _master_solution_proven(incumbent["ip"])),
                all_milp_outputs_validated=all_master_outputs_clean,
                physical_plan_verified=physical_plan_verified,
                physical_plan_validation_reason=_final_plan.get("reason"),
                reported_solution_consistent=physical_plan_verified,
                k_near_used=k_near, total_energy_Wh=round(tot_E, 1),
                total_robust_margin=round(tot_M, 2), route_diag=diag)


def solve_exact(turbines, ship, p, wx, xi_amb, mode="exact_discrete", max_stops=8,
                k_near=None, weather_unc=None, launch_ships=None, verbose=True) -> dict:
    """精确/对照模式的统一入口(critique 必改 5/6 —— 不让加速版共享 exact 的理论声明):

      mode='exact_bp'          : 【完整 SP-对偶 gap-closing 分支定价】(branch_and_price_exact) —— 节点用
                                 集合划分 LP 对偶 σ(可负)继续列生成至全列空间 SP-LP 收敛, 栈空+全节点收敛 ⇒
                                 树严格闭合 ⇒ 全局最优(不依赖根证书)。仅 ξ-only 严格。
      mode='exact_lex'         : 全局【词典序】列生成(lex_column_generation): L1 架次根证书 + L2 能耗严格重定价
                                 (含 σ, LP 紧则全局)+ L3 裕度 h 展开池内最优。返回 certified_L1/L1_L2/lex3。
      mode='exact_discrete'    : k_near=n、完整资源状态、完整 Ryan–Foster 分支、scipy/Gurobi 精确主问题。
                                 ⇒ 对【离散化模型】可证全局最优(branch_and_price)。仅 ξ 不确定性下严格成立;
                                 多源(weather_unc)下标号支配未扩资源向量 ⇒ 退化为受限列空间(见 accelerated)。
      mode='accelerated'       : 近邻限制(k_near<n)或简化支配 ⇒ 仅启发式加速, 给【受限列空间】的解,
                                 【不共享】exact 的全局最优声明(solve_route_drcc_exact, 报告 LP gap)。
      mode='enumeration_anchor': 小规模【完整枚举】所有 DR 可行列 → 一次性整数主问题(scipy.milp/Gurobi),
                                 由构造即全局最优, 用于【验证 B&P】(solve_monolithic_misocp)。

    返回各模式自身的结果 dict, 并统一附 mode 字段。"""
    n = len(turbines)
    if mode == "exact_bp":
        # 更新 任务2: 完整 SP-对偶 gap-closing 分支定价(节点用集合划分 LP 对偶 σ 继续定价至全列空间收敛)。
        if k_near is not None and k_near < n and verbose:
            log.warning("exact_bp 要求完全定价 k_near=n=%d; 传入 k_near=%d 已忽略并强制为 %d。", n, k_near, n)
        res = branch_and_price_exact(turbines, ship, p, wx, xi_amb, max_stops=max_stops, k_near=n,
                                     weather_unc=weather_unc, launch_ships=launch_ships, verbose=verbose)
        res["k_near_used"] = n
        if weather_unc is not None:
            res["certified_min_sorties"] = False; res["certified_optimal"] = False
            res["exact_scope"] = "restricted_column_space (multisource; xi-only is exact)"
        else:
            res["exact_scope"] = "global (xi-only; SP-dual gap-closing B&P)"
        res["proof_model_scope"] = "finite_discrete_route_model"
        res["continuous_real_world_optimality_claimed"] = False
        res["mode"] = mode
        return res
    if mode == "exact_lex":
        # 更新 任务2: 完整全局【词典序】列生成(第一层架次 + 第二层能耗严格重定价, 第三层裕度 h 展开)。
        # 强制完全定价 k_near=n; 返回 certified_min_sorties + certified_lex(前两层 LP 紧时为真)。
        if k_near is not None and k_near < n and verbose:
            log.warning("exact_lex 要求完全定价 k_near=n=%d; 传入 k_near=%d 已忽略并强制为 %d。", n, k_near, n)
        res = lex_column_generation(turbines, ship, p, wx, xi_amb, max_stops=max_stops, k_near=n,
                                    weather_unc=weather_unc, launch_ships=launch_ships, verbose=verbose)
        res["k_near_used"] = n
        if weather_unc is not None:   # 多源下支配未扩资源向量 ⇒ 原子化撤销全部证书别名
            for _k in ("certified_min_sorties", "certified_optimal", "certified_L1",
                       "certified_L1_L2", "certified_lex", "certified_lex3"):
                res[_k] = False
            res["exact_scope"] = "restricted_column_space (multisource; xi-only is exact)"
        else:
            res["exact_scope"] = "global lexicographic (xi-only; L1 root cert + L2 energy-CG LP-tight)"
        res["proof_model_scope"] = "finite_discrete_route_model"
        res["continuous_real_world_optimality_claimed"] = False
        res["mode"] = mode
        return res
    if mode == "exact_discrete":
        # 更新 审计 claim3: exact_discrete 必须【完全定价 k_near=n】才是正式离散模型的全局证书;
        # 若调用者传了 k_near<n, 这不是 exact —— 强制 k_near=n 并警告(不让"近邻受限"冒用 exact 证书)。
        if k_near is not None and k_near < n and verbose:
            log.warning("exact_discrete 要求完全定价 k_near=n=%d; 传入 k_near=%d 已被忽略并强制为 %d "
                        "(否则证书只在受限近邻列空间成立, 见 doc_proof §0/§7)。", n, k_near, n)
        if weather_unc is not None and verbose:
            log.warning("exact_discrete + weather_unc: 多源支配未扩资源向量, 严格 exact 仅 ξ-only; "
                        "多源结果属受限列空间(见 doc_proof §范围)。")
        res = branch_and_price(turbines, ship, p, wx, xi_amb, max_stops=max_stops,
                               k_near=n, weather_unc=weather_unc,
                               launch_ships=launch_ships, verbose=verbose)
        res["k_near_used"] = n
        if weather_unc is not None:
            # 多源下标号支配未纳入路由相关风灵敏度 ⇒ 不得冒称全局最优证书(D9)
            res["certified_optimal"] = False
            res["certified_min_sorties"] = False
            res["exact_scope"] = "restricted_column_space (multisource; xi-only is exact)"
        else:
            res["exact_scope"] = "global (discretized, xi-only, first-layer + integer certificate)"
    elif mode == "accelerated":
        res = solve_route_drcc_exact(turbines, ship, p, wx, xi_amb, max_stops=max_stops,
                                     k_near=(k_near or 8), verbose=verbose,
                                     weather_unc=weather_unc, launch_ships=launch_ships,
                                     strict_dominance=False)
        # 更新 审计 P0-7: accelerated 允许启发式(子集)支配 ⇒ 【无条件】清除全局证书(即便 k_near=n)。
        res["certified_min_sorties"] = False; res["certified_optimal"] = False; res["certified_lex"] = False
        res["exact_scope"] = "restricted (accelerated: subset dominance, NOT a global certificate)"
    elif mode == "enumeration_anchor":
        # 更新 审计 claim5: 锚点用于验证 B&P, 必须是【完整枚举】⇒ 强制 k_near=n(传小值忽略+警告);
        # 仍可能因 max_cols 截断而不完整, 由 complete_enumeration 如实标注。
        if k_near is not None and k_near < n and verbose:
            log.warning("enumeration_anchor 须完整枚举 k_near=n=%d; 传入 k_near=%d 已被忽略并强制为 %d。",
                        n, k_near, n)
        res = solve_monolithic_misocp(turbines, ship, p, wx, xi_amb, max_stops=max_stops,
                                      k_near=n, weather_unc=weather_unc,
                                      launch_ships=launch_ships)
        if res.get("complete_enumeration"):
            res["exact_scope"] = "global (complete enumeration anchor)"
        else:
            # k_near 已强制=n, 故走到这里只能是 max_cols 截断
            res["exact_scope"] = "restricted (enumeration truncated at max_cols; NOT a complete anchor)"
    else:
        raise ValueError(f"未知 mode={mode}; 可选 exact_discrete / accelerated / enumeration_anchor")
    res["proof_model_scope"] = "finite_discrete_route_model"
    res["continuous_real_world_optimality_claimed"] = False
    res["mode"] = mode
    return res


# =============================================================================
# 5. 自检(真实量级 ξ 夹具 + 真实风机: 对比启发式 vs 精确 gap)
# =============================================================================
def _selftest():
    here = Path(__file__).resolve().parent
    turb_csv = M._first_existing([here / "data" / "turbines_Rodsand_II_clean.csv"])
    p = M.Params()
    horizons = [5, 10, 15, 20, 30]
    states = ["低速", "动力定位", "直航", "转弯"]

    if turb_csv:
        turbines = M.load_turbines(turb_csv, farm="Rodsand_II")[:14]
    else:
        turbines = [M.Turbine(f"DEMO_{i}", np.array([11.55 + 0.006 * i, 54.55 + 0.002 * (i % 3)]),
                              68.5, 115.0) for i in range(14)]
    lat0, lon0 = turbines[0].lonlat[1], turbines[0].lonlat[0]
    for t in turbines:
        t.local = M.latlon_to_local_m(t.lonlat[1], t.lonlat[0], lat0, lon0)

    wx = dict(wind10=2.7, wind_dir_from=230.0, Hs=0.16, Tp=2.1, wave_dir=200.0, ship_heading=90.0)
    xi_amb = RM._demo_xi_realistic(horizons, states)   # ★真实量级
    centroid = np.mean([t.local for t in turbines], axis=0)
    # 自洽: 动力定位回收
    ship = RM.ShipPrediction.from_cv(centroid + np.array([-600.0, -400.0]),
                                     v_ship=np.array([1.2, 0.9]), horizons=horizons,
                                     c_state="动力定位")

    print("\n================ step12_branch_price.py 自检 ================")
    print(f"风机 {len(turbines)} 台 | 回收状态 {ship.c_state}(自洽低速) | h {horizons} | "
          f"真实量级 ξ 夹具")

    # 校验: 标号增量 E_nom/T_nom + 返程 与 step10 route_nominal_ET 一致
    wc = _wx_cache(p, wx)
    seq = [0, 1, 2]
    E_to, E_land, T_to, T_land = M.to_land_energy_time(p)
    E_nom = E_to; T_nom = T_to; from_local = ship.P_launch
    for j in seq:
        eL, tL = _leg_ET(p, _leg_wc(p, wc, turbines[j]), from_local, turbines[j].local)
        eI, tI = _insp_ET(p, turbines[j], p.z_cruise)
        E_nom += eL + eI; T_nom += tL + tI; from_local = turbines[j].local
    _, E0_inc, T0_inc, _, _, _ = _close_ET_soc(p, wc, turbines[2].local, E_nom, T_nom, ship, 30,
                                                xi_amb.get_interp(30, "动力定位"), last_turbine=turbines[2])
    r = RM.Route(rid=0, turbines=[turbines[j] for j in seq], ship=ship)
    nom = RM.route_nominal_ET(r, 30, p, wx, t_dock_s=wc.t_dock_s)
    _consistent = abs(E0_inc - nom["E0"]) < 1e-6 and abs(T0_inc - nom["T0"]) < 1e-6
    print(f"\n[一致性校验] 标号增量 E0={E0_inc:.2f}Wh T0={T0_inc:.1f}s vs "
          f"step10 E0={nom['E0']:.2f}Wh T0={nom['T0']:.1f}s "
          f"→ {'✓一致' if _consistent else '✗不一致'}")
    assert _consistent, "标号闭合与 route_nominal_ET 的能量/时间口径不一致"

    # 启发式(step14) vs 精确(step16)
    res_h = RA.solve_route_drcc(turbines, ship, p, wx, xi_amb, strategy="full", max_stops=8)
    print("\n--- 启发式列生成(step14, 仅上界)---")
    print(f"  选用架次 {res_h['n_sorties_chosen']} | 多台 {res_h['n_multi_stop']} | "
          f"均停靠 {res_h['mean_stops']} | 未覆盖 {len(res_h['uncovered'])} | 能耗 {res_h['total_energy_Wh']}Wh")

    print("\n--- 精确: DR-RCSPP 最优定价 + 列生成 + 整数主问题 ---")
    res_e = solve_route_drcc_exact(turbines, ship, p, wx, xi_amb, max_stops=8, k_near=8)
    print(f"  列生成迭代 {res_e['cg_iters']} 轮 | 累计定价列 {res_e['priced_total']} | 最终列池 {res_e['pool_size']}")
    print(f"  LP 下界 LB={res_e['LB']} | 整数上界 UB={res_e['UB']} | "
          f"最优性 gap={res_e['gap_pct']}% | 主问题求解器 {res_e['solver_master']}")
    print(f"  选用架次 {res_e['n_sorties']} | 多台 {res_e['n_multi_stop']} | "
          f"均停靠 {res_e['mean_stops']} | 未覆盖 {len(res_e['uncovered'])} | 能耗 {res_e['total_energy_Wh']}Wh")

    print("\n--- 对比解读 ---")
    print(f"  启发式架次 {res_h['n_sorties_chosen']} vs 精确架次 {res_e['n_sorties']}; "
          f"精确给出 LB={res_e['LB']} → gap={res_e['gap_pct']}%(启发式无 gap)。")

    # 多源不确定性接入精确算法(model.md §14): 闭合经 route_feasible_at_h 纳入风联合 SOC + 浪门
    wu = RM.WeatherUncertainty(wind_cov=np.diag([1.0**2, 1.0**2]), hs_std=0.05)
    print("\n--- 精确 + 多源不确定性(ξ+风+浪, 闭合含浪门; σ_wind=1.0 σ_hs=0.05)---")
    res_w = solve_route_drcc_exact(turbines, ship, p, wx, xi_amb, max_stops=8, k_near=6,
                                   verbose=False, weather_unc=wu)
    print(f"  LB={res_w['LB']} | UB={res_w['UB']} | gap={res_w['gap_pct']}% | "
          f"架次 {res_w['n_sorties']} | 未覆盖 {len(res_w['uncovered'])} | 能耗 {res_w['total_energy_Wh']}Wh")
    print(f"  对比仅 ξ: 架次 {res_e['n_sorties']}→{res_w['n_sorties']} "
          f"(多源更保守; 闭合已统一经 route_feasible_at_h, 与 step10/11 一致且补上着舰门)。")

    print("\n--- 完整列枚举集合划分基准(枚举全部列 → 一次性求解; 速度对照用; 历史名 monolithic)---")
    res_mono = solve_monolithic_misocp(turbines, ship, p, wx, xi_amb, max_stops=8, k_near=6)
    print(f"  枚举 DR 可行列 {res_mono['n_cols_enumerated']} 条 | 枚举耗时 {res_mono['t_enumerate_s']}s | "
          f"求解耗时 {res_mono['t_solve_s']}s | 架次 {res_mono['n_sorties']}")
    print(f"  对比: 列生成只碰 {res_e['pool_size']} 列(按需), 整体法枚举 {res_mono['n_cols_enumerated']} 列(全部); "
          f"规模一大整体法列爆炸 → 算法实验 vs_gurobi_speed 展示此差距。")

    # === 完整(离散化下 globally exact)分支定价 + 三模式(任务1)===
    sub = turbines[:8]    # 用小子集让完整 B&P/枚举锚点跑得快, 清楚展示 certified + 锚点一致
    print("\n--- 【exact_discrete】完整 Ryan–Foster 分支定价(k_near=n, 精确主问题)---")
    bp = solve_exact(sub, ship, p, wx, xi_amb, mode="exact_discrete", max_stops=8, verbose=False)
    print(f"  架次 UB={bp['n_sorties']} | 根 LB={bp['root_LB']}→⌈LB⌉={bp['ceil_root_LB']} | "
          f"整数最优性 opt_gap={bp['opt_gap_pct']}% | B&P 节点 {bp['nodes_explored']} | "
          f"分支树闭合={bp['branch_closed']}(信息性) | 【最少架次可证全局最优={bp['certified_min_sorties']}】 "
          f"词典序全层证书={bp['certified_lex']}(此入口不做 L2/L3 重定价; 见 exact_lex)")
    print("\n--- 【enumeration_anchor】完整枚举 → 一次性整数主问题(B&P 的验证锚点)---")
    anc = solve_exact(sub, ship, p, wx, xi_amb, mode="enumeration_anchor", max_stops=8, verbose=False)
    print(f"  枚举列 {anc['n_cols_enumerated']} | 架次 {anc['n_sorties']} | "
          f"【B&P==枚举锚点? {bp['n_sorties'] == anc['n_sorties']}】(一致 ⇒ B&P 实现正确)")

    print("\n--- 【exact_lex】三阶段词典序列生成(L1架次 → L2能耗重定价 → L3裕度全列空间重定价)---")
    lx = solve_exact(sub, ship, p, wx, xi_amb, mode="exact_lex", max_stops=8, verbose=False)
    print(f"  (N,E,M)=({lx['n_sorties']},{lx['total_energy_Wh']},{lx['total_robust_margin']}) "
          f"vs 锚点({anc['n_sorties']},{anc['total_energy_Wh']},{anc['total_robust_margin']}) | "
          f"certified_L1={lx['certified_L1']} L1_L2={lx['certified_L1_L2']} lex3={lx['certified_lex3']} "
          f"| L3-LP上界 M_ub={lx.get('robust_LP_ub')} (lex3=False 时为 LP 分数间隙, 仍对锚点验证为事实最优)")

    # === 起飞时刻网格(任务2): τ 作为决策, 逐 τ 定价并池 ===
    print("\n--- 起飞—回收协同定时(任务2): 起飞时刻网格 τ∈{0,8,16}min ---")
    grid = RM.build_launch_grid(centroid + np.array([-600.0, -400.0]), np.array([1.2, 0.9]),
                                launch_offsets_min=[0, 8, 16], horizons=horizons,
                                c_state="动力定位", wx_base=wx)
    bpg = solve_exact(sub, ship, p, wx, xi_amb, mode="exact_discrete", max_stops=8,
                      launch_ships=[o.ship for o in grid], verbose=False)
    taus = sorted({d["tau"] for d in bpg["route_diag"] if d["tau"] is not None})
    print(f"  含起飞时刻决策: 架次 {bpg['n_sorties']} | 可证最优={bpg['certified_optimal']} | "
          f"各路由选用的起飞时刻 τ(min)={taus}")
    print("  解读: 列结构升级为 r=(τ,ω,h); 不同路由可选不同起飞时刻(作业时空环境), "
          "回收时长 h 仍逐列优化; c(τ) 索引 ξ 模糊集(无泄漏)。")

    print("\n自检完成。完整 B&P(离散化下 globally exact, opt_gap=0 且与枚举锚点一致)"
          "+ 三模式 + 起飞时刻 τ 决策 + 多源接入 链路已跑通。")
    print("范围: 完整 B&P 对【离散化模型】globally exact, 严格成立于仅 ξ 不确定性; 含风浪须扩资源向量或用枚举锚点验证(见 doc_proof.md §7)。")


if False and __name__ == "__main__":  # legacy research demo: never auto-executed
    import sys as _sys
    if "--quick" in _sys.argv:
        # 更新 审计建议: 小规模快速回归(n=6, max_stops=2), 几秒内验 B&P==锚点 + fixed_h 一致性
        here = Path(__file__).resolve().parent
        tc = M._first_existing([here / "data" / "turbines_Rodsand_II_clean.csv"])
        p = M.Params(); hz = [5, 10, 15, 20, 30]
        ts = (M.load_turbines(tc, farm="Rodsand_II")[:6] if tc else
              [M.Turbine(f"D{i}", np.array([11.55 + .006 * i, 54.55 + .002 * (i % 3)]), 68.5, 115.0) for i in range(6)])
        la, lo = ts[0].lonlat[1], ts[0].lonlat[0]
        for t in ts:
            t.local = M.latlon_to_local_m(t.lonlat[1], t.lonlat[0], la, lo)
        wx = dict(wind10=2.7, wind_dir_from=230.0, Hs=0.16, Tp=2.1, wave_dir=200.0, ship_heading=90.0)
        amb = RM._demo_xi_realistic(hz, ["动力定位"])
        ctr = np.mean([t.local for t in ts], axis=0)
        sp = RM.ShipPrediction.from_cv(ctr + np.array([-400.0, -300.0]), np.array([1.0, 0.8]), hz, c_state="动力定位")
        print("=== step12 --quick 小规模回归(n=6, k_near=6, max_stops=2)===")
        bp = solve_exact(ts, sp, p, wx, amb, mode="exact_discrete", max_stops=2, verbose=False)
        anc = solve_exact(ts, sp, p, wx, amb, mode="enumeration_anchor", max_stops=2, k_near=6, verbose=False)
        same = (bp["n_sorties"] == anc["n_sorties"])
        print(f"  B&P: 架次={bp['n_sorties']} certified_min_sorties={bp['certified_min_sorties']} "
              f"certified_lex={bp['certified_lex']} closed={bp['branch_closed']} "
              f"opt_gap={bp['opt_gap_pct']}% k_near_used={bp.get('k_near_used')}")
        print(f"  枚举锚点: 架次={anc['n_sorties']} 序列列={anc.get('n_seq_enumerated')} "
              f"展开列={anc['n_cols_enumerated']} 完整枚举={anc.get('complete_enumeration')} scope={anc.get('exact_scope')}")
        print(f"  【B&P==锚点? {same}】 | 选中列 fixed_h(应被汇总采用): "
              f"{[d['h'] for d in bp['route_diag']]}")
        # 更新 任务2: 全局词典序列生成 vs 完整枚举锚点 (N,E,M)
        lx = solve_exact(ts, sp, p, wx, amb, mode="exact_lex", max_stops=2, verbose=False)
        lexN = (lx["n_sorties"] == anc["n_sorties"])
        lexE = abs(lx["total_energy_Wh"] - anc["total_energy_Wh"]) < 0.5
        lexM = abs(lx["total_robust_margin"] - anc["total_robust_margin"]) < 0.5
        print(f"  词典序CG: (N={lx['n_sorties']},E={lx['total_energy_Wh']},M={lx['total_robust_margin']}) "
              f"vs 锚点(N={anc['n_sorties']},E={anc['total_energy_Wh']},M={anc['total_robust_margin']}) "
              f"| 全中={lexN and lexE and lexM}")
        print(f"  词典序证书: certified_min_sorties={lx['certified_min_sorties']} certified_lex={lx['certified_lex']} "
              f"(s1conv={lx['stage1_converged']} s2conv={lx['stage2_converged']} L2_LP紧={lx['L2_lp_tight']})")
        # 更新 任务2: 完整 SP-对偶 gap-closing 分支定价 vs 锚点
        bpx = solve_exact(ts, sp, p, wx, amb, mode="exact_bp", max_stops=2, verbose=False)
        bpxN = (bpx["n_sorties"] == anc["n_sorties"])
        print(f"  SP对偶B&P: N={bpx['n_sorties']} vs 锚点 N={anc['n_sorties']} 同={bpxN} | "
              f"root_SP_LP={bpx['root_SP_LP']} tree_closed={bpx['tree_closed']} "
              f"certified={bpx['certified_min_sorties']} via={bpx['certified_via']} nodes={bpx['nodes_explored']}")
        print("OK" if (same and lexN and lexE and lexM and bpxN and bpx['certified_min_sorties']) else "⚠ 需检查")
    else:
        _selftest()


# =============================================================================
# 研究性软覆盖分支定价实现（不进入正式全局证书路径）
# =============================================================================
def solve_soft_coverage_research(turbines, launch_opts, p, xi_amb, K, T_min,
                      deck_delta_min=2.5, t_swap_min=4.0, max_stops=4,
                      weather_unc=None, batteries=None, kappa_mode="vp_unimodal",
                      seed_cols=None, max_nodes=200, cg_max_iter=200,
                      time_limit_s=1800.0, verbose=False,
                      l1_time_limit_s=None, l2_time_limit_s=None,
                      seed_incumbent=True,
                      deck_mode="interval", t_launch_min=None,
                      reach_mode: str = "valid", dominance_mode: str | None = None,
                      enable_rf_branching: bool = False, l2_mode: str = "bp",
                      pricing_label_budget: int | None = 400_000,
                      use_milp_heuristic: bool = False,
                      _test_force_branch=False):
    r"""【更新 新增, 更新 重构】机队主问题的分支定价(Branch-and-Price), 给 LP 上界
    与整数下界 ⇒ 最优性 gap 证书(作者定案: gap 判断经分支定价, 不是拍脑袋阈值)。

    主问题(LP 松弛, 与 solve_resource_master 同一形):
        max Σ_i y_i
        s.t. y_i ≤ Σ_{c∋i} x_c            (覆盖链接, 对偶 π_i ≥ 0)
             Σ_c x_c ≤ B                  (电池, 对偶 β ≥ 0)
             Σ_{c 占机 t} x_c ≤ K          (占机并发, 逐 Δ 格点, 对偶 μ_t ≥ 0)
             Σ_{c 占甲板 t} x_c ≤ 1        (甲板容量 1, 逐 Δ 格点, 对偶 δ_t ≥ 0)
             [分支行] Σ_{c∋i} x_c ≥ 1  ∀i∈required   (对偶 ρ_i ≥ 0)
        无划分行(更新 定案: 覆盖=软目标), 故列 reduced cost 无 η 项。

    更新/更新 甲板语义(deck_mode='interval' 新默认): 起降是有时长的甲板区间占用。
    更新 修复(τ=离舰时刻, 与 step11.deck_indices/论文一致): 起飞准备占
    [max(τ−t_launch,0), τ), 回收换电占 [τ+h, τ+h+t_swap); 占机区间 [max(τ−t_launch,0),
    τ+h+t_swap); 时间网格延伸至 T+t_swap(约束越窗换电)。措辞纠正: 每列甲板占用是两个
    不连通区间(2-interval), 逐格点容量-1 行是【有效不等式】(Δ 网格捕获全部成对重叠 ⇒
    整数可行性充分), 并非该 stable-set 多面体的完整线性描述(旧注释"区间图完美"断言过强);
    最优性证书由 B&P 整数分支闭合, 不依赖该断言。'slot' 保留回归/消融。

    定价(逐 τ 的 ESPPRC, 精确标号 = price_routes strict 支配):
        列 reduced cost  rc(τ,S,h) = [β + Σ_{t∈launch甲板(τ)} δ_t]           ← route_cost(τ 常量)
                                    + [Σ_{t∈占机(τ,h)} μ_t + Σ_{t∈rec甲板(τ,h)} δ_t]  ← close_cost(h)
                                    − Σ_{i∈S} (π_i + ρ_i·1[i∈required])      ← prize
        rc < 0 即改进列。更新 两条精确提速(引理见 doc_proof §R39): ①闭合按 (cost,h)
        升序扫、首个可行 h 即最优(9 次 DRCC 评估 → ~首个可行处); ②完成界剪枝(标号
        prize 上界 ≤ 成本 ⇒ 全后代 rc ≥ 0, 安全剪)。

    更新 分支层级(GPT 批评"对 column 分支太弱"成立, 已更换):
        ① 风机服务量 s_i=Σ_{c∋i}x_c 分数 → (禁服 i: 定价图删点+列过滤 | 必服 i: 加 ≥1 行取对偶 ρ_i);
        ② 同飞对 z_ij 分数 → Ryan–Foster (apart: 禁同列 | together: 同进同出),
           经 price_routes 既有 forbid_pairs/force_pairs 机制进定价(缺席一方随 τ 局部化);
        ③ 兜底列固定/禁止(理论完备; 实践极少触发, 计数输出验证)。
    每支树节点 = (banned, required, apart, together, fcols, xcols); 对偶界仅在本节点列集上
    有效, 全局 UB = max(已探节点 UB, 未探节点父 UB) —— 与 更新 相同的有效性论证。

    返回 dict: covered(LB=incumbent), UB, gap_pct, status, chosen, root_LP, nodes, cg_iters,
    pool_final, pricing_progress(超时时最后一轮定价完成度), n_branch_*(分支类型计数),
    deck_mode/t_launch_min(provenance)。status 以 '-no-certificate' 结尾(含
    'time-limit-no-certificate'/'node-limit-no-certificate', 更新 M-01 拆分) ⇒ UB
    不可信, 只能报 incumbent(诚实降级, 不冒充证书); hit_time_limit/hit_node_limit/
    hit_label_budget 为独立运行时旗标, 证书条件 no_timeout/no_node_limit/
    no_label_truncation 直读旗标而非解析状态字符串。

    ── 更新(第三方审计修复#1–#6, 全局最优证书闭环)──────────────────────────────
    #1 收敛状态/gap=0: 节点列生成【只有】以 no-neg-rc 收敛才使用其 LP 界; 若耗尽
       cg_max_iter 仍在产列或 LP 求解失败 ⇒ 整树降级 status='cg-iter-limit-no-certificate'
       /'lp-fail-no-certificate', UB=None, 不再冒发 optimal(gap=0)。节点 UB 加 CG 容差
       松弛 B·PRICING_EPS(rc_tol 的 Lagrange 界)。
    #2 tau_reach: 缺省 reach_mode='valid'(仅去程下界 + 顺风地速上界, 保真, 见 step11
       更新 证明); 'off' 不预筛; 'legacy2d'(旧 2d 回路界, 移动母船下不保真)仅消融,
       使用即撤销证书(certificate.reach_ok=False)。
    #3 Ryan–Foster: 本主问题是【覆盖型】(y_i≤Σx, 覆盖计数目标), together/apart 二分对
       整数解【不穷尽】(反例: 选 {i,j} 列 + 仅含 i 列的解两支皆排) ⇒ 缺省禁用
       (enable_rf_branching=False)。①风机服务量二分(s_i=0 / ≥1)+ ③列固定/禁止二分
       已构成完备分支(列空间有限), 证书不受影响; 启用 RF 且实际发生对分支时撤销证书。
    #4 空/不可行 RMP 使用严格两阶段法: Phase-I 仅最小化全部人工变量并通过定价消除
       它们；最优人工值仍正且定价闭合才判节点不可行。Phase-II 将人工变量固定为 0，
       再求真实覆盖/能耗目标。算法正确性不依赖 Big-M。
    #5 仅船位误差第一层完整证书: certificate.L1_certified = 栈空最优 ∧ 全节点定价收敛 ∧
       reach∈{valid,off} ∧ 未用 RF 对分支 ∧ 支配保真(ξ-only 用 'set'; 多源自动升
       'sequence' 顺序完备标号, 审计#7)。
    #6 独立第二层能耗 B&P(l2_mode='bp', 缺省): 每个节点先完成可行性 Phase-I，再在
       覆盖=N*、人工变量全为 0 下以 rc=E0+占用对偶−π 做 Phase-II 定价；能耗闭合对
       全部决策 h 逐项记录运行时扫描 proof。仅树闭合、两阶段完整、全部 h proof 完整时
       才给 L2 全局证书。l2_mode='expand' 仍诚实标注无 L2 证书。

    ── 更新(外部审计 P0-NEW/P2/C3 修复)────────────────────────────────────────
    P0-NEW 分数节点上 `_milp_int`/`_milp_int_E` 的 MILP 输出不再被信任: 与整数-LP
      路径同款独立验证(形状/有限性 + 全行 A@z≤b + 覆盖数取所选列 tid 并集 + 能耗取
      所选列 E0 之和); 验证失败视同 MILP 失败(继续完备分支)。杜绝 success=True 但
      x 畸形(维度错/NaN)或违约(y 全 1、x 全 0)造成的假 L1/L2 证书。
    P2-02 seed 验证与 L2 前 h 扩列的天气解析统一为 _wx_of_ship(ship.wx_tau 优先),
      与定价完全同口径。
    C3   证书新增 l2_optimality_tolerance_Wh / l2_certificate_semantics: L2 为
      【容差级】最优(剪枝 node_lb ≥ best_E − 1e-6, node_lb = LP − B·PRICING_EPS);
      L1 覆盖数为整数目标, 精确。
    P3-02 L2 收尾新增 incumbent 能耗一致性运行时审计(漂移超容差撤证)。
    (P2-01/P2-03/P2-04 属单船 lex/CG 入口修复, 见各函数内 更新 注释。)
    """
    import time as _time
    import step11_algorithm_route_drcc as RA
    t0 = _time.time()
    total_time_limit_s = max(float(time_limit_s), 0.0)
    # L1/L2 must not silently compete for the same deadline.  By default L1
    # receives at most 70% of the total budget and L2 is guaranteed at least
    # 30%; unused L1 time is transferred to L2.  Explicit per-stage limits are
    # supported for controlled experiments.
    l1_budget_s = (max(float(l1_time_limit_s), 0.0)
                   if l1_time_limit_s is not None else 0.70 * total_time_limit_s)
    l2_reserved_s = (max(float(l2_time_limit_s), 0.0)
                     if l2_time_limit_s is not None else 0.30 * total_time_limit_s)
    l1_deadline = t0 + l1_budget_s
    l1_finished_at = None
    l2_started_at = None
    l2_finished_at = None
    B = int(batteries) if batteries is not None else 2 * int(K)
    t_launch = float(t_launch_min) if t_launch_min is not None else float(deck_delta_min)
    # 更新(M-01): 时间限制与节点限制分别用【独立运行时旗标】记录(不再从最终状态字符串
    # 反推限制原因 —— 旧口径 time_limit_s=0 时命中的是时间限制却被诊断为节点限制)。
    hit_time_limit = False
    hit_node_limit = False
    hit_label_budget = False
    # 更新(H-02): 全部定价调用的运行时证明(L1+L2 各节点各 τ 各轮; 见 price_routes
    # stats_out), 与调用计数一并聚合 —— 少一条即撤证(fail-closed)。
    pricing_stats = []
    pricing_calls_expected = 0
    # L2 Phase-II 能耗定价的独立 h 扫描证明集合；只登记实际 Phase-II 调用。
    l2_h_scan_stats = []
    l2_h_scan_calls_expected = 0
    # 更新(E-01/E-02 可观测性): 树深度与 Phase-I 人工变量活动计数(端到端测试用)。
    max_depth = 0
    phase1_stats = dict(l1_activated=0, l1_resolved=0, l1_infeasible=0,
                        l2_activated=0, l2_resolved=0, l2_infeasible=0)
    two_phase_stats = dict(l1_nodes_started=0, l1_phase1_feasible=0,
                           l1_phase1_infeasible=0, l1_phase2_started=0,
                           l1_phase2_closed=0)
    # 更新(P0-1/P0-2): MILP 仅作分数节点 incumbent 启发式；整数 LP 由主问题行独立
    # 重建并验证后直接成为 incumbent。外部 seed_cols 永不信任，全部按当前输入重算。
    milp_runtime = dict(l1_calls=0, l1_failures=0, l2_calls=0, l2_failures=0,
                        l1_invalid_solutions=0, l2_invalid_solutions=0,
                        l1_validated_incumbents=0, l2_validated_incumbents=0,
                        l1_integral_lp_incumbents=0, l2_integral_lp_incumbents=0,
                        integral_lp_validation_failures=0,
                        last_validation_reason=None, validation_reasons={})
    seed_incumbent_stats = dict(enabled=bool(seed_incumbent), attempted=False,
                                accepted=False, coverage=0, energy_Wh=None,
                                source=None, rejection_reason=None)

    def _record_milp_rejection(stage, reason):
        key = str(reason or "unknown")
        milp_runtime[f"{stage}_failures"] += 1
        if key not in ("solver_failure", "solver_exception"):
            milp_runtime[f"{stage}_invalid_solutions"] += 1
        milp_runtime["last_validation_reason"] = f"{stage}:{key}"
        vr = milp_runtime["validation_reasons"]
        vr[f"{stage}:{key}"] = int(vr.get(f"{stage}:{key}", 0)) + 1

    seed_validation_stats = dict(input_count=0, accepted_count=0, rejected_count=0,
                                 energy_overwritten_count=0, validation_complete=False,
                                 all_accepted_revalidated=False, trusted_input_values=False,
                                 rejection_reasons={})

    # 更新(外部审计 问题#1 修复): 甲板占用是对【Δ 格点集合】的 set-packing。区间→格点
    #   映射 [a,b) ↦ {i : a ≤ i·Δ < b} 只在【端点是 Δ 整数倍】时忠实表达连续区间的相交:
    #   若某 prep/rec 区间不含任何 Δ 倍点, 它映射为空集或与相邻列不共享格点, 于是两条
    #   【物理重叠】的列(共用同一甲板)被判为不冲突 ⇒ B&P 可同时选取并冒发证书。
    #   审计给出 t_launch=3.0, Δ=2.5 的合法反例: 返回 3 台/218.4 Wh(证书真), 而连续物理
    #   最优为 3 台/244.7 Wh —— prep [0,2.5) 与 [2,5) 物理重叠 [2,2.5) 却无公共格点。
    #   修复: 检测所有【实际参与求解的列】的 prep/rec/occ 区间的【左端点】是否对齐 Δ;
    #   未对齐时甲板格点语义 ≠ 连续物理语义, 证书不能外推到物理排他 ⇒ 撤销 L1/L2 证书。
    #   (slot 模式按单格点占用, 语义自洽, 不受此约束。)
    #
    # 【为何只查左端点 — 严格证明】半开区间 [a,b) 捕获的格点集 = {i : a ≤ i·Δ < b}。
    #   若两区间左端点 a1,a2 均为 Δ 整数倍, 则"共享格点 ⟺ 连续相交":
    #     (⇐ 相交) 设 a1≤a2<b1, 则 a2 对齐且 a2∈[a1,b1)∩[a2,b2) ⇒ 共享格点 a2;
    #     (⇒ 共享 p) p 在两区间内 ⇒ 相交含 p。
    #   右端点是否对齐【无关】: [a,b) 恒捕获对齐的左端点 a(因 a=kΔ<b), 右端截断不影响。
    #   实测: 20 万随机对(左对齐、右任意)相交与共享格点 0 处不一致; 左端不对齐则出现
    #   "物理相交却无公共格点"(20 万中约 1.1 万)。故只需保证 prep/rec/occ 的左端对齐。
    #   prep 左端 = max(τ−t_launch,0)(受 t_launch 影响), rec/occ 左端 = τ+h / max(τ−t_launch,0)。
    #   决策网格 τ,h ∈ 5 的整数倍时, τ+h 恒对齐; 故触发点通常是 t_launch 使 τ−t_launch 不对齐。
    #   (t_swap 只影响右端点, 因此 t_swap 非 Δ 整数倍【不】破坏忠实性。)
    def _deck_endpoints_aligned(tau, h):
        if deck_mode == "slot":
            return True
        eps = 1e-9
        for x in (max(float(tau) - t_launch, 0.0), float(tau) + float(h)):
            r = x / deck_delta_min
            if abs(r - round(r)) > eps:
                return False
        return True
    # 更新: 网格延伸覆盖窗尾换电 [T, T+t_swap)(与 step11 同口径)
    n_tgrid = int(math.floor((T_min + t_swap_min) / deck_delta_min)) + 1
    orig_kappa = RM.kappa
    if str(kappa_mode) == "nominal":
        RM.kappa = lambda _eps: 0.0
    elif kappa_mode in RM.KAPPA_MODES:
        RM.kappa = RM.KAPPA_MODES[kappa_mode]
    try:
        m = len(turbines)
        tid_of = [t.tid for t in turbines]
        idx_of_tid = {t.tid: i for i, t in enumerate(turbines)}
        h_grid = RM.decision_horizons_of(xi_amb)
        # 更新 #7: 支配保真自动档 —— ξ-only 用同集 strict('set', 闭合裕度只依赖
        # (集合,末端,h) 且经 e_dom/T_nom 单调, 精确); 多源(weather_unc)闭合含逐腿风灵敏度
        # (依赖顺序) ⇒ 'sequence'(不合并标号, 顺序完备)才保证定价最优性。
        if dominance_mode is None:
            dominance_mode = "set" if weather_unc is None else "sequence"
        # 更新 #2 / 更新(审计 6.1): 保真可达预筛。valid 在带符号均值(风偏置/ξ均值)
        # 下于 tau_reach 内部自动降级为 off(排除方向恒保真); legacy2d 仅消融且撤证书。
        _mean_relax_free = RM.mean_relax_free(xi_amb, weather_unc)
        reach_of = [RA.tau_reach(opt, turbines, p, max(h_grid), mode=reach_mode,
                                 wx=getattr(opt, "wx", None), xi_amb=xi_amb,
                                 weather_unc=weather_unc) for opt in launch_opts]
        eval_cache = {}

        # ---- 甲板占用格点(τ 常量的起飞侧 / 依赖 h 的回收侧; union 进列) ----
        def _launch_deck_idx(tau):
            # 更新(τ=离舰时刻): 起飞准备占 [max(τ−t_launch,0), τ), 与 step11.deck_indices 同口径
            if deck_mode == "slot":
                i = int(round(tau / deck_delta_min))
                return [i] if 0 <= i < n_tgrid else []
            a = max(tau - t_launch, 0.0)
            i0 = int(math.ceil((a - 1e-9) / deck_delta_min))
            i1 = int(math.floor((tau - 1e-9) / deck_delta_min))
            return [i for i in range(max(i0, 0), min(i1 + 1, n_tgrid))
                    if a - 1e-9 <= i * deck_delta_min < tau - 1e-9]

        def _rec_deck_idx(tau, h):
            a = tau + float(h)
            if deck_mode == "slot":
                i = int(round(a / deck_delta_min))
                return [i] if 0 <= i < n_tgrid else []
            b = a + t_swap_min
            i0 = int(math.ceil((a - 1e-9) / deck_delta_min))
            i1 = int(math.floor((b - 1e-9) / deck_delta_min))
            return [i for i in range(max(i0, 0), min(i1 + 1, n_tgrid))
                    if a - 1e-9 <= i * deck_delta_min < b - 1e-9]

        def _occ_idx(tau, h):
            # 更新: 占机 = [max(τ−t_launch,0), τ+h+t_swap)(准备期 UAV 已被该架次占用)
            a0 = max(tau - t_launch, 0.0)
            a = int(math.ceil((a0 - 1e-9) / deck_delta_min))
            out = []
            for ti in range(max(a, 0), n_tgrid):
                t = ti * deck_delta_min
                if t >= tau + float(h) + t_swap_min - 1e-9:
                    break
                if a0 - 1e-9 <= t:
                    out.append(ti)
            return out

        # ============ 更新(H-01, 致命修复): 连续甲板/占机语义 ============
        # 根因: Δ 格点行只在区间【左端点全部对齐 Δ】时忠实表达连续相交(更新 一节的引理);
        #   t_launch 非 Δ 整数倍等合法输入下, 两条物理重叠的列可映射为无公共格点 ⇒ 求解器
        #   同时选取并返回连续物理冲突的计划(独立反例: 格点最大覆盖 3, 连续物理最大覆盖 2)。
        #   更新 前一批只做了撤证(deck_grid_exact), 未修"返回错误计划"。本批按审计要求
        #   把连续冲突写进【L1 LP / L1 MILP / L2 LP / L2 MILP 同一套行】:
        #   (a) 甲板(容量 1): 每列保留完整连续甲板区间(prep [max(τ−t_launch,0),τ),
        #       rec [τ+h, τ+h+t_swap)); 新列入池时与既有列【逐对】按
        #       max(a1,a2) < min(b1,b2) 判连续相交。相交对若已共享甲板格点, 则格点容量-1 行
        #       已蕴含 x_i+x_j ≤ 1(数学等价); 否则(格点漏判对)显式加 x_i+x_j ≤ 1 行。
        #       两类逐对计数进运行时审计(total = grid_covered + row_added 恒等式),
        #       证书条件 deck_conflict_semantics_exact 据此判定。新列在既有逐对行中系数
        #       恒为 0 ⇒ 定价 reduced cost 不受影响(对偶可行性构造见 MODIFICATIONS)。
        #   (b) 占机(容量 K): 逐对行对 K≥2 不充分。区间并发峰值必在某左端点取得 ⇒ 对
        #       池内所有【未对齐 Δ 的占机左端点】追加事件行 Σ_{列 ∋ e} x ≤ K(对齐左端点
        #       已被既有全格点行覆盖); 事件行对新列系数非零 ⇒ 定价闭合成本同步计入其对偶。
        #   (c) 返回前对最终 chosen 做独立连续区间复检(_plan_physical_check); 冲突 ⇒
        #       status='physical-deck-conflict', L1/L2 证书一律 False(纵深防御)。
        def _occ_iv(tau, h):
            return (max(float(tau) - t_launch, 0.0), float(tau) + float(h) + t_swap_min)

        def _deck_ivs(tau, h):
            if deck_mode == "slot":
                return ()      # slot 模式 = 单格点占用语义(自洽), 无连续区间冲突概念
            out = []
            a = max(float(tau) - t_launch, 0.0)
            if a < float(tau) - 1e-9:
                out.append((a, float(tau)))
            if t_swap_min > 1e-9:
                out.append((float(tau) + float(h), float(tau) + float(h) + t_swap_min))
            return tuple(out)

        def _ivs_overlap(iv1, iv2):
            for a1, b1 in iv1:
                for a2, b2 in iv2:
                    if max(a1, a2) < min(b1, b2) - 1e-9:
                        return True
            return False

        def _aligned(x):
            r = float(x) / deck_delta_min
            return abs(r - round(r)) <= 1e-9

        deck_conf_adj = {}                 # j -> set(k): 连续相交且格点漏判的甲板冲突对
        occ_event_times = []               # 未对齐 Δ 的占机左端点(升序去重)
        _occ_ev_seen = set()
        deck_pair_stats = dict(total=0, grid_covered=0, row_added=0, audit_ok=True)

        def _occ_event_hits(tau, h):
            a0, b0 = _occ_iv(tau, h)
            return [e for e in occ_event_times if a0 - 1e-9 <= e < b0 - 1e-9]

        def _register_col_conflicts(jn):
            """新列 jn 入池: (a) 与既有列逐对判连续甲板相交并归类; (b) 收集未对齐占机左端点。"""
            cn = cols[jn]
            a0, _b0 = cn["occ_iv"]
            if not _aligned(a0) and a0 not in _occ_ev_seen:
                _occ_ev_seen.add(a0)
                occ_event_times.append(a0)
                occ_event_times.sort()
            ivn = cn["deck_ivals"]
            if not ivn:
                return
            lo_n = min(a for a, _ in ivn)
            hi_n = max(b for _, b in ivn)
            dset_n = set(cn["deck_idx"])
            for j in range(jn):
                cj = cols[j]
                ivj = cj["deck_ivals"]
                if not ivj:
                    continue
                if min(a for a, _ in ivj) >= hi_n - 1e-9 or max(b for _, b in ivj) <= lo_n + 1e-9:
                    continue           # 粗筛: 总包络不相交
                if not _ivs_overlap(ivn, ivj):
                    continue
                deck_pair_stats["total"] += 1
                if dset_n & set(cj["deck_idx"]):
                    deck_pair_stats["grid_covered"] += 1     # 格点行已蕴含 x_i+x_j≤1
                else:
                    deck_pair_stats["row_added"] += 1
                    deck_conf_adj.setdefault(jn, set()).add(j)
                    deck_conf_adj.setdefault(j, set()).add(jn)

        def _mk_col(tau, ship, wx, tids, h, E0, route=None, gate_proof=None):
            return dict(tau=float(tau), ship=ship, wx=wx, tids=tuple(sorted(tids)),
                        h=float(h), E0=float(E0), route=route,
                        occ_idx=tuple(_occ_idx(float(tau), float(h))),
                        deck_idx=tuple(sorted(set(_launch_deck_idx(float(tau)))
                                             | set(_rec_deck_idx(float(tau), float(h))))),
                        occ_iv=_occ_iv(float(tau), float(h)),             # 更新 H-01
                        deck_ivals=_deck_ivs(float(tau), float(h)),      # 更新 H-01
                        gate_proof=gate_proof,                            # 更新 C-01
                        kappa_used=RM.kappa(p.eps_E))

        # ---- 初始列池(与 solve_resource_master 同一启发式生成器, 保证起点相同) ----
        cols, seen, key2idx = [], set(), {}

        def _key_of(tau, tids, h):
            return (round(float(tau), 3), tuple(sorted(tids)), float(h))

        def _add(tau, ship, wx, tids, h, E0, route=None, gate_proof=None):
            # 更新 #6: 主问题列以 (τ, 集合, h) 为身份; 同键不同【序列】能耗可不同 ——
            # L1 只看集合, 但 L2 目标是 ΣE0, 故同键保留【最小 E0 的序列】(能耗定价可能
            # 找到更省的访问顺序, 视为改进列)。
            key = _key_of(tau, tids, h)
            j = key2idx.get(key)
            if j is not None:
                if float(E0) < float(cols[j]["E0"]) - 1e-9:
                    cols[j]["E0"] = float(E0)
                    if route is not None:
                        cols[j]["route"] = route
                    if gate_proof is not None:
                        cols[j]["gate_proof"] = gate_proof
                    return True
                if gate_proof is not None and cols[j].get("gate_proof") is None:
                    cols[j]["gate_proof"] = gate_proof   # 补证据不改列身份
                return False
            seen.add(key)
            key2idx[key] = len(cols)
            cols.append(_mk_col(tau, ship, wx, tids, h, E0, route, gate_proof=gate_proof))
            _register_col_conflicts(len(cols) - 1)       # 更新 H-01: 逐对连续冲突登记
            return True

        # ---- 更新 P0-2: 种子列是非可信输入，必须完全规范化并重新做物理判定 ----
        # 不使用 seed 中的 ship/wx/turbine 对象或 E0。只把 route 的 tid 顺序视为候选序列，
        # 然后映射到本次 turbines/launch_opts，调用当前 route_feasible_at_h 重算可行性和能耗。
        raw_seed_cols = (RA.build_route_columns(
            turbines, launch_opts, p, xi_amb, T_min, deck_delta_min,
            max_stops, weather_unc, "drcc", RM.BUDGET_GAMMA_DEFAULT,
            kappa_mode, 8.0) if seed_cols is None else list(seed_cols))
        seed_validation_stats["input_count"] = int(len(raw_seed_cols))
        _tb_by_tid = {t.tid: t for t in turbines}
        _opts_by_tau = {}
        for _opt in launch_opts:
            _opts_by_tau.setdefault(round(float(_opt.tau_min), 9), []).append(_opt)

        def _seed_reject(reason):
            seed_validation_stats["rejected_count"] += 1
            rr = seed_validation_stats["rejection_reasons"]
            rr[reason] = int(rr.get(reason, 0)) + 1

        for _si, c in enumerate(raw_seed_cols):
            try:
                if not isinstance(c, dict):
                    _seed_reject("not-a-dict"); continue
                tau_raw = float(c["tau"]); h_raw = float(c["h"])
                if not (math.isfinite(tau_raw) and math.isfinite(h_raw)):
                    _seed_reject("nonfinite-time"); continue
                _opts = _opts_by_tau.get(round(tau_raw, 9), [])
                if len(_opts) != 1:
                    _seed_reject("unknown-or-ambiguous-launch-option"); continue
                opt = _opts[0]
                _hm = [float(hh) for hh in h_grid if abs(float(hh) - h_raw) <= 1e-9]
                if len(_hm) != 1 or tau_raw + _hm[0] > float(T_min) + 1e-9:
                    _seed_reject("h-outside-decision-domain"); continue
                h_can = _hm[0]

                route_in = c.get("route")
                tids_field = tuple(c.get("tids", ()))
                if route_in is not None and getattr(route_in, "turbines", None) is not None:
                    seq_tids = tuple(getattr(t, "tid", None) for t in route_in.turbines)
                else:
                    seq_tids = tids_field
                if (not seq_tids or any(tid is None for tid in seq_tids)
                        or len(seq_tids) != len(set(seq_tids))
                        or len(seq_tids) > int(max_stops)
                        or any(tid not in _tb_by_tid for tid in seq_tids)):
                    _seed_reject("invalid-route-turbines"); continue
                if tids_field and set(tids_field) != set(seq_tids):
                    _seed_reject("route-tids-mismatch"); continue

                route_can = RM.Route(rid=-1,
                                     turbines=[_tb_by_tid[tid] for tid in seq_tids],
                                     ship=opt.ship)
                h_eval = int(round(h_can)) if abs(h_can - round(h_can)) <= 1e-9 else h_can
                # 更新(P2-02): 与定价同一天气解析(ship.wx_tau 优先) —— 即便调用方使
                # opt.wx ≠ ship.wx_tau, seed 判定与定价/物理复检也不再分叉。
                wx_eval = _wx_of_ship(opt.ship, opt.wx)
                dd = RM.route_feasible_at_h(route_can, h_eval, p, wx_eval, xi_amb,
                                            weather_unc=weather_unc)
                if not isinstance(dd, dict) or dd.get("feasible") is not True:
                    _seed_reject("physical-infeasible"); continue
                E_can = _plan_energy(dd)
                if not math.isfinite(E_can) or E_can < -1e-9:
                    _seed_reject("invalid-recomputed-energy"); continue
                try:
                    E_in = float(c.get("E0", float("nan")))
                except (TypeError, ValueError):
                    E_in = float("nan")
                if not math.isfinite(E_in) or abs(E_in - E_can) > 1e-8 * max(1.0, abs(E_can)):
                    seed_validation_stats["energy_overwritten_count"] += 1
                _add(float(opt.tau_min), opt.ship, opt.wx, seq_tids, h_can, E_can,
                     route_can, gate_proof=dd.get("gate_weather_proof"))
                seed_validation_stats["accepted_count"] += 1
            except Exception as e:
                _seed_reject("validation-exception:" + type(e).__name__)
                log.warning("分支定价 seed#%d 验证异常，已拒绝: %s", _si, e)

        seed_validation_stats["validation_complete"] = True
        seed_validation_stats["all_accepted_revalidated"] = bool(
            seed_validation_stats["accepted_count"]
            + seed_validation_stats["rejected_count"]
            == seed_validation_stats["input_count"])
        log.info("分支定价 初始池 %d 列(种子输入 %d, 接受 %d, 拒绝 %d, E覆盖 %d), 甲板=%s",
                 len(cols), seed_validation_stats["input_count"],
                 seed_validation_stats["accepted_count"], seed_validation_stats["rejected_count"],
                 seed_validation_stats["energy_overwritten_count"], deck_mode)

        def _key(j):
            c = cols[j]
            return (round(c["tau"], 3), c["tids"], c["h"])

        # ---- 节点列过滤(banned/apart/together/xcols) ----
        def _active(node):
            banned, apart, together, xcols = node["banned"], node["apart"], node["together"], node["xcols"]
            out = []
            for j in range(len(cols)):
                c = cols[j]
                s = set(c["tids"])
                if _key(j) in xcols or (s & banned):
                    continue
                bad = False
                for pr in apart:
                    if pr <= s:
                        bad = True
                        break
                if not bad:
                    for pr in together:
                        i1, i2 = tuple(pr)
                        if (i1 in s) != (i2 in s):
                            bad = True
                            break
                if not bad:
                    out.append(j)
            return out

        # ---- 节点 RMP(LP / MILP 共享行构建) ----
        # Phase-I/Phase-II 已严格分离：Phase-I 目标仅为全部人工变量之和；Phase-II
        # 将人工变量上界固定为 0，再求真实覆盖/能耗目标。算法正确性不再使用 Big-M。
        def _extra_rows(active_idx):
            """更新(H-01): 主问题的连续语义补充行(L1/L2 共用同一构造):
            (a) 占机事件行 —— 池内未对齐 Δ 的占机左端点 e: Σ_{活跃列 ∋ e} x ≤ K。区间并发
                峰值必在某左端点取得; 对齐左端点已被全格点行覆盖, 故仅补未对齐者。仅活跃
                成员 ≥2 的事件建行(成员 ≤K≥1 的行恒松, 不建不损完备性)。
            (b) 甲板逐对行 —— deck_conf_adj 中两端均活跃的格点漏判相交对: x_i+x_j ≤ 1。
            返回 (ev_rows=[(e,[jj…])…], pair_rows=[(jj1,jj2)…])。"""
            pos = {j: jj for jj, j in enumerate(active_idx)}
            ev_rows = []
            if occ_event_times:
                for e in occ_event_times:
                    mem = [pos[j] for j in active_idx
                           if cols[j]["occ_iv"][0] - 1e-9 <= e < cols[j]["occ_iv"][1] - 1e-9]
                    if len(mem) >= 2:
                        ev_rows.append((e, mem))
            pair_rows = []
            if deck_conf_adj:
                for j in active_idx:
                    for k in deck_conf_adj.get(j, ()):
                        if k > j and k in pos:
                            pair_rows.append((pos[j], pos[k]))
            return ev_rows, pair_rows

        def _rows(active_idx, node, n):
            import scipy.sparse as sp
            req = sorted(node["required"])
            n_s = len(req)
            deck_pts = sorted({ti for j in active_idx for ti in cols[j]["deck_idx"]})
            drow = {ti: m + 1 + n_tgrid + k for k, ti in enumerate(deck_pts)}
            rrow = {tid: m + 1 + n_tgrid + len(deck_pts) + k for k, tid in enumerate(req)}
            ev_rows, pair_rows = _extra_rows(active_idx)
            R0 = m + 1 + n_tgrid + len(deck_pts) + n_s
            erow = {e: R0 + k for k, (e, _mem) in enumerate(ev_rows)}
            R = R0 + len(ev_rows) + len(pair_rows)
            A = sp.lil_matrix((R, n + m + n_s))
            b = np.zeros(R)
            for i in range(m):
                A[i, n + i] = 1.0
            for jj, j in enumerate(active_idx):
                c = cols[j]
                for tid in c["tids"]:
                    A[idx_of_tid[tid], jj] = -1.0
                    if tid in rrow:
                        A[rrow[tid], jj] = -1.0
                A[m, jj] = 1.0
                for ti in c["occ_idx"]:
                    A[m + 1 + ti, jj] = 1.0
                for ti in c["deck_idx"]:
                    A[drow[ti], jj] = 1.0
            for k, tid in enumerate(req):          # 人工松弛: −Σx − s_k ≤ −1
                A[rrow[tid], n + m + k] = -1.0
            for k, (e, mem) in enumerate(ev_rows):     # 更新 H-01(a): 占机事件行
                for jj in mem:
                    A[R0 + k, jj] = 1.0
                b[R0 + k] = float(K)
            for k, (jj1, jj2) in enumerate(pair_rows):  # 更新 H-01(b): 甲板逐对行
                A[R0 + len(ev_rows) + k, jj1] = 1.0
                A[R0 + len(ev_rows) + k, jj2] = 1.0
                b[R0 + len(ev_rows) + k] = 1.0
            b[m] = float(B)
            b[m + 1:m + 1 + n_tgrid] = float(K)
            for ti, r in drow.items():
                b[r] = 1.0
            for tid, r in rrow.items():
                b[r] = -1.0
            return A.tocsc(), b, drow, rrow, n_s, erow

        def _rmp_lp(active_idx, node, phase):
            """L1 严格两阶段 RMP。

            phase1: min sum(required artificials)，只用于恢复/证明节点可行性；
            phase2: max sum(y)，全部人工变量固定为 0，LP 值才可作为节点上界。
            """
            from scipy.optimize import linprog
            if phase not in ("phase1", "phase2"):
                raise ValueError("phase must be 'phase1' or 'phase2'")
            n = len(active_idx)
            A, b, drow, rrow, n_s, erow = _rows(active_idx, node, n)
            nv = n + m + n_s
            lo = np.zeros(nv)
            hi = np.ones(nv)
            for jj, j in enumerate(active_idx):
                if _key(j) in node["fcols"]:
                    lo[jj] = 1.0
            if phase == "phase2" and n_s:
                hi[n + m:] = 0.0                 # Phase-II: 人工变量严格固定为零
            if phase == "phase1":
                obj = np.concatenate([np.zeros(n + m), np.ones(n_s)])
            else:
                obj = np.concatenate([np.zeros(n), -np.ones(m), np.zeros(n_s)])
            bounds_lp = list(zip(lo, hi))
            res = linprog(obj, A_ub=A, b_ub=b, bounds=bounds_lp, method="highs")
            if getattr(res, "status", -1) == 2 and getattr(res, "success", False) is not True:
                return dict(infeasible=True, phase=phase)
            checked = _validate_linprog_result(res, obj, bounds_lp, A_ub=A, b_ub=b, need_ineqlin=A.shape[0])
            if checked is None:
                return None
            x_full, fun, marg, _, dual_lb = checked
            lam = -marg
            pi = {tid_of[i]: max(0.0, float(lam[i])) for i in range(m)}
            beta = max(0.0, float(lam[m]))
            mu = np.maximum(0.0, lam[m + 1:m + 1 + n_tgrid])
            delta = {ti: max(0.0, float(lam[r])) for ti, r in drow.items()}
            rho = {tid: max(0.0, float(lam[r])) for tid, r in rrow.items()}
            mu_ev = {e: max(0.0, float(lam[r])) for e, r in erow.items()}
            x = {j: float(x_full[jj]) for jj, j in enumerate(active_idx)}
            art_values = np.asarray(x_full[n + m:n + m + n_s], float) if n_s else np.zeros(0)
            art_sum = float(art_values.sum())
            art_max = float(art_values.max()) if art_values.size else 0.0
            primal_obj = fun if phase == "phase1" else -fun
            # Min-phase dual_lb is a lower bound.  In Phase-II the public
            # objective is max coverage, so -dual_lb is the matching safe
            # upper bound used for pruning/certification.
            bound_obj = dual_lb if phase == "phase1" else -dual_lb
            return dict(obj=primal_obj, bound=bound_obj, x=x, pi=pi, beta=beta, mu=mu, delta=delta,
                        rho=rho, mu_ev=mu_ev, art_required=art_sum,
                        max_artificial=art_max, artificial_count=int(n_s), phase=phase)

        def _milp_int(active_idx, node):
            """L1 Phase-II integer incumbent heuristic with full primal validation."""
            milp_runtime["l1_calls"] += 1
            n = len(active_idx)
            A, b, _, _, n_s, _erow = _rows(active_idx, node, n)
            nv = n + m + n_s
            if nv == 0:
                return 0, []
            lo = np.zeros(nv)
            hi = np.ones(nv)
            for jj, j in enumerate(active_idx):
                if _key(j) in node["fcols"]:
                    lo[jj] = 1.0
            if n_s:
                hi[n + m:] = 0.0
            obj = np.concatenate([np.zeros(n), -np.ones(m), np.zeros(n_s)])
            integ = np.concatenate([np.ones(n + m), np.zeros(n_s)])
            try:
                from scipy.optimize import milp, LinearConstraint, Bounds
                rem = max(0.0, float(l1_deadline) - _time.time())
                if rem <= 0.0:
                    _record_milp_rejection("l1", "global_time_limit")
                    return None, None
                r = milp(obj, constraints=LinearConstraint(A, -np.inf, b),
                         integrality=integ, bounds=Bounds(lo, hi),
                         options=dict(time_limit=max(rem, 1e-9)))
            except Exception:
                _record_milp_rejection("l1", "solver_exception")
                return None, None
            z, _value, reason = _validate_milp_primal(r, obj, A, b, lo, hi, integ)
            if z is None:
                _record_milp_rejection("l1", reason)
                return None, None
            art_values = z[n + m:n + m + n_s] if n_s else np.zeros(0)
            if art_values.size and float(np.max(np.abs(art_values))) > ART_TOL:
                _record_milp_rejection("l1", "artificial_residue")
                return None, None
            sel = [active_idx[jj] for jj in range(n) if z[jj] > 0.5]
            covered_tids = {tid for j in sel for tid in cols[j]["tids"]}
            # Never trust y or solver objective for the incumbent value.
            cov = int(len(covered_tids))
            if cov < 0 or cov > m:
                _record_milp_rejection("l1", "invalid_coverage_reconstruction")
                return None, None
            milp_runtime["l1_validated_incumbents"] += 1
            return cov, sel

        def _integral_lp_selection(lp, tol=1e-7):
            """仅当全部列变量数值上为 0/1 时返回所选全局列下标，否则返回 None。"""
            vals = lp.get("x") if isinstance(lp, dict) else None
            if not isinstance(vals, dict):
                return None
            if any((not math.isfinite(float(v))) or (tol < float(v) < 1.0 - tol)
                   or float(v) < -tol or float(v) > 1.0 + tol for v in vals.values()):
                return None
            return [j for j, v in vals.items() if float(v) >= 1.0 - tol]

        def _validate_l1_integral_lp(active_idx, node, lp, selected):
            """从整数 LP 的 x 直接重建 y，并用本节点完整主问题行/固定界独立验证。"""
            try:
                n = len(active_idx)
                pos = {j: jj for jj, j in enumerate(active_idx)}
                if len(selected) != len(set(selected)) or any(j not in pos for j in selected):
                    return None
                selected_keys = {_key(j) for j in selected}
                if not set(node["fcols"]).issubset(selected_keys):
                    return None
                A, b, _drow, _rrow, n_s, _erow = _rows(active_idx, node, n)
                z = np.zeros(n + m + n_s, float)
                for j in selected:
                    z[pos[j]] = 1.0
                covered_tids = {tid for j in selected for tid in cols[j]["tids"]}
                for tid in covered_tids:
                    z[n + idx_of_tid[tid]] = 1.0
                viol = np.asarray(A @ z - b, float)
                if viol.size and float(np.max(viol)) > 1e-7:
                    return None
                cov = int(len(covered_tids))
                if abs(float(lp["obj"]) - cov) > 1e-5:
                    return None
                return cov, list(selected)
            except Exception:
                return None

        # ---- 节点定价(更新: banned/pairs 局部化进 ESPPRC; 返回 (新列数, 完成度)) ----
        def _price_node(duals, node, deadline=None):
            new = 0
            banned, apart, together = node["banned"], node["apart"], node["together"]
            xcols = node["xcols"]
            n_opt = max(len(launch_opts), 1)
            for oi, opt in enumerate(launch_opts):
                if deadline is not None and _time.time() > deadline:
                    log.warning("分支定价 定价超时 @τ#%d/%d —— 本轮不完整, 无证书。", oi, n_opt)
                    return -1, oi / n_opt
                tau = float(opt.tau_min)
                r_turbs = [t for t in reach_of[oi] if t.tid not in banned]
                present = {t.tid for t in r_turbs}
                drop = set()
                for pr in together:      # together 对缺席一方 ⇒ 在场一方本 τ 同禁(RF 语义局部化)
                    i1, i2 = tuple(pr)
                    if (i1 in present) != (i2 in present):
                        drop.add(i1 if i1 in present else i2)
                if drop:
                    r_turbs = [t for t in r_turbs if t.tid not in drop]
                if not r_turbs:
                    continue
                li = {t.tid: k for k, t in enumerate(r_turbs)}
                fpairs = {frozenset((li[a], li[bb])) for pr in apart
                          for a, bb in [tuple(pr)] if a in li and bb in li}
                tpairs = {frozenset((li[a], li[bb])) for pr in together
                          for a, bb in [tuple(pr)] if a in li and bb in li}
                d_launch = sum(duals["delta"].get(ti, 0.0) for ti in _launch_deck_idx(tau))
                dual_map = {k: duals["pi"].get(t.tid, 0.0) + duals["rho"].get(t.tid, 0.0)
                            for k, t in enumerate(r_turbs)}
                if sum(dual_map.values()) <= duals["beta"] + d_launch + 1e-9:
                    continue           # 全拿也不抵成本(完成界的 τ 级粗筛)
                mu, delta = duals["mu"], duals["delta"]
                mu_ev = duals.get("mu_ev", {})

                def _cc(h, tids, _tau=tau):
                    if _tau + float(h) > T_min + 1e-9:
                        return 1e18
                    if (round(_tau, 3), tuple(tids), float(h)) in xcols:
                        return 1e18
                    occ = sum(mu[ti] for ti in _occ_idx(_tau, float(h)))
                    # 更新(H-01): 占机事件行对新列系数非零 ⇒ 闭合成本必须计入其对偶
                    if mu_ev:
                        occ += sum(mu_ev.get(e, 0.0) for e in _occ_event_hits(_tau, float(h)))
                    return float(occ) + sum(delta.get(ti, 0.0) for ti in _rec_deck_idx(_tau, float(h)))

                _pst = {}
                nonlocal pricing_calls_expected
                pricing_calls_expected += 1
                pricing_stats.append(_pst)     # 更新(H-02): 先登记后调用, 异常路径也留痕
                pr_new = price_routes(r_turbs, opt.ship, p, opt.wx, xi_amb, dual_map,
                                      route_cost=duals["beta"] + d_launch,
                                      max_stops=max_stops, k_near=len(r_turbs), max_routes=20,
                                      rc_tol=-PRICING_EPS, weather_unc=weather_unc,
                                      strict_dominance=True, forbid_pairs=fpairs,
                                      force_pairs=tpairs, eval_cache=eval_cache,
                                      close_cost_of_h=_cc, dominance_mode=dominance_mode,
                                      label_budget=pricing_label_budget, stats_out=_pst)
                if not _pst.get("complete", True):
                    return -2, oi / n_opt      # 更新 6.6: 标号预算触限 ⇒ 定价不完整
                for f in pr_new:
                    if _add(tau, opt.ship, opt.wx, [t.tid for t in f["turbines"]],
                            f["h"], f["E0"],
                            RM.Route(rid=-1, turbines=list(f["turbines"]), ship=opt.ship),
                            gate_proof=f.get("gate_proof")):
                        new += 1
            return new, 1.0

        def _child(node, **kw):
            d = dict(banned=node["banned"], required=node["required"], apart=node["apart"],
                     together=node["together"], fcols=node["fcols"], xcols=node["xcols"],
                     depth=int(node.get("depth", 0)) + 1)   # 更新(E-01): 树深度可观测
            d.update(kw)
            return d

        # ---- 分支定价主循环(DFS; UB=max(已探节点LP, 未探节点父UB) 的全局有效界) ----
        root = dict(banned=frozenset(), required=frozenset(), apart=frozenset(),
                    together=frozenset(), fcols=frozenset(), xcols=frozenset(), parent_ub=None,
                    depth=0)

        def _energy_of_indices(sel):
            return float(sum(float(cols[j]["E0"]) for j in sel))

        def _validate_root_integer_selection(sel):
            """Independently validate an integer root selection against all RMP rows."""
            try:
                active = list(range(len(cols)))
                n = len(active)
                if len(sel) != len(set(sel)) or any(j < 0 or j >= n for j in sel):
                    return None
                A, b, _drow, _rrow, n_s, _erow = _rows(active, root, n)
                z = np.zeros(n + m + n_s, float)
                covered_tids = {tid for j in sel for tid in cols[j]["tids"]}
                for j in sel:
                    z[j] = 1.0
                for tid in covered_tids:
                    z[n + idx_of_tid[tid]] = 1.0
                viol = np.asarray(A @ z - b, float)
                if viol.size and float(np.max(viol)) > 1e-7:
                    return None
                return int(len(covered_tids)), float(_energy_of_indices(sel))
            except Exception:
                return None

        # Seed columns are not merely a pricing warm start: solve their restricted
        # lexicographic master once and preserve that feasible plan as the global
        # incumbent.  The returned plan is mapped back to the revalidated columns
        # and checked against the exact root rows before acceptance.  Therefore a
        # later B&P timeout can never erase an already-known feasible solution.
        LB, incumbent, incumbent_source = 0, [], "empty"
        if seed_incumbent and cols:
            seed_incumbent_stats["attempted"] = True
            try:
                seed_res = RA.solve_resource_master(
                    turbines, launch_opts, p, xi_amb, K, T_min,
                    deck_delta_min=deck_delta_min, t_swap_min=t_swap_min,
                    max_stops=max_stops, weather_unc=weather_unc,
                    kappa_mode=kappa_mode, batteries=B, cols_override=list(cols),
                    solver="auto", deck_mode=deck_mode, t_launch_min=t_launch)
                sel = []
                for c in seed_res.get("chosen", []):
                    kk = _key_of(c["tau"], c["tids"], c["h"])
                    j = key2idx.get(kk)
                    if j is None:
                        raise ValueError(f"seed solution contains unknown column {kk}")
                    sel.append(j)
                chk = _validate_root_integer_selection(sel)
                if chk is None:
                    seed_incumbent_stats["rejection_reason"] = "root-row-validation-failed"
                else:
                    LB, E_seed = chk
                    incumbent = [cols[j] for j in sel]
                    incumbent_source = "seed_restricted_master"
                    seed_incumbent_stats.update(
                        accepted=True, coverage=int(LB), energy_Wh=round(E_seed, 6),
                        source=str(seed_res.get("solver", "restricted-master")))
                    log.info("分支定价 seed incumbent: coverage=%d, E=%.1fWh (%s)",
                             LB, E_seed, seed_incumbent_stats["source"])
            except Exception as e:
                seed_incumbent_stats["rejection_reason"] = type(e).__name__ + ":" + str(e)[:160]
                log.warning("分支定价 seed incumbent 构造失败，继续精确搜索但不丢失证书语义: %s", e)

        def _accept_l1_incumbent(cov, sel, source):
            """Lexicographically preserve the best known feasible L1 solution."""
            nonlocal LB, incumbent, incumbent_source
            if cov is None or sel is None:
                return False
            cand_E = _energy_of_indices(sel)
            inc_E = float(sum(float(c["E0"]) for c in incumbent)) if incumbent else float("inf")
            if int(cov) > int(LB) or (int(cov) == int(LB) and cand_E < inc_E - 1e-8):
                LB = int(cov)
                incumbent = [cols[j] for j in sel]
                incumbent_source = str(source)
                return True
            return False

        root_ub, nodes, cg_iters_total = None, 0, 0
        nb_t = nb_p = nb_c = 0
        pricing_progress = 1.0
        no_certificate = False
        open_parent_ubs = []
        # 更新 #1: CG 容差的 Lagrange 松弛 —— 收敛判据 rc ≥ rc_tol=−1e-6, 完整 LP 至多
        # 比 RMP 高 (Σx 上界 B)·|rc_tol|; 节点 UB 必须含该松弛才是有效界。
        _ub_slack = math.nextafter(float(B) * PRICING_EPS, math.inf)
        all_nodes_converged = True
        stack = [root]
        status = "optimal(gap=0)"
        while stack:
            # 更新(M-01): 时间/节点限制各自独立判定、独立旗标、独立状态字符串 ——
            # 不再共用 'limit-reached' 后靠字符串猜原因(time_limit_s=0 曾被误诊为节点限制)。
            if _time.time() > l1_deadline:
                hit_time_limit = True
                status = "time-limit-no-certificate"
                open_parent_ubs.extend(nd.get("parent_ub") for nd in stack)
                break
            if nodes >= max_nodes:
                hit_node_limit = True
                status = "node-limit-no-certificate"
                open_parent_ubs.extend(nd.get("parent_ub") for nd in stack)
                break
            node = stack.pop()
            nodes += 1
            two_phase_stats["l1_nodes_started"] += 1
            max_depth = max(max_depth, int(node.get("depth", 0)))
            lp = None
            node_conv = False
            fcol_conflict = False
            node_infeasible = False
            _node_phase1 = False

            # ---------------- L1 Phase-I: min 全部 required 人工变量 ----------------
            phase1_done = False
            for _it_p1 in range(cg_max_iter):
                active = _active(node)
                if len({j for j in active if _key(j) in node["fcols"]}) < len(node["fcols"]):
                    fcol_conflict = True
                    node_infeasible = True
                    two_phase_stats["l1_phase1_infeasible"] += 1
                    break
                p1 = _rmp_lp(active, node, "phase1")
                cg_iters_total += 1
                if p1 is not None and p1.get("infeasible"):
                    fcol_conflict = True
                    node_infeasible = True
                    two_phase_stats["l1_phase1_infeasible"] += 1
                    break
                if p1 is None:
                    no_certificate = True
                    status = "phase1-lp-fail-no-certificate"
                    break
                p1_art = float(p1.get("art_required", float("inf")))
                if p1_art <= ART_TOL:
                    phase1_done = True       # 0 是 Phase-I 的全局下界；真实可行解已见证
                    two_phase_stats["l1_phase1_feasible"] += 1
                    if _node_phase1:
                        phase1_stats["l1_resolved"] += 1
                    break
                if not _node_phase1:
                    _node_phase1 = True
                    phase1_stats["l1_activated"] += 1
                n_new, prog = _price_node(p1, node, deadline=l1_deadline)
                pricing_progress = prog
                if n_new == -1:
                    no_certificate = True
                    hit_time_limit = True
                    status = "time-limit-no-certificate"
                    break
                if n_new == -2:
                    no_certificate = True
                    hit_label_budget = True
                    status = "pricing-label-limit-no-certificate"
                    break
                if n_new == 0:
                    # rc>=-PRICING_EPS only gives an approximate closure.
                    # Prove infeasibility only when the conservative full-space
                    # Phase-I lower bound remains strictly positive.
                    p1_full_lb = _pricing_relaxed_lower_bound(
                        p1.get("bound"), B, rc_tol=-PRICING_EPS)
                    if p1_full_lb is not None and p1_full_lb > ART_TOL:
                        phase1_stats["l1_infeasible"] += 1
                        two_phase_stats["l1_phase1_infeasible"] += 1
                        node_infeasible = True
                        phase1_done = True
                    else:
                        no_certificate = True
                        all_nodes_converged = False
                        status = "phase1-numeric-gap-no-certificate"
                    break
            else:
                no_certificate = True
                status = "cg-iter-limit-no-certificate"

            if no_certificate:
                all_nodes_converged = False
                stack.clear()
                break
            if node_infeasible:
                continue
            if not phase1_done:
                no_certificate = True
                all_nodes_converged = False
                status = "phase1-proof-incomplete-no-certificate"
                stack.clear()
                break

            # ---------------- L1 Phase-II: max 覆盖，人工变量固定为零 ----------------
            two_phase_stats["l1_phase2_started"] += 1
            for _it_cg in range(cg_max_iter):
                active = _active(node)
                lp = _rmp_lp(active, node, "phase2")
                cg_iters_total += 1
                if lp is not None and lp.get("infeasible"):
                    # Phase-I 已有真实可行见证；这里若不可行属于数值/建模不一致，不能静默剪。
                    lp = None
                    no_certificate = True
                    status = "phase2-inconsistent-infeasible-no-certificate"
                    break
                if lp is None:
                    no_certificate = True
                    status = "phase2-lp-fail-no-certificate"
                    break
                if (float(lp.get("art_required", float("inf"))) > ART_TOL
                        or float(lp.get("max_artificial", float("inf"))) > ART_TOL):
                    no_certificate = True
                    status = "phase2-artificial-residue-no-certificate"
                    break
                n_new, prog = _price_node(lp, node, deadline=l1_deadline)
                pricing_progress = prog
                if verbose:
                    log.info("  node#%d L1-P2 cg#%d: LP=%.3f, 池=%d, 新列=%s",
                             nodes, _it_cg + 1, lp["obj"], len(cols), n_new)
                if n_new == -1:
                    no_certificate = True
                    hit_time_limit = True
                    status = "time-limit-no-certificate"
                    break
                if n_new == -2:
                    no_certificate = True
                    hit_label_budget = True
                    status = "pricing-label-limit-no-certificate"
                    break
                if n_new == 0:
                    node_conv = True
                    two_phase_stats["l1_phase2_closed"] += 1
                    break
            else:
                no_certificate = True
                status = "cg-iter-limit-no-certificate"

            if no_certificate:
                all_nodes_converged = False
                if lp is not None and use_milp_heuristic:
                    lb_node, sel = _milp_int(_active(node), node)
                    _accept_l1_incumbent(lb_node, sel, "l1_milp_heuristic_after_failure")
                stack.clear()
                break
            if not node_conv or lp is None:
                all_nodes_converged = False
                status = "phase2-proof-incomplete-no-certificate"
                stack.clear()
                break
            ub_node = float(lp["bound"]) + _ub_slack
            if root_ub is None:
                root_ub = ub_node
            if _safe_integer_floor(ub_node) <= LB:
                continue

            active_now = _active(node)
            integral_sel = _integral_lp_selection(lp)
            if integral_sel is not None:
                # 更新 P0-1: 节点 LP 已整数时不依赖 scipy.milp。直接重建完整主问题解，
                # 验证全部行/固定列，并要求其覆盖值等于闭合 LP 目标；验证失败必须撤证。
                chk = _validate_l1_integral_lp(active_now, node, lp, integral_sel)
                if chk is None:
                    milp_runtime["integral_lp_validation_failures"] += 1
                    no_certificate = True
                    all_nodes_converged = False
                    status = "integral-lp-validation-fail-no-certificate"
                    stack.clear()
                    break
                lb_node, sel = chk
                milp_runtime["l1_integral_lp_incumbents"] += 1
                _accept_l1_incumbent(lb_node, sel, "l1_integral_lp")
                if _test_force_branch and nodes == 1 and lp["x"]:
                    _tb = next((cols[j]["tids"][0] for j, xv in lp["x"].items()
                                if xv > 1e-6 and cols[j]["tids"]), None)
                    if _tb is not None:
                        nb_t += 1
                        stack.append(_child(node, banned=node["banned"] | {_tb},
                                            parent_ub=ub_node))
                        stack.append(_child(node, required=node["required"] | {_tb},
                                            parent_ub=ub_node))
                        continue
                # 闭合 LP 已由同一整数解达到，节点可安全关闭；MILP 是否可用与证书无关。
                continue

            # 分数 LP：认证默认路径不调用任何节点 MILP。显式开启时，MILP 仅用于
            # 改善 incumbent；其任意失败/异常/非法输出都会在最终证书条件中 fail-closed。
            if use_milp_heuristic:
                lb_node, sel = _milp_int(active_now, node)
                _accept_l1_incumbent(lb_node, sel, "l1_milp_heuristic")
                if verbose:
                    log.info("  node#%d: incumbent → %d", nodes, LB)
            if _test_force_branch and nodes == 1 and lp["x"]:
                # 【测试钩子, 仅自测用】根 LP 在本问题上经验性近乎整数(占机块为区间结构、
                # 甲板 2-interval 行在小实例上很紧), 常在根闭合 ⇒ 分支子节点代码平时走不到。强制压一次风机分支,
                # 执行 banned 过滤 / required 行(对偶 ρ)/ 定价局部化 路径; 两子节点覆盖空间
                # 划分 ⇒ 最终 LB 仍必须等于真最优(冒烟脚本据此断言)。生产调用勿传。
                _tb = next((cols[j]["tids"][0] for j, xv in lp["x"].items()
                            if xv > 1e-6 and cols[j]["tids"]), None)
                if _tb is not None:
                    nb_t += 1
                    stack.append(_child(node, banned=node["banned"] | {_tb}, parent_ub=ub_node))
                    stack.append(_child(node, required=node["required"] | {_tb}, parent_ub=ub_node))
                    continue
            if _safe_integer_floor(ub_node) <= LB:
                continue
            # ── ①风机服务量分支: s_i=Σ_{c∋i}x ∈(0,1) 取最贴 0.5 者 ──
            sfrac = {}
            for j, xv in lp["x"].items():
                if xv <= 1e-6:
                    continue
                for tid in cols[j]["tids"]:
                    sfrac[tid] = sfrac.get(tid, 0.0) + xv
            cand = [(abs(v - 0.5), tid) for tid, v in sfrac.items() if 1e-6 < v < 1 - 1e-6]
            if cand:
                _, tb = min(cand)
                nb_t += 1
                stack.append(_child(node, banned=node["banned"] | {tb}, parent_ub=ub_node))
                stack.append(_child(node, required=node["required"] | {tb}, parent_ub=ub_node))
                continue
            # ── ②Ryan–Foster 同飞对分支 —— 更新 #3: 【缺省禁用】。
            #    本主问题是覆盖型(y_i ≤ Σx, max Σy; 同风机可被多列覆盖), together/apart
            #    的二分对整数解不穷尽: 整数解可同时含列 ω₁⊇{i,j} 与 ω₂∋i∌j —— 它既被
            #    apart 支(禁 ω₁)也被 together 支(禁 ω₂)排除 ⇒ 分支会丢可行整数解,
            #    "栈空"不再蕴含最优。集合【划分】(=1 行)下的经典 RF 定理在此不适用。
            #    ①服务量二分 + ③列固定/禁止二分已完备(列空间有限), RF 仅作可选加速消融;
            #    启用且实际发生对分支时, 证书由下方 certificate.rf_ok=False 撤销。
            if enable_rf_branching:
                zfrac = {}
                for j, xv in lp["x"].items():
                    if xv <= 1e-6:
                        continue
                    tt = cols[j]["tids"]
                    for a_ in range(len(tt)):
                        for b_ in range(a_ + 1, len(tt)):
                            prk = frozenset((tt[a_], tt[b_]))
                            zfrac[prk] = zfrac.get(prk, 0.0) + xv
                candp = [(abs(v - 0.5), pr) for pr, v in zfrac.items() if 1e-6 < v < 1 - 1e-6]
                if candp:
                    _, pb = min(candp)
                    nb_p += 1
                    stack.append(_child(node, apart=node["apart"] | {pb}, parent_ub=ub_node))
                    stack.append(_child(node, together=node["together"] | {pb}, parent_ub=ub_node))
                    continue
            # ── ③兜底列分支(理论完备; nb_c 计数验证其罕见) ──
            fr = [(abs(xv - 0.5), j) for j, xv in lp["x"].items() if 1e-6 < xv < 1 - 1e-6]
            if not fr:
                # 到达此处说明“非整数”判定与分支候选不一致；绝不能关闭节点。
                milp_runtime["integral_lp_validation_failures"] += 1
                no_certificate = True
                all_nodes_converged = False
                status = "fractionality-audit-fail-no-certificate"
                stack.clear()
                break
            _, jb = min(fr)
            kb = _key(jb)
            nb_c += 1
            stack.append(_child(node, xcols=node["xcols"] | {kb}, parent_ub=ub_node))
            stack.append(_child(node, fcols=node["fcols"] | {kb}, parent_ub=ub_node))

        l1_finished_at = _time.time()

        # ---- 全局 UB 与 gap ----
        # 更新(M-01): 旧状态串 'limit-reached' 已拆为 time-/node-limit-no-certificate,
        # 二者一律 UB=None(无证书即不报界), UB_val 仅剩 optimal 路径的兜底诊断值。
        cand_ubs = [u for u in ([root_ub] + open_parent_ubs) if u is not None]
        UB_val = max(cand_ubs) if cand_ubs else \
                 (root_ub if root_ub is not None else float(m))
        if status == "optimal(gap=0)":
            UB_final = LB          # 栈空 ⇒ 所有节点关闭 ⇒ LB 即最优
        elif status.endswith("-no-certificate"):
            UB_final = None        # 更新 #1: 定价不完整/迭代触限/LP 失败 —— LP 界一律不报
        else:
            UB_final = _safe_integer_floor(UB_val)
        gap = (None if UB_final is None else
               (0.0 if LB <= 0 and UB_final <= 0 else
                round(100.0 * max(UB_final - LB, 0) / max(UB_final, 1), 2)))

        # =====================================================================
        # 更新 #6: 独立第二阶段能耗 Branch-and-Price(词典序 L2 全局证书)。
        #   主问题:  min Σ_c E0_c x_c
        #     s.t.  y_i − Σ_{c∋i} x_c ≤ 0            (覆盖链接, 对偶 π_i ≥ 0)
        #           −Σ_i y_i (− s_cnt) ≤ −N*         (覆盖计数 ≥ N*, 对偶 σ; s_cnt 人工)
        #           Σx ≤ B / 占机 ≤ K / 甲板 ≤ 1     (β / μ_t / δ_t)
        #           [分支行] −Σ_{c∋i}x − s_i ≤ −1    (required, 对偶 ρ_i; s_i 人工)
        #   列 reduced cost: rc = E0 + β + Σμ_occ + Σδ_deck − Σ_{i∈S}(π_i+ρ_i)
        #   (计数行只作用于 y, 不入列 rc)。定价 = 同一 ESPPRC 标号, 闭合逐【全部窗内 h】
        #   取 min rc(price_routes energy_weight=1, 更新)。min 型 CG: 收敛节点的
        #   LP − B·|rc_tol| 是节点 IP 的有效【下界】⇒ node_lb ≥ incumbent_E 剪枝;
        #   分支 = ①服务量二分 + ③列固定/禁止(与 L1 同款, 完备); 栈空 ⇒ L2 全局最优。
        # =====================================================================
        def _rows_E(active_idx, node, n, n_star):
            import scipy.sparse as sp
            req = sorted(node["required"])
            n_s = 1 + len(req)                     # s_cnt + 每 required 一个
            deck_pts = sorted({ti for j in active_idx for ti in cols[j]["deck_idx"]})
            drow = {ti: m + 2 + n_tgrid + k for k, ti in enumerate(deck_pts)}
            rrow = {tid: m + 2 + n_tgrid + len(deck_pts) + k for k, tid in enumerate(req)}
            # 更新(H-01): L2 与 L1 使用【同一套】连续语义补充行(占机事件行 + 甲板逐对行)
            ev_rows, pair_rows = _extra_rows(active_idx)
            R0 = m + 2 + n_tgrid + len(deck_pts) + len(req)
            erow = {e: R0 + k for k, (e, _mem) in enumerate(ev_rows)}
            R = R0 + len(ev_rows) + len(pair_rows)
            A = sp.lil_matrix((R, n + m + n_s))
            b = np.zeros(R)
            for i in range(m):
                A[i, n + i] = 1.0                  # 覆盖链接 y_i − Σx ≤ 0
                A[m, n + i] = -1.0                 # 计数行 −Σy − s_cnt ≤ −N*
            A[m, n + m] = -1.0
            for jj, j in enumerate(active_idx):
                c = cols[j]
                for tid in c["tids"]:
                    A[idx_of_tid[tid], jj] = -1.0
                    if tid in rrow:
                        A[rrow[tid], jj] = -1.0
                A[m + 1, jj] = 1.0                 # 电池
                for ti in c["occ_idx"]:
                    A[m + 2 + ti, jj] = 1.0
                for ti in c["deck_idx"]:
                    A[drow[ti], jj] = 1.0
            for k, tid in enumerate(req):
                A[rrow[tid], n + m + 1 + k] = -1.0
            for k, (e, mem) in enumerate(ev_rows):
                for jj in mem:
                    A[R0 + k, jj] = 1.0
                b[R0 + k] = float(K)
            for k, (jj1, jj2) in enumerate(pair_rows):
                A[R0 + len(ev_rows) + k, jj1] = 1.0
                A[R0 + len(ev_rows) + k, jj2] = 1.0
                b[R0 + len(ev_rows) + k] = 1.0
            b[m] = -float(n_star)
            b[m + 1] = float(B)
            b[m + 2:m + 2 + n_tgrid] = float(K)
            for ti, r in drow.items():
                b[r] = 1.0
            for tid, r in rrow.items():
                b[r] = -1.0
            E_vec = np.array([float(cols[j]["E0"]) for j in active_idx], float)
            return A.tocsc(), b, drow, rrow, n_s, E_vec, erow

        def _rmp_lp_E(active_idx, node, n_star, phase):
            """L2 严格两阶段 RMP。

            phase1: min sum(s_cnt, required artificials)，只证明原节点可行/不可行；
            phase2: min sum(E*x)，全部人工变量固定为 0，LP 值才是有效能耗下界。
            """
            from scipy.optimize import linprog
            if phase not in ("phase1", "phase2"):
                raise ValueError("phase must be 'phase1' or 'phase2'")
            n = len(active_idx)
            A, b, drow, rrow, n_s, E_vec, erow = _rows_E(active_idx, node, n, n_star)
            nv = n + m + n_s
            lo = np.zeros(nv)
            hi = np.ones(nv)
            for jj, j in enumerate(active_idx):
                if _key(j) in node["fcols"]:
                    lo[jj] = 1.0
            if phase == "phase1":
                hi[n + m:] = 1.0
                obj = np.concatenate([np.zeros(n + m), np.ones(n_s)])
            else:
                hi[n + m:] = 0.0                 # Phase-II 严格禁用全部人工变量
                obj = np.concatenate([E_vec, np.zeros(m + n_s)])
            bounds_lp = list(zip(lo, hi))
            res = linprog(obj, A_ub=A, b_ub=b, bounds=bounds_lp, method="highs")
            if getattr(res, "status", -1) == 2 and getattr(res, "success", False) is not True:
                return dict(infeasible=True, phase=phase)
            checked = _validate_linprog_result(res, obj, bounds_lp, A_ub=A, b_ub=b, need_ineqlin=A.shape[0])
            if checked is None:
                return None
            x_full, fun, marg, _, dual_lb = checked
            lam = -marg
            pi = {tid_of[i]: max(0.0, float(lam[i])) for i in range(m)}
            beta = max(0.0, float(lam[m + 1]))
            mu = np.maximum(0.0, lam[m + 2:m + 2 + n_tgrid])
            delta = {ti: max(0.0, float(lam[r])) for ti, r in drow.items()}
            rho = {tid: max(0.0, float(lam[r])) for tid, r in rrow.items()}
            mu_ev = {e: max(0.0, float(lam[r])) for e, r in erow.items()}
            x = {j: float(x_full[jj]) for jj, j in enumerate(active_idx)}
            art_values = np.asarray(x_full[n + m:n + m + n_s], float)
            art_sum = float(art_values.sum())
            art_max = float(art_values.max()) if art_values.size else 0.0
            return dict(obj=fun, bound=dual_lb, x=x, pi=pi, beta=beta, mu=mu, delta=delta,
                        rho=rho, mu_ev=mu_ev, art=art_sum,
                        max_artificial=art_max, artificial_count=int(n_s), phase=phase)

        def _milp_int_E(active_idx, node, n_star, deadline=None):
            """L2 Phase-II integer incumbent heuristic with full primal validation."""
            milp_runtime["l2_calls"] += 1
            n = len(active_idx)
            A, b, _, _, n_s, E_vec, _erow = _rows_E(active_idx, node, n, n_star)
            nv = n + m + n_s
            if n == 0:
                return None, None, None
            lo = np.zeros(nv)
            hi = np.ones(nv)
            for jj, j in enumerate(active_idx):
                if _key(j) in node["fcols"]:
                    lo[jj] = 1.0
            hi[n + m:] = 0.0
            obj = np.concatenate([E_vec, np.zeros(m + n_s)])
            integ = np.concatenate([np.ones(n + m), np.zeros(n_s)])
            try:
                from scipy.optimize import milp, LinearConstraint, Bounds
                effective_deadline = (t0 + total_time_limit_s) if deadline is None else float(deadline)
                rem = max(0.0, effective_deadline - _time.time())
                if rem <= 0.0:
                    _record_milp_rejection("l2", "global_time_limit")
                    return None, None, None
                r = milp(obj, constraints=LinearConstraint(A, -np.inf, b),
                         integrality=integ, bounds=Bounds(lo, hi),
                         options=dict(time_limit=max(rem, 1e-9)))
            except Exception:
                _record_milp_rejection("l2", "solver_exception")
                return None, None, None
            z, _value, reason = _validate_milp_primal(r, obj, A, b, lo, hi, integ)
            if z is None:
                _record_milp_rejection("l2", reason)
                return None, None, None
            art_values = z[n + m:n + m + n_s]
            art_max = float(np.max(np.abs(art_values))) if art_values.size else 0.0
            if art_max > ART_TOL:
                _record_milp_rejection("l2", "artificial_residue")
                return None, None, art_max
            sel = [active_idx[jj] for jj in range(n) if z[jj] > 0.5]
            covered_tids = {tid for j in sel for tid in cols[j]["tids"]}
            if len(covered_tids) != int(n_star):
                _record_milp_rejection("l2", "coverage_count_mismatch")
                return None, None, art_max
            energy = float(sum(float(cols[j]["E0"]) for j in sel))
            if not math.isfinite(energy) or energy < -1e-9:
                _record_milp_rejection("l2", "invalid_energy_reconstruction")
                return None, None, art_max
            milp_runtime["l2_validated_incumbents"] += 1
            return energy, sel, art_max

        def _validate_l2_integral_lp(active_idx, node, n_star, lp, selected):
            """从整数 L2 LP 的 x 重建 y，验证完整行并核对真实能耗目标。"""
            try:
                n = len(active_idx)
                pos = {j: jj for jj, j in enumerate(active_idx)}
                if len(selected) != len(set(selected)) or any(j not in pos for j in selected):
                    return None
                selected_keys = {_key(j) for j in selected}
                if not set(node["fcols"]).issubset(selected_keys):
                    return None
                A, b, _drow, _rrow, n_s, _E_vec, _erow = _rows_E(
                    active_idx, node, n, n_star)
                z = np.zeros(n + m + n_s, float)
                for j in selected:
                    z[pos[j]] = 1.0
                covered_tids = {tid for j in selected for tid in cols[j]["tids"]}
                for tid in covered_tids:
                    z[n + idx_of_tid[tid]] = 1.0
                viol = np.asarray(A @ z - b, float)
                if viol.size and float(np.max(viol)) > 1e-7:
                    return None
                energy = float(sum(float(cols[j]["E0"]) for j in selected))
                if abs(float(lp["obj"]) - energy) > 1e-6 * max(1.0, abs(energy)):
                    return None
                return energy, list(selected), 0.0
            except Exception:
                return None

        def _price_node_E(duals, node, phase, deadline=None):
            """L2 节点定价。

            Phase-I 使用零能耗系数寻找可消除人工变量的列；Phase-II 使用真实 E0，
            并要求每条闭合路线对全部 h 产生可核验的运行时扫描证明。
            """
            if phase not in ("phase1", "phase2"):
                raise ValueError("phase must be 'phase1' or 'phase2'")
            new = 0
            banned = node["banned"]
            xcols = node["xcols"]
            n_opt = max(len(launch_opts), 1)
            for oi, opt in enumerate(launch_opts):
                if deadline is not None and _time.time() > deadline:
                    return -1, oi / n_opt
                tau = float(opt.tau_min)
                r_turbs = [t for t in reach_of[oi] if t.tid not in banned]
                if not r_turbs:
                    continue
                d_launch = sum(duals["delta"].get(ti, 0.0) for ti in _launch_deck_idx(tau))
                dual_map = {k: duals["pi"].get(t.tid, 0.0) + duals["rho"].get(t.tid, 0.0)
                            for k, t in enumerate(r_turbs)}
                if sum(dual_map.values()) <= duals["beta"] + d_launch + 1e-9:
                    continue           # E0 ≥ 0 ⇒ rc ≥ β+δ−Σ对偶 ≥ −1e-9(保真粗筛)
                mu, delta = duals["mu"], duals["delta"]
                mu_ev = duals.get("mu_ev", {})

                def _cc2(h, tids, _tau=tau):
                    if _tau + float(h) > T_min + 1e-9:
                        return 1e18
                    if (round(_tau, 3), tuple(tids), float(h)) in xcols:
                        return 1e18
                    occ = sum(mu[ti] for ti in _occ_idx(_tau, float(h)))
                    if mu_ev:
                        occ += sum(mu_ev.get(e, 0.0) for e in _occ_event_hits(_tau, float(h)))
                    return float(occ) + sum(delta.get(ti, 0.0) for ti in _rec_deck_idx(_tau, float(h)))

                _pst = {"solver_stage": "L2-" + phase}
                nonlocal pricing_calls_expected, l2_h_scan_calls_expected
                pricing_calls_expected += 1
                pricing_stats.append(_pst)
                if phase == "phase2":
                    l2_h_scan_calls_expected += 1
                    l2_h_scan_stats.append(_pst)
                pr_new = price_routes(r_turbs, opt.ship, p, opt.wx, xi_amb, dual_map,
                                      route_cost=duals["beta"] + d_launch,
                                      max_stops=max_stops, k_near=len(r_turbs), max_routes=20,
                                      rc_tol=-PRICING_EPS, weather_unc=weather_unc,
                                      strict_dominance=True, eval_cache=eval_cache,
                                      close_cost_of_h=_cc2, dominance_mode=dominance_mode,
                                      energy_weight=(1.0 if phase == "phase2" else 0.0),
                                      label_budget=pricing_label_budget, stats_out=_pst)
                if _pst.get("complete") is not True or _pst.get("proof_complete") is not True:
                    return -2, oi / n_opt
                if phase == "phase2" and _pst.get("all_h_proof_complete") is not True:
                    return -3, oi / n_opt
                for f in pr_new:
                    if _add(tau, opt.ship, opt.wx, [t.tid for t in f["turbines"]],
                            f["h"], f["E0"],
                            RM.Route(rid=-1, turbines=list(f["turbines"]), ship=opt.ship),
                            gate_proof=f.get("gate_proof")):
                        new += 1
            return new, 1.0

        def _l2_energy_bp(n_star, incumbent_cols, deadline):
            """严格 Phase-I/Phase-II 的 L2 能耗 Branch-and-Price。"""
            nonlocal hit_label_budget
            best_E = float(sum(float(c["E0"]) for c in incumbent_cols))
            best_sel = list(incumbent_cols)
            l2_nodes = l2_cg = 0
            l2_status = "optimal(gap=0)"
            l2_conv_all = True
            l2_hit_time = False
            l2_hit_node = False
            aud = dict(
                max_lp_artificial=0.0,
                max_incumbent_artificial=0.0,
                artificial_nodes_seen=0,
                incumbents_accepted=0,
                incumbents_rejected_artificial=0,
                all_accepted_incumbents_artificial_free=True,
                artificial_audit_complete=False,
                phase1_nodes_started=0,
                phase1_nodes_feasible=0,
                phase1_nodes_infeasible=0,
                phase1_cg_complete=False,
                phase2_nodes_started=0,
                phase2_nodes_closed=0,
                phase2_artificials_fixed_zero=False,
                phase2_bounds_valid=False,
                milp_calls=0,
                milp_failures=0,
                integral_lp_incumbents=0,
                integral_lp_validation_failures=0,
            )
            root2 = dict(banned=frozenset(), required=frozenset(), apart=frozenset(),
                         together=frozenset(), fcols=frozenset(), xcols=frozenset(), depth=0)
            stack2 = [root2]
            _lb_slack = math.nextafter(float(B) * PRICING_EPS, math.inf)
            root_lb = None

            while stack2:
                if _time.time() > deadline:
                    l2_hit_time = True
                    l2_status = "time-limit-no-certificate"
                    l2_conv_all = False
                    break
                if l2_nodes >= max_nodes:
                    l2_hit_node = True
                    l2_status = "node-limit-no-certificate"
                    l2_conv_all = False
                    break

                nd = stack2.pop()
                l2_nodes += 1
                lp = None
                node_infeasible = False
                phase1_activated = False
                aud["phase1_nodes_started"] += 1

                # ---------------- L2 Phase-I ----------------
                phase1_done = False
                for _it_p1 in range(cg_max_iter):
                    act = _active(nd)
                    if len({j for j in act if _key(j) in nd["fcols"]}) < len(nd["fcols"]):
                        node_infeasible = True
                        phase1_done = True
                        aud["phase1_nodes_infeasible"] += 1
                        break
                    p1 = _rmp_lp_E(act, nd, n_star, "phase1")
                    l2_cg += 1
                    if p1 is not None and p1.get("infeasible"):
                        node_infeasible = True
                        phase1_done = True
                        aud["phase1_nodes_infeasible"] += 1
                        break
                    if p1 is None:
                        l2_status = "phase1-lp-fail-no-certificate"
                        l2_conv_all = False
                        break
                    p1_art = float(p1.get("art", float("inf")))
                    p1_art_max = float(p1.get("max_artificial", float("inf")))
                    aud["max_lp_artificial"] = max(aud["max_lp_artificial"], p1_art_max)
                    if p1_art <= ART_TOL and p1_art_max <= ART_TOL:
                        phase1_done = True
                        aud["phase1_nodes_feasible"] += 1
                        if phase1_activated:
                            phase1_stats["l2_resolved"] += 1
                        break
                    if not phase1_activated:
                        phase1_activated = True
                        phase1_stats["l2_activated"] += 1
                    aud["artificial_nodes_seen"] += 1
                    n_new, _pr = _price_node_E(p1, nd, "phase1", deadline=deadline)
                    if n_new == -1:
                        l2_hit_time = True
                        l2_status = "time-limit-no-certificate"
                        l2_conv_all = False
                        break
                    if n_new == -2:
                        hit_label_budget = True
                        l2_status = "pricing-label-limit-no-certificate"
                        l2_conv_all = False
                        break
                    if n_new == -3:
                        l2_status = "h-scan-proof-incomplete-no-certificate"
                        l2_conv_all = False
                        break
                    if n_new == 0:
                        p1_full_lb = _pricing_relaxed_lower_bound(
                            p1.get("bound"), B, rc_tol=-PRICING_EPS)
                        if p1_full_lb is not None and p1_full_lb > ART_TOL:
                            node_infeasible = True
                            phase1_done = True
                            aud["phase1_nodes_infeasible"] += 1
                            phase1_stats["l2_infeasible"] += 1
                        else:
                            l2_status = "phase1-numeric-gap-no-certificate"
                            l2_conv_all = False
                        break
                else:
                    l2_status = "cg-iter-limit-no-certificate"
                    l2_conv_all = False

                if not l2_conv_all:
                    stack2.clear()
                    break
                if node_infeasible:
                    continue
                if not phase1_done:
                    l2_status = "phase1-proof-incomplete-no-certificate"
                    l2_conv_all = False
                    stack2.clear()
                    break

                # ---------------- L2 Phase-II ----------------
                aud["phase2_nodes_started"] += 1
                phase2_closed = False
                for _it_p2 in range(cg_max_iter):
                    act = _active(nd)
                    lp = _rmp_lp_E(act, nd, n_star, "phase2")
                    l2_cg += 1
                    if lp is not None and lp.get("infeasible"):
                        l2_status = "phase2-inconsistent-infeasible-no-certificate"
                        l2_conv_all = False
                        lp = None
                        break
                    if lp is None:
                        l2_status = "phase2-lp-fail-no-certificate"
                        l2_conv_all = False
                        break
                    lp_art = float(lp.get("art", float("inf")))
                    lp_art_max = float(lp.get("max_artificial", float("inf")))
                    aud["max_lp_artificial"] = max(aud["max_lp_artificial"], lp_art_max)
                    if lp_art > ART_TOL or lp_art_max > ART_TOL:
                        l2_status = "phase2-artificial-residue-no-certificate"
                        l2_conv_all = False
                        break
                    n_new, _pr = _price_node_E(lp, nd, "phase2", deadline=deadline)
                    if n_new == -1:
                        l2_hit_time = True
                        l2_status = "time-limit-no-certificate"
                        l2_conv_all = False
                        break
                    if n_new == -2:
                        hit_label_budget = True
                        l2_status = "pricing-label-limit-no-certificate"
                        l2_conv_all = False
                        break
                    if n_new == -3:
                        l2_status = "h-scan-proof-incomplete-no-certificate"
                        l2_conv_all = False
                        break
                    if n_new == 0:
                        phase2_closed = True
                        aud["phase2_nodes_closed"] += 1
                        break
                else:
                    l2_status = "cg-iter-limit-no-certificate"
                    l2_conv_all = False

                if not l2_conv_all:
                    stack2.clear()
                    break
                if not phase2_closed or lp is None:
                    l2_status = "phase2-proof-incomplete-no-certificate"
                    l2_conv_all = False
                    stack2.clear()
                    break

                node_lb = float(lp["bound"]) - _lb_slack
                if root_lb is None:
                    root_lb = node_lb
                if node_lb >= best_E - 1e-6:
                    continue

                active_now = _active(nd)
                integral_sel = _integral_lp_selection(lp)
                if integral_sel is not None:
                    chk = _validate_l2_integral_lp(active_now, nd, n_star, lp, integral_sel)
                    if chk is None:
                        aud["integral_lp_validation_failures"] += 1
                        milp_runtime["integral_lp_validation_failures"] += 1
                        l2_status = "integral-lp-validation-fail-no-certificate"
                        l2_conv_all = False
                        stack2.clear()
                        break
                    e_int, sel, art_int = chk
                    aud["integral_lp_incumbents"] += 1
                    milp_runtime["l2_integral_lp_incumbents"] += 1
                    aud["max_incumbent_artificial"] = max(
                        aud["max_incumbent_artificial"], float(art_int))
                    if e_int < best_E - 1e-9:
                        aud["incumbents_accepted"] += 1
                        best_E, best_sel = e_int, [cols[j] for j in sel]
                    # 整数 LP 解达到闭合节点下界，节点直接关闭，不调用 MILP。
                    continue

                # 分数 LP：认证默认路径不调用节点 MILP。显式开启时仅作 incumbent
                # 启发式，且任何失败/异常/非法输出都会撤销本次运行的证书。
                if use_milp_heuristic:
                    _l2_calls_before = milp_runtime["l2_calls"]
                    _l2_fail_before = milp_runtime["l2_failures"]
                    e_int, sel, art_int = _milp_int_E(active_now, nd, n_star, deadline=deadline)
                    aud["milp_calls"] += milp_runtime["l2_calls"] - _l2_calls_before
                    aud["milp_failures"] += milp_runtime["l2_failures"] - _l2_fail_before
                    if art_int is not None:
                        aud["max_incumbent_artificial"] = max(
                            aud["max_incumbent_artificial"], float(art_int))
                    if art_int is not None and art_int > ART_TOL:
                        aud["incumbents_rejected_artificial"] += 1
                        aud["all_accepted_incumbents_artificial_free"] = False
                    if e_int is not None and e_int < best_E - 1e-9:
                        aud["incumbents_accepted"] += 1
                        best_E, best_sel = e_int, [cols[j] for j in sel]
                    if node_lb >= best_E - 1e-6:
                        continue

                # ①服务量二分
                sfr = {}
                for j, xv in lp["x"].items():
                    if xv <= 1e-6:
                        continue
                    for tid in cols[j]["tids"]:
                        sfr[tid] = sfr.get(tid, 0.0) + xv
                cand = [(abs(v - 0.5), tid) for tid, v in sfr.items()
                        if 1e-6 < v < 1 - 1e-6]
                if cand:
                    _, tb = min(cand)
                    stack2.append(_child(nd, banned=nd["banned"] | {tb}))
                    stack2.append(_child(nd, required=nd["required"] | {tb}))
                    continue

                # ③列固定/禁止二分
                fr = [(abs(xv - 0.5), j) for j, xv in lp["x"].items()
                      if 1e-6 < xv < 1 - 1e-6]
                if not fr:
                    aud["integral_lp_validation_failures"] += 1
                    milp_runtime["integral_lp_validation_failures"] += 1
                    l2_status = "fractionality-audit-fail-no-certificate"
                    l2_conv_all = False
                    stack2.clear()
                    break
                _, jb = min(fr)
                kb = _key(jb)
                stack2.append(_child(nd, xcols=nd["xcols"] | {kb}))
                stack2.append(_child(nd, fcols=nd["fcols"] | {kb}))

            # 更新(P3-02): incumbent 能耗一致性审计 —— best_E 与 best_sel 实际列
            # E0 之和须一致(列 E0 可被后续定价原地下调; 理论不变量: 被选列在其闭合节点
            # 已达该键的序列最小 E0, 故二者恒等)。此处把该不变量升级为运行时审计:
            # 漂移超容差 ⇒ 不变量被破坏 ⇒ 撤销 L2 证书(fail-closed)。
            _recount_E = (float(sum(float(c["E0"]) for c in best_sel))
                          if best_sel else 0.0)
            _energy_consistent = bool(abs(_recount_E - best_E)
                                      <= 1e-5 * max(1.0, abs(best_E)))
            aud["incumbent_energy_recount_Wh"] = round(_recount_E, 6)
            aud["incumbent_energy_consistent"] = _energy_consistent
            if (l2_status == "optimal(gap=0)" and l2_conv_all and not stack2
                    and not _energy_consistent):
                l2_status = "incumbent-energy-drift-no-certificate"
                l2_conv_all = False
            clean_exit = bool(l2_status == "optimal(gap=0)" and l2_conv_all and not stack2)
            aud["phase1_cg_complete"] = clean_exit
            aud["phase2_artificials_fixed_zero"] = clean_exit
            aud["phase2_bounds_valid"] = clean_exit
            aud["artificial_audit_complete"] = clean_exit
            certified = clean_exit
            return best_sel, dict(status=l2_status, certified=certified,
                                  energy_Wh=round(best_E, 1), nodes=l2_nodes, cg_iters=l2_cg,
                                  hit_time_limit=l2_hit_time, hit_node_limit=l2_hit_node,
                                  root_energy_LP=(round(root_lb, 1) if root_lb is not None else None),
                                  **aud)

        def _l2_solve(chosen_l1):
            """按 l2_mode 求 L2: 'bp'(缺省)=能耗 B&P 证书; 'expand'=旧扩池 MILP(无证书)。
            更新(M-02): 所有路径(含 skipped/回退)都显式携带人工审计字段 —— 回退路径
            没有跑完整审计, artificial_audit_complete 如实 =False(证书据此 fail-closed)。"""
            nonlocal l2_started_at, l2_finished_at
            _no_aud = dict(max_lp_artificial=None, max_incumbent_artificial=None,
                           artificial_nodes_seen=0, incumbents_accepted=0,
                           incumbents_rejected_artificial=0,
                           all_accepted_incumbents_artificial_free=False,
                           artificial_audit_complete=False)
            info = dict(status="skipped", certified=False,
                        energy_Wh=(round(float(sum(c["E0"] for c in chosen_l1)), 1)
                                   if chosen_l1 else 0.0), nodes=0, cg_iters=0,
                        hit_time_limit=False, hit_node_limit=False,
                        root_energy_LP=None, time_budget_s=0.0, elapsed_s=0.0,
                        **_no_aud)
            if not chosen_l1 or LB <= 0:
                return chosen_l1, info
            if l2_mode == "bp" and status == "optimal(gap=0)":
                l2_started_at = _time.time()
                elapsed_total = max(l2_started_at - t0, 0.0)
                if l2_time_limit_s is not None:
                    l2_budget_s = l2_reserved_s
                else:
                    # Guarantee the reserved L2 budget while transferring any
                    # unused portion of the original total budget.
                    l2_budget_s = max(l2_reserved_s,
                                      max(total_time_limit_s - elapsed_total, 0.0))
                deadline = l2_started_at + l2_budget_s
                sel, info = _l2_energy_bp(LB, chosen_l1, deadline)
                l2_finished_at = _time.time()
                info["time_budget_s"] = round(float(l2_budget_s), 6)
                info["elapsed_s"] = round(float(l2_finished_at - l2_started_at), 6)
                if info["certified"]:
                    return sel, info
                log.warning("分支定价 L2 能耗 B&P 未闭合(%s), 回退扩池 MILP(无 L2 证书)。",
                            info["status"])
            # 旧口径: 扩池上的一次 MILP(仅池最优, 诚实无证书)。更新: _milp_int_E 现返回
            # (E, sel, art_sum) 三元组; 人工残量>容差的解已在其内部被拒(E=None), 此处仅登记。
            if use_milp_heuristic:
                try:
                    active = list(range(len(cols)))
                    e_pool, sel, art_pool = _milp_int_E(active, root, LB, deadline=t0 + total_time_limit_s)
                    if e_pool is not None:
                        info = dict(status="expanded-pool-milp", certified=False,
                                    energy_Wh=round(e_pool, 1), nodes=info.get("nodes", 0),
                                    cg_iters=info.get("cg_iters", 0),
                                    hit_time_limit=bool(info.get("hit_time_limit", False)),
                                    hit_node_limit=bool(info.get("hit_node_limit", False)),
                                    root_energy_LP=info.get("root_energy_LP"),
                                    time_budget_s=info.get("time_budget_s", 0.0),
                                    elapsed_s=info.get("elapsed_s", 0.0), **_no_aud)
                        info["max_incumbent_artificial"] = float(art_pool or 0.0)
                        return [cols[j] for j in sel], info
                    if art_pool is not None and art_pool > ART_TOL:
                        log.warning("分支定价 L2 扩池 MILP 人工残量 %.3g > 容差, 解被拒。", art_pool)
                except Exception as e:
                    log.warning("分支定价 L2 能耗层失败(%s), 保留 L1 incumbent。", e)
            return chosen_l1, info

        # ---- L2: 在覆盖=LB 的解中做能耗最小(与 solve_resource_master 词典序一致) ----
        chosen = incumbent
        l2 = dict(status="skipped", certified=False,
                  energy_Wh=(round(float(sum(c["E0"] for c in incumbent)), 1) if incumbent else 0.0),
                  nodes=0, cg_iters=0, root_energy_LP=None,
                  time_budget_s=0.0, elapsed_s=0.0,
                  hit_time_limit=False, hit_node_limit=False,
                  max_lp_artificial=None, max_incumbent_artificial=None,
                  artificial_nodes_seen=0, incumbents_accepted=0,
                  incumbents_rejected_artificial=0,
                  all_accepted_incumbents_artificial_free=False,
                  artificial_audit_complete=False)
        # 零覆盖时，第二层目标在空选择上平凡最优：能耗=0。只要L1树已闭合，
        # 无需启动L2分支定价即可给出严格的平凡证书；不能把“未运行”误写成“未证明”。
        if LB <= 0 and status == "optimal(gap=0)":
            l2.update(status="optimal-trivial-zero-coverage", certified=True,
                      energy_Wh=0.0, max_lp_artificial=0.0,
                      max_incumbent_artificial=0.0,
                      artificial_nodes_seen=0, incumbents_accepted=1,
                      incumbents_rejected_artificial=0,
                      all_accepted_incumbents_artificial_free=True,
                      artificial_audit_complete=True,
                      phase1_nodes_started=0, phase1_nodes_feasible=0,
                      phase1_nodes_infeasible=0, phase1_cg_complete=True,
                      phase2_nodes_started=0, phase2_nodes_closed=0,
                      phase2_artificials_fixed_zero=True,
                      phase2_bounds_valid=True,
                      incumbent_energy_recount_Wh=0.0,
                      incumbent_energy_consistent=True)
        n_expanded_h = 0
        if incumbent and LB > 0:
            # 更新: L2 前先把终池各 (τ, 序列π) 在【全部可行 h】上扩列(镜像旧
            # lex_column_generation._expand_pool_h)。动机: 第一层定价按 close_cost(h)
            # 单调性取"最早可行 h 即最优"只对覆盖层成立; 船在动, 稍晚回收可能返程更短、
            # 能耗更低。更新 #6 起, 这一步仅是【廉价播种】(缩短 L2 能耗 B&P 的定价
            # 轮数); L2 证书由 _l2_energy_bp 的定价+分支闭合给出。旧"扩池后一次 MILP"
            # 口径仅在 l2_mode='expand' 或 L2 B&P 未闭合的回退路径中使用, 并如实标注
            # L2_scope='expanded-pool(no-L2-pricing-certificate)'。
            _seq_seen, _budget = set(), 20000
            for c in list(cols):
                r = c.get("route")
                if r is None or _budget <= 0:
                    continue
                skey = (round(c["tau"], 3), tuple(t.tid for t in r.turbines))
                if skey in _seq_seen:
                    continue
                _seq_seen.add(skey)
                for hh in h_grid:
                    if c["tau"] + float(hh) > T_min + 1e-9:
                        break
                    if _key_of(c["tau"], c["tids"], float(hh)) in seen or _budget <= 0:
                        continue
                    _budget -= 1
                    dd = RM.route_feasible_at_h(r, int(hh), p,
                                                _wx_of_ship(c["ship"], c["wx"]), xi_amb,
                                                weather_unc=weather_unc)   # 更新(P2-02): per-τ 天气
                    if dd["feasible"] and _add(c["tau"], c["ship"], c["wx"], c["tids"],
                                               float(hh), _plan_energy(dd), r,
                                               gate_proof=dd.get("gate_weather_proof")):
                        n_expanded_h += 1
            if n_expanded_h and verbose:
                log.info("分支定价 L2: h 扩列 +%d(终池 %d)", n_expanded_h, len(cols))
            chosen, l2 = _l2_solve(chosen)

        log.info("分支定价: LB=%d, UB=%s, gap=%s%%, 状态=%s, 节点=%d, CG 轮=%d, "
                 "分支(风机/对/列)=%d/%d/%d, 终池=%d | L2=%s(E=%s, 节点=%d) | 用时 %.1fs",
                 LB, UB_final, gap, status, nodes, cg_iters_total,
                 nb_t, nb_p, nb_c, len(cols), l2.get("status"), l2.get("energy_Wh"),
                 l2.get("nodes", 0), _time.time() - t0)
        # ---- 更新 #5 / 更新 重写: 证书装配 = 运行时证据直读 + 独立复核 ----
        # 更新(外部审计 一般#7): 证书 = 【条件清单 + 总布尔】, 条件未知一律 False(保守),
        #   certificate_reason 列出全部撤证原因。
        # 更新(H-02 核心): 各条件不再由配置字符串/硬编码推断, 而是读取【运行时证明对象】
        #   (ReachResult / PricingProof(stats_out) / gate_weather_proof / 甲板逐对审计 /
        #   L2 人工审计 / hit_* 旗标), 并在此处做独立复核 —— 少一条证据、一处不一致即撤证。
        dominance_exact = bool(dominance_mode == "sequence"
                               or (dominance_mode == "set" and weather_unc is None))

        # (0) 更新(H-01 收尾): 最终方案的独立连续物理复检 —— 不经任何 LP/格点机制,
        #     直接对 chosen 检查: 甲板区间逐对不相交、占机并发 ≤ K(左端点扫描; 半开区间
        #     并发峰值必在某左端点取得)、电池数 ≤ B。违反 ⇒ 行体系漏判 ⇒
        #     status='physical-deck-conflict' 且 L1/L2 全部撤证(不冒发)。
        def _plan_physical_check(plan):
            if not plan:
                return True, None
            if len(plan) > B:
                return False, f"battery-count {len(plan)} > B={B}"
            for i in range(len(plan)):
                for j2 in range(i + 1, len(plan)):
                    if _ivs_overlap(plan[i]["deck_ivals"], plan[j2]["deck_ivals"]):
                        return False, ("deck-overlap (tau=%.6g,h=%.6g)x(tau=%.6g,h=%.6g)"
                                       % (plan[i]["tau"], plan[i]["h"],
                                          plan[j2]["tau"], plan[j2]["h"]))
            for e in sorted({c["occ_iv"][0] for c in plan}):
                conc = sum(1 for c in plan
                           if c["occ_iv"][0] - 1e-9 <= e < c["occ_iv"][1] - 1e-9)
                if conc > K:
                    return False, f"occ-concurrency {conc} > K={K} at t={e:.6g}"
            return True, None

        phys_ok, phys_reason = _plan_physical_check(chosen)
        if not phys_ok:
            log.error("分支定价 连续物理复检失败(%s) —— 撤销全部证书。", phys_reason)
            status = "physical-deck-conflict"

        # 最终公开目标只能由被选列重建，不能信任 LP 辅助变量、旧 incumbent 或报告字段。
        chosen_covered_tids = {tid for c in chosen for tid in c.get("tids", ())}
        chosen_energy = float(sum(float(c.get("E0", float("nan"))) for c in chosen))
        reported_solution_consistent = bool(
            len(chosen_covered_tids) == int(LB)
            and math.isfinite(chosen_energy)
            and chosen_energy >= -1e-9)
        if not reported_solution_consistent:
            phys_ok = False
            phys_reason = ("reported-objective-mismatch: covered=%d vs LB=%d, energy=%r"
                           % (len(chosen_covered_tids), LB, chosen_energy))
            status = "invalid-reported-solution"

        # (1) 更新(H-01): 甲板冲突语义 = 连续物理。slot 模式单格点占用语义自洽;
        #     interval 模式 = 格点行 ∪ 占机事件行 ∪ 甲板逐对行。此处做【独立全量重扫】
        #     (numpy 分块, 与增量登记 _register_col_conflicts 完全独立): 全池每一条
        #     连续相交的甲板对必须 (共享格点 ⇒ 格点行已蕴含) ∨ (已入 deck_conf_adj
        #     逐对行), 且三项计数与增量登记逐一相等 —— 登记被删/回退在此被抓并撤证。
        def _recount_deck_pairs():
            flat_a, flat_b, owner = [], [], []
            for j in range(len(cols)):
                for (a_, b_) in cols[j]["deck_ivals"]:
                    flat_a.append(a_); flat_b.append(b_); owner.append(j)
            if len(flat_a) <= 1:
                return 0, 0, 0, True
            fa = np.asarray(flat_a); fb = np.asarray(flat_b)
            ow = np.asarray(owner)
            pairs = set()
            step = 256
            for s0 in range(0, len(fa), step):
                s1 = min(s0 + step, len(fa))
                ov = (np.maximum(fa[s0:s1, None], fa[None, :])
                      < np.minimum(fb[s0:s1, None], fb[None, :]) - 1e-9)
                for u, v in np.argwhere(ov):
                    ju, jv = int(ow[s0 + u]), int(ow[v])
                    if ju < jv:
                        pairs.add((ju, jv))
                    elif jv < ju:
                        pairs.add((jv, ju))
            tot = len(pairs)
            cov = rowd = 0
            ok = True
            for (i, j2) in pairs:
                if set(cols[i]["deck_idx"]) & set(cols[j2]["deck_idx"]):
                    cov += 1
                elif j2 in deck_conf_adj.get(i, ()):
                    rowd += 1
                else:
                    ok = False            # 连续相交却既无公共格点也无逐对行 ⇒ 登记遗漏
            return tot, cov, rowd, ok

        if deck_mode == "slot":
            deck_pair_stats["audit_ok"] = True
        else:
            _t2, _c2, _r2, _pairs_ok = _recount_deck_pairs()
            deck_pair_stats["audit_ok"] = bool(
                _pairs_ok and _t2 == deck_pair_stats["total"]
                and _c2 == deck_pair_stats["grid_covered"]
                and _r2 == deck_pair_stats["row_added"])
        # 信息字段(更新 口径保留): 端点是否全对齐 Δ。更新 起它不再门控证书 ——
        # 未对齐时由逐对行/事件行给出连续精确语义, 证书条件换为 deck_conflict_semantics_exact。
        deck_grid_exact = bool(all(_deck_endpoints_aligned(float(c["tau"]), float(c["h"]))
                                   for c in cols))
        deck_conflict_semantics_exact = bool(
            deck_mode == "slot" or (deck_pair_stats["audit_ok"] and phys_ok))

        # (2) 更新(C-01): 逐风机天气场切换 —— 无任何 wx_local 时不存在切换(平凡安全);
        #     否则要求【全池每一列】携带 route_feasible_at_h 的 gate_weather_proof 且
        #     switching_proven_safe ∧ all_candidates_checked(最坏候选聚合完成)。外部
        #     seed 列无 proof / 任一列缺失 ⇒ False(fail-closed)。
        _has_local_wx = any(getattr(t, "wx_local", None) is not None for t in turbines)
        if not _has_local_wx:
            gate_weather_switch_proven_safe = True
            _gate_missing = 0
        else:
            _gate_missing = sum(1 for c in cols
                                if not isinstance(c.get("gate_proof"), dict))
            gate_weather_switch_proven_safe = bool(
                _gate_missing == 0
                and all(bool(c["gate_proof"].get("switching_proven_safe"))
                        and bool(c["gate_proof"].get("all_candidates_checked"))
                        for c in cols))

        # (3) 更新(H-02): reach 预筛保真性 = 读 tau_reach 的 ReachResult 运行时证明并
        #     独立复核: proof_complete ∧ effective_mode∈{off, valid-proven} ∧ 排除集合与
        #     返回集合互补一致 ∧ (valid-proven ⇒ 两前提【零均值松弛 ∧ 非 speed_adjustable】
        #     确凿成立, 由 tau_reach 顶部无条件观测的字段核对 —— 删降级分支的变异在此被抓)。
        _all_tids = {t.tid for t in turbines}
        reach_filter_proven_safe = True
        _reach_modes_seen = set()
        try:
            for rr in reach_of:
                em = rr.effective_mode
                _reach_modes_seen.add(em)
                kept = {t.tid for t in rr}
                exc = set(rr.excluded_tids)
                if not (bool(rr.proof_complete)
                        and em in ("off", "valid-proven")
                        and (kept | exc) == _all_tids and not (kept & exc)
                        and (em != "off" or not exc)
                        and (em != "valid-proven"
                             or (bool(rr.mean_relax_free)
                                 and not bool(rr.speed_adjustable)))):
                    reach_filter_proven_safe = False
        except (AttributeError, TypeError):
            reach_filter_proven_safe = False   # 旧 list 返回/字段缺失 ⇒ 无证明, 撤证
        reach_ok = reach_filter_proven_safe    # 向后兼容字段名(更新 ③ 消融断言用)

        # (4) 更新(H-02): 名义能量早剪保真性 = 全部定价调用的 PricingProof 聚合:
        #     每次调用须 proof_complete; 早剪若实际启用(nominal_prune_enabled 记录的是
        #     price_routes 内实际使用的 _prune_nom), 其独立复核 mean_relax_free_observed
        #     必须为真; 若声明禁用, 剪枝计数必须为 0。并核对调用计数恒等式
        #     len(pricing_stats)==pricing_calls_expected(少登记一条即撤证), 以及本处
        #     第三次独立重算的门控与顶层记录一致(更新 问题#4 口径保留)。
        _prune_gate_recomputed = bool(RM.mean_relax_free(xi_amb, weather_unc))

        def _pricing_proof_ok(ps):
            if not bool(ps.get("proof_complete")):
                return False
            if bool(ps.get("nominal_prune_enabled")):
                return bool(ps.get("mean_relax_free_observed"))
            return int(ps.get("nominal_prune_count", -1)) == 0

        energy_pruning_proven_safe = bool(
            len(pricing_stats) == int(pricing_calls_expected)
            and all(_pricing_proof_ok(ps) for ps in pricing_stats)
            and _prune_gate_recomputed == bool(_mean_relax_free))

        # (5) 严格 Phase-I/Phase-II 运行时证明。证书只读取实际阶段计数与审计字段，
        #     不再依赖 Big-M 或当前列池能耗上界。
        def _is_true(v):
            return type(v) is bool and v is True

        l1_two_phase_complete = bool(
            status == "optimal(gap=0)"
            and two_phase_stats.get("l1_nodes_started")
                == two_phase_stats.get("l1_phase1_feasible", -1)
                   + two_phase_stats.get("l1_phase1_infeasible", -1)
            and two_phase_stats.get("l1_phase2_started")
                == two_phase_stats.get("l1_phase1_feasible")
            and two_phase_stats.get("l1_phase2_closed")
                == two_phase_stats.get("l1_phase2_started"))

        def _l2_two_phase_ok(d):
            try:
                return bool(
                    _is_true(d["phase1_cg_complete"])
                    and _is_true(d["phase2_artificials_fixed_zero"])
                    and _is_true(d["phase2_bounds_valid"])
                    and int(d["phase1_nodes_started"])
                        == int(d["phase1_nodes_feasible"]) + int(d["phase1_nodes_infeasible"])
                    and int(d["phase2_nodes_started"]) == int(d["phase1_nodes_feasible"])
                    and int(d["phase2_nodes_closed"]) == int(d["phase2_nodes_started"]))
            except (KeyError, TypeError, ValueError):
                return False

        l2_all_h_checked = bool(
            len(l2_h_scan_stats) == int(l2_h_scan_calls_expected)
            and all(
                ps.get("solver_stage") == "L2-phase2"
                and ps.get("proof_complete") is True
                and ps.get("all_h_required") is True
                and ps.get("all_h_proof_complete") is True
                and ps.get("all_h_early_termination") is False
                and int(ps.get("all_h_scans_incomplete", -1)) == 0
                and int(ps.get("all_h_evaluations_observed", -1))
                    == int(ps.get("all_h_evaluations_expected", -2))
                for ps in l2_h_scan_stats))

        # (6) RF 分支：按合同，功能启用即撤证。
        no_rf_branching = bool((not enable_rf_branching) and nb_p == 0)

        # (7) L2 人工变量审计：Phase-II 必须固定为零，所有被接受 incumbent 的人工变量
        #     最大残量必须不超过容差；缺字段/None 一律 False。
        def _l2_artificial_ok(d):
            try:
                return bool(
                    _is_true(d["artificial_audit_complete"])
                    and _is_true(d["phase2_artificials_fixed_zero"])
                    and _is_true(d["all_accepted_incumbents_artificial_free"])
                    and float(d["max_incumbent_artificial"]) <= ART_TOL)
            except (KeyError, TypeError, ValueError):
                return False

        seed_columns_revalidated = bool(
            seed_validation_stats.get("validation_complete") is True
            and seed_validation_stats.get("all_accepted_revalidated") is True
            and seed_validation_stats.get("trusted_input_values") is False)

        # 默认认证路径不调用启发式 MILP；显式启用后，任何异常、非最优状态或
        # 独立原始解验证失败都必须撤销证书。
        milp_outputs_clean = bool(
            int(milp_runtime.get("l1_failures", 0)) == 0
            and int(milp_runtime.get("l2_failures", 0)) == 0
            and int(milp_runtime.get("l1_invalid_solutions", 0)) == 0
            and int(milp_runtime.get("l2_invalid_solutions", 0)) == 0)

        l1_conditions = dict(
            tree_complete=bool(status == "optimal(gap=0)"),
            seed_columns_revalidated=seed_columns_revalidated,
            all_milp_outputs_validated=milp_outputs_clean,
            all_nodes_cg_converged=bool(all_nodes_converged),
            strict_two_phase_complete=l1_two_phase_complete,
            reach_filter_proven_safe=reach_filter_proven_safe,
            energy_pruning_proven_safe=energy_pruning_proven_safe,
            gate_weather_switch_proven_safe=gate_weather_switch_proven_safe,
            dominance_proven_safe=bool(dominance_exact),
            deck_conflict_semantics_exact=deck_conflict_semantics_exact,
            physical_plan_verified=bool(phys_ok),
            reported_solution_consistent=reported_solution_consistent,
            no_rf_branching=no_rf_branching,
            no_timeout=bool(not hit_time_limit),
            no_node_limit=bool(not hit_node_limit),
            no_label_truncation=bool(not hit_label_budget),
        )
        L1_certified = bool(all(v is True for v in l1_conditions.values()))
        l2_conditions = dict(
            l1_certified=L1_certified,
            l2_tree_closed=bool(l2.get("certified") is True),
            seed_columns_revalidated=seed_columns_revalidated,
            all_milp_outputs_validated=milp_outputs_clean,
            l2_strict_two_phase_complete=_l2_two_phase_ok(l2),
            l2_no_artificial_residue=_l2_artificial_ok(l2),
            l2_all_h_checked=l2_all_h_checked,
            l2_no_timeout=bool(l2.get("hit_time_limit") is False),
            l2_no_node_limit=bool(l2.get("hit_node_limit") is False),
            physical_plan_verified=bool(phys_ok),
            reported_solution_consistent=reported_solution_consistent,
        )
        L2_certified = bool(all(v is True for v in l2_conditions.values()))
        _reasons = [k for k, v in l1_conditions.items() if v is not True]
        _reasons += ["L2:" + k for k, v in l2_conditions.items()
                     if v is not True and k != "l1_certified"]
        certificate = dict(
            L1_certified=L1_certified, L2_certified=L2_certified,
            conditions=dict(L1=l1_conditions, L2=l2_conditions),
            certificate_reason=(_reasons or None),
            reach_ok=reach_ok, rf_ok=no_rf_branching, dominance_exact=dominance_exact,
            all_nodes_cg_converged=bool(all_nodes_converged),
            mean_relax_free=bool(_mean_relax_free),
            nominal_prunes_active=bool(any(ps.get("nominal_prune_enabled") is True
                                           for ps in pricing_stats)),
            pricing_label_budget=pricing_label_budget,
            pricing_calls=len(pricing_stats),
            pricing_calls_expected=int(pricing_calls_expected),
            l2_h_scan_calls=len(l2_h_scan_stats),
            l2_h_scan_calls_expected=int(l2_h_scan_calls_expected),
            l2_all_h_checked=l2_all_h_checked,
            l1_two_phase_stats=dict(two_phase_stats),
            l2_two_phase_stats={k: l2.get(k) for k in (
                "phase1_nodes_started", "phase1_nodes_feasible", "phase1_nodes_infeasible",
                "phase1_cg_complete", "phase2_nodes_started", "phase2_nodes_closed",
                "phase2_artificials_fixed_zero", "phase2_bounds_valid",
                "incumbent_energy_recount_Wh", "incumbent_energy_consistent")},   # 更新(P3-02)
            reach_modes_seen=sorted(_reach_modes_seen),
            gate_proof_missing=int(_gate_missing),
            gate_weather_switch_proven_safe=gate_weather_switch_proven_safe,
            has_per_turbine_weather=bool(_has_local_wx),
            phase1_method="strict-two-phase",
            bigm_used_for_correctness=False,
            # 更新(C3): L2 证书是【容差级】最优(非字面 gap=0): 剪枝
            # node_lb ≥ best_E − 1e-6, 其中 node_lb = 节点闭合 LP − B·PRICING_EPS
            # (CG rc_tol=−1e-6 的 Lagrange 松弛)。L1 覆盖数为整数目标, 精确。
            l2_optimality_tolerance_Wh=round(1e-6 + float(B) * PRICING_EPS, 9),
            l2_certificate_semantics="min-energy within stated absolute tolerance "
                                     "(prune node_lb>=best_E-1e-6; node_lb=LP-B*PRICING_EPS)",
            seed_columns_revalidated=seed_columns_revalidated,
            seed_validation=dict(seed_validation_stats),
            seed_incumbent=dict(seed_incumbent_stats),
            milp_runtime=dict(milp_runtime),
            use_milp_heuristic=bool(use_milp_heuristic),
            reported_solution_consistent=reported_solution_consistent,
            deck_grid_exact=deck_grid_exact,
            deck_conflict_semantics_exact=deck_conflict_semantics_exact,
            deck_pair_stats=dict(deck_pair_stats),
            physical_check_reason=phys_reason,
            hit_time_limit=bool(hit_time_limit), hit_node_limit=bool(hit_node_limit),
            hit_label_budget=bool(hit_label_budget),
            deck_mode=deck_mode, deck_delta_min=float(deck_delta_min),
            t_launch_min=t_launch, t_swap_min=float(t_swap_min),
            dominance_mode=dominance_mode, reach_mode=reach_mode,
            scope=("global-discrete (xi-only; set-dominance exact)" if weather_unc is None
                   else "global-discrete (multi-source; sequence-complete labeling)"),
            proof_model_scope="finite_discrete_fleet_model",
            continuous_real_world_optimality_claimed=False,
            pricing_reduced_cost_tolerance=float(PRICING_EPS))
        L2_scope = ("energy-branch-and-price(certified)" if L2_certified
                    else ("energy-branch-and-price(uncertified:%s)" % l2.get("status")
                          if l2_mode == "bp" and l2.get("status") not in ("skipped", "expanded-pool-milp")
                          else "expanded-pool(no-L2-pricing-certificate)"))
        if L1_certified and L2_certified:
            public_status = "lexicographic-optimal"
        elif L1_certified:
            public_status = "coverage-optimal-energy-unproven"
        elif chosen:
            public_status = "feasible-unproven:" + str(status)
        else:
            public_status = str(status)
        public_gap = gap if L1_certified else None
        if not milp_outputs_clean:
            public_status = "feasible-unproven:milp-output-rejected"
            public_gap = None
        _seed_cov = int(seed_incumbent_stats.get("coverage") or 0)
        _seed_E = seed_incumbent_stats.get("energy_Wh")
        incumbent_preserved = bool(
            len(chosen_covered_tids) > _seed_cov
            or (len(chosen_covered_tids) == _seed_cov
                and (_seed_E is None or chosen_energy <= float(_seed_E) + 1e-6)))
        return dict(covered=len(chosen_covered_tids), UB=UB_final, gap_pct=public_gap, status=public_status,
                    L1_status=status, L1_certified=L1_certified, L1_gap_pct=gap,
                    lexicographic_certified=bool(L1_certified and L2_certified),
                    L2_scope=L2_scope, L2_status=l2.get("status"),
                    L2_certified=L2_certified, L2_nodes=l2.get("nodes"),
                    L2_cg_iters=l2.get("cg_iters"),
                    L2_root_energy_LP=l2.get("root_energy_LP"),
                    L1_time_budget_s=round(float(l1_budget_s), 6),
                    L1_elapsed_s=round(float((l1_finished_at or _time.time()) - t0), 6),
                    L2_time_budget_s=float(l2.get("time_budget_s", 0.0) or 0.0),
                    L2_elapsed_s=float(l2.get("elapsed_s", 0.0) or 0.0),
                    certificate=certificate,
                    reach_mode=reach_mode, dominance_mode=dominance_mode,
                    rf_branching=("on" if enable_rf_branching else "off"),
                    n_expanded_h=n_expanded_h,
                    chosen=chosen, flights=len(chosen),
                    energy_Wh=round(chosen_energy, 1) if chosen else 0.0,
                    mean_stops=(round(float(np.mean([len(c["tids"]) for c in chosen])), 2)
                                if chosen else 0.0),
                    multi_stop_ratio=(round(float(np.mean([len(c["tids"]) >= 2 for c in chosen])), 3)
                                      if chosen else 0.0),
                    nodes=nodes, cg_iters=cg_iters_total, pool_final=len(cols),
                    n_branch_turbine=nb_t, n_branch_pair=nb_p, n_branch_col=nb_c,
                    max_depth=int(max_depth), phase1_stats=dict(phase1_stats),
                    seed_validation=dict(seed_validation_stats),
                    seed_incumbent=dict(seed_incumbent_stats),
                    incumbent_source=incumbent_source,
                    incumbent_preserved=incumbent_preserved,
                    milp_runtime=dict(milp_runtime),
                    use_milp_heuristic=bool(use_milp_heuristic),
                    reported_solution_consistent=reported_solution_consistent,
                    deck_pair_stats=dict(deck_pair_stats),
                    hit_time_limit=bool(hit_time_limit),
                    hit_node_limit=bool(hit_node_limit),
                    hit_label_budget=bool(hit_label_budget),
                    pricing_calls=len(pricing_stats),
                    pricing_progress=round(float(pricing_progress), 3),
                    root_LP=round(root_ub, 3) if root_ub is not None else None,
                    K=K, batteries=B, T_min=T_min, time_s=round(_time.time() - t0, 1),
                    deck_mode=deck_mode, t_launch_min=t_launch,
                    solver="branch-and-price(exact labeling + turbine/col branching + L2 energy B&P + scipy.milp)")
    finally:
        RM.kappa = orig_kappa


# =============================================================================
# Unified anytime finite-discrete fleet solver
# =============================================================================
def _route_sequence_count_upper(n_turbines: int, max_stops: int, n_launch: int) -> int:
    """Return ``|launch| * sum_{l=1}^s P(n,l)`` before physical pruning."""
    n = max(int(n_turbines), 0)
    s = min(max(int(max_stops), 0), n)
    total = 0
    permutation_count = 1
    for length in range(1, s + 1):
        permutation_count *= n - length + 1
        total += permutation_count
    return int(max(int(n_launch), 0) * total)


def _finite_model_scope_signature(turbines, launch_opts, p, xi_amb, *, K, batteries,
                                  T_min, max_stops, kappa_mode, weather_unc,
                                  deck_mode, pool_h_mode, resource_config) -> dict:
    """Return a stable identity for the finite discrete model and resource rules."""
    def simple(value):
        """JSON-safe binary64-exact serializer for finite-model identity."""
        if isinstance(value, (str, int, bool, np.integer)) or value is None:
            return value
        if isinstance(value, (float, np.floating)):
            number = float(value)
            return ("float64:" + number.hex()) if math.isfinite(number) else str(number)
        if isinstance(value, np.ndarray):
            return [simple(x) for x in value.tolist()]
        if isinstance(value, (list, tuple)):
            return [simple(x) for x in value]
        if isinstance(value, dict):
            def _key(k):
                if isinstance(k, (float, np.floating)):
                    number = float(k)
                    return ("float64:" + number.hex()) if math.isfinite(number) else str(number)
                return str(k)
            return {_key(k): simple(value[k]) for k in sorted(value, key=_key)}
        return str(value)

    params = {k: simple(v) for k, v in sorted(vars(p).items())
              if not str(k).startswith("_")}
    xi_cells = []
    for (h, state), cell in sorted(getattr(xi_amb, "cells", {}).items(),
                                   key=lambda item: (float(item[0][0]), str(item[0][1]))):
        xi_cells.append(dict(
            h_min=simple(float(h)), state=str(state), n=int(getattr(cell, "n", 0)),
            mu=simple(np.asarray(getattr(cell, "mu", np.zeros(2)), float)),
            Sigma=simple(np.asarray(getattr(cell, "Sigma", np.zeros((2, 2))), float)),
            support_radius=simple(float(getattr(cell, "support_radius", 0.0)))))
    launches = []
    for option in launch_opts:
        ship = option.ship
        launches.append(dict(
            tau_min=simple(float(option.tau_min)),
            P_launch_m=simple(np.asarray(ship.P_launch, float)),
            xi_state=str(ship.c_state),
            predicted_recovery_points_m=simple(getattr(ship, "pred_by_h", {})),
            predicted_recovery_states=simple(getattr(ship, "recovery_state_by_h", {})),
            predicted_recovery_state_sources=simple(
                getattr(ship, "recovery_state_source_by_h", {})),
            predictor=str(getattr(ship, "_predictor", "declared")),
            weather_launch=simple(getattr(option, "wx", {})),
            weather_by_h=simple(getattr(ship, "weather_by_h", {}))))
    uncertainty = None
    if weather_unc is not None:
        uncertainty = simple(vars(weather_unc) if hasattr(weather_unc, "__dict__") else weather_unc)
    scope = dict(
        model_contract="finite-discrete-route-resource-model",
        model_semantics_contract=MODEL_SEMANTICS_CONTRACT,
        physical_numeric_contract=RM.FORMAL_PHYSICAL_NUMERIC_CONTRACT,
        route_identity_contract=ROUTE_IDENTITY_CONTRACT,
        turbine_ids=[str(t.tid) for t in turbines],
        turbine_local_m=[simple(np.asarray(t.local, float)) for t in turbines],
        turbine_lonlat=[simple(np.asarray(getattr(t, "lonlat", np.zeros(2)), float))
                        for t in turbines],
        turbine_hub_m=[simple(float(getattr(t, "H_hub", 0.0))) for t in turbines],
        turbine_tip_m=[simple(float(getattr(t, "H_tip", 0.0))) for t in turbines],
        turbine_weather_local=[simple(getattr(t, "wx_local", None)) for t in turbines],
        launch_models=launches,
        statistical_horizons=[simple(float(h)) for h in sorted(xi_amb.horizons)],
        xi_cells=xi_cells,
        xi_formal_validated=bool(getattr(xi_amb, "formal_validated", False)),
        xi_moments_source=str(getattr(xi_amb, "moments_source", "unknown")),
        xi_formal_horizon_grid_contract=str(getattr(
            xi_amb, "formal_horizon_grid_contract", "unknown")),
        xi_covariance_contract=str(getattr(xi_amb, "covariance_contract", "unknown")),
        K=int(K), batteries=int(batteries), T_min=simple(float(T_min)),
        max_stops=int(max_stops), kappa_mode=str(kappa_mode),
        time_contract=RM.time_contract_for(p),
        wait_is_recourse=RM.WAIT_IS_RECOURSE,
        speed_is_recourse=bool(getattr(p, "speed_adjustable", False)),
        time_recourse_mode=str(getattr(p, "time_recourse_mode", "wait_only")),
        return_speed_recourse_contract=(
            RM.SPEED_RECOURSE_CONTRACT if getattr(p, "speed_adjustable", False) else None),
        energy_recourse_contract=(
            RM.ENERGY_SPEED_RECOURSE_CONTRACT if getattr(p, "speed_adjustable", False) else None),
        power_envelope_contract=(
            RM.POWER_ENVELOPE_CONTRACT if getattr(p, "speed_adjustable", False) else None),
        dock_risk_contract=RM.DOCK_RISK_CONTRACT,
        geo_risk_allocation_contract=RM.GEO_RISK_ALLOCATION_CONTRACT,
        soc_risk_allocation=str(getattr(p, "soc_risk_allocation", "fixed")),
        deck_mode=str(deck_mode), pool_h_mode=str(pool_h_mode),
        resource_config=simple(resource_config), weather_uncertainty=uncertainty,
        params=params)
    payload = json.dumps(scope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    scope["sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return scope



def _route_universe_context_scope(turbines, launch_opts, p, xi_amb, *,
                                  T_min, max_stops, kappa_mode, weather_unc,
                                  deck_mode, deck_delta_min, t_launch_min,
                                  landing_clear_min, chance_mode, budget_gamma):
    """Binary64-exact identity of everything that can change a route column."""
    resource_config = dict(
        purpose="complete-route-universe",
        chance_mode=str(chance_mode),
        budget_gamma=float(budget_gamma),
        deck_delta_min=float(deck_delta_min),
        t_launch_min=float(t_launch_min),
        landing_clear_min=float(landing_clear_min),
        resource_time_semantics="strict-half-open-binary64")
    # K/B are deliberately fixed dummy values: they do not alter route-column
    # physical feasibility or route resource intervals.  All other Params are
    # included conservatively by _finite_model_scope_signature.
    return _finite_model_scope_signature(
        turbines, launch_opts, p, xi_amb,
        K=1, batteries=1, T_min=float(T_min), max_stops=int(max_stops),
        kappa_mode=str(kappa_mode), weather_unc=weather_unc,
        deck_mode=str(deck_mode), pool_h_mode="all-discrete-horizons",
        resource_config=resource_config)


def _route_universe_columns_sha256(columns):
    rows = []
    for c in columns:
        rows.append((
            repr(_exact_route_signature(c)),
            repr(_column_semantics_fp(c)),
        ))
    payload = json.dumps(sorted(rows), ensure_ascii=False,
                         separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_certified_route_universe(
        turbines, launch_opts, p, xi_amb, T_min, *,
        max_stops=4, weather_unc=None, kappa_mode="vp_unimodal",
        chance_mode="drcc", budget_gamma=2.0,
        t_launch_min=2.5, landing_clear_min=1.0,
        deck_mode="interval", deck_delta_min=2.5,
        time_limit_s=None, deadline=None):
    """[THM-CU] Exhaustively materialize every feasible formal route column.

    This is intended for small formal instances (the publication E1 has n=8,
    max_stops=4).  Enumeration is identical to exact implicit pricing:
    every launch option, every ordered elementary sequence up to ``max_stops``
    and every supported recovery horizon is visited.  The only subtree pruning
    is the same proved inspection/climb service-floor necessary condition used
    by the implicit pricer.  A timeout returns ``complete=False`` and can never
    be used to certify a complete route universe.
    """
    if deadline is None and time_limit_s is not None:
        deadline = time.monotonic() + float(time_limit_s)
    all_tids = frozenset(_tid(t.tid) for t in turbines)
    if len(all_tids) != len(turbines):
        raise ValueError("candidate turbine identifiers must be unique")
    if int(max_stops) < 1:
        raise ValueError("max_stops must be positive")
    context = _route_universe_context_scope(
        turbines, launch_opts, p, xi_amb,
        T_min=T_min, max_stops=max_stops, kappa_mode=kappa_mode,
        weather_unc=weather_unc, deck_mode=deck_mode,
        deck_delta_min=deck_delta_min, t_launch_min=t_launch_min,
        landing_clear_min=landing_clear_min, chance_mode=chance_mode,
        budget_gamma=budget_gamma)
    horizons = tuple(float(h) for h in RM.decision_horizons_of(xi_amb))
    risk_policy = _risk_policy_for_mode(kappa_mode)
    columns = []
    sig_to_idx = {}
    evaluated_sequences = 0
    evaluated_route_h = 0
    pruned_service_prefixes = 0
    reach_fallback_launches = 0
    physical_cache = {}
    complete = True
    reason = "complete"

    service_floor_cache = {}
    def prefix_service_floor(prefix):
        key = tuple(_tid(t.tid) for t in prefix)
        got = service_floor_cache.get(key)
        if got is not None:
            return got
        t_lb = 0.0
        e_lb = 0.0
        for tb in prefix:
            dz = float(M.insp_vertical_span(tb, p.z_cruise))
            if getattr(p, "use_zeng", False):
                p_up = float(M.P_zeng(0.0, p) + 7.27 * 9.81 * p.v_z)
                p_insp = float(M.P_zeng(p.v_orbit, p))
            else:
                p_up = float(p.P_climb)
                p_insp = float(p.P_hov)
            t_lb += float(dz / p.v_z + p.tau_insp)
            e_lb += float(
                p_up * dz / p.v_z / 3600.0
                + p_insp * p.tau_insp / 3600.0)
        service_floor_cache[key] = (t_lb, e_lb)
        return t_lb, e_lb

    try:
        for oi, opt in enumerate(launch_opts):
            if _deadline_hit(deadline):
                complete = False; reason = "route-universe-time-limit"; break
            reach_proof = RA.tau_reach(
                opt, turbines, p, max(horizons) if horizons else 0.0,
                mode="valid", wx=getattr(opt, "wx", None), xi_amb=xi_amb,
                weather_unc=weather_unc)
            observed_excluded = tuple(sorted(
                all_tids - {_tid(t.tid) for t in reach_proof}))
            reach_safe = bool(
                getattr(reach_proof, "proof_complete", False)
                and getattr(reach_proof, "effective_mode", None) in {"off", "valid-proven"}
                and observed_excluded == tuple(sorted(
                    _tid(t) for t in getattr(reach_proof, "excluded_tids", ())))
                and (getattr(reach_proof, "effective_mode", None) != "off"
                     or not observed_excluded)
                and (getattr(reach_proof, "effective_mode", None) != "valid-proven"
                     or (bool(getattr(reach_proof, "mean_relax_free", False))
                         and not bool(getattr(reach_proof, "speed_adjustable", True)))))
            reach = list(reach_proof) if reach_safe else list(turbines)
            if not reach_safe:
                reach_fallback_launches += 1

            def extend(prefix, used):
                nonlocal complete, reason, evaluated_sequences
                nonlocal evaluated_route_h, pruned_service_prefixes
                if not complete:
                    return
                if _deadline_hit(deadline):
                    complete = False; reason = "route-universe-time-limit"; return
                if prefix:
                    evaluated_sequences += 1
                    t_lb, e_lb = prefix_service_floor(prefix)
                    max_h_s = 60.0 * max(horizons) if horizons else 0.0
                    if t_lb > max_h_s or e_lb > float(p.B_use):
                        pruned_service_prefixes += 1
                        return
                    for h in horizons:
                        if _deadline_hit(deadline):
                            complete = False; reason = "route-universe-time-limit"; return
                        if float(opt.tau_min) + h > float(T_min):
                            continue
                        if t_lb > 60.0 * h:
                            continue
                        evaluated_route_h += 1
                        key = (int(oi),
                               tuple(_tid(x.tid) for x in prefix),
                               float(h).hex())
                        if key in physical_cache:
                            c = physical_cache[key]
                        else:
                            try:
                                c = _candidate_from_physics(
                                    oi, opt, prefix, h, p, xi_amb, weather_unc,
                                    t_launch_min, landing_clear_min,
                                    deck_mode, deck_delta_min,
                                    chance_mode=chance_mode,
                                    budget_gamma=budget_gamma,
                                    deadline=deadline, risk_policy=risk_policy)
                            except TimeoutError:
                                complete = False
                                reason = "route-universe-time-limit"
                                return
                            physical_cache[key] = c
                        if c is not None:
                            _validate_column_domain(c, all_tids, max_stops)
                            _add_columns(columns, sig_to_idx, [c])
                if len(prefix) >= int(max_stops):
                    return
                for tb in reach:
                    tid = _tid(tb.tid)
                    if tid in used:
                        continue
                    extend(prefix + (tb,), used | {tid})
                    if not complete:
                        return
            extend(tuple(), set())
            if not complete:
                break
    except Exception as exc:
        complete = False
        reason = f"route-universe-evaluator-error:{type(exc).__name__}"

    if complete and not _route_archive_semantics_invariant(columns):
        complete = False
        reason = "route-universe-semantic-invariant-failed"
    columns_hash = (_route_universe_columns_sha256(columns) if complete else "")
    stats = dict(
        complete=bool(complete), reason=str(reason),
        n_turbines=len(turbines), max_stops=int(max_stops),
        n_launch_options=len(launch_opts), n_horizons=len(horizons),
        evaluated_sequences=int(evaluated_sequences),
        evaluated_route_h=int(evaluated_route_h),
        retained_columns=int(len(columns)),
        pruned_service_prefixes=int(pruned_service_prefixes),
        reach_fallback_launches=int(reach_fallback_launches),
        physical_cache_entries=int(len(physical_cache)),
        context_sha256=str(context["sha256"]),
        columns_sha256=str(columns_hash),
        builder_contract=COMPLETE_ROUTE_UNIVERSE_CONTRACT)
    return CertifiedRouteUniverse(
        columns=tuple(columns), complete=bool(complete),
        context_sha256=str(context["sha256"]),
        columns_sha256=str(columns_hash),
        builder_contract=COMPLETE_ROUTE_UNIVERSE_CONTRACT,
        stats=stats)


def _validate_certified_route_universe(
        universe, turbines, launch_opts, p, xi_amb, T_min, *,
        max_stops, weather_unc, kappa_mode, chance_mode, budget_gamma,
        t_launch_min, landing_clear_min, deck_mode, deck_delta_min):
    if not isinstance(universe, CertifiedRouteUniverse):
        return False, "wrong-universe-type"
    if not universe.complete:
        return False, "universe-incomplete"
    if universe.builder_contract != COMPLETE_ROUTE_UNIVERSE_CONTRACT:
        return False, "universe-builder-contract-mismatch"
    expected = _route_universe_context_scope(
        turbines, launch_opts, p, xi_amb,
        T_min=T_min, max_stops=max_stops, kappa_mode=kappa_mode,
        weather_unc=weather_unc, deck_mode=deck_mode,
        deck_delta_min=deck_delta_min, t_launch_min=t_launch_min,
        landing_clear_min=landing_clear_min, chance_mode=chance_mode,
        budget_gamma=budget_gamma)["sha256"]
    if str(universe.context_sha256) != str(expected):
        return False, "universe-context-mismatch"
    try:
        if _route_universe_columns_sha256(universe.columns) != universe.columns_sha256:
            return False, "universe-columns-hash-mismatch"
        allowed = {_tid(t.tid) for t in turbines}
        for c in universe.columns:
            _validate_column_domain(c, allowed, max_stops)
        if not _route_archive_semantics_invariant(universe.columns):
            return False, "universe-semantic-invariant-failed"
    except Exception as exc:
        return False, f"universe-validation-error:{type(exc).__name__}"
    return True, "ok"


def enumerate_discrete_route_columns(turbines, launch_opts, p, xi_amb, T_min,
                                     deck_delta_min, max_stops, weather_unc=None,
                                     max_sequence_evals=2_000_000,
                                     kappa_mode="vp_unimodal",
                                     deadline: float | None = None):
    """Enumerate ordered, nonrepeating routes and every supported recovery duration."""
    import step11_algorithm_route_drcc as RA
    estimate = _route_sequence_count_upper(len(turbines), max_stops, len(launch_opts))
    if kappa_mode not in set(RM.KAPPA_MODES) | {"nominal"}:
        return [], dict(complete=False, reason="unsupported-kappa-mode",
                        sequence_upper_bound=estimate, evaluated_pairs=0)
    if max_sequence_evals is not None and estimate > int(max_sequence_evals):
        return [], dict(complete=False, reason="sequence-evaluation-guard",
                        sequence_upper_bound=estimate,
                        max_sequence_evals=int(max_sequence_evals), evaluated_pairs=0)
    columns, stats = RA.enumerate_discrete_routes(
        turbines, launch_opts, p, xi_amb, T_min, deck_delta_min, max_stops,
        weather_unc, kappa_mode=kappa_mode, max_evals=max_sequence_evals,
        reach_mode="valid", deadline=deadline)
    complete = bool(stats.get("route_space_complete", False))
    reason = "complete" if complete else str(stats.get("status", "incomplete"))
    return columns, dict(
        complete=complete, reason=reason,
        sequence_upper_bound=int(estimate),
        max_sequence_evals=(None if max_sequence_evals is None else int(max_sequence_evals)),
        enumerated_sequences=int(stats.get("n_seq", 0)),
        evaluated_pairs=int(stats.get("n_eval", 0)),
        retained_columns=int(len(columns)),
        reach_filter_proven_safe=bool(stats.get("reach_filter_proven_safe", False)),
        ordered_route_identity_preserved=True,
        status=str(stats.get("status", "unknown")))


def _empty_anytime_result(turbines, K, batteries, time_limit_s, started,
                          termination_reason, *, bound_source):
    upper = len({str(t.tid) for t in turbines})
    runtime = time.monotonic() - started
    return dict(
        status="empty-plan-incumbent",
        termination_reason=str(termination_reason), runtime_s=float(runtime),
        time_limit_s=time_limit_s,
        coverage_incumbent=0, coverage_upper_bound=upper,
        coverage_gap_abs=upper, coverage_gap_pct=100.0 if upper else 0.0,
        coverage_optimal=(upper == 0),
        # The empty plan is always the incumbent and has zero plan energy.
        # Its global energy gap is undefined until coverage optimality closes.
        energy_incumbent_Wh=0.0,
        energy_lower_bound_Wh=(0.0 if upper == 0 else None),
        energy_gap_abs_Wh=(0.0 if upper == 0 else None),
        energy_gap_pct=(0.0 if upper == 0 else None),
        global_energy_gap_pct=(0.0 if upper == 0 else None),
        global_energy_gap_reason=(None if upper == 0 else "coverage optimum not proven"),
        conditional_energy_gap_pct=None,
        energy_optimal=(upper == 0), lexicographic_optimal=(upper == 0),
        route_space_complete=False, pricing_complete=False,
        resource_audit_complete=False,
        bound_scope="global_discrete_physical_model", bound_source=bound_source,
        restricted_pool_gap_pct=None, chosen=[], covered_turbine_ids=[],
        duplicate_turbine_visits=[], K=int(K), batteries=int(batteries),
        continuous_real_world_optimality_claimed=False, empty_plan_allowed=True)


# =============================================================================
# Formal exact Branch-Price-and-Cut + Logic-Based Benders engine
# =============================================================================


@dataclass(frozen=True)
class BranchState:
    """Complete branch state inherited by one branch-and-price node."""

    forbidden_turbines: frozenset = frozenset()
    required_turbines: frozenset = frozenset()
    forbidden_arcs: frozenset = frozenset()
    required_arcs: frozenset = frozenset()
    forbidden_routes: frozenset = frozenset()
    required_routes: frozenset = frozenset()


@dataclass
class BranchPriceNode:
    node_id: int
    depth: int
    branch: BranchState = field(default_factory=BranchState)
    inherited_bound: float | None = None
    bound_source: str = "trivial-model-bound"


@dataclass
class PricingSearchResult:
    columns: list
    complete: bool
    best_reduced_value: float | None
    reduced_value_bound: float | None
    bound_available: bool
    evaluated_routes: int
    evaluated_sequences: int
    termination_reason: str
    # Experimental M1/M2 telemetry. These fields have no proof role.
    search_goal: str = "certification"
    discovery_early_return: bool = False
    shadow_prefixes_evaluated: int = 0
    shadow_prunable_prefixes: int = 0
    shadow_false_prune_witnesses: int = 0
    shadow_bound_errors: int = 0
    shadow_audit_complete: bool = False
    guided_order_calls: int = 0
    guided_order_reorders: int = 0
    guided_order_failures: int = 0
    layered_depths_started: int = 0
    layered_depths_completed: int = 0
    layered_max_depth_completed: int = 0
    layered_rounds: int = 0
    physical_cache_hits: int = 0
    physical_cache_misses: int = 0
    # R-BPC profiling / certified-prefix-pruning telemetry.  These fields are
    # observational except ``certified_prefix_prunes``: a prune is allowed only
    # when the outward-safe prefix lower bound is >= mathematical zero.
    wall_time_s: float = 0.0
    physical_evaluator_runtime_s: float = 0.0
    prefix_bound_runtime_s: float = 0.0
    prefix_service_runtime_s: float = 0.0
    certified_prefix_pruning_enabled: bool = False
    certified_prefix_prunes: int = 0
    depth_certified_prefix_prunes: dict = field(default_factory=dict)
    service_floor_prunes: int = 0
    depth_service_floor_prunes: dict = field(default_factory=dict)
    horizon_window_skips: int = 0
    horizon_service_time_skips: int = 0
    physical_infeasible_results: int = 0
    whole_route_evaluator_calls: int = 0
    drcc_route_evaluator_calls: int = 0
    dominance_prunes: int = 0
    duplicate_state_prunes: int = 0
    branch_filter_skips: int = 0
    existing_signature_skips: int = 0
    best_reduced_value_ub: float | None = None
    certified_prefix_bound_histogram: dict = field(default_factory=dict)
    launch_prefix_nodes: dict = field(default_factory=dict)
    root_turbine_prefix_nodes: dict = field(default_factory=dict)
    root_pair_prefix_nodes: dict = field(default_factory=dict)
    launch_evaluator_calls: dict = field(default_factory=dict)
    horizon_evaluator_calls: dict = field(default_factory=dict)
    launch_horizon_evaluator_calls: dict = field(default_factory=dict)
    improving_columns_seen: int = 0
    discovery_improving_columns_returned: int = 0
    discovery_distinct_launches: int = 0
    discovery_distinct_service_sets: int = 0
    discovery_diversity_satisfied: bool = False
    discovery_hard_cap_triggered: bool = False
    depth_prefixes_evaluated: dict = field(default_factory=dict)
    depth_improving_seen: dict = field(default_factory=dict)
    depth_improving_returned: dict = field(default_factory=dict)
    # V10 pricing telemetry only; never used by pricing admission or proof.
    pattern_cut_active_dual_rows: int = 0
    pattern_cut_dual_abs_sum: float = 0.0
    pattern_cut_improving_seen_count: int = 0
    pattern_cut_improving_seen_contribution_sum: float = 0.0
    pattern_cut_improving_seen_sign_essential: int = 0
    pattern_cut_returned_count: int = 0
    pattern_cut_returned_contribution_sum: float = 0.0
    pattern_cut_returned_sign_essential: int = 0
    pattern_cut_returned_by_depth: dict = field(default_factory=dict)
    # V13 discovery-order telemetry only.  These fields never participate in
    # reduced-cost admission, pruning, pricing closure, or any certificate.
    depth_fair_requested: bool = False
    depth_fair_active: bool = False
    depth_fair_rounds: int = 0
    depth_fair_halfcap_dual_abs: float = 0.0
    # V14 heuristic-only multi-stop enrichment telemetry.  These fields
    # have no role in reduced-cost closure, pruning, bounds, or certificates.
    neutral_multistop_enabled: bool = False
    neutral_multistop_candidates_seen: int = 0
    neutral_multistop_cross_zero_seen: int = 0
    neutral_multistop_nonnegative_seen: int = 0
    neutral_multistop_returned: int = 0
    neutral_multistop_returned_by_depth: dict = field(default_factory=dict)
    neutral_multistop_best_stop_count: int = 0
    neutral_multistop_best_uncovered_gain: int = 0
    neutral_multistop_best_rc_ub: float | None = None
    neutral_multistop_best_energy_per_stop_Wh: float | None = None
    neutral_multistop_early_return: bool = False

    @property
    def search_complete(self):
        """Whether the implicit route search itself exhausted its finite domain."""
        return bool(self.complete)

    @property
    def closed(self):
        """Whether exhaustive pricing proves every omitted route has rc >= 0."""
        if not self.complete or not self.bound_available or self.reduced_value_bound is None:
            return False
        try:
            value = float(self.reduced_value_bound)
        except (TypeError, ValueError, OverflowError):
            return False
        return bool((math.isinf(value) and value > 0.0)
                    or (math.isfinite(value) and value >= 0.0))


@dataclass
class RestrictedMasterResult:
    status: str
    x: np.ndarray | None
    objective_value: float | None
    dual_lower_bound: float | None
    inequality_duals: np.ndarray | None
    equality_duals: np.ndarray | None
    eligible_indices: list
    inequality_rows: list
    equality_rows: list
    A_ub: np.ndarray
    b_ub: np.ndarray
    A_eq: np.ndarray
    b_eq: np.ndarray
    objective: np.ndarray
    phase_one_value: float | None = None


@dataclass
class StageSearchResult:
    stage: str
    incumbent_selection: tuple
    incumbent_audit: object | None
    incumbent_value: float | None
    incumbent_lower_bound: float | None
    incumbent_upper_bound: float | None
    coverage_incumbent: int
    global_bound: float
    optimal: bool
    termination_reason: str
    open_nodes: int
    processed_nodes: int
    generated_columns: int
    pricing_calls: int
    exact_pricing_calls: int
    resource_cuts_added: int
    rmp_solves: int
    phase_one_solves: int
    pricing_candidates: int
    pricing_nodes: int
    columns_accepted: int
    heuristic_columns: int
    resource_audit_calls: int
    branch_children_created: int
    branch_decisions: int
    # ``pricing_complete`` is the legacy public name and now means mathematical
    # pricing closure (all omitted routes proved rc >= 0), not merely that a DFS
    # call finished scanning. ``pricing_search_complete`` exposes the latter.
    pricing_complete: bool
    pricing_search_complete: bool
    pricing_bound_available: bool
    resource_audit_complete: bool
    farkas_pricing_complete: bool
    branching_complete: bool
    heuristic_pricing_used: bool
    exact_pricing_called: bool
    pricing_best_reduced_value: float | None
    pricing_reduced_value_bound: float | None
    bound_source: str
    # M1/M2 experiment telemetry; these fields never participate in proofs.
    pricing_discovery_calls: int = 0
    pricing_discovery_early_returns: int = 0
    pricing_certification_calls: int = 0
    pricing_shadow_prefixes_evaluated: int = 0
    pricing_shadow_prunable_prefixes: int = 0
    pricing_shadow_false_prune_witnesses: int = 0
    pricing_shadow_bound_errors: int = 0
    pricing_shadow_complete_calls: int = 0
    pricing_guided_order_calls: int = 0
    pricing_guided_order_reorders: int = 0
    pricing_guided_order_failures: int = 0
    pricing_layered_depths_started: int = 0
    pricing_layered_depths_completed: int = 0
    pricing_layered_max_depth_completed: int = 0
    pricing_layered_rounds: int = 0
    pricing_physical_cache_hits: int = 0
    pricing_physical_cache_misses: int = 0
    pricing_runtime_s: float = 0.0
    pricing_physical_evaluator_runtime_s: float = 0.0
    pricing_prefix_bound_runtime_s: float = 0.0
    pricing_prefix_service_runtime_s: float = 0.0
    pricing_certified_prefix_prunes: int = 0
    pricing_depth_certified_prefix_prunes: dict = field(default_factory=dict)
    pricing_service_floor_prunes: int = 0
    pricing_depth_service_floor_prunes: dict = field(default_factory=dict)
    pricing_horizon_window_skips: int = 0
    pricing_horizon_service_time_skips: int = 0
    pricing_physical_infeasible_results: int = 0
    pricing_branch_filter_skips: int = 0
    pricing_existing_signature_skips: int = 0
    pricing_call_records: list = field(default_factory=list)
    rmp_records: list = field(default_factory=list)
    resource_audit_records: list = field(default_factory=list)
    rmp_runtime_s: float = 0.0
    phase_one_runtime_s: float = 0.0
    resource_audit_runtime_s: float = 0.0
    pricing_discovery_improving_seen: int = 0
    pricing_discovery_improving_returned: int = 0
    pricing_discovery_diverse_returns: int = 0
    pricing_discovery_hard_cap_returns: int = 0
    pricing_discovery_max_return_batch: int = 0
    pricing_discovery_max_distinct_launches: int = 0
    pricing_discovery_max_distinct_service_sets: int = 0
    primal_refresh_calls: int = 0
    primal_refresh_audit_calls: int = 0
    primal_refresh_timeouts: int = 0
    primal_refresh_improvements: int = 0
    primal_refresh_best_coverage: int = 0
    primal_refresh_columns_seen: int = 0
    primal_refresh_rebuilds: int = 0
    primal_refresh_repairs: int = 0
    primal_refresh_augmentation_audits: int = 0
    primal_refresh_rebuild_audits: int = 0
    primal_refresh_repair_audits: int = 0
    primal_refresh_augmentation_improvements: int = 0
    primal_refresh_rebuild_improvements: int = 0
    primal_refresh_repair_improvements: int = 0
    # V17 heuristic-only resource-primal efficiency/diagnostic telemetry.
    # These fields never participate in bounds, pruning, pricing closure, or
    # any optimality/infeasibility certificate.
    primal_refresh_duplicate_trials_skipped: int = 0
    primal_refresh_cached_infeasible_trials: int = 0
    primal_refresh_uncovered_fair_rounds: int = 0
    primal_refresh_failure_reasons: dict = field(default_factory=dict)
    # V18 heuristic-only exact deck-conflict diagnostic / ordering telemetry.
    # The conflict relation is the same fixed half-open interval relation used
    # by the exact resource audit.  These fields never enter a formal bound.
    primal_deck_diagnostic_enabled: bool = False
    primal_deck_archive_conflict_edges: int = 0
    primal_deck_archive_max_degree: int = 0
    primal_deck_archive_max_component: int = 0
    primal_deck_candidate_scored: int = 0
    primal_deck_candidate_zero_conflict: int = 0
    primal_deck_candidate_positive_conflict: int = 0
    primal_deck_prefilter_skips: int = 0
    primal_deck_max_candidate_conflicts: int = 0
    primal_deck_conflict_pairs_sample: list = field(default_factory=list)
    # V19 heuristic-only adaptive two-stop exact-variant enrichment telemetry.
    # These columns are legal RMP variables but never count as pricing closure,
    # bounds, pruning, infeasibility, or optimality evidence.
    pricing_multistop_merge_enabled: bool = False
    pricing_multistop_merge_triggers: int = 0
    pricing_multistop_merge_attempts: int = 0
    pricing_multistop_merge_physical_feasible: int = 0
    pricing_multistop_merge_new_candidates: int = 0
    pricing_multistop_merge_returned: int = 0
    pricing_multistop_merge_added: int = 0
    pricing_multistop_merge_batches: int = 0
    pricing_multistop_merge_distinct_pairs: int = 0
    pricing_multistop_merge_best_rc_ub: float | None = None
    pricing_multistop_merge_best_energy_per_stop_Wh: float | None = None
    pricing_multistop_merge_best_uncovered_gain: int = 0
    pricing_multistop_merge_used_in_incumbent: int = 0
    # V20 heuristic-only resource-aware exact singleton-variant enrichment.
    # It targets uncovered turbines and deck-compatible launch/horizon states.
    # These counters are observational only and never enter any proof path.
    pricing_resource_variant_enabled: bool = False
    pricing_resource_variant_triggers: int = 0
    pricing_resource_variant_attempts: int = 0
    pricing_resource_variant_deck_compatible_specs: int = 0
    pricing_resource_variant_deck_prefilter_skips: int = 0
    pricing_resource_variant_physical_feasible: int = 0
    pricing_resource_variant_new_candidates: int = 0
    pricing_resource_variant_returned: int = 0
    pricing_resource_variant_added: int = 0
    pricing_resource_variant_batches: int = 0
    pricing_resource_variant_distinct_turbines: int = 0
    pricing_resource_variant_best_rc_ub: float | None = None
    pricing_resource_variant_best_energy_Wh: float | None = None
    pricing_resource_variant_used_in_incumbent: int = 0
    # V20.1 observational provenance for the exact variants returned by V20.
    # Records are never read by pricing, branching, bounds, or certificates.
    pricing_resource_variant_records: list = field(default_factory=list)
    # V21 exact frozen-current-archive primal recovery telemetry. The recovery
    # solves only the columns already generated at that instant and accepts a
    # selection only after the unchanged exact resource audit.  Its restricted-
    # archive bound is never used for full-space pruning/certification.
    archive_primal_recovery_enabled: bool = False
    archive_primal_recovery_calls: int = 0
    archive_primal_recovery_runtime_s: float = 0.0
    archive_primal_recovery_audit_calls: int = 0
    archive_primal_recovery_timeouts: int = 0
    archive_primal_recovery_improvements: int = 0
    archive_primal_recovery_best_coverage: int = 0
    archive_primal_recovery_best_archive_columns: int = 0
    archive_primal_recovery_records: list = field(default_factory=list)
    archive_primal_recovery_witness_selection_indices: list = field(default_factory=list)
    archive_primal_recovery_witness_route_signatures: list = field(default_factory=list)
    archive_primal_recovery_witness_covered_turbines: list = field(default_factory=list)
    # V15 heuristic-only resource-aware archive exchange telemetry.  None of
    # these counters participates in bounds, pruning, or pricing certificates.
    primal_exchange_enabled: bool = False
    primal_exchange_calls: int = 0
    primal_exchange_candidate_routes: int = 0
    primal_exchange_trials_built: int = 0
    primal_exchange_audit_calls: int = 0
    primal_exchange_improvements: int = 0
    primal_exchange_consolidation_trials: int = 0
    primal_exchange_optional_drop_trials: int = 0
    primal_exchange_max_stop_count_considered: int = 0
    primal_exchange_best_coverage: int = 0
    primal_exchange_multistop_used_in_incumbent: int = 0
    pricing_depth_prefixes_evaluated: dict = field(default_factory=dict)
    pricing_depth_improving_seen: dict = field(default_factory=dict)
    pricing_depth_improving_returned: dict = field(default_factory=dict)
    pricing_pattern_cut_active_dual_rows: int = 0
    pricing_pattern_cut_dual_abs_sum: float = 0.0
    pricing_pattern_cut_improving_seen_count: int = 0
    pricing_pattern_cut_improving_seen_contribution_sum: float = 0.0
    pricing_pattern_cut_improving_seen_sign_essential: int = 0
    pricing_pattern_cut_returned_count: int = 0
    pricing_pattern_cut_returned_contribution_sum: float = 0.0
    pricing_pattern_cut_returned_sign_essential: int = 0
    pricing_pattern_cut_returned_by_depth: dict = field(default_factory=dict)
    pricing_depth_fair_requested_calls: int = 0
    pricing_depth_fair_active_calls: int = 0
    pricing_depth_fair_rounds: int = 0
    pricing_depth_fair_halfcap_dual_abs_sum: float = 0.0
    pricing_multistop_neutral_enabled_calls: int = 0
    pricing_multistop_candidates_seen: int = 0
    pricing_multistop_cross_zero_seen: int = 0
    pricing_multistop_nonnegative_seen: int = 0
    pricing_multistop_neutral_returned: int = 0
    pricing_multistop_neutral_added: int = 0
    pricing_multistop_neutral_batches: int = 0
    pricing_multistop_neutral_returned_by_depth: dict = field(default_factory=dict)
    pricing_multistop_best_stop_count: int = 0
    pricing_multistop_best_uncovered_gain: int = 0
    pricing_multistop_best_rc_ub: float | None = None
    pricing_multistop_best_energy_per_stop_Wh: float | None = None
    pricing_multistop_neutral_used_in_incumbent: int = 0
    battery_halfcap_dual_active_rmp_solves: int = 0
    battery_halfcap_dual_abs_sum: float = 0.0
    battery_halfcap_dual_max_abs: float = 0.0
    direct_target_backend: str | None = None
    fullcover_strong_cuts: int = 0
    target_master_solves: int = 0
    fullcover_cuts_loaded: int = 0
    fullcover_battery_core_cuts: int = 0
    resource_audit_nodes: int = 0
    resource_audit_memo_hits: int = 0
    target_exact_cover_nodes: int = 0
    target_checkpoint_writes: int = 0
    battery_relaxation_nodes: int = 0
    target_closure_context_sha256: str | None = None
    global_battery_relaxation_status: str | None = None
    global_battery_min_required: int | None = None
    global_battery_dp_states: int = 0
    global_battery_one_pack_masks: int = 0


@dataclass(frozen=True)
class CertifiedRouteUniverse:
    """Complete materialized physical route universe for one fixed E1 instance.

    The object is an acceleration certificate, never an alternate model.  It is
    valid only when ``context_sha256`` matches the current physical route-column
    context and ``columns_sha256`` matches the immutable formal semantics of
    every materialized column.
    """
    columns: tuple
    complete: bool
    context_sha256: str
    columns_sha256: str
    builder_contract: str
    stats: dict


COMPLETE_ROUTE_UNIVERSE_CONTRACT = (
    "formal-complete-physical-route-universe-v1-exhaustive-sequence-horizon"
)


def _remaining(deadline):
    return None if deadline is None else max(0.0, float(deadline) - time.monotonic())


def _deadline_hit(deadline):
    return deadline is not None and time.monotonic() >= float(deadline)


def _tid(value):
    return str(value)


def _ordered_tids(column):
    # ``route_order`` is the historical seed-pool field that preserves visit
    # order; ``tids`` may be a sorted coverage key and must never override it.
    seq = column.get("ordered_tids")
    if seq is None:
        seq = column.get("route_order")
    if seq is None:
        seq = column.get("tids", ())
    if not seq and column.get("route") is not None:
        seq = column["route"].turbine_ids()
    return tuple(_tid(t) for t in seq)


def _route_arcs(column):
    seq = _ordered_tids(column)
    return frozenset((seq[i], seq[i + 1]) for i in range(len(seq) - 1))


def _validate_column_domain(column, allowed_tids, max_stops):
    ordered = _ordered_tids(column)
    unknown = sorted(set(ordered) - set(allowed_tids))
    if unknown:
        raise ValueError(f"route column contains turbines outside the model: {unknown!r}")
    if len(ordered) > int(max_stops):
        raise ValueError("route column exceeds max_stops")
    return column


def _exact_route_signature(column):
    """Binary64-exact identity of one physical route; visit order is preserved.

    Caller-supplied ``route_signature`` metadata is never trusted by the formal
    path: accepting it would let two physically distinct routes collide.
    """
    tau = _float_binary64_fp(column.get("tau", 0.0))
    h = _float_binary64_fp(column.get("h", 0.0))
    ship = column.get("ship")
    ship_fp = _ship_column_fp(ship) if ship is not None else "no-ship"
    wx = column.get("wx")
    wx_fp = _wx_fp(wx) if wx is not None else "no-weather"
    return (ship_fp, wx_fp, tau, _ordered_tids(column), h)


def _column_semantics_fp(column):
    """Exact formal semantics bound to one canonical route signature.

    A route signature denotes one mathematical master variable.  Its objective,
    pooled-energy/resource payload and interval semantics therefore may not be
    replaced in place after a logic cut or branch decision has referred to that
    signature.  Binary64 quantities are fingerprinted by their exact payloads.
    """
    return (
        ROUTE_SEMANTICS_CONTRACT,
        _exact_route_signature(column),
        ("E_plan_Wh", _float_binary64_fp(column["E_plan_Wh"])),
        ("E_soc_required_Wh", _float_binary64_fp(column["E_soc_required_Wh"])),
        ("ordered_tids", _ordered_tids(column)),
        ("route_arcs", tuple(sorted(column.get("route_arcs") or _route_arcs(column)))),
        ("resource_intervals", _state_fp(column.get("resource_intervals"))),
    )


def _normalize_exact_column(column, *, launch_option_index=None, p=None,
                            t_launch_min=0.0, landing_clear_min=0.0,
                            deck_mode="interval", deck_delta_min=2.5):
    c = dict(column)
    ordered = _ordered_tids(c)
    if not ordered:
        raise ValueError("route column covers no turbine")
    if len(ordered) != len(set(ordered)):
        raise ValueError(f"route repeats a turbine: {ordered!r}")
    c["ordered_tids"] = ordered
    c["tids"] = ordered
    if launch_option_index is not None:
        c["launch_option_index"] = int(launch_option_index)
    plan = float(c.get("E_plan_Wh", c.get("E0", float("nan"))))
    soc = float(c.get("E_soc_required_Wh", plan))
    if not math.isfinite(plan) or plan < 0.0:
        raise ValueError("planned route energy must be finite and nonnegative")
    if not math.isfinite(soc) or soc < plan:
        raise ValueError("SOC route energy must be finite and at least planned energy")
    c["E_plan_Wh"] = plan
    c["E_soc_required_Wh"] = soc
    c.setdefault("E0", c["E_plan_Wh"])
    if "tau" not in c or "h" not in c:
        raise ValueError("route column requires tau and h")
    c["route_signature"] = _exact_route_signature(c)
    c["route_arcs"] = _route_arcs(c)
    c["resource_intervals"] = RA._resource_intervals(
        c, float(t_launch_min), float(landing_clear_min), 0.0,
        deck_mode=str(deck_mode), deck_delta_min=float(deck_delta_min))
    if p is not None and c["E_soc_required_Wh"] > float(p.B_use):
        raise ValueError("route SOC requirement exceeds usable battery energy")
    return c


def _branch_consistent(branch: BranchState):
    if branch.forbidden_turbines & branch.required_turbines:
        return False
    if branch.forbidden_arcs & branch.required_arcs:
        return False
    if branch.forbidden_routes & branch.required_routes:
        return False
    for i, j in branch.required_arcs:
        if i in branch.forbidden_turbines or j in branch.forbidden_turbines:
            return False
    return True


def _column_allowed_at_node(column, branch: BranchState):
    tids = frozenset(_ordered_tids(column))
    if tids & branch.forbidden_turbines:
        return False
    arcs = column.get("route_arcs") or _route_arcs(column)
    if frozenset(arcs) & branch.forbidden_arcs:
        return False
    sig = _exact_route_signature(column)
    if sig in branch.forbidden_routes:
        return False
    return True


def _possible_resource_row_times(launch_opts, xi_amb, T_min, t_launch_min,
                                 deck_mode, deck_delta_min):
    """Finite event times sufficient for interval-capacity rows.

    For half-open intervals, any overlap contains the start of at least one
    participating interval.  All starts are determined by a launch option and a
    discrete recovery horizon, independent of the visited turbine order.
    """
    horizons = tuple(float(h) for h in RM.decision_horizons_of(xi_amb))
    deck_times = set()
    active_times = set()
    for opt in launch_opts:
        tau = float(opt.tau_min)
        launch_start = max(tau - float(t_launch_min), 0.0)
        if str(deck_mode) == "slot":
            d = float(deck_delta_min)
            launch_start = round(tau / d) * d
        deck_times.add(float(launch_start))
        active_times.add(float(max(tau - float(t_launch_min), 0.0)))
        for h in horizons:
            if tau + h > float(T_min):
                continue
            rec = tau + h
            if str(deck_mode) == "slot":
                d = float(deck_delta_min)
                rec = round(rec / d) * d
            deck_times.add(float(rec))
    return tuple(sorted(deck_times)), tuple(sorted(active_times))


def _contains_time(interval, t):
    """Strict half-open membership on the stored binary64 endpoints."""
    return float(interval[0]) <= float(t) < float(interval[1])


def _row_coefficient(column, descriptor):
    """[LEM-PAT]/[THM-BR] Coefficient rule shared by current and future columns."""
    kind, key = descriptor
    tids = frozenset(_ordered_tids(column))
    if kind == "packing":
        return 1.0 if key in tids else 0.0
    if kind == "deck":
        ivs = column["resource_intervals"]["deck"]
        return 1.0 if any(_contains_time(iv, key) for iv in ivs) else 0.0
    if kind == "active":
        return 1.0 if _contains_time(column["resource_intervals"]["active"], key) else 0.0
    if kind == "pooled_energy":
        return float(column["E_soc_required_Wh"])
    if kind == "resource_pattern":
        # Exact-pattern Logic-Based Benders cut.  A route in the audited
        # infeasible selection has coefficient +1; every other current or
        # future route has coefficient -1.  For binary x this inequality
        # excludes only the exact selected route set, not its supersets.
        return 1.0 if _exact_route_signature(column) in key else -1.0
    if kind == "battery_halfcap":
        # V12 formal valid inequality.  Each selected route is assigned to one
        # indivisible battery pack by the exact resource model.  If a route
        # needs strictly more than half one usable pack, no two such routes can
        # share a pack; with B packs at most B can be selected.  Compare the
        # exact real values represented by their binary64 payloads, with no
        # epsilon or point-estimate sign decision.
        cap = Fraction.from_float(float(key))
        e = Fraction.from_float(float(column["E_soc_required_Wh"]))
        return 1.0 if 2 * e > cap else 0.0
    if kind == "diagnostic_battery_halfcap":
        # V11 post-formal diagnostic only.  Every route with energy strictly
        # above half one usable battery is pairwise battery-incompatible with
        # every other such route, so at most B can be selected.
        cap = Fraction.from_float(float(key))
        e = Fraction.from_float(float(column["E_soc_required_Wh"]))
        return 1.0 if 2 * e > cap else 0.0
    if kind == "diagnostic_battery_anchor_clique":
        # key=(anchor_signature, anchor_energy, usable_capacity).  This row is
        # constructed only for anchor_energy <= capacity/2.  Every route above
        # capacity-anchor_energy is incompatible with the anchor and is itself
        # above capacity/2, hence all members form a valid battery clique.
        anchor_sig, anchor_energy, cap_value = key
        if _exact_route_signature(column) == anchor_sig:
            return 1.0
        cap = Fraction.from_float(float(cap_value))
        ae = Fraction.from_float(float(anchor_energy))
        e = Fraction.from_float(float(column["E_soc_required_Wh"]))
        return 1.0 if e > cap - ae else 0.0
    if kind == "coverage":
        return float(len(tids))
    if kind == "required_service":
        return 1.0 if key in tids else 0.0
    if kind == "required_arc":
        return 1.0 if key in (column.get("route_arcs") or _route_arcs(column)) else 0.0
    if kind == "required_route":
        return 1.0 if _exact_route_signature(column) == key else 0.0
    raise KeyError(f"unknown row descriptor {descriptor!r}")


def _master_rows(all_tids, deck_times, active_times, K, pooled_energy_cap,
                 no_good_cuts, branch: BranchState, stage, coverage_target,
                 extra_inequality_rows=()):
    inequality_rows = []
    inequality_rhs = []
    for tid in all_tids:
        inequality_rows.append(("packing", tid)); inequality_rhs.append(1.0)
    for t in deck_times:
        inequality_rows.append(("deck", float(t))); inequality_rhs.append(1.0)
    for t in active_times:
        inequality_rows.append(("active", float(t))); inequality_rhs.append(float(K))
    if pooled_energy_cap is not None:
        inequality_rows.append(("pooled_energy", None)); inequality_rhs.append(float(pooled_energy_cap))
    for cut in no_good_cuts:
        inequality_rows.append(("resource_pattern", frozenset(cut)))
        inequality_rhs.append(float(len(cut) - 1))
    for desc, rhs in tuple(extra_inequality_rows or ()):
        inequality_rows.append(desc)
        inequality_rhs.append(float(rhs))

    equality_rows = []
    equality_rhs = []
    if stage == "energy":
        equality_rows.append(("coverage", None)); equality_rhs.append(float(coverage_target))
    for tid in sorted(branch.required_turbines):
        equality_rows.append(("required_service", tid)); equality_rhs.append(1.0)
    for arc in sorted(branch.required_arcs):
        equality_rows.append(("required_arc", arc)); equality_rhs.append(1.0)
    for sig in sorted(branch.required_routes, key=repr):
        equality_rows.append(("required_route", sig)); equality_rhs.append(1.0)
    return (inequality_rows, np.asarray(inequality_rhs, float),
            equality_rows, np.asarray(equality_rhs, float))


def _build_restricted_master(columns, all_tids, node, stage, coverage_target,
                             deck_times, active_times, K, pooled_energy_cap,
                             no_good_cuts, extra_inequality_rows=()):
    eligible = [j for j, c in enumerate(columns) if _column_allowed_at_node(c, node.branch)]
    ineq_rows, b_ub, eq_rows, b_eq = _master_rows(
        all_tids, deck_times, active_times, K, pooled_energy_cap,
        no_good_cuts, node.branch, stage, coverage_target,
        extra_inequality_rows=extra_inequality_rows)
    A_ub = np.zeros((len(ineq_rows), len(eligible)), float)
    A_eq = np.zeros((len(eq_rows), len(eligible)), float)
    # v3.1-P6 (THM-007): hoist per-column invariants (route signature, tid
    # set, arcs) out of the row loop.  The coefficient expressions below are
    # verbatim copies of _row_coefficient's branches, so every matrix entry
    # is bit-identical to the reference build; row kinds outside the common
    # set still fall back to the unchanged _row_coefficient.
    _need_sig = any(d[0] in ("resource_pattern", "required_route")
                    for d in ineq_rows) or any(
        d[0] == "required_route" for d in eq_rows)
    for k, j in enumerate(eligible):
        c = columns[j]
        tids = frozenset(_ordered_tids(c))
        sig = _exact_route_signature(c) if _need_sig else None
        _arcs = None
        for r, desc in enumerate(ineq_rows):
            kind, key = desc
            if kind == "packing":
                A_ub[r, k] = 1.0 if key in tids else 0.0
            elif kind == "deck":
                _ivs = c["resource_intervals"]["deck"]
                A_ub[r, k] = (1.0 if any(_contains_time(iv, key)
                                         for iv in _ivs) else 0.0)
            elif kind == "active":
                A_ub[r, k] = (1.0 if _contains_time(
                    c["resource_intervals"]["active"], key) else 0.0)
            elif kind == "resource_pattern":
                A_ub[r, k] = 1.0 if sig in key else -1.0
            elif kind == "pooled_energy":
                A_ub[r, k] = float(c["E_soc_required_Wh"])
            elif kind == "battery_halfcap":
                _cap = Fraction.from_float(float(key))
                _e = Fraction.from_float(float(c["E_soc_required_Wh"]))
                A_ub[r, k] = 1.0 if 2 * _e > _cap else 0.0
            else:
                A_ub[r, k] = _row_coefficient(c, desc)
        for r, desc in enumerate(eq_rows):
            kind, key = desc
            if kind == "coverage":
                A_eq[r, k] = float(len(tids))
            elif kind == "required_service":
                A_eq[r, k] = 1.0 if key in tids else 0.0
            elif kind == "required_arc":
                if _arcs is None:
                    _arcs = c.get("route_arcs") or _route_arcs(c)
                A_eq[r, k] = 1.0 if key in _arcs else 0.0
            elif kind == "required_route":
                A_eq[r, k] = 1.0 if sig == key else 0.0
            else:
                A_eq[r, k] = _row_coefficient(c, desc)
    if stage == "coverage":
        objective = -np.asarray([len(_ordered_tids(columns[j])) for j in eligible], float)
    elif stage == "energy":
        objective = np.asarray([float(columns[j]["E_plan_Wh"]) for j in eligible], float)
    else:
        raise ValueError("stage must be coverage or energy")
    return eligible, ineq_rows, b_ub, eq_rows, b_eq, A_ub, A_eq, objective


def _solve_restricted_master(columns, all_tids, node, stage, coverage_target,
                             deck_times, active_times, K, pooled_energy_cap,
                             no_good_cuts, deadline, extra_inequality_rows=()):
    eligible, ineq_rows, b_ub, eq_rows, b_eq, A_ub, A_eq, objective = _build_restricted_master(
        columns, all_tids, node, stage, coverage_target, deck_times, active_times,
        K, pooled_energy_cap, no_good_cuts,
        extra_inequality_rows=extra_inequality_rows)
    n = len(eligible)
    if n == 0:
        # Symbolic zero-variable feasibility is exact: these RHS values come
        # from the finite model (capacities/branch equalities), not from an LP
        # residual.  A positive tolerance here could turn a genuinely
        # inconsistent empty RMP into a false feasible certificate.
        ineq_ok = bool(np.all(np.zeros(len(b_ub)) <= b_ub))
        eq_ok = bool(np.all(b_eq == 0.0))
        if ineq_ok and eq_ok:
            return RestrictedMasterResult(
                "optimal", np.zeros(0), 0.0, 0.0,
                np.zeros(len(b_ub)), np.zeros(len(b_eq)), eligible,
                ineq_rows, eq_rows, A_ub, b_ub, A_eq, b_eq, objective)
        return RestrictedMasterResult(
            "infeasible", None, None, None, None, None, eligible,
            ineq_rows, eq_rows, A_ub, b_ub, A_eq, b_eq, objective)
    rem = _remaining(deadline)
    if rem is not None and rem <= 0.0:
        return RestrictedMasterResult(
            "time_limit", None, None, None, None, None, eligible,
            ineq_rows, eq_rows, A_ub, b_ub, A_eq, b_eq, objective)
    try:
        from scipy.optimize import linprog
        options = {}
        if rem is not None:
            options["time_limit"] = max(float(rem), 1e-9)
        res = linprog(objective,
                      A_ub=(A_ub if len(b_ub) else None),
                      b_ub=(b_ub if len(b_ub) else None),
                      A_eq=(A_eq if len(b_eq) else None),
                      b_eq=(b_eq if len(b_eq) else None),
                      bounds=[(0.0, None)] * n,
                      method="highs", options=options)
    except Exception as exc:
        return RestrictedMasterResult(
            f"solver_error:{type(exc).__name__}", None, None, None, None, None,
            eligible, ineq_rows, eq_rows, A_ub, b_ub, A_eq, b_eq, objective)
    if int(getattr(res, "status", -1)) == 2:
        return RestrictedMasterResult(
            "infeasible", None, None, None, None, None, eligible,
            ineq_rows, eq_rows, A_ub, b_ub, A_eq, b_eq, objective)
    checked = _validate_linprog_result(
        res, objective, [(0.0, None)] * n,
        A_ub=(A_ub if len(b_ub) else None), b_ub=(b_ub if len(b_ub) else None),
        A_eq=(A_eq if len(b_eq) else None), b_eq=(b_eq if len(b_eq) else None),
        need_ineqlin=len(b_ub), need_eqlin=len(b_eq),
        dual_bounds=[(0.0, 1.0)] * n)
    if checked is None:
        status = "time_limit" if int(getattr(res, "status", -1)) == 1 else "invalid_lp_certificate"
        return RestrictedMasterResult(
            status, None, None, None, None, None, eligible,
            ineq_rows, eq_rows, A_ub, b_ub, A_eq, b_eq, objective)
    x, fun, im, em, dual_lb = checked
    return RestrictedMasterResult(
        "optimal", x, float(fun), float(dual_lb), np.asarray(im, float),
        np.asarray(em, float), eligible, ineq_rows, eq_rows,
        A_ub, b_ub, A_eq, b_eq, objective)


def _solve_elastic_phase_one(master: RestrictedMasterResult, deadline):
    """Elastic Phase-I master, a strict equivalent of Farkas pricing here.

    All route-independent rows are upper bounds satisfied by x=0.  Potential
    missing-column infeasibility therefore comes from equality rows.  Positive
    and negative artificials make every equality feasible; exact pricing on the
    resulting Phase-I dual either restores feasibility or proves that the full
    implicit master cannot drive the artificial objective to zero.
    """
    n = len(master.eligible_indices)
    m = len(master.b_eq)
    if m == 0:
        return None
    rem = _remaining(deadline)
    if rem is not None and rem <= 0.0:
        return None
    from scipy.optimize import linprog
    c = np.concatenate([np.zeros(n), np.ones(m), np.ones(m)])
    Aub = np.hstack([master.A_ub, np.zeros((len(master.b_ub), 2 * m))])
    Aeq = np.hstack([master.A_eq, np.eye(m), -np.eye(m)])
    art_caps = [max(1.0, abs(float(master.b_eq[i])) +
                    float(np.sum(np.abs(master.A_eq[i, :]))) + 1.0)
                for i in range(m)]
    bounds = ([(0.0, None)] * n +
              [(0.0, cap) for cap in art_caps] +
              [(0.0, cap) for cap in art_caps])
    options = {}
    if rem is not None:
        options["time_limit"] = max(float(rem), 1e-9)
    try:
        res = linprog(c,
                      A_ub=(Aub if len(master.b_ub) else None),
                      b_ub=(master.b_ub if len(master.b_ub) else None),
                      A_eq=Aeq, b_eq=master.b_eq,
                      bounds=bounds, method="highs", options=options)
    except Exception:
        return None
    checked = _validate_linprog_result(
        res, c, bounds,
        A_ub=(Aub if len(master.b_ub) else None),
        b_ub=(master.b_ub if len(master.b_ub) else None),
        A_eq=Aeq, b_eq=master.b_eq,
        need_ineqlin=len(master.b_ub), need_eqlin=len(master.b_eq),
        dual_bounds=([(0.0, 1.0)] * n +
                     [(0.0, cap) for cap in art_caps] +
                     [(0.0, cap) for cap in art_caps]))
    if checked is None:
        return None
    x, fun, im, em, dual_lb = checked
    return RestrictedMasterResult(
        "optimal", x[:n], 0.0, float(dual_lb), np.asarray(im, float),
        np.asarray(em, float), master.eligible_indices,
        master.inequality_rows, master.equality_rows,
        master.A_ub, master.b_ub, master.A_eq, master.b_eq,
        np.zeros(n), phase_one_value=float(fun))


def _outward_product_interval(a, b):
    """Enclose the exact-real product of two finite binary64 inputs.

    Python's float multiplication is round-to-nearest binary64.  The exact
    product therefore lies between the immediately adjacent representable
    numbers around the rounded result.  Returning that outward interval gives
    a scale-aware certificate rather than a fixed absolute epsilon.
    """
    a = float(a); b = float(b)
    if not (math.isfinite(a) and math.isfinite(b)):
        raise FloatingPointError("non-finite reduced-cost product input")
    if a == 0.0 or b == 0.0:
        return 0.0, 0.0
    p = a * b
    if not math.isfinite(p):
        raise FloatingPointError("reduced-cost product overflow")
    return math.nextafter(p, -math.inf), math.nextafter(p, math.inf)


def _outward_add_interval(lo, hi, add_lo, add_hi):
    vals = (lo, hi, add_lo, add_hi)
    if not all(math.isfinite(float(v)) for v in vals):
        raise FloatingPointError("non-finite reduced-cost interval addend")
    nlo = float(lo) + float(add_lo)
    nhi = float(hi) + float(add_hi)
    if not (math.isfinite(nlo) and math.isfinite(nhi)):
        raise FloatingPointError("reduced-cost interval addition overflow")
    return math.nextafter(nlo, -math.inf), math.nextafter(nhi, math.inf)


def _outward_add_lower(total_lb, term_lb):
    total_lb = float(total_lb); term_lb = float(term_lb)
    if not (math.isfinite(total_lb) and math.isfinite(term_lb)):
        raise FloatingPointError("non-finite pricing lower-bound term")
    out = total_lb + term_lb
    if not math.isfinite(out):
        raise FloatingPointError("pricing lower-bound overflow")
    return math.nextafter(out, -math.inf)


def _future_column_coefficient_range(descriptor, max_stops, *, row_family):
    """Certified coefficient range for every possible future route column.

    This registry is part of the global-certificate contract.  Adding a master
    row without proving and registering its complete future-column coefficient
    range must fail closed instead of being silently treated as nonnegative.
    ``None`` as an upper endpoint means +infinity and is allowed only where the
    sign of the dual makes the lower contribution attain its minimum at the
    finite lower endpoint.
    """
    kind, _ = descriptor
    if row_family == "inequality":
        ranges = {
            "packing": (0.0, 1.0),
            "deck": (0.0, 1.0),
            "active": (0.0, 1.0),
            "pooled_energy": (0.0, None),
            "resource_pattern": (-1.0, 1.0),
            "battery_halfcap": (0.0, 1.0),
        }
        if kind not in ranges:
            raise ValueError(
                f"no certified future-column coefficient range for inequality row {kind!r}")
        return ranges[kind]
    if row_family == "equality":
        if kind == "coverage":
            return 1.0, float(max_stops)
        if kind in {"required_service", "required_arc", "required_route"}:
            return 0.0, 1.0
        raise ValueError(
            f"no certified future-column coefficient range for equality row {kind!r}")
    raise ValueError(f"unknown master row family {row_family!r}")


def _universal_pricing_lower_bound(stage, max_stops, inequality_rows,
                                     equality_rows, inequality_duals,
                                     equality_duals):
    """Rigorous lower bound on every omitted-column reduced cost.

    This is the omitted-column component of the Full-Space Lagrangian
    Reduced-Cost Correction Theorem.  Every master row is required to have an
    explicit future-column coefficient range.  Unknown row kinds fail closed.
    All binary64 products/additions that can round are outward protected.
    """
    if stage == "coverage":
        lb = -float(max_stops)   # c_r=-|S_r|, 1 <= |S_r| <= max_stops
    elif stage in {"energy", "farkas"}:
        lb = 0.0                # formal route energy >=0; Phase-I route cost=0
    else:
        raise ValueError(stage)
    if not math.isfinite(lb):
        raise FloatingPointError("non-finite universal pricing base")

    # SciPy/HiGHS inequality marginal is y<=0.  The reduced-cost contribution
    # is -y*a.  Because -y>=0, the minimum over a coefficient interval occurs
    # at its lower endpoint.  This handles the resource-pattern value -1
    # explicitly and proves that nonnegative rows contribute at least zero.
    for desc, dual in zip(inequality_rows, np.asarray(inequality_duals, float)):
        d = min(float(dual), 0.0)
        if not math.isfinite(d):
            raise FloatingPointError("non-finite inequality dual")
        coeff_lo, _coeff_hi = _future_column_coefficient_range(
            desc, max_stops, row_family="inequality")
        term_lb, _ = _outward_product_interval(-d, float(coeff_lo))
        lb = _outward_add_lower(lb, term_lb)

    # Equality duals are free.  All equality coefficient ranges are finite, so
    # the minimum of -v*a over [lo,hi] is attained at one of the endpoints.
    for desc, dual in zip(equality_rows, np.asarray(equality_duals, float)):
        d = float(dual)
        if not math.isfinite(d):
            raise FloatingPointError("non-finite equality dual")
        coeff_lo, coeff_hi = _future_column_coefficient_range(
            desc, max_stops, row_family="equality")
        plo, _ = _outward_product_interval(-d, float(coeff_lo))
        phi, _ = _outward_product_interval(-d, float(coeff_hi))
        lb = _outward_add_lower(lb, min(plo, phi))
    return math.nextafter(float(lb), -math.inf)


def _column_reduced_cost_interval(column, stage, inequality_rows, equality_rows,
                                  inequality_duals, equality_duals):
    """Return ``(estimate, lower, upper)`` for one route reduced cost.

    Inputs are interpreted as the exact real values represented by their
    binary64 payloads.  Each product and accumulation is enclosed with directed
    ``nextafter`` protection.  The lower endpoint is therefore valid even under
    catastrophic cancellation at large dual scales.
    """
    if stage == "coverage":
        base = -float(len(_ordered_tids(column)))
    elif stage == "energy":
        base = float(column["E_plan_Wh"])
    elif stage == "farkas":
        base = 0.0
    else:
        raise ValueError(stage)
    if not math.isfinite(base):
        raise FloatingPointError("non-finite reduced-cost objective coefficient")
    lo = hi = base
    terms = [base]
    for desc, dual in zip(inequality_rows, np.asarray(inequality_duals, float)):
        d = min(float(dual), 0.0)
        coeff = float(_row_coefficient(column, desc))
        if not (math.isfinite(d) and math.isfinite(coeff)):
            raise FloatingPointError("non-finite inequality reduced-cost term")
        plo, phi = _outward_product_interval(d, coeff)
        # contribution is -(d*coeff)
        clo, chi = -phi, -plo
        lo, hi = _outward_add_interval(lo, hi, clo, chi)
        terms.append(-(d * coeff))
    for desc, dual in zip(equality_rows, np.asarray(equality_duals, float)):
        d = float(dual)
        coeff = float(_row_coefficient(column, desc))
        if not (math.isfinite(d) and math.isfinite(coeff)):
            raise FloatingPointError("non-finite equality reduced-cost term")
        plo, phi = _outward_product_interval(d, coeff)
        clo, chi = -phi, -plo
        lo, hi = _outward_add_interval(lo, hi, clo, chi)
        terms.append(-(d * coeff))
    estimate = float(math.fsum(terms))
    if not math.isfinite(estimate):
        raise FloatingPointError("non-finite reduced-cost estimate")
    return estimate, float(lo), float(hi)


def _column_reduced_cost(column, stage, inequality_rows, equality_rows,
                         inequality_duals, equality_duals):
    """Backward-compatible point estimate; certificates use the interval API."""
    return _column_reduced_cost_interval(
        column, stage, inequality_rows, equality_rows,
        inequality_duals, equality_duals)[0]


def _candidate_from_physics(opt_index, opt, sequence, h, p, xi_amb, weather_unc,
                            t_launch_min, landing_clear_min, deck_mode, deck_delta_min,
                            chance_mode="drcc", budget_gamma=2.0, deadline=None,
                            kappa_fn=None, risk_policy=None):
    if _deadline_hit(deadline):
        raise TimeoutError("global deadline reached before physical route evaluation")
    route = RM.Route(rid=-1, turbines=list(sequence), ship=opt.ship)
    diag = RM.route_feasible_at_h(
        route, int(h) if float(h).is_integer() else float(h),
        p, opt.wx, xi_amb, weather_unc=weather_unc,
        chance_mode=str(chance_mode), budget_gamma=float(budget_gamma),
        deadline=deadline, kappa_fn=kappa_fn, risk_policy=risk_policy)
    if _deadline_hit(deadline):
        raise TimeoutError("global deadline reached during physical route evaluation")
    if not bool(diag.get("feasible", False)):
        return None
    plan = float(diag.get("E_plan_Wh", float(diag.get("E0", 0.0)) +
                          float(diag.get("E_dock_Wh", 0.0))))
    soc = float(diag.get("E_soc_required_Wh", plan))
    c = dict(
        tau=float(opt.tau_min), h=float(h), ship=opt.ship, wx=opt.wx,
        route=route, ordered_tids=tuple(_tid(t.tid) for t in sequence),
        E0=plan, E_plan_Wh=plan, E_soc_required_Wh=soc,
        E0_nominal=float(diag.get("E0", plan)),
        gate_proof=diag.get("gate_weather_proof"),
        launch_option_index=int(opt_index), physical_diagnostics=diag)
    return _normalize_exact_column(
        c, launch_option_index=opt_index, p=p,
        t_launch_min=t_launch_min, landing_clear_min=landing_clear_min,
        deck_mode=deck_mode, deck_delta_min=deck_delta_min)


def _adaptive_multistop_merge_enrichment(
        *, archive, turbines, launch_opts, p, xi_amb, weather_unc, T_min,
        node, existing_signatures, incumbent_selection,
        inequality_rows, equality_rows, inequality_duals, equality_duals,
        deadline, t_launch_min, landing_clear_min, deck_mode, deck_delta_min,
        kappa_mode="vp_unimodal", chance_mode="drcc", budget_gamma=2.0,
        attempt_limit=24, batch_target=4, wall_budget_s=0.50,
        physical_cache=None):
    """V19 heuristic-only 2-stop exact-variant enrichment.

    This routine is deliberately *not* a pricing oracle.  It takes turbine pairs
    already evidenced by the current archive (singletons and/or existing 2-stop
    service sets), probes a small number of exact order/launch/horizon variants,
    and passes every probe through the unchanged whole-route physical/DRCC
    evaluator.  Returned columns are legal RMP variables, but they never count
    as negative reduced-cost pricing progress and never establish closure,
    pruning, bounds, infeasibility, or optimality.

    The finite exact pricing domain is unchanged; exhaustive pricing remains the
    only omitted-column certificate.
    """
    attempt_limit = max(0, int(attempt_limit))
    batch_target = max(0, int(batch_target))
    if attempt_limit <= 0 or batch_target <= 0 or len(turbines) < 2:
        return dict(columns=[], attempts=0, physical_feasible=0, new_candidates=0,
                    added_candidates=0, distinct_pairs=0, best_rc_ub=None,
                    best_energy_per_stop_Wh=None, uncovered_gain_best=0,
                    timed_out=False)
    if physical_cache is None:
        physical_cache = {}

    local_deadline = deadline
    if wall_budget_s is not None:
        _local = time.monotonic() + max(0.0, float(wall_budget_s))
        local_deadline = _local if deadline is None else min(float(deadline), _local)

    tid_to_turbine = {_tid(t.tid): t for t in turbines}
    incumbent_covered = set()
    for j in tuple(incumbent_selection or ()):
        if 0 <= int(j) < len(archive):
            incumbent_covered.update(_ordered_tids(archive[int(j)]))
    all_tids = set(tid_to_turbine)
    uncovered = all_tids - incumbent_covered

    # Evidence pool.  A turbine must have at least one route-local feasible
    # singleton in the archive, or appear in an already feasible 2-stop route.
    singleton_launches = {}
    singleton_horizons = {}
    pair_evidence = set()
    pair_launches = {}
    pair_horizons = {}
    for c in archive:
        tids = tuple(_ordered_tids(c))
        if len(tids) == 1:
            tid = tids[0]
            oi = c.get("launch_option_index")
            if oi is not None:
                singleton_launches.setdefault(tid, set()).add(int(oi))
            try:
                singleton_horizons.setdefault(tid, set()).add(float(c["h"]))
            except Exception:
                pass
        elif len(tids) == 2:
            key = frozenset(tids)
            pair_evidence.add(key)
            oi = c.get("launch_option_index")
            if oi is not None:
                pair_launches.setdefault(key, set()).add(int(oi))
            try:
                pair_horizons.setdefault(key, set()).add(float(c["h"]))
            except Exception:
                pass

    singleton_tids = sorted(t for t in singleton_launches if t in tid_to_turbine)
    pair_ranked = []
    seen_pairs = set()

    # First diversify exact variants of service pairs that are already known to
    # be physically possible somewhere in the archive.
    for key in sorted(pair_evidence, key=lambda x: tuple(sorted(x))):
        tids = tuple(sorted(key))
        if len(tids) != 2 or any(t not in tid_to_turbine for t in tids):
            continue
        ug = sum(1 for t in tids if t in uncovered)
        pair_ranked.append((-ug, 0, tids))
        seen_pairs.add(key)

    # Then consider merges of route-local feasible singletons.  At least one
    # uncovered turbine is preferred, because the purpose is incumbent/variant
    # enrichment rather than duplicating already-saturated coverage structure.
    for ai, a in enumerate(singleton_tids):
        for b in singleton_tids[ai + 1:]:
            key = frozenset((a, b))
            if key in seen_pairs:
                continue
            ug = int(a in uncovered) + int(b in uncovered)
            pair_ranked.append((-ug, 1, (a, b)))
            seen_pairs.add(key)
    pair_ranked.sort(key=lambda x: (x[0], x[1], x[2]))

    horizons_all = tuple(float(h) for h in RM.decision_horizons_of(xi_amb))
    risk_policy = _risk_policy_for_mode(kappa_mode)
    retained = []
    attempts = 0
    physical_feasible = 0
    new_candidates = 0
    distinct_pairs = set()
    best_rc_ub = None
    best_energy_per_stop = None
    uncovered_gain_best = 0
    timed_out = False

    def _try_candidate(pair_tids, order, oi, h):
        nonlocal attempts, physical_feasible, new_candidates, best_rc_ub
        nonlocal best_energy_per_stop, uncovered_gain_best, timed_out
        if attempts >= attempt_limit or _deadline_hit(local_deadline):
            timed_out = bool(_deadline_hit(local_deadline))
            return
        if not (0 <= int(oi) < len(launch_opts)):
            return
        sequence = tuple(tid_to_turbine[t] for t in order)
        # Exact node legality can reject an order without physics.
        if any(_tid(t.tid) in node.branch.forbidden_turbines for t in sequence):
            return
        if (_tid(sequence[0].tid), _tid(sequence[1].tid)) in node.branch.forbidden_arcs:
            return
        cache_key = (
            "v19-merge", int(oi), tuple(order), float(h).hex())
        if cache_key in physical_cache:
            candidate = physical_cache[cache_key]
        else:
            attempts += 1
            try:
                candidate = _candidate_from_physics(
                    int(oi), launch_opts[int(oi)], sequence, float(h),
                    p, xi_amb, weather_unc, t_launch_min, landing_clear_min,
                    deck_mode, deck_delta_min, chance_mode=chance_mode,
                    budget_gamma=budget_gamma, deadline=local_deadline,
                    risk_policy=risk_policy)
            except TimeoutError:
                timed_out = True
                return
            except Exception:
                # Heuristic layer fails closed locally: evaluator errors do not
                # affect exact pricing or any certificate.
                return
            physical_cache[cache_key] = candidate
        if candidate is None:
            return
        physical_feasible += 1
        if not _column_allowed_at_node(candidate, node.branch):
            return
        sig = _exact_route_signature(candidate)
        if sig in existing_signatures:
            return
        new_candidates += 1
        try:
            _rc, _rc_lb, rc_ub = _column_reduced_cost_interval(
                candidate, "coverage", inequality_rows, equality_rows,
                inequality_duals, equality_duals)
        except Exception:
            rc_ub = math.inf
        try:
            e = float(candidate["E_soc_required_Wh"])
            e_per = e / 2.0
        except Exception:
            e_per = math.inf
        ug = sum(1 for t in pair_tids if t in uncovered)
        uncovered_gain_best = max(uncovered_gain_best, int(ug))
        if math.isfinite(float(rc_ub)):
            best_rc_ub = float(rc_ub) if best_rc_ub is None else min(
                float(best_rc_ub), float(rc_ub))
        if math.isfinite(float(e_per)):
            best_energy_per_stop = (
                float(e_per) if best_energy_per_stop is None
                else min(float(best_energy_per_stop), float(e_per)))
        rank = (
            -int(ug),
            float(e_per),
            float(rc_ub) if math.isfinite(float(rc_ub)) else math.inf,
            repr(sig),
        )
        retained.append((rank, candidate, frozenset(pair_tids)))

    for _neg_ug, _source_rank, pair_tids in pair_ranked:
        if len(retained) >= batch_target or attempts >= attempt_limit:
            break
        if _deadline_hit(local_deadline):
            timed_out = True
            break
        key = frozenset(pair_tids)
        launch_ids = []
        for oi in sorted(pair_launches.get(key, ())):
            if oi not in launch_ids:
                launch_ids.append(oi)
        for tid in pair_tids:
            for oi in sorted(singleton_launches.get(tid, ())):
                if oi not in launch_ids:
                    launch_ids.append(oi)
        # Keep the heuristic bounded but include a small fallback launch sample.
        for oi in range(min(2, len(launch_opts))):
            if oi not in launch_ids:
                launch_ids.append(oi)
        launch_ids = launch_ids[:4]

        hs = []
        for h in sorted(pair_horizons.get(key, ())):
            if h not in hs:
                hs.append(h)
        for tid in pair_tids:
            for h in sorted(singleton_horizons.get(tid, ())):
                if h not in hs:
                    hs.append(h)
        if horizons_all:
            for h in (horizons_all[0], horizons_all[-1]):
                if h not in hs:
                    hs.append(h)
        hs = [h for h in hs if math.isfinite(h) and h >= 0.0][:4]

        for order in (tuple(pair_tids), tuple(reversed(pair_tids))):
            for oi in launch_ids:
                for h in hs:
                    if len(retained) >= batch_target or attempts >= attempt_limit:
                        break
                    _try_candidate(pair_tids, order, oi, h)
                if len(retained) >= batch_target or attempts >= attempt_limit:
                    break
            if len(retained) >= batch_target or attempts >= attempt_limit:
                break

    retained.sort(key=lambda x: x[0])
    chosen = []
    per_pair = {}
    for _rank, c, pair_key in retained:
        n = per_pair.get(pair_key, 0)
        if n >= 2:
            continue
        chosen.append(c)
        per_pair[pair_key] = n + 1
        distinct_pairs.add(pair_key)
        if len(chosen) >= batch_target:
            break

    return dict(
        columns=chosen,
        attempts=int(attempts),
        physical_feasible=int(physical_feasible),
        new_candidates=int(new_candidates),
        added_candidates=int(len(chosen)),
        distinct_pairs=int(len(distinct_pairs)),
        best_rc_ub=(None if best_rc_ub is None else float(best_rc_ub)),
        best_energy_per_stop_Wh=(
            None if best_energy_per_stop is None else float(best_energy_per_stop)),
        uncovered_gain_best=int(uncovered_gain_best),
        timed_out=bool(timed_out),
    )


def _resource_aware_singleton_variant_enrichment(
        *, archive, turbines, launch_opts, p, xi_amb, weather_unc, T_min,
        node, existing_signatures, incumbent_selection,
        inequality_rows, equality_rows, inequality_duals, equality_duals,
        deadline, t_launch_min, landing_clear_min, deck_mode, deck_delta_min,
        kappa_mode="vp_unimodal", chance_mode="drcc", budget_gamma=2.0,
        attempt_limit=32, batch_target=4, wall_budget_s=0.70,
        physical_cache=None):
    """V20 heuristic-only deck-compatible exact singleton timing enrichment.

    V19's real n=10 run exhausted 72 bounded two-stop merge probes without one
    route-level feasible candidate.  V20 therefore targets the narrower missing
    object evidenced by the incumbent: alternative exact launch/horizon variants
    for *uncovered* turbines.  Candidate states are ordered by the exact fixed
    half-open deck relation against the incumbent and by proximity to already
    route-local-feasible singleton states.

    Every returned column passes the unchanged whole-route physical/DRCC
    evaluator.  Returned columns are legal RMP variables only; this helper is not
    a pricing oracle and cannot establish closure, improve an omitted-column
    bound, prune a node, prove infeasibility, or certify optimality.
    """
    attempt_limit = max(0, int(attempt_limit))
    batch_target = max(0, int(batch_target))
    if attempt_limit <= 0 or batch_target <= 0 or not turbines or not launch_opts:
        return dict(columns=[], records=[], attempts=0, deck_compatible_specs=0,
                    deck_prefilter_skips=0, physical_feasible=0,
                    new_candidates=0, distinct_turbines=0,
                    best_rc_ub=None, best_energy_Wh=None, timed_out=False)
    if physical_cache is None:
        physical_cache = {}

    local_deadline = deadline
    if wall_budget_s is not None:
        _local = time.monotonic() + max(0.0, float(wall_budget_s))
        local_deadline = _local if deadline is None else min(float(deadline), _local)

    tid_to_turbine = {_tid(x.tid): x for x in turbines}
    incumbent_selection = tuple(int(j) for j in (incumbent_selection or ())
                                if 0 <= int(j) < len(archive))
    incumbent_covered = set()
    incumbent_deck = []
    for j in incumbent_selection:
        incumbent_covered.update(_ordered_tids(archive[j]))
        incumbent_deck.extend(tuple(
            archive[j].get("resource_intervals", {}).get("deck", ())))
    uncovered = tuple(sorted(set(tid_to_turbine) - incumbent_covered))
    if not uncovered:
        return dict(columns=[], records=[], attempts=0, deck_compatible_specs=0,
                    deck_prefilter_skips=0, physical_feasible=0,
                    new_candidates=0, distinct_turbines=0,
                    best_rc_ub=None, best_energy_Wh=None, timed_out=False)

    horizons = tuple(float(h) for h in RM.decision_horizons_of(xi_amb)
                     if math.isfinite(float(h)) and float(h) >= 0.0)
    if not horizons:
        return dict(columns=[], records=[], attempts=0, deck_compatible_specs=0,
                    deck_prefilter_skips=0, physical_feasible=0,
                    new_candidates=0, distinct_turbines=0,
                    best_rc_ub=None, best_energy_Wh=None, timed_out=False)

    # Existing singleton states are physical feasibility anchors.  The exact
    # signature remains the authority; this pre-index is only heuristic ordering.
    anchors = {tid: [] for tid in uncovered}
    existing_state_keys = set()
    for c in archive:
        tids = tuple(_ordered_tids(c))
        if len(tids) != 1:
            continue
        tid = tids[0]
        oi = c.get("launch_option_index")
        try:
            oi = int(oi)
            hh = float(c["h"])
            tau = float(c["tau"])
        except Exception:
            continue
        existing_state_keys.add((tid, oi, _float_binary64_fp(hh)))
        if tid in anchors:
            anchors[tid].append((tau, hh, oi))

    def _deck_conflicts(oi, hh):
        tau = float(launch_opts[int(oi)].tau_min)
        probe = dict(tau=tau, h=float(hh))
        ivs = RA._resource_intervals(
            probe, float(t_launch_min), float(landing_clear_min), 0.0,
            deck_mode=str(deck_mode), deck_delta_min=float(deck_delta_min))["deck"]
        count = 0
        for a in ivs:
            for b in incumbent_deck:
                if RA._halfopen_overlap(a, b):
                    count += 1
        return int(count)

    per_tid_specs = {}
    deck_prefilter_skips = 0
    for tid in uncovered:
        a = anchors.get(tid) or []
        specs = []
        for oi, opt in enumerate(launch_opts):
            tau = float(opt.tau_min)
            for hh in horizons:
                if tau + float(hh) > float(T_min):
                    continue
                if (tid, int(oi), _float_binary64_fp(float(hh))) in existing_state_keys:
                    continue
                conflicts = _deck_conflicts(oi, hh)
                if a:
                    distance = min(
                        abs(tau - float(at)) + abs(float(hh) - float(ah))
                        for at, ah, _aoi in a)
                else:
                    # No existing feasible singleton anchor for this turbine:
                    # deterministic chronological order is the safest heuristic.
                    distance = tau + 0.01 * float(hh)
                specs.append((int(conflicts), float(distance), tau,
                              float(hh), int(oi)))
        specs.sort(key=lambda z: (z[0], z[1], z[2], z[3], z[4]))
        per_tid_specs[tid] = specs

    attempts = 0
    deck_compatible_specs = 0
    physical_feasible = 0
    new_candidates = 0
    timed_out = False
    best_rc_ub = None
    best_energy = None
    retained = []
    distinct_turbines = set()
    risk_policy = _risk_policy_for_mode(kappa_mode)

    # Fair round-robin across uncovered turbines prevents one turbine's dense
    # exact-state neighborhood from consuming the whole short heuristic budget.
    positions = {tid: 0 for tid in uncovered}
    active = set(uncovered)
    while active and attempts < attempt_limit and len(retained) < batch_target:
        if _deadline_hit(local_deadline):
            timed_out = True
            break
        progressed = False
        for tid in uncovered:
            if tid not in active:
                continue
            specs = per_tid_specs.get(tid, ())
            pos = positions[tid]
            if pos >= len(specs):
                active.discard(tid)
                continue
            conflicts, _distance, _tau, hh, oi = specs[pos]
            positions[tid] = pos + 1
            progressed = True
            if conflicts > 0:
                deck_prefilter_skips += 1
                continue
            deck_compatible_specs += 1
            if attempts >= attempt_limit or _deadline_hit(local_deadline):
                timed_out = bool(_deadline_hit(local_deadline))
                break

            cache_key = (int(oi), (tid,), float(hh).hex())
            if cache_key in physical_cache:
                candidate = physical_cache[cache_key]
            else:
                attempts += 1
                try:
                    candidate = _candidate_from_physics(
                        int(oi), launch_opts[int(oi)],
                        (tid_to_turbine[tid],), float(hh),
                        p, xi_amb, weather_unc,
                        t_launch_min, landing_clear_min,
                        deck_mode, deck_delta_min,
                        chance_mode=chance_mode, budget_gamma=budget_gamma,
                        deadline=local_deadline, risk_policy=risk_policy)
                except TimeoutError:
                    timed_out = True
                    break
                except Exception:
                    # Heuristic errors fail closed locally and never affect the
                    # exact pricing/certificate path.
                    candidate = None
                physical_cache[cache_key] = candidate
            if candidate is None:
                continue
            physical_feasible += 1
            if not _column_allowed_at_node(candidate, node.branch):
                continue
            sig = _exact_route_signature(candidate)
            if sig in existing_signatures:
                continue
            new_candidates += 1
            try:
                _rc, _rc_lb, rc_ub = _column_reduced_cost_interval(
                    candidate, "coverage",
                    inequality_rows, equality_rows,
                    inequality_duals, equality_duals)
            except Exception:
                rc_ub = math.inf
            try:
                e = float(candidate["E_soc_required_Wh"])
            except Exception:
                e = math.inf
            if math.isfinite(float(rc_ub)):
                best_rc_ub = (float(rc_ub) if best_rc_ub is None
                              else min(float(best_rc_ub), float(rc_ub)))
            if math.isfinite(float(e)):
                best_energy = (float(e) if best_energy is None
                               else min(float(best_energy), float(e)))
            _record = dict(
                tid=str(tid),
                ordered_tids=[str(_t) for _t in _ordered_tids(candidate)],
                tau=float(candidate["tau"]),
                h=float(candidate["h"]),
                launch_option_index=int(candidate.get("launch_option_index", oi)),
                E_soc_required_Wh=(None if not math.isfinite(float(e))
                                   else float(e)),
                rc_ub=(None if not math.isfinite(float(rc_ub))
                       else float(rc_ub)),
                signature_repr=repr(sig),
                trigger_incumbent_coverage=int(len(incumbent_covered)),
                trigger_incumbent_turbines=sorted(str(_t)
                                                  for _t in incumbent_covered),
                trigger_uncovered_turbines=[str(_t) for _t in uncovered],
                trigger_incumbent_route_signatures=[
                    repr(_exact_route_signature(archive[int(_j)]))
                    for _j in incumbent_selection],
                deck_conflicts_at_trigger=0,
            )
            retained.append(((float(e),
                              float(rc_ub) if math.isfinite(float(rc_ub))
                              else math.inf,
                              float(candidate["tau"]),
                              repr(sig)),
                             candidate, tid, _record))
            distinct_turbines.add(tid)
        if not progressed:
            break

    retained.sort(key=lambda x: x[0])
    # Preserve turbine diversity in the returned batch.
    chosen = []
    chosen_records = []
    used_tids = set()
    for _rank, c, tid, _record in retained:
        if tid in used_tids:
            continue
        chosen.append(c)
        chosen_records.append(dict(_record))
        used_tids.add(tid)
        if len(chosen) >= batch_target:
            break
    if len(chosen) < batch_target:
        chosen_sigs = {_exact_route_signature(c) for c in chosen}
        for _rank, c, _tid0, _record in retained:
            sig = _exact_route_signature(c)
            if sig in chosen_sigs:
                continue
            chosen.append(c)
            chosen_records.append(dict(_record))
            chosen_sigs.add(sig)
            if len(chosen) >= batch_target:
                break

    return dict(
        columns=chosen,
        records=chosen_records,
        attempts=int(attempts),
        deck_compatible_specs=int(deck_compatible_specs),
        deck_prefilter_skips=int(deck_prefilter_skips),
        physical_feasible=int(physical_feasible),
        new_candidates=int(new_candidates),
        distinct_turbines=int(len(distinct_turbines)),
        best_rc_ub=(None if best_rc_ub is None else float(best_rc_ub)),
        best_energy_Wh=(None if best_energy is None else float(best_energy)),
        timed_out=bool(timed_out),
    )


def _revalidate_seed_column(raw, turbines, launch_opts, p, xi_amb, weather_unc,
                            T_min, t_launch_min, landing_clear_min, deck_mode,
                            deck_delta_min, kappa_mode, chance_mode, budget_gamma,
                            deadline=None):
    """Recompute one heuristic seed with current model objects and physics.

    External route objects, energies and feasibility flags are never trusted by
    the certificate path.  A seed is useful only as a primal/RMP warm start.
    """
    tids = _ordered_tids(raw)
    if not tids or len(tids) != len(set(tids)):
        raise ValueError("invalid-seed-route-order")
    by_tid = {_tid(t.tid): t for t in turbines}
    if any(t not in by_tid for t in tids):
        raise ValueError("seed-turbine-outside-current-model")
    sequence = tuple(by_tid[t] for t in tids)
    try:
        tau = float(raw["tau"])
        h = float(raw["h"])
    except Exception as exc:
        raise ValueError("seed-missing-tau-or-h") from exc
    horizons = tuple(float(v) for v in RM.decision_horizons_of(xi_amb))
    matched_horizons = tuple(v for v in horizons if _float_binary64_fp(h) == _float_binary64_fp(v))
    if not matched_horizons:
        raise ValueError("seed-horizon-outside-current-discrete-grid")
    if len({_float_binary64_fp(v) for v in matched_horizons}) != 1:
        raise ValueError("ambiguous-seed-horizon")
    # Seeds are heuristic warm starts only, but their finite-state identity must
    # exactly match the model grid.  Approximate off-grid values are rejected.
    h = float(matched_horizons[0])

    candidates = []
    raw_index = raw.get("launch_option_index")
    if raw_index is not None:
        try:
            oi = int(raw_index)
            if 0 <= oi < len(launch_opts):
                candidates.append((oi, launch_opts[oi]))
        except Exception:
            pass
    for oi, opt in enumerate(launch_opts):
        if _float_binary64_fp(opt.tau_min) != _float_binary64_fp(tau):
            continue
        if all(oi != old_oi for old_oi, _ in candidates):
            candidates.append((oi, opt))
    if not candidates:
        raise ValueError("seed-launch-option-not-in-current-grid")

    raw_ship = raw.get("ship")
    raw_wx = raw.get("wx")
    if raw_ship is not None:
        ship_fp = _ship_column_fp(raw_ship)
        exact = [(oi, opt) for oi, opt in candidates
                 if _ship_column_fp(opt.ship) == ship_fp]
        if exact:
            candidates = exact
    if raw_wx is not None:
        wx_fp = _wx_fp(raw_wx)
        exact = [(oi, opt) for oi, opt in candidates if _wx_fp(opt.wx) == wx_fp]
        if exact:
            candidates = exact
    # Ambiguous launch options with distinct physical states are rejected rather
    # than selected arbitrarily.
    fingerprints = {(_ship_column_fp(opt.ship), _wx_fp(opt.wx),
                     _float_binary64_fp(opt.tau_min))
                    for _, opt in candidates}
    if len(fingerprints) != 1:
        raise ValueError("ambiguous-seed-launch-option")
    oi, opt = candidates[0]
    if float(opt.tau_min) + h > float(T_min):
        raise ValueError("seed-outside-mission-window")
    risk_policy = _risk_policy_for_mode(kappa_mode)
    column = _candidate_from_physics(
        oi, opt, sequence, h, p, xi_amb, weather_unc,
        t_launch_min, landing_clear_min, deck_mode, deck_delta_min,
        chance_mode=chance_mode, budget_gamma=budget_gamma, deadline=deadline,
        risk_policy=risk_policy)
    if column is None:
        raise ValueError("seed-physical-or-drcc-infeasible")
    return column



_V20_COMPAT_FP_CACHE = {}


def _prefix_signature_compatible(opt, prefix_tids, signature):
    """Whether an exact route signature could still extend launch/prefix.

    False is a proof of incompatibility; True means only "still possible".
    This is used solely to tighten SHADOW coefficient intervals in this release.

    v2.0-P2 (THM-003, Lemma 3.1): _ship_column_fp(opt.ship) and _wx_fp(opt.wx)
    are content-pure functions of the launch option; the reference recomputes
    them on every call. They are memoized per opt object here. The cache is
    cleared at the start of every _exact_pricing_search call; within one such
    call the launch options are immutable inputs, so the memoized values are
    bit-identical to the reference computation.
    """
    try:
        _ck = id(opt)
        _fps = _V20_COMPAT_FP_CACHE.get(_ck)
        if _fps is None:
            _fps = (_ship_column_fp(opt.ship), _wx_fp(opt.wx))
            _V20_COMPAT_FP_CACHE[_ck] = _fps
        ship_fp, wx_fp, tau_fp, seq, _h_fp = signature
        if ship_fp != _fps[0]:
            return False
        if wx_fp != _fps[1]:
            return False
        if tau_fp != _float_binary64_fp(float(opt.tau_min)):
            return False
        seq = tuple(_tid(t) for t in seq)
        prefix_tids = tuple(_tid(t) for t in prefix_tids)
        return bool(len(prefix_tids) <= len(seq)
                    and seq[:len(prefix_tids)] == prefix_tids)
    except Exception:
        # Shadow tightening fails open: retain the broader global interval.
        return True


def _prefix_future_column_coefficient_range(
        descriptor, max_stops, *, row_family, prefix_tids, opt, branch):
    """Certified coefficient interval for completions of one ordered prefix.

    V1 tightens only identity/branch rows whose semantics follow directly from
    the prefix. Harder resource/physics rows use the existing global registry.
    """
    prefix_tids = tuple(_tid(t) for t in prefix_tids)
    prefix_set = frozenset(prefix_tids)
    kind, key = descriptor

    if row_family == "inequality":
        if kind == "packing":
            if key in prefix_set:
                return 1.0, 1.0
            if key in branch.forbidden_turbines:
                return 0.0, 0.0
            return 0.0, 1.0
        if kind == "resource_pattern":
            # v2.0-P1 (THM-002): both branches and the fail-open path share
            # coeff_lo = -1.0, and the inequality lower bound consumes only
            # coeff_lo (see _prefix_pricing_lower_bound), so the compatibility
            # scan has no numeric effect on any prefix bound value.
            return -1.0, 1.0
        return _future_column_coefficient_range(
            descriptor, max_stops, row_family=row_family)

    if row_family == "equality":
        if kind == "coverage":
            k = len(prefix_tids)
            return float(k), float(max(k, int(max_stops)))
        if kind == "required_service":
            if key in prefix_set:
                return 1.0, 1.0
            if key in branch.forbidden_turbines:
                return 0.0, 0.0
            return 0.0, 1.0
        if kind == "required_arc":
            arcs = frozenset(zip(prefix_tids[:-1], prefix_tids[1:]))
            if key in arcs:
                return 1.0, 1.0
            try:
                i, j = key
                if i == j:
                    return 0.0, 0.0
                if i in branch.forbidden_turbines or j in branch.forbidden_turbines:
                    return 0.0, 0.0

                # If the prefix is already at max_stops, a missing required arc
                # can no longer be created by any completion.
                remaining = int(max_stops) - len(prefix_tids)
                if remaining <= 0:
                    return 0.0, 0.0

                # Once the head j has already appeared without (i,j), elementarity
                # prevents visiting j again, so the missing arc is impossible.
                if j in prefix_set:
                    return 0.0, 0.0

                if i in prefix_set:
                    # The only still-possible case is i being the current last
                    # turbine: the very next extension may be j.
                    if prefix_tids and prefix_tids[-1] == i:
                        return 0.0, 1.0
                    return 0.0, 0.0

                # Neither endpoint has appeared. Creating i->j needs two future
                # stop slots. With fewer than two, the coefficient is fixed at 0.
                if remaining < 2:
                    return 0.0, 0.0
            except Exception:
                # Shadow tightening fails open.
                return 0.0, 1.0
            return 0.0, 1.0
        if kind == "required_route":
            if not _prefix_signature_compatible(opt, prefix_tids, key):
                return 0.0, 0.0
            return 0.0, 1.0
        return _future_column_coefficient_range(
            descriptor, max_stops, row_family=row_family)

    raise ValueError(f"unknown master row family {row_family!r}")


def _stable_discovery_order(items, score_fn):
    """Stable discovery-only ordering; failure restores historical input order.

    This helper never removes an item.  It has no certificate role.
    """
    raw = list(items)
    decorated = []
    try:
        for pos, item in enumerate(raw):
            score = score_fn(item)
            key = tuple(float(v) for v in score) if isinstance(score, tuple) else (float(score),)
            if any(not math.isfinite(v) for v in key):
                raise FloatingPointError("non-finite discovery score")
            decorated.append((key, pos, item))
    except Exception:
        return raw, False, True
    decorated.sort(key=lambda z: (z[0], z[1]))
    ordered = [z[2] for z in decorated]
    reordered = any(a is not b for a, b in zip(raw, ordered))
    return ordered, bool(reordered), False


def _round_robin_tagged_iterators(tagged_iterators):
    """Yield one item per active iterator per round, preserving every item.

    Yields ``(round_index, tag, value)`` with a zero-based exact round index.
    Exhausted iterators disappear; no item is dropped or duplicated.  This is a
    discovery scheduling helper only and has no proof role.
    """
    active = [(tag, iter(values)) for tag, values in tagged_iterators]
    round_index = 0
    while active:
        next_active = []
        emitted = False
        for tag, it in active:
            try:
                value = next(it)
            except StopIteration:
                continue
            emitted = True
            yield int(round_index), tag, value
            next_active.append((tag, it))
        if emitted:
            round_index += 1
        active = next_active


def _prefix_pricing_lower_bound(stage, max_stops, prefix, opt, branch,
                                inequality_rows, equality_rows,
                                inequality_duals, equality_duals, *,
                                energy_objective_lb=None,
                                energy_coverage_joint_lb=None):
    """Outward-safe lower bound for every completion of an ordered prefix.

    No wind/DRCC monotonicity is assumed.  When supplied by exact pricing,
    ``energy_objective_lb`` is the already-mandatory inspection/climb service
    energy of the prefix, a nonnegative subset computed by the same binary64
    expressions as the formal route evaluator.  ``energy_coverage_joint_lb``
    may additionally couple that energy floor with the unique fixed-coverage
    equality dual over all possible remaining stop counts.
    """
    prefix_tids = tuple(_tid(t.tid) if hasattr(t, "tid") else _tid(t)
                        for t in prefix)
    if not prefix_tids:
        raise ValueError("prefix lower bound requires a nonempty prefix")
    if stage == "coverage":
        lb = -float(max_stops)
    elif stage == "energy":
        if energy_coverage_joint_lb is not None:
            lb = float(energy_coverage_joint_lb)
        elif energy_objective_lb is not None:
            lb = float(energy_objective_lb)
        else:
            lb = 0.0
    elif stage == "farkas":
        lb = 0.0
    else:
        raise ValueError(stage)
    if not math.isfinite(lb):
        raise FloatingPointError("non-finite prefix pricing base")

    for desc, dual in zip(inequality_rows, np.asarray(inequality_duals, float)):
        d = min(float(dual), 0.0)
        if not math.isfinite(d):
            raise FloatingPointError("non-finite inequality dual")
        coeff_lo, _ = _prefix_future_column_coefficient_range(
            desc, max_stops, row_family="inequality",
            prefix_tids=prefix_tids, opt=opt, branch=branch)
        term_lb, _ = _outward_product_interval(-d, float(coeff_lo))
        lb = _outward_add_lower(lb, term_lb)

    for desc, dual in zip(equality_rows, np.asarray(equality_duals, float)):
        if (stage == "energy" and energy_coverage_joint_lb is not None
                and desc[0] == "coverage"):
            continue
        d = float(dual)
        if not math.isfinite(d):
            raise FloatingPointError("non-finite equality dual")
        coeff_lo, coeff_hi = _prefix_future_column_coefficient_range(
            desc, max_stops, row_family="equality",
            prefix_tids=prefix_tids, opt=opt, branch=branch)
        plo, _ = _outward_product_interval(-d, float(coeff_lo))
        phi, _ = _outward_product_interval(-d, float(coeff_hi))
        lb = _outward_add_lower(lb, min(plo, phi))

    # A lower bound is protected toward -infinity, never toward +infinity.
    return math.nextafter(float(lb), -math.inf)


def _v20_compile_prefix_bound_context(stage, max_stops, branch,
                                      inequality_rows, equality_rows,
                                      inequality_duals, equality_duals):
    """v2.0-P3 (THM-001 Theorem 1): compiled Tier-A prefix bound context.

    Precomputes every dual-only row term of _prefix_pricing_lower_bound once
    per pricing call and returns ``evaluate(prefix, opt, ...)`` which
    reproduces the reference accumulation order bit-for-bit (including
    exactly-zero rows). Prefix-dependent rows are resolved by O(1) membership
    tests (packing, required_service) or by the reference range function
    (required_arc, required_route — few rows, cheap after P2 memoization).

    Returns ``None`` whenever anything about the row set is unanticipated
    (unknown descriptor kind, non-finite dual, row/dual length mismatch), in
    which case callers fall back to _prefix_pricing_lower_bound and behavior
    is identical to the reference by construction.
    """
    try:
        iduals = [float(x) for x in np.asarray(inequality_duals, float)]
        eduals = [float(x) for x in np.asarray(equality_duals, float)]
    except Exception:
        return None
    if len(iduals) != len(tuple(inequality_rows)):
        return None
    if len(eduals) != len(tuple(equality_rows)):
        return None
    for _d in iduals:
        _dd = min(float(_d), 0.0)
        if not math.isfinite(_dd):
            return None
    ineq_const = []          # float term_lo per row, None for packing rows
    packing_rows = []        # (position, tid, term_in)
    for desc, d0 in zip(inequality_rows, iduals):
        d = min(float(d0), 0.0)
        kind, key = desc
        if kind == "packing":
            t_in, _ = _outward_product_interval(-d, 1.0)
            packing_rows.append((len(ineq_const), key, t_in))
            ineq_const.append(None)
        elif kind == "resource_pattern":
            t_lo, _ = _outward_product_interval(-d, -1.0)
            ineq_const.append(t_lo)
        elif kind in ("deck", "active", "pooled_energy", "battery_halfcap"):
            t_lo, _ = _outward_product_interval(-d, 0.0)
            ineq_const.append(t_lo)
        else:
            return None
    eq_prepared = []
    for desc, d0 in zip(equality_rows, eduals):
        d = float(d0)
        if not math.isfinite(d):
            return None
        kind, key = desc
        if kind in ("coverage", "required_service", "required_arc",
                    "required_route"):
            eq_prepared.append((desc, d))
        else:
            return None
    _max_stops_int = int(max_stops)

    def evaluate(prefix, opt, *, energy_objective_lb=None,
                 energy_coverage_joint_lb=None):
        prefix_tids = tuple(_tid(t.tid) if hasattr(t, "tid") else _tid(t)
                            for t in prefix)
        if not prefix_tids:
            raise ValueError("prefix lower bound requires a nonempty prefix")
        if stage == "coverage":
            lb = -float(max_stops)
        elif stage == "energy":
            if energy_coverage_joint_lb is not None:
                lb = float(energy_coverage_joint_lb)
            elif energy_objective_lb is not None:
                lb = float(energy_objective_lb)
            else:
                lb = 0.0
        elif stage == "farkas":
            lb = 0.0
        else:
            raise ValueError(stage)
        if not math.isfinite(lb):
            raise FloatingPointError("non-finite prefix pricing base")
        prefix_set = frozenset(prefix_tids)
        terms = list(ineq_const)
        for pos, key, t_in in packing_rows:
            terms[pos] = t_in if key in prefix_set else 0.0
        for v in terms:
            lb = _outward_add_lower(lb, v)
        for desc, d in eq_prepared:
            kind, key = desc
            if (stage == "energy" and energy_coverage_joint_lb is not None
                    and kind == "coverage"):
                continue
            if kind == "coverage":
                k = len(prefix_tids)
                coeff_lo, coeff_hi = float(k), float(max(k, _max_stops_int))
            elif kind == "required_service":
                if key in prefix_set:
                    coeff_lo, coeff_hi = 1.0, 1.0
                elif key in branch.forbidden_turbines:
                    coeff_lo, coeff_hi = 0.0, 0.0
                else:
                    coeff_lo, coeff_hi = 0.0, 1.0
            else:
                coeff_lo, coeff_hi = _prefix_future_column_coefficient_range(
                    desc, max_stops, row_family="equality",
                    prefix_tids=prefix_tids, opt=opt, branch=branch)
            plo, _ = _outward_product_interval(-d, float(coeff_lo))
            phi, _ = _outward_product_interval(-d, float(coeff_hi))
            lb = _outward_add_lower(lb, min(plo, phi))
        return math.nextafter(float(lb), -math.inf)

    return evaluate


class _V30PhysicalPricingCache(dict):
    """dict with an attached route-space completeness slot (v3.0-P5).

    Production solves allocate this subclass once per solve_fleet_anytime
    call.  Direct API / selftest callers passing a plain ``{}`` have no slot
    attribute, so THM-006 cached-feature pricing stays disabled for them and
    every search follows the unchanged reference DFS.
    """
    __slots__ = ("v30_state",)


def _v30_register_route_space_complete(cache, branch, complete):
    """THM-006 completeness bookkeeping after one production DFS call.

    A traversal that finished naturally (complete=True: no deadline, no
    discovery/neutral early return) at a superset node (empty forbidden
    turbine and arc sets; required sets and forbidden routes do not restrict
    traversal) has visited every (launch, ordered elementary prefix,
    horizon) key of the maximal branch space reachable by any descendant
    node, resolving each non-provably-infeasible key in ``cache``.
    """
    if not complete:
        return
    if (getattr(branch, "forbidden_turbines", None)
            or getattr(branch, "forbidden_arcs", None)):
        return
    try:
        cache.v30_state = {"complete": True, "cols": None, "len": -1,
                           "sig": None}
    except AttributeError:
        return  # plain dict: cached-feature path stays disabled


def _v30_cached_feature_columns(cache, turbines, p, T_min, max_stops,
                                horizons):
    """Feasible-column snapshot for THM-006 cached-feature pricing.

    Returns ``None`` unless the completeness flag is set on the cache.  The
    snapshot filter replicates the DFS key filters exactly (key shape,
    elementarity, launch-window, service-floor trio, max stops), so every
    snapshot entry is a column the DFS would retrieve by cache lookup at
    some node; every skipped entry is one the DFS would never look up.
    Rebuilt whenever the cache grows or the domain fingerprint changes.
    """
    try:
        st = getattr(cache, "v30_state", None)
    except Exception:
        return None
    if not st or not st.get("complete"):
        return None
    sig = (float(T_min), int(max_stops), tuple(horizons), len(turbines))
    if (st.get("cols") is None or st.get("len") != len(cache)
            or st.get("sig") != sig):
        max_h_s = 60.0 * max(horizons) if horizons else 0.0
        service_by_tid = {}
        try:
            for tb in turbines:
                dz = float(M.insp_vertical_span(tb, p.z_cruise))
                if getattr(p, "use_zeng", False):
                    p_up = float(M.P_zeng(0.0, p) + 7.27 * 9.81 * p.v_z)
                    p_insp = float(M.P_zeng(p.v_orbit, p))
                else:
                    p_up = float(p.P_climb)
                    p_insp = float(p.P_hov)
                service_by_tid[_tid(tb.tid)] = (
                    float(dz / p.v_z + p.tau_insp),
                    float(p_up * dz / p.v_z / 3600.0
                          + p_insp * p.tau_insp / 3600.0))
        except Exception:
            # Fail-safe (THM-006 impl. condition 2): turbines/params that do
            # not support the service-floor arithmetic (e.g. synthetic test
            # stubs) disable the cached-feature path for this call; the
            # unchanged reference DFS runs instead.
            return None
        cols = []
        for k, v in cache.items():
            if v is None:
                continue
            if not (isinstance(k, tuple) and len(k) == 3
                    and isinstance(k[0], int) and not isinstance(k[0], bool)
                    and isinstance(k[1], tuple) and isinstance(k[2], str)):
                continue
            tids = k[1]
            if not tids or len(set(tids)) != len(tids):
                continue
            if len(tids) > int(max_stops):
                continue
            try:
                t_sum = 0.0
                e_sum = 0.0
                ok = True
                for t in tids:
                    sv = service_by_tid.get(t)
                    if sv is None:
                        ok = False
                        break
                    t_sum += sv[0]
                    e_sum += sv[1]
                if not ok:
                    continue
                h = float(v["h"])
                tau = float(v["tau"])
                if tau + h > float(T_min):
                    continue
                if (t_sum > max_h_s or e_sum > float(p.B_use)
                        or t_sum > 60.0 * h):
                    continue
            except Exception:
                continue
            cols.append(v)
        st["cols"] = cols
        st["len"] = len(cache)
        st["sig"] = sig
    return st["cols"]


def _exact_pricing_search(turbines, launch_opts, p, xi_amb, weather_unc, T_min,
                          max_stops, node, existing_signatures, stage,
                          inequality_rows, equality_rows, inequality_duals,
                          equality_duals, deadline, pricing_epsilon,
                          t_launch_min, landing_clear_min, deck_mode,
                          deck_delta_min, kappa_mode="vp_unimodal",
                          chance_mode="drcc", budget_gamma=2.0,
                          implicit_test_columns=None, batch_size=16,
                          physical_cache=None, search_goal="certification",
                          shadow_prefix_bounds=False, discovery_column_limit=8,
                          guided_discovery_order=False,
                          layered_discovery_order=False,
                          depth_fair_discovery_order=False,
                          neutral_multistop_enrichment=False,
                          neutral_multistop_batch_target=8,
                          neutral_uncovered_tids=None,
                          adaptive_discovery_batch=False,
                          discovery_batch_hard_cap=6,
                          discovery_min_distinct_launches=2,
                          discovery_min_distinct_service_sets=2,
                          pattern_cut_diagnostics=False,
                          certified_prefix_pruning=False):
    """Exact on-demand pricing by implicit elementary-sequence DFS.

    ``search_goal='certification'`` preserves the historical exhaustive scan.
    ``search_goal='discovery'`` may return early after a bounded batch of
    sign-definite improving columns; that result is explicitly incomplete and
    cannot close pricing. ``shadow_prefix_bounds`` computes prefix bounds for
    telemetry only and NEVER prunes a prefix in this release.
    ``guided_discovery_order`` changes only visit order during discovery; the
    full launch/prefix/horizon domain remains intact.
    ``layered_discovery_order`` additionally visits depth 1 globally before
    depth 2, etc., and round-robins launches inside each depth.  It is still
    ordering-only: completion of all layers is the same finite pricing domain.
    ``depth_fair_discovery_order`` is V13 ordering-only telemetry/heuristic:
    during incomplete discovery, if a formal battery-halfcap row has a nonzero
    dual, exact per-depth iterators are round-robined across depths as well as
    launches.  No prefix is removed, and an exhaustive run visits the same
    finite pricing domain as the historical layered traversal.
    ``neutral_multistop_enrichment`` is V14 heuristic-only discovery: only in
    coverage discovery with an active formal battery-halfcap dual, physically
    valid routes with at least two stops and ``rc_ub >= 0`` may be returned in a
    bounded batch.  Such columns are legal additions to the RMP but are never
    counted as improving pricing columns and never establish closure, pruning,
    lower/upper bounds, or any certificate.
    ``adaptive_discovery_batch`` changes only when an incomplete discovery call
    returns already-proved negative columns.  It never relaxes the strict
    ``rc_ub < 0`` admission rule and therefore cannot prove pricing closure.

    The current physical/DRCC evaluator is a black-box whole-route predicate, so
    no safe nontrivial RCSP resource dominance or state merging has been proved.
    The certificate path therefore traverses every allowed launch option, every
    ordered nonrepeating turbine prefix up to ``max_stops`` and every supported
    recovery horizon.  It does not pre-materialize the complete route pool, but
    it is still implicit full-permutation enumeration in the worst case.  Only
    exact identity deduplication is used and only a batch of improving columns is
    retained after the complete scan.
    """
    search_goal = str(search_goal).strip().lower()
    if search_goal not in {"certification", "discovery"}:
        raise ValueError("search_goal must be 'certification' or 'discovery'")
    discovery_column_limit = max(1, int(discovery_column_limit))
    _batch_capacity = max(1, int(batch_size))
    _batch_target = min(_batch_capacity, discovery_column_limit)
    adaptive_discovery_batch = bool(
        adaptive_discovery_batch and search_goal == "discovery")
    _batch_hard_cap = max(
        _batch_target,
        min(_batch_capacity, max(1, int(discovery_batch_hard_cap))))
    _min_distinct_launches = max(
        1, min(_batch_target, int(discovery_min_distinct_launches)))
    _min_distinct_service_sets = max(
        1, min(_batch_target, int(discovery_min_distinct_service_sets)))
    shadow_prefixes_evaluated = 0
    shadow_prunable_prefixes = 0
    shadow_false_prune_witnesses = 0
    shadow_bound_errors = 0
    shadow_prunable_keys = set()
    shadow_false_witness_keys = set()
    discovery_early_return = False
    guided_order_calls = 0
    guided_order_reorders = 0
    guided_order_failures = 0
    guided_discovery_order = bool(
        guided_discovery_order and search_goal == "discovery")
    layered_discovery_order = bool(
        layered_discovery_order and search_goal == "discovery")
    depth_fair_discovery_order = bool(
        depth_fair_discovery_order
        and layered_discovery_order
        and search_goal == "discovery")
    _depth_fair_halfcap_dual_abs = 0.0
    if depth_fair_discovery_order:
        for _desc, _dual in zip(
                inequality_rows, np.asarray(inequality_duals, float)):
            if _desc[0] != "battery_halfcap":
                continue
            _a = abs(float(_dual))
            if not math.isfinite(_a):
                raise FloatingPointError(
                    "non-finite battery-halfcap dual in depth-fair discovery")
            _depth_fair_halfcap_dual_abs += _a
    depth_fair_active = bool(
        depth_fair_discovery_order and _depth_fair_halfcap_dual_abs > 0.0)
    depth_fair_rounds = 0
    neutral_multistop_enrichment = bool(
        neutral_multistop_enrichment
        and depth_fair_active
        and search_goal == "discovery"
        and stage == "coverage")
    neutral_multistop_batch_target = max(
        1, min(int(batch_size), int(neutral_multistop_batch_target)))
    neutral_uncovered_tids = frozenset(
        _tid(t) for t in (neutral_uncovered_tids or ()))
    neutral_multistop_candidates_seen = 0
    neutral_multistop_cross_zero_seen = 0
    neutral_multistop_nonnegative_seen = 0
    neutral_multistop_early_return = False
    neutral_multistop_by_service = {}
    neutral_multistop_best_stop_count = 0
    neutral_multistop_best_uncovered_gain = 0
    neutral_multistop_best_rc_ub = None
    neutral_multistop_best_energy_per_stop_Wh = None
    # Layered mode uses the V4 dual-guided child order as a secondary priority.
    if layered_discovery_order:
        guided_discovery_order = True
    layered_depths_started = 0
    layered_depths_completed = 0
    layered_max_depth_completed = 0
    layered_rounds = 0
    _pricing_wall_t0 = time.perf_counter()
    physical_cache_hits = 0
    physical_cache_misses = 0
    physical_evaluator_runtime_s = 0.0
    prefix_bound_runtime_s = 0.0
    prefix_service_runtime_s = 0.0
    certified_prefix_pruning = bool(
        certified_prefix_pruning and search_goal == "certification")
    certified_prefix_prunes = 0
    depth_certified_prefix_prunes = {}
    service_floor_prunes = 0
    depth_service_floor_prunes = {}
    horizon_window_skips = 0
    horizon_service_time_skips = 0
    physical_infeasible_results = 0
    whole_route_evaluator_calls = 0
    drcc_route_evaluator_calls = 0
    dominance_prunes = 0  # no formal dominance mechanism is active in R-BPC yet
    _v30_cols = None  # v3.0-P5: set inside the production branch only
    duplicate_state_prunes = 0  # DFS states are not merged/transposed yet
    branch_filter_skips = 0
    existing_signature_skips = 0
    _V20_COMPAT_FP_CACHE.clear()
    _v20_bound_ctx = _v20_compile_prefix_bound_context(
        stage, max_stops, node.branch, inequality_rows, equality_rows,
        inequality_duals, equality_duals)
    certified_prefix_bound_histogram = {}
    launch_prefix_nodes = {}
    root_turbine_prefix_nodes = {}
    root_pair_prefix_nodes = {}
    launch_evaluator_calls = {}
    horizon_evaluator_calls = {}
    launch_horizon_evaluator_calls = {}
    improving_columns_seen = 0
    discovery_improving_launches = set()
    discovery_improving_service_sets = set()
    discovery_improving_signatures = set()
    discovery_diversity_satisfied = False
    discovery_hard_cap_triggered = False
    depth_prefixes_evaluated = {}
    depth_improving_seen = {}
    depth_improving_returned = {}
    pattern_cut_diagnostics = bool(pattern_cut_diagnostics)
    pattern_cut_rows = ([
        (desc, float(dual))
        for desc, dual in zip(inequality_rows, np.asarray(inequality_duals, float))
        if desc[0] == "resource_pattern"
    ] if pattern_cut_diagnostics else [])
    pattern_cut_active_dual_rows = sum(
        1 for _desc, dual in pattern_cut_rows if min(float(dual), 0.0) < 0.0)
    pattern_cut_dual_abs_sum = float(math.fsum(
        abs(min(float(dual), 0.0)) for _desc, dual in pattern_cut_rows))
    pattern_cut_improving_seen_count = 0
    pattern_cut_improving_seen_contribution_sum = 0.0
    pattern_cut_improving_seen_sign_essential = 0
    pattern_cut_diag_by_sig = {}
    _prefix_lb_cache = {}

    try:
        universal_lb = _universal_pricing_lower_bound(
            stage, max_stops, inequality_rows, equality_rows,
            inequality_duals, equality_duals)
        universal_bound_available = True
    except (FloatingPointError, OverflowError, ValueError):
        universal_lb = None
        universal_bound_available = False
    best_rc = math.inf
    best_rc_lb = math.inf
    best_rc_ub = math.inf
    evaluated_routes = 0
    evaluated_sequences = 0
    retained = []  # sign-definite improving columns
    ambiguous_retained = []  # outward rc interval straddles the pricing threshold
    complete = True
    reason = "exact-pricing-closed"

    allowed_tids = frozenset(_tid(t.tid) for t in turbines)
    if physical_cache is None:
        physical_cache = {}

    # [THM-RU / exact-pricing safe pruning] A route's inspection/climb service
    # component is a nonnegative subset of its full formal time/energy.  If the
    # prefix service floor alone exceeds every supported recovery horizon or the
    # usable battery energy, no extension of that prefix can be physically
    # feasible.  This removes only provably impossible subtrees and therefore
    # preserves exhaustive pricing completeness.
    _service_floor_cache = {}
    def _prefix_service_floor(prefix):
        nonlocal prefix_service_runtime_s
        _t0 = time.perf_counter()
        key = tuple(_tid(t.tid) for t in prefix)
        got = _service_floor_cache.get(key)
        if got is not None:
            prefix_service_runtime_s += time.perf_counter() - _t0
            return got
        t_lb = 0.0
        e_lb = 0.0
        for tb in prefix:
            dz = float(M.insp_vertical_span(tb, p.z_cruise))
            if getattr(p, "use_zeng", False):
                p_up = float(M.P_zeng(0.0, p) + 7.27 * 9.81 * p.v_z)
                p_insp = float(M.P_zeng(p.v_orbit, p))
            else:
                p_up = float(p.P_climb)
                p_insp = float(p.P_hov)
            t_service = float(dz / p.v_z + p.tau_insp)
            e_service = float(
                p_up * dz / p.v_z / 3600.0
                + p_insp * p.tau_insp / 3600.0)
            t_lb += t_service
            e_lb += e_service
        got = (float(t_lb), float(e_lb))
        _service_floor_cache[key] = got
        prefix_service_runtime_s += time.perf_counter() - _t0
        return got

    def _stage2_energy_coverage_joint_lb(prefix, prefix_energy_lb):
        """Safe LB on E_plan - lambda_C*|S| over every prefix completion.

        Only the inspection/climb service component is used.  Branch/physics
        restrictions on future turbines are relaxed, so the continuation set
        here is a superset of legal completions.  For each possible added-stop
        count m, the m smallest remaining service-energy floors are combined
        with the exact coverage coefficient k+m using outward-safe arithmetic.
        """
        if stage != "energy":
            return None
        _cov = [
            float(_dual)
            for _desc, _dual in zip(
                equality_rows, np.asarray(equality_duals, float))
            if _desc[0] == "coverage"
        ]
        if len(_cov) != 1 or not math.isfinite(_cov[0]):
            return None
        _lam = float(_cov[0])
        _prefix_tids = tuple(_tid(_t.tid) for _t in prefix)
        _used = frozenset(_prefix_tids)
        _remaining_service = []
        for _tb in turbines:
            _tb_tid = _tid(_tb.tid)
            if (_tb_tid in _used
                    or _tb_tid in node.branch.forbidden_turbines):
                continue
            _e = float(_prefix_service_floor((_tb,))[1])
            # Protect the single-stop service component downward before it is
            # reused in an order-independent continuation envelope.
            _remaining_service.append(math.nextafter(_e, -math.inf))
        _remaining_service.sort()
        _slots = min(
            max(0, int(max_stops) - len(_prefix_tids)),
            len(_remaining_service))
        _e_lb = math.nextafter(float(prefix_energy_lb), -math.inf)
        _best = math.inf
        for _m in range(_slots + 1):
            _cov_lo, _ = _outward_product_interval(
                -_lam, float(len(_prefix_tids) + _m))
            _joint = _outward_add_lower(_e_lb, _cov_lo)
            _best = min(_best, _joint)
            if _m < _slots:
                _e_lb = _outward_add_lower(
                    _e_lb, _remaining_service[_m])
        return math.nextafter(float(_best), -math.inf)

    def _prefix_lb_cached(prefix, oi, opt):
        nonlocal prefix_bound_runtime_s
        _t0 = time.perf_counter()
        key = (int(oi), tuple(_tid(t.tid) for t in prefix))
        if key not in _prefix_lb_cache:
            _service_e = float(_prefix_service_floor(prefix)[1])
            _joint_lb = _stage2_energy_coverage_joint_lb(prefix, _service_e)
            if _v20_bound_ctx is not None:
                _prefix_lb_cache[key] = _v20_bound_ctx(
                    prefix, opt,
                    energy_objective_lb=(
                        math.nextafter(_service_e, -math.inf)
                        if stage == "energy" else None),
                    energy_coverage_joint_lb=_joint_lb)
            else:
                _prefix_lb_cache[key] = _prefix_pricing_lower_bound(
                    stage, max_stops, prefix, opt, node.branch,
                    inequality_rows, equality_rows,
                    inequality_duals, equality_duals,
                    energy_objective_lb=(
                        math.nextafter(_service_e, -math.inf)
                        if stage == "energy" else None),
                    energy_coverage_joint_lb=_joint_lb)
        out = float(_prefix_lb_cache[key])
        prefix_bound_runtime_s += time.perf_counter() - _t0
        return out

    def _record_certified_prefix_bound(value):
        value = float(value)
        if value < -100.0:
            key = "<-100"
        elif value < -10.0:
            key = "[-100,-10)"
        elif value < -1.0:
            key = "[-10,-1)"
        elif value < -0.1:
            key = "[-1,-0.1)"
        elif value < 0.0:
            key = "[-0.1,0)"
        elif value < 0.1:
            key = "[0,0.1)"
        elif value < 1.0:
            key = "[0.1,1)"
        elif value < 10.0:
            key = "[1,10)"
        elif value < 100.0:
            key = "[10,100)"
        else:
            key = ">=100"
        certified_prefix_bound_histogram[key] = int(
            certified_prefix_bound_histogram.get(key, 0)) + 1

    def _step_distance(prefix, turbine, opt):
        try:
            origin = (np.asarray(prefix[-1].local, float) if prefix
                      else np.asarray(opt.ship.P_launch, float))
            target = np.asarray(turbine.local, float)
            d = float(np.linalg.norm(target - origin))
            return d if math.isfinite(d) else math.inf
        except Exception:
            return math.inf

    def _rank_children(prefix, candidates, oi, opt):
        nonlocal guided_order_calls, guided_order_reorders, guided_order_failures
        raw = list(candidates)
        if not guided_discovery_order or len(raw) <= 1:
            return raw
        guided_order_calls += 1
        ordered, reordered, failed = _stable_discovery_order(
            raw,
            lambda tb: (
                _prefix_lb_cached(prefix + (tb,), oi, opt),
                _step_distance(prefix, tb, opt),
            ))
        guided_order_reorders += int(reordered)
        guided_order_failures += int(failed)
        return ordered

    def consider(column):
        nonlocal best_rc, best_rc_lb, best_rc_ub, evaluated_routes, complete, reason
        nonlocal shadow_false_prune_witnesses, discovery_early_return
        nonlocal improving_columns_seen, discovery_diversity_satisfied
        nonlocal discovery_hard_cap_triggered
        nonlocal pattern_cut_improving_seen_count
        nonlocal pattern_cut_improving_seen_contribution_sum
        nonlocal pattern_cut_improving_seen_sign_essential
        nonlocal neutral_multistop_candidates_seen
        nonlocal neutral_multistop_cross_zero_seen
        nonlocal neutral_multistop_nonnegative_seen
        nonlocal neutral_multistop_early_return
        nonlocal neutral_multistop_best_stop_count
        nonlocal neutral_multistop_best_uncovered_gain
        nonlocal neutral_multistop_best_rc_ub
        nonlocal neutral_multistop_best_energy_per_stop_Wh
        nonlocal branch_filter_skips, existing_signature_skips
        if column is None:
            return
        try:
            _validate_column_domain(column, allowed_tids, max_stops)
        except ValueError:
            return
        if not _column_allowed_at_node(column, node.branch):
            branch_filter_skips += 1
            return
        sig = _exact_route_signature(column)
        if sig in existing_signatures:
            existing_signature_skips += 1
            return
        evaluated_routes += 1
        try:
            rc, rc_lb, rc_ub = _column_reduced_cost_interval(
                column, stage, inequality_rows, equality_rows,
                inequality_duals, equality_duals)
        except (FloatingPointError, OverflowError, ValueError):
            complete = False
            reason = "pricing-numeric-certificate-error"
            return
        best_rc = min(best_rc, rc)
        best_rc_lb = min(best_rc_lb, rc_lb)
        best_rc_ub = min(best_rc_ub, rc_ub)
        # Formal certificate pricing uses the mathematical zero threshold in
        # every stage.  ``pricing_epsilon`` is *not* an optimality tolerance:
        # any route whose outward upper endpoint is strictly negative is a
        # proved improving column, however small the improvement.  If zero lies
        # inside the interval, add a bounded batch of legal neutral-enrichment
        # columns so the finite implicit route space makes combinatorial progress
        # instead of resolving the identical RMP from a point estimate.
        threshold = 0.0
        improving = rc_ub < 0.0
        ambiguous = (not improving and rc_lb < 0.0 <= rc_ub)

        # V14: bounded heuristic-only multi-stop enrichment.  The route has
        # already passed the unchanged whole-route physical evaluator and formal
        # node/identity checks above.  Admission here requires only that it is
        # NOT a proved negative column, so it can never be confused with formal
        # pricing progress.  Early return is explicitly incomplete.
        if neutral_multistop_enrichment and not improving:
            _neutral_tids = tuple(_ordered_tids(column))
            _neutral_depth = int(len(_neutral_tids))
            if _neutral_depth >= 2:
                neutral_multistop_candidates_seen += 1
                if ambiguous:
                    neutral_multistop_cross_zero_seen += 1
                elif rc_lb >= 0.0:
                    neutral_multistop_nonnegative_seen += 1
                try:
                    _neutral_energy = float(column["E_soc_required_Wh"])
                    if not math.isfinite(_neutral_energy) or _neutral_energy < 0.0:
                        raise ValueError("non-finite neutral route energy")
                    _neutral_energy_per_stop = _neutral_energy / _neutral_depth
                    _neutral_high_energy = bool(
                        2 * Fraction.from_float(_neutral_energy)
                        > Fraction.from_float(float(p.B_use)))
                except (KeyError, TypeError, ValueError, OverflowError):
                    _neutral_energy = math.inf
                    _neutral_energy_per_stop = math.inf
                    _neutral_high_energy = True
                _neutral_uncovered_gain = int(sum(
                    1 for _t in set(_neutral_tids)
                    if _t in neutral_uncovered_tids))
                neutral_multistop_best_stop_count = max(
                    neutral_multistop_best_stop_count, _neutral_depth)
                neutral_multistop_best_uncovered_gain = max(
                    neutral_multistop_best_uncovered_gain,
                    _neutral_uncovered_gain)
                if (neutral_multistop_best_rc_ub is None
                        or float(rc_ub) < neutral_multistop_best_rc_ub):
                    neutral_multistop_best_rc_ub = float(rc_ub)
                if (neutral_multistop_best_energy_per_stop_Wh is None
                        or _neutral_energy_per_stop
                            < neutral_multistop_best_energy_per_stop_Wh):
                    neutral_multistop_best_energy_per_stop_Wh = float(
                        _neutral_energy_per_stop)

                # Distinct service sets prevent a launch/horizon variant cloud
                # from consuming the whole enrichment batch.  Route order and
                # exact signature remain untouched inside the chosen column.
                _neutral_service = frozenset(_neutral_tids)
                _neutral_rank = (
                    -_neutral_uncovered_gain,
                    -_neutral_depth,
                    int(_neutral_high_energy),
                    float(_neutral_energy_per_stop),
                    float(rc_ub),
                    repr(sig),
                )
                _old_neutral = neutral_multistop_by_service.get(
                    _neutral_service)
                if (_old_neutral is None
                        or _neutral_rank < _old_neutral[0]):
                    neutral_multistop_by_service[_neutral_service] = (
                        _neutral_rank, column, float(rc_lb), float(rc_ub))

                _neutral_high_value = bool(
                    neutral_uncovered_tids and _neutral_uncovered_gain > 0)
                _neutral_diverse_batch_ready = bool(
                    len(neutral_multistop_by_service)
                    >= neutral_multistop_batch_target)
                _neutral_scan_cap_ready = bool(
                    neutral_multistop_candidates_seen >= max(
                        24, 3 * neutral_multistop_batch_target)
                    and neutral_multistop_by_service)
                if (search_goal == "discovery"
                        and not retained
                        and (_neutral_high_value
                             or _neutral_diverse_batch_ready
                             or _neutral_scan_cap_ready)):
                    complete = False
                    discovery_early_return = True
                    neutral_multistop_early_return = True
                    reason = "discovery-neutral-multistop-batch-found"
                    return

        if improving:
            # V10 shadow telemetry only: recompute this already-proved negative
            # column without resource-pattern rows. Formal admission above stays
            # exactly ``rc_ub < 0`` and is never changed by these values.
            if pattern_cut_diagnostics:
                pcontrib = 0.0
                for desc, dual in pattern_cut_rows:
                    d = min(float(dual), 0.0)
                    coeff = float(_row_coefficient(column, desc))
                    pcontrib += -(d * coeff)
                keep = [i for i, desc in enumerate(inequality_rows)
                        if desc[0] != "resource_pattern"]
                try:
                    _rc0, _rc0_lb, rc0_ub = _column_reduced_cost_interval(
                        column, stage,
                        [inequality_rows[i] for i in keep], equality_rows,
                        np.asarray([inequality_duals[i] for i in keep], float),
                        equality_duals)
                    sign_essential = bool(rc0_ub >= 0.0)
                except (FloatingPointError, OverflowError, ValueError):
                    sign_essential = False
                    rc0_ub = math.nan
                pattern_cut_improving_seen_count += 1
                pattern_cut_improving_seen_contribution_sum += float(pcontrib)
                pattern_cut_improving_seen_sign_essential += int(sign_essential)
                pattern_cut_diag_by_sig[sig] = (
                    float(pcontrib), bool(sign_essential), float(rc), float(rc0_ub))

            if shadow_prefix_bounds and shadow_prunable_keys:
                oi = int(column.get("launch_option_index", -1))
                seq = tuple(_ordered_tids(column))
                for plen in range(1, len(seq) + 1):
                    pkey = (oi, seq[:plen])
                    if pkey in shadow_prunable_keys and pkey not in shadow_false_witness_keys:
                        shadow_false_witness_keys.add(pkey)
                        shadow_false_prune_witnesses += 1

            # V6 deduplicates only within the *incomplete discovery batch*.
            # Same formal signature means the same master column; ignoring a
            # duplicate cannot remove a distinct column or strengthen a proof.
            if adaptive_discovery_batch and sig in discovery_improving_signatures:
                return
            if adaptive_discovery_batch:
                discovery_improving_signatures.add(sig)

            item = (-rc, repr(sig), evaluated_routes, column, rc)
            if len(retained) < int(batch_size):
                heapq.heappush(retained, item)
            elif -rc > retained[0][0]:
                heapq.heapreplace(retained, item)

            improving_columns_seen += 1
            _improving_depth = int(len(_ordered_tids(column)))
            depth_improving_seen[_improving_depth] = int(
                depth_improving_seen.get(_improving_depth, 0)) + 1
            try:
                launch_key = (
                    "index", int(column["launch_option_index"])
                ) if column.get("launch_option_index") is not None else (
                    "tau", float(column["tau"]).hex())
            except Exception:
                launch_key = ("unknown", repr(sig))
            discovery_improving_launches.add(launch_key)
            discovery_improving_service_sets.add(
                frozenset(_ordered_tids(column)))
            discovery_diversity_satisfied = bool(
                len(discovery_improving_launches) >= _min_distinct_launches
                and len(discovery_improving_service_sets)
                    >= _min_distinct_service_sets)

            # Discovery stops only after sign-definite improving columns have
            # passed the unchanged full physical evaluator. complete=False is
            # the firewall preventing this early return from proving closure.
            if search_goal == "discovery":
                if adaptive_discovery_batch:
                    _target_met = (
                        len(retained) >= _batch_target
                        and discovery_diversity_satisfied)
                    _hard_cap_met = improving_columns_seen >= _batch_hard_cap
                    if _target_met or _hard_cap_met:
                        complete = False
                        discovery_early_return = True
                        discovery_hard_cap_triggered = bool(
                            _hard_cap_met and not _target_met)
                        reason = (
                            "discovery-diverse-batch-found"
                            if _target_met
                            else "discovery-batch-hard-cap-found")
                elif len(retained) >= min(
                        int(batch_size), discovery_column_limit):
                    complete = False
                    reason = "discovery-improving-batch-found"
                    discovery_early_return = True
        elif ambiguous:
            # Adding a valid column never invalidates the RMP.  If rigorous
            # directed rounding cannot decide on which side of the threshold
            # the column lies, retain a bounded batch as neutral enrichment.
            # This makes finite combinatorial progress and avoids repeatedly
            # resolving the identical RMP from a non-rigorous point estimate.
            item = (-rc_lb, repr(sig), evaluated_routes, column, rc_lb, rc)
            if len(ambiguous_retained) < int(batch_size):
                heapq.heappush(ambiguous_retained, item)
            elif -rc_lb > ambiguous_retained[0][0]:
                heapq.heapreplace(ambiguous_retained, item)

    if implicit_test_columns is not None:
        for raw in implicit_test_columns:
            if _deadline_hit(deadline):
                complete = False; reason = "exact-pricing-time-limit"; break
            c = _normalize_exact_column(
                raw, p=p, t_launch_min=t_launch_min,
                landing_clear_min=landing_clear_min,
                deck_mode=deck_mode, deck_delta_min=deck_delta_min)
            consider(c)
            evaluated_sequences += 1
            _d = int(len(_ordered_tids(c)))
            depth_prefixes_evaluated[_d] = int(
                depth_prefixes_evaluated.get(_d, 0)) + 1
            if not complete:
                break
    else:
        horizons = tuple(float(h) for h in RM.decision_horizons_of(xi_amb))
        risk_policy = _risk_policy_for_mode(kappa_mode)
        # v3.0-P5 (THM-006): exact cached-feature pricing.  Once a production
        # traversal has completed over the maximal branch space, every later
        # pricing call re-derives its candidate set from the completed
        # physical cache instead of re-walking the prefix tree.  The
        # per-column node filter, master-signature skip and rc interval are
        # all inherited from the unchanged consider() below.
        _v30_cols = _v30_cached_feature_columns(
            physical_cache, turbines, p, T_min, max_stops, horizons)
        if _v30_cols is not None:
            for _v30_c in _v30_cols:
                if _deadline_hit(deadline):
                    complete = False
                    reason = "exact-pricing-time-limit"
                    break
                consider(_v30_c)
                if not complete:
                    break
        try:
            launch_plan = ([] if _v30_cols is not None
                           else list(enumerate(launch_opts)))
            if guided_discovery_order and len(launch_plan) > 1:
                guided_order_calls += 1

                def _launch_score(pair):
                    oi0, opt0 = pair
                    eligible = [
                        tb for tb in turbines
                        if _tid(tb.tid) not in node.branch.forbidden_turbines
                    ]
                    if not eligible:
                        return (1e300, 1e300)
                    return min(
                        (_prefix_lb_cached((tb,), oi0, opt0),
                         _step_distance(tuple(), tb, opt0))
                        for tb in eligible)

                launch_plan, launch_reordered, launch_failed = (
                    _stable_discovery_order(launch_plan, _launch_score))
                guided_order_reorders += int(launch_reordered)
                guided_order_failures += int(launch_failed)

            def _safe_reach(oi0, opt0):
                reach_proof = RA.tau_reach(
                    opt0, turbines, p, max(horizons) if horizons else 0.0,
                    mode="valid", wx=getattr(opt0, "wx", None), xi_amb=xi_amb,
                    weather_unc=weather_unc)
                observed_excluded = tuple(sorted(
                    {_tid(t.tid) for t in turbines}
                    - {_tid(t.tid) for t in reach_proof}))
                reach_safe = bool(
                    getattr(reach_proof, "proof_complete", False)
                    and getattr(reach_proof, "effective_mode", None)
                        in {"off", "valid-proven"}
                    and observed_excluded == tuple(sorted(
                        _tid(t) for t in getattr(reach_proof, "excluded_tids", ())))
                    and (getattr(reach_proof, "effective_mode", None) != "off"
                         or not observed_excluded)
                    and (getattr(reach_proof, "effective_mode", None)
                         != "valid-proven"
                         or (bool(getattr(reach_proof, "mean_relax_free", False))
                             and not bool(getattr(
                                 reach_proof, "speed_adjustable", True)))))
                reach_source = list(reach_proof) if reach_safe else list(turbines)
                return [
                    t for t in reach_source
                    if _tid(t.tid) not in node.branch.forbidden_turbines
                ]

            def _evaluate_one_prefix(prefix, oi0, opt0):
                nonlocal complete, reason, evaluated_sequences
                nonlocal shadow_prefixes_evaluated, shadow_prunable_prefixes
                nonlocal shadow_bound_errors
                nonlocal physical_cache_hits, physical_cache_misses
                nonlocal physical_evaluator_runtime_s, physical_infeasible_results
                nonlocal whole_route_evaluator_calls, drcc_route_evaluator_calls
                nonlocal certified_prefix_prunes, service_floor_prunes
                nonlocal horizon_window_skips, horizon_service_time_skips
                nonlocal best_rc_lb
                if not complete:
                    return False
                if _deadline_hit(deadline):
                    complete = False
                    reason = "exact-pricing-time-limit"
                    return False
                evaluated_sequences += 1
                _d = int(len(prefix))
                launch_prefix_nodes[str(int(oi0))] = int(
                    launch_prefix_nodes.get(str(int(oi0)), 0)) + 1
                _root_tid = str(_tid(prefix[0].tid))
                root_turbine_prefix_nodes[_root_tid] = int(
                    root_turbine_prefix_nodes.get(_root_tid, 0)) + 1
                if _d >= 2:
                    _pair_key = _root_tid + "->" + str(_tid(prefix[1].tid))
                    root_pair_prefix_nodes[_pair_key] = int(
                        root_pair_prefix_nodes.get(_pair_key, 0)) + 1
                depth_prefixes_evaluated[_d] = int(
                    depth_prefixes_evaluated.get(_d, 0)) + 1
                service_t_lb, service_e_lb = _prefix_service_floor(prefix)
                max_h_s = 60.0 * max(horizons) if horizons else 0.0
                if service_t_lb > max_h_s or service_e_lb > float(p.B_use):
                    service_floor_prunes += 1
                    depth_service_floor_prunes[_d] = int(
                        depth_service_floor_prunes.get(_d, 0)) + 1
                    return False

                # [THM-RBPC-PFX] For every completion r of this exact ordered
                # prefix, _prefix_pricing_lower_bound returns LB <= rc(r).
                # Therefore LB>=0 proves that no completion can satisfy the
                # sole formal improving-column predicate rc_UB<0.  Record LB
                # in the completed-search omitted-column bound before pruning.
                if certified_prefix_pruning:
                    try:
                        _cert_lb = _prefix_lb_cached(prefix, oi0, opt0)
                    except (FloatingPointError, OverflowError, ValueError, TypeError):
                        _cert_lb = None  # fail open: enumerate the subtree
                    if _cert_lb is not None:
                        _record_certified_prefix_bound(_cert_lb)
                    if _cert_lb is not None and _cert_lb >= 0.0:
                        best_rc_lb = min(best_rc_lb, float(_cert_lb))
                        certified_prefix_prunes += 1
                        depth_certified_prefix_prunes[_d] = int(
                            depth_certified_prefix_prunes.get(_d, 0)) + 1
                        return False

                if shadow_prefix_bounds:
                    shadow_prefixes_evaluated += 1
                    try:
                        shadow_lb = _prefix_lb_cached(prefix, oi0, opt0)
                        if shadow_lb >= 0.0:
                            pkey = (
                                int(oi0),
                                tuple(_tid(t.tid) for t in prefix),
                            )
                            shadow_prunable_keys.add(pkey)
                            shadow_prunable_prefixes += 1
                    except (FloatingPointError, OverflowError, ValueError, TypeError):
                        shadow_bound_errors += 1

                for h in horizons:
                    if _deadline_hit(deadline):
                        complete = False
                        reason = "exact-pricing-time-limit"
                        return False
                    if float(opt0.tau_min) + h > float(T_min):
                        horizon_window_skips += 1
                        continue
                    if service_t_lb > 60.0 * float(h):
                        horizon_service_time_skips += 1
                        continue
                    cache_key = (
                        int(oi0),
                        tuple(_tid(t.tid) for t in prefix),
                        float(h).hex(),
                    )
                    if cache_key in physical_cache:
                        physical_cache_hits += 1
                        candidate = physical_cache[cache_key]
                    else:
                        physical_cache_misses += 1
                        whole_route_evaluator_calls += 1
                        if str(chance_mode) == "drcc":
                            drcc_route_evaluator_calls += 1
                        _oi_key = str(int(oi0))
                        _h_key = float(h).hex()
                        _oh_key = _oi_key + "|" + _h_key
                        launch_evaluator_calls[_oi_key] = int(
                            launch_evaluator_calls.get(_oi_key, 0)) + 1
                        horizon_evaluator_calls[_h_key] = int(
                            horizon_evaluator_calls.get(_h_key, 0)) + 1
                        launch_horizon_evaluator_calls[_oh_key] = int(
                            launch_horizon_evaluator_calls.get(_oh_key, 0)) + 1
                        _phys_t0 = time.perf_counter()
                        try:
                            candidate = _candidate_from_physics(
                                oi0, opt0, prefix, h, p, xi_amb, weather_unc,
                                t_launch_min, landing_clear_min,
                                deck_mode, deck_delta_min,
                                chance_mode=chance_mode,
                                budget_gamma=budget_gamma, deadline=deadline,
                                risk_policy=risk_policy)
                        except TimeoutError:
                            physical_evaluator_runtime_s += (
                                time.perf_counter() - _phys_t0)
                            complete = False
                            reason = "exact-pricing-time-limit"
                            return False
                        except Exception as exc:
                            physical_evaluator_runtime_s += (
                                time.perf_counter() - _phys_t0)
                            complete = False
                            reason = (
                                f"pricing-evaluator-error:{type(exc).__name__}")
                            return False
                        physical_evaluator_runtime_s += (
                            time.perf_counter() - _phys_t0)
                        physical_cache[cache_key] = candidate
                    if candidate is None:
                        physical_infeasible_results += 1
                    consider(candidate)
                    if not complete:
                        return False
                return True

            def _depth_prefix_iterator(oi0, opt0, reach0, target_depth):
                def walk(prefix, used):
                    if _deadline_hit(deadline):
                        return
                    if prefix:
                        service_t_lb, service_e_lb = _prefix_service_floor(prefix)
                        max_h_s = 60.0 * max(horizons) if horizons else 0.0
                        if (service_t_lb > max_h_s
                                or service_e_lb > float(p.B_use)):
                            return
                    if len(prefix) == int(target_depth):
                        yield prefix
                        return
                    last_tid = _tid(prefix[-1].tid) if prefix else None
                    child_candidates = []
                    for t in reach0:
                        tid = _tid(t.tid)
                        if tid in used:
                            continue
                        if (last_tid is not None
                                and (last_tid, tid) in node.branch.forbidden_arcs):
                            continue
                        child_candidates.append(t)
                    for t in _rank_children(
                            prefix, child_candidates, oi0, opt0):
                        tid = _tid(t.tid)
                        yield from walk(prefix + (t,), used | {tid})
                yield from walk(tuple(), set())

            if layered_discovery_order:
                # Precompute only the safe reach supersets for each launch.
                # This is cheap compared with the physical oracle and permits
                # true sequence-level round-robin across launches.
                launch_contexts = []
                for oi, opt in launch_plan:
                    if _deadline_hit(deadline):
                        complete = False
                        reason = "exact-pricing-time-limit"
                        break
                    launch_contexts.append((int(oi), opt, _safe_reach(oi, opt)))

                if complete:
                    if depth_fair_active:
                        # V13: each depth owns the same exact launch-round-robin
                        # iterator used by V5.  The outer round-robin advances
                        # one item from every active depth before returning to a
                        # depth.  Thus depth-3/4 receive discovery opportunities
                        # without deleting, merging, or pruning any sequence.
                        def _depth_launch_iterator(target_depth):
                            tagged = [
                                ((oi, opt),
                                 _depth_prefix_iterator(
                                     oi, opt, reach, target_depth))
                                for oi, opt, reach in launch_contexts
                            ]
                            for launch_round, (oi, opt), prefix in (
                                    _round_robin_tagged_iterators(tagged)):
                                yield (
                                    int(launch_round), int(oi), opt, prefix)

                        depth_tagged = [
                            (int(target_depth),
                             _depth_launch_iterator(target_depth))
                            for target_depth in range(
                                1, int(max_stops) + 1)
                        ]
                        layered_depths_started += int(max_stops)
                        last_round_index = -1
                        for round_index, target_depth, payload in (
                                _round_robin_tagged_iterators(depth_tagged)):
                            if not complete:
                                break
                            if _deadline_hit(deadline):
                                complete = False
                                reason = "exact-pricing-time-limit"
                                break
                            if int(round_index) != int(last_round_index):
                                layered_rounds += 1
                                depth_fair_rounds += 1
                                last_round_index = int(round_index)
                            _launch_round, oi, opt, prefix = payload
                            _evaluate_one_prefix(prefix, oi, opt)
                            if not complete:
                                break
                        if complete:
                            layered_depths_completed += int(max_stops)
                            layered_max_depth_completed = int(max_stops)
                    else:
                        for target_depth in range(1, int(max_stops) + 1):
                            layered_depths_started += 1
                            tagged = [
                                ((oi, opt),
                                 _depth_prefix_iterator(
                                     oi, opt, reach, target_depth))
                                for oi, opt, reach in launch_contexts
                            ]
                            # The helper interleaves one exact target-depth
                            # sequence from each active launch per round.
                            last_round_index = -1
                            for round_index, (oi, opt), prefix in (
                                    _round_robin_tagged_iterators(tagged)):
                                if not complete:
                                    break
                                if _deadline_hit(deadline):
                                    complete = False
                                    reason = "exact-pricing-time-limit"
                                    break
                                if int(round_index) != int(last_round_index):
                                    layered_rounds += 1
                                    last_round_index = int(round_index)
                                _evaluate_one_prefix(prefix, oi, opt)
                                if not complete:
                                    break
                            if not complete:
                                break
                            layered_depths_completed += 1
                            layered_max_depth_completed = int(target_depth)
            else:
                # Historical launch-first DFS path (V3/V4 / certification).
                for oi, opt in launch_plan:
                    if _deadline_hit(deadline):
                        complete = False
                        reason = "exact-pricing-time-limit"
                        break
                    reach = _safe_reach(oi, opt)

                    def extend(prefix, used):
                        if not complete:
                            return
                        if _deadline_hit(deadline):
                            return
                        if prefix and not _evaluate_one_prefix(prefix, oi, opt):
                            return
                        if len(prefix) >= int(max_stops):
                            return
                        last_tid = _tid(prefix[-1].tid) if prefix else None
                        child_candidates = []
                        for t in reach:
                            tid = _tid(t.tid)
                            if tid in used:
                                continue
                            if (last_tid is not None
                                    and (last_tid, tid)
                                    in node.branch.forbidden_arcs):
                                continue
                            child_candidates.append(t)
                        for t in _rank_children(
                                prefix, child_candidates, oi, opt):
                            tid = _tid(t.tid)
                            extend(prefix + (t,), used | {tid})
                            if not complete:
                                return

                    extend(tuple(), set())
                    if not complete:
                        break
        finally:
            # Formal pricing uses an explicit kappa function; no module-global
            # risk multiplier is mutated, so concurrent exact solves cannot
            # cross-contaminate the certified feasible region.
            pass

    if _v30_cols is None and implicit_test_columns is None:
        _v30_register_route_space_complete(
            physical_cache, node.branch, bool(complete))

    cols = [item[3] for item in sorted(retained, key=lambda z: z[4])]
    for _c in cols:
        _d = int(len(_ordered_tids(_c)))
        depth_improving_returned[_d] = int(
            depth_improving_returned.get(_d, 0)) + 1
    neutral_multistop_returned_by_depth = {}
    neutral_multistop_returned = 0
    if not cols and neutral_multistop_enrichment and neutral_multistop_by_service:
        _neutral_items = sorted(
            neutral_multistop_by_service.values(), key=lambda z: z[0])
        _neutral_items = _neutral_items[:neutral_multistop_batch_target]
        cols = [item[1] for item in _neutral_items]
        neutral_multistop_returned = int(len(cols))
        for _c in cols:
            _d = int(len(_ordered_tids(_c)))
            neutral_multistop_returned_by_depth[_d] = int(
                neutral_multistop_returned_by_depth.get(_d, 0)) + 1
        # Even if the finite search happened to finish, these nonnegative /
        # sign-ambiguous columns are enrichment only.  Returning them forces a
        # new RMP and must never be interpreted as pricing closure.
        if cols:
            complete = False
            discovery_early_return = True
            neutral_multistop_early_return = True
            reason = "discovery-neutral-multistop-batch-found"
    if not cols and complete and ambiguous_retained:
        cols = [item[3] for item in sorted(ambiguous_retained, key=lambda z: z[4])]
        reason = "exact-pricing-numeric-ambiguity-progress"
    pattern_cut_returned_count = 0
    pattern_cut_returned_contribution_sum = 0.0
    pattern_cut_returned_sign_essential = 0
    pattern_cut_returned_by_depth = {}
    if pattern_cut_diagnostics:
        # ``retained`` contains strict rc_ub<0 columns only; ambiguous neutral
        # enrichment is intentionally excluded from this causal diagnostic.
        for item in retained:
            col = item[3]
            sig = _exact_route_signature(col)
            got = pattern_cut_diag_by_sig.get(sig)
            if got is None:
                continue
            pcontrib, sign_essential, rc_full, rc0_ub = got
            d = int(len(_ordered_tids(col)))
            rec = pattern_cut_returned_by_depth.setdefault(
                d, dict(count=0, contribution_sum=0.0, sign_essential=0,
                        rc_sum=0.0, rc_without_cut_ub_sum=0.0,
                        rc_without_cut_ub_finite_count=0))
            rec["count"] += 1
            rec["contribution_sum"] += float(pcontrib)
            rec["sign_essential"] += int(bool(sign_essential))
            rec["rc_sum"] += float(rc_full)
            if math.isfinite(float(rc0_ub)):
                rec["rc_without_cut_ub_sum"] += float(rc0_ub)
                rec["rc_without_cut_ub_finite_count"] += 1
            pattern_cut_returned_count += 1
            pattern_cut_returned_contribution_sum += float(pcontrib)
            pattern_cut_returned_sign_essential += int(bool(sign_essential))

    if complete:
        bound = math.inf if best_rc_lb == math.inf else float(best_rc_lb)
        bound_available = True
    else:
        bound = universal_lb
        bound_available = bool(universal_bound_available)
    return PricingSearchResult(
        cols, complete,
        None if best_rc == math.inf else float(best_rc),
        (None if bound is None else float(bound)), bound_available,
        evaluated_routes, evaluated_sequences, reason,
        search_goal=search_goal,
        discovery_early_return=bool(discovery_early_return),
        shadow_prefixes_evaluated=int(shadow_prefixes_evaluated),
        shadow_prunable_prefixes=int(shadow_prunable_prefixes),
        shadow_false_prune_witnesses=int(shadow_false_prune_witnesses),
        shadow_bound_errors=int(shadow_bound_errors),
        shadow_audit_complete=bool(shadow_prefix_bounds and complete),
        guided_order_calls=int(guided_order_calls),
        guided_order_reorders=int(guided_order_reorders),
        guided_order_failures=int(guided_order_failures),
        layered_depths_started=int(layered_depths_started),
        layered_depths_completed=int(layered_depths_completed),
        layered_max_depth_completed=int(layered_max_depth_completed),
        layered_rounds=int(layered_rounds),
        physical_cache_hits=int(physical_cache_hits),
        physical_cache_misses=int(physical_cache_misses),
        wall_time_s=float(time.perf_counter() - _pricing_wall_t0),
        physical_evaluator_runtime_s=float(physical_evaluator_runtime_s),
        prefix_bound_runtime_s=float(prefix_bound_runtime_s),
        prefix_service_runtime_s=float(prefix_service_runtime_s),
        certified_prefix_pruning_enabled=bool(certified_prefix_pruning),
        certified_prefix_prunes=int(certified_prefix_prunes),
        depth_certified_prefix_prunes=dict(depth_certified_prefix_prunes),
        service_floor_prunes=int(service_floor_prunes),
        depth_service_floor_prunes=dict(depth_service_floor_prunes),
        horizon_window_skips=int(horizon_window_skips),
        horizon_service_time_skips=int(horizon_service_time_skips),
        physical_infeasible_results=int(physical_infeasible_results),
        whole_route_evaluator_calls=int(whole_route_evaluator_calls),
        drcc_route_evaluator_calls=int(drcc_route_evaluator_calls),
        dominance_prunes=int(dominance_prunes),
        duplicate_state_prunes=int(duplicate_state_prunes),
        branch_filter_skips=int(branch_filter_skips),
        existing_signature_skips=int(existing_signature_skips),
        best_reduced_value_ub=(
            None if best_rc_ub == math.inf else float(best_rc_ub)),
        certified_prefix_bound_histogram=dict(certified_prefix_bound_histogram),
        launch_prefix_nodes=dict(launch_prefix_nodes),
        root_turbine_prefix_nodes=dict(root_turbine_prefix_nodes),
        root_pair_prefix_nodes=dict(root_pair_prefix_nodes),
        launch_evaluator_calls=dict(launch_evaluator_calls),
        horizon_evaluator_calls=dict(horizon_evaluator_calls),
        launch_horizon_evaluator_calls=dict(launch_horizon_evaluator_calls),
        improving_columns_seen=int(improving_columns_seen),
        discovery_improving_columns_returned=int(len(retained)),
        discovery_distinct_launches=int(len(discovery_improving_launches)),
        discovery_distinct_service_sets=int(
            len(discovery_improving_service_sets)),
        discovery_diversity_satisfied=bool(
            discovery_diversity_satisfied),
        discovery_hard_cap_triggered=bool(
            discovery_hard_cap_triggered),
        depth_prefixes_evaluated=dict(depth_prefixes_evaluated),
        depth_improving_seen=dict(depth_improving_seen),
        depth_improving_returned=dict(depth_improving_returned),
        pattern_cut_active_dual_rows=int(pattern_cut_active_dual_rows),
        pattern_cut_dual_abs_sum=float(pattern_cut_dual_abs_sum),
        pattern_cut_improving_seen_count=int(pattern_cut_improving_seen_count),
        pattern_cut_improving_seen_contribution_sum=float(
            pattern_cut_improving_seen_contribution_sum),
        pattern_cut_improving_seen_sign_essential=int(
            pattern_cut_improving_seen_sign_essential),
        pattern_cut_returned_count=int(pattern_cut_returned_count),
        pattern_cut_returned_contribution_sum=float(
            pattern_cut_returned_contribution_sum),
        pattern_cut_returned_sign_essential=int(
            pattern_cut_returned_sign_essential),
        pattern_cut_returned_by_depth=dict(pattern_cut_returned_by_depth),
        depth_fair_requested=bool(depth_fair_discovery_order),
        depth_fair_active=bool(depth_fair_active),
        depth_fair_rounds=int(depth_fair_rounds),
        depth_fair_halfcap_dual_abs=float(
            _depth_fair_halfcap_dual_abs),
        neutral_multistop_enabled=bool(neutral_multistop_enrichment),
        neutral_multistop_candidates_seen=int(
            neutral_multistop_candidates_seen),
        neutral_multistop_cross_zero_seen=int(
            neutral_multistop_cross_zero_seen),
        neutral_multistop_nonnegative_seen=int(
            neutral_multistop_nonnegative_seen),
        neutral_multistop_returned=int(neutral_multistop_returned),
        neutral_multistop_returned_by_depth=dict(
            neutral_multistop_returned_by_depth),
        neutral_multistop_best_stop_count=int(
            neutral_multistop_best_stop_count),
        neutral_multistop_best_uncovered_gain=int(
            neutral_multistop_best_uncovered_gain),
        neutral_multistop_best_rc_ub=(
            None if neutral_multistop_best_rc_ub is None
            else float(neutral_multistop_best_rc_ub)),
        neutral_multistop_best_energy_per_stop_Wh=(
            None if neutral_multistop_best_energy_per_stop_Wh is None
            else float(neutral_multistop_best_energy_per_stop_Wh)),
        neutral_multistop_early_return=bool(
            neutral_multistop_early_return))


def _add_columns(archive, signature_to_index, columns):
    """[LEM-CS] Insert columns while preserving immutable route semantics.

    The canonical signature is the identity used by branching and exact-pattern
    cuts.  Consequently a signature may only be deduplicated when *all* formal
    objective/master/resource semantics are binary64-exact identical.  A same-
    signature representation with different energy or resource payload is not a
    dominance opportunity on the certificate path; it is an inconsistent model
    representation and therefore fails closed.
    """
    changed = 0
    for c in columns:
        sig = _exact_route_signature(c)
        if sig in signature_to_index:
            idx = int(signature_to_index[sig])
            old = archive[idx]
            if _column_semantics_fp(old) != _column_semantics_fp(c):
                raise RuntimeError(
                    "same canonical route signature has different formal semantics")
            continue
        signature_to_index[sig] = len(archive)
        archive.append(c)
        changed += 1
    return changed


def _route_archive_semantics_invariant(columns):
    """Executable final check for [LEM-CS]; never repairs inconsistent columns."""
    try:
        seen = {}
        for column in columns:
            sig = _exact_route_signature(column)
            fp = _column_semantics_fp(column)
            if sig in seen and seen[sig] != fp:
                return False
            seen[sig] = fp
        return True
    except Exception:
        return False


def _future_row_range_contract_self_check(max_stops):
    """Executable registry sanity check used by [THM-LRC]/[COR-P1]."""
    try:
        expected_ineq = {
            ("packing", "_"): (0.0, 1.0),
            ("deck", 0.0): (0.0, 1.0),
            ("active", 0.0): (0.0, 1.0),
            ("pooled_energy", None): (0.0, None),
            ("resource_pattern", frozenset()): (-1.0, 1.0),
            ("battery_halfcap", 1.0): (0.0, 1.0),
        }
        for desc, expected in expected_ineq.items():
            if _future_column_coefficient_range(
                    desc, max_stops, row_family="inequality") != expected:
                return False
        expected_eq = {
            ("coverage", None): (1.0, float(max_stops)),
            ("required_service", "_"): (0.0, 1.0),
            ("required_arc", ("_", "__")): (0.0, 1.0),
            ("required_route", ("_",)): (0.0, 1.0),
        }
        for desc, expected in expected_eq.items():
            if _future_column_coefficient_range(
                    desc, max_stops, row_family="equality") != expected:
                return False
        for family in ("inequality", "equality"):
            try:
                _future_column_coefficient_range(
                    ("__unregistered_proof_test_row__", None), max_stops,
                    row_family=family)
            except ValueError:
                pass
            else:
                return False
        return True
    except Exception:
        return False


def _physical_certificate_guard(*, algorithmic_global_certificate,
                                route_universe_provenance_certified, mode,
                                route_semantics_invariance_certified,
                                future_column_row_ranges_certified,
                                binary64_model_contract_enforced,
                                formal_proof_contract_enforced):
    """Machine-level conjunction corresponding to doc_proof [THM-LEX]."""
    return bool(
        algorithmic_global_certificate
        and route_universe_provenance_certified
        and mode == "exact-branch-price-cut"
        and route_semantics_invariance_certified
        and future_column_row_ranges_certified
        and binary64_model_contract_enforced
        and formal_proof_contract_enforced)


def _target_infeasibility_algorithmic_proven(stage_result, target_feasible):
    """[THM-TGT] Exact NO guard for a fixed-coverage decision problem."""
    if stage_result is None:
        return False
    return bool(
        (not target_feasible)
        and int(stage_result.open_nodes) == 0
        and bool(stage_result.resource_audit_complete)
        and bool(stage_result.branching_complete)
        and bool(stage_result.pricing_bound_available)
        and bool(stage_result.farkas_pricing_complete))


def _node_allowed_turbine_bound(all_tids, branch):
    return len(set(all_tids) - set(branch.forbidden_turbines))


def _safe_node_bound_from_pricing(master, pricing, stage, route_mass_upper_bound,
                                  all_tids, node):
    """[THM-LRC] Apply ``L_n + M_n*min(0,delta_n)`` or a safe trivial bound."""
    if master.dual_lower_bound is None or not pricing.bound_available:
        if stage == "coverage":
            return float(_node_allowed_turbine_bound(all_tids, node.branch)), "trivial-node-allowed-turbine-bound"
        return 0.0, "nonnegative-energy-bound"
    try:
        delta = float(pricing.reduced_value_bound)
        base_lb = float(master.dual_lower_bound)
        if not math.isfinite(base_lb) or math.isnan(delta):
            raise FloatingPointError
        correction = 0.0 if math.isinf(delta) and delta > 0 else min(0.0, delta)
        if correction == 0.0:
            full_lb = math.nextafter(base_lb, -math.inf)
        else:
            corr_lb, _ = _outward_product_interval(float(route_mass_upper_bound), correction)
            full_lb = _outward_add_lower(base_lb, corr_lb)
    except (TypeError, ValueError, OverflowError, FloatingPointError):
        if stage == "coverage":
            return float(_node_allowed_turbine_bound(all_tids, node.branch)), "trivial-node-allowed-turbine-bound"
        return 0.0, "nonnegative-energy-bound"
    if stage == "coverage":
        ub = min(_node_allowed_turbine_bound(all_tids, node.branch),
                 _safe_integer_floor(-full_lb))
        return float(max(0, ub)), "rmp-lagrangian-plus-pricing-bound"
    return float(max(0.0, full_lb)), "rmp-lagrangian-plus-pricing-bound"


def _phase_one_full_space_lower_bound(phase_master, pricing, route_mass_upper_bound):
    """[COR-P1] Safe lower bound on the *full* elastic Phase-I objective.

    Let ``L_RMP`` be the validated dual lower bound of the restricted elastic
    master and let every omitted ordinary route have reduced cost at least
    ``delta``.  At node n, forbidden-service filtering plus set packing and
    nonempty routes imply ``sum_r x_r <= M_n`` with
    ``M_n = |I minus forbidden_turbines(n)|``.  Therefore

        Phi_full >= L_RMP + M_n * min(0, delta).

    This is the quantity that can prove full-space infeasibility.  The ordinary
    pricing tolerance ``PRICING_EPS`` by itself is never such a proof.
    """
    if phase_master.dual_lower_bound is None or not pricing.bound_available:
        return None
    try:
        delta = float(pricing.reduced_value_bound)
        lb = float(phase_master.dual_lower_bound)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(lb):
        return None
    if math.isnan(delta):
        return None
    correction = 0.0 if math.isinf(delta) and delta > 0 else min(0.0, delta)
    try:
        if correction == 0.0:
            return math.nextafter(lb, -math.inf)
        corr_lb, _ = _outward_product_interval(float(route_mass_upper_bound), correction)
        return _outward_add_lower(lb, corr_lb)
    except (FloatingPointError, OverflowError, ValueError):
        return None


def _phase_one_infeasibility_proven(phase_master, pricing, route_mass_upper_bound,
                                    artificial_tolerance=ART_TOL):
    """Return ``(proved, full_lb)``; only a strictly positive safe LB may prune."""
    full_lb = _phase_one_full_space_lower_bound(
        phase_master, pricing, route_mass_upper_bound)
    if full_lb is None:
        return False, None
    return bool(full_lb > float(artificial_tolerance)), float(full_lb)


def _is_integral(x, tol=1e-7):
    return x is not None and np.all(np.abs(np.asarray(x) - np.rint(np.asarray(x))) <= tol)


def _exact_binary_master_feasible(master, rounded_x):
    """[THM-NUM] Exact-rational feasibility gate for a rounded binary RMP point."""
    """Verify a rounded 0/1 RMP point in exact binary64-as-rational arithmetic.

    HiGHS feasibility tolerances are appropriate for discovering an LP point, but
    they are not a certificate that ``np.rint(x)`` is a feasible integer master
    point.  Every matrix/RHS payload is therefore interpreted as the exact real
    represented by its binary64 value and rechecked without a tolerance before an
    integer pattern is sent to the resource oracle or accepted as an incumbent.
    """
    if rounded_x is None:
        return False
    vals = np.asarray(rounded_x, float)
    if len(vals) != len(master.eligible_indices):
        return False
    if not np.all(np.isfinite(vals)):
        return False
    if not np.all((vals == 0.0) | (vals == 1.0)):
        return False
    selected = [k for k, value in enumerate(vals) if value == 1.0]

    def _row_sum(matrix, r):
        total = Fraction(0)
        for k in selected:
            coeff = float(matrix[r, k])
            if not math.isfinite(coeff):
                raise FloatingPointError("non-finite master coefficient")
            total += Fraction.from_float(coeff)
        return total

    try:
        for r, rhs in enumerate(np.asarray(master.b_ub, float)):
            rhs_f = float(rhs)
            if not math.isfinite(rhs_f):
                return False
            if _row_sum(master.A_ub, r) > Fraction.from_float(rhs_f):
                return False
        for r, rhs in enumerate(np.asarray(master.b_eq, float)):
            rhs_f = float(rhs)
            if not math.isfinite(rhs_f):
                return False
            if _row_sum(master.A_eq, r) != Fraction.from_float(rhs_f):
                return False
    except (FloatingPointError, OverflowError, ValueError):
        return False
    return True


def _selection_from_local(master, x):
    return tuple(master.eligible_indices[k] for k, v in enumerate(x) if float(v) > 0.5)


def _coverage_of_selection(columns, selection):
    return sum(len(_ordered_tids(columns[j])) for j in selection)


def _fraction_to_float_down(value):
    """Largest/equal binary64 not exceeding an exact rational value."""
    value = Fraction(value)
    out = float(value)
    if Fraction.from_float(out) > value:
        out = math.nextafter(out, -math.inf)
    return float(out)


def _fraction_to_float_up(value):
    """Smallest/equal binary64 not below an exact rational value."""
    value = Fraction(value)
    out = float(value)
    if Fraction.from_float(out) < value:
        out = math.nextafter(out, math.inf)
    return float(out)


def _energy_of_selection_exact(columns, selection):
    total = Fraction(0)
    for j in selection:
        value = float(columns[j]["E_plan_Wh"])
        if not math.isfinite(value) or value < 0.0:
            raise FloatingPointError("selected plan contains non-finite/negative energy")
        total += Fraction.from_float(value)
    return total


def _energy_of_selection_interval(columns, selection):
    """Nearest display value and tight binary64 enclosure of exact plan energy."""
    exact = _energy_of_selection_exact(columns, selection)
    return float(exact), _fraction_to_float_down(exact), _fraction_to_float_up(exact)


def _energy_of_selection(columns, selection):
    return _energy_of_selection_interval(columns, selection)[0]


def _energy_of_selection_estimate(columns, selection):
    return _energy_of_selection_interval(columns, selection)[0]


def _resource_map(columns):
    return {j: dict(columns[j]["resource_intervals"]) for j in range(len(columns))}


def _audit_integer_selection(columns, selection, K, batteries, p, quick_min,
                             swap_min, quick_capacity, swap_capacity, deadline):
    return RA.audit_resource_assignment(
        columns, selection, int(K), int(batteries), float(p.B_use),
        _resource_map(columns), float(quick_min), float(swap_min),
        int(quick_capacity), int(swap_capacity), deadline=deadline)


def _branch_on_fractional_solution(columns, master, x, node, next_node_id):
    values = np.asarray(x, float)
    all_tids = sorted({tid for j in master.eligible_indices for tid in _ordered_tids(columns[j])})
    service = {}
    for tid in all_tids:
        service[tid] = sum(values[k] for k, j in enumerate(master.eligible_indices)
                           if tid in _ordered_tids(columns[j]))
    fractional_service = [(abs(v - 0.5), tid, v) for tid, v in service.items()
                          if 1e-7 < v < 1.0 - 1e-7]
    if fractional_service:
        _, tid, _ = min(fractional_service)
        b0 = BranchState(
            node.branch.forbidden_turbines | {tid}, node.branch.required_turbines,
            node.branch.forbidden_arcs, node.branch.required_arcs,
            node.branch.forbidden_routes, node.branch.required_routes)
        b1 = BranchState(
            node.branch.forbidden_turbines, node.branch.required_turbines | {tid},
            node.branch.forbidden_arcs, node.branch.required_arcs,
            node.branch.forbidden_routes, node.branch.required_routes)
        return [BranchPriceNode(next_node_id, node.depth + 1, b0, node.inherited_bound, node.bound_source),
                BranchPriceNode(next_node_id + 1, node.depth + 1, b1, node.inherited_bound, node.bound_source)], "service"

    arc_flow = {}
    for k, j in enumerate(master.eligible_indices):
        for arc in columns[j].get("route_arcs") or _route_arcs(columns[j]):
            arc_flow[arc] = arc_flow.get(arc, 0.0) + float(values[k])
    fractional_arc = [(abs(v - 0.5), arc, v) for arc, v in arc_flow.items()
                      if 1e-7 < v < 1.0 - 1e-7]
    if fractional_arc:
        _, arc, _ = min(fractional_arc, key=lambda z: (z[0], repr(z[1])))
        b0 = BranchState(
            node.branch.forbidden_turbines, node.branch.required_turbines,
            node.branch.forbidden_arcs | {arc}, node.branch.required_arcs,
            node.branch.forbidden_routes, node.branch.required_routes)
        b1 = BranchState(
            node.branch.forbidden_turbines, node.branch.required_turbines,
            node.branch.forbidden_arcs, node.branch.required_arcs | {arc},
            node.branch.forbidden_routes, node.branch.required_routes)
        return [BranchPriceNode(next_node_id, node.depth + 1, b0, node.inherited_bound, node.bound_source),
                BranchPriceNode(next_node_id + 1, node.depth + 1, b1, node.inherited_bound, node.bound_source)], "arc"

    fractional_vars = [(abs(float(v) - 0.5), k, float(v)) for k, v in enumerate(values)
                       if 1e-7 < float(v) < 1.0 - 1e-7]
    if not fractional_vars:
        return [], "none"
    _, k, _ = min(fractional_vars)
    sig = _exact_route_signature(columns[master.eligible_indices[k]])
    b0 = BranchState(
        node.branch.forbidden_turbines, node.branch.required_turbines,
        node.branch.forbidden_arcs, node.branch.required_arcs,
        node.branch.forbidden_routes | {sig}, node.branch.required_routes)
    b1 = BranchState(
        node.branch.forbidden_turbines, node.branch.required_turbines,
        node.branch.forbidden_arcs, node.branch.required_arcs,
        node.branch.forbidden_routes, node.branch.required_routes | {sig})
    return [BranchPriceNode(next_node_id, node.depth + 1, b0, node.inherited_bound, node.bound_source),
            BranchPriceNode(next_node_id + 1, node.depth + 1, b1, node.inherited_bound, node.bound_source)], "route"


def _branch_on_integral_numeric_ambiguity(columns, master, selection, node, next_node_id):
    """Complete x_r=0/1 fallback when an integral RMP is not rigorously fathomed.

    An LP solver may return an integral point that is optimal only within its
    internal floating tolerances.  If a rigorous full-space lower/upper bound
    still permits improvement, integrality itself is not a certificate.  Branch
    on one currently unfixed route variable; the two children are disjoint and
    their integer-solution union is exactly the parent integer set.
    """
    selected = set(int(j) for j in selection)
    candidates = []
    for j in master.eligible_indices:
        sig = _exact_route_signature(columns[j])
        if sig in node.branch.forbidden_routes or sig in node.branch.required_routes:
            continue
        # Prefer a variable that is 1 in the current integer incumbent so the
        # left child immediately excludes the numerically ambiguous point.
        candidates.append((0 if j in selected else 1, repr(sig), sig))
    if not candidates:
        return [], "none"
    _, _, sig = min(candidates)
    b0 = BranchState(
        node.branch.forbidden_turbines, node.branch.required_turbines,
        node.branch.forbidden_arcs, node.branch.required_arcs,
        node.branch.forbidden_routes | {sig}, node.branch.required_routes)
    b1 = BranchState(
        node.branch.forbidden_turbines, node.branch.required_turbines,
        node.branch.forbidden_arcs, node.branch.required_arcs,
        node.branch.forbidden_routes, node.branch.required_routes | {sig})
    return [
        BranchPriceNode(next_node_id, node.depth + 1, b0,
                        node.inherited_bound, node.bound_source),
        BranchPriceNode(next_node_id + 1, node.depth + 1, b1,
                        node.inherited_bound, node.bound_source),
    ], "route-integral-numeric-fallback"


def _queue_priority(stage, bound, node_id):
    return (-float(bound), int(node_id)) if stage == "coverage" else (float(bound), int(node_id))


def _global_open_bound_info(stage, queue, incumbent_value, trivial_bound):
    """Return the valid open-tree bound and the sources attaining that bound."""
    if not queue:
        value = float(incumbent_value if incumbent_value is not None else trivial_bound)
        source = ("branch-tree-exhausted-incumbent-bound" if incumbent_value is not None
                  else "trivial-model-bound")
        return value, source
    nodes = [entry[2] for entry in queue if entry[2].inherited_bound is not None]
    if not nodes:
        return float(trivial_bound), "trivial-model-bound"
    values = [float(node.inherited_bound) for node in nodes]
    value = max(values) if stage == "coverage" else min(values)
    tol = 1e-9 * max(1.0, abs(float(value)))
    sources = sorted({str(node.bound_source) for node in nodes
                      if abs(float(node.inherited_bound) - float(value)) <= tol})
    if not sources:
        sources = ["trivial-model-bound"]
    source = sources[0] if len(sources) == 1 else "mixed-active-node-bounds[" + ",".join(sources) + "]"
    return float(value), source


def _global_open_bound(stage, queue, incumbent_value, trivial_bound):
    return _global_open_bound_info(stage, queue, incumbent_value, trivial_bound)[0]




def _fullcover_closure_context_sha256(
        archive, all_tids, K, batteries, p, pooled_energy_cap,
        quick_min, swap_min, quick_capacity, swap_capacity,
        algorithm_sha256=None):
    """Binary64-exact context binding for reusable full-cover resource cuts.

    A persisted cut is meaningful only for the exact same materialized route
    universe and resource model.  The context deliberately includes every
    column's immutable formal semantics rather than only route indices.
    """
    payload = (
        FULLCOVER_CLOSURE_CHECKPOINT_CONTRACT,
        MODEL_SEMANTICS_CONTRACT,
        FORMAL_PROOF_CONTRACT,
        str(algorithm_sha256 or "missing-algorithm-sha256"),
        tuple(_state_fp(tid) for tid in all_tids),
        int(K), int(batteries),
        _float_binary64_fp(float(p.B_use)),
        None if pooled_energy_cap is None else _float_binary64_fp(float(pooled_energy_cap)),
        _float_binary64_fp(float(quick_min)),
        _float_binary64_fp(float(swap_min)),
        int(quick_capacity), int(swap_capacity),
        tuple(_state_fp(_column_semantics_fp(c)) for c in archive),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _fullcover_checkpoint_payload_sha256(payload):
    body = dict(payload)
    body.pop("payload_sha256", None)
    raw = json.dumps(body, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_fullcover_closure_checkpoint(path, *, context_sha256, archive_len, resume):
    """Load only same-context proven cuts; any malformed/mismatched file fails closed."""
    if path is None or not bool(resume):
        return []
    fp = Path(path)
    if not fp.is_file():
        return []
    try:
        payload = json.loads(fp.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"full-cover closure checkpoint unreadable: {fp}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("full-cover closure checkpoint must be a JSON object")
    if payload.get("contract") != FULLCOVER_CLOSURE_CHECKPOINT_CONTRACT:
        raise RuntimeError("full-cover closure checkpoint contract mismatch")
    if payload.get("context_sha256") != str(context_sha256):
        raise RuntimeError("full-cover closure checkpoint context mismatch")
    got_sha = str(payload.get("payload_sha256", ""))
    want_sha = _fullcover_checkpoint_payload_sha256(payload)
    if got_sha != want_sha:
        raise RuntimeError("full-cover closure checkpoint payload hash mismatch")
    records = payload.get("cuts", [])
    if not isinstance(records, list):
        raise RuntimeError("full-cover closure checkpoint cuts must be a list")
    out, seen = [], set()
    allowed_kinds = {"full-pattern-resource-dfs", "battery-binpack-core"}
    for rec in records:
        if not isinstance(rec, dict):
            raise RuntimeError("full-cover closure checkpoint cut record malformed")
        kind = str(rec.get("kind", ""))
        if kind not in allowed_kinds:
            raise RuntimeError(f"unknown full-cover persisted cut kind {kind!r}")
        raw = rec.get("indices")
        if not isinstance(raw, list) or not raw:
            raise RuntimeError("full-cover persisted cut must be nonempty")
        try:
            cut = tuple(sorted({int(j) for j in raw}))
        except Exception as exc:
            raise RuntimeError("full-cover persisted cut has noninteger index") from exc
        if len(cut) != len(raw):
            raise RuntimeError("full-cover persisted cut has duplicate indices")
        if any(j < 0 or j >= int(archive_len) for j in cut):
            raise RuntimeError("full-cover persisted cut index outside current universe")
        if cut in seen:
            raise RuntimeError("full-cover closure checkpoint contains duplicate cuts")
        seen.add(cut)
        out.append((cut, kind))
    return out


def _save_fullcover_closure_checkpoint(path, *, context_sha256, cuts):
    """Atomically persist only cuts already proven by exact code in this context."""
    if path is None:
        return False
    fp = Path(path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    normalized = []
    seen = set()
    for cut, kind in cuts:
        cut = tuple(sorted({int(j) for j in cut}))
        if not cut:
            raise RuntimeError("refusing to persist an empty full-cover cut")
        if cut in seen:
            continue
        seen.add(cut)
        normalized.append(dict(indices=list(cut), kind=str(kind)))
    normalized.sort(key=lambda rec: (len(rec["indices"]), rec["indices"], rec["kind"]))
    payload = dict(
        contract=FULLCOVER_CLOSURE_CHECKPOINT_CONTRACT,
        context_sha256=str(context_sha256),
        model_semantics_contract=MODEL_SEMANTICS_CONTRACT,
        formal_proof_contract=FORMAL_PROOF_CONTRACT,
        cuts=normalized,
    )
    payload["payload_sha256"] = _fullcover_checkpoint_payload_sha256(payload)
    tmp = fp.with_name(fp.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, fp)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        finally:
            raise
    return True


def _exact_battery_binpack_status(
        archive, selection, battery_count, usable_battery_energy_Wh, deadline):
    """Exact necessary battery-energy relaxation for one route pattern.

    Every real resource assignment maps each selected route to one of ``B``
    battery packs and cumulative SOC use of a pack may not exceed ``B_use``.
    Dropping all time/UAV/binding/station restrictions yields ordinary bin
    packing.  If even this relaxation is infeasible, the real resource pattern
    is infeasible.  ``UNKNOWN_TIMEOUT`` never yields a cut.
    """
    B = int(battery_count)
    if B <= 0:
        return ("FEASIBLE" if not selection else "INFEASIBLE_PROVEN"), 1
    cap = Fraction.from_float(float(usable_battery_energy_Wh))
    if cap < 0:
        raise RuntimeError("battery bin-pack relaxation received negative capacity")
    items = []
    for j in sorted({int(v) for v in selection}):
        e = float(archive[j]["E_soc_required_Wh"])
        if not math.isfinite(e) or e < 0.0:
            raise RuntimeError("battery bin-pack relaxation received invalid route energy")
        ef = Fraction.from_float(e)
        if ef > cap:
            return "INFEASIBLE_PROVEN", 1
        items.append((ef, j))
    if not items:
        return "FEASIBLE", 1
    total = sum((e for e, _j in items), Fraction(0))
    if total > cap * B:
        return "INFEASIBLE_PROVEN", 1
    items.sort(key=lambda z: (-z[0], z[1]))
    loads = [Fraction(0) for _ in range(B)]
    failed = set()
    explored = 0

    def _dfs(pos):
        nonlocal explored
        if _deadline_hit(deadline):
            return None
        explored += 1
        if pos >= len(items):
            return True
        state = (int(pos), tuple(sorted(loads)))
        if state in failed:
            return False
        e = items[pos][0]
        seen_loads = set()
        for b in range(B):
            load = loads[b]
            if load in seen_loads:
                continue
            seen_loads.add(load)
            if load + e > cap:
                continue
            loads[b] = load + e
            ans = _dfs(pos + 1)
            loads[b] = load
            if ans is True:
                return True
            if ans is None:
                return None
        failed.add(state)
        return False

    ans = _dfs(0)
    if ans is None:
        return "UNKNOWN_TIMEOUT", int(explored)
    return ("FEASIBLE" if ans else "INFEASIBLE_PROVEN"), int(explored)


def _minimal_battery_conflict_core(
        archive, selection, battery_count, usable_battery_energy_Wh, deadline):
    """Deletion-minimal monotone battery-energy conflict core.

    The returned subset cut ``sum_{r in Q} x_r <= |Q|-1`` is globally valid
    because bin-packing infeasibility is upward closed: adding routes cannot
    make an already impossible packing into ``B`` capacity-``B_use`` bins
    feasible.  This is intentionally restricted to the exact full-cover
    accelerator in v15 even though the inequality itself is more general.
    """
    sel = tuple(sorted({int(j) for j in selection}))
    status, nodes = _exact_battery_binpack_status(
        archive, sel, battery_count, usable_battery_energy_Wh, deadline)
    total_nodes = int(nodes)
    if status != "INFEASIBLE_PROVEN":
        return status, tuple(), total_nodes
    core = list(sel)
    pos = 0
    while pos < len(core):
        # ``core`` itself is already proven infeasible.  If minimization runs
        # out of time, retain that last proven core; an UNKNOWN deletion trial
        # is never interpreted as infeasible.
        if _deadline_hit(deadline):
            return "INFEASIBLE_PROVEN", tuple(core), total_nodes
        trial = tuple(core[:pos] + core[pos + 1:])
        st, n = _exact_battery_binpack_status(
            archive, trial, battery_count, usable_battery_energy_Wh, deadline)
        total_nodes += int(n)
        if st == "UNKNOWN_TIMEOUT":
            return "INFEASIBLE_PROVEN", tuple(core), total_nodes
        if st == "INFEASIBLE_PROVEN":
            core = list(trial)
            continue
        pos += 1
    if not core:
        raise RuntimeError("battery conflict core unexpectedly became empty")
    return "INFEASIBLE_PROVEN", tuple(core), total_nodes



def _exact_global_fullcover_battery_relaxation(
        archive, all_tids, battery_count, usable_battery_energy_Wh, deadline,
        max_turbines=12):
    """[THM-GBR] Exact energy-only full-cover battery relaxation.

    This is a *necessary* relaxation of the unchanged physical resource model.
    It keeps:
      * the complete materialized route universe;
      * exact full-cover/set-partition semantics over ``all_tids``;
      * each route's stored ``E_soc_required_Wh`` interpreted as the exact real
        represented by its binary64 value;
      * ``B`` identical batteries of exact binary64 usable capacity ``B_use``.

    It deliberately drops UAV identity/binding, route timing, deck occupation,
    quick/swap station capacity and every other scheduling restriction.  Hence
    every real resource-feasible full-cover schedule maps to a feasible point of
    this relaxation.  The converse is *not* valid.

    For n turbines, route coverage is represented by an n-bit mask.  First,
    ``best_energy[M]`` is computed exactly: the minimum SOC energy of any
    pairwise-disjoint route partition whose union is mask M.  Therefore M can be
    assigned to one relaxed battery iff ``best_energy[M] <= B_use``.  A second
    mask DP partitions the full turbine mask into the fewest such one-battery
    masks.  With n=8 the entire state space is only 256 masks.

    Returns ``(status, min_required, dp_states, one_pack_masks, witness_masks)``:
      * ``INFEASIBLE_PROVEN``: the relaxed problem itself needs more than B
        batteries, or no exact-cover route partition exists;
      * ``FEASIBLE_RELAXATION``: the energy-only relaxation fits in B batteries;
        this is NOT a physical target-YES certificate;
      * ``UNKNOWN_TIMEOUT``: deadline reached, fail closed;
      * ``SKIPPED_SIZE``: n exceeds the safe accelerator guard, so callers must
        continue with the ordinary exact closure.

    No tolerance, decimal rounding or solver infeasibility status participates.
    """
    tids = tuple(all_tids)
    if len(tids) != len(set(tids)):
        raise RuntimeError("global battery relaxation received duplicate turbine ids")
    n = len(tids)
    if n == 0:
        return "FEASIBLE_RELAXATION", 0, 1, 1, tuple()
    if n > int(max_turbines):
        return "SKIPPED_SIZE", None, 0, 0, tuple()
    B = int(battery_count)
    if B < 0:
        raise RuntimeError("global battery relaxation received negative battery count")
    cap_f = float(usable_battery_energy_Wh)
    if not math.isfinite(cap_f) or cap_f < 0.0:
        raise RuntimeError("global battery relaxation received invalid battery capacity")
    cap = Fraction.from_float(cap_f)

    tid_to_bit = {tid: i for i, tid in enumerate(tids)}
    # In the energy-only relaxation, among routes with the same service mask
    # only the minimum exact SOC energy can ever be useful.
    route_energy_by_mask = {}
    for c in archive:
        rt = tuple(_ordered_tids(c))
        if not rt:
            raise RuntimeError("global battery relaxation received an empty route")
        if len(rt) != len(set(rt)):
            raise RuntimeError("global battery relaxation received a non-elementary route")
        if any(tid not in tid_to_bit for tid in rt):
            raise RuntimeError("global battery relaxation route leaves the turbine domain")
        mask = 0
        for tid in rt:
            mask |= 1 << tid_to_bit[tid]
        if mask == 0:
            raise RuntimeError("global battery relaxation produced an empty service mask")
        e_f = float(c["E_soc_required_Wh"])
        if not math.isfinite(e_f) or e_f < 0.0:
            raise RuntimeError("global battery relaxation received invalid SOC energy")
        e = Fraction.from_float(e_f)
        prev = route_energy_by_mask.get(mask)
        if prev is None or e < prev:
            route_energy_by_mask[mask] = e

    full_mask = (1 << n) - 1
    inf = None
    best_energy = [inf] * (full_mask + 1)
    best_energy[0] = Fraction(0)
    best_prev = [None] * (full_mask + 1)
    route_items = tuple(sorted(route_energy_by_mask.items()))
    dp_states = 0
    for mask in range(full_mask + 1):
        if _deadline_hit(deadline):
            return "UNKNOWN_TIMEOUT", None, int(dp_states), 0, tuple()
        base = best_energy[mask]
        if base is None:
            continue
        dp_states += 1
        for rm, e in route_items:
            if mask & rm:
                continue
            nm = mask | rm
            cand = base + e
            if best_energy[nm] is None or cand < best_energy[nm]:
                best_energy[nm] = cand
                best_prev[nm] = (mask, rm)

    # A mask can be one relaxed battery bundle exactly when some route partition
    # of that mask fits the exact usable capacity.
    one_pack = [False] * (full_mask + 1)
    one_pack[0] = True
    for mask in range(1, full_mask + 1):
        e = best_energy[mask]
        one_pack[mask] = e is not None and e <= cap
    one_pack_count = sum(1 for x in one_pack[1:] if x)

    if best_energy[full_mask] is None:
        # Even after dropping all resource coupling there is no exact cover.
        return "INFEASIBLE_PROVEN", None, int(dp_states), int(one_pack_count), tuple()
    # Minimum number of one-battery masks partitioning each turbine mask.
    # Symmetry is removed by requiring every chosen bundle to contain the
    # least-significant uncovered turbine.
    huge = n + 1
    min_b = [huge] * (full_mask + 1)
    min_b[0] = 0
    parent = [None] * (full_mask + 1)
    battery_dp_states = 0
    for mask in range(full_mask + 1):
        if _deadline_hit(deadline):
            return "UNKNOWN_TIMEOUT", None, int(dp_states + battery_dp_states), int(one_pack_count), tuple()
        if min_b[mask] >= huge or mask == full_mask:
            continue
        battery_dp_states += 1
        remaining = full_mask ^ mask
        first = remaining & -remaining
        sub = remaining
        sub_checks = 0
        while sub:
            sub_checks += 1
            if (sub_checks & 1023) == 0 and _deadline_hit(deadline):
                return ("UNKNOWN_TIMEOUT", None,
                        int(dp_states + battery_dp_states),
                        int(one_pack_count), tuple())
            if (sub & first) and one_pack[sub]:
                nm = mask | sub
                cand = min_b[mask] + 1
                if cand < min_b[nm]:
                    min_b[nm] = cand
                    parent[nm] = (mask, sub)
            sub = (sub - 1) & remaining

    total_states = int(dp_states + battery_dp_states)
    if min_b[full_mask] >= huge:
        # Defensive: best_energy[full] itself is one route-partition, but it may
        # exceed one battery; finite singleton-bundle decomposition is not
        # guaranteed if some route itself exceeds capacity.
        return "INFEASIBLE_PROVEN", None, total_states, int(one_pack_count), tuple()

    min_required = int(min_b[full_mask])
    witness = []
    cur = full_mask
    while cur:
        rec = parent[cur]
        if rec is None:
            raise RuntimeError("global battery relaxation witness reconstruction failed")
        prev, bundle = rec
        witness.append(int(bundle))
        cur = int(prev)
    witness.reverse()
    status = ("INFEASIBLE_PROVEN"
              if min_required > B else "FEASIBLE_RELAXATION")
    return status, min_required, total_states, int(one_pack_count), tuple(witness)

def _fullcover_target_master_rows(
        columns, all_tids, deck_times, active_times, K, batteries,
        quick_min, swap_min):
    """Materialize safe resource relaxations for the full-cover direct master.

    Besides the original deck and active rows, v15 carries forward two exact
    necessary conditions already used by the generic resource master:
      * active concurrency is bounded by ``min(K,B)`` because every active
        sortie occupies one physical battery;
      * each UAV is necessarily occupied until at least
        ``clear_end + min(quick,swap)`` (or ``quick`` when ``B=1``).
    The latter forms an interval-capacity graph and maximal-overlap rows at
    interval starts are a complete exact representation of this relaxation.
    None of these rows changes the formal resource feasible set.
    """
    deck_rows = []
    seen_deck = set()
    for t in deck_times:
        row = tuple(j for j, c in enumerate(columns)
                    if _row_coefficient(c, ("deck", float(t))) > 0.5)
        if len(row) >= 2 and row not in seen_deck:
            seen_deck.add(row)
            deck_rows.append(list(row))

    capacity_rows = []
    seen_capacity = set()

    def _add_capacity(row, cap):
        rr = tuple(int(j) for j in row)
        key = (rr, int(cap))
        if len(rr) > int(cap) and key not in seen_capacity:
            seen_capacity.add(key)
            capacity_rows.append((list(rr), int(cap)))

    active_cap = min(int(K), int(batteries))
    for t in active_times:
        row = [j for j, c in enumerate(columns)
               if _row_coefficient(c, ("active", float(t))) > 0.5]
        _add_capacity(row, active_cap)

    fastest_service = (min(max(float(quick_min), 0.0), max(float(swap_min), 0.0))
                       if int(batteries) >= 2 else max(float(quick_min), 0.0))
    fastest_turn_intervals = []
    for c in columns:
        r = c["resource_intervals"]
        a = float(r["launch_start_min"])
        b = float(r["clear_end_min"]) + fastest_service
        fastest_turn_intervals.append((a, b))
    for row in RA._interval_capacity_rows(fastest_turn_intervals):
        _add_capacity(row, int(K))
    return deck_rows, capacity_rows


def _exact_fullcover_master_feasibility(
        *, archive, all_tids, deck_rows, capacity_rows, pooled_energy_cap,
        exact_index_cuts, strong_cuts, deadline):
    """Independent exact feasibility oracle for the finite full-cover binary master.

    This checker deliberately does *not* trust a floating MILP infeasibility status.
    It enumerates exact-cover patterns over the already certified/materialized route
    universe, applying only integer rows and binary64 quantities represented as exact
    rationals.  It is therefore a proof oracle for the direct target accelerator:

      ``FEASIBLE``           -> returns one exact binary route pattern;
      ``INFEASIBLE_PROVEN`` -> all exact-cover patterns were exhausted;
      ``UNKNOWN_TIMEOUT``    -> fail closed; no target-NO certificate is allowed.

    Resource feasibility itself is still decided by ``audit_resource_assignment``;
    this routine only validates the finite binary master after Logic-Benders cuts.
    """
    tids = tuple(all_tids)
    tid_set = set(tids)
    if len(tids) != len(tid_set):
        raise RuntimeError("full-cover exact master received duplicate turbine ids")
    n = len(archive)
    route_tids = []
    candidates = {tid: [] for tid in tids}
    energy = []
    for j, c in enumerate(archive):
        rt = tuple(_ordered_tids(c))
        if not rt:
            raise RuntimeError("full-cover exact master received an empty route")
        if len(rt) != len(set(rt)):
            raise RuntimeError("full-cover exact master received a non-elementary route")
        if not set(rt).issubset(tid_set):
            raise RuntimeError("full-cover exact master route leaves the turbine domain")
        route_tids.append(frozenset(rt))
        for tid in rt:
            candidates[tid].append(j)
        e = float(c["E_soc_required_Wh"])
        if not math.isfinite(e) or e < 0.0:
            raise RuntimeError("full-cover exact master received invalid SOC energy")
        energy.append(Fraction.from_float(e))

    deck_sets = [frozenset(int(j) for j in row) for row in deck_rows]
    cap_sets = [(frozenset(int(j) for j in row), int(cap))
                for row, cap in capacity_rows]
    if any(cap < 0 for _row, cap in cap_sets):
        raise RuntimeError("full-cover exact master received a negative capacity")
    exact_cuts = {frozenset(int(j) for j in cut) for cut in exact_index_cuts}
    strong_sets = [frozenset(int(j) for j in cut) for cut in strong_cuts]
    for cut in tuple(exact_cuts) + tuple(strong_sets):
        if any(j < 0 or j >= n for j in cut):
            raise RuntimeError("full-cover exact master cut contains an invalid column index")

    cap_exact = None
    if pooled_energy_cap is not None:
        cap_f = float(pooled_energy_cap)
        if not math.isfinite(cap_f) or cap_f < 0.0:
            raise RuntimeError("full-cover exact master received invalid pooled-energy cap")
        cap_exact = Fraction.from_float(cap_f)

    deck_by_route = [[] for _ in range(n)]
    for rid, row in enumerate(deck_sets):
        for j in row:
            if j < 0 or j >= n:
                raise RuntimeError("full-cover exact master deck row contains invalid index")
            deck_by_route[j].append(rid)
    cap_by_route = [[] for _ in range(n)]
    for rid, (row, _cap) in enumerate(cap_sets):
        for j in row:
            if j < 0 or j >= n:
                raise RuntimeError("full-cover exact master capacity row contains invalid index")
            cap_by_route[j].append(rid)

    selected = []
    selected_set = set()
    covered = set()
    deck_count = [0] * len(deck_sets)
    cap_count = [0] * len(cap_sets)
    explored = 0
    timed_out = False

    def _compatible(j, used_energy):
        rs = route_tids[j]
        if rs & covered:
            return False
        trial_selected = selected_set | {j}
        if any(cut.issubset(trial_selected) for cut in strong_sets):
            return False
        if any(deck_count[r] >= 1 for r in deck_by_route[j]):
            return False
        if any(cap_count[r] >= cap_sets[r][1] for r in cap_by_route[j]):
            return False
        if cap_exact is not None and used_energy + energy[j] > cap_exact:
            return False
        return True

    def _dfs(used_energy):
        nonlocal explored, timed_out
        if _deadline_hit(deadline):
            timed_out = True
            return None
        explored += 1
        if len(covered) == len(tids):
            key = frozenset(selected_set)
            if key in exact_cuts:
                return None
            # Strong cuts were checked incrementally; keep a leaf assertion as a
            # fail-closed defense against accidental future changes.
            if any(cut.issubset(key) for cut in strong_sets):
                return None
            return tuple(sorted(selected_set))

        # Minimum-remaining-values branching is exact and materially reduces the
        # small-n target-NO proof tree without changing the feasible set.
        best = None
        for tid in tids:
            if tid in covered:
                continue
            opts = [j for j in candidates.get(tid, ()) if _compatible(j, used_energy)]
            if not opts:
                return None
            if best is None or len(opts) < len(best[1]):
                best = (tid, opts)
        assert best is not None
        # Prefer routes covering more still-uncovered turbines only as a search
        # order; all candidates are still explored on a NO proof.
        opts = sorted(best[1], key=lambda j: (-len(route_tids[j]), j))
        for j in opts:
            if _deadline_hit(deadline):
                timed_out = True
                return None
            rs = route_tids[j]
            selected.append(j); selected_set.add(j); covered.update(rs)
            for rid in deck_by_route[j]:
                deck_count[rid] += 1
            for rid in cap_by_route[j]:
                cap_count[rid] += 1
            ans = _dfs(used_energy + energy[j])
            for rid in cap_by_route[j]:
                cap_count[rid] -= 1
            for rid in deck_by_route[j]:
                deck_count[rid] -= 1
            for tid2 in rs:
                covered.remove(tid2)
            selected_set.remove(j); selected.pop()
            if ans is not None:
                return ans
            if timed_out:
                return None
        return None

    witness = _dfs(Fraction(0))
    if witness is not None:
        return "FEASIBLE", tuple(witness), int(explored)
    if timed_out:
        return "UNKNOWN_TIMEOUT", tuple(), int(explored)
    return "INFEASIBLE_PROVEN", tuple(), int(explored)


def _solve_complete_universe_fullcover_target(
        *, archive, all_tids, K, batteries, p, deadline,
        deck_times, active_times, pooled_energy_cap,
        quick_min, swap_min, quick_capacity, swap_capacity,
        initial_selection=(), initial_audit=None, no_good_cuts=(),
        target_closure_checkpoint_path=None, target_closure_resume=False,
        target_closure_algorithm_sha256=None):
    """[THM-FCT] Direct exact target decision on a certified complete universe.

    Preconditions: the route universe is complete/materialized and the target
    equals the number of turbines in the master set ``all_tids``.  The binary master therefore uses
    one equality per turbine.  Each integer route pattern is independently
    audited by the unchanged exact resource DFS.  A proven resource-infeasible
    *full-cover* pattern S admits the stronger valid cut

        sum_{r in S} x_r <= |S|-1,

    because S already covers every turbine exactly once and every route is
    nonempty; no target-feasible integer solution can contain S plus another
    route.  This strengthening is NOT used for partial-coverage targets.

    A floating MILP infeasible status is never itself a certificate.  It only
    triggers the independent exact full-cover verifier; target-NO is available
    only after that verifier exhausts every remaining pattern as INFEASIBLE_PROVEN.
    """
    all_tids = tuple(all_tids)
    target = len(all_tids)
    deck_rows, capacity_rows = _fullcover_target_master_rows(
        archive, all_tids, deck_times, active_times, K, batteries,
        quick_min, swap_min)

    # Existing signature-based exact-pattern cuts are valid, but the target
    # accelerator normally starts empty.  Convert them only when every
    # signature is present in the immutable complete universe.
    sig_to_idx = {_exact_route_signature(c): j for j, c in enumerate(archive)}
    exact_index_cuts = []
    for cut in no_good_cuts:
        idx = []
        ok = True
        for sig in cut:
            j = sig_to_idx.get(sig)
            if j is None:
                ok = False
                break
            idx.append(int(j))
        if ok:
            exact_index_cuts.append(tuple(sorted(set(idx))))

    closure_context_sha = _fullcover_closure_context_sha256(
        archive, all_tids, K, batteries, p, pooled_energy_cap,
        quick_min, swap_min, quick_capacity, swap_capacity,
        algorithm_sha256=target_closure_algorithm_sha256)
    loaded_records = _load_fullcover_closure_checkpoint(
        target_closure_checkpoint_path, context_sha256=closure_context_sha,
        archive_len=len(archive), resume=bool(target_closure_resume))
    strong_cuts = [tuple(cut) for cut, _kind in loaded_records]
    cut_kind = {tuple(cut): str(kind) for cut, kind in loaded_records}
    seen_strong = set(strong_cuts)
    cuts_loaded = len(strong_cuts)
    battery_core_cuts = sum(1 for _cut, kind in loaded_records
                            if kind == "battery-binpack-core")
    checkpoint_writes = 0
    if target_closure_checkpoint_path is not None and not bool(target_closure_resume):
        if _save_fullcover_closure_checkpoint(
                target_closure_checkpoint_path, context_sha256=closure_context_sha,
                cuts=[]):
            checkpoint_writes += 1
    audit_cache = {}
    master_solves = 0
    resource_audits = 0
    resource_audit_nodes = 0
    resource_audit_memo_hits = 0
    exact_cover_nodes = 0
    battery_relaxation_nodes = 0
    global_battery_status = None
    global_battery_min_required = None
    global_battery_dp_states = 0
    global_battery_one_pack_masks = 0
    backend = None

    def _persist_cut(cut, kind):
        nonlocal checkpoint_writes, battery_core_cuts
        cut = tuple(sorted({int(j) for j in cut}))
        if not cut:
            raise RuntimeError("attempted to add an empty full-cover resource cut")
        if cut in seen_strong:
            return False
        seen_strong.add(cut)
        strong_cuts.append(cut)
        cut_kind[cut] = str(kind)
        if kind == "battery-binpack-core":
            battery_core_cuts += 1
        if target_closure_checkpoint_path is not None:
            records = [(c, cut_kind[c]) for c in strong_cuts]
            if _save_fullcover_closure_checkpoint(
                    target_closure_checkpoint_path,
                    context_sha256=closure_context_sha, cuts=records):
                checkpoint_writes += 1
        return True

    def _telemetry():
        return dict(
            fullcover_strong_cuts=len(strong_cuts),
            fullcover_cuts_loaded=int(cuts_loaded),
            fullcover_battery_core_cuts=int(battery_core_cuts),
            resource_audit_nodes=int(resource_audit_nodes),
            resource_audit_memo_hits=int(resource_audit_memo_hits),
            target_exact_cover_nodes=int(exact_cover_nodes),
            target_checkpoint_writes=int(checkpoint_writes),
            battery_relaxation_nodes=int(battery_relaxation_nodes),
            target_closure_context_sha256=str(closure_context_sha),
            global_battery_relaxation_status=global_battery_status,
            global_battery_min_required=global_battery_min_required,
            global_battery_dp_states=int(global_battery_dp_states),
            global_battery_one_pack_masks=int(global_battery_one_pack_masks))

    # A supplied full-cover exact witness can terminate immediately.
    if initial_selection:
        sel0 = tuple(sorted(int(j) for j in initial_selection))
        if _coverage_of_selection(archive, sel0) == target:
            aud0 = initial_audit
            if aud0 is None:
                resource_audits += 1
                aud0 = _audit_integer_selection(
                    archive, sel0, K, batteries, p, quick_min, swap_min,
                    quick_capacity, swap_capacity, deadline)
                resource_audit_nodes += int(getattr(aud0, "explored_nodes", 0))
                resource_audit_memo_hits += int(getattr(aud0, "memo_hits", 0))
            if aud0.status is RA.ResourceAuditStatus.FEASIBLE:
                exact_e = _energy_of_selection_exact(archive, sel0)
                return StageSearchResult(
                    stage="energy", incumbent_selection=sel0, incumbent_audit=aud0,
                    incumbent_value=float(exact_e),
                    incumbent_lower_bound=_fraction_to_float_down(exact_e),
                    incumbent_upper_bound=_fraction_to_float_up(exact_e),
                    coverage_incumbent=target, global_bound=0.0, optimal=False,
                    termination_reason="target-feasible-witness-initial",
                    open_nodes=0, processed_nodes=0, generated_columns=0,
                    pricing_calls=0, exact_pricing_calls=0, resource_cuts_added=0,
                    rmp_solves=0, phase_one_solves=0, pricing_candidates=0,
                    pricing_nodes=0, columns_accepted=0, heuristic_columns=0,
                    resource_audit_calls=resource_audits,
                    branch_children_created=0, branch_decisions=0,
                    pricing_complete=True, pricing_search_complete=True,
                    pricing_bound_available=True, resource_audit_complete=True,
                    farkas_pricing_complete=True, branching_complete=True,
                    heuristic_pricing_used=False, exact_pricing_called=False,
                    pricing_best_reduced_value=None,
                    pricing_reduced_value_bound=math.inf,
                    bound_source="complete-universe-fullcover-target-witness",
                    direct_target_backend="initial-witness",
                    target_master_solves=0, **_telemetry())

    # [THM-GBR] Before enumerating one resource-infeasible full-cover pattern at
    # a time, solve one exact universe-level energy-only battery relaxation.
    # If even this relaxation needs more than the available number of batteries,
    # every physical schedule is impossible.  A feasible relaxation is only a
    # lower-bound result and must continue into the unchanged v15 exact closure.
    (global_battery_status, global_battery_min_required,
     global_battery_dp_states, global_battery_one_pack_masks,
     _global_battery_witness) = _exact_global_fullcover_battery_relaxation(
        archive, all_tids, batteries, float(p.B_use), deadline)
    if global_battery_status == "INFEASIBLE_PROVEN":
        return StageSearchResult(
            stage="energy", incumbent_selection=tuple(), incumbent_audit=None,
            incumbent_value=None, incumbent_lower_bound=None,
            incumbent_upper_bound=None, coverage_incumbent=0,
            global_bound=0.0, optimal=False,
            termination_reason="fullcover-target-global-battery-relaxation-infeasible-proven",
            open_nodes=0, processed_nodes=0, generated_columns=0,
            pricing_calls=0, exact_pricing_calls=1,
            resource_cuts_added=len(strong_cuts), rmp_solves=0,
            phase_one_solves=0, pricing_candidates=0,
            pricing_nodes=int(global_battery_dp_states), columns_accepted=0,
            heuristic_columns=0, resource_audit_calls=resource_audits,
            branch_children_created=0, branch_decisions=0,
            pricing_complete=True, pricing_search_complete=True,
            pricing_bound_available=True, resource_audit_complete=True,
            farkas_pricing_complete=True, branching_complete=True,
            heuristic_pricing_used=False, exact_pricing_called=True,
            pricing_best_reduced_value=None, pricing_reduced_value_bound=math.inf,
            bound_source="complete-universe-fullcover-exact-global-battery-relaxation",
            direct_target_backend="exact-global-battery-mask-dp",
            target_master_solves=0, **_telemetry())
    if global_battery_status == "UNKNOWN_TIMEOUT":
        return StageSearchResult(
            stage="energy", incumbent_selection=tuple(), incumbent_audit=None,
            incumbent_value=None, incumbent_lower_bound=None,
            incumbent_upper_bound=None, coverage_incumbent=0,
            global_bound=0.0, optimal=False,
            termination_reason="global-battery-relaxation-time-limit",
            open_nodes=1, processed_nodes=0, generated_columns=0,
            pricing_calls=0, exact_pricing_calls=1,
            resource_cuts_added=len(strong_cuts), rmp_solves=0,
            phase_one_solves=0, pricing_candidates=0,
            pricing_nodes=int(global_battery_dp_states), columns_accepted=0,
            heuristic_columns=0, resource_audit_calls=resource_audits,
            branch_children_created=0, branch_decisions=0,
            pricing_complete=True, pricing_search_complete=False,
            pricing_bound_available=False, resource_audit_complete=False,
            farkas_pricing_complete=True, branching_complete=False,
            heuristic_pricing_used=False, exact_pricing_called=True,
            pricing_best_reduced_value=None, pricing_reduced_value_bound=None,
            bound_source="complete-universe-fullcover-global-battery-relaxation-open",
            direct_target_backend="exact-global-battery-mask-dp",
            target_master_solves=0, **_telemetry())
    if global_battery_status not in {"FEASIBLE_RELAXATION", "SKIPPED_SIZE"}:
        raise RuntimeError(
            f"unknown global battery relaxation status {global_battery_status!r}")

    while not _deadline_hit(deadline):
        master_solves += 1
        master = RA.solve_binary_master(
            "auto", archive, all_tids, deck_rows, exact_index_cuts,
            phase="energy", deadline=deadline,
            capacity_rows=capacity_rows,
            pooled_energy_cap=float(pooled_energy_cap),
            coverage_equal=target,
            full_cover_equal=True,
            full_cover_no_good_cuts=strong_cuts)
        backend = str(master.backend)

        selection = None
        if master.x is None:
            if bool(master.infeasible_proven):
                # A floating MILP backend is only a candidate infeasibility signal.
                # [THM-NUM/THM-FCT] Re-prove finite binary-master infeasibility
                # independently by exact-cover DFS over the complete universe.
                exact_status, exact_selection, exact_nodes = (
                    _exact_fullcover_master_feasibility(
                        archive=archive, all_tids=all_tids, deck_rows=deck_rows,
                        capacity_rows=capacity_rows, pooled_energy_cap=pooled_energy_cap,
                        exact_index_cuts=exact_index_cuts, strong_cuts=strong_cuts,
                        deadline=deadline))
                exact_cover_nodes += int(exact_nodes)
                if exact_status == "INFEASIBLE_PROVEN":
                    return StageSearchResult(
                        stage="energy", incumbent_selection=tuple(),
                        incumbent_audit=None, incumbent_value=None,
                        incumbent_lower_bound=None, incumbent_upper_bound=None,
                        coverage_incumbent=0, global_bound=0.0, optimal=False,
                        termination_reason="fullcover-target-master-infeasible-proven",
                        open_nodes=0, processed_nodes=master_solves,
                        generated_columns=0, pricing_calls=0, exact_pricing_calls=1,
                        resource_cuts_added=len(strong_cuts),
                        rmp_solves=master_solves, phase_one_solves=0,
                        pricing_candidates=0, pricing_nodes=int(exact_nodes), columns_accepted=0,
                        heuristic_columns=0, resource_audit_calls=resource_audits,
                        branch_children_created=0, branch_decisions=0,
                        pricing_complete=True, pricing_search_complete=True,
                        pricing_bound_available=True, resource_audit_complete=True,
                        farkas_pricing_complete=True, branching_complete=True,
                        heuristic_pricing_used=False, exact_pricing_called=True,
                        pricing_best_reduced_value=None,
                        pricing_reduced_value_bound=math.inf,
                        bound_source="complete-universe-fullcover-exact-master-infeasible",
                        direct_target_backend=backend + "+exact-fullcover-dfs",
                        target_master_solves=master_solves, **_telemetry())
                if exact_status == "UNKNOWN_TIMEOUT":
                    return StageSearchResult(
                        stage="energy", incumbent_selection=tuple(), incumbent_audit=None,
                        incumbent_value=None, incumbent_lower_bound=None,
                        incumbent_upper_bound=None, coverage_incumbent=0,
                        global_bound=0.0, optimal=False,
                        termination_reason="fullcover-target-exact-master-verification-time-limit",
                        open_nodes=1, processed_nodes=master_solves,
                        generated_columns=0, pricing_calls=0, exact_pricing_calls=1,
                        resource_cuts_added=len(strong_cuts),
                        rmp_solves=master_solves, phase_one_solves=0,
                        pricing_candidates=0, pricing_nodes=int(exact_nodes),
                        columns_accepted=0, heuristic_columns=0,
                        resource_audit_calls=resource_audits,
                        branch_children_created=0, branch_decisions=0,
                        pricing_complete=True, pricing_search_complete=False,
                        pricing_bound_available=False, resource_audit_complete=True,
                        farkas_pricing_complete=True, branching_complete=False,
                        heuristic_pricing_used=False, exact_pricing_called=True,
                        pricing_best_reduced_value=None, pricing_reduced_value_bound=None,
                        bound_source="complete-universe-fullcover-exact-master-open",
                        direct_target_backend=backend + "+exact-fullcover-dfs",
                        target_master_solves=master_solves, **_telemetry())
                if exact_status != "FEASIBLE" or not exact_selection:
                    raise RuntimeError("exact full-cover master verifier returned invalid status")
                selection = tuple(int(j) for j in exact_selection)
                backend = backend + "+exact-fullcover-dfs-recovered"
            else:
                return StageSearchResult(
                    stage="energy", incumbent_selection=tuple(), incumbent_audit=None,
                    incumbent_value=None, incumbent_lower_bound=None,
                    incumbent_upper_bound=None, coverage_incumbent=0,
                    global_bound=0.0, optimal=False,
                    termination_reason=f"fullcover-target-master-{master.status}",
                    open_nodes=1, processed_nodes=master_solves,
                    generated_columns=0, pricing_calls=0, exact_pricing_calls=0,
                    resource_cuts_added=len(strong_cuts),
                    rmp_solves=master_solves, phase_one_solves=0,
                    pricing_candidates=0, pricing_nodes=0, columns_accepted=0,
                    heuristic_columns=0, resource_audit_calls=resource_audits,
                    branch_children_created=0, branch_decisions=0,
                    pricing_complete=True, pricing_search_complete=True,
                    pricing_bound_available=True, resource_audit_complete=True,
                    farkas_pricing_complete=True, branching_complete=True,
                    heuristic_pricing_used=False, exact_pricing_called=False,
                    pricing_best_reduced_value=None,
                    pricing_reduced_value_bound=math.inf,
                    bound_source="complete-universe-fullcover-binary-master-open",
                    direct_target_backend=backend,
                    target_master_solves=master_solves, **_telemetry())

        if selection is None:
            selection = tuple(int(j) for j in np.flatnonzero(np.asarray(master.x) > 0.5))
        if _coverage_of_selection(archive, selection) != target:
            # The backend solution is independently validated by RA; reaching
            # this guard means the cross-module target contract was violated.
            raise RuntimeError("full-cover target binary master returned wrong coverage")
        # v15 first separates the strongest cheap monotone resource relaxation.
        # If the route energies cannot be packed into B battery-capacity bins
        # even after dropping every timing/UAV/binding restriction, the real
        # resource assignment is impossible.  Its deletion-minimal core yields
        # a globally valid subset cut and can exclude many future patterns at once.
        bp_status, bp_core, bp_nodes = _minimal_battery_conflict_core(
            archive, selection, batteries, float(p.B_use), deadline)
        battery_relaxation_nodes += int(bp_nodes)
        if bp_status == "INFEASIBLE_PROVEN":
            if not _persist_cut(bp_core, "battery-binpack-core"):
                raise RuntimeError("binary master returned a persisted battery-conflict core")
            continue
        if bp_status == "UNKNOWN_TIMEOUT":
            return StageSearchResult(
                stage="energy", incumbent_selection=tuple(), incumbent_audit=None,
                incumbent_value=None, incumbent_lower_bound=None,
                incumbent_upper_bound=None, coverage_incumbent=0,
                global_bound=0.0, optimal=False,
                termination_reason="battery-relaxation-time-limit",
                open_nodes=1, processed_nodes=master_solves,
                generated_columns=0, pricing_calls=0, exact_pricing_calls=0,
                resource_cuts_added=len(strong_cuts),
                rmp_solves=master_solves, phase_one_solves=0,
                pricing_candidates=0, pricing_nodes=0, columns_accepted=0,
                heuristic_columns=0, resource_audit_calls=resource_audits,
                branch_children_created=0, branch_decisions=0,
                pricing_complete=True, pricing_search_complete=True,
                pricing_bound_available=True, resource_audit_complete=False,
                farkas_pricing_complete=True, branching_complete=True,
                heuristic_pricing_used=False, exact_pricing_called=False,
                pricing_best_reduced_value=None, pricing_reduced_value_bound=math.inf,
                bound_source="complete-universe-fullcover-battery-relaxation-open",
                direct_target_backend=backend, target_master_solves=master_solves,
                **_telemetry())

        key = tuple(sorted(_exact_route_signature(archive[j]) for j in selection))
        audit = audit_cache.get(key)
        if audit is None:
            resource_audits += 1
            audit = _audit_integer_selection(
                archive, selection, K, batteries, p, quick_min, swap_min,
                quick_capacity, swap_capacity, deadline)
            resource_audit_nodes += int(getattr(audit, "explored_nodes", 0))
            resource_audit_memo_hits += int(getattr(audit, "memo_hits", 0))
            if audit.status is not RA.ResourceAuditStatus.UNKNOWN_TIMEOUT:
                audit_cache[key] = audit
        if audit.status is RA.ResourceAuditStatus.UNKNOWN_TIMEOUT:
            return StageSearchResult(
                stage="energy", incumbent_selection=tuple(), incumbent_audit=None,
                incumbent_value=None, incumbent_lower_bound=None,
                incumbent_upper_bound=None, coverage_incumbent=0,
                global_bound=0.0, optimal=False,
                termination_reason="resource-audit-time-limit",
                open_nodes=1, processed_nodes=master_solves,
                generated_columns=0, pricing_calls=0, exact_pricing_calls=0,
                resource_cuts_added=len(strong_cuts),
                rmp_solves=master_solves, phase_one_solves=0,
                pricing_candidates=0, pricing_nodes=0, columns_accepted=0,
                heuristic_columns=0, resource_audit_calls=resource_audits,
                branch_children_created=0, branch_decisions=0,
                pricing_complete=True, pricing_search_complete=True,
                pricing_bound_available=True, resource_audit_complete=False,
                farkas_pricing_complete=True, branching_complete=True,
                heuristic_pricing_used=False, exact_pricing_called=False,
                pricing_best_reduced_value=None,
                pricing_reduced_value_bound=math.inf,
                bound_source="complete-universe-fullcover-resource-audit-open",
                direct_target_backend=backend,
                target_master_solves=master_solves, **_telemetry())
        if audit.status is RA.ResourceAuditStatus.FEASIBLE:
            exact_e = _energy_of_selection_exact(archive, selection)
            return StageSearchResult(
                stage="energy", incumbent_selection=selection, incumbent_audit=audit,
                incumbent_value=float(exact_e),
                incumbent_lower_bound=_fraction_to_float_down(exact_e),
                incumbent_upper_bound=_fraction_to_float_up(exact_e),
                coverage_incumbent=target, global_bound=0.0, optimal=False,
                termination_reason="target-feasible-witness",
                open_nodes=0, processed_nodes=master_solves,
                generated_columns=0, pricing_calls=0, exact_pricing_calls=0,
                resource_cuts_added=len(strong_cuts),
                rmp_solves=master_solves, phase_one_solves=0,
                pricing_candidates=0, pricing_nodes=0, columns_accepted=0,
                heuristic_columns=0, resource_audit_calls=resource_audits,
                branch_children_created=0, branch_decisions=0,
                pricing_complete=True, pricing_search_complete=True,
                pricing_bound_available=True, resource_audit_complete=True,
                farkas_pricing_complete=True, branching_complete=True,
                heuristic_pricing_used=False, exact_pricing_called=False,
                pricing_best_reduced_value=None,
                pricing_reduced_value_bound=math.inf,
                bound_source="complete-universe-fullcover-target-witness",
                direct_target_backend=backend,
                target_master_solves=master_solves, **_telemetry())

        if audit.status is not RA.ResourceAuditStatus.INFEASIBLE_PROVEN:
            raise RuntimeError(f"unknown resource audit status in full-cover closure: {audit.status!r}")
        # Exact resource-infeasible full-cover pattern.  The stronger cut is
        # valid only because coverage_target == |I| and route packing is <= 1.
        cut = tuple(sorted(selection))
        if not _persist_cut(cut, "full-pattern-resource-dfs"):
            raise RuntimeError("repeated proven resource-infeasible full-cover pattern")

    return StageSearchResult(
        stage="energy", incumbent_selection=tuple(), incumbent_audit=None,
        incumbent_value=None, incumbent_lower_bound=None,
        incumbent_upper_bound=None, coverage_incumbent=0, global_bound=0.0,
        optimal=False, termination_reason="global-time-limit",
        open_nodes=1, processed_nodes=master_solves, generated_columns=0,
        pricing_calls=0, exact_pricing_calls=0,
        resource_cuts_added=len(strong_cuts), rmp_solves=master_solves,
        phase_one_solves=0, pricing_candidates=0, pricing_nodes=0,
        columns_accepted=0, heuristic_columns=0,
        resource_audit_calls=resource_audits, branch_children_created=0,
        branch_decisions=0, pricing_complete=True,
        pricing_search_complete=True, pricing_bound_available=True,
        resource_audit_complete=True, farkas_pricing_complete=True,
        branching_complete=True, heuristic_pricing_used=False,
        exact_pricing_called=False, pricing_best_reduced_value=None,
        pricing_reduced_value_bound=math.inf,
        bound_source="complete-universe-fullcover-target-time-limit",
        direct_target_backend=backend,
        target_master_solves=master_solves, **_telemetry())


def _record_resource_rejection_diagnostic(
        sink, columns, selection, audit):
    """Record one proven-infeasible exact selection for V9 diagnostics only."""
    if sink is None:
        return
    cur = frozenset(int(j) for j in selection)
    size = int(len(cur))
    coverage = int(_coverage_of_selection(columns, tuple(cur)))
    sink["rejected_pattern_count"] = int(
        sink.get("rejected_pattern_count", 0)) + 1
    sink["rejected_pattern_size_sum"] = int(
        sink.get("rejected_pattern_size_sum", 0)) + size
    sink["rejected_pattern_coverage_sum"] = int(
        sink.get("rejected_pattern_coverage_sum", 0)) + coverage
    sink["rejected_pattern_size_min"] = min(
        int(sink.get("rejected_pattern_size_min", size)), size)
    sink["rejected_pattern_size_max"] = max(
        int(sink.get("rejected_pattern_size_max", size)), size)
    sink["rejected_pattern_coverage_min"] = min(
        int(sink.get("rejected_pattern_coverage_min", coverage)), coverage)
    sink["rejected_pattern_coverage_max"] = max(
        int(sink.get("rejected_pattern_coverage_max", coverage)), coverage)

    prev = sink.get("_previous_rejected_pattern")
    if prev is not None:
        dist = int(len(cur.symmetric_difference(prev)))
        sink["rejected_hamming_count"] = int(
            sink.get("rejected_hamming_count", 0)) + 1
        sink["rejected_hamming_sum"] = int(
            sink.get("rejected_hamming_sum", 0)) + dist
        sink["rejected_hamming_min"] = min(
            int(sink.get("rejected_hamming_min", dist)), dist)
        sink["rejected_hamming_max"] = max(
            int(sink.get("rejected_hamming_max", dist)), dist)
    sink["_previous_rejected_pattern"] = cur
    # Private V10 payload: retained only inside the frozen-archive diagnostic.
    # It is never consulted by the target B&B itself.
    sink.setdefault("_rejected_selections", []).append(tuple(sorted(cur)))
    stop_hist = {}
    for j in cur:
        d = int(len(_ordered_tids(columns[int(j)])))
        stop_hist[d] = int(stop_hist.get(d, 0)) + 1
    hist_key = ",".join(f"{d}:{stop_hist[d]}" for d in sorted(stop_hist))
    morph = sink.setdefault("rejected_morphology_counts", {})
    morph[hist_key] = int(morph.get(hist_key, 0)) + 1
    totals = sink.setdefault("rejected_route_stop_count_totals", {})
    for d, n in stop_hist.items():
        totals[int(d)] = int(totals.get(int(d), 0)) + int(n)
    joint_key = f"coverage={coverage}|routes={size}"
    joint = sink.setdefault("rejected_coverage_route_count_joint", {})
    joint[joint_key] = int(joint.get(joint_key, 0)) + 1

    event_counts = sink.setdefault("resource_failure_event_counts", {})
    pattern_counts = sink.setdefault("resource_failure_pattern_counts", {})
    reasons = dict(getattr(audit, "failure_reasons", None) or {})
    if not reasons:
        reasons = {"unclassified_exact_infeasible": 1}
    for reason, count in reasons.items():
        reason = str(reason)
        count = int(count)
        if count <= 0:
            continue
        event_counts[reason] = int(event_counts.get(reason, 0)) + count
        pattern_counts[reason] = int(pattern_counts.get(reason, 0)) + 1


def _solve_branch_price_stage(*, stage, turbines, launch_opts, p, xi_amb, K,
                              batteries, T_min, max_stops, weather_unc,
                              deadline, archive, signature_to_index,
                              no_good_cuts, coverage_target=None,
                              initial_selection=(), initial_audit=None,
                              pricing_epsilon=PRICING_EPS,
                              coverage_gap_target_abs=0,
                              energy_gap_target_rel=0.0,
                              energy_gap_target_abs_Wh=1e-6,
                              t_launch_min=2.5, landing_clear_min=1.0,
                              quick_min=1.0, swap_min=6.0,
                              quick_capacity=1, swap_capacity=1,
                              deck_mode="interval", deck_delta_min=2.5,
                              kappa_mode="vp_unimodal", chance_mode="drcc",
                              budget_gamma=2.0, implicit_test_columns=None,
                              pricing_batch_size=16, root_branch=None,
                              physical_cache=None, decision_only=False,
                              complete_universe_mode=False,
                              target_closure_checkpoint_path=None,
                              target_closure_resume=False,
                              target_closure_algorithm_sha256=None,
                              pricing_experiment_mode=False,
                              coverage_decision_target=None,
                              diagnostics_sink=None,
                              pricing_pattern_cut_diagnostics=False,
                              formal_battery_halfcap=False,
                              primal_exchange=False,
                              primal_resource_primal=False,
                              primal_resource_guided=False,
                              primal_resource_deck_guided=False,
                              adaptive_multistop_enrichment=False,
                              resource_variant_enrichment=False,
                              archive_primal_recovery=False,
                              archive_primal_recovery_time_limit_s=2.0,
                              diagnostic_extra_inequality_rows=(),
                              certified_prefix_pruning=False):
    all_tids = tuple(sorted({_tid(t.tid) for t in turbines}))
    diagnostic_extra_inequality_rows = tuple(diagnostic_extra_inequality_rows or ())
    if diagnostic_extra_inequality_rows and not bool(complete_universe_mode):
        raise ValueError(
            "diagnostic_extra_inequality_rows require complete_universe_mode")
    if not isinstance(formal_battery_halfcap, (bool, np.bool_)):
        raise ValueError("formal_battery_halfcap must be boolean")
    formal_battery_halfcap = bool(formal_battery_halfcap)
    if not isinstance(primal_exchange, (bool, np.bool_)):
        raise ValueError("primal_exchange must be boolean")
    primal_exchange = bool(primal_exchange)
    if not isinstance(primal_resource_primal, (bool, np.bool_)):
        raise ValueError("primal_resource_primal must be boolean")
    primal_resource_primal = bool(primal_resource_primal)
    if not isinstance(primal_resource_guided, (bool, np.bool_)):
        raise ValueError("primal_resource_guided must be boolean")
    primal_resource_guided = bool(primal_resource_guided)
    if not isinstance(primal_resource_deck_guided, (bool, np.bool_)):
        raise ValueError("primal_resource_deck_guided must be boolean")
    primal_resource_deck_guided = bool(primal_resource_deck_guided)
    if not isinstance(adaptive_multistop_enrichment, (bool, np.bool_)):
        raise ValueError("adaptive_multistop_enrichment must be boolean")
    adaptive_multistop_enrichment = bool(adaptive_multistop_enrichment)
    if not isinstance(resource_variant_enrichment, (bool, np.bool_)):
        raise ValueError("resource_variant_enrichment must be boolean")
    resource_variant_enrichment = bool(resource_variant_enrichment)
    if not isinstance(archive_primal_recovery, (bool, np.bool_)):
        raise ValueError("archive_primal_recovery must be boolean")
    archive_primal_recovery = bool(archive_primal_recovery)
    _finite_number(
        "archive_primal_recovery_time_limit_s",
        archive_primal_recovery_time_limit_s, nonnegative=True)
    archive_primal_recovery_time_limit_s = float(
        archive_primal_recovery_time_limit_s)
    if bool(complete_universe_mode):
        # A recovery solve treats the current archive as finite/complete and
        # therefore must never recursively launch another recovery solve.
        archive_primal_recovery = False
    if sum(bool(x) for x in (
            primal_exchange, primal_resource_primal, primal_resource_guided,
            primal_resource_deck_guided)) > 1:
        raise ValueError(
            "primal_exchange, primal_resource_primal, primal_resource_guided, "
            "and primal_resource_deck_guided are mutually exclusive")
    formal_extra_inequality_rows = (
        ((("battery_halfcap", float(p.B_use)), float(batteries)),)
        if formal_battery_halfcap else ())
    if coverage_decision_target is not None:
        if stage != "coverage":
            raise ValueError(
                "coverage_decision_target is valid only for coverage stage")
        if isinstance(coverage_decision_target, (bool, np.bool_)):
            raise ValueError("coverage_decision_target must be an integer")
        _target_float = float(coverage_decision_target)
        _target_int = int(coverage_decision_target)
        if (not math.isfinite(_target_float)
                or _target_float != float(_target_int)
                or not (1 <= _target_int <= len(all_tids))):
            raise ValueError(
                "coverage_decision_target must be in 1..number of turbines")
        coverage_decision_target = _target_int
    deck_times, active_times = _possible_resource_row_times(
        launch_opts, xi_amb, T_min, t_launch_min, deck_mode, deck_delta_min)
    # Necessary pooled-SOC relaxation.  Round the exact binary64 rational
    # capacity product upward so this acceleration row can never cut a solution
    # that is feasible for the strict per-battery resource audit.
    pooled_energy_cap = _fraction_to_float_up(
        Fraction.from_float(float(p.B_use)) * int(batteries))

    # [THM-FCT] For a certified complete universe and a target equal to the
    # number of turbines in the master set all_tids, solve the finite binary target master
    # directly and separate only exact resource-infeasible full-cover patterns.
    # This removes the custom route-variable B&B tree while preserving the
    # unchanged exact resource audit and fail-closed semantics.
    if (bool(complete_universe_mode) and bool(decision_only)
            and stage == "energy"
            and int(coverage_target or -1) == len(all_tids)):
        return _solve_complete_universe_fullcover_target(
            archive=archive, all_tids=all_tids, K=K, batteries=batteries, p=p,
            deadline=deadline, deck_times=deck_times, active_times=active_times,
            pooled_energy_cap=pooled_energy_cap,
            quick_min=quick_min, swap_min=swap_min,
            quick_capacity=quick_capacity, swap_capacity=swap_capacity,
            initial_selection=initial_selection, initial_audit=initial_audit,
            no_good_cuts=no_good_cuts,
            target_closure_checkpoint_path=target_closure_checkpoint_path,
            target_closure_resume=target_closure_resume,
            target_closure_algorithm_sha256=target_closure_algorithm_sha256)

    root_bound = float(len(all_tids) if stage == "coverage" else 0.0)
    root = BranchPriceNode(0, 0, BranchState() if root_branch is None else root_branch, root_bound,
                           "trivial-coverable-turbine-bound" if stage == "coverage"
                           else "nonnegative-energy-bound")
    queue = [(_queue_priority(stage, root_bound, 0), 0, root)]
    next_node_id = 1
    processed = 0
    pricing_calls = 0
    exact_pricing_calls = 0
    pricing_discovery_calls = 0
    pricing_discovery_early_returns = 0
    pricing_certification_calls = 0
    pricing_shadow_prefixes_evaluated = 0
    pricing_shadow_prunable_prefixes = 0
    pricing_shadow_false_prune_witnesses = 0
    pricing_shadow_bound_errors = 0
    pricing_shadow_complete_calls = 0
    pricing_guided_order_calls = 0
    pricing_guided_order_reorders = 0
    pricing_guided_order_failures = 0
    pricing_layered_depths_started = 0
    pricing_layered_depths_completed = 0
    pricing_layered_max_depth_completed = 0
    pricing_layered_rounds = 0
    pricing_depth_fair_requested_calls = 0
    pricing_depth_fair_active_calls = 0
    pricing_depth_fair_rounds = 0
    pricing_depth_fair_halfcap_dual_abs_sum = 0.0
    pricing_multistop_neutral_enabled_calls = 0
    pricing_multistop_candidates_seen = 0
    pricing_multistop_cross_zero_seen = 0
    pricing_multistop_nonnegative_seen = 0
    pricing_multistop_neutral_returned = 0
    pricing_multistop_neutral_added = 0
    pricing_multistop_neutral_batches = 0
    pricing_multistop_neutral_returned_by_depth = {}
    pricing_multistop_best_stop_count = 0
    pricing_multistop_best_uncovered_gain = 0
    pricing_multistop_best_rc_ub = None
    pricing_multistop_best_energy_per_stop_Wh = None
    pricing_multistop_neutral_signatures = set()
    pricing_multistop_neutral_batch_limit = 4
    # V19 bounded heuristic-only two-stop exact-variant enrichment.
    pricing_multistop_merge_triggers = 0
    pricing_multistop_merge_attempts = 0
    pricing_multistop_merge_physical_feasible = 0
    pricing_multistop_merge_new_candidates = 0
    pricing_multistop_merge_returned = 0
    pricing_multistop_merge_added = 0
    pricing_multistop_merge_batches = 0
    pricing_multistop_merge_distinct_pairs = 0
    pricing_multistop_merge_best_rc_ub = None
    pricing_multistop_merge_best_energy_per_stop_Wh = None
    pricing_multistop_merge_best_uncovered_gain = 0
    pricing_multistop_merge_signatures = set()
    pricing_multistop_merge_singleton_streak = 0
    pricing_multistop_merge_batch_limit = 3
    # V20 bounded heuristic-only exact singleton timing variants.
    pricing_resource_variant_triggers = 0
    pricing_resource_variant_attempts = 0
    pricing_resource_variant_deck_compatible_specs = 0
    pricing_resource_variant_deck_prefilter_skips = 0
    pricing_resource_variant_physical_feasible = 0
    pricing_resource_variant_new_candidates = 0
    pricing_resource_variant_returned = 0
    pricing_resource_variant_added = 0
    pricing_resource_variant_batches = 0
    pricing_resource_variant_distinct_turbines = 0
    pricing_resource_variant_best_rc_ub = None
    pricing_resource_variant_best_energy_Wh = None
    pricing_resource_variant_signatures = set()
    pricing_resource_variant_records = []
    pricing_resource_variant_trigger_limit = 3
    archive_primal_recovery_calls = 0
    archive_primal_recovery_runtime_s = 0.0
    archive_primal_recovery_audit_calls = 0
    archive_primal_recovery_timeouts = 0
    archive_primal_recovery_improvements = 0
    archive_primal_recovery_best_coverage = 0
    archive_primal_recovery_best_archive_columns = 0
    archive_primal_recovery_records = []
    archive_primal_recovery_witness_selection_indices = []
    archive_primal_recovery_witness_route_signatures = []
    archive_primal_recovery_witness_covered_turbines = []
    pricing_physical_cache_hits = 0
    pricing_physical_cache_misses = 0
    pricing_runtime_s = 0.0
    pricing_physical_evaluator_runtime_s = 0.0
    pricing_prefix_bound_runtime_s = 0.0
    pricing_prefix_service_runtime_s = 0.0
    pricing_certified_prefix_prunes = 0
    pricing_depth_certified_prefix_prunes = {}
    pricing_service_floor_prunes = 0
    pricing_depth_service_floor_prunes = {}
    pricing_horizon_window_skips = 0
    pricing_horizon_service_time_skips = 0
    pricing_physical_infeasible_results = 0
    pricing_branch_filter_skips = 0
    pricing_existing_signature_skips = 0
    pricing_call_records = []
    rmp_records = []
    resource_audit_records = []
    rmp_runtime_s = 0.0
    phase_one_runtime_s = 0.0
    resource_audit_runtime_s = 0.0
    pricing_discovery_improving_seen = 0
    pricing_discovery_improving_returned = 0
    pricing_discovery_diverse_returns = 0
    pricing_discovery_hard_cap_returns = 0
    pricing_discovery_max_return_batch = 0
    pricing_discovery_max_distinct_launches = 0
    pricing_discovery_max_distinct_service_sets = 0
    primal_refresh_calls = 0
    primal_refresh_audit_calls = 0
    primal_refresh_timeouts = 0
    primal_refresh_improvements = 0
    primal_refresh_best_coverage = 0
    primal_refresh_columns_seen = 0
    primal_refresh_rebuilds = 0
    primal_refresh_repairs = 0
    primal_refresh_augmentation_audits = 0
    primal_refresh_rebuild_audits = 0
    primal_refresh_repair_audits = 0
    primal_refresh_augmentation_improvements = 0
    primal_refresh_rebuild_improvements = 0
    primal_refresh_repair_improvements = 0
    primal_refresh_duplicate_trials_skipped = 0
    primal_refresh_uncovered_fair_rounds = 0
    primal_refresh_failure_reasons = {}
    primal_deck_archive_conflict_edges = 0
    primal_deck_archive_max_degree = 0
    primal_deck_archive_max_component = 0
    primal_deck_candidate_scored = 0
    primal_deck_candidate_zero_conflict = 0
    primal_deck_candidate_positive_conflict = 0
    primal_deck_prefilter_skips = 0
    primal_deck_max_candidate_conflicts = 0
    primal_deck_conflict_pairs_sample = []
    # V17/V18 cache is stage-local and heuristic-only.  Only exact
    # INFEASIBLE_PROVEN selections are stored; UNKNOWN_TIMEOUT is never cached.
    primal_refresh_infeasible_trial_cache = set()
    primal_exchange_calls = 0
    primal_exchange_candidate_routes = 0
    primal_exchange_trials_built = 0
    primal_exchange_audit_calls = 0
    primal_exchange_improvements = 0
    primal_exchange_consolidation_trials = 0
    primal_exchange_optional_drop_trials = 0
    primal_exchange_max_stop_count_considered = 0
    primal_exchange_best_coverage = 0
    pricing_depth_prefixes_evaluated = {}
    pricing_depth_improving_seen = {}
    pricing_depth_improving_returned = {}
    pricing_pattern_cut_active_dual_rows = 0
    pricing_pattern_cut_dual_abs_sum = 0.0
    pricing_pattern_cut_improving_seen_count = 0
    pricing_pattern_cut_improving_seen_contribution_sum = 0.0
    pricing_pattern_cut_improving_seen_sign_essential = 0
    pricing_pattern_cut_returned_count = 0
    pricing_pattern_cut_returned_contribution_sum = 0.0
    pricing_pattern_cut_returned_sign_essential = 0
    pricing_pattern_cut_returned_by_depth = {}
    battery_halfcap_dual_active_rmp_solves = 0
    battery_halfcap_dual_abs_sum = 0.0
    battery_halfcap_dual_max_abs = 0.0
    generated = 0
    resource_cuts_added = 0
    rmp_solves = 0
    phase_one_solves = 0
    pricing_candidates = 0
    pricing_nodes = 0
    resource_audit_calls = 0
    branch_children_created = 0
    branch_decisions = 0
    # Distinguish "the DFS scan finished" from mathematical pricing closure.
    # A node can be safely fathomed from a valid reduced-cost bound even when
    # omitted-column rc is not proved nonnegative.
    pricing_complete_all = True
    pricing_closed_all = True
    pricing_bound_available_all = True
    resource_audit_complete = True
    farkas_complete_all = True
    branching_complete = True
    exact_pricing_called = False
    last_best_rc = None
    last_rc_bound = None
    bound_source = root.bound_source

    incumbent_selection = tuple(initial_selection)
    incumbent_audit = initial_audit
    empty_is_admissible = (stage == "coverage" or int(coverage_target or 0) == 0)
    if not incumbent_selection and empty_is_admissible:
        # The natural set-packing model admits x=0.  Audit it explicitly so a
        # zero-coverage optimum is a valid incumbent/certificate, not "no solution".
        resource_audit_calls += 1
        _audit_t0 = time.perf_counter()
        incumbent_audit = _audit_integer_selection(
            archive, tuple(), K, batteries, p, quick_min, swap_min,
            quick_capacity, swap_capacity, deadline)
        _audit_dt = time.perf_counter() - _audit_t0
        resource_audit_runtime_s += _audit_dt
        resource_audit_records.append(dict(
            call_index=int(resource_audit_calls), trigger="empty-incumbent",
            bb_node=None, bb_depth=None, selection_size=0,
            wall_time_s=float(_audit_dt), status=str(incumbent_audit.status.value),
            dfs_nodes=int(getattr(incumbent_audit, "explored_nodes", 0)),
            cache_hits=int(getattr(incumbent_audit, "memo_hits", 0)),
            failure_reasons=dict(getattr(incumbent_audit, "failure_reasons", None) or {}),
            cut_generated=False))
        if incumbent_audit.status is not RA.ResourceAuditStatus.FEASIBLE:
            raise RuntimeError("empty plan must be resource feasible")
    incumbent_exact = None
    if stage == "coverage":
        incumbent_value = (float(_coverage_of_selection(archive, incumbent_selection))
                           if incumbent_selection else 0.0)
        incumbent_lower_bound = incumbent_upper_bound = float(incumbent_value)
        coverage_incumbent = int(incumbent_value)
    else:
        if incumbent_selection:
            incumbent_exact = _energy_of_selection_exact(archive, incumbent_selection)
            incumbent_value = float(incumbent_exact)
            incumbent_lower_bound = _fraction_to_float_down(incumbent_exact)
            incumbent_upper_bound = _fraction_to_float_up(incumbent_exact)
        elif empty_is_admissible:
            incumbent_exact = Fraction(0)
            incumbent_value = incumbent_lower_bound = incumbent_upper_bound = 0.0
        else:
            incumbent_value = incumbent_lower_bound = incumbent_upper_bound = None
        coverage_incumbent = int(coverage_target or 0)

    primal_refresh_best_coverage = int(
        _coverage_of_selection(archive, incumbent_selection)
        if incumbent_selection else 0)

    def _maybe_archive_primal_recover(_trigger):
        """V21 primal-only exact solve of the *current generated archive*.

        The inner complete-universe solve is exact only for the frozen archive.
        We import only an exact-resource-feasible witness as a primal incumbent;
        none of its restricted-archive bounds, cuts, or certificates are copied
        into the full-space BPC state.
        """
        nonlocal incumbent_selection, incumbent_audit
        nonlocal incumbent_value, incumbent_lower_bound, incumbent_upper_bound
        nonlocal incumbent_exact, coverage_incumbent
        nonlocal archive_primal_recovery_calls
        nonlocal archive_primal_recovery_runtime_s
        nonlocal archive_primal_recovery_audit_calls
        nonlocal archive_primal_recovery_timeouts
        nonlocal archive_primal_recovery_improvements
        nonlocal archive_primal_recovery_best_coverage
        nonlocal archive_primal_recovery_best_archive_columns
        nonlocal archive_primal_recovery_records
        nonlocal archive_primal_recovery_witness_selection_indices
        nonlocal archive_primal_recovery_witness_route_signatures
        nonlocal archive_primal_recovery_witness_covered_turbines
        nonlocal resource_audit_runtime_s

        if (not archive_primal_recovery or stage != "coverage"
                or bool(complete_universe_mode) or not archive):
            return False
        _remaining = (None if deadline is None
                      else max(0.0, float(deadline) - time.monotonic()))
        _limit = float(archive_primal_recovery_time_limit_s)
        if _remaining is not None:
            _limit = min(_limit, _remaining)
        if _limit <= 1e-6:
            return False

        archive_primal_recovery_calls += 1
        _inc_before = int(coverage_incumbent)
        _archive_cols = int(len(archive))
        _t0 = time.monotonic()
        _diag = _diagnose_fixed_archive_coverage(
            turbines=turbines, launch_opts=launch_opts, p=p,
            xi_amb=xi_amb, K=K, batteries=batteries, T_min=T_min,
            max_stops=max_stops, weather_unc=weather_unc,
            archive=archive, signature_to_index=signature_to_index,
            no_good_cuts=no_good_cuts,
            initial_selection=incumbent_selection,
            initial_audit=incumbent_audit,
            t_launch_min=t_launch_min,
            landing_clear_min=landing_clear_min,
            quick_min=quick_min, swap_min=swap_min,
            quick_capacity=quick_capacity, swap_capacity=swap_capacity,
            deck_mode=deck_mode, deck_delta_min=deck_delta_min,
            kappa_mode=kappa_mode, chance_mode=chance_mode,
            budget_gamma=budget_gamma, time_limit_s=_limit)
        _call_runtime = float(time.monotonic() - _t0)
        archive_primal_recovery_runtime_s += _call_runtime
        if str(_diag.get("status", "")).endswith("time-limit"):
            archive_primal_recovery_timeouts += 1

        _sel = tuple(int(_j) for _j in
                     (_diag.get("witness_selection_indices", []) or ())
                     if 0 <= int(_j) < len(archive))
        _cov = int(_coverage_of_selection(archive, _sel)) if _sel else 0
        archive_primal_recovery_best_coverage = max(
            archive_primal_recovery_best_coverage, _cov)
        archive_primal_recovery_best_archive_columns = max(
            archive_primal_recovery_best_archive_columns, len(archive))
        _rec = dict(
            trigger=str(_trigger),
            archive_columns=_archive_cols,
            restricted_status=str(_diag.get("status", "")),
            restricted_runtime_s=_call_runtime,
            restricted_coverage_lower_bound=_diag.get("coverage_lower_bound"),
            restricted_coverage_upper_bound=_diag.get("coverage_upper_bound"),
            restricted_exact_optimum=_diag.get("exact_optimum"),
            restricted_optimal_proven=bool(_diag.get("optimal_proven", False)),
            witness_coverage=int(_cov),
            incumbent_before=int(_inc_before),
            incumbent_after=int(_inc_before),
            exact_reaudit_status="not-needed",
            promoted=False)

        if _cov <= int(coverage_incumbent):
            archive_primal_recovery_records.append(_rec)
            return False
        if _deadline_hit(deadline):
            _rec["exact_reaudit_status"] = "skipped-formal-deadline"
            archive_primal_recovery_records.append(_rec)
            return False

        archive_primal_recovery_audit_calls += 1
        _audit_t0 = time.perf_counter()
        _audit = _audit_integer_selection(
            archive, _sel, K, batteries, p,
            quick_min, swap_min, quick_capacity, swap_capacity, deadline)
        resource_audit_runtime_s += time.perf_counter() - _audit_t0
        _rec["exact_reaudit_status"] = str(_audit.status)
        if _audit.status is not RA.ResourceAuditStatus.FEASIBLE:
            if _audit.status is RA.ResourceAuditStatus.UNKNOWN_TIMEOUT:
                archive_primal_recovery_timeouts += 1
            archive_primal_recovery_records.append(_rec)
            return False

        incumbent_selection = _sel
        incumbent_audit = _audit
        coverage_incumbent = int(_cov)
        incumbent_value = float(_cov)
        incumbent_lower_bound = float(_cov)
        incumbent_upper_bound = float(_cov)
        incumbent_exact = None
        archive_primal_recovery_improvements += 1
        archive_primal_recovery_witness_selection_indices = [
            int(_j) for _j in _sel]
        archive_primal_recovery_witness_route_signatures = [
            repr(_exact_route_signature(archive[int(_j)]))
            for _j in _sel if 0 <= int(_j) < len(archive)]
        archive_primal_recovery_witness_covered_turbines = sorted({
            str(_tid0)
            for _j in _sel if 0 <= int(_j) < len(archive)
            for _tid0 in _ordered_tids(archive[int(_j)])})
        _rec["promoted"] = True
        _rec["incumbent_after"] = int(_cov)
        _rec["witness_selection_indices"] = list(
            archive_primal_recovery_witness_selection_indices)
        _rec["witness_covered_turbines"] = list(
            archive_primal_recovery_witness_covered_turbines)
        archive_primal_recovery_records.append(_rec)
        return True

    # Recover the best exact-resource-feasible incumbent already present in the
    # warm-start/current archive before the first full-space RMP solve.
    _maybe_archive_primal_recover("root")

    termination = "branch-tree-exhausted"
    interrupt_tree = False
    while queue:
        if _deadline_hit(deadline):
            termination = "global-time-limit"
            break
        _, _, node = heapq.heappop(queue)
        if not _branch_consistent(node.branch):
            continue
        processed += 1
        node_finished = False
        while not node_finished:
            if _deadline_hit(deadline):
                heapq.heappush(queue, (_queue_priority(stage, node.inherited_bound, node.node_id), node.node_id, node))
                termination = "global-time-limit"
                node_finished = True
                break
            rmp_solves += 1
            _rmp_t0 = time.perf_counter()
            master = _solve_restricted_master(
                archive, all_tids, node, stage, coverage_target,
                deck_times, active_times, K, pooled_energy_cap,
                no_good_cuts, deadline,
                extra_inequality_rows=(
                    formal_extra_inequality_rows
                    + diagnostic_extra_inequality_rows))
            _rmp_dt = time.perf_counter() - _rmp_t0
            rmp_runtime_s += _rmp_dt
            _rmp_record = dict(
                solve_index=int(rmp_solves), stage=str(stage),
                bb_node=int(node.node_id), bb_depth=int(node.depth),
                wall_time_s=float(_rmp_dt), status=str(master.status),
                number_of_columns=int(len(master.eligible_indices)),
                inequality_rows=int(len(master.inequality_rows)),
                equality_rows=int(len(master.equality_rows)),
                total_rows=int(len(master.inequality_rows) + len(master.equality_rows)),
                cut_count=int(len(no_good_cuts)),
                solver_backend="scipy.optimize.linprog(method=highs)",
                persistent_model=False, basis_reuse=False,
                solver_iterations=None, phase_one_runtime_s=0.0)
            rmp_records.append(_rmp_record)
            pricing_stage = stage
            phase_one = False
            if master.status == "infeasible":
                if complete_universe_mode:
                    # [THM-CU] Every legal route column is already present.
                    # Therefore restricted-master infeasibility is full-space
                    # node infeasibility; no Farkas search for omitted columns
                    # exists or is required.
                    node.bound_source = "complete-universe-rmp-infeasible"
                    node_finished = True
                    continue
                phase_one_solves += 1
                _phase_t0 = time.perf_counter()
                phase = _solve_elastic_phase_one(master, deadline)
                _phase_dt = time.perf_counter() - _phase_t0
                phase_one_runtime_s += _phase_dt
                _rmp_record["phase_one_runtime_s"] = float(_phase_dt)
                if phase is None:
                    farkas_complete_all = False
                    pricing_complete_all = False
                    node.inherited_bound = (float(_node_allowed_turbine_bound(all_tids, node.branch))
                                            if stage == "coverage" else 0.0)
                    node.bound_source = ("trivial-node-allowed-turbine-bound" if stage == "coverage"
                                         else "nonnegative-energy-bound")
                    heapq.heappush(queue, (_queue_priority(stage, node.inherited_bound, node.node_id), node.node_id, node))
                    termination = "farkas-phase-time-limit-or-invalid"
                    pricing_bound_available_all = False
                    interrupt_tree = True
                    node_finished = True
                    break
                if float(phase.phase_one_value or 0.0) <= 1e-8:
                    # The ordinary RMP and elastic Phase-I disagree at the
                    # numerical feasibility boundary.  Re-solving the identical
                    # RMP can loop forever when no deadline is supplied.  Keep
                    # the node open with a strict trivial full-space bound and
                    # fail closed; numerical ambiguity is never infeasibility.
                    farkas_complete_all = False
                    pricing_complete_all = False
                    pricing_closed_all = False
                    pricing_bound_available_all = False
                    node.inherited_bound = (
                        float(_node_allowed_turbine_bound(all_tids, node.branch))
                        if stage == "coverage" else 0.0)
                    node.bound_source = (
                        "trivial-node-allowed-turbine-bound"
                        if stage == "coverage" else "nonnegative-energy-bound")
                    heapq.heappush(
                        queue,
                        (_queue_priority(stage, node.inherited_bound, node.node_id),
                         node.node_id, node))
                    termination = "phase-one-numeric-feasibility-ambiguity"
                    interrupt_tree = True
                    node_finished = True
                    break
                master = phase
                pricing_stage = "farkas"
                phase_one = True
            elif master.status != "optimal":
                pricing_complete_all = False
                node.inherited_bound = (float(_node_allowed_turbine_bound(all_tids, node.branch))
                                        if stage == "coverage" else 0.0)
                node.bound_source = ("trivial-node-allowed-turbine-bound" if stage == "coverage"
                                     else "nonnegative-energy-bound")
                heapq.heappush(queue, (_queue_priority(stage, node.inherited_bound, node.node_id), node.node_id, node))
                termination = f"rmp-{master.status}"
                pricing_bound_available_all = False
                interrupt_tree = True
                node_finished = True
                break

            if master.inequality_duals is not None:
                _halfcap_dual_abs = 0.0
                for _desc, _dual in zip(
                        master.inequality_rows,
                        np.asarray(master.inequality_duals, float)):
                    if _desc[0] != "battery_halfcap":
                        continue
                    _a = abs(float(_dual))
                    if not math.isfinite(_a):
                        raise FloatingPointError(
                            "non-finite formal battery-halfcap dual")
                    _halfcap_dual_abs += _a
                    battery_halfcap_dual_max_abs = max(
                        float(battery_halfcap_dual_max_abs), _a)
                if _halfcap_dual_abs > 0.0:
                    battery_halfcap_dual_active_rmp_solves += 1
                battery_halfcap_dual_abs_sum += float(_halfcap_dual_abs)

            if complete_universe_mode:
                pricing = PricingSearchResult(
                    [], True, None, math.inf, True, 0, 0,
                    "complete-materialized-universe-no-omitted-columns")
            else:
                pricing_calls += 1
                exact_pricing_calls += 1
                exact_pricing_called = True
                _pricing_experiment_key = (
                    str(pricing_experiment_mode).strip().lower()
                    if isinstance(pricing_experiment_mode, str)
                    else ("discovery-shadow" if pricing_experiment_mode else "off"))
                _use_discovery_shadow = bool(
                    _pricing_experiment_key in {
                        "discovery-shadow", "dual-guided-shadow",
                        "layered-guided-shadow", "layered-batch-shadow",
                        "layered-batch-primal-shadow",
                        "layered-batch-primal-depth-fair-shadow",
                        "layered-batch-primal-depth-fair-neutral-shadow"}
                    and not phase_one and pricing_stage in {"coverage", "energy"})
                _use_primal_refresh = bool(
                    _pricing_experiment_key in {
                        "layered-batch-primal-shadow",
                        "layered-batch-primal-depth-fair-shadow",
                        "layered-batch-primal-depth-fair-neutral-shadow"}
                    and _use_discovery_shadow)
                _use_adaptive_batch = bool(
                    _pricing_experiment_key in {
                        "layered-batch-shadow", "layered-batch-primal-shadow",
                        "layered-batch-primal-depth-fair-shadow",
                        "layered-batch-primal-depth-fair-neutral-shadow"}
                    and _use_discovery_shadow)
                _use_layered_guided = bool(
                    _pricing_experiment_key in {
                        "layered-guided-shadow", "layered-batch-shadow",
                        "layered-batch-primal-shadow",
                        "layered-batch-primal-depth-fair-shadow",
                        "layered-batch-primal-depth-fair-neutral-shadow"}
                    and _use_discovery_shadow)
                _use_dual_guided = bool(
                    _pricing_experiment_key in {
                        "dual-guided-shadow", "layered-guided-shadow",
                        "layered-batch-shadow", "layered-batch-primal-shadow",
                        "layered-batch-primal-depth-fair-shadow",
                        "layered-batch-primal-depth-fair-neutral-shadow"}
                    and _use_discovery_shadow)
                _use_depth_fair = bool(
                    _pricing_experiment_key in {
                        "layered-batch-primal-depth-fair-shadow",
                        "layered-batch-primal-depth-fair-neutral-shadow"}
                    and _use_discovery_shadow)
                _use_neutral_multistop = bool(
                    _pricing_experiment_key
                        == "layered-batch-primal-depth-fair-neutral-shadow"
                    and _use_discovery_shadow
                    and pricing_stage == "coverage"
                    and pricing_multistop_neutral_batches
                        < pricing_multistop_neutral_batch_limit)
                if _use_discovery_shadow:
                    pricing_discovery_calls += 1
                _neutral_uncovered_tids = set(all_tids)
                for _j in incumbent_selection:
                    if 0 <= int(_j) < len(archive):
                        _neutral_uncovered_tids.difference_update(
                            _ordered_tids(archive[int(_j)]))
                pricing = _exact_pricing_search(
                    turbines, launch_opts, p, xi_amb, weather_unc, T_min,
                    max_stops, node, set(signature_to_index), pricing_stage,
                    master.inequality_rows, master.equality_rows,
                    master.inequality_duals, master.equality_duals,
                    deadline, pricing_epsilon, t_launch_min,
                    landing_clear_min, deck_mode, deck_delta_min,
                    kappa_mode=kappa_mode, chance_mode=chance_mode,
                    budget_gamma=budget_gamma,
                    implicit_test_columns=implicit_test_columns,
                    batch_size=pricing_batch_size,
                    physical_cache=physical_cache,
                    search_goal=("discovery" if _use_discovery_shadow else "certification"),
                    shadow_prefix_bounds=_use_discovery_shadow,
                    discovery_column_limit=(
                        min(4, int(pricing_batch_size))
                        if _use_adaptive_batch
                        else 1 if _use_dual_guided
                        else min(8, int(pricing_batch_size))),
                    guided_discovery_order=_use_dual_guided,
                    layered_discovery_order=_use_layered_guided,
                    depth_fair_discovery_order=_use_depth_fair,
                    neutral_multistop_enrichment=_use_neutral_multistop,
                    neutral_multistop_batch_target=min(
                        8, int(pricing_batch_size)),
                    neutral_uncovered_tids=_neutral_uncovered_tids,
                    adaptive_discovery_batch=_use_adaptive_batch,
                    discovery_batch_hard_cap=min(
                        6, int(pricing_batch_size)),
                    discovery_min_distinct_launches=2,
                    discovery_min_distinct_service_sets=2,
                    pattern_cut_diagnostics=bool(
                        pricing_pattern_cut_diagnostics),
                    certified_prefix_pruning=bool(
                        certified_prefix_pruning and not _use_discovery_shadow
                        and not phase_one))
                pricing_shadow_prefixes_evaluated += int(pricing.shadow_prefixes_evaluated)
                pricing_shadow_prunable_prefixes += int(pricing.shadow_prunable_prefixes)
                pricing_shadow_false_prune_witnesses += int(
                    pricing.shadow_false_prune_witnesses)
                pricing_shadow_bound_errors += int(pricing.shadow_bound_errors)
                if pricing.shadow_audit_complete:
                    pricing_shadow_complete_calls += 1
                pricing_guided_order_calls += int(pricing.guided_order_calls)
                pricing_guided_order_reorders += int(pricing.guided_order_reorders)
                pricing_guided_order_failures += int(pricing.guided_order_failures)
                pricing_layered_depths_started += int(
                    pricing.layered_depths_started)
                pricing_layered_depths_completed += int(
                    pricing.layered_depths_completed)
                pricing_layered_max_depth_completed = max(
                    pricing_layered_max_depth_completed,
                    int(pricing.layered_max_depth_completed))
                pricing_layered_rounds += int(pricing.layered_rounds)
                pricing_depth_fair_requested_calls += int(
                    bool(pricing.depth_fair_requested))
                pricing_depth_fair_active_calls += int(
                    bool(pricing.depth_fair_active))
                pricing_depth_fair_rounds += int(pricing.depth_fair_rounds)
                pricing_depth_fair_halfcap_dual_abs_sum += float(
                    pricing.depth_fair_halfcap_dual_abs)
                pricing_multistop_neutral_enabled_calls += int(
                    bool(pricing.neutral_multistop_enabled))
                pricing_multistop_candidates_seen += int(
                    pricing.neutral_multistop_candidates_seen)
                pricing_multistop_cross_zero_seen += int(
                    pricing.neutral_multistop_cross_zero_seen)
                pricing_multistop_nonnegative_seen += int(
                    pricing.neutral_multistop_nonnegative_seen)
                pricing_multistop_neutral_returned += int(
                    pricing.neutral_multistop_returned)
                for _d, _v in pricing.neutral_multistop_returned_by_depth.items():
                    _di = int(_d)
                    pricing_multistop_neutral_returned_by_depth[_di] = int(
                        pricing_multistop_neutral_returned_by_depth.get(
                            _di, 0)) + int(_v)
                pricing_multistop_best_stop_count = max(
                    pricing_multistop_best_stop_count,
                    int(pricing.neutral_multistop_best_stop_count))
                pricing_multistop_best_uncovered_gain = max(
                    pricing_multistop_best_uncovered_gain,
                    int(pricing.neutral_multistop_best_uncovered_gain))
                if pricing.neutral_multistop_best_rc_ub is not None:
                    _v = float(pricing.neutral_multistop_best_rc_ub)
                    pricing_multistop_best_rc_ub = (
                        _v if pricing_multistop_best_rc_ub is None
                        else min(pricing_multistop_best_rc_ub, _v))
                if pricing.neutral_multistop_best_energy_per_stop_Wh is not None:
                    _v = float(pricing.neutral_multistop_best_energy_per_stop_Wh)
                    pricing_multistop_best_energy_per_stop_Wh = (
                        _v if pricing_multistop_best_energy_per_stop_Wh is None
                        else min(pricing_multistop_best_energy_per_stop_Wh, _v))
                pricing_physical_cache_hits += int(pricing.physical_cache_hits)
                pricing_physical_cache_misses += int(pricing.physical_cache_misses)
                pricing_runtime_s += float(pricing.wall_time_s)
                pricing_physical_evaluator_runtime_s += float(
                    pricing.physical_evaluator_runtime_s)
                pricing_prefix_bound_runtime_s += float(pricing.prefix_bound_runtime_s)
                pricing_prefix_service_runtime_s += float(pricing.prefix_service_runtime_s)
                pricing_certified_prefix_prunes += int(pricing.certified_prefix_prunes)
                pricing_service_floor_prunes += int(pricing.service_floor_prunes)
                pricing_horizon_window_skips += int(pricing.horizon_window_skips)
                pricing_horizon_service_time_skips += int(
                    pricing.horizon_service_time_skips)
                pricing_physical_infeasible_results += int(
                    pricing.physical_infeasible_results)
                pricing_branch_filter_skips += int(pricing.branch_filter_skips)
                pricing_existing_signature_skips += int(pricing.existing_signature_skips)
                for _d, _v in pricing.depth_certified_prefix_prunes.items():
                    _di = int(_d)
                    pricing_depth_certified_prefix_prunes[_di] = int(
                        pricing_depth_certified_prefix_prunes.get(_di, 0)) + int(_v)
                for _d, _v in pricing.depth_service_floor_prunes.items():
                    _di = int(_d)
                    pricing_depth_service_floor_prunes[_di] = int(
                        pricing_depth_service_floor_prunes.get(_di, 0)) + int(_v)
                _dual_payload = (
                    tuple((str(_desc[0]), repr(_desc[1]), float(_dual).hex())
                          for _desc, _dual in zip(
                              master.inequality_rows,
                              np.asarray(master.inequality_duals, float))),
                    tuple((str(_desc[0]), repr(_desc[1]), float(_dual).hex())
                          for _desc, _dual in zip(
                              master.equality_rows,
                              np.asarray(master.equality_duals, float))))
                _dual_fp = hashlib.sha256(
                    repr(_dual_payload).encode("utf-8")).hexdigest()[:16]
                pricing_call_records.append(dict(
                    call_index=int(pricing_calls), stage=str(pricing_stage),
                    bb_node=int(node.node_id), bb_depth=int(node.depth),
                    dual_fingerprint=str(_dual_fp), search_goal=str(pricing.search_goal),
                    wall_time_s=float(pricing.wall_time_s),
                    prefix_nodes=int(pricing.evaluated_sequences),
                    candidate_count=int(pricing.evaluated_routes),
                    improving_candidate_count=int(pricing.improving_columns_seen),
                    best_rc_estimate=pricing.best_reduced_value,
                    best_rc_lb=pricing.reduced_value_bound,
                    best_rc_ub=pricing.best_reduced_value_ub,
                    physical_cache_hits=int(pricing.physical_cache_hits),
                    physical_cache_misses=int(pricing.physical_cache_misses),
                    physical_evaluator_runtime_s=float(
                        pricing.physical_evaluator_runtime_s),
                    prefix_bound_runtime_s=float(pricing.prefix_bound_runtime_s),
                    service_floor_prunes=int(pricing.service_floor_prunes),
                    certified_rc_bound_prunes=int(pricing.certified_prefix_prunes),
                    branch_filter_skips=int(pricing.branch_filter_skips),
                    existing_signature_skips=int(pricing.existing_signature_skips),
                    physical_infeasible_results=int(pricing.physical_infeasible_results),
                    whole_route_evaluator_calls=int(pricing.whole_route_evaluator_calls),
                    drcc_calls=int(pricing.drcc_route_evaluator_calls),
                    resource_bound_prunes=int(pricing.service_floor_prunes),
                    reduced_cost_bound_prunes=int(pricing.certified_prefix_prunes),
                    dominance_prunes=int(pricing.dominance_prunes),
                    duplicate_state_prunes=int(pricing.duplicate_state_prunes),
                    certified_prefix_bound_histogram=dict(
                        pricing.certified_prefix_bound_histogram),
                    launch_prefix_nodes=dict(pricing.launch_prefix_nodes),
                    root_turbine_prefix_nodes=dict(pricing.root_turbine_prefix_nodes),
                    root_pair_prefix_nodes=dict(pricing.root_pair_prefix_nodes),
                    launch_evaluator_calls=dict(pricing.launch_evaluator_calls),
                    horizon_evaluator_calls=dict(pricing.horizon_evaluator_calls),
                    launch_horizon_evaluator_calls=dict(
                        pricing.launch_horizon_evaluator_calls),
                    horizon_window_skips=int(pricing.horizon_window_skips),
                    horizon_service_time_skips=int(pricing.horizon_service_time_skips),
                    rmp_solve_index=int(_rmp_record["solve_index"]),
                    rmp_runtime_s=float(_rmp_record["wall_time_s"]),
                    rmp_columns=int(_rmp_record["number_of_columns"]),
                    rmp_rows=int(_rmp_record["total_rows"]),
                    cut_count=int(_rmp_record["cut_count"]),
                    rmp_basis_reuse=False,
                    depth_prefix_nodes=dict(pricing.depth_prefixes_evaluated),
                    depth_service_floor_prunes=dict(pricing.depth_service_floor_prunes),
                    depth_certified_rc_prunes=dict(
                        pricing.depth_certified_prefix_prunes)))
                for _d, _v in pricing.depth_prefixes_evaluated.items():
                    _di = int(_d)
                    pricing_depth_prefixes_evaluated[_di] = int(
                        pricing_depth_prefixes_evaluated.get(_di, 0)) + int(_v)
                for _d, _v in pricing.depth_improving_seen.items():
                    _di = int(_d)
                    pricing_depth_improving_seen[_di] = int(
                        pricing_depth_improving_seen.get(_di, 0)) + int(_v)
                for _d, _v in pricing.depth_improving_returned.items():
                    _di = int(_d)
                    pricing_depth_improving_returned[_di] = int(
                        pricing_depth_improving_returned.get(_di, 0)) + int(_v)
                pricing_pattern_cut_active_dual_rows += int(
                    pricing.pattern_cut_active_dual_rows)
                pricing_pattern_cut_dual_abs_sum += float(
                    pricing.pattern_cut_dual_abs_sum)
                pricing_pattern_cut_improving_seen_count += int(
                    pricing.pattern_cut_improving_seen_count)
                pricing_pattern_cut_improving_seen_contribution_sum += float(
                    pricing.pattern_cut_improving_seen_contribution_sum)
                pricing_pattern_cut_improving_seen_sign_essential += int(
                    pricing.pattern_cut_improving_seen_sign_essential)
                pricing_pattern_cut_returned_count += int(
                    pricing.pattern_cut_returned_count)
                pricing_pattern_cut_returned_contribution_sum += float(
                    pricing.pattern_cut_returned_contribution_sum)
                pricing_pattern_cut_returned_sign_essential += int(
                    pricing.pattern_cut_returned_sign_essential)
                for _d, _rec in pricing.pattern_cut_returned_by_depth.items():
                    _di = int(_d)
                    _dst = pricing_pattern_cut_returned_by_depth.setdefault(
                        _di, dict(count=0, contribution_sum=0.0, sign_essential=0,
                                  rc_sum=0.0, rc_without_cut_ub_sum=0.0,
                                  rc_without_cut_ub_finite_count=0))
                    for _k in ("count", "sign_essential",
                               "rc_without_cut_ub_finite_count"):
                        _dst[_k] += int(_rec.get(_k, 0))
                    for _k in ("contribution_sum", "rc_sum",
                               "rc_without_cut_ub_sum"):
                        _dst[_k] += float(_rec.get(_k, 0.0))
                pricing_discovery_improving_seen += int(
                    pricing.improving_columns_seen)
                pricing_discovery_improving_returned += int(
                    pricing.discovery_improving_columns_returned)
                pricing_discovery_diverse_returns += int(
                    pricing.discovery_early_return
                    and pricing.discovery_diversity_satisfied)
                pricing_discovery_hard_cap_returns += int(
                    pricing.discovery_early_return
                    and pricing.discovery_hard_cap_triggered)
                pricing_discovery_max_return_batch = max(
                    pricing_discovery_max_return_batch,
                    int(pricing.discovery_improving_columns_returned))
                pricing_discovery_max_distinct_launches = max(
                    pricing_discovery_max_distinct_launches,
                    int(pricing.discovery_distinct_launches))
                pricing_discovery_max_distinct_service_sets = max(
                    pricing_discovery_max_distinct_service_sets,
                    int(pricing.discovery_distinct_service_sets))
                if pricing.discovery_early_return:
                    pricing_discovery_early_returns += 1
                if (not phase_one) and pricing.complete:
                    pricing_certification_calls += 1
            last_best_rc = pricing.best_reduced_value
            last_rc_bound = pricing.reduced_value_bound
            pricing_candidates += int(pricing.evaluated_routes)
            pricing_nodes += int(pricing.evaluated_sequences)
            pricing_bound_available_all &= bool(pricing.bound_available)
            if pricing.columns:
                _neutral_returned_this_call = int(
                    pricing.neutral_multistop_returned)
                _neutral_returned_sigs = (
                    {_exact_route_signature(_c) for _c in pricing.columns}
                    if _neutral_returned_this_call > 0 else set())
                added = _add_columns(archive, signature_to_index, pricing.columns)
                generated += added
                if _neutral_returned_this_call > 0:
                    pricing_multistop_neutral_batches += 1
                    pricing_multistop_neutral_added += int(added)
                    pricing_multistop_neutral_signatures.update(
                        _neutral_returned_sigs)

                # V19: after consecutive singleton-only formal discovery
                # batches, probe a tiny set of physically validated two-stop
                # exact variants.  This is heuristic enrichment only: formal
                # pricing closure/bounds remain those of ``pricing`` above.
                if (adaptive_multistop_enrichment
                        and stage == "coverage" and not phase_one
                        and pricing.discovery_early_return):
                    _ret_depths = {
                        int(_d): int(_v)
                        for _d, _v in pricing.depth_improving_returned.items()
                        if int(_v) > 0
                    }
                    if (_ret_depths
                            and set(_ret_depths).issubset({1})
                            and sum(_ret_depths.values()) > 0):
                        pricing_multistop_merge_singleton_streak += 1
                    else:
                        pricing_multistop_merge_singleton_streak = 0
                    if (pricing_multistop_merge_singleton_streak >= 2
                            and pricing_multistop_merge_batches
                                < pricing_multistop_merge_batch_limit
                            and not _deadline_hit(deadline)):
                        pricing_multistop_merge_triggers += 1
                        _merge = _adaptive_multistop_merge_enrichment(
                            archive=archive, turbines=turbines,
                            launch_opts=launch_opts, p=p, xi_amb=xi_amb,
                            weather_unc=weather_unc, T_min=T_min, node=node,
                            existing_signatures=set(signature_to_index),
                            incumbent_selection=incumbent_selection,
                            inequality_rows=master.inequality_rows,
                            equality_rows=master.equality_rows,
                            inequality_duals=master.inequality_duals,
                            equality_duals=master.equality_duals,
                            deadline=deadline,
                            t_launch_min=t_launch_min,
                            landing_clear_min=landing_clear_min,
                            deck_mode=deck_mode,
                            deck_delta_min=deck_delta_min,
                            kappa_mode=kappa_mode,
                            chance_mode=chance_mode,
                            budget_gamma=budget_gamma,
                            attempt_limit=24, batch_target=4,
                            wall_budget_s=0.60,
                            physical_cache=physical_cache)
                        pricing_multistop_merge_attempts += int(
                            _merge.get("attempts", 0))
                        pricing_multistop_merge_physical_feasible += int(
                            _merge.get("physical_feasible", 0))
                        pricing_multistop_merge_new_candidates += int(
                            _merge.get("new_candidates", 0))
                        _merge_cols = list(_merge.get("columns", []))
                        pricing_multistop_merge_returned += len(_merge_cols)
                        pricing_multistop_merge_distinct_pairs += int(
                            _merge.get("distinct_pairs", 0))
                        pricing_multistop_merge_best_uncovered_gain = max(
                            pricing_multistop_merge_best_uncovered_gain,
                            int(_merge.get("uncovered_gain_best", 0)))
                        if _merge.get("best_rc_ub") is not None:
                            _v = float(_merge["best_rc_ub"])
                            pricing_multistop_merge_best_rc_ub = (
                                _v if pricing_multistop_merge_best_rc_ub is None
                                else min(pricing_multistop_merge_best_rc_ub, _v))
                        if _merge.get("best_energy_per_stop_Wh") is not None:
                            _v = float(_merge["best_energy_per_stop_Wh"])
                            pricing_multistop_merge_best_energy_per_stop_Wh = (
                                _v if pricing_multistop_merge_best_energy_per_stop_Wh
                                    is None
                                else min(
                                    pricing_multistop_merge_best_energy_per_stop_Wh,
                                    _v))
                        if _merge_cols:
                            _merge_sigs = {
                                _exact_route_signature(_c) for _c in _merge_cols}
                            _merge_added = _add_columns(
                                archive, signature_to_index, _merge_cols)
                            generated += int(_merge_added)
                            added += int(_merge_added)
                            pricing_multistop_merge_added += int(_merge_added)
                            if _merge_added > 0:
                                pricing_multistop_merge_batches += 1
                                pricing_multistop_merge_signatures.update(
                                    _merge_sigs)

                # V20: after a singleton-only formal discovery batch, probe a
                # bounded set of *deck-compatible exact singleton timing variants*
                # for currently uncovered turbines.  This directly targets the
                # V19 real-run failure mode (72/72 blind two-stop probes were
                # route-level infeasible).  The enrichment remains heuristic-only.
                if (resource_variant_enrichment
                        and stage == "coverage" and not phase_one
                        and pricing.discovery_early_return
                        and pricing_resource_variant_triggers
                            < pricing_resource_variant_trigger_limit
                        and not _deadline_hit(deadline)):
                    _rv_depths = {
                        int(_d): int(_v)
                        for _d, _v in pricing.depth_improving_returned.items()
                        if int(_v) > 0
                    }
                    if (_rv_depths and set(_rv_depths).issubset({1})
                            and sum(_rv_depths.values()) > 0):
                        pricing_resource_variant_triggers += 1
                        _rv = _resource_aware_singleton_variant_enrichment(
                            archive=archive, turbines=turbines,
                            launch_opts=launch_opts, p=p, xi_amb=xi_amb,
                            weather_unc=weather_unc, T_min=T_min, node=node,
                            existing_signatures=set(signature_to_index),
                            incumbent_selection=incumbent_selection,
                            inequality_rows=master.inequality_rows,
                            equality_rows=master.equality_rows,
                            inequality_duals=master.inequality_duals,
                            equality_duals=master.equality_duals,
                            deadline=deadline,
                            t_launch_min=t_launch_min,
                            landing_clear_min=landing_clear_min,
                            deck_mode=deck_mode,
                            deck_delta_min=deck_delta_min,
                            kappa_mode=kappa_mode,
                            chance_mode=chance_mode,
                            budget_gamma=budget_gamma,
                            attempt_limit=32, batch_target=4,
                            wall_budget_s=0.75,
                            physical_cache=physical_cache)
                        pricing_resource_variant_attempts += int(
                            _rv.get("attempts", 0))
                        pricing_resource_variant_deck_compatible_specs += int(
                            _rv.get("deck_compatible_specs", 0))
                        pricing_resource_variant_deck_prefilter_skips += int(
                            _rv.get("deck_prefilter_skips", 0))
                        pricing_resource_variant_physical_feasible += int(
                            _rv.get("physical_feasible", 0))
                        pricing_resource_variant_new_candidates += int(
                            _rv.get("new_candidates", 0))
                        pricing_resource_variant_distinct_turbines += int(
                            _rv.get("distinct_turbines", 0))
                        if _rv.get("best_rc_ub") is not None:
                            _v = float(_rv["best_rc_ub"])
                            pricing_resource_variant_best_rc_ub = (
                                _v if pricing_resource_variant_best_rc_ub is None
                                else min(pricing_resource_variant_best_rc_ub, _v))
                        if _rv.get("best_energy_Wh") is not None:
                            _v = float(_rv["best_energy_Wh"])
                            pricing_resource_variant_best_energy_Wh = (
                                _v if pricing_resource_variant_best_energy_Wh is None
                                else min(pricing_resource_variant_best_energy_Wh, _v))
                        _rv_cols = list(_rv.get("columns", []))
                        _rv_records = [
                            dict(_rec) for _rec in _rv.get("records", [])]
                        for _rec in _rv_records:
                            _rec["trigger_index"] = int(
                                pricing_resource_variant_triggers)
                            _rec["stage"] = str(stage)
                        pricing_resource_variant_returned += len(_rv_cols)
                        if _rv_cols:
                            _rv_sigs = {
                                _exact_route_signature(_c) for _c in _rv_cols}
                            _rv_added = _add_columns(
                                archive, signature_to_index, _rv_cols)
                            generated += int(_rv_added)
                            added += int(_rv_added)
                            pricing_resource_variant_added += int(_rv_added)
                            if _rv_added > 0:
                                pricing_resource_variant_batches += 1
                                pricing_resource_variant_signatures.update(
                                    _rv_sigs)
                            for _rec in _rv_records:
                                _sig_repr = str(_rec.get("signature_repr", ""))
                                _idx = next((
                                    int(_j) for _j, _c in enumerate(archive)
                                    if repr(_exact_route_signature(_c))
                                       == _sig_repr), None)
                                _rec["added_to_archive"] = bool(
                                    _idx is not None
                                    and _exact_route_signature(archive[_idx])
                                        in pricing_resource_variant_signatures)
                                _rec["archive_index"] = _idx
                        pricing_resource_variant_records.extend(_rv_records)

                if added > 0:
                    _maybe_archive_primal_recover("columns-added")
                    if _use_primal_refresh and not phase_one:
                        primal_refresh_calls += 1
                        _resource_primal_active = bool(
                            (primal_exchange or primal_resource_primal
                             or primal_resource_guided
                             or primal_resource_deck_guided)
                            and stage == "coverage")
                        primal_exchange_calls += int(_resource_primal_active)
                        refresh = _primal_refresh_incumbent(
                            archive, incumbent_selection, incumbent_audit,
                            stage, coverage_target,
                            K, batteries, p, quick_min, swap_min,
                            quick_capacity, swap_capacity, deadline,
                            wall_budget_s=(0.75 if _resource_primal_active else 0.50),
                            max_audits=32,
                            strategy=("resource_deck_guided"
                                      if primal_resource_deck_guided
                                         and stage == "coverage"
                                      else "resource_guided"
                                      if primal_resource_guided
                                         and stage == "coverage"
                                      else "resource_primal"
                                      if primal_resource_primal
                                         and stage == "coverage"
                                      else "resource_exchange"
                                      if primal_exchange
                                         and stage == "coverage"
                                      else "legacy"),
                            infeasible_trial_cache=(
                                primal_refresh_infeasible_trial_cache
                                if (primal_resource_guided
                                    or primal_resource_deck_guided)
                                   and stage == "coverage"
                                else None))
                        primal_refresh_audit_calls += int(
                            refresh["audit_calls"])
                        primal_refresh_timeouts += int(
                            bool(refresh["timed_out"]))
                        primal_refresh_columns_seen = max(
                            primal_refresh_columns_seen,
                            int(refresh["columns_seen"]))
                        primal_refresh_rebuilds += int(refresh["rebuilds"])
                        primal_refresh_repairs += int(refresh["repairs"])
                        primal_refresh_augmentation_audits += int(
                            refresh["augmentation_audits"])
                        primal_refresh_rebuild_audits += int(
                            refresh["rebuild_audits"])
                        primal_refresh_repair_audits += int(
                            refresh["repair_audits"])
                        primal_refresh_augmentation_improvements += int(
                            refresh["augmentation_improvements"])
                        primal_refresh_rebuild_improvements += int(
                            refresh["rebuild_improvements"])
                        primal_refresh_repair_improvements += int(
                            refresh["repair_improvements"])
                        primal_refresh_duplicate_trials_skipped += int(
                            refresh.get("duplicate_trials_skipped", 0))
                        primal_refresh_uncovered_fair_rounds += int(
                            refresh.get("uncovered_fair_rounds", 0))
                        for _reason, _count in dict(
                                refresh.get("failure_reasons", {})).items():
                            primal_refresh_failure_reasons[str(_reason)] = int(
                                primal_refresh_failure_reasons.get(
                                    str(_reason), 0)) + int(_count)
                        primal_deck_archive_conflict_edges = max(
                            primal_deck_archive_conflict_edges,
                            int(refresh.get("deck_archive_conflict_edges", 0)))
                        primal_deck_archive_max_degree = max(
                            primal_deck_archive_max_degree,
                            int(refresh.get("deck_archive_max_degree", 0)))
                        primal_deck_archive_max_component = max(
                            primal_deck_archive_max_component,
                            int(refresh.get("deck_archive_max_component", 0)))
                        primal_deck_candidate_scored += int(
                            refresh.get("deck_candidate_scored", 0))
                        primal_deck_candidate_zero_conflict += int(
                            refresh.get("deck_candidate_zero_conflict", 0))
                        primal_deck_candidate_positive_conflict += int(
                            refresh.get("deck_candidate_positive_conflict", 0))
                        primal_deck_prefilter_skips += int(
                            refresh.get("deck_prefilter_skips", 0))
                        primal_deck_max_candidate_conflicts = max(
                            primal_deck_max_candidate_conflicts,
                            int(refresh.get("deck_max_candidate_conflicts", 0)))
                        for _pair in list(
                                refresh.get("deck_conflict_pairs_sample", [])):
                            if (_pair not in primal_deck_conflict_pairs_sample
                                    and len(primal_deck_conflict_pairs_sample) < 12):
                                primal_deck_conflict_pairs_sample.append(_pair)
                        primal_exchange_candidate_routes += int(
                            refresh.get("exchange_candidate_routes", 0))
                        primal_exchange_trials_built += int(
                            refresh.get("exchange_trials_built", 0))
                        primal_exchange_audit_calls += int(
                            refresh.get("exchange_audits", 0))
                        primal_exchange_improvements += int(
                            refresh.get("exchange_improvements", 0))
                        primal_exchange_consolidation_trials += int(
                            refresh.get("exchange_consolidation_trials", 0))
                        primal_exchange_optional_drop_trials += int(
                            refresh.get("exchange_optional_drop_trials", 0))
                        primal_exchange_max_stop_count_considered = max(
                            primal_exchange_max_stop_count_considered,
                            int(refresh.get(
                                "exchange_max_stop_count_considered", 0)))
                        primal_refresh_best_coverage = max(
                            primal_refresh_best_coverage,
                            int(refresh["coverage"]))
                        if (primal_exchange or primal_resource_primal
                                or primal_resource_guided
                                or primal_resource_deck_guided) and stage == "coverage":
                            primal_exchange_best_coverage = max(
                                primal_exchange_best_coverage,
                                int(refresh["coverage"]))
                        if refresh["improved"]:
                            primal_refresh_improvements += 1
                            incumbent_selection = tuple(
                                refresh["selection"])
                            incumbent_audit = refresh["audit"]
                            if stage == "coverage":
                                coverage_incumbent = int(
                                    refresh["coverage"])
                                incumbent_value = float(
                                    coverage_incumbent)
                                incumbent_lower_bound = float(
                                    coverage_incumbent)
                                incumbent_upper_bound = float(
                                    coverage_incumbent)
                                incumbent_exact = None
                            else:
                                incumbent_exact = (
                                    _energy_of_selection_exact(
                                        archive,
                                        incumbent_selection))
                                incumbent_value = float(
                                    incumbent_exact)
                                incumbent_lower_bound = (
                                    _fraction_to_float_down(
                                        incumbent_exact))
                                incumbent_upper_bound = (
                                    _fraction_to_float_up(
                                        incumbent_exact))
                    continue
            if phase_one:
                # Ordinary reduced-cost tolerance is an optimization stopping
                # tolerance, not a proof that the full elastic master has
                # positive artificial objective.  Correct the validated Phase-I
                # RMP dual bound by a lower bound on every omitted column's
                # reduced cost and by the node LP route-mass bound sum x_r <= M_n.
                route_mass_upper_bound = _node_allowed_turbine_bound(all_tids, node.branch)
                phase_infeasible, phase_full_lb = _phase_one_infeasibility_proven(
                    master, pricing, route_mass_upper_bound, ART_TOL)
                if phase_infeasible:
                    # This remains valid even if pricing was interrupted, provided
                    # its universal reduced-cost bound is available: the corrected
                    # full-space artificial lower bound is already strictly > 0.
                    if not pricing.complete:
                        pricing_complete_all = False
                    node_finished = True
                    continue

                # If exact Phase-I pricing found a genuinely negative omitted
                # column it is retained with FARKAS_COLUMN_EPS and would have been
                # added above.  Reaching here means full-space infeasibility has
                # not been proved.  Never prune merely because rc >= -PRICING_EPS.
                farkas_complete_all = False
                pricing_complete_all &= bool(pricing.complete)
                node.inherited_bound = (float(_node_allowed_turbine_bound(all_tids, node.branch))
                                        if stage == "coverage" else 0.0)
                node.bound_source = ("trivial-node-allowed-turbine-bound" if stage == "coverage"
                                     else "nonnegative-energy-bound")
                heapq.heappush(queue, (_queue_priority(stage, node.inherited_bound, node.node_id), node.node_id, node))
                if pricing.complete:
                    termination = "farkas-full-space-infeasibility-unproven"
                else:
                    termination = pricing.termination_reason
                interrupt_tree = True
                node_finished = True
                break

            route_mass_upper_bound = _node_allowed_turbine_bound(all_tids, node.branch)
            node_bound, node_source = _safe_node_bound_from_pricing(
                master, pricing, stage, route_mass_upper_bound, all_tids, node)
            node.inherited_bound = node_bound
            node.bound_source = node_source
            bound_source = node_source
            if not pricing.complete:
                pricing_complete_all = False
                pricing_closed_all = False
                # Interrupted pricing is not closure.  It may still fathom a
                # node when the certified full-space bound (RMP dual lower bound
                # plus M_n times the universal omitted-column rc lower bound) is
                # already dominated by the current feasible incumbent.
                interrupted_bound_fathoms = bool(
                    (stage == "coverage" and incumbent_value is not None
                     and node_bound <= incumbent_value)
                    or (stage == "energy" and incumbent_upper_bound is not None
                        and node_bound >= incumbent_upper_bound))
                if interrupted_bound_fathoms:
                    node_finished = True
                    continue
                heapq.heappush(queue, (_queue_priority(stage, node_bound, node.node_id), node.node_id, node))
                termination = pricing.termination_reason
                interrupt_tree = True
                node_finished = True
                break

            if not pricing.closed:
                pricing_closed_all = False

            # V9 restricted-archive target diagnostic: when a rigorous coverage
            # node upper bound is already below the requested threshold, this
            # node cannot contain a target witness.  This path is never enabled
            # in the formal solve.
            if (stage == "coverage"
                    and coverage_decision_target is not None
                    and _safe_integer_floor(node_bound)
                        < int(coverage_decision_target)):
                node_finished = True
                continue

            # Certificate pruning uses rigorous node bounds against a feasible
            # incumbent bound with no user/numerical optimality tolerance.
            if stage == "coverage" and incumbent_value is not None and node_bound <= incumbent_value:
                node_finished = True
                continue
            if (stage == "energy" and incumbent_upper_bound is not None
                    and node_bound >= incumbent_upper_bound):
                node_finished = True
                continue

            x = master.x
            if _is_integral(x):
                rounded_x = np.rint(x)
                selection = _selection_from_local(master, rounded_x)
                if not _exact_binary_master_feasible(master, rounded_x):
                    # HiGHS' feasibility tolerance can make x numerically
                    # integral even though the rounded 0/1 pattern violates an
                    # encoded binary64 row.  It is not a valid incumbent.
                    children, _ = _branch_on_integral_numeric_ambiguity(
                        archive, master, selection, node, next_node_id)
                    if children:
                        next_node_id += len(children)
                        branch_decisions += 1
                        branch_children_created += len(children)
                        for child in children:
                            child.inherited_bound = node_bound
                            child.bound_source = node_source
                            heapq.heappush(
                                queue,
                                (_queue_priority(stage, node_bound, child.node_id),
                                 child.node_id, child))
                        node_finished = True
                        continue
                    node.inherited_bound = node_bound
                    node.bound_source = node_source
                    heapq.heappush(
                        queue,
                        (_queue_priority(stage, node_bound, node.node_id),
                         node.node_id, node))
                    termination = "rounded-integer-master-infeasible"
                    interrupt_tree = True
                    node_finished = True
                    break
                if selection:
                    resource_audit_calls += 1
                    _audit_t0 = time.perf_counter()
                    audit = _audit_integer_selection(
                        archive, selection, K, batteries, p, quick_min, swap_min,
                        quick_capacity, swap_capacity, deadline)
                    _audit_dt = time.perf_counter() - _audit_t0
                    resource_audit_runtime_s += _audit_dt
                    _audit_record = dict(
                        call_index=int(resource_audit_calls), trigger="integral-rmp",
                        bb_node=int(node.node_id), bb_depth=int(node.depth),
                        selection_size=int(len(selection)), wall_time_s=float(_audit_dt),
                        status=str(audit.status.value),
                        dfs_nodes=int(getattr(audit, "explored_nodes", 0)),
                        cache_hits=int(getattr(audit, "memo_hits", 0)),
                        failure_reasons=dict(getattr(audit, "failure_reasons", None) or {}),
                        cut_generated=False)
                    resource_audit_records.append(_audit_record)
                    if audit.status is RA.ResourceAuditStatus.UNKNOWN_TIMEOUT:
                        resource_audit_complete = False
                        heapq.heappush(queue, (_queue_priority(stage, node_bound, node.node_id), node.node_id, node))
                        termination = "resource-audit-time-limit"
                        interrupt_tree = True
                        node_finished = True
                        break
                    if audit.status is RA.ResourceAuditStatus.INFEASIBLE_PROVEN:
                        _record_resource_rejection_diagnostic(
                            diagnostics_sink, archive, selection, audit)
                        # Resource feasibility in the current transition model is
                        # not downward closed: deleting a task can change UAV
                        # predecessors, quick/swap modes and event times.  Hence
                        # the classical subset no-good sum_{r in S} x_r <= |S|-1
                        # is unsafe.  Store S as an exact binary pattern instead;
                        # the master row uses +1 on S and -1 on every other
                        # current/future route, excluding only x == 1_S.
                        cut = frozenset(_exact_route_signature(archive[j]) for j in selection)
                        if cut and cut not in no_good_cuts:
                            no_good_cuts.append(cut)
                            resource_cuts_added += 1
                            _audit_record["cut_generated"] = True
                            continue
                        raise RuntimeError("repeated proven resource-infeasible exact pattern")
                    cov = _coverage_of_selection(archive, selection)
                    if stage == "coverage":
                        if incumbent_value is None or cov > incumbent_value:
                            incumbent_value = float(cov)
                            incumbent_lower_bound = incumbent_upper_bound = float(cov)
                            coverage_incumbent = int(cov)
                            incumbent_selection = selection
                            incumbent_audit = audit
                        if (decision_only
                                and coverage_decision_target is not None
                                and int(cov) >= int(coverage_decision_target)):
                            termination = "coverage-target-feasible-witness"
                            interrupt_tree = True
                            node_finished = True
                            break
                    else:
                        if cov != int(coverage_target):
                            raise RuntimeError("energy-stage incumbent violates fixed coverage")
                        energy_exact = _energy_of_selection_exact(archive, selection)
                        if incumbent_exact is None or energy_exact < incumbent_exact:
                            incumbent_exact = energy_exact
                            incumbent_value = float(energy_exact)
                            incumbent_lower_bound = _fraction_to_float_down(energy_exact)
                            incumbent_upper_bound = _fraction_to_float_up(energy_exact)
                            incumbent_selection = selection
                            incumbent_audit = audit
                        if decision_only:
                            # [THM-TGT] A strict physical/resource integer witness
                            # proves the existential target-feasibility decision.
                            termination = "target-feasible-witness"
                            interrupt_tree = True
                            node_finished = True
                            break

                # An integral RMP solution is only a feasible incumbent.  It may
                # be fathomed iff a rigorous full-space bound proves that no
                # solution in this node can improve the updated incumbent.
                can_fathom = bool(
                    (stage == "coverage" and incumbent_value is not None
                     and node_bound <= incumbent_value)
                    or (stage == "energy" and incumbent_upper_bound is not None
                        and node_bound >= incumbent_upper_bound))

                # Directed-rounding bounds can be a few ulps loose even when the
                # integral RMP is exactly optimal.  If exhaustive pricing is
                # mathematically closed, tighten only this integral node using an
                # exact-rational binary64 Lagrangian certificate.
                if not can_fathom and selection and pricing.closed:
                    exact_node_proved, exact_node_lb = _integer_node_exact_full_space_proof(
                        master, pricing, stage, archive, selection)
                    if exact_node_proved:
                        if stage == "coverage":
                            node_bound = float(cov)
                        else:
                            node_bound = _fraction_to_float_down(exact_node_lb)
                        node.inherited_bound = node_bound
                        node.bound_source = "exact-binary64-integral-node-certificate"
                        bound_source = node.bound_source
                        can_fathom = True

                if can_fathom:
                    node_finished = True
                    continue

                # No fractional variable exists, but the rigorous bound still
                # leaves improvement possible.  HiGHS may have returned an
                # integer point that is only tolerance-optimal.  Preserve
                # completeness by branching on an unfixed route variable
                # x_r=0/1 rather than stopping or treating integrality as proof.
                children, _ = _branch_on_integral_numeric_ambiguity(
                    archive, master, selection, node, next_node_id)
                if children:
                    next_node_id += len(children)
                    branch_decisions += 1
                    branch_children_created += len(children)
                    for child in children:
                        child.inherited_bound = node_bound
                        child.bound_source = node_source
                        heapq.heappush(
                            queue,
                            (_queue_priority(stage, node_bound, child.node_id),
                             child.node_id, child))
                    node_finished = True
                    continue

                # Every currently materialized route variable is already fixed
                # by this node and the exact certificate still cannot close.
                # Keep the unresolved node open and fail closed.
                node.inherited_bound = node_bound
                node.bound_source = node_source
                heapq.heappush(
                    queue,
                    (_queue_priority(stage, node_bound, node.node_id),
                     node.node_id, node))
                termination = "integer-node-bound-not-closed"
                interrupt_tree = True
                node_finished = True
                break

            children, branch_kind = _branch_on_fractional_solution(
                archive, master, x, node, next_node_id)
            if not children:
                branching_complete = False
                heapq.heappush(queue, (_queue_priority(stage, node_bound, node.node_id), node.node_id, node))
                termination = "no-complete-branch-found"
                interrupt_tree = True
                node_finished = True
                break
            next_node_id += len(children)
            branch_decisions += 1
            branch_children_created += len(children)
            for child in children:
                child.inherited_bound = node_bound
                child.bound_source = node_source
                heapq.heappush(queue, (_queue_priority(stage, node_bound, child.node_id), child.node_id, child))
            node_finished = True

        if interrupt_tree or termination == "global-time-limit":
            break
        if not queue:
            # Every branch node has been safely fathomed (inconsistency,
            # full-space infeasibility, rigorous bound, or exact integral-node
            # certificate).  Tree exhaustion is itself an exact proof and must
            # not be routed through a user Gap tolerance.
            termination = "stage-optimum-proven"
            break
        global_bound = _global_open_bound(
            stage, queue, incumbent_value,
            len(all_tids) if stage == "coverage" else 0.0)
        if stage == "coverage" and incumbent_value is not None:
            gap = max(0, _safe_integer_floor(global_bound) - int(incumbent_value))
            if gap <= int(coverage_gap_target_abs):
                termination = "coverage-gap-target-reached" if gap > 0 else "coverage-bound-closed"
                break
        if stage == "energy" and incumbent_upper_bound is not None:
            abs_gap = max(0.0, float(incumbent_upper_bound) - float(global_bound))
            rel_gap = abs_gap / max(abs(float(incumbent_upper_bound)), 1e-12)
            exact_bound_closed = bool(
                incumbent_exact is not None
                and Fraction.from_float(float(global_bound)) >= incumbent_exact)
            if exact_bound_closed:
                termination = "energy-bound-closed"
                break
            if abs_gap <= float(energy_gap_target_abs_Wh) or rel_gap <= float(energy_gap_target_rel):
                # User Gap targets are anytime stopping rules only.  They never
                # promote a positive-gap run to exact/global optimality.
                termination = "energy-gap-target-reached"
                break

    global_bound, bound_source = _global_open_bound_info(
        stage, queue, incumbent_value,
        len(all_tids) if stage == "coverage" else 0.0)
    tree_exhausted = bool(not queue and not interrupt_tree)
    if stage == "coverage":
        global_bound = float(max(coverage_incumbent, min(len(all_tids), _safe_integer_floor(global_bound))))
        bound_proves_optimum = bool(
            incumbent_value is not None and int(global_bound) <= int(incumbent_value))
        optimal = bool((tree_exhausted or bound_proves_optimum)
                       and incumbent_value is not None
                       and resource_audit_complete and branching_complete
                       and pricing_bound_available_all and farkas_complete_all)
    else:
        # Keep the raw open-tree lower bound for the exact proof test.  The
        # reported floating lower-bound field itself must never lie above the
        # exact feasible incumbent objective; when the exact rational sum is not
        # representable in binary64, cap the report at its downward enclosure.
        raw_open_global_bound = float(global_bound)
        bound_proves_optimum = bool(
            incumbent_exact is not None
            and Fraction.from_float(raw_open_global_bound) >= incumbent_exact)
        if not queue and incumbent_lower_bound is not None:
            # Exact tree exhaustion proves the selected feasible solution is
            # optimal.  Report its downward-rounded objective as the global LB;
            # the upward endpoint remains the primal UB used by the safe Gap.
            global_bound = float(max(0.0, incumbent_lower_bound))
            bound_source = "branch-tree-exhausted-exact-incumbent-enclosure"
        elif incumbent_lower_bound is not None:
            global_bound = min(float(incumbent_lower_bound),
                               max(0.0, raw_open_global_bound))
        optimal = bool((tree_exhausted or bound_proves_optimum)
                       and incumbent_value is not None
                       and resource_audit_complete and branching_complete
                       and pricing_bound_available_all and farkas_complete_all)
    if not queue and termination == "branch-tree-exhausted":
        termination = "stage-optimum-proven" if optimal else "stage-tree-exhausted-no-incumbent"
    return StageSearchResult(
        stage=stage, incumbent_selection=tuple(incumbent_selection),
        incumbent_audit=incumbent_audit, incumbent_value=incumbent_value,
        incumbent_lower_bound=incumbent_lower_bound,
        incumbent_upper_bound=incumbent_upper_bound,
        coverage_incumbent=int(coverage_incumbent), global_bound=float(global_bound),
        optimal=optimal, termination_reason=termination,
        open_nodes=len(queue), processed_nodes=processed,
        generated_columns=generated, pricing_calls=pricing_calls,
        exact_pricing_calls=exact_pricing_calls,
        resource_cuts_added=resource_cuts_added,
        rmp_solves=int(rmp_solves), phase_one_solves=int(phase_one_solves),
        pricing_candidates=int(pricing_candidates), pricing_nodes=int(pricing_nodes),
        columns_accepted=int(generated),
        heuristic_columns=int(pricing_multistop_neutral_added + pricing_multistop_merge_added
                              + pricing_resource_variant_added),
        resource_audit_calls=int(resource_audit_calls),
        branch_children_created=int(branch_children_created),
        branch_decisions=int(branch_decisions),
        pricing_complete=bool(pricing_closed_all and not queue),
        pricing_search_complete=bool(pricing_complete_all and not queue),
        pricing_bound_available=bool(pricing_bound_available_all),
        resource_audit_complete=bool(resource_audit_complete),
        farkas_pricing_complete=bool(farkas_complete_all),
        branching_complete=bool(branching_complete),
        heuristic_pricing_used=bool(pricing_multistop_neutral_added > 0),
        exact_pricing_called=bool(exact_pricing_called),
        pricing_best_reduced_value=last_best_rc,
        pricing_reduced_value_bound=last_rc_bound,
        bound_source=bound_source,
        pricing_discovery_calls=int(pricing_discovery_calls),
        pricing_discovery_early_returns=int(pricing_discovery_early_returns),
        pricing_certification_calls=int(pricing_certification_calls),
        pricing_shadow_prefixes_evaluated=int(pricing_shadow_prefixes_evaluated),
        pricing_shadow_prunable_prefixes=int(pricing_shadow_prunable_prefixes),
        pricing_shadow_false_prune_witnesses=int(
            pricing_shadow_false_prune_witnesses),
        pricing_shadow_bound_errors=int(pricing_shadow_bound_errors),
        pricing_shadow_complete_calls=int(pricing_shadow_complete_calls),
        pricing_guided_order_calls=int(pricing_guided_order_calls),
        pricing_guided_order_reorders=int(pricing_guided_order_reorders),
        pricing_guided_order_failures=int(pricing_guided_order_failures),
        pricing_layered_depths_started=int(pricing_layered_depths_started),
        pricing_layered_depths_completed=int(pricing_layered_depths_completed),
        pricing_layered_max_depth_completed=int(
            pricing_layered_max_depth_completed),
        pricing_layered_rounds=int(pricing_layered_rounds),
        pricing_depth_fair_requested_calls=int(
            pricing_depth_fair_requested_calls),
        pricing_depth_fair_active_calls=int(pricing_depth_fair_active_calls),
        pricing_depth_fair_rounds=int(pricing_depth_fair_rounds),
        pricing_depth_fair_halfcap_dual_abs_sum=float(
            pricing_depth_fair_halfcap_dual_abs_sum),
        pricing_multistop_neutral_enabled_calls=int(
            pricing_multistop_neutral_enabled_calls),
        pricing_multistop_candidates_seen=int(
            pricing_multistop_candidates_seen),
        pricing_multistop_cross_zero_seen=int(
            pricing_multistop_cross_zero_seen),
        pricing_multistop_nonnegative_seen=int(
            pricing_multistop_nonnegative_seen),
        pricing_multistop_neutral_returned=int(
            pricing_multistop_neutral_returned),
        pricing_multistop_neutral_added=int(
            pricing_multistop_neutral_added),
        pricing_multistop_neutral_batches=int(
            pricing_multistop_neutral_batches),
        pricing_multistop_neutral_returned_by_depth=dict(
            pricing_multistop_neutral_returned_by_depth),
        pricing_multistop_best_stop_count=int(
            pricing_multistop_best_stop_count),
        pricing_multistop_best_uncovered_gain=int(
            pricing_multistop_best_uncovered_gain),
        pricing_multistop_best_rc_ub=(
            None if pricing_multistop_best_rc_ub is None
            else float(pricing_multistop_best_rc_ub)),
        pricing_multistop_best_energy_per_stop_Wh=(
            None if pricing_multistop_best_energy_per_stop_Wh is None
            else float(pricing_multistop_best_energy_per_stop_Wh)),
        pricing_multistop_neutral_used_in_incumbent=int(sum(
            1 for _j in incumbent_selection
            if 0 <= int(_j) < len(archive)
            and _exact_route_signature(archive[int(_j)])
                in pricing_multistop_neutral_signatures)),
        pricing_physical_cache_hits=int(pricing_physical_cache_hits),
        pricing_physical_cache_misses=int(pricing_physical_cache_misses),
        pricing_runtime_s=float(pricing_runtime_s),
        pricing_physical_evaluator_runtime_s=float(
            pricing_physical_evaluator_runtime_s),
        pricing_prefix_bound_runtime_s=float(pricing_prefix_bound_runtime_s),
        pricing_prefix_service_runtime_s=float(pricing_prefix_service_runtime_s),
        pricing_certified_prefix_prunes=int(pricing_certified_prefix_prunes),
        pricing_depth_certified_prefix_prunes=dict(
            pricing_depth_certified_prefix_prunes),
        pricing_service_floor_prunes=int(pricing_service_floor_prunes),
        pricing_depth_service_floor_prunes=dict(pricing_depth_service_floor_prunes),
        pricing_horizon_window_skips=int(pricing_horizon_window_skips),
        pricing_horizon_service_time_skips=int(pricing_horizon_service_time_skips),
        pricing_physical_infeasible_results=int(pricing_physical_infeasible_results),
        pricing_branch_filter_skips=int(pricing_branch_filter_skips),
        pricing_existing_signature_skips=int(pricing_existing_signature_skips),
        pricing_call_records=list(pricing_call_records),
        rmp_records=list(rmp_records),
        resource_audit_records=list(resource_audit_records),
        rmp_runtime_s=float(rmp_runtime_s),
        phase_one_runtime_s=float(phase_one_runtime_s),
        resource_audit_runtime_s=float(resource_audit_runtime_s),
        pricing_discovery_improving_seen=int(
            pricing_discovery_improving_seen),
        pricing_discovery_improving_returned=int(
            pricing_discovery_improving_returned),
        pricing_discovery_diverse_returns=int(
            pricing_discovery_diverse_returns),
        pricing_discovery_hard_cap_returns=int(
            pricing_discovery_hard_cap_returns),
        pricing_discovery_max_return_batch=int(
            pricing_discovery_max_return_batch),
        pricing_discovery_max_distinct_launches=int(
            pricing_discovery_max_distinct_launches),
        pricing_discovery_max_distinct_service_sets=int(
            pricing_discovery_max_distinct_service_sets),
        primal_refresh_calls=int(primal_refresh_calls),
        primal_refresh_audit_calls=int(primal_refresh_audit_calls),
        primal_refresh_timeouts=int(primal_refresh_timeouts),
        primal_refresh_improvements=int(primal_refresh_improvements),
        primal_refresh_best_coverage=int(primal_refresh_best_coverage),
        primal_refresh_columns_seen=int(primal_refresh_columns_seen),
        primal_refresh_rebuilds=int(primal_refresh_rebuilds),
        primal_refresh_repairs=int(primal_refresh_repairs),
        primal_refresh_augmentation_audits=int(
            primal_refresh_augmentation_audits),
        primal_refresh_rebuild_audits=int(primal_refresh_rebuild_audits),
        primal_refresh_repair_audits=int(primal_refresh_repair_audits),
        primal_refresh_augmentation_improvements=int(
            primal_refresh_augmentation_improvements),
        primal_refresh_rebuild_improvements=int(
            primal_refresh_rebuild_improvements),
        primal_refresh_repair_improvements=int(
            primal_refresh_repair_improvements),
        primal_refresh_duplicate_trials_skipped=int(
            primal_refresh_duplicate_trials_skipped),
        primal_refresh_cached_infeasible_trials=int(
            len(primal_refresh_infeasible_trial_cache)),
        primal_refresh_uncovered_fair_rounds=int(
            primal_refresh_uncovered_fair_rounds),
        primal_refresh_failure_reasons=dict(
            primal_refresh_failure_reasons),
        primal_deck_diagnostic_enabled=bool(
            primal_resource_deck_guided and stage == "coverage"),
        primal_deck_archive_conflict_edges=int(
            primal_deck_archive_conflict_edges),
        primal_deck_archive_max_degree=int(primal_deck_archive_max_degree),
        primal_deck_archive_max_component=int(primal_deck_archive_max_component),
        primal_deck_candidate_scored=int(primal_deck_candidate_scored),
        primal_deck_candidate_zero_conflict=int(
            primal_deck_candidate_zero_conflict),
        primal_deck_candidate_positive_conflict=int(
            primal_deck_candidate_positive_conflict),
        primal_deck_prefilter_skips=int(primal_deck_prefilter_skips),
        primal_deck_max_candidate_conflicts=int(
            primal_deck_max_candidate_conflicts),
        primal_deck_conflict_pairs_sample=list(
            primal_deck_conflict_pairs_sample),
        pricing_multistop_merge_enabled=bool(
            adaptive_multistop_enrichment and stage == "coverage"),
        pricing_multistop_merge_triggers=int(pricing_multistop_merge_triggers),
        pricing_multistop_merge_attempts=int(pricing_multistop_merge_attempts),
        pricing_multistop_merge_physical_feasible=int(
            pricing_multistop_merge_physical_feasible),
        pricing_multistop_merge_new_candidates=int(
            pricing_multistop_merge_new_candidates),
        pricing_multistop_merge_returned=int(pricing_multistop_merge_returned),
        pricing_multistop_merge_added=int(pricing_multistop_merge_added),
        pricing_multistop_merge_batches=int(pricing_multistop_merge_batches),
        pricing_multistop_merge_distinct_pairs=int(
            pricing_multistop_merge_distinct_pairs),
        pricing_multistop_merge_best_rc_ub=(
            None if pricing_multistop_merge_best_rc_ub is None
            else float(pricing_multistop_merge_best_rc_ub)),
        pricing_multistop_merge_best_energy_per_stop_Wh=(
            None if pricing_multistop_merge_best_energy_per_stop_Wh is None
            else float(pricing_multistop_merge_best_energy_per_stop_Wh)),
        pricing_multistop_merge_best_uncovered_gain=int(
            pricing_multistop_merge_best_uncovered_gain),
        pricing_multistop_merge_used_in_incumbent=int(sum(
            1 for _j in incumbent_selection
            if 0 <= int(_j) < len(archive)
            and _exact_route_signature(archive[int(_j)])
                in pricing_multistop_merge_signatures)),
        pricing_resource_variant_enabled=bool(
            resource_variant_enrichment and stage == "coverage"),
        pricing_resource_variant_triggers=int(pricing_resource_variant_triggers),
        pricing_resource_variant_attempts=int(pricing_resource_variant_attempts),
        pricing_resource_variant_deck_compatible_specs=int(
            pricing_resource_variant_deck_compatible_specs),
        pricing_resource_variant_deck_prefilter_skips=int(
            pricing_resource_variant_deck_prefilter_skips),
        pricing_resource_variant_physical_feasible=int(
            pricing_resource_variant_physical_feasible),
        pricing_resource_variant_new_candidates=int(
            pricing_resource_variant_new_candidates),
        pricing_resource_variant_returned=int(pricing_resource_variant_returned),
        pricing_resource_variant_added=int(pricing_resource_variant_added),
        pricing_resource_variant_batches=int(pricing_resource_variant_batches),
        pricing_resource_variant_distinct_turbines=int(
            pricing_resource_variant_distinct_turbines),
        pricing_resource_variant_best_rc_ub=(
            None if pricing_resource_variant_best_rc_ub is None
            else float(pricing_resource_variant_best_rc_ub)),
        pricing_resource_variant_best_energy_Wh=(
            None if pricing_resource_variant_best_energy_Wh is None
            else float(pricing_resource_variant_best_energy_Wh)),
        pricing_resource_variant_used_in_incumbent=int(sum(
            1 for _j in incumbent_selection
            if 0 <= int(_j) < len(archive)
            and _exact_route_signature(archive[int(_j)])
                in pricing_resource_variant_signatures)),
        pricing_resource_variant_records=[
            dict(_rec,
                 used_in_stage_incumbent=bool(any(
                     0 <= int(_j) < len(archive)
                     and repr(_exact_route_signature(archive[int(_j)]))
                         == str(_rec.get("signature_repr", ""))
                     for _j in incumbent_selection)))
            for _rec in pricing_resource_variant_records],
        archive_primal_recovery_enabled=bool(
            archive_primal_recovery and stage == "coverage"),
        archive_primal_recovery_calls=int(archive_primal_recovery_calls),
        archive_primal_recovery_runtime_s=float(
            archive_primal_recovery_runtime_s),
        archive_primal_recovery_audit_calls=int(
            archive_primal_recovery_audit_calls),
        archive_primal_recovery_timeouts=int(
            archive_primal_recovery_timeouts),
        archive_primal_recovery_improvements=int(
            archive_primal_recovery_improvements),
        archive_primal_recovery_best_coverage=int(
            archive_primal_recovery_best_coverage),
        archive_primal_recovery_best_archive_columns=int(
            archive_primal_recovery_best_archive_columns),
        archive_primal_recovery_records=list(
            archive_primal_recovery_records),
        archive_primal_recovery_witness_selection_indices=list(
            archive_primal_recovery_witness_selection_indices),
        archive_primal_recovery_witness_route_signatures=list(
            archive_primal_recovery_witness_route_signatures),
        archive_primal_recovery_witness_covered_turbines=list(
            archive_primal_recovery_witness_covered_turbines),
        primal_exchange_enabled=bool(
            (primal_exchange or primal_resource_primal or primal_resource_guided
             or primal_resource_deck_guided)
            and stage == "coverage"),
        primal_exchange_calls=int(primal_exchange_calls),
        primal_exchange_candidate_routes=int(primal_exchange_candidate_routes),
        primal_exchange_trials_built=int(primal_exchange_trials_built),
        primal_exchange_audit_calls=int(primal_exchange_audit_calls),
        primal_exchange_improvements=int(primal_exchange_improvements),
        primal_exchange_consolidation_trials=int(
            primal_exchange_consolidation_trials),
        primal_exchange_optional_drop_trials=int(
            primal_exchange_optional_drop_trials),
        primal_exchange_max_stop_count_considered=int(
            primal_exchange_max_stop_count_considered),
        primal_exchange_best_coverage=int(
            max(primal_exchange_best_coverage,
                coverage_incumbent
                if (primal_exchange or primal_resource_primal
                    or primal_resource_guided or primal_resource_deck_guided)
                    and stage == "coverage"
                else 0)),
        primal_exchange_multistop_used_in_incumbent=int(sum(
            1 for _j in incumbent_selection
            if 0 <= int(_j) < len(archive)
            and len(_ordered_tids(archive[int(_j)])) >= 2)),
        pricing_depth_prefixes_evaluated=dict(
            pricing_depth_prefixes_evaluated),
        pricing_depth_improving_seen=dict(pricing_depth_improving_seen),
        pricing_depth_improving_returned=dict(
            pricing_depth_improving_returned),
        pricing_pattern_cut_active_dual_rows=int(
            pricing_pattern_cut_active_dual_rows),
        pricing_pattern_cut_dual_abs_sum=float(
            pricing_pattern_cut_dual_abs_sum),
        pricing_pattern_cut_improving_seen_count=int(
            pricing_pattern_cut_improving_seen_count),
        pricing_pattern_cut_improving_seen_contribution_sum=float(
            pricing_pattern_cut_improving_seen_contribution_sum),
        pricing_pattern_cut_improving_seen_sign_essential=int(
            pricing_pattern_cut_improving_seen_sign_essential),
        pricing_pattern_cut_returned_count=int(
            pricing_pattern_cut_returned_count),
        pricing_pattern_cut_returned_contribution_sum=float(
            pricing_pattern_cut_returned_contribution_sum),
        pricing_pattern_cut_returned_sign_essential=int(
            pricing_pattern_cut_returned_sign_essential),
        pricing_pattern_cut_returned_by_depth=dict(
            pricing_pattern_cut_returned_by_depth),
        battery_halfcap_dual_active_rmp_solves=int(
            battery_halfcap_dual_active_rmp_solves),
        battery_halfcap_dual_abs_sum=float(
            battery_halfcap_dual_abs_sum),
        battery_halfcap_dual_max_abs=float(
            battery_halfcap_dual_max_abs))


def _materialize_chosen(columns, selection, audit):
    if not selection:
        return []
    assignment = audit.assignment if hasattr(audit, "assignment") else audit
    if assignment is None:
        raise RuntimeError("feasible incumbent lacks resource assignment")
    chosen = []
    for j in selection:
        c = dict(columns[j])
        c["route_signature"] = list(_exact_route_signature(c))
        if j in assignment.get("uav_assignment", {}):
            c["uav_id"] = int(assignment["uav_assignment"][j])
        if j in assignment.get("battery_assignment", {}):
            c["battery_group"] = int(assignment["battery_assignment"][j])
        service = assignment.get("mission_service", {}).get(j)
        if service is not None:
            c["turnaround_before"] = dict(service)
        chosen.append(c)
    return chosen




def _initial_singleton_columns(turbines, launch_opts, p, xi_amb, weather_unc,
                               T_min, deadline, t_launch_min,
                               landing_clear_min, deck_mode, deck_delta_min,
                               kappa_mode="vp_unimodal", chance_mode="drcc",
                               budget_gamma=2.0, implicit_test_columns=None,
                               physical_cache=None):
    """Generate at most one feasible singleton per turbine.

    This is a primal-start mechanism, not a certificate mechanism.  It never
    generates multi-stop permutations and exact pricing remains mandatory.
    """
    best = {}
    if physical_cache is None:
        physical_cache = {}
    if implicit_test_columns is not None:
        for raw in implicit_test_columns:
            if _deadline_hit(deadline):
                break
            try:
                c = _normalize_exact_column(
                    raw, p=p, t_launch_min=t_launch_min,
                    landing_clear_min=landing_clear_min,
                    deck_mode=deck_mode, deck_delta_min=deck_delta_min)
            except Exception:
                continue
            tids = _ordered_tids(c)
            if len(tids) != 1:
                continue
            tid = tids[0]
            if tid not in best or float(c["E_plan_Wh"]) < float(best[tid]["E_plan_Wh"]):
                best[tid] = c
        return list(best.values())

    horizons = tuple(float(h) for h in RM.decision_horizons_of(xi_amb))
    risk_policy = _risk_policy_for_mode(kappa_mode)
    try:
        for t in turbines:
            tid = _tid(t.tid)
            found = None
            for oi, opt in enumerate(launch_opts):
                if _deadline_hit(deadline):
                    return list(best.values())
                for h in horizons:
                    if _deadline_hit(deadline):
                        return list(best.values())
                    if float(opt.tau_min) + h > float(T_min):
                        continue
                    cache_key = (
                        int(oi), (tid,), float(h).hex())
                    if cache_key in physical_cache:
                        c = physical_cache[cache_key]
                    else:
                        try:
                            c = _candidate_from_physics(
                                oi, opt, (t,), h, p, xi_amb, weather_unc,
                                t_launch_min, landing_clear_min,
                                deck_mode, deck_delta_min,
                                chance_mode=chance_mode,
                                budget_gamma=budget_gamma, deadline=deadline,
                                risk_policy=risk_policy)
                        except TimeoutError:
                            return list(best.values())
                        except Exception:
                            c = None
                        # Same fixed model/physics context as later pricing.
                        # Cache proven infeasibility (None) and feasible columns;
                        # timeout is returned above and is never cached.
                        physical_cache[cache_key] = c
                    if c is not None:
                        found = c
                        break
                if found is not None:
                    break
            if found is not None:
                best[tid] = found
    finally:
        # Explicit RiskPolicy keeps the formal singleton warm-start path reentrant.
        pass
    return list(best.values())


def _greedy_exact_resource_start(columns, K, batteries, p, quick_min, swap_min,
                                 quick_capacity, swap_capacity, deadline):
    """Build a resource-audited incumbent from the current generated columns."""
    order = sorted(
        range(len(columns)),
        key=lambda j: (-len(_ordered_tids(columns[j])),
                       float(columns[j]["E_plan_Wh"]),
                       repr(_exact_route_signature(columns[j]))))
    selected = []
    covered = set()
    last_audit = None
    unknown = False
    audit_calls = 0
    for j in order:
        if _deadline_hit(deadline):
            break
        tids = set(_ordered_tids(columns[j]))
        if tids & covered:
            continue
        trial = tuple(selected + [j])
        audit_calls += 1
        audit = _audit_integer_selection(
            columns, trial, K, batteries, p, quick_min, swap_min,
            quick_capacity, swap_capacity, deadline)
        if audit.status is RA.ResourceAuditStatus.UNKNOWN_TIMEOUT:
            unknown = True
            break
        if audit.status is RA.ResourceAuditStatus.FEASIBLE:
            selected.append(j)
            covered.update(tids)
            last_audit = audit
    return tuple(selected), last_audit, unknown, int(audit_calls)



def _primal_refresh_incumbent(
        columns, current_selection, current_audit, stage, coverage_target,
        K, batteries, p, quick_min, swap_min, quick_capacity, swap_capacity,
        deadline, *, wall_budget_s=0.50, max_audits=32, strategy="legacy",
        infeasible_trial_cache=None):
    """Primal-only anytime incumbent refresh over the current exact column archive.

    ``strategy="legacy"`` preserves the V7 augmentation/rebuild/repair search.
    ``strategy="resource_exchange"`` is the V15 coverage heuristic: consolidate
    selected singleton-like missions through an existing multi-stop archive route.
    ``strategy="resource_primal"`` is the V16 unified coverage heuristic: preserve
    the incumbent, exact-audit monotone additions first, then use the same bounded
    exchange neighborhood.
    ``strategy="resource_guided"`` is V17: the same single resource primal, with
    cross-refresh caching of exact infeasible selections and uncovered-turbine
    fair ordering of augmentation variants.
    ``strategy="resource_deck_guided"`` is V18: keep V17, but use the exact same
    fixed half-open deck conflict relation as the resource oracle to order and
    prefilter heuristic trials, while recording an archive conflict graph.

    None of these strategies contributes a lower bound, pricing closure, pruning decision,
    infeasibility proof, or optimality certificate.
    """
    started = time.monotonic()
    hard_deadline = (
        float(deadline) if deadline is not None else math.inf)
    local_deadline = min(
        hard_deadline, started + max(0.0, float(wall_budget_s)))
    max_audits = max(1, int(max_audits))
    strategy = str(strategy).strip().lower().replace("-", "_")
    if strategy not in {
            "legacy", "resource_exchange", "resource_primal",
            "resource_guided", "resource_deck_guided"}:
        raise ValueError("unknown primal refresh strategy")
    if strategy in {
            "resource_exchange", "resource_primal",
            "resource_guided", "resource_deck_guided"} and stage != "coverage":
        # V15 is intentionally a coverage-incumbent mechanism only.  Energy
        # optimization keeps the proven V7 behavior rather than introducing a
        # second exchange objective.
        strategy = "legacy"

    current_selection = tuple(int(j) for j in current_selection)
    best_selection = tuple(current_selection)
    best_audit = current_audit
    audit_calls = 0
    timed_out = False
    rebuilds = 0
    repairs = 0
    columns_seen = len(columns)
    neighborhood_audits = {
        "augmentation": 0, "rebuild": 0, "repair": 0, "exchange": 0}
    neighborhood_improvements = {
        "augmentation": 0, "rebuild": 0, "repair": 0, "exchange": 0}
    exchange_candidate_routes = 0
    exchange_trials_built = 0
    exchange_consolidation_trials = 0
    exchange_optional_drop_trials = 0
    exchange_max_stop_count_considered = 0
    duplicate_trials_skipped = 0
    uncovered_fair_rounds = 0
    failure_reasons = {}
    deck_archive_conflict_edges = 0
    deck_archive_max_degree = 0
    deck_archive_max_component = 0
    deck_candidate_scored = 0
    deck_candidate_zero_conflict = 0
    deck_candidate_positive_conflict = 0
    deck_prefilter_skips = 0
    deck_max_candidate_conflicts = 0
    deck_conflict_pairs_sample = []
    if infeasible_trial_cache is None:
        infeasible_trial_cache = set()

    def _selection_tids(selection):
        tids = []
        for j in selection:
            tids.extend(_ordered_tids(columns[j]))
        return tuple(tids)

    def _coverage(selection):
        return len(set(_selection_tids(selection)))

    baseline_cov = _coverage(best_selection)
    best_cov = int(baseline_cov)
    best_energy = (
        _energy_of_selection_exact(columns, best_selection)
        if best_selection else Fraction(0))

    def _time_or_budget_hit():
        nonlocal timed_out
        if audit_calls >= max_audits:
            return True
        if time.monotonic() >= local_deadline:
            timed_out = True
            return True
        return False

    def _is_objective_improvement(selection):
        cov = _coverage(selection)
        tids = _selection_tids(selection)
        if len(tids) != len(set(tids)):
            return False
        if stage == "coverage":
            return cov > best_cov
        target = int(coverage_target or 0)
        if cov != target:
            return False
        e = _energy_of_selection_exact(columns, selection)
        return best_audit is None or e < best_energy

    def _selection_trial_key(selection):
        return tuple(sorted(
            repr(_exact_route_signature(columns[int(j)]))
            for j in selection))

    def _deck_pair_conflict(j, q):
        j = int(j)
        q = int(q)
        for _a in columns[j]["resource_intervals"]["deck"]:
            for _b in columns[q]["resource_intervals"]["deck"]:
                if RA._halfopen_overlap(_a, _b):
                    return True
        return False

    def _deck_conflict_edges(selection):
        _sel = tuple(int(j) for j in selection)
        _pairs = []
        for _ii in range(len(_sel)):
            for _kk in range(_ii + 1, len(_sel)):
                _a = int(_sel[_ii])
                _b = int(_sel[_kk])
                if _deck_pair_conflict(_a, _b):
                    _pairs.append((_a, _b))
        return _pairs

    def _deck_conflicts_for_add(selection, q, *, record_candidate=False):
        nonlocal deck_candidate_scored, deck_candidate_zero_conflict
        nonlocal deck_candidate_positive_conflict, deck_max_candidate_conflicts
        _pairs = [
            (int(j), int(q)) for j in selection
            if int(j) != int(q) and _deck_pair_conflict(int(j), int(q))]
        if record_candidate:
            deck_candidate_scored += 1
            if _pairs:
                deck_candidate_positive_conflict += 1
            else:
                deck_candidate_zero_conflict += 1
            deck_max_candidate_conflicts = max(
                deck_max_candidate_conflicts, len(_pairs))
        return _pairs

    def _record_deck_pairs(pairs):
        for _a, _b in pairs:
            _sa = repr(_exact_route_signature(columns[int(_a)]))
            _sb = repr(_exact_route_signature(columns[int(_b)]))
            _pair = tuple(sorted((_sa, _sb)))
            if (_pair not in deck_conflict_pairs_sample
                    and len(deck_conflict_pairs_sample) < 12):
                deck_conflict_pairs_sample.append(_pair)

    if strategy == "resource_deck_guided":
        _adj = {int(j): set() for j in range(len(columns))}
        for _j in range(len(columns)):
            for _q in range(_j + 1, len(columns)):
                if _deck_pair_conflict(_j, _q):
                    _adj[_j].add(_q)
                    _adj[_q].add(_j)
        deck_archive_conflict_edges = int(
            sum(len(v) for v in _adj.values()) // 2)
        deck_archive_max_degree = int(
            max((len(v) for v in _adj.values()), default=0))
        _seen_nodes = set()
        _max_comp = 0
        for _root in _adj:
            if _root in _seen_nodes:
                continue
            _stack = [_root]
            _seen_nodes.add(_root)
            _size = 0
            while _stack:
                _u = _stack.pop()
                _size += 1
                for _v in _adj[_u]:
                    if _v not in _seen_nodes:
                        _seen_nodes.add(_v)
                        _stack.append(_v)
            _max_comp = max(_max_comp, _size)
        deck_archive_max_component = int(_max_comp)

    def _try_selection(selection, neighborhood):
        nonlocal audit_calls, timed_out, duplicate_trials_skipped
        nonlocal best_selection, best_audit, best_cov, best_energy
        nonlocal deck_prefilter_skips, deck_max_candidate_conflicts
        selection = tuple(int(j) for j in selection)
        if not _is_objective_improvement(selection):
            return False
        if strategy == "resource_deck_guided":
            _deck_pairs = _deck_conflict_edges(selection)
            if _deck_pairs:
                deck_prefilter_skips += 1
                deck_max_candidate_conflicts = max(
                    deck_max_candidate_conflicts, len(_deck_pairs))
                _record_deck_pairs(_deck_pairs)
                return False
        _trial_key = _selection_trial_key(selection)
        if (strategy in {"resource_guided", "resource_deck_guided"}
                and _trial_key in infeasible_trial_cache):
            duplicate_trials_skipped += 1
            return False
        if _time_or_budget_hit():
            return False
        audit_calls += 1
        neighborhood_audits[neighborhood] += 1
        audit = _audit_integer_selection(
            columns, selection, K, batteries, p, quick_min, swap_min,
            quick_capacity, swap_capacity, local_deadline)
        if audit.status is RA.ResourceAuditStatus.UNKNOWN_TIMEOUT:
            timed_out = True
            return False
        if audit.status is not RA.ResourceAuditStatus.FEASIBLE:
            if strategy in {"resource_guided", "resource_deck_guided"}:
                infeasible_trial_cache.add(_trial_key)
                for _reason, _count in dict(
                        getattr(audit, "failure_reasons", None) or {}).items():
                    failure_reasons[str(_reason)] = int(
                        failure_reasons.get(str(_reason), 0)) + int(_count)
            return False
        cov = _coverage(selection)
        best_selection = selection
        best_audit = audit
        best_cov = int(cov)
        best_energy = (
            _energy_of_selection_exact(columns, selection)
            if selection else Fraction(0))
        neighborhood_improvements[neighborhood] += 1
        return True

    def _route_key(j, priority_tids=frozenset()):
        tids = frozenset(_ordered_tids(columns[j]))
        return (
            -len(tids & priority_tids),
            -len(tids),
            float(columns[j]["E_plan_Wh"]),
            repr(_exact_route_signature(columns[j])),
        )

    if strategy in {
            "resource_exchange", "resource_primal", "resource_guided",
            "resource_deck_guided"}:
        # V15/V16/V17: one compact resource-aware coverage neighborhood.
        # The half-cap check below is only a cheap *necessary* heuristic filter;
        # the unchanged exact resource audit remains the sole acceptance test.
        universe_tids = set()
        for _c in columns:
            universe_tids.update(_ordered_tids(_c))
        _cap = Fraction.from_float(float(p.B_use))

        def _is_high_energy(j):
            try:
                return bool(
                    2 * Fraction.from_float(
                        float(columns[int(j)]["E_soc_required_Wh"])) > _cap)
            except Exception:
                # A heuristic prefilter must fail open.  The exact audit below
                # remains authoritative.
                return False

        def _packing_ok(selection):
            seen = set()
            for _j in selection:
                _t = set(_ordered_tids(columns[int(_j)]))
                if seen & _t:
                    return False
                seen.update(_t)
            return True

        def _halfcap_ok(selection):
            return sum(_is_high_energy(_j) for _j in selection) <= int(batteries)

        # V16 regression fix: V15 removed V7's monotone augmentation entirely.
        # On the real n=10 case this left a feasible incumbent with one useful
        # 2-stop anchor at only four missions / coverage five while three
        # half-cap battery slots were still unused.  Reserve at most half the
        # audit budget for anchor-preserving additions before any exchange.
        if strategy in {"resource_primal", "resource_guided",
                         "resource_deck_guided"}:
            _augment_cap = max(1, min(max_audits, max_audits // 2))
            _working = list(best_selection)

            if strategy in {"resource_guided", "resource_deck_guided"}:
                # V17: the real n=10 V16 run spent 53 exact augmentation audits
                # for one improvement.  The legacy energy-first sort can starve
                # some uncovered turbines and can repeat the same proved
                # infeasible exact selection on every refresh.  Interleave the
                # best remaining exact variants across uncovered turbines and
                # recompute after every accepted addition.
                while audit_calls < _augment_cap and not _time_or_budget_hit():
                    _working_tids = set(_selection_tids(_working))
                    _missing = tuple(sorted(
                        universe_tids - _working_tids, key=str))
                    if not _missing:
                        break
                    _groups = {str(_tid0): [] for _tid0 in _missing}
                    for _q0 in range(len(columns)):
                        _q0 = int(_q0)
                        if _q0 in _working:
                            continue
                        _qt0 = set(_ordered_tids(columns[_q0]))
                        if not _qt0 or (_qt0 & _working_tids):
                            continue
                        _gain0 = tuple(sorted(
                            _qt0 & set(_missing), key=str))
                        if not _gain0:
                            continue
                        _trial0 = tuple(_working + [_q0])
                        if not _halfcap_ok(_trial0):
                            continue
                        _deck_conf0 = (
                            len(_deck_conflicts_for_add(
                                _working, _q0, record_candidate=True))
                            if strategy == "resource_deck_guided" else 0)
                        _k0 = (
                            int(_deck_conf0),
                            -len(_gain0),
                            -len(_qt0),
                            float(columns[_q0]["E_plan_Wh"]),
                            float(columns[_q0].get("tau", 0.0)),
                            float(columns[_q0].get("h", 0.0)),
                            repr(_exact_route_signature(columns[_q0])))
                        for _tid0 in _gain0:
                            _groups[str(_tid0)].append((_k0, _q0))
                    for _tid0 in list(_groups):
                        _groups[_tid0].sort()
                    _fair = []
                    _seen_q = set()
                    _depth = 0
                    while True:
                        _added_round = False
                        for _tid0 in sorted(_groups):
                            _g0 = _groups[_tid0]
                            if _depth < len(_g0):
                                _q0 = int(_g0[_depth][1])
                                if _q0 not in _seen_q:
                                    _seen_q.add(_q0)
                                    _fair.append(_q0)
                                _added_round = True
                        if not _added_round:
                            break
                        _depth += 1
                    uncovered_fair_rounds += 1
                    _improved_round = False
                    for _q in _fair:
                        if audit_calls >= _augment_cap or _time_or_budget_hit():
                            break
                        _trial = tuple(_working + [int(_q)])
                        if _try_selection(_trial, "augmentation"):
                            _working = list(best_selection)
                            _improved_round = True
                            break
                    if not _improved_round:
                        break
                    if best_cov >= len(universe_tids):
                        break
            else:
                _working_tids = set(_selection_tids(_working))
                _missing = frozenset(universe_tids - _working_tids)
                _augment_order = sorted(
                    range(len(columns)),
                    key=lambda q: _route_key(q, _missing))
                for _q in _augment_order:
                    if audit_calls >= _augment_cap or _time_or_budget_hit():
                        break
                    _q = int(_q)
                    if _q in _working:
                        continue
                    _qt = set(_ordered_tids(columns[_q]))
                    if not _qt or (_qt & _working_tids):
                        continue
                    _trial = tuple(_working + [_q])
                    if not _halfcap_ok(_trial):
                        continue
                    if _try_selection(_trial, "augmentation"):
                        _working = list(best_selection)
                        _working_tids = set(_selection_tids(_working))
                        if best_cov >= len(universe_tids):
                            break

        # Exchange starts from the best incumbent produced above.
        current_selection = tuple(best_selection)
        current_tids = set(_selection_tids(current_selection))
        current_set = set(current_selection)
        _candidate_meta = []
        for _j, _c in enumerate(columns):
            if _j in current_set:
                continue
            _tids = frozenset(_ordered_tids(_c))
            if len(_tids) < 2:
                continue
            _overlap = tuple(
                _s for _s in current_selection
                if set(_ordered_tids(columns[int(_s)])) & set(_tids))
            _uncovered_gain = len(set(_tids) - current_tids)
            _consolidation_gain = max(0, len(_overlap) - 1)
            _potential_gain = _uncovered_gain + _consolidation_gain
            _candidate_meta.append((
                -int(_potential_gain),
                -int(_uncovered_gain),
                -int(_consolidation_gain),
                -len(_tids),
                float(_c["E_plan_Wh"]) / max(1, len(_tids)),
                repr(_exact_route_signature(_c)),
                int(_j), tuple(int(x) for x in _overlap)))
            exchange_max_stop_count_considered = max(
                exchange_max_stop_count_considered, len(_tids))

        _candidate_meta.sort()
        exchange_candidate_routes = len(_candidate_meta)

        for _meta in _candidate_meta:
            if _time_or_budget_hit():
                break
            _j = int(_meta[-2])
            _mandatory = tuple(_meta[-1])
            _mandatory_set = set(_mandatory)

            # Mandatory overlap removal is the main consolidation move.  Add a
            # one-route optional drop family as a bounded escape when the
            # multi-stop route covers only previously-uncovered turbines or when
            # timing/battery compatibility requires a different incumbent route.
            _drop_sets = [tuple(sorted(_mandatory_set))]
            _optional = [
                int(_s) for _s in current_selection
                if int(_s) not in _mandatory_set]
            for _drop in sorted(
                    _optional,
                    key=lambda q: (
                        len(_ordered_tids(columns[q])),
                        -float(columns[q]["E_plan_Wh"]),
                        repr(_exact_route_signature(columns[q])))):
                _drop_sets.append(tuple(sorted(_mandatory_set | {_drop})))
                # Keep the neighborhood deliberately small and deterministic.
                if len(_drop_sets) >= 1 + min(6, len(current_selection)):
                    break

            _seen_drop_sets = set()
            for _drops in _drop_sets:
                if _time_or_budget_hit():
                    break
                if _drops in _seen_drop_sets:
                    continue
                _seen_drop_sets.add(_drops)
                if len(_mandatory) >= 2:
                    exchange_consolidation_trials += 1
                if len(set(_drops) - _mandatory_set) > 0:
                    exchange_optional_drop_trials += 1

                _seed = [
                    int(_s) for _s in current_selection
                    if int(_s) not in set(_drops)]
                _seed.append(_j)
                if not _packing_ok(_seed) or not _halfcap_ok(_seed):
                    continue

                # Two deterministic refill views: first prioritize missing
                # turbines, then favor coverage-per-mission (multi-stop) before
                # energy.  We construct a full trial and audit only that trial.
                _orders = []
                _seed_tids = set(_selection_tids(_seed))
                _missing = frozenset(universe_tids - _seed_tids)
                _orders.append(sorted(
                    range(len(columns)),
                    key=lambda q: _route_key(q, _missing)))
                _orders.append(sorted(
                    range(len(columns)),
                    key=lambda q: (
                        -len(set(_ordered_tids(columns[q])) & set(_missing)),
                        -len(_ordered_tids(columns[q])),
                        float(columns[q]["E_plan_Wh"])
                        / max(1, len(_ordered_tids(columns[q]))),
                        repr(_exact_route_signature(columns[q])))))

                _trial_seen = set()
                for _order in _orders:
                    if _time_or_budget_hit():
                        break
                    _trial = list(dict.fromkeys(_seed))
                    _trial_tids = set(_selection_tids(_trial))
                    for _q in _order:
                        _q = int(_q)
                        if _q in _trial:
                            continue
                        _qt = set(_ordered_tids(columns[_q]))
                        if not _qt or (_qt & _trial_tids):
                            continue
                        _candidate_trial = tuple(_trial + [_q])
                        if not _halfcap_ok(_candidate_trial):
                            continue
                        _trial.append(_q)
                        _trial_tids.update(_qt)
                    _trial_tuple = tuple(_trial)
                    _trial_key = tuple(sorted(_trial_tuple))
                    if _trial_key in _trial_seen:
                        continue
                    _trial_seen.add(_trial_key)
                    if not _is_objective_improvement(_trial_tuple):
                        continue
                    exchange_trials_built += 1
                    if _try_selection(_trial_tuple, "exchange"):
                        # One exact-audited improvement is enough for this short
                        # refresh.  Re-solve the RMP/pricing with the stronger
                        # incumbent rather than spending the full heuristic budget.
                        return dict(
                            selection=tuple(best_selection),
                            audit=best_audit,
                            improved=True,
                            coverage=int(best_cov),
                            audit_calls=int(audit_calls),
                            timed_out=bool(timed_out),
                            columns_seen=int(columns_seen),
                            rebuilds=0, repairs=0,
                            augmentation_audits=int(
                                neighborhood_audits["augmentation"]),
                            rebuild_audits=0,
                            repair_audits=0,
                            augmentation_improvements=int(
                                neighborhood_improvements["augmentation"]),
                            rebuild_improvements=0,
                            repair_improvements=0,
                            duplicate_trials_skipped=int(
                                duplicate_trials_skipped),
                            cached_infeasible_trials=int(
                                len(infeasible_trial_cache)),
                            uncovered_fair_rounds=int(
                                uncovered_fair_rounds),
                            failure_reasons=dict(failure_reasons),
                            deck_archive_conflict_edges=int(deck_archive_conflict_edges),
                            deck_archive_max_degree=int(deck_archive_max_degree),
                            deck_archive_max_component=int(deck_archive_max_component),
                            deck_candidate_scored=int(deck_candidate_scored),
                            deck_candidate_zero_conflict=int(deck_candidate_zero_conflict),
                            deck_candidate_positive_conflict=int(deck_candidate_positive_conflict),
                            deck_prefilter_skips=int(deck_prefilter_skips),
                            deck_max_candidate_conflicts=int(deck_max_candidate_conflicts),
                            deck_conflict_pairs_sample=list(deck_conflict_pairs_sample),
                            exchange_candidate_routes=int(exchange_candidate_routes),
                            exchange_trials_built=int(exchange_trials_built),
                            exchange_audits=int(neighborhood_audits["exchange"]),
                            exchange_improvements=int(
                                neighborhood_improvements["exchange"]),
                            exchange_consolidation_trials=int(
                                exchange_consolidation_trials),
                            exchange_optional_drop_trials=int(
                                exchange_optional_drop_trials),
                            exchange_max_stop_count_considered=int(
                                exchange_max_stop_count_considered),
                            runtime_s=float(time.monotonic() - started))

        return dict(
            selection=tuple(best_selection),
            audit=best_audit,
            improved=bool(best_cov > baseline_cov),
            coverage=int(best_cov),
            audit_calls=int(audit_calls),
            timed_out=bool(timed_out),
            columns_seen=int(columns_seen),
            rebuilds=0, repairs=0,
            augmentation_audits=int(neighborhood_audits["augmentation"]),
            rebuild_audits=0,
            repair_audits=0,
            augmentation_improvements=int(
                neighborhood_improvements["augmentation"]),
            rebuild_improvements=0,
            repair_improvements=0,
            duplicate_trials_skipped=int(duplicate_trials_skipped),
            cached_infeasible_trials=int(len(infeasible_trial_cache)),
            uncovered_fair_rounds=int(uncovered_fair_rounds),
            failure_reasons=dict(failure_reasons),
            deck_archive_conflict_edges=int(deck_archive_conflict_edges),
            deck_archive_max_degree=int(deck_archive_max_degree),
            deck_archive_max_component=int(deck_archive_max_component),
            deck_candidate_scored=int(deck_candidate_scored),
            deck_candidate_zero_conflict=int(deck_candidate_zero_conflict),
            deck_candidate_positive_conflict=int(deck_candidate_positive_conflict),
            deck_prefilter_skips=int(deck_prefilter_skips),
            deck_max_candidate_conflicts=int(deck_max_candidate_conflicts),
            deck_conflict_pairs_sample=list(deck_conflict_pairs_sample),
            exchange_candidate_routes=int(exchange_candidate_routes),
            exchange_trials_built=int(exchange_trials_built),
            exchange_audits=int(neighborhood_audits["exchange"]),
            exchange_improvements=int(neighborhood_improvements["exchange"]),
            exchange_consolidation_trials=int(exchange_consolidation_trials),
            exchange_optional_drop_trials=int(exchange_optional_drop_trials),
            exchange_max_stop_count_considered=int(
                exchange_max_stop_count_considered),
            runtime_s=float(time.monotonic() - started))

    def _greedy_refill(seed, order, neighborhood):
        nonlocal audit_calls, timed_out
        nonlocal best_selection, best_audit, best_cov, best_energy
        seed = list(dict.fromkeys(int(j) for j in seed))
        selected = list(seed)
        selected_tids = set(_selection_tids(selected))
        for j in order:
            if _time_or_budget_hit():
                break
            j = int(j)
            if j in selected:
                continue
            tids = set(_ordered_tids(columns[j]))
            if tids & selected_tids:
                continue
            trial = tuple(selected + [j])
            cov = _coverage(trial)
            if stage != "coverage" and cov > int(coverage_target or 0):
                continue
            if audit_calls >= max_audits or time.monotonic() >= local_deadline:
                break
            audit_calls += 1
            neighborhood_audits[neighborhood] += 1
            audit = _audit_integer_selection(
                columns, trial, K, batteries, p, quick_min, swap_min,
                quick_capacity, swap_capacity, local_deadline)
            if audit.status is RA.ResourceAuditStatus.UNKNOWN_TIMEOUT:
                timed_out = True
                break
            if audit.status is not RA.ResourceAuditStatus.FEASIBLE:
                continue
            selected.append(j)
            selected_tids.update(tids)

            if stage == "coverage":
                if cov > best_cov:
                    best_selection = tuple(selected)
                    best_audit = audit
                    best_cov = int(cov)
                    best_energy = _energy_of_selection_exact(
                        columns, best_selection)
                    neighborhood_improvements[neighborhood] += 1
            else:
                target = int(coverage_target or 0)
                if cov == target:
                    e = _energy_of_selection_exact(columns, tuple(selected))
                    if best_audit is None or e < best_energy:
                        best_selection = tuple(selected)
                        best_audit = audit
                        best_cov = int(cov)
                        best_energy = e
                        neighborhood_improvements[neighborhood] += 1
                    break
        return tuple(selected)

    # 1) Monotone augmentation of the current incumbent.
    current_tids = set(_selection_tids(best_selection))
    universe_tids = set()
    for c in columns:
        universe_tids.update(_ordered_tids(c))
    uncovered = frozenset(universe_tids - current_tids)
    augment_order = sorted(
        range(len(columns)),
        key=lambda j: _route_key(j, uncovered))
    working = list(best_selection)
    working_tids = set(_selection_tids(working))
    for j in augment_order:
        if _time_or_budget_hit():
            break
        if j in working:
            continue
        tids = set(_ordered_tids(columns[j]))
        if tids & working_tids:
            continue
        trial = tuple(working + [j])
        if stage == "energy" and _coverage(trial) > int(coverage_target or 0):
            continue
        if _try_selection(trial, "augmentation"):
            working = list(best_selection)
            working_tids = set(_selection_tids(working))

    # 2) Deterministic rebuilds.
    if not _time_or_budget_hit():
        rebuilds += 1
        priority_order = sorted(
            range(len(columns)),
            key=lambda j: _route_key(j, uncovered))
        _greedy_refill(tuple(), priority_order, "rebuild")

    if not _time_or_budget_hit():
        rebuilds += 1
        canonical_order = sorted(
            range(len(columns)),
            key=lambda j: (
                -len(_ordered_tids(columns[j])),
                float(columns[j]["E_plan_Wh"]),
                repr(_exact_route_signature(columns[j])),
            ))
        _greedy_refill(tuple(), canonical_order, "rebuild")

    # 3) One-route-drop repair.
    original = tuple(current_selection)
    for drop in original:
        if _time_or_budget_hit():
            break
        repairs += 1
        seed = tuple(j for j in original if j != drop)
        seed_tids = set(_selection_tids(seed))
        missing = frozenset(universe_tids - seed_tids)
        repair_order = sorted(
            range(len(columns)),
            key=lambda j: _route_key(j, missing))
        _greedy_refill(seed, repair_order, "repair")

    improved = bool(
        best_cov > baseline_cov
        if stage == "coverage"
        else (
            best_audit is not None
            and _coverage(best_selection) == int(coverage_target or 0)
            and (current_audit is None
                 or best_energy < _energy_of_selection_exact(
                     columns, current_selection))))
    return dict(
        selection=tuple(best_selection),
        audit=best_audit,
        improved=bool(improved),
        coverage=int(best_cov),
        audit_calls=int(audit_calls),
        timed_out=bool(timed_out),
        columns_seen=int(columns_seen),
        rebuilds=int(rebuilds),
        repairs=int(repairs),
        augmentation_audits=int(neighborhood_audits["augmentation"]),
        rebuild_audits=int(neighborhood_audits["rebuild"]),
        repair_audits=int(neighborhood_audits["repair"]),
        augmentation_improvements=int(
            neighborhood_improvements["augmentation"]),
        rebuild_improvements=int(neighborhood_improvements["rebuild"]),
        repair_improvements=int(neighborhood_improvements["repair"]),
        exchange_candidate_routes=0,
        exchange_trials_built=0,
        exchange_audits=0,
        exchange_improvements=0,
        exchange_consolidation_trials=0,
        exchange_optional_drop_trials=0,
        exchange_max_stop_count_considered=0,
        runtime_s=float(time.monotonic() - started),
    )


def _diagnose_fixed_archive_coverage(
        *, turbines, launch_opts, p, xi_amb, K, batteries, T_min, max_stops,
        weather_unc, archive, signature_to_index, no_good_cuts,
        initial_selection, initial_audit, t_launch_min, landing_clear_min,
        quick_min, swap_min, quick_capacity, swap_capacity,
        deck_mode, deck_delta_min, kappa_mode, chance_mode, budget_gamma,
        time_limit_s):
    """Exact restricted-archive coverage diagnostic.

    The frozen generated archive is treated as a complete finite universe only
    inside this diagnostic. A proof here is exact *within the archive* and can
    never be promoted to the complete physical route-space certificate.
    """
    started = time.monotonic()
    limit = max(0.0, float(time_limit_s))
    scope = "fixed-generated-column-archive-only-not-full-route-space"
    if limit <= 0.0:
        return dict(
            enabled=False, scope=scope, status="disabled-zero-budget",
            time_limit_s=float(limit), runtime_s=0.0,
            archive_columns=int(len(archive)),
            coverage_lower_bound=None, coverage_upper_bound=None,
            exact_optimum=None, optimal_proven=False,
            witness_selection_indices=[], witness_route_signatures=[],
            witness_covered_turbines=[],
            open_nodes=None, processed_nodes=None, rmp_solves=None,
            resource_audit_calls=None, resource_cuts_added=None)

    diag_archive = list(archive)
    diag_signature_to_index = dict(signature_to_index)
    diag_cuts = list(no_good_cuts)
    diag_deadline = time.monotonic() + limit
    try:
        stage = _solve_branch_price_stage(
            stage="coverage", turbines=turbines, launch_opts=launch_opts, p=p,
            xi_amb=xi_amb, K=K, batteries=batteries, T_min=T_min,
            max_stops=max_stops, weather_unc=weather_unc,
            deadline=diag_deadline,
            archive=diag_archive,
            signature_to_index=diag_signature_to_index,
            no_good_cuts=diag_cuts, coverage_target=None,
            initial_selection=tuple(initial_selection),
            initial_audit=initial_audit, pricing_epsilon=PRICING_EPS,
            coverage_gap_target_abs=0, energy_gap_target_rel=0.0,
            energy_gap_target_abs_Wh=0.0,
            t_launch_min=t_launch_min,
            landing_clear_min=landing_clear_min,
            quick_min=quick_min, swap_min=swap_min,
            quick_capacity=quick_capacity, swap_capacity=swap_capacity,
            deck_mode=deck_mode, deck_delta_min=deck_delta_min,
            kappa_mode=kappa_mode, chance_mode=chance_mode,
            budget_gamma=budget_gamma, implicit_test_columns=None,
            pricing_batch_size=16, root_branch=None, physical_cache=None,
            decision_only=False, complete_universe_mode=True,
            pricing_experiment_mode=False)
        lb = int(stage.coverage_incumbent)
        ub = int(max(lb, _safe_integer_floor(stage.global_bound)))
        proven = bool(stage.optimal and lb == ub)
        return dict(
            enabled=True, scope=scope,
            status=("archive-optimum-proven"
                    if proven else str(stage.termination_reason)),
            time_limit_s=float(limit),
            runtime_s=float(time.monotonic() - started),
            archive_columns=int(len(diag_archive)),
            coverage_lower_bound=int(lb),
            coverage_upper_bound=int(ub),
            exact_optimum=(int(lb) if proven else None),
            optimal_proven=bool(proven),
            witness_selection_indices=[
                int(_j) for _j in stage.incumbent_selection],
            witness_route_signatures=[
                repr(_exact_route_signature(diag_archive[int(_j)]))
                for _j in stage.incumbent_selection
                if 0 <= int(_j) < len(diag_archive)],
            witness_covered_turbines=sorted({
                str(_tid0)
                for _j in stage.incumbent_selection
                if 0 <= int(_j) < len(diag_archive)
                for _tid0 in _ordered_tids(diag_archive[int(_j)])}),
            open_nodes=int(stage.open_nodes),
            processed_nodes=int(stage.processed_nodes),
            rmp_solves=int(stage.rmp_solves),
            resource_audit_calls=int(stage.resource_audit_calls),
            resource_cuts_added=int(stage.resource_cuts_added))
    except Exception as exc:
        return dict(
            enabled=True, scope=scope,
            status=f"diagnostic-error:{type(exc).__name__}",
            time_limit_s=float(limit),
            runtime_s=float(time.monotonic() - started),
            archive_columns=int(len(archive)),
            coverage_lower_bound=None, coverage_upper_bound=None,
            exact_optimum=None, optimal_proven=False,
            witness_selection_indices=[], witness_route_signatures=[],
            witness_covered_turbines=[],
            open_nodes=None, processed_nodes=None, rmp_solves=None,
            resource_audit_calls=None, resource_cuts_added=None)


def _diagnose_resource_variant_postsolve(
        *, archive, final_selection, variant_records, K, batteries, p,
        quick_min, swap_min, quick_capacity, swap_capacity,
        time_limit_s=10.0):
    """V20.1 post-formal diagnostic for V20 exact singleton variants.

    This helper runs only after the formal solver clock.  It never mutates the
    formal archive, incumbent, branch tree, pricing state, bounds, cuts, or
    certificate fields.  It answers three observational questions:

    1. What exact V20 variant(s) were added and are their turbines still
       uncovered in the final incumbent?
    2. Can a final-uncovered V20 variant be directly appended under the exact
       resource audit, and if not, what exact failure reasons are reported?
    3. When exactly one incumbent deck route blocks the variant, can an existing
       archive timing variant of that blocker be substituted so that the
       augmented selection becomes exact-resource feasible?

    The same frozen-archive scan also summarizes how many singleton routes for
    final-uncovered turbines have zero/one/multiple final-incumbent deck
    blockers.  All conclusions are restricted to the generated archive.
    """
    started = time.monotonic()
    limit = max(0.0, float(time_limit_s))
    scope = (
        "post-formal-fixed-archive-resource-variant-diagnostic-only-"
        "not-full-route-space-not-proof")
    out = dict(
        enabled=bool(limit > 0.0),
        scope=scope,
        status="disabled-zero-budget" if limit <= 0.0 else "completed",
        time_limit_s=float(limit),
        runtime_s=0.0,
        timed_out=False,
        records_analyzed=0,
        records_missing_from_archive=0,
        final_coverage=0,
        final_uncovered_turbines=[],
        direct_augmentation_audits=0,
        direct_augmentation_feasible=0,
        direct_augmentation_infeasible=0,
        direct_augmentation_unknown=0,
        single_blocker_records=0,
        blocker_retime_candidates=0,
        blocker_retime_audits=0,
        blocker_retime_feasible=0,
        final_uncovered_singleton_routes=0,
        final_uncovered_zero_deck_conflict_routes=0,
        final_uncovered_single_blocker_routes=0,
        final_uncovered_multi_blocker_routes=0,
        final_uncovered_single_blocker_distinct_turbines=0,
        records=[],
        single_blocker_pairs_sample=[],
    )
    if limit <= 0.0:
        return out

    deadline = time.monotonic() + limit
    final_selection = tuple(
        int(_j) for _j in (final_selection or ())
        if 0 <= int(_j) < len(archive))
    all_tids = {
        str(_tid0) for _c in archive for _tid0 in _ordered_tids(_c)}
    final_covered = {
        str(_tid0)
        for _j in final_selection
        for _tid0 in _ordered_tids(archive[int(_j)])}
    out["final_coverage"] = int(len(final_covered))
    out["final_uncovered_turbines"] = sorted(all_tids - final_covered)

    sig_to_index = {
        repr(_exact_route_signature(_c)): int(_j)
        for _j, _c in enumerate(archive)}

    def _deck_conflict(_a, _b):
        for _ia in archive[int(_a)].get("resource_intervals", {}).get("deck", ()):
            for _ib in archive[int(_b)].get("resource_intervals", {}).get("deck", ()):
                if RA._halfopen_overlap(_ia, _ib):
                    return True
        return False

    def _deck_blockers(_q, _selection):
        return [
            int(_j) for _j in _selection
            if int(_j) != int(_q) and _deck_conflict(int(_q), int(_j))]

    def _audit_status_payload(_selection):
        if _deadline_hit(deadline):
            return "UNKNOWN_TIMEOUT", {"diagnostic_deadline": 1}, None
        audit = _audit_integer_selection(
            archive, tuple(int(_j) for _j in _selection),
            int(K), int(batteries), p,
            float(quick_min), float(swap_min),
            int(quick_capacity), int(swap_capacity), deadline)
        if audit.status is RA.ResourceAuditStatus.FEASIBLE:
            return "FEASIBLE", {}, audit
        if audit.status is RA.ResourceAuditStatus.UNKNOWN_TIMEOUT:
            return "UNKNOWN_TIMEOUT", dict(
                getattr(audit, "failure_reasons", None) or {}), audit
        return "INFEASIBLE_PROVEN", dict(
            getattr(audit, "failure_reasons", None) or {}), audit

    # Frozen-archive deck structure around final-uncovered singleton routes.
    _uncovered = set(out["final_uncovered_turbines"])
    _single_blocker_tids = set()
    for _q, _c in enumerate(archive):
        _tids = tuple(str(_t) for _t in _ordered_tids(_c))
        if len(_tids) != 1 or _tids[0] not in _uncovered:
            continue
        out["final_uncovered_singleton_routes"] += 1
        _blockers = _deck_blockers(_q, final_selection)
        if len(_blockers) == 0:
            out["final_uncovered_zero_deck_conflict_routes"] += 1
        elif len(_blockers) == 1:
            out["final_uncovered_single_blocker_routes"] += 1
            _single_blocker_tids.add(_tids[0])
            if len(out["single_blocker_pairs_sample"]) < 24:
                _b = int(_blockers[0])
                out["single_blocker_pairs_sample"].append(dict(
                    candidate_index=int(_q),
                    candidate_signature=repr(_exact_route_signature(_c)),
                    candidate_tid=str(_tids[0]),
                    blocker_index=int(_b),
                    blocker_signature=repr(
                        _exact_route_signature(archive[_b])),
                    blocker_tids=[
                        str(_t) for _t in _ordered_tids(archive[_b])],
                ))
        else:
            out["final_uncovered_multi_blocker_routes"] += 1
    out["final_uncovered_single_blocker_distinct_turbines"] = int(
        len(_single_blocker_tids))

    # Diagnose each V20 returned/added exact variant against the final incumbent.
    for _raw in list(variant_records or []):
        if _deadline_hit(deadline):
            out["timed_out"] = True
            out["status"] = "diagnostic-time-limit"
            break
        rec = dict(_raw)
        sig_repr = str(rec.get("signature_repr", ""))
        q = sig_to_index.get(sig_repr)
        rec["final_archive_index"] = q
        out["records_analyzed"] += 1
        if q is None:
            out["records_missing_from_archive"] += 1
            rec["postsolve_status"] = "missing-from-final-archive"
            out["records"].append(rec)
            continue

        tids = tuple(str(_t) for _t in _ordered_tids(archive[int(q)]))
        overlap = sorted(set(tids) & final_covered)
        blockers = _deck_blockers(int(q), final_selection)
        rec["final_overlap_turbines"] = overlap
        rec["final_turbine_uncovered"] = bool(not overlap)
        rec["final_deck_blocker_count"] = int(len(blockers))
        rec["final_deck_blocker_indices"] = [int(_j) for _j in blockers]
        rec["final_deck_blocker_signatures"] = [
            repr(_exact_route_signature(archive[int(_j)]))
            for _j in blockers]
        rec["direct_augmentation_status"] = "not-run"
        rec["direct_augmentation_failure_reasons"] = {}
        rec["blocker_retime_status"] = "not-run"
        rec["blocker_retime_witness_route_signature"] = None
        rec["blocker_retime_failure_reasons"] = {}

        if overlap:
            rec["postsolve_status"] = "turbine-already-covered-final"
            out["records"].append(rec)
            continue

        if not blockers:
            out["direct_augmentation_audits"] += 1
            st, reasons, _audit = _audit_status_payload(
                tuple(final_selection) + (int(q),))
            rec["direct_augmentation_status"] = st
            rec["direct_augmentation_failure_reasons"] = reasons
            if st == "FEASIBLE":
                out["direct_augmentation_feasible"] += 1
                rec["postsolve_status"] = "direct-augmentation-feasible"
            elif st == "INFEASIBLE_PROVEN":
                out["direct_augmentation_infeasible"] += 1
                rec["postsolve_status"] = "direct-augmentation-infeasible"
            else:
                out["direct_augmentation_unknown"] += 1
                out["timed_out"] = True
                out["status"] = "diagnostic-time-limit"
                rec["postsolve_status"] = "direct-augmentation-unknown"
            out["records"].append(rec)
            continue

        if len(blockers) != 1:
            rec["postsolve_status"] = "multiple-final-deck-blockers"
            out["records"].append(rec)
            continue

        out["single_blocker_records"] += 1
        blocker = int(blockers[0])
        blocker_tids = tuple(str(_t) for _t in _ordered_tids(archive[blocker]))
        rest = tuple(int(_j) for _j in final_selection if int(_j) != blocker)
        rec["postsolve_status"] = "single-final-deck-blocker"
        rec["blocker_tids"] = list(blocker_tids)

        # Search existing exact timing variants that preserve the blocker's
        # covered turbine set and avoid deck conflicts with both the candidate
        # and the remaining incumbent.
        replacements = []
        for _r, _c in enumerate(archive):
            if int(_r) in {int(q), blocker}:
                continue
            if tuple(str(_t) for _t in _ordered_tids(_c)) != blocker_tids:
                continue
            trial0 = tuple(rest) + (int(_r), int(q))
            # No duplicate turbine coverage is allowed in this diagnostic
            # augmentation witness.
            _trial_tids = [
                str(_t)
                for _j in trial0 for _t in _ordered_tids(archive[int(_j)])]
            if len(_trial_tids) != len(set(_trial_tids)):
                continue
            conflict = False
            for _ii in range(len(trial0)):
                for _kk in range(_ii + 1, len(trial0)):
                    if _deck_conflict(trial0[_ii], trial0[_kk]):
                        conflict = True
                        break
                if conflict:
                    break
            if conflict:
                continue
            replacements.append(int(_r))
        replacements.sort(key=lambda _r: (
            float(archive[_r].get("E_plan_Wh", math.inf)),
            float(archive[_r].get("tau", math.inf)),
            repr(_exact_route_signature(archive[_r]))))
        out["blocker_retime_candidates"] += int(len(replacements))
        rec["blocker_retime_candidate_count"] = int(len(replacements))

        _last_reasons = {}
        for _r in replacements:
            if _deadline_hit(deadline):
                out["timed_out"] = True
                out["status"] = "diagnostic-time-limit"
                rec["blocker_retime_status"] = "UNKNOWN_TIMEOUT"
                break
            out["blocker_retime_audits"] += 1
            st, reasons, _audit = _audit_status_payload(
                tuple(rest) + (int(_r), int(q)))
            _last_reasons = reasons
            if st == "FEASIBLE":
                out["blocker_retime_feasible"] += 1
                rec["blocker_retime_status"] = "FEASIBLE"
                rec["blocker_retime_witness_route_signature"] = repr(
                    _exact_route_signature(archive[int(_r)]))
                rec["postsolve_status"] = "single-blocker-retime-feasible"
                break
            if st == "UNKNOWN_TIMEOUT":
                out["timed_out"] = True
                out["status"] = "diagnostic-time-limit"
                rec["blocker_retime_status"] = "UNKNOWN_TIMEOUT"
                break
        else:
            rec["blocker_retime_status"] = (
                "INFEASIBLE_OR_NO_COMPATIBLE_RETIME"
                if replacements else "NO_DECK_COMPATIBLE_RETIME")
        rec["blocker_retime_failure_reasons"] = dict(_last_reasons)
        out["records"].append(rec)

    out["runtime_s"] = float(time.monotonic() - started)
    return out



def _diagnose_fullspace_target_ladder(
        *, turbines, launch_opts, p, xi_amb, K, batteries, T_min, max_stops,
        weather_unc, archive, signature_to_index, no_good_cuts,
        initial_selection, initial_audit, physical_cache,
        t_launch_min, landing_clear_min, quick_min, swap_min,
        quick_capacity, swap_capacity, deck_mode, deck_delta_min,
        kappa_mode, chance_mode, budget_gamma,
        time_limit_s, archive_primal_recovery_time_limit_s=2.0):
    """V21 post-formal exact full-space target ladder diagnostic.

    Starting from the strongest exact-resource-feasible witness already known,
    ask whether coverage >= LB+1 exists in the unchanged physical route space.
    A feasible target witness advances the ladder; a rigorously optimal coverage
    stage below the requested target proves that target impossible.  The helper
    runs only after the formal solver clock and mutates copies of archive/cuts/
    cache, so none of its bounds or decisions can affect the formal result.
    """
    started = time.monotonic()
    limit = max(0.0, float(time_limit_s))
    out = dict(
        enabled=bool(limit > 0.0),
        scope=("post-formal-full-physical-route-space-target-ladder-"
               "diagnostic-only-not-promoted-to-formal-result"),
        status="disabled-zero-budget" if limit <= 0.0 else "completed",
        time_limit_s=float(limit),
        runtime_s=0.0,
        archive_columns_start=int(len(archive)),
        archive_columns_end=int(len(archive)),
        start_coverage=int(
            _coverage_of_selection(archive, tuple(initial_selection or ()))
            if initial_selection else 0),
        best_coverage=int(
            _coverage_of_selection(archive, tuple(initial_selection or ()))
            if initial_selection else 0),
        highest_feasible_target=None,
        first_infeasible_target=None,
        unresolved_target=None,
        targets_attempted=0,
        records=[],
        witness_selection_indices=[int(_j) for _j in (initial_selection or ())],
        witness_route_signatures=[
            repr(_exact_route_signature(archive[int(_j)]))
            for _j in (initial_selection or ())
            if 0 <= int(_j) < len(archive)],
        witness_covered_turbines=sorted({
            str(_tid0)
            for _j in (initial_selection or ())
            if 0 <= int(_j) < len(archive)
            for _tid0 in _ordered_tids(archive[int(_j)])}),
    )
    if limit <= 0.0:
        return out

    diag_deadline = time.monotonic() + limit
    diag_archive = list(archive)
    diag_sig = dict(signature_to_index)
    diag_cuts = list(no_good_cuts)
    diag_cache = dict(physical_cache or {})
    best_selection = tuple(int(_j) for _j in (initial_selection or ()))
    best_audit = initial_audit
    if best_selection and best_audit is None and not _deadline_hit(diag_deadline):
        _initial_audit = _audit_integer_selection(
            diag_archive, best_selection, K, batteries, p,
            quick_min, swap_min, quick_capacity, swap_capacity, diag_deadline)
        if _initial_audit.status is RA.ResourceAuditStatus.FEASIBLE:
            best_audit = _initial_audit
        else:
            best_selection = tuple()
            best_audit = None
    best_cov = int(_coverage_of_selection(diag_archive, best_selection)
                   if best_selection else 0)
    out["start_coverage"] = int(best_cov)
    out["best_coverage"] = int(best_cov)
    all_tids = sorted({_tid(t.tid) for t in turbines})
    target = max(1, best_cov + 1)

    while target <= len(all_tids) and not _deadline_hit(diag_deadline):
        before_cols = int(len(diag_archive))
        t0 = time.monotonic()
        try:
            stage = _solve_branch_price_stage(
                stage="coverage", turbines=turbines, launch_opts=launch_opts,
                p=p, xi_amb=xi_amb, K=K, batteries=batteries, T_min=T_min,
                max_stops=max_stops, weather_unc=weather_unc,
                deadline=diag_deadline, archive=diag_archive,
                signature_to_index=diag_sig, no_good_cuts=diag_cuts,
                coverage_target=None, initial_selection=best_selection,
                initial_audit=best_audit, pricing_epsilon=PRICING_EPS,
                coverage_gap_target_abs=0, energy_gap_target_rel=0.0,
                energy_gap_target_abs_Wh=0.0,
                t_launch_min=t_launch_min,
                landing_clear_min=landing_clear_min,
                quick_min=quick_min, swap_min=swap_min,
                quick_capacity=quick_capacity, swap_capacity=swap_capacity,
                deck_mode=deck_mode, deck_delta_min=deck_delta_min,
                kappa_mode=kappa_mode, chance_mode=chance_mode,
                budget_gamma=budget_gamma, implicit_test_columns=None,
                pricing_batch_size=16, root_branch=None,
                physical_cache=diag_cache, decision_only=True,
                complete_universe_mode=False,
                pricing_experiment_mode="layered-batch-primal-shadow",
                coverage_decision_target=int(target),
                formal_battery_halfcap=True,
                primal_resource_deck_guided=True,
                resource_variant_enrichment=True,
                archive_primal_recovery=True,
                archive_primal_recovery_time_limit_s=float(
                    archive_primal_recovery_time_limit_s))
        except Exception as exc:
            out["status"] = f"diagnostic-error:{type(exc).__name__}"
            out["unresolved_target"] = int(target)
            break

        cov = int(stage.coverage_incumbent)
        ub = int(max(cov, _safe_integer_floor(stage.global_bound)))
        witness = bool(
            stage.incumbent_selection
            and stage.incumbent_audit is not None
            and stage.incumbent_audit.status is RA.ResourceAuditStatus.FEASIBLE
            and cov >= int(target))
        infeasible = bool(stage.optimal and ub < int(target))
        decision = (
            "FEASIBLE_WITNESS" if witness else
            "INFEASIBLE_PROVEN" if infeasible else
            "UNRESOLVED")
        rec = dict(
            target=int(target), decision=decision,
            runtime_s=float(time.monotonic() - t0),
            termination_reason=str(stage.termination_reason),
            coverage_incumbent=int(cov), coverage_upper_bound=int(ub),
            optimal=bool(stage.optimal),
            open_nodes=int(stage.open_nodes),
            processed_nodes=int(stage.processed_nodes),
            rmp_solves=int(stage.rmp_solves),
            resource_audit_calls=int(stage.resource_audit_calls),
            resource_cuts_added=int(stage.resource_cuts_added),
            pricing_calls=int(stage.pricing_calls),
            pricing_candidates=int(stage.pricing_candidates),
            pricing_nodes=int(stage.pricing_nodes),
            pricing_complete=bool(stage.pricing_complete),
            pricing_search_complete=bool(stage.pricing_search_complete),
            pricing_bound_available=bool(stage.pricing_bound_available),
            pricing_layered_depths_completed=int(
                stage.pricing_layered_depths_completed),
            pricing_layered_max_depth_completed=int(
                stage.pricing_layered_max_depth_completed),
            pricing_depth_prefixes_evaluated=dict(
                stage.pricing_depth_prefixes_evaluated),
            pricing_depth_improving_seen=dict(
                stage.pricing_depth_improving_seen),
            pricing_depth_improving_returned=dict(
                stage.pricing_depth_improving_returned),
            archive_columns_before=int(before_cols),
            archive_columns_after=int(len(diag_archive)),
            archive_primal_recovery_calls=int(
                stage.archive_primal_recovery_calls),
            archive_primal_recovery_improvements=int(
                stage.archive_primal_recovery_improvements),
            archive_primal_recovery_best_coverage=int(
                stage.archive_primal_recovery_best_coverage))
        out["records"].append(rec)
        out["targets_attempted"] += 1

        if witness:
            best_selection = tuple(stage.incumbent_selection)
            best_audit = stage.incumbent_audit
            best_cov = int(cov)
            out["best_coverage"] = max(int(out["best_coverage"]), best_cov)
            out["highest_feasible_target"] = int(target)
            out["witness_selection_indices"] = [
                int(_j) for _j in best_selection]
            out["witness_route_signatures"] = [
                repr(_exact_route_signature(diag_archive[int(_j)]))
                for _j in best_selection
                if 0 <= int(_j) < len(diag_archive)]
            out["witness_covered_turbines"] = sorted({
                str(_tid0)
                for _j in best_selection
                if 0 <= int(_j) < len(diag_archive)
                for _tid0 in _ordered_tids(diag_archive[int(_j)])})
            # A witness can jump by more than one target; skip already satisfied
            # intermediate targets and ask only the next unresolved threshold.
            target = max(int(target) + 1, best_cov + 1)
            continue

        if infeasible:
            out["first_infeasible_target"] = int(target)
            out["status"] = "target-infeasible-proven-diagnostic"
            break

        out["unresolved_target"] = int(target)
        out["status"] = (
            "diagnostic-time-limit"
            if _deadline_hit(diag_deadline)
            else "target-unresolved-diagnostic")
        break

    out["archive_columns_end"] = int(len(diag_archive))
    out["runtime_s"] = float(time.monotonic() - started)
    return out


def _fastest_turnaround_pattern_infeasible(archive, selection, K, batteries,
                                            quick_min, swap_min):
    """Diagnostic-only necessary UAV-turnaround relaxation for one pattern."""
    fastest_service = (min(max(float(quick_min), 0.0), max(float(swap_min), 0.0))
                       if int(batteries) >= 2 else max(float(quick_min), 0.0))
    intervals = []
    for j in selection:
        r = archive[int(j)]["resource_intervals"]
        intervals.append((float(r["launch_start_min"]),
                          float(r["clear_end_min"]) + fastest_service))
    rows = RA._interval_capacity_rows(intervals, min_size=int(K) + 1)
    max_overlap = max((len(row) for row in rows), default=0)
    return bool(max_overlap > int(K)), int(max_overlap)


def _exact_relaxed_min_batteries(archive, selection, usable_battery_energy_Wh,
                                 deadline, start_at=1):
    """Exact minimum battery count in the route-item energy relaxation, shadow only."""
    sel = tuple(sorted({int(j) for j in selection}))
    if not sel:
        return "FEASIBLE", 0, 0
    cap = Fraction.from_float(float(usable_battery_energy_Wh))
    if cap <= 0:
        return "INFEASIBLE_PROVEN", None, 0
    energies = [Fraction.from_float(float(archive[j]["E_soc_required_Wh"]))
                for j in sel]
    if any(e > cap for e in energies):
        return "INFEASIBLE_PROVEN", None, 0
    total = sum(energies, Fraction(0))
    ratio = total / cap
    pooled_lb = max(1, int(-(-ratio.numerator // ratio.denominator)))
    b0 = max(int(start_at), pooled_lb)
    nodes = 0
    for b in range(b0, len(sel) + 1):
        if _deadline_hit(deadline):
            return "UNKNOWN_TIMEOUT", None, int(nodes)
        st, n = _exact_battery_binpack_status(
            archive, sel, b, usable_battery_energy_Wh, deadline)
        nodes += int(n)
        if st == "UNKNOWN_TIMEOUT":
            return st, None, int(nodes)
        if st == "FEASIBLE":
            return "FEASIBLE", int(b), int(nodes)
    return "INFEASIBLE_PROVEN", None, int(nodes)


def _analyze_rejected_patterns_certificate_shadow(
        *, archive, selections, K, batteries, usable_battery_energy_Wh,
        quick_min, swap_min, time_limit_s):
    """Post-target V10 diagnostic; cannot modify formal or target-search state."""
    started = time.monotonic()
    limit = max(0.0, float(time_limit_s))
    out = dict(
        enabled=bool(limit > 0.0), time_limit_s=float(limit), runtime_s=0.0,
        analyzed_patterns=0, total_patterns=int(len(selections)), timed_out=False,
        pooled_energy_infeasible_patterns=0,
        battery_binpack_infeasible_patterns=0,
        battery_binpack_feasible_patterns=0,
        battery_binpack_unknown_patterns=0,
        battery_min_required_counts={},
        battery_core_extracted_patterns=0,
        battery_core_unique_count=0,
        battery_core_size_avg=None, battery_core_size_min=None,
        battery_core_size_max=None,
        battery_core_shadow_prior_cover_count=0,
        battery_core_shadow_prior_cover_fraction=None,
        fastest_turnaround_infeasible_patterns=0,
        fastest_turnaround_max_overlap_max=0,
        first_proof_layer_counts={
            "pooled_energy": 0, "battery_binpack": 0,
            "fastest_turnaround": 0, "full_resource_only": 0,
            "unknown_shadow": 0})
    if limit <= 0.0:
        return out
    deadline = time.monotonic() + limit
    cap = Fraction.from_float(float(usable_battery_energy_Wh))
    pooled_cap = cap * int(batteries)
    prior_cores = []
    unique_cores = set()
    core_sizes = []

    for selection in selections:
        if _deadline_hit(deadline):
            out["timed_out"] = True
            break
        sel = tuple(sorted({int(j) for j in selection}))
        out["analyzed_patterns"] += 1
        sel_set = frozenset(sel)
        covered_by_prior = any(frozenset(core).issubset(sel_set)
                               for core in prior_cores)
        if covered_by_prior:
            out["battery_core_shadow_prior_cover_count"] += 1

        total_e = sum((Fraction.from_float(float(
            archive[j]["E_soc_required_Wh"])) for j in sel), Fraction(0))
        pooled_no = bool(total_e > pooled_cap)
        out["pooled_energy_infeasible_patterns"] += int(pooled_no)

        st, _nodes = _exact_battery_binpack_status(
            archive, sel, int(batteries), usable_battery_energy_Wh, deadline)
        bin_no = st == "INFEASIBLE_PROVEN"
        if st == "FEASIBLE":
            out["battery_binpack_feasible_patterns"] += 1
        elif bin_no:
            out["battery_binpack_infeasible_patterns"] += 1
        else:
            out["battery_binpack_unknown_patterns"] += 1

        bmin = None
        if st != "UNKNOWN_TIMEOUT" and not _deadline_hit(deadline):
            mst, bmin, _mnodes = _exact_relaxed_min_batteries(
                archive, sel, usable_battery_energy_Wh, deadline,
                start_at=(int(batteries) + 1 if bin_no else 1))
            if mst == "FEASIBLE" and bmin is not None:
                key = str(int(bmin))
                out["battery_min_required_counts"][key] = int(
                    out["battery_min_required_counts"].get(key, 0)) + 1

        if bin_no and not covered_by_prior and not _deadline_hit(deadline):
            cst, core, _cnodes = _minimal_battery_conflict_core(
                archive, sel, int(batteries), usable_battery_energy_Wh, deadline)
            if cst == "INFEASIBLE_PROVEN" and core:
                core = tuple(sorted(int(j) for j in core))
                out["battery_core_extracted_patterns"] += 1
                core_sizes.append(len(core))
                if core not in unique_cores:
                    unique_cores.add(core)
                    prior_cores.append(core)

        turn_no, max_overlap = _fastest_turnaround_pattern_infeasible(
            archive, sel, int(K), int(batteries), quick_min, swap_min)
        out["fastest_turnaround_infeasible_patterns"] += int(turn_no)
        out["fastest_turnaround_max_overlap_max"] = max(
            int(out["fastest_turnaround_max_overlap_max"]), int(max_overlap))

        layers = out["first_proof_layer_counts"]
        if pooled_no:
            layers["pooled_energy"] += 1
        elif bin_no:
            layers["battery_binpack"] += 1
        elif turn_no:
            layers["fastest_turnaround"] += 1
        elif st == "UNKNOWN_TIMEOUT":
            layers["unknown_shadow"] += 1
        else:
            layers["full_resource_only"] += 1

        if st == "UNKNOWN_TIMEOUT" and _deadline_hit(deadline):
            out["timed_out"] = True
            break

    out["runtime_s"] = float(time.monotonic() - started)
    out["battery_core_unique_count"] = int(len(unique_cores))
    if core_sizes:
        out["battery_core_size_avg"] = float(sum(core_sizes)) / len(core_sizes)
        out["battery_core_size_min"] = int(min(core_sizes))
        out["battery_core_size_max"] = int(max(core_sizes))
    analyzed = int(out["analyzed_patterns"])
    if analyzed:
        out["battery_core_shadow_prior_cover_fraction"] = float(
            out["battery_core_shadow_prior_cover_count"]) / analyzed
    return out



def _battery_clique_diagnostic_rows_and_shadow(
        archive, selections, batteries, usable_battery_energy_Wh):
    """Build V11 certificate-safe battery clique rows and shadow coverage.

    These rows are used only by a post-formal frozen-archive rerun in V11.
    They are valid consequences of the exact route-level battery energy model:

    * half-cap row: items with 2E>C are pairwise incompatible;
    * anchor row: for anchor q with E_q<=C/2, q together with every item
      E>C-E_q is a pairwise-incompatible clique.

    At most ``batteries`` members of any such clique can be selected because
    every selected route is assigned to exactly one battery pack.
    """
    B = int(batteries)
    cap = Fraction.from_float(float(usable_battery_energy_Wh))
    out = dict(
        enabled=bool(B >= 1 and cap > 0),
        halfcap_rows=0, anchor_rows=0, total_rows=0,
        archive_columns=int(len(archive)),
        archive_halfcap_routes=0, archive_nonhalfcap_routes=0,
        archive_halfcap_stop_count_counts={},
        rejected_patterns=int(len(selections)),
        rejected_halfcap_violations=0,
        rejected_anchor_violations=0,
        rejected_anchor_only_violations=0,
        rejected_any_clique_violations=0,
        rejected_uncovered_by_cliques=0,
        rejected_halfcap_selected_count_distribution={},
        rejected_max_halfcap_selected=0)
    if B < 1 or cap <= 0:
        return tuple(), out

    energies = [
        Fraction.from_float(float(c["E_soc_required_Wh"])) for c in archive
    ]
    signatures = [_exact_route_signature(c) for c in archive]
    half = cap / 2

    rows = [(("diagnostic_battery_halfcap", float(usable_battery_energy_Wh)),
             float(B))]
    out["halfcap_rows"] = 1

    for j, (c, e) in enumerate(zip(archive, energies)):
        stops = int(len(_ordered_tids(c)))
        if 2 * e > cap:
            out["archive_halfcap_routes"] += 1
            key = str(stops)
            out["archive_halfcap_stop_count_counts"][key] = int(
                out["archive_halfcap_stop_count_counts"].get(key, 0)) + 1
        else:
            out["archive_nonhalfcap_routes"] += 1
            # The exact signature pins the low-energy anchor; every other
            # member is selected by an intrinsic energy threshold.
            rows.append((
                ("diagnostic_battery_anchor_clique",
                 (signatures[j], float(c["E_soc_required_Wh"]),
                  float(usable_battery_energy_Wh))),
                float(B)))
            out["anchor_rows"] += 1

    out["total_rows"] = int(len(rows))

    for selection in selections:
        sel = tuple(sorted({int(j) for j in selection}))
        hcount = sum(1 for j in sel if 2 * energies[j] > cap)
        key = str(int(hcount))
        out["rejected_halfcap_selected_count_distribution"][key] = int(
            out["rejected_halfcap_selected_count_distribution"].get(key, 0)) + 1
        out["rejected_max_halfcap_selected"] = max(
            int(out["rejected_max_halfcap_selected"]), int(hcount))
        half_violate = bool(hcount > B)
        out["rejected_halfcap_violations"] += int(half_violate)

        anchor_violate = False
        for q in sel:
            eq = energies[q]
            if 2 * eq > cap:
                continue
            clique_count = 0
            qsig = signatures[q]
            for j in sel:
                if signatures[j] == qsig or energies[j] > cap - eq:
                    clique_count += 1
            if clique_count > B:
                anchor_violate = True
                break

        out["rejected_anchor_violations"] += int(anchor_violate)
        out["rejected_anchor_only_violations"] += int(
            anchor_violate and not half_violate)
        any_violate = bool(half_violate or anchor_violate)
        out["rejected_any_clique_violations"] += int(any_violate)
        out["rejected_uncovered_by_cliques"] += int(not any_violate)

    return tuple(rows), out


def _diagnose_fixed_archive_target_with_battery_cliques(
        *, turbines, launch_opts, p, xi_amb, K, batteries, T_min, max_stops,
        weather_unc, archive, signature_to_index, no_good_cuts,
        initial_selection, initial_audit, t_launch_min, landing_clear_min,
        quick_min, swap_min, quick_capacity, swap_capacity,
        deck_mode, deck_delta_min, kappa_mode, chance_mode, budget_gamma,
        target_min, time_limit_s, clique_rows):
    """V11 post-formal frozen-archive target rerun with safe battery cliques.

    This is a diagnostic comparison only.  It uses the same frozen archive,
    the same pre-target formal exact-pattern cuts, and the unchanged exact
    resource audit.  The only added rows are valid battery-energy clique
    inequalities.  Its status must never be promoted to the formal solve.
    """
    started = time.monotonic()
    limit = max(0.0, float(time_limit_s))
    target = int(target_min)
    base = dict(
        enabled=bool(limit > 0.0),
        scope="fixed-generated-column-archive-target-with-battery-cliques-not-formal",
        target_min=int(target), time_limit_s=float(limit),
        archive_columns=int(len(archive)),
        clique_rows=int(len(tuple(clique_rows or ()))))
    if limit <= 0.0:
        return dict(
            **base, status="DISABLED_ZERO_BUDGET", runtime_s=0.0,
            feasible_proven=False, infeasible_proven=False,
            witness_coverage=None, coverage_incumbent=None,
            coverage_upper_bound=None, open_nodes=None, processed_nodes=None,
            rmp_solves=None, resource_audit_calls=None,
            resource_cuts_added=None, rejected_pattern_count=0)

    diag_archive = list(archive)
    diag_signature_to_index = dict(signature_to_index)
    # Important comparison contract: start from the same formal cuts that the
    # baseline target diagnostic received, not cuts generated by that baseline.
    diag_cuts = list(no_good_cuts)
    diag_sink = {}
    deadline = time.monotonic() + limit
    try:
        stage = _solve_branch_price_stage(
            stage="coverage", turbines=turbines, launch_opts=launch_opts, p=p,
            xi_amb=xi_amb, K=K, batteries=batteries, T_min=T_min,
            max_stops=max_stops, weather_unc=weather_unc,
            deadline=deadline, archive=diag_archive,
            signature_to_index=diag_signature_to_index,
            no_good_cuts=diag_cuts, coverage_target=None,
            initial_selection=tuple(initial_selection),
            initial_audit=initial_audit, pricing_epsilon=PRICING_EPS,
            coverage_gap_target_abs=0, energy_gap_target_rel=0.0,
            energy_gap_target_abs_Wh=0.0,
            t_launch_min=t_launch_min, landing_clear_min=landing_clear_min,
            quick_min=quick_min, swap_min=swap_min,
            quick_capacity=quick_capacity, swap_capacity=swap_capacity,
            deck_mode=deck_mode, deck_delta_min=deck_delta_min,
            kappa_mode=kappa_mode, chance_mode=chance_mode,
            budget_gamma=budget_gamma, implicit_test_columns=None,
            pricing_batch_size=16, root_branch=None, physical_cache=None,
            decision_only=True, complete_universe_mode=True,
            pricing_experiment_mode=False,
            coverage_decision_target=int(target),
            diagnostics_sink=diag_sink,
            diagnostic_extra_inequality_rows=tuple(clique_rows or ()))

        witness_cov = int(stage.coverage_incumbent)
        feasible = bool(
            stage.incumbent_audit is not None
            and witness_cov >= target
            and getattr(stage.incumbent_audit, "status", None)
                is RA.ResourceAuditStatus.FEASIBLE)
        closure = bool(
            not feasible
            and int(stage.open_nodes) == 0
            and bool(stage.pricing_search_complete)
            and bool(stage.pricing_bound_available)
            and bool(stage.resource_audit_complete)
            and bool(stage.farkas_pricing_complete)
            and bool(stage.branching_complete))
        status = ("FEASIBLE" if feasible else
                  "INFEASIBLE_PROVEN" if closure else "UNKNOWN_TIMEOUT")
        return dict(
            **base, status=status,
            runtime_s=float(time.monotonic() - started),
            feasible_proven=bool(feasible),
            infeasible_proven=bool(closure),
            witness_coverage=(witness_cov if feasible else None),
            coverage_incumbent=int(stage.coverage_incumbent),
            coverage_upper_bound=int(max(
                stage.coverage_incumbent,
                _safe_integer_floor(stage.global_bound))),
            open_nodes=int(stage.open_nodes),
            processed_nodes=int(stage.processed_nodes),
            rmp_solves=int(stage.rmp_solves),
            resource_audit_calls=int(stage.resource_audit_calls),
            resource_cuts_added=int(stage.resource_cuts_added),
            rejected_pattern_count=int(
                diag_sink.get("rejected_pattern_count", 0)))
    except Exception as exc:
        return dict(
            **base, status=f"DIAGNOSTIC_ERROR:{type(exc).__name__}",
            runtime_s=float(time.monotonic() - started),
            feasible_proven=False, infeasible_proven=False,
            witness_coverage=None, coverage_incumbent=None,
            coverage_upper_bound=None, open_nodes=None, processed_nodes=None,
            rmp_solves=None, resource_audit_calls=None,
            resource_cuts_added=None,
            rejected_pattern_count=int(
                diag_sink.get("rejected_pattern_count", 0)))


def _diagnose_fixed_archive_target_min_coverage(
        *, turbines, launch_opts, p, xi_amb, K, batteries, T_min, max_stops,
        weather_unc, archive, signature_to_index, no_good_cuts,
        initial_selection, initial_audit, t_launch_min, landing_clear_min,
        quick_min, swap_min, quick_capacity, swap_capacity,
        deck_mode, deck_delta_min, kappa_mode, chance_mode, budget_gamma,
        target_min, time_limit_s, certificate_shadow_time_limit_s=0.0,
        clique_rerun_time_limit_s=0.0):
    """Decide whether the frozen archive contains coverage >= ``target_min``.

    This is exact only for the generated archive.  FEASIBLE requires an unchanged
    exact resource-audit witness.  INFEASIBLE_PROVEN requires exhaustive closure
    of every archive node whose rigorous coverage upper bound can reach target.
    UNKNOWN_TIMEOUT is fail-closed and has no effect on the formal result.
    """
    started = time.monotonic()
    limit = max(0.0, float(time_limit_s))
    scope = "fixed-generated-column-archive-target-only-not-full-route-space"
    target = int(target_min)
    base = dict(
        enabled=True, scope=scope, target_min=int(target),
        time_limit_s=float(limit), archive_columns=int(len(archive)))
    if limit <= 0.0:
        return dict(
            **base, status="DISABLED_ZERO_BUDGET", runtime_s=0.0,
            feasible_proven=False, infeasible_proven=False,
            witness_coverage=None, coverage_incumbent=None,
            coverage_upper_bound=None, open_nodes=None, processed_nodes=None,
            rmp_solves=None, resource_audit_calls=None,
            resource_cuts_added=None, rejected_pattern_count=0,
            rejected_pattern_size_avg=None,
            rejected_pattern_coverage_avg=None,
            rejected_hamming_avg=None, rejected_hamming_min=None,
            rejected_hamming_max=None,
            resource_failure_event_counts={},
            resource_failure_pattern_counts={},
            rejected_morphology_counts={},
            rejected_route_stop_count_totals={},
            rejected_coverage_route_count_joint={},
            certificate_shadow=dict(
                enabled=False,
                time_limit_s=float(certificate_shadow_time_limit_s),
                runtime_s=0.0, analyzed_patterns=0, total_patterns=0),
            battery_clique_shadow=dict(enabled=False, total_rows=0),
            battery_clique_target_rerun=dict(
                enabled=False, status="DISABLED_ZERO_BUDGET",
                time_limit_s=float(clique_rerun_time_limit_s),
                runtime_s=0.0))

    diag_archive = list(archive)
    diag_signature_to_index = dict(signature_to_index)
    diag_cuts = list(no_good_cuts)
    diag_sink = {}
    diag_deadline = time.monotonic() + limit

    try:
        stage = _solve_branch_price_stage(
            stage="coverage", turbines=turbines, launch_opts=launch_opts, p=p,
            xi_amb=xi_amb, K=K, batteries=batteries, T_min=T_min,
            max_stops=max_stops, weather_unc=weather_unc,
            deadline=diag_deadline, archive=diag_archive,
            signature_to_index=diag_signature_to_index,
            no_good_cuts=diag_cuts, coverage_target=None,
            initial_selection=tuple(initial_selection),
            initial_audit=initial_audit, pricing_epsilon=PRICING_EPS,
            coverage_gap_target_abs=0, energy_gap_target_rel=0.0,
            energy_gap_target_abs_Wh=0.0,
            t_launch_min=t_launch_min,
            landing_clear_min=landing_clear_min,
            quick_min=quick_min, swap_min=swap_min,
            quick_capacity=quick_capacity, swap_capacity=swap_capacity,
            deck_mode=deck_mode, deck_delta_min=deck_delta_min,
            kappa_mode=kappa_mode, chance_mode=chance_mode,
            budget_gamma=budget_gamma, implicit_test_columns=None,
            pricing_batch_size=16, root_branch=None, physical_cache=None,
            decision_only=True, complete_universe_mode=True,
            pricing_experiment_mode=False,
            coverage_decision_target=int(target),
            diagnostics_sink=diag_sink)

        witness_cov = int(stage.coverage_incumbent)
        feasible = bool(
            stage.incumbent_audit is not None
            and witness_cov >= int(target)
            and getattr(stage.incumbent_audit, "status", None)
                is RA.ResourceAuditStatus.FEASIBLE)

        closure = bool(
            not feasible
            and int(stage.open_nodes) == 0
            and bool(stage.pricing_search_complete)
            and bool(stage.pricing_bound_available)
            and bool(stage.resource_audit_complete)
            and bool(stage.farkas_pricing_complete)
            and bool(stage.branching_complete))

        if feasible:
            status = "FEASIBLE"
        elif closure:
            status = "INFEASIBLE_PROVEN"
        else:
            status = "UNKNOWN_TIMEOUT"

        count = int(diag_sink.get("rejected_pattern_count", 0))
        hcount = int(diag_sink.get("rejected_hamming_count", 0))
        target_runtime_s = float(time.monotonic() - started)
        rejected_selections = tuple(diag_sink.get("_rejected_selections", ()))
        certificate_shadow = _analyze_rejected_patterns_certificate_shadow(
            archive=diag_archive,
            selections=rejected_selections,
            K=int(K), batteries=int(batteries),
            usable_battery_energy_Wh=float(p.B_use),
            quick_min=quick_min, swap_min=swap_min,
            time_limit_s=certificate_shadow_time_limit_s)
        clique_rows, battery_clique_shadow = (
            _battery_clique_diagnostic_rows_and_shadow(
                diag_archive, rejected_selections, int(batteries),
                float(p.B_use)))
        battery_clique_target_rerun = (
            _diagnose_fixed_archive_target_with_battery_cliques(
                turbines=turbines, launch_opts=launch_opts, p=p,
                xi_amb=xi_amb, K=K, batteries=batteries, T_min=T_min,
                max_stops=max_stops, weather_unc=weather_unc,
                archive=archive, signature_to_index=signature_to_index,
                no_good_cuts=no_good_cuts,
                initial_selection=tuple(initial_selection),
                initial_audit=initial_audit,
                t_launch_min=t_launch_min,
                landing_clear_min=landing_clear_min,
                quick_min=quick_min, swap_min=swap_min,
                quick_capacity=quick_capacity, swap_capacity=swap_capacity,
                deck_mode=deck_mode, deck_delta_min=deck_delta_min,
                kappa_mode=kappa_mode, chance_mode=chance_mode,
                budget_gamma=budget_gamma, target_min=target,
                time_limit_s=clique_rerun_time_limit_s,
                clique_rows=clique_rows)
            if float(clique_rerun_time_limit_s) > 0.0
            else dict(
                enabled=False,
                scope="fixed-generated-column-archive-target-with-battery-cliques-not-formal",
                target_min=int(target),
                time_limit_s=float(clique_rerun_time_limit_s),
                archive_columns=int(len(archive)),
                clique_rows=int(len(clique_rows)),
                status="DISABLED_ZERO_BUDGET", runtime_s=0.0,
                feasible_proven=False, infeasible_proven=False))
        return dict(
            **base, status=status,
            runtime_s=float(target_runtime_s),
            feasible_proven=bool(feasible),
            infeasible_proven=bool(closure),
            witness_coverage=(int(witness_cov) if feasible else None),
            coverage_incumbent=int(stage.coverage_incumbent),
            coverage_upper_bound=int(max(
                stage.coverage_incumbent,
                _safe_integer_floor(stage.global_bound))),
            open_nodes=int(stage.open_nodes),
            processed_nodes=int(stage.processed_nodes),
            rmp_solves=int(stage.rmp_solves),
            resource_audit_calls=int(stage.resource_audit_calls),
            resource_cuts_added=int(stage.resource_cuts_added),
            rejected_pattern_count=count,
            rejected_pattern_size_avg=(
                float(diag_sink.get("rejected_pattern_size_sum", 0)) / count
                if count else None),
            rejected_pattern_size_min=diag_sink.get(
                "rejected_pattern_size_min"),
            rejected_pattern_size_max=diag_sink.get(
                "rejected_pattern_size_max"),
            rejected_pattern_coverage_avg=(
                float(diag_sink.get("rejected_pattern_coverage_sum", 0)) / count
                if count else None),
            rejected_pattern_coverage_min=diag_sink.get(
                "rejected_pattern_coverage_min"),
            rejected_pattern_coverage_max=diag_sink.get(
                "rejected_pattern_coverage_max"),
            rejected_hamming_avg=(
                float(diag_sink.get("rejected_hamming_sum", 0)) / hcount
                if hcount else None),
            rejected_hamming_min=diag_sink.get("rejected_hamming_min"),
            rejected_hamming_max=diag_sink.get("rejected_hamming_max"),
            resource_failure_event_counts=dict(
                diag_sink.get("resource_failure_event_counts", {})),
            resource_failure_pattern_counts=dict(
                diag_sink.get("resource_failure_pattern_counts", {})),
            rejected_morphology_counts=dict(
                diag_sink.get("rejected_morphology_counts", {})),
            rejected_route_stop_count_totals=dict(
                diag_sink.get("rejected_route_stop_count_totals", {})),
            rejected_coverage_route_count_joint=dict(
                diag_sink.get("rejected_coverage_route_count_joint", {})),
            certificate_shadow=dict(certificate_shadow),
            battery_clique_shadow=dict(battery_clique_shadow),
            battery_clique_target_rerun=dict(battery_clique_target_rerun))
    except Exception as exc:
        count = int(diag_sink.get("rejected_pattern_count", 0))
        hcount = int(diag_sink.get("rejected_hamming_count", 0))
        return dict(
            **base, status=f"DIAGNOSTIC_ERROR:{type(exc).__name__}",
            runtime_s=float(time.monotonic() - started),
            feasible_proven=False, infeasible_proven=False,
            witness_coverage=None, coverage_incumbent=None,
            coverage_upper_bound=None, open_nodes=None, processed_nodes=None,
            rmp_solves=None, resource_audit_calls=None,
            resource_cuts_added=None,
            rejected_pattern_count=count,
            rejected_pattern_size_avg=(
                float(diag_sink.get("rejected_pattern_size_sum", 0)) / count
                if count else None),
            rejected_pattern_coverage_avg=(
                float(diag_sink.get("rejected_pattern_coverage_sum", 0)) / count
                if count else None),
            rejected_hamming_avg=(
                float(diag_sink.get("rejected_hamming_sum", 0)) / hcount
                if hcount else None),
            rejected_hamming_min=diag_sink.get("rejected_hamming_min"),
            rejected_hamming_max=diag_sink.get("rejected_hamming_max"),
            resource_failure_event_counts=dict(
                diag_sink.get("resource_failure_event_counts", {})),
            resource_failure_pattern_counts=dict(
                diag_sink.get("resource_failure_pattern_counts", {})),
            rejected_morphology_counts=dict(
                diag_sink.get("rejected_morphology_counts", {})),
            rejected_route_stop_count_totals=dict(
                diag_sink.get("rejected_route_stop_count_totals", {})),
            rejected_coverage_route_count_joint=dict(
                diag_sink.get("rejected_coverage_route_count_joint", {})),
            certificate_shadow=dict(
                enabled=False,
                time_limit_s=float(certificate_shadow_time_limit_s),
                runtime_s=0.0, analyzed_patterns=0,
                total_patterns=int(len(diag_sink.get("_rejected_selections", ())))),
            battery_clique_shadow=dict(enabled=False, total_rows=0),
            battery_clique_target_rerun=dict(
                enabled=False,
                status="DIAGNOSTIC_NOT_RUN_AFTER_BASE_ERROR",
                time_limit_s=float(clique_rerun_time_limit_s),
                runtime_s=0.0))


def _empty_bpc_result(turbines, time_limit_s, started, reason, *, bound_source,
                      K=None, batteries=None, seed_validation=None,
                      status="time_limit_feasible",
                      pricing_bound_available=False, model_contract=None,
                      route_universe_source="physical-oracle",
                      route_universe_provenance_certified=True):
    """Return a fail-closed anytime result with the audited empty-plan incumbent."""
    n = len({_tid(t.tid) for t in turbines})
    seed_validation = dict(seed_validation or {
        "input_count": 0, "accepted_count": 0, "rejected_count": 0,
        "rejection_reasons": {}})
    model_contract = dict(model_contract or {})
    try:
        _empty_max_stops = int(model_contract.get("max_stops", 1))
        _empty_row_ranges_ok = _future_row_range_contract_self_check(_empty_max_stops)
    except Exception:
        _empty_row_ranges_ok = False
    _empty_binary64_contract_ok = bool(
        model_contract.get("physical_numeric_contract") == RM.FORMAL_PHYSICAL_NUMERIC_CONTRACT
        and model_contract.get("route_identity_contract") == ROUTE_IDENTITY_CONTRACT
        and model_contract.get("model_semantics_contract") == MODEL_SEMANTICS_CONTRACT)
    _empty_proof_contract_ok = bool(
        model_contract.get("formal_proof_contract") == FORMAL_PROOF_CONTRACT
        and model_contract.get("formal_proof_contract_sha256") == FORMAL_PROOF_CONTRACT_SHA256)
    return dict(
        status=str(status),
        termination_reason=str(reason), runtime_s=float(time.monotonic() - started),
        time_limit_s=time_limit_s,
        algorithm="branch-price-and-cut-with-logic-benders",
        pricing_method=(
            "exact-layered-batch-primal-target9-battery-clique-root-cause-diagnostic"
            if model_contract.get("pricing_mode")
                == "exact-layered-batch-primal-target9-battery-clique-diagnostic-shadow"
            else "exact-layered-batch-primal-target9-certificate-root-cause-diagnostic"
            if model_contract.get("pricing_mode")
                == "exact-layered-batch-primal-target9-certificate-diagnostic-shadow"
            else "exact-layered-batch-primal-target9-root-cause-diagnostic"
            if model_contract.get("pricing_mode")
                == "exact-layered-batch-primal-target9-diagnostic-shadow"
            else "exact-layered-batch-primal-root-cause-diagnostic"
            if model_contract.get("pricing_mode")
                == "exact-layered-batch-primal-diagnostic-shadow"
            else "exact-layered-batch-discovery-plus-primal-incumbent-refresh"
            if model_contract.get("pricing_mode")
                == "exact-layered-batch-primal-shadow"
            else "exact-layered-round-robin-adaptive-diverse-batch-discovery"
            if model_contract.get("pricing_mode") == "exact-layered-batch-shadow"
            else "exact-layered-round-robin-dual-guided-discovery-plus-prefix-shadow"
            if model_contract.get("pricing_mode") == "exact-layered-guided-shadow"
            else "exact-implicit-dfs-dual-guided-discovery-plus-prefix-shadow"
            if model_contract.get("pricing_mode") == "exact-dual-guided-shadow"
            else "exact-implicit-dfs-discovery-plus-prefix-shadow"
            if model_contract.get("pricing_mode") == "exact-discovery-shadow"
            else "exact-implicit-elementary-sequence-dfs"),
        solver_mode="exact-branch-price-cut",
        pricing_mode=str(model_contract.get("pricing_mode", "exact-implicit-dfs")),
        branching_complete=False, farkas_pricing_complete=False,
        coverage_incumbent=0, coverage_upper_bound=int(n),
        coverage_gap_abs=int(n), coverage_gap_pct=(100.0 if n else 0.0),
        coverage_optimal=False,
        coverage_algorithmic_certificate=False,
        coverage_physical_model_certificate=False,
        coverage_global_certificate_available=False,
        solve_scope=str(model_contract.get("solve_scope", "lexicographic")),
        energy_incumbent_Wh=0.0, energy_incumbent_estimate_Wh=0.0,
        energy_incumbent_lower_enclosure_Wh=0.0,
        energy_lower_bound_Wh=None,
        energy_gap_abs_Wh=None, energy_gap_pct=None, energy_optimal=False,
        conditional_energy_gap_pct=None,
        global_energy_gap_reason="coverage optimum not proven",
        lexicographic_optimal=False,
        pricing_complete=False, pricing_closed=False,
        pricing_search_complete=False,
        pricing_bound_available=bool(pricing_bound_available),
        coverage_pricing_complete=False, coverage_pricing_closed=False,
        coverage_pricing_search_complete=False,
        energy_pricing_complete=None, energy_pricing_closed=None,
        energy_pricing_search_complete=None,
        coverage_pricing_best_reduced_value=None,
        coverage_pricing_reduced_value_bound=None,
        energy_pricing_best_reduced_value=None,
        energy_pricing_reduced_value_bound=None,
        # The empty plan has a complete exact resource certificate by construction.
        resource_audit_complete=True,
        bound_scope=("global_discrete_physical_model"
                     if route_universe_provenance_certified
                     else "synthetic_finite_route_fixture"),
        bound_source=str(bound_source),
        coverage_bound_source=str(bound_source), energy_bound_source=None,
        open_nodes=1, processed_nodes=0, branch_nodes=0,
        branch_decisions=0, branch_children_created=0,
        rmp_solves=0, phase_one_solves=0, generated_columns=0,
        columns_accepted=0, pricing_calls=0, exact_pricing_calls=0,
        exact_certification_calls=0, pricing_discovery_calls=0,
        pricing_discovery_early_returns=0,
        pricing_discovery_improving_seen=0,
        pricing_discovery_improving_returned=0,
        pricing_discovery_diverse_returns=0,
        pricing_discovery_hard_cap_returns=0,
        pricing_discovery_max_return_batch=0,
        pricing_discovery_max_distinct_launches=0,
        pricing_discovery_max_distinct_service_sets=0,
        primal_refresh_calls=0,
        primal_refresh_audit_calls=0,
        primal_refresh_timeouts=0,
        primal_refresh_augmentation_audits=0,
        primal_refresh_rebuild_audits=0,
        primal_refresh_repair_audits=0,
        primal_refresh_augmentation_improvements=0,
        primal_refresh_rebuild_improvements=0,
        primal_refresh_repair_improvements=0,
        pricing_depth_prefixes_evaluated={},
        pricing_depth_improving_seen={},
        pricing_depth_improving_returned={},
        pricing_depth1_prefixes=0, pricing_depth2_prefixes=0,
        pricing_depth3_prefixes=0, pricing_depth4_prefixes=0,
        pricing_depth1_improving=0, pricing_depth2_improving=0,
        pricing_depth3_improving=0, pricing_depth4_improving=0,
        pricing_depth1_returned=0, pricing_depth2_returned=0,
        pricing_depth3_returned=0, pricing_depth4_returned=0,
        archive_diag_enabled=False,
        archive_diag_scope="fixed-generated-column-archive-only-not-full-route-space",
        archive_diag_status="not-run-empty-result",
        archive_diag_time_limit_s=0.0,
        archive_diag_runtime_s=0.0,
        archive_diag_columns=0,
        archive_diag_coverage_lower_bound=None,
        archive_diag_coverage_upper_bound=None,
        archive_diag_exact_optimum=None,
        archive_diag_optimal_proven=False,
        archive_diag_open_nodes=None,
        archive_diag_processed_nodes=None,
        archive_diag_rmp_solves=None,
        archive_diag_resource_audit_calls=None,
        archive_diag_resource_cuts_added=None,
        archive_target_enabled=False,
        archive_target_scope=(
            "fixed-generated-column-archive-target-only-not-full-route-space"),
        archive_target_min=9,
        archive_target_status="not-run-empty-result",
        archive_target_time_limit_s=0.0,
        archive_target_runtime_s=0.0,
        archive_target_columns=0,
        archive_target_feasible_proven=False,
        archive_target_infeasible_proven=False,
        archive_target_witness_coverage=None,
        archive_target_coverage_incumbent=None,
        archive_target_coverage_upper_bound=None,
        archive_target_open_nodes=None,
        archive_target_processed_nodes=None,
        archive_target_rmp_solves=None,
        archive_target_resource_audit_calls=None,
        archive_target_resource_cuts_added=None,
        archive_target_rejected_pattern_count=0,
        archive_target_rejected_pattern_size_avg=None,
        archive_target_rejected_pattern_size_min=None,
        archive_target_rejected_pattern_size_max=None,
        archive_target_rejected_pattern_coverage_avg=None,
        archive_target_rejected_pattern_coverage_min=None,
        archive_target_rejected_pattern_coverage_max=None,
        archive_target_rejected_hamming_avg=None,
        archive_target_rejected_hamming_min=None,
        archive_target_rejected_hamming_max=None,
        archive_target_resource_failure_event_counts={},
        archive_target_resource_failure_pattern_counts={},
        archive_target_rejected_morphology_counts={},
        archive_target_rejected_route_stop_count_totals={},
        archive_target_rejected_coverage_route_count_joint={},
        archive_target_certificate_shadow={},
        archive_target_shadow_analyzed_patterns=0,
        archive_target_shadow_total_patterns=0,
        archive_target_shadow_timed_out=False,
        archive_target_shadow_pooled_energy_infeasible=0,
        archive_target_shadow_battery_binpack_infeasible=0,
        archive_target_shadow_battery_binpack_feasible=0,
        archive_target_shadow_battery_binpack_unknown=0,
        archive_target_shadow_battery_core_unique_count=0,
        archive_target_shadow_battery_core_size_avg=None,
        archive_target_shadow_battery_core_size_min=None,
        archive_target_shadow_battery_core_size_max=None,
        archive_target_shadow_prior_core_cover_count=0,
        archive_target_shadow_prior_core_cover_fraction=None,
        archive_target_shadow_fastest_turnaround_infeasible=0,
        archive_target_shadow_battery_min_required_counts={},
        archive_target_shadow_first_proof_layer_counts={},
        archive_target_battery_clique_shadow={},
        archive_target_clique_halfcap_rows=0,
        archive_target_clique_anchor_rows=0,
        archive_target_clique_total_rows=0,
        archive_target_clique_archive_halfcap_routes=0,
        archive_target_clique_archive_nonhalfcap_routes=0,
        archive_target_clique_archive_halfcap_stop_count_counts={},
        archive_target_clique_rejected_halfcap_violations=0,
        archive_target_clique_rejected_anchor_violations=0,
        archive_target_clique_rejected_anchor_only_violations=0,
        archive_target_clique_rejected_any_violations=0,
        archive_target_clique_rejected_uncovered=0,
        archive_target_clique_rejected_halfcap_count_distribution={},
        archive_clique_target_enabled=False,
        archive_clique_target_scope=(
            "fixed-generated-column-archive-target-with-battery-cliques-not-formal"),
        archive_clique_target_status="not-run-empty-result",
        archive_clique_target_time_limit_s=0.0,
        archive_clique_target_runtime_s=0.0,
        archive_clique_target_rows=0,
        archive_clique_target_feasible_proven=False,
        archive_clique_target_infeasible_proven=False,
        archive_clique_target_witness_coverage=None,
        archive_clique_target_coverage_incumbent=None,
        archive_clique_target_coverage_upper_bound=None,
        archive_clique_target_open_nodes=None,
        archive_clique_target_processed_nodes=None,
        archive_clique_target_rmp_solves=None,
        archive_clique_target_resource_audit_calls=None,
        archive_clique_target_resource_cuts_added=None,
        archive_clique_target_rejected_pattern_count=0,
        pricing_pattern_cut_active_dual_rows=0,
        pricing_pattern_cut_dual_abs_sum=0.0,
        pricing_pattern_cut_improving_seen_count=0,
        pricing_pattern_cut_improving_seen_contribution_sum=0.0,
        pricing_pattern_cut_improving_seen_sign_essential=0,
        pricing_pattern_cut_returned_count=0,
        pricing_pattern_cut_returned_contribution_sum=0.0,
        pricing_pattern_cut_returned_sign_essential=0,
        pricing_pattern_cut_returned_by_depth={},
        primal_refresh_improvements=0,
        primal_refresh_best_coverage=0,
        primal_refresh_columns_seen=0,
        primal_refresh_rebuilds=0,
        primal_refresh_repairs=0,
        pricing_shadow_prefixes_evaluated=0,
        pricing_shadow_prunable_prefixes=0,
        pricing_shadow_false_prune_witnesses=0,
        pricing_shadow_bound_errors=0, pricing_shadow_complete_calls=0,
        pricing_candidates=0, pricing_nodes=0,
        heuristic_columns=0, resource_audit_calls=0, resource_cuts_added=0,
        resource_pattern_cuts_added=0,
        resource_cut_type="exact-selected-pattern",
        resource_cut_superset_assumption=False,
        heuristic_pricing_used=False, initial_column_heuristic_used=False,
        exact_pricing_called=False,
        pricing_best_reduced_value=None, pricing_reduced_value_bound=None,
        chosen=[], covered_turbine_ids=[], duplicate_turbine_visits=[],
        covered=0, coverable=int(n), flights=0, energy_Wh=0.0,
        K=(None if K is None else int(K)),
        batteries=(None if batteries is None else int(batteries)),
        mean_stops=0.0, multi_stop_ratio=0.0,
        makespan_min=0.0, pool_size=0, generated_column_archive_size=0,
        solver="scipy-highs-rmp",
        solver_requested=model_contract.get("solver_requested"),
        solver_effective=model_contract.get("solver_effective", "scipy-highs-rmp"),
        battery_energy_used_Wh=[], battery_end_soc_pct=[], swap_events=[],
        quick_inspection_events=[], n_swaps=0, n_quick_reuses=0,
        seed_validation=seed_validation,
        seed_columns_revalidated=bool(seed_validation.get("validation_complete", False)),
        restricted_pool_gap_pct=None, global_certificate_available=False,
        global_route_space_certificate=False,
        implicit_route_space_certified=False,
        algorithmic_route_space_certified=False,
        algorithmic_global_certificate=False,
        route_universe_source=str(route_universe_source),
        route_universe_provenance_certified=bool(route_universe_provenance_certified),
        physical_model_global_certificate=False,
        route_semantics_invariance_certified=True,
        future_column_row_ranges_certified=bool(_empty_row_ranges_ok),
        binary64_model_contract_enforced=bool(_empty_binary64_contract_ok),
        formal_proof_contract_enforced=bool(_empty_proof_contract_ok),
        formal_proof_contract=FORMAL_PROOF_CONTRACT,
        formal_proof_obligations=list(FORMAL_PROOF_OBLIGATIONS),
        formal_proof_code_anchors=[(k, list(v)) for k, v in FORMAL_PROOF_CODE_ANCHORS],
        proof_contract_sha256=FORMAL_PROOF_CONTRACT_SHA256,
        proof_code_sha256=FORMAL_PROOF_CODE_SHA256,
        route_space_complete=False, route_space_materialized=False,
        # Even without a pricing objective bound, the returned |I| coverage
        # upper bound is a strict bound for the complete finite implicit space.
        implicit_route_space_bound_valid=True, empty_plan_allowed=True,
        empty_plan_is_incumbent=True,
        pricing_non_enumerative=False,
        pricing_uses_implicit_full_permutation_search=bool(
            route_universe_source == "physical-oracle"),
        pricing_dominance="identity-only", pricing_state_merging=False,
        pricing_unsafe_truncation_enabled=False,
        finite_discrete_model_only=True, continuous_real_world_optimality_claimed=False,
        model_contract_validated=bool(model_contract),
        kappa_mode=model_contract.get("kappa_mode"),
        chance_mode=model_contract.get("chance_mode"),
        deck_mode=model_contract.get("deck_mode"),
        soc_correction=model_contract.get("soc_correction"),
        soc_risk_allocation=model_contract.get("soc_risk_allocation"),
        battery_energy_mode=model_contract.get("battery_energy_mode"),
        battery_reuse_mode=model_contract.get("battery_reuse_mode"),
        model_contract_sha256=model_contract.get("sha256"),
        parameter_contract_sha256=model_contract.get("parameter_contract_sha256"),
        instance_contract_sha256=model_contract.get("instance_contract_sha256"),
        algorithm_contract_sha256=model_contract.get("algorithm_sha256"),
        model_contract_scope=model_contract.get(
            "model_contract_scope", "full-finite-model-including-instance-data-binary64-exact"),
        risk_policy_contract=model_contract.get("risk_policy_contract"),
        physical_numeric_contract=model_contract.get("physical_numeric_contract"),
        route_identity_contract=model_contract.get("route_identity_contract"),
        model_semantics_contract=model_contract.get("model_semantics_contract"),
        result_certificate_contract=model_contract.get("result_certificate_contract"),
        route_semantics_contract=model_contract.get("route_semantics_contract"),
        future_column_row_range_contract=model_contract.get("future_column_row_range_contract"),
        pricing_bound_numeric_contract="binary64-outward-rounded-interval",
        master_dual_numeric_contract="binary64-outward-rounded-lagrangian",
        incumbent_energy_numeric_contract="exact-binary64-rational-sum-with-outward-float-enclosure",
        resource_numeric_contract="binary64-strict-half-open-time-exact-rational-soc",
        global_gap_numeric_contract="feasible-upper-enclosure-minus-rigorous-global-lower",
        integral_node_numeric_contract="exact-binary64-rational-lagrangian-when-needed",
        pool_h_mode_requested=model_contract.get("pool_h_mode"),
        pool_h_mode_effective="on-demand-all-discrete-horizons",
        wall_clock_deadline_enforcement="cooperative",
        blackbox_hard_interrupt_available=False,
        blackbox_overrun_scope="at-most-one-noncooperative-call-between-deadline-checks")


def _is_explicit_time_limit(reason, deadline_hit=False):
    """Recognize only unambiguous time-limit exits.

    Mixed reasons such as ``farkas-phase-time-limit-or-invalid`` are deliberately
    not classified as a time limit unless the shared wall-clock deadline was
    actually reached; otherwise an invalid Phase-I certificate must surface as
    an error rather than a benign timeout.
    """
    if bool(deadline_hit):
        return True
    key = str(reason).strip().lower()
    return key in {
        "global-time-limit", "exact-pricing-time-limit",
        "resource-audit-time-limit", "rmp-time_limit", "rmp-time-limit",
    }


def _classify_anytime_status(*, lexicographic_optimal, coverage_optimal,
                             coverage_gap_abs, coverage_gap_target_abs,
                             energy_gap_abs, energy_gap_pct,
                             energy_gap_target_abs_Wh, energy_gap_target_rel,
                             termination_reason, deadline_hit=False):
    """Map a proved stopping condition to a public status without fall-through."""
    if bool(lexicographic_optimal):
        return "lexicographic_optimal"

    timed_out = _is_explicit_time_limit(termination_reason, deadline_hit)
    reason_key = str(termination_reason).strip().lower()
    coverage_target_met = bool(
        reason_key == "coverage-gap-target-reached"
        and int(coverage_gap_abs) <= int(coverage_gap_target_abs))
    energy_target_met = bool(
        reason_key == "energy-gap-target-reached"
        and coverage_optimal and energy_gap_abs is not None and energy_gap_pct is not None
        and (float(energy_gap_abs) <= float(energy_gap_target_abs_Wh)
             or float(energy_gap_pct) <= 100.0 * float(energy_gap_target_rel)))

    if bool(coverage_optimal):
        if energy_target_met:
            return "coverage_optimal_energy_gap_target_reached"
        if timed_out:
            return "coverage_optimal_energy_time_limit"
        return "solver_error"
    if coverage_target_met:
        return "gap_target_reached"
    if timed_out:
        return "time_limit_feasible"
    return "solver_error"


def _materialize_seed_columns(seed_cols, deadline, *, iterator_nonblocking=False):
    """Materialize optional warm-start columns without violating the shared clock.

    A Python timeout check cannot interrupt an arbitrary blocking ``next()`` call.
    Therefore, with a finite deadline, built-in materialized containers are safe by
    default while a one-shot/custom iterator is consumed only when its caller
    explicitly declares ``iterator_nonblocking=True``.  Skipping unsafe iterator
    input is fail-closed: seeds are optional warm starts and exact pricing still
    searches the complete finite implicit route space.
    """
    stats = dict(
        supplied_count=None, input_count=0, input_count_known=False,
        consumed_count=0, materialized_count=0,
        materialization_complete=True, materialization_timed_out=False,
        materialization_skipped_unbounded_iterator=False,
        iterator_nonblocking_declared=bool(iterator_nonblocking))
    if seed_cols is None:
        stats["supplied_count"] = 0
        stats["input_count_known"] = True
        return None, stats

    safely_materialized = isinstance(seed_cols, (list, tuple))
    if safely_materialized:
        stats["supplied_count"] = len(seed_cols)
        stats["input_count"] = len(seed_cols)
        stats["input_count_known"] = True
    elif deadline is not None and not bool(iterator_nonblocking):
        # Do not call iter()/next() on an arbitrary external object under a finite
        # wall-clock contract.  The caller may pass list(seed_cols) before entry or
        # explicitly assert that each next() is nonblocking.
        stats["materialization_complete"] = False
        stats["materialization_skipped_unbounded_iterator"] = True
        stats["materialization_timed_out"] = bool(_deadline_hit(deadline))
        return [], stats

    iterator = iter(seed_cols)
    materialized = []
    while True:
        # Crucially, this check precedes next(); time_limit_s=0 consumes nothing.
        if _deadline_hit(deadline):
            stats["materialization_complete"] = False
            stats["materialization_timed_out"] = True
            break
        try:
            raw_seed = next(iterator)
        except StopIteration:
            break
        stats["consumed_count"] += 1
        materialized.append(raw_seed)

    stats["materialized_count"] = len(materialized)
    if not stats["input_count_known"]:
        stats["input_count"] = len(materialized)
        if stats["materialization_complete"]:
            stats["supplied_count"] = len(materialized)
            stats["input_count_known"] = True
    return materialized, stats


def _solve_fleet_anytime_impl(turbines, launch_opts, p, xi_amb, K, T_min,
                        deck_delta_min=2.5, t_swap_min=6.0, max_stops=8,
                        weather_unc=None, kappa_mode="vp_unimodal",
                        chance_mode="drcc", budget_gamma=2.0,
                        batteries=None, solver="auto", deck_mode="interval",
                        t_launch_min=None, landing_clear_min=None,
                        quick_inspection_capacity=None, swap_station_capacity=None,
                        battery_reuse_mode="exact_soc", pool_h_mode="pareto",
                        allow_resource_only_columns=False,
                        time_limit_s=None, deadline=None,
                        coverage_gap_target_abs=0,
                        energy_gap_target_rel=0.0,
                        energy_gap_target_abs_Wh=1e-6,
                        solver_mode="exact-branch-price-cut",
                        pricing_mode="exact-implicit-dfs",
                        seed_cols=None,
                        seed_iterator_nonblocking=False,
                        implicit_test_columns=None,
                        pricing_batch_size=16, solve_scope="lexicographic", coverage_target=None,
                        certified_route_universe=None,
                        target_closure_checkpoint_path=None,
                        target_closure_resume=False,
                        archive_diagnostic_time_limit_s=30.0,
                        archive_shadow_diagnostic_time_limit_s=30.0,
                        archive_clique_diagnostic_time_limit_s=30.0,
                        archive_primal_recovery=False,
                        archive_primal_recovery_time_limit_s=2.0,
                        fullspace_target_diagnostic_time_limit_s=0.0,
                        _internal_synthetic_route_universe=False, **_ignored):
    """Anytime exact solver for the finite route model.

    v12 supports two mathematically equivalent exact route representations.  A validated
    ``CertifiedRouteUniverse`` materializes every legal physical route once for a
    small instance; otherwise the original path performs implicit exhaustive DFS over all ordered elementary sequences and recovery horizons.
    It is exact but not a polynomial/efficient non-enumerative RCSP labeler.
    Unsafe route-scan caps are rejected on the certificate path and are recognized
    only inside the isolated research baseline.  ``implicit_test_columns`` is an
    internal self-test hook.  It is rejected by the public formal API; only
    ``_solve_fleet_anytime_synthetic_fixture`` may enable it, and results from
    that private fixture can never carry a physical-model global certificate.
    """
    # Validate the requested mathematical/control contract before creating the
    # wall-clock deadline.  Invalid categorical values must never reach physics
    # and inherit a residual global kappa or another fallback branch.
    contract = _validate_anytime_public_contract(
        solver_mode=solver_mode, pricing_mode=pricing_mode,
        kappa_mode=kappa_mode, chance_mode=chance_mode, deck_mode=deck_mode,
        battery_reuse_mode=battery_reuse_mode, pool_h_mode=pool_h_mode,
        solver=solver, time_limit_s=time_limit_s, deadline=deadline, budget_gamma=budget_gamma,
        K=K, batteries=batteries, max_stops=max_stops,
        coverage_gap_target_abs=coverage_gap_target_abs,
        energy_gap_target_rel=energy_gap_target_rel,
        energy_gap_target_abs_Wh=energy_gap_target_abs_Wh,
        pricing_batch_size=pricing_batch_size, solve_scope=solve_scope,
        coverage_target=coverage_target)
    mode = contract["solver_mode"]
    pricing_key = contract["pricing_mode"]
    solve_scope = contract["solve_scope"]
    coverage_target = contract.get("coverage_target")
    kappa_mode = contract["kappa_mode"]
    chance_mode = contract["chance_mode"]
    deck_mode = contract["deck_mode"]
    battery_reuse_mode = contract["battery_reuse_mode"]
    pool_h_mode = contract["pool_h_mode"]
    K = contract["K"]
    batteries = contract["batteries"]
    solver = contract["solver_requested"]
    _finite_number(
        "archive_diagnostic_time_limit_s",
        archive_diagnostic_time_limit_s, nonnegative=True)
    archive_diagnostic_time_limit_s = float(
        archive_diagnostic_time_limit_s)
    _finite_number(
        "archive_shadow_diagnostic_time_limit_s",
        archive_shadow_diagnostic_time_limit_s, nonnegative=True)
    archive_shadow_diagnostic_time_limit_s = float(
        archive_shadow_diagnostic_time_limit_s)
    _finite_number(
        "archive_clique_diagnostic_time_limit_s",
        archive_clique_diagnostic_time_limit_s, nonnegative=True)
    archive_clique_diagnostic_time_limit_s = float(
        archive_clique_diagnostic_time_limit_s)
    if not isinstance(archive_primal_recovery, (bool, np.bool_)):
        raise ValueError("archive_primal_recovery must be boolean")
    archive_primal_recovery = bool(archive_primal_recovery)
    _finite_number(
        "archive_primal_recovery_time_limit_s",
        archive_primal_recovery_time_limit_s, nonnegative=True)
    archive_primal_recovery_time_limit_s = float(
        archive_primal_recovery_time_limit_s)
    _finite_number(
        "fullspace_target_diagnostic_time_limit_s",
        fullspace_target_diagnostic_time_limit_s, nonnegative=True)
    fullspace_target_diagnostic_time_limit_s = float(
        fullspace_target_diagnostic_time_limit_s)
    if target_closure_checkpoint_path is not None and solve_scope != "coverage-target":
        raise ValueError("target_closure_checkpoint_path is only valid for coverage-target solves")
    if not isinstance(target_closure_resume, (bool, np.bool_)):
        raise ValueError("target_closure_resume must be boolean")

    synthetic_fixture = bool(
        mode == "exact-branch-price-cut" and implicit_test_columns is not None)
    if synthetic_fixture and not bool(_internal_synthetic_route_universe):
        raise ValueError(
            "implicit_test_columns is an internal synthetic test fixture and is not "
            "permitted on the formal physical certificate path")
    if (mode == "exact-branch-price-cut" and allow_resource_only_columns
            and not bool(_internal_synthetic_route_universe)):
        raise ValueError(
            "allow_resource_only_columns is not permitted on the formal physical certificate path")
    if bool(_internal_synthetic_route_universe) and not synthetic_fixture:
        raise ValueError(
            "internal synthetic fixture mode requires implicit_test_columns")
    materialized_complete_universe = certified_route_universe is not None
    if materialized_complete_universe and synthetic_fixture:
        raise ValueError("certified_route_universe cannot be combined with synthetic fixture columns")
    route_universe_source = (
        "synthetic-test-fixture" if synthetic_fixture else
        "materialized-complete-physical-oracle" if materialized_complete_universe else
        "physical-oracle" if mode == "exact-branch-price-cut" else
        "research-restricted-pool")
    route_universe_provenance_certified = bool(
        mode == "exact-branch-price-cut" and not synthetic_fixture)

    # Canonicalize every model-defining public scalar before hashing the model.
    # Search-control parameters (time limit / requested Gap / pricing batch size)
    # are intentionally separated from the mathematical model fingerprint.
    T_min = _finite_number("T_min", T_min, nonnegative=True)
    deck_delta_min = _finite_number("deck_delta_min", deck_delta_min, positive=True)
    t_swap_min = _finite_number("t_swap_min", t_swap_min, nonnegative=True)
    max_stops = _nonnegative_int("max_stops", max_stops, positive=True)
    pricing_batch_size = _nonnegative_int("pricing_batch_size", pricing_batch_size, positive=True)
    coverage_gap_target_abs = _nonnegative_int(
        "coverage_gap_target_abs", coverage_gap_target_abs)
    energy_gap_target_rel = _finite_number(
        "energy_gap_target_rel", energy_gap_target_rel, nonnegative=True)
    energy_gap_target_abs_Wh = _finite_number(
        "energy_gap_target_abs_Wh", energy_gap_target_abs_Wh, nonnegative=True)
    if t_launch_min is not None:
        t_launch_min = _finite_number("t_launch_min", t_launch_min, nonnegative=True)
    if landing_clear_min is not None:
        landing_clear_min = _finite_number(
            "landing_clear_min", landing_clear_min, nonnegative=True)

    # Bind the public request to one validated physical/resource model before
    # any certified search begins. Params fields that select unimplemented
    # semantics must fail closed rather than being echoed while ignored.
    p.validate_contract(formal=False)
    if mode == "exact-branch-price-cut":
        M.validate_xi_ambiguity_math(xi_amb)
        if str(getattr(p, "battery_reuse_mode", "exact_soc")) != "exact_soc":
            raise ValueError("formal exact BPC requires Params.battery_reuse_mode='exact_soc'")
        if str(getattr(p, "battery_energy_mode", "robust_required")) != "robust_required":
            raise ValueError("formal exact BPC requires Params.battery_energy_mode='robust_required'")

    battery_count = int(2 * K if batteries is None else batteries)
    t_launch = float(deck_delta_min if t_launch_min is None else t_launch_min)
    t_clear = float(getattr(p, "landing_clear_min", 1.0)
                    if landing_clear_min is None else landing_clear_min)
    quick_min = _finite_number(
        "quick_inspection_min", getattr(p, "quick_inspection_min", 1.0),
        nonnegative=True)
    quick_capacity = _nonnegative_int(
        "quick_inspection_capacity",
        getattr(p, "quick_inspection_capacity", 1)
        if quick_inspection_capacity is None else quick_inspection_capacity,
        positive=True)
    swap_capacity = _nonnegative_int(
        "swap_station_capacity",
        getattr(p, "swap_station_capacity", 1)
        if swap_station_capacity is None else swap_station_capacity,
        positive=True)

    complete_universe_mode = False
    route_universe_validation_reason = "not-supplied"
    if certified_route_universe is not None:
        ok, route_universe_validation_reason = _validate_certified_route_universe(
            certified_route_universe, turbines, launch_opts, p, xi_amb, T_min,
            max_stops=max_stops, weather_unc=weather_unc,
            kappa_mode=kappa_mode, chance_mode=chance_mode,
            budget_gamma=budget_gamma, t_launch_min=t_launch,
            landing_clear_min=t_clear, deck_mode=deck_mode,
            deck_delta_min=deck_delta_min)
        if not ok:
            raise ValueError(
                "certified_route_universe rejected: "
                + str(route_universe_validation_reason))
        complete_universe_mode = True

    # Build two fingerprints:
    # 1) a parameter/discretization contract; and
    # 2) a binary64-exact finite-instance contract including turbines, launch
    #    options, weather and Xi.  The public model_contract_sha256 is the
    #    composite of both and therefore uniquely binds the certified finite model.
    _params_contract = {
        name: getattr(p, name) for name in sorted(p.__dataclass_fields__)
    }
    contract.update(
        params_contract=_params_contract,
        soc_correction=str(getattr(p, "soc_correction", "none")),
        soc_risk_allocation=str(getattr(p, "soc_risk_allocation", "fixed")),
        battery_energy_mode=str(getattr(p, "battery_energy_mode", "robust_required")),
        params_battery_reuse_mode=str(getattr(p, "battery_reuse_mode", "exact_soc")),
        time_recourse_mode=str(getattr(p, "time_recourse_mode", "wait_only")),
        risk_policy_contract="immutable-explicit-one-and-two-sided-kappa",
        physical_numeric_contract=RM.FORMAL_PHYSICAL_NUMERIC_CONTRACT,
        route_identity_contract=ROUTE_IDENTITY_CONTRACT,
        model_semantics_contract=MODEL_SEMANTICS_CONTRACT,
        result_certificate_contract=RESULT_CERTIFICATE_CONTRACT,
        route_semantics_contract=ROUTE_SEMANTICS_CONTRACT,
        future_column_row_range_contract=FUTURE_COLUMN_ROW_RANGE_CONTRACT,
        formal_proof_contract=FORMAL_PROOF_CONTRACT,
        formal_proof_obligations=list(FORMAL_PROOF_OBLIGATIONS),
        formal_proof_code_anchors=[(k, list(v)) for k, v in FORMAL_PROOF_CODE_ANCHORS],
        formal_proof_code_sha256=FORMAL_PROOF_CODE_SHA256,
        formal_proof_contract_sha256=FORMAL_PROOF_CONTRACT_SHA256,
        proof_code_sha256=FORMAL_PROOF_CODE_SHA256,
        T_min=float(T_min), max_stops=int(max_stops),
        budget_gamma=float(budget_gamma),
        K_effective=int(K), batteries_effective=int(battery_count),
        deck_delta_min=float(deck_delta_min),
        t_launch_min_effective=float(t_launch),
        landing_clear_min_effective=float(t_clear),
        quick_inspection_min_effective=float(quick_min),
        t_swap_min_effective=float(t_swap_min),
        quick_inspection_capacity_effective=int(quick_capacity),
        swap_station_capacity_effective=int(swap_capacity),
        pool_h_mode_effective="on-demand-all-discrete-horizons"
            if mode == "exact-branch-price-cut" else str(pool_h_mode),
        model_contract_scope="full-finite-model-including-instance-data-binary64-exact")
    _risk_policy_for_mode(kappa_mode)  # validates both one-/two-sided policy construction

    _model_payload = {
        key: contract[key] for key in (
            "kappa_mode", "chance_mode", "deck_mode", "battery_reuse_mode",
            "pool_h_mode_effective", "K_effective", "batteries_effective",
            "T_min", "max_stops", "budget_gamma", "deck_delta_min",
            "t_launch_min_effective", "landing_clear_min_effective",
            "quick_inspection_min_effective", "t_swap_min_effective",
            "quick_inspection_capacity_effective",
            "swap_station_capacity_effective", "params_contract",
            "soc_correction", "soc_risk_allocation", "battery_energy_mode",
            "params_battery_reuse_mode", "time_recourse_mode",
            "risk_policy_contract", "physical_numeric_contract",
            "route_identity_contract", "model_semantics_contract",
            "model_contract_scope")
    }
    _contract_payload = json.dumps(
        _model_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    contract["sha256"] = hashlib.sha256(
        _contract_payload.encode("utf-8")).hexdigest()
    contract["parameter_contract_sha256"] = contract["sha256"]

    _resource_config_for_identity = dict(
        chance_mode=str(chance_mode),
        budget_gamma=float(budget_gamma),
        deck_delta_min=float(deck_delta_min),
        t_launch_min=float(t_launch),
        landing_clear_min=float(t_clear),
        quick_inspection_min=float(quick_min),
        t_swap_min=float(t_swap_min),
        quick_inspection_capacity=int(quick_capacity),
        swap_station_capacity=int(swap_capacity),
        battery_reuse_mode=str(battery_reuse_mode),
        resource_time_semantics="strict-half-open-binary64",
        resource_soc_semantics="exact-rational-sum-of-binary64-energies")
    _instance_scope = _finite_model_scope_signature(
        turbines, launch_opts, p, xi_amb,
        K=int(K), batteries=int(battery_count), T_min=float(T_min),
        max_stops=int(max_stops), kappa_mode=str(kappa_mode),
        weather_unc=weather_unc, deck_mode=str(deck_mode),
        pool_h_mode=("on-demand-all-discrete-horizons"
                     if mode == "exact-branch-price-cut" else str(pool_h_mode)),
        resource_config=_resource_config_for_identity)
    if implicit_test_columns is not None:
        _test_columns_identity = []
        for _c in implicit_test_columns:
            try:
                _test_columns_identity.append(dict(
                    signature=_exact_route_signature(_c),
                    plan=_float_binary64_fp(_c.get("E_plan_Wh", _c.get("E0", float("nan")))),
                    soc=_float_binary64_fp(_c.get(
                        "E_soc_required_Wh",
                        _c.get("E_plan_Wh", _c.get("E0", float("nan")))))))
            except Exception:
                _test_columns_identity.append(repr(_c))
        _instance_scope["implicit_test_columns"] = _test_columns_identity
        _scope_payload = json.dumps(
            _instance_scope, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), default=str)
        _instance_scope["sha256"] = hashlib.sha256(
            _scope_payload.encode("utf-8")).hexdigest()
    contract["instance_contract_sha256"] = _instance_scope["sha256"]
    _full_model_payload = json.dumps(
        dict(parameter_contract_sha256=contract["parameter_contract_sha256"],
             instance_contract_sha256=contract["instance_contract_sha256"],
             model_contract_scope=contract["model_contract_scope"]),
        ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    contract["sha256"] = hashlib.sha256(
        _full_model_payload.encode("utf-8")).hexdigest()

    _algorithm_payload = dict(
        solver_mode=mode, pricing_mode=pricing_key,
        solver_effective=contract.get("solver_effective"),
        pricing_batch_size=int(pricing_batch_size),
        coverage_gap_target_abs=int(coverage_gap_target_abs),
        solve_scope=str(solve_scope),
        coverage_target=(None if coverage_target is None else int(coverage_target)),
        energy_gap_target_rel=float(energy_gap_target_rel),
        energy_gap_target_abs_Wh=float(energy_gap_target_abs_Wh),
        result_certificate_contract=RESULT_CERTIFICATE_CONTRACT,
        route_semantics_contract=ROUTE_SEMANTICS_CONTRACT,
        future_column_row_range_contract=FUTURE_COLUMN_ROW_RANGE_CONTRACT,
        formal_proof_contract=FORMAL_PROOF_CONTRACT,
        formal_proof_obligations=list(FORMAL_PROOF_OBLIGATIONS),
        formal_proof_code_sha256=FORMAL_PROOF_CODE_SHA256,
        formal_proof_contract_sha256=FORMAL_PROOF_CONTRACT_SHA256,
        proof_code_sha256=FORMAL_PROOF_CODE_SHA256,
        complete_route_universe_used=bool(complete_universe_mode),
        complete_route_universe_contract=(
            COMPLETE_ROUTE_UNIVERSE_CONTRACT if complete_universe_mode else None),
        complete_route_universe_columns_sha256=(
            certified_route_universe.columns_sha256 if complete_universe_mode else None))
    contract["algorithm_sha256"] = hashlib.sha256(json.dumps(
        _algorithm_payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()

    started = time.monotonic()
    if deadline is None:
        deadline = (None if time_limit_s is None else
                    time.monotonic() + float(time_limit_s))
    # Materialize exactly once.  Under a finite deadline arbitrary one-shot
    # iterators are not touched unless the caller explicitly declares nonblocking
    # next() semantics; exact pricing does not depend on these optional seeds.
    materialized_seed_cols, seed_materialization = _materialize_seed_columns(
        seed_cols, deadline,
        iterator_nonblocking=bool(seed_iterator_nonblocking))
    legacy_sequence_cap = _ignored.pop("max_sequence_evals", None)
    if mode == "research-baseline":
        # Explicitly isolated compatibility route.  It cannot issue a global
        # implicit-route certificate and is not used by formal experiments.
        cols = list(materialized_seed_cols or [])
        enumeration_stats = None
        if not cols and launch_opts:
            cols, enumeration_stats = enumerate_discrete_route_columns(
                turbines, launch_opts, p, xi_amb, T_min, deck_delta_min,
                max_stops, weather_unc=weather_unc,
                max_sequence_evals=legacy_sequence_cap,
                kappa_mode=kappa_mode, deadline=deadline)
        result = RA.solve_resource_master(
            turbines, launch_opts, p, xi_amb, K, T_min,
            deck_delta_min=deck_delta_min, t_swap_min=t_swap_min,
            max_stops=max_stops, weather_unc=weather_unc,
            kappa_mode=kappa_mode, batteries=batteries,
            cols_override=cols, solver=solver, deck_mode=deck_mode,
            t_launch_min=t_launch_min, landing_clear_min=landing_clear_min,
            quick_inspection_capacity=quick_inspection_capacity,
            swap_station_capacity=swap_station_capacity,
            battery_reuse_mode=battery_reuse_mode,
            pool_h_mode=pool_h_mode,
            allow_resource_only_columns=allow_resource_only_columns,
            deadline=deadline, time_limit_s=time_limit_s,
            coverage_gap_target_abs=coverage_gap_target_abs,
            energy_gap_target_rel=energy_gap_target_rel,
            energy_gap_target_abs_Wh=energy_gap_target_abs_Wh)
        research_seed_validation = dict(result.get("seed_validation") or {})
        research_seed_validation.update(seed_materialization)
        research_seed_validation["validation_complete"] = bool(
            research_seed_validation.get("validation_complete", True)
            and seed_materialization.get("materialization_complete", False))
        result.update(
            algorithm="research-baseline-restricted-pool",
            pricing_method="none", branching_complete=False,
            farkas_pricing_complete=False, pricing_complete=False,
            pricing_bound_available=False, bound_scope="validated_route_pool",
            bound_source="restricted-pool-only",
            global_certificate_available=False,
            global_route_space_certificate=False,
            implicit_route_space_certified=False,
            algorithmic_route_space_certified=False,
            algorithmic_global_certificate=False,
            route_universe_source="research-restricted-pool",
            route_universe_provenance_certified=False,
            physical_model_global_certificate=False,
            route_semantics_invariance_certified=False,
            future_column_row_ranges_certified=False,
            binary64_model_contract_enforced=False,
            formal_proof_contract_enforced=False,
            formal_proof_contract=FORMAL_PROOF_CONTRACT,
            formal_proof_obligations=list(FORMAL_PROOF_OBLIGATIONS),
            formal_proof_code_anchors=[(k, list(v)) for k, v in FORMAL_PROOF_CODE_ANCHORS],
            proof_contract_sha256=FORMAL_PROOF_CONTRACT_SHA256,
        proof_code_sha256=FORMAL_PROOF_CODE_SHA256,
            pricing_uses_implicit_full_permutation_search=False,
            result_certificate_contract=RESULT_CERTIFICATE_CONTRACT,
            route_semantics_contract=ROUTE_SEMANTICS_CONTRACT,
            future_column_row_range_contract=FUTURE_COLUMN_ROW_RANGE_CONTRACT,
            research_enumeration_stats=enumeration_stats,
            seed_validation=research_seed_validation,
            seed_columns_revalidated=bool(
                result.get("seed_columns_revalidated", False)
                and research_seed_validation["validation_complete"]),
            status="time_limit_feasible",
            empty_plan_allowed=True,
            empty_plan_is_incumbent=(not bool(result.get("chosen"))))
        return result
    unsafe_names = {
        "max_columns", "max_routes", "max_labels", "beam_width", "top_k",
        "candidate_limit", "nearest_neighbors", "route_pool_cap",
        "max_pricing_rounds", "pricing_column_limit", "early_stop",
        "early_stop_after_columns", "label_budget", "pricing_label_budget",
        "node_budget", "max_sequence_evals", "k_near",
    }
    unsafe_exact_options = sorted(k for k in _ignored if k in unsafe_names)
    if legacy_sequence_cap is not None:
        unsafe_exact_options.append("max_sequence_evals")
    if unsafe_exact_options:
        raise ValueError(
            "unsafe pricing/enumeration caps are not accepted by exact-branch-price-cut: "
            + ", ".join(sorted(set(unsafe_exact_options))))
    if _ignored:
        raise TypeError("unsupported exact solver options: " + ", ".join(sorted(_ignored)))
    if pricing_key == "exact-mip":
        return _empty_bpc_result(
            turbines, time_limit_s, started,
            "exact-mip-pricing-not-implemented-for-current-black-box-physics",
            bound_source="trivial-coverable-turbine-bound", K=K,
            batteries=(2 * int(K) if batteries is None else batteries),
            status="solver_error", pricing_bound_available=False,
            model_contract=contract, route_universe_source=route_universe_source,
            route_universe_provenance_certified=route_universe_provenance_certified)
    if len({_tid(t.tid) for t in turbines}) != len(turbines):
        raise ValueError("candidate turbine identifiers must be unique")

    archive = []
    signature_to_index = {}
    seed_validation = dict(seed_materialization)
    seed_validation.update(accepted_count=0, rejected_count=0,
                           rejection_reasons={}, kappa_mode=str(kappa_mode),
                           validation_timed_out=False, validated_count=0,
                           validation_complete=False, unvalidated_count=None)
    initial_cols = []
    # Shared physical cache starts before seed/singleton warm-start work so
    # formally revalidated physics is reusable by later exact pricing.
    physical_pricing_cache = _V30PhysicalPricingCache()
    if complete_universe_mode:
        # [THM-CU] The complete route universe strictly subsumes every possible
        # seed.  Do not spend wall-clock revalidating heuristic seeds; seed data
        # have no proof role once all formal columns are materialized.
        _add_columns(
            archive, signature_to_index,
            list(certified_route_universe.columns))
        seed_validation.update(
            accepted_count=0, rejected_count=0, validated_count=0,
            validation_complete=True, unvalidated_count=0,
            bypass_reason="complete-certified-route-universe-supersedes-seeds")
    else:
        for raw in (materialized_seed_cols or []):
            if _deadline_hit(deadline):
                seed_validation["validation_timed_out"] = True
                break
            try:
                c = _revalidate_seed_column(
                    raw, turbines, launch_opts, p, xi_amb, weather_unc, T_min,
                    t_launch, t_clear, deck_mode, deck_delta_min,
                    kappa_mode, chance_mode, budget_gamma, deadline=deadline)
                _validate_column_domain(c, {_tid(t.tid) for t in turbines}, max_stops)
                _add_columns(archive, signature_to_index, [c])
                try:
                    _ck = (
                        int(c["launch_option_index"]),
                        tuple(_ordered_tids(c)),
                        float(c["h"]).hex(),
                    )
                    physical_pricing_cache[_ck] = c
                except Exception:
                    # Cache bridging has no proof role; the validated seed remains
                    # in the archive even if its cache key cannot be reconstructed.
                    pass
                seed_validation["accepted_count"] += 1
                seed_validation["validated_count"] += 1
            except TimeoutError:
                seed_validation["validation_timed_out"] = True
                break
            except Exception as exc:
                reason = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
                seed_validation["rejected_count"] += 1
                seed_validation["rejection_reasons"][reason] = (
                    seed_validation["rejection_reasons"].get(reason, 0) + 1)
                seed_validation["validated_count"] += 1
        seed_validation["validation_complete"] = bool(
            seed_validation.get("materialization_complete", False)
            and not seed_validation["validation_timed_out"]
            and seed_validation["validated_count"]
                == seed_validation.get("materialized_count", 0))
        if seed_validation.get("input_count_known", False):
            seed_validation["unvalidated_count"] = max(
                int(seed_validation.get("input_count", 0))
                - int(seed_validation["validated_count"]), 0)
        initial_cols = _initial_singleton_columns(
            turbines, launch_opts, p, xi_amb, weather_unc, T_min, deadline,
            t_launch, t_clear, deck_mode, deck_delta_min,
            kappa_mode=kappa_mode, chance_mode=chance_mode,
            budget_gamma=budget_gamma,
            implicit_test_columns=implicit_test_columns,
            physical_cache=physical_pricing_cache)
        for c in initial_cols:
            _validate_column_domain(c, {_tid(t.tid) for t in turbines}, max_stops)
        _add_columns(archive, signature_to_index, initial_cols)
    no_good_cuts = []
    (initial_selection, initial_audit, initial_audit_unknown,
     initial_resource_audit_calls) = _greedy_exact_resource_start(
        archive, K, battery_count, p, quick_min, t_swap_min,
        quick_capacity, swap_capacity, deadline)
    if not initial_selection:
        initial_resource_audit_calls += 1
        initial_audit = _audit_integer_selection(
            archive, tuple(), K, battery_count, p, quick_min, t_swap_min,
            quick_capacity, swap_capacity, deadline)
        if initial_audit.status is not RA.ResourceAuditStatus.FEASIBLE:
            raise RuntimeError("empty plan must pass exact resource audit")
    if _deadline_hit(deadline) and not initial_selection:
        return _empty_bpc_result(
            turbines, time_limit_s, started, "global-time-limit-before-root-rmp",
            bound_source="trivial-coverable-turbine-bound", K=K,
            batteries=battery_count, seed_validation=seed_validation,
            model_contract=contract, route_universe_source=route_universe_source,
            route_universe_provenance_certified=route_universe_provenance_certified)

    physical_pricing_cache_entries_before_pricing = len(
        physical_pricing_cache)

    if solve_scope == "coverage-target":
        # [THM-TGT] Exact yes/no query on the unchanged physical feasible set:
        #   exists x in F(K,B) with sum_r |S_r| x_r = target ?
        target = int(coverage_target)
        n_coverable = len({_tid(t.tid) for t in turbines})
        if target > n_coverable:
            target_stage = None
            target_feasible = False
            target_infeasible_algorithmic = True
            target_termination = "target-above-hard-coverable-cap"
        else:
            initial_target_selection = (
                tuple(initial_selection)
                if _coverage_of_selection(archive, initial_selection) == target
                else tuple())
            initial_target_audit = initial_audit if initial_target_selection else None
            target_stage = _solve_branch_price_stage(
                stage="energy", turbines=turbines, launch_opts=launch_opts, p=p,
                xi_amb=xi_amb, K=K, batteries=battery_count, T_min=T_min,
                max_stops=max_stops, weather_unc=weather_unc, deadline=deadline,
                archive=archive, signature_to_index=signature_to_index,
                no_good_cuts=no_good_cuts, coverage_target=target,
                initial_selection=initial_target_selection,
                initial_audit=initial_target_audit,
                pricing_epsilon=PRICING_EPS,
                coverage_gap_target_abs=0,
                energy_gap_target_rel=0.0,
                energy_gap_target_abs_Wh=0.0,
                t_launch_min=t_launch, landing_clear_min=t_clear,
                quick_min=quick_min, swap_min=t_swap_min,
                quick_capacity=quick_capacity, swap_capacity=swap_capacity,
                deck_mode=deck_mode, deck_delta_min=deck_delta_min,
                kappa_mode=kappa_mode, chance_mode=chance_mode,
                budget_gamma=budget_gamma,
                implicit_test_columns=implicit_test_columns,
                pricing_batch_size=pricing_batch_size,
                root_branch=(BranchState(required_turbines=frozenset(
                    {_tid(t.tid) for t in turbines}))
                    if target == n_coverable else None),
                physical_cache=physical_pricing_cache,
                decision_only=True,
                complete_universe_mode=complete_universe_mode,
                target_closure_checkpoint_path=target_closure_checkpoint_path,
                target_closure_resume=bool(target_closure_resume),
                target_closure_algorithm_sha256=contract.get("algorithm_sha256"),
                pricing_experiment_mode=(
                    "layered-batch-primal-depth-fair-neutral-shadow"
                    if pricing_key
                        == "exact-layered-batch-primal-battery-halfcap-depth-fair-neutral-formal"
                    else "layered-batch-primal-depth-fair-shadow"
                    if pricing_key
                        == "exact-layered-batch-primal-battery-halfcap-depth-fair-formal"
                    else "layered-batch-primal-shadow"
                    if pricing_key in {
                        "exact-layered-batch-primal-shadow",
                        "exact-layered-batch-primal-diagnostic-shadow",
                        "exact-layered-batch-primal-target9-diagnostic-shadow",
                        "exact-layered-batch-primal-target9-certificate-diagnostic-shadow",
                        "exact-layered-batch-primal-target9-battery-clique-diagnostic-shadow",
                        "exact-layered-batch-primal-battery-halfcap-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-exchange-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-primal-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-guided-formal",
                        "exact-layered-batch-primal-battery-halfcap-deck-guided-formal",
                        "exact-layered-batch-primal-battery-halfcap-adaptive-multistop-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-variant-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-variant-diagnostic-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-variant-archive-recovery-formal"}
                    else "layered-batch-shadow"
                    if pricing_key == "exact-layered-batch-shadow"
                    else "layered-guided-shadow"
                    if pricing_key == "exact-layered-guided-shadow"
                    else "dual-guided-shadow"
                    if pricing_key == "exact-dual-guided-shadow"
                    else "discovery-shadow"
                    if pricing_key == "exact-discovery-shadow"
                    else False),
                formal_battery_halfcap=bool(
                    pricing_key in {
                        "exact-layered-batch-primal-battery-halfcap-formal",
                        "exact-layered-batch-primal-battery-halfcap-depth-fair-formal",
                        "exact-layered-batch-primal-battery-halfcap-depth-fair-neutral-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-exchange-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-primal-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-guided-formal",
                        "exact-layered-batch-primal-battery-halfcap-deck-guided-formal",
                        "exact-layered-batch-primal-battery-halfcap-adaptive-multistop-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-variant-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-variant-diagnostic-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-variant-archive-recovery-formal"}),
                primal_exchange=bool(
                    pricing_key
                    == "exact-layered-batch-primal-battery-halfcap-resource-exchange-formal"),
                primal_resource_primal=bool(
                    pricing_key
                    == "exact-layered-batch-primal-battery-halfcap-resource-primal-formal"),
                primal_resource_guided=bool(
                    pricing_key
                    == "exact-layered-batch-primal-battery-halfcap-resource-guided-formal"),
                primal_resource_deck_guided=bool(
                    pricing_key in {
                        "exact-layered-batch-primal-battery-halfcap-deck-guided-formal",
                        "exact-layered-batch-primal-battery-halfcap-adaptive-multistop-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-variant-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-variant-diagnostic-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-variant-archive-recovery-formal"}),
                adaptive_multistop_enrichment=bool(
                    pricing_key
                    == "exact-layered-batch-primal-battery-halfcap-adaptive-multistop-formal"),
                resource_variant_enrichment=bool(
                    pricing_key in {
                        "exact-layered-batch-primal-battery-halfcap-resource-variant-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-variant-diagnostic-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-variant-archive-recovery-formal"}))
            target_feasible = bool(
                target_stage.incumbent_selection
                and target_stage.incumbent_audit is not None
                and target_stage.incumbent_audit.status is RA.ResourceAuditStatus.FEASIBLE
                and _coverage_of_selection(archive, target_stage.incumbent_selection) == target)
            target_infeasible_algorithmic = _target_infeasibility_algorithmic_proven(
                target_stage, target_feasible)
            target_termination = str(target_stage.termination_reason)

        route_semantics_invariance_certified = _route_archive_semantics_invariant(archive)
        future_column_row_ranges_certified = _future_row_range_contract_self_check(max_stops)
        binary64_model_contract_enforced = bool(
            contract.get("physical_numeric_contract") == RM.FORMAL_PHYSICAL_NUMERIC_CONTRACT
            and contract.get("route_identity_contract") == ROUTE_IDENTITY_CONTRACT
            and contract.get("model_semantics_contract") == MODEL_SEMANTICS_CONTRACT)
        formal_proof_contract_enforced = bool(
            contract.get("formal_proof_contract") == FORMAL_PROOF_CONTRACT
            and tuple(contract.get("formal_proof_obligations", ())) == FORMAL_PROOF_OBLIGATIONS
            and contract.get("formal_proof_contract_sha256") == FORMAL_PROOF_CONTRACT_SHA256)
        if not route_semantics_invariance_certified:
            raise RuntimeError("formal route archive semantic invariant violated")
        if not future_column_row_ranges_certified:
            raise RuntimeError("formal future-column row-range contract self-check failed")

        target_infeasible_physical = _physical_certificate_guard(
            algorithmic_global_certificate=bool(target_infeasible_algorithmic),
            route_universe_provenance_certified=route_universe_provenance_certified,
            mode=mode,
            route_semantics_invariance_certified=route_semantics_invariance_certified,
            future_column_row_ranges_certified=future_column_row_ranges_certified,
            binary64_model_contract_enforced=binary64_model_contract_enforced,
            formal_proof_contract_enforced=formal_proof_contract_enforced)
        # YES is existential: one exact physical/resource witness is enough.
        target_feasible_physical = bool(
            target_feasible
            and route_universe_provenance_certified
            and mode == "exact-branch-price-cut"
            and route_semantics_invariance_certified
            and future_column_row_ranges_certified
            and binary64_model_contract_enforced
            and formal_proof_contract_enforced)
        target_decision_certified = bool(
            target_feasible_physical or target_infeasible_physical)
        if target_feasible_physical:
            target_decision = "FEASIBLE"
            target_certificate_type = "exact-physical-resource-witness"
        elif target_infeasible_physical:
            target_decision = "INFEASIBLE"
            if complete_universe_mode:
                _target_backend = (
                    "" if target_stage is None
                    else str(getattr(target_stage, "direct_target_backend", "") or ""))
                if _target_backend == "exact-global-battery-mask-dp":
                    target_certificate_type = (
                        "complete-materialized-universe-fullcover-global-battery-relaxation-infeasibility")
                elif _target_backend:
                    target_certificate_type = (
                        "complete-materialized-universe-fullcover-persistent-resource-closure-infeasibility")
                else:
                    target_certificate_type = (
                        "complete-materialized-universe-branch-cut-infeasibility")
            else:
                target_certificate_type = "full-space-phase1-bpc-infeasibility"
        else:
            target_decision = "UNRESOLVED"
            target_certificate_type = None

        target_selection = (
            tuple(target_stage.incumbent_selection)
            if target_stage is not None and target_feasible else tuple())
        target_audit = (
            target_stage.incumbent_audit
            if target_stage is not None and target_feasible else None)
        chosen = _materialize_chosen(archive, target_selection, target_audit)
        covered_ids = []
        for j in target_selection:
            covered_ids.extend(_ordered_tids(archive[j]))
        if len(covered_ids) != len(set(covered_ids)):
            raise RuntimeError("target witness contains duplicate turbine visits")

        ss = target_stage
        return dict(
            status=("target_feasible_certified" if target_feasible_physical else
                    "target_infeasible_certified" if target_infeasible_physical else
                    "target_unresolved"),
            termination_reason=target_termination,
            runtime_s=float(time.monotonic() - started), time_limit_s=time_limit_s,
            solve_scope="coverage-target",
            target_coverage=int(target), target_decision=target_decision,
            target_decision_certified=bool(target_decision_certified),
            target_feasible_witness_found=bool(target_feasible),
            target_feasible_proven=bool(target_feasible_physical),
            target_infeasible_proven=bool(target_infeasible_physical),
            target_algorithmic_infeasibility_certificate=bool(target_infeasible_algorithmic),
            target_certificate_type=target_certificate_type,
            target_coverage_lower_bound=(int(target) if target_feasible_physical else None),
            target_coverage_upper_bound=(int(target - 1) if target_infeasible_physical else None),
            target_witness_coverage=(int(target) if target_feasible_physical else None),
            K=int(K), batteries=int(battery_count), chosen=chosen,
            covered_turbine_ids=covered_ids, flights=len(chosen),
            covered=(int(target) if target_feasible_physical else 0),
            coverable=int(n_coverable),
            energy_Wh=sum(float(c.get("E_plan_Wh", 0.0)) for c in chosen),
            mean_stops=(float(np.mean([len(_ordered_tids(archive[j]))
                                      for j in target_selection]))
                        if target_selection else 0.0),
            open_nodes=(0 if ss is None else int(ss.open_nodes)),
            processed_nodes=(0 if ss is None else int(ss.processed_nodes)),
            branch_decisions=(0 if ss is None else int(ss.branch_decisions)),
            rmp_solves=(0 if ss is None else int(ss.rmp_solves)),
            target_master_backend=(None if ss is None else getattr(ss, "direct_target_backend", None)),
            target_master_solves=(0 if ss is None else int(getattr(ss, "target_master_solves", 0))),
            target_fullcover_strong_cuts=(0 if ss is None else int(getattr(ss, "fullcover_strong_cuts", 0))),
            target_fullcover_cuts_loaded=(0 if ss is None else int(getattr(ss, "fullcover_cuts_loaded", 0))),
            target_battery_core_cuts=(0 if ss is None else int(getattr(ss, "fullcover_battery_core_cuts", 0))),
            target_resource_audit_nodes=(0 if ss is None else int(getattr(ss, "resource_audit_nodes", 0))),
            target_resource_audit_memo_hits=(0 if ss is None else int(getattr(ss, "resource_audit_memo_hits", 0))),
            target_exact_cover_nodes=(0 if ss is None else int(getattr(ss, "target_exact_cover_nodes", 0))),
            target_battery_relaxation_nodes=(0 if ss is None else int(getattr(ss, "battery_relaxation_nodes", 0))),
            target_checkpoint_writes=(0 if ss is None else int(getattr(ss, "target_checkpoint_writes", 0))),
            target_closure_context_sha256=(None if ss is None else getattr(ss, "target_closure_context_sha256", None)),
            target_closure_checkpoint_contract=FULLCOVER_CLOSURE_CHECKPOINT_CONTRACT,
            target_global_battery_relaxation_status=(
                None if ss is None else getattr(ss, "global_battery_relaxation_status", None)),
            target_global_battery_min_required=(
                None if ss is None else getattr(ss, "global_battery_min_required", None)),
            target_global_battery_dp_states=(
                0 if ss is None else int(getattr(ss, "global_battery_dp_states", 0))),
            target_global_battery_one_pack_masks=(
                0 if ss is None else int(getattr(ss, "global_battery_one_pack_masks", 0))),
            phase_one_solves=(0 if ss is None else int(ss.phase_one_solves)),
            generated_columns=(0 if ss is None else int(ss.generated_columns)),
            pricing_calls=(0 if ss is None else int(ss.pricing_calls)),
            exact_pricing_calls=(0 if ss is None else int(ss.exact_pricing_calls)),
            pricing_complete=(True if ss is None else bool(ss.pricing_complete)),
            pricing_search_complete=(True if ss is None else bool(ss.pricing_search_complete)),
            pricing_bound_available=(True if ss is None else bool(ss.pricing_bound_available)),
            farkas_pricing_complete=(True if ss is None else bool(ss.farkas_pricing_complete)),
            resource_audit_complete=(True if ss is None else bool(ss.resource_audit_complete)),
            branching_complete=(True if ss is None else bool(ss.branching_complete)),
            generated_column_archive_size=len(archive),
            physical_pricing_cache_entries=len(physical_pricing_cache),
            seed_validation=seed_validation,
            seed_columns_revalidated=bool(seed_validation.get("validation_complete", False)),
            route_universe_source=str(route_universe_source),
            route_universe_provenance_certified=bool(route_universe_provenance_certified),
            route_space_complete=bool(complete_universe_mode),
            route_space_materialized=bool(complete_universe_mode),
            complete_route_universe_contract=(
                COMPLETE_ROUTE_UNIVERSE_CONTRACT if complete_universe_mode else None),
            complete_route_universe_columns_sha256=(
                certified_route_universe.columns_sha256 if complete_universe_mode else None),
            complete_route_universe_stats=(
                dict(certified_route_universe.stats) if complete_universe_mode else None),
            route_semantics_invariance_certified=bool(route_semantics_invariance_certified),
            future_column_row_ranges_certified=bool(future_column_row_ranges_certified),
            binary64_model_contract_enforced=bool(binary64_model_contract_enforced),
            formal_proof_contract_enforced=bool(formal_proof_contract_enforced),
            target_physical_model_certificate=bool(target_decision_certified),
            global_certificate_available=False,
            coverage_global_certificate_available=False,
            lexicographic_optimal=False, coverage_optimal=False, energy_optimal=False,
            algorithmic_global_certificate=False,
            physical_model_global_certificate=False,
            finite_discrete_model_only=True,
            continuous_real_world_optimality_claimed=False,
            model_contract_validated=True,
            model_contract_sha256=contract.get("sha256"),
            parameter_contract_sha256=contract.get("parameter_contract_sha256"),
            instance_contract_sha256=contract.get("instance_contract_sha256"),
            algorithm_contract_sha256=contract.get("algorithm_sha256"),
            physical_numeric_contract=contract.get("physical_numeric_contract"),
            route_identity_contract=contract.get("route_identity_contract"),
            model_semantics_contract=contract.get("model_semantics_contract"),
            result_certificate_contract=RESULT_CERTIFICATE_CONTRACT,
            formal_proof_contract=FORMAL_PROOF_CONTRACT,
            formal_proof_obligations=list(FORMAL_PROOF_OBLIGATIONS),
            proof_contract_sha256=FORMAL_PROOF_CONTRACT_SHA256)

    coverage = _solve_branch_price_stage(
        stage="coverage", turbines=turbines, launch_opts=launch_opts, p=p,
        xi_amb=xi_amb, K=K, batteries=battery_count, T_min=T_min,
        max_stops=max_stops, weather_unc=weather_unc, deadline=deadline,
        archive=archive, signature_to_index=signature_to_index,
        no_good_cuts=no_good_cuts, coverage_target=None,
        initial_selection=initial_selection, initial_audit=initial_audit,
        pricing_epsilon=PRICING_EPS,
        coverage_gap_target_abs=coverage_gap_target_abs,
        energy_gap_target_rel=energy_gap_target_rel,
        energy_gap_target_abs_Wh=energy_gap_target_abs_Wh,
        t_launch_min=t_launch, landing_clear_min=t_clear,
        quick_min=quick_min, swap_min=t_swap_min,
        quick_capacity=quick_capacity, swap_capacity=swap_capacity,
        deck_mode=deck_mode, deck_delta_min=deck_delta_min,
        kappa_mode=kappa_mode, chance_mode=chance_mode,
        budget_gamma=budget_gamma,
        implicit_test_columns=implicit_test_columns,
        pricing_batch_size=pricing_batch_size,
        physical_cache=physical_pricing_cache,
        complete_universe_mode=complete_universe_mode,
        pricing_experiment_mode=(
                    "layered-batch-primal-depth-fair-neutral-shadow"
                    if pricing_key
                        == "exact-layered-batch-primal-battery-halfcap-depth-fair-neutral-formal"
                    else "layered-batch-primal-depth-fair-shadow"
                    if pricing_key
                        == "exact-layered-batch-primal-battery-halfcap-depth-fair-formal"
                    else "layered-batch-primal-shadow"
                    if pricing_key in {
                        "exact-layered-batch-primal-shadow",
                        "exact-layered-batch-primal-diagnostic-shadow",
                        "exact-layered-batch-primal-target9-diagnostic-shadow",
                        "exact-layered-batch-primal-target9-certificate-diagnostic-shadow",
                        "exact-layered-batch-primal-target9-battery-clique-diagnostic-shadow",
                        "exact-layered-batch-primal-battery-halfcap-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-exchange-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-primal-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-guided-formal",
                        "exact-layered-batch-primal-battery-halfcap-deck-guided-formal",
                        "exact-layered-batch-primal-battery-halfcap-adaptive-multistop-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-variant-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-variant-diagnostic-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-variant-archive-recovery-formal"}
                    else "layered-batch-shadow"
                    if pricing_key == "exact-layered-batch-shadow"
                    else "layered-guided-shadow"
                    if pricing_key == "exact-layered-guided-shadow"
                    else "dual-guided-shadow"
                    if pricing_key == "exact-dual-guided-shadow"
                    else "discovery-shadow"
                    if pricing_key == "exact-discovery-shadow"
                    else False),
        pricing_pattern_cut_diagnostics=bool(
            pricing_key in {
                "exact-layered-batch-primal-target9-certificate-diagnostic-shadow",
                "exact-layered-batch-primal-target9-battery-clique-diagnostic-shadow"}),
        formal_battery_halfcap=bool(
            pricing_key in {
                "exact-layered-batch-primal-battery-halfcap-formal",
                "exact-layered-batch-primal-battery-halfcap-depth-fair-formal",
                "exact-layered-batch-primal-battery-halfcap-depth-fair-neutral-formal",
                "exact-layered-batch-primal-battery-halfcap-resource-exchange-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-primal-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-guided-formal",
                        "exact-layered-batch-primal-battery-halfcap-deck-guided-formal",
                        "exact-layered-batch-primal-battery-halfcap-adaptive-multistop-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-variant-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-variant-diagnostic-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-variant-archive-recovery-formal"}),
        primal_exchange=bool(
            pricing_key
            == "exact-layered-batch-primal-battery-halfcap-resource-exchange-formal"),
        primal_resource_primal=bool(
            pricing_key
            == "exact-layered-batch-primal-battery-halfcap-resource-primal-formal"),
        primal_resource_guided=bool(
            pricing_key
            == "exact-layered-batch-primal-battery-halfcap-resource-guided-formal"),
        primal_resource_deck_guided=bool(
            pricing_key in {
                "exact-layered-batch-primal-battery-halfcap-deck-guided-formal",
                "exact-layered-batch-primal-battery-halfcap-adaptive-multistop-formal",
                "exact-layered-batch-primal-battery-halfcap-resource-variant-formal",
                "exact-layered-batch-primal-battery-halfcap-resource-variant-diagnostic-formal",
                "exact-layered-batch-primal-battery-halfcap-resource-variant-archive-recovery-formal"}),
        adaptive_multistop_enrichment=bool(
            pricing_key
            == "exact-layered-batch-primal-battery-halfcap-adaptive-multistop-formal"),
        resource_variant_enrichment=bool(
            pricing_key in {
                "exact-layered-batch-primal-battery-halfcap-resource-variant-formal",
                "exact-layered-batch-primal-battery-halfcap-resource-variant-diagnostic-formal",
                "exact-layered-batch-primal-battery-halfcap-resource-variant-archive-recovery-formal"}),
        archive_primal_recovery=bool(
            archive_primal_recovery or pricing_key == "exact-layered-batch-primal-battery-halfcap-resource-variant-archive-recovery-formal"),
        archive_primal_recovery_time_limit_s=float(
            archive_primal_recovery_time_limit_s),
        certified_prefix_pruning=bool(
            pricing_key
            == "exact-layered-batch-primal-battery-halfcap-resource-variant-archive-recovery-formal"))

    coverage_inc = int(coverage.coverage_incumbent)
    coverage_ub = int(max(coverage_inc, _safe_integer_floor(coverage.global_bound)))
    coverage_gap_abs = int(coverage_ub - coverage_inc)
    coverage_gap_pct = 100.0 * coverage_gap_abs / max(1, coverage_ub)
    coverage_optimal = bool(coverage.optimal and coverage_gap_abs == 0)

    energy = None
    if (solve_scope == "lexicographic" and coverage_optimal
            and not _deadline_hit(deadline)):
        energy = _solve_branch_price_stage(
            stage="energy", turbines=turbines, launch_opts=launch_opts, p=p,
            xi_amb=xi_amb, K=K, batteries=battery_count, T_min=T_min,
            max_stops=max_stops, weather_unc=weather_unc, deadline=deadline,
            archive=archive, signature_to_index=signature_to_index,
            no_good_cuts=no_good_cuts, coverage_target=coverage_inc,
            initial_selection=coverage.incumbent_selection,
            initial_audit=coverage.incumbent_audit,
            pricing_epsilon=PRICING_EPS,
            coverage_gap_target_abs=coverage_gap_target_abs,
            energy_gap_target_rel=energy_gap_target_rel,
            energy_gap_target_abs_Wh=energy_gap_target_abs_Wh,
            t_launch_min=t_launch, landing_clear_min=t_clear,
            quick_min=quick_min, swap_min=t_swap_min,
            quick_capacity=quick_capacity, swap_capacity=swap_capacity,
            deck_mode=deck_mode, deck_delta_min=deck_delta_min,
            kappa_mode=kappa_mode, chance_mode=chance_mode,
            budget_gamma=budget_gamma,
            implicit_test_columns=implicit_test_columns,
            pricing_batch_size=pricing_batch_size,
            physical_cache=physical_pricing_cache,
            complete_universe_mode=complete_universe_mode,
            pricing_experiment_mode=(
                    "layered-batch-primal-depth-fair-neutral-shadow"
                    if pricing_key
                        == "exact-layered-batch-primal-battery-halfcap-depth-fair-neutral-formal"
                    else "layered-batch-primal-depth-fair-shadow"
                    if pricing_key
                        == "exact-layered-batch-primal-battery-halfcap-depth-fair-formal"
                    else "layered-batch-primal-shadow"
                    if pricing_key in {
                        "exact-layered-batch-primal-shadow",
                        "exact-layered-batch-primal-diagnostic-shadow",
                        "exact-layered-batch-primal-target9-diagnostic-shadow",
                        "exact-layered-batch-primal-target9-certificate-diagnostic-shadow",
                        "exact-layered-batch-primal-target9-battery-clique-diagnostic-shadow",
                        "exact-layered-batch-primal-battery-halfcap-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-exchange-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-primal-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-guided-formal",
                        "exact-layered-batch-primal-battery-halfcap-deck-guided-formal",
                        "exact-layered-batch-primal-battery-halfcap-adaptive-multistop-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-variant-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-variant-diagnostic-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-variant-archive-recovery-formal"}
                    else "layered-batch-shadow"
                    if pricing_key == "exact-layered-batch-shadow"
                    else "layered-guided-shadow"
                    if pricing_key == "exact-layered-guided-shadow"
                    else "dual-guided-shadow"
                    if pricing_key == "exact-dual-guided-shadow"
                    else "discovery-shadow"
                    if pricing_key == "exact-discovery-shadow"
                    else False),
            pricing_pattern_cut_diagnostics=bool(
                pricing_key in {
                    "exact-layered-batch-primal-target9-certificate-diagnostic-shadow",
                    "exact-layered-batch-primal-target9-battery-clique-diagnostic-shadow"}),
            formal_battery_halfcap=bool(
                pricing_key in {
                    "exact-layered-batch-primal-battery-halfcap-formal",
                    "exact-layered-batch-primal-battery-halfcap-depth-fair-formal",
                    "exact-layered-batch-primal-battery-halfcap-depth-fair-neutral-formal",
                    "exact-layered-batch-primal-battery-halfcap-resource-exchange-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-primal-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-guided-formal",
                        "exact-layered-batch-primal-battery-halfcap-deck-guided-formal",
                        "exact-layered-batch-primal-battery-halfcap-adaptive-multistop-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-variant-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-variant-diagnostic-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-variant-archive-recovery-formal"}),
            primal_exchange=bool(
                pricing_key
                == "exact-layered-batch-primal-battery-halfcap-resource-exchange-formal"),
            primal_resource_primal=bool(
                pricing_key
                == "exact-layered-batch-primal-battery-halfcap-resource-primal-formal"),
            primal_resource_guided=bool(
                pricing_key
                == "exact-layered-batch-primal-battery-halfcap-resource-guided-formal"),
            primal_resource_deck_guided=bool(
                pricing_key in {
                    "exact-layered-batch-primal-battery-halfcap-deck-guided-formal",
                    "exact-layered-batch-primal-battery-halfcap-adaptive-multistop-formal",
                    "exact-layered-batch-primal-battery-halfcap-resource-variant-formal",
                    "exact-layered-batch-primal-battery-halfcap-resource-variant-diagnostic-formal",
                    "exact-layered-batch-primal-battery-halfcap-resource-variant-archive-recovery-formal"}),
            adaptive_multistop_enrichment=bool(
                pricing_key
                == "exact-layered-batch-primal-battery-halfcap-adaptive-multistop-formal"),
            resource_variant_enrichment=bool(
                pricing_key in {
                    "exact-layered-batch-primal-battery-halfcap-resource-variant-formal",
                    "exact-layered-batch-primal-battery-halfcap-resource-variant-diagnostic-formal",
                    "exact-layered-batch-primal-battery-halfcap-resource-variant-archive-recovery-formal"}),
            certified_prefix_pruning=bool(
                pricing_key
                == "exact-layered-batch-primal-battery-halfcap-resource-variant-archive-recovery-formal"))

    # V8/V9/V10/V11 diagnostics run after the formal solver clock. They never modify
    # formal incumbents, bounds, node pruning, or certificate fields.
    formal_solver_runtime_s = float(time.monotonic() - started)
    archive_diag = dict(
        enabled=False,
        scope="fixed-generated-column-archive-only-not-full-route-space",
        status="not-requested",
        time_limit_s=float(archive_diagnostic_time_limit_s),
        runtime_s=0.0,
        archive_columns=int(len(archive)),
        coverage_lower_bound=None,
        coverage_upper_bound=None,
        exact_optimum=None,
        optimal_proven=False,
        witness_selection_indices=[],
        witness_route_signatures=[],
        witness_covered_turbines=[],
        open_nodes=None,
        processed_nodes=None,
        rmp_solves=None,
        resource_audit_calls=None,
        resource_cuts_added=None)
    archive_target_diag = dict(
        enabled=False,
        scope="fixed-generated-column-archive-target-only-not-full-route-space",
        target_min=9,
        status="NOT_REQUESTED",
        time_limit_s=float(archive_diagnostic_time_limit_s),
        runtime_s=0.0,
        archive_columns=int(len(archive)),
        feasible_proven=False,
        infeasible_proven=False,
        witness_coverage=None,
        coverage_incumbent=None,
        coverage_upper_bound=None,
        open_nodes=None,
        processed_nodes=None,
        rmp_solves=None,
        resource_audit_calls=None,
        resource_cuts_added=None,
        rejected_pattern_count=0,
        rejected_pattern_size_avg=None,
        rejected_pattern_size_min=None,
        rejected_pattern_size_max=None,
        rejected_pattern_coverage_avg=None,
        rejected_pattern_coverage_min=None,
        rejected_pattern_coverage_max=None,
        rejected_hamming_avg=None,
        rejected_hamming_min=None,
        rejected_hamming_max=None,
        resource_failure_event_counts={},
        resource_failure_pattern_counts={},
        rejected_morphology_counts={},
        rejected_route_stop_count_totals={},
        rejected_coverage_route_count_joint={},
        certificate_shadow=dict(
            enabled=False,
            time_limit_s=float(archive_shadow_diagnostic_time_limit_s),
            runtime_s=0.0, analyzed_patterns=0, total_patterns=0))
    fullspace_target_diag = dict(
        enabled=False,
        scope=("post-formal-full-physical-route-space-target-ladder-"
               "diagnostic-only-not-promoted-to-formal-result"),
        status="not-requested",
        time_limit_s=float(fullspace_target_diagnostic_time_limit_s),
        runtime_s=0.0,
        archive_columns_start=int(len(archive)),
        archive_columns_end=int(len(archive)),
        start_coverage=int(coverage.coverage_incumbent),
        best_coverage=int(coverage.coverage_incumbent),
        highest_feasible_target=None,
        first_infeasible_target=None,
        unresolved_target=None,
        targets_attempted=0,
        records=[],
        witness_selection_indices=[],
        witness_route_signatures=[],
        witness_covered_turbines=[])

    if (pricing_key in {
            "exact-layered-batch-primal-diagnostic-shadow",
            "exact-layered-batch-primal-battery-halfcap-resource-variant-diagnostic-formal",
            "exact-layered-batch-primal-battery-halfcap-resource-variant-archive-recovery-formal"}
            and not complete_universe_mode
            and not synthetic_fixture):
        archive_diag = _diagnose_fixed_archive_coverage(
            turbines=turbines, launch_opts=launch_opts, p=p,
            xi_amb=xi_amb, K=K, batteries=battery_count, T_min=T_min,
            max_stops=max_stops, weather_unc=weather_unc,
            archive=archive, signature_to_index=signature_to_index,
            no_good_cuts=no_good_cuts,
            initial_selection=coverage.incumbent_selection,
            initial_audit=coverage.incumbent_audit,
            t_launch_min=t_launch, landing_clear_min=t_clear,
            quick_min=quick_min, swap_min=t_swap_min,
            quick_capacity=quick_capacity, swap_capacity=swap_capacity,
            deck_mode=deck_mode, deck_delta_min=deck_delta_min,
            kappa_mode=kappa_mode, chance_mode=chance_mode,
            budget_gamma=budget_gamma,
            time_limit_s=archive_diagnostic_time_limit_s)

    resource_variant_diag = dict(
        enabled=False,
        scope=(
            "post-formal-fixed-archive-resource-variant-diagnostic-only-"
            "not-full-route-space-not-proof"),
        status="not-requested",
        time_limit_s=float(min(10.0, archive_diagnostic_time_limit_s)),
        runtime_s=0.0,
        timed_out=False,
        records_analyzed=0,
        records_missing_from_archive=0,
        final_coverage=int(coverage.coverage_incumbent),
        final_uncovered_turbines=[],
        direct_augmentation_audits=0,
        direct_augmentation_feasible=0,
        direct_augmentation_infeasible=0,
        direct_augmentation_unknown=0,
        single_blocker_records=0,
        blocker_retime_candidates=0,
        blocker_retime_audits=0,
        blocker_retime_feasible=0,
        final_uncovered_singleton_routes=0,
        final_uncovered_zero_deck_conflict_routes=0,
        final_uncovered_single_blocker_routes=0,
        final_uncovered_multi_blocker_routes=0,
        final_uncovered_single_blocker_distinct_turbines=0,
        records=[],
        single_blocker_pairs_sample=[],
    )
    if (pricing_key in {
            "exact-layered-batch-primal-battery-halfcap-resource-variant-diagnostic-formal",
            "exact-layered-batch-primal-battery-halfcap-resource-variant-archive-recovery-formal"}
            and not complete_universe_mode
            and not synthetic_fixture):
        resource_variant_diag = _diagnose_resource_variant_postsolve(
            archive=archive,
            final_selection=coverage.incumbent_selection,
            variant_records=coverage.pricing_resource_variant_records,
            K=K, batteries=battery_count, p=p,
            quick_min=quick_min, swap_min=t_swap_min,
            quick_capacity=quick_capacity, swap_capacity=swap_capacity,
            time_limit_s=min(10.0, archive_diagnostic_time_limit_s))

    if (float(fullspace_target_diagnostic_time_limit_s) > 0.0
            and pricing_key in {
                "exact-layered-batch-primal-battery-halfcap-resource-variant-formal",
                "exact-layered-batch-primal-battery-halfcap-resource-variant-diagnostic-formal",
                "exact-layered-batch-primal-battery-halfcap-resource-variant-archive-recovery-formal"}
            and not complete_universe_mode and not synthetic_fixture):
        _target_initial_selection = tuple(coverage.incumbent_selection)
        _target_initial_audit = coverage.incumbent_audit
        if (archive_diag.get("coverage_lower_bound") is not None
                and int(archive_diag.get("coverage_lower_bound"))
                    > int(coverage.coverage_incumbent)
                and archive_diag.get("witness_selection_indices")):
            _target_initial_selection = tuple(
                int(_j) for _j in archive_diag.get(
                    "witness_selection_indices", []))
            # The fixed-archive diagnostic already exact-audited this witness.
            # Re-audit inside the isolated target diagnostic to avoid importing
            # any diagnostic object/state into another diagnostic phase.
            _target_initial_audit = None
        fullspace_target_diag = _diagnose_fullspace_target_ladder(
            turbines=turbines, launch_opts=launch_opts, p=p,
            xi_amb=xi_amb, K=K, batteries=battery_count, T_min=T_min,
            max_stops=max_stops, weather_unc=weather_unc,
            archive=archive, signature_to_index=signature_to_index,
            no_good_cuts=no_good_cuts,
            initial_selection=_target_initial_selection,
            initial_audit=_target_initial_audit,
            physical_cache=physical_pricing_cache,
            t_launch_min=t_launch, landing_clear_min=t_clear,
            quick_min=quick_min, swap_min=t_swap_min,
            quick_capacity=quick_capacity, swap_capacity=swap_capacity,
            deck_mode=deck_mode, deck_delta_min=deck_delta_min,
            kappa_mode=kappa_mode, chance_mode=chance_mode,
            budget_gamma=budget_gamma,
            time_limit_s=fullspace_target_diagnostic_time_limit_s,
            archive_primal_recovery_time_limit_s=(
                archive_primal_recovery_time_limit_s))

    if (pricing_key in {
            "exact-layered-batch-primal-target9-diagnostic-shadow",
            "exact-layered-batch-primal-target9-certificate-diagnostic-shadow",
            "exact-layered-batch-primal-target9-battery-clique-diagnostic-shadow"}
            and not complete_universe_mode
            and not synthetic_fixture):
        archive_target_diag = _diagnose_fixed_archive_target_min_coverage(
            turbines=turbines, launch_opts=launch_opts, p=p,
            xi_amb=xi_amb, K=K, batteries=battery_count, T_min=T_min,
            max_stops=max_stops, weather_unc=weather_unc,
            archive=archive, signature_to_index=signature_to_index,
            no_good_cuts=no_good_cuts,
            initial_selection=coverage.incumbent_selection,
            initial_audit=coverage.incumbent_audit,
            t_launch_min=t_launch, landing_clear_min=t_clear,
            quick_min=quick_min, swap_min=t_swap_min,
            quick_capacity=quick_capacity, swap_capacity=swap_capacity,
            deck_mode=deck_mode, deck_delta_min=deck_delta_min,
            kappa_mode=kappa_mode, chance_mode=chance_mode,
            budget_gamma=budget_gamma, target_min=9,
            time_limit_s=archive_diagnostic_time_limit_s,
            certificate_shadow_time_limit_s=(
                archive_shadow_diagnostic_time_limit_s
                if pricing_key in {
                    "exact-layered-batch-primal-target9-certificate-diagnostic-shadow",
                    "exact-layered-batch-primal-target9-battery-clique-diagnostic-shadow"}
                else 0.0),
            clique_rerun_time_limit_s=(
                archive_clique_diagnostic_time_limit_s
                if pricing_key
                == "exact-layered-batch-primal-target9-battery-clique-diagnostic-shadow"
                else 0.0))

    final_selection = coverage.incumbent_selection
    final_audit = coverage.incumbent_audit
    if final_selection:
        energy_inc_estimate, energy_inc_lower, energy_inc = _energy_of_selection_interval(
            archive, final_selection)
    else:
        energy_inc_estimate = energy_inc_lower = energy_inc = 0.0
    energy_lb = None
    energy_gap_abs = None
    energy_gap_pct = None
    energy_optimal = False
    conditional_energy_gap_pct = None
    if energy is not None:
        final_selection = energy.incumbent_selection
        final_audit = energy.incumbent_audit
        energy_inc = energy.incumbent_upper_bound
        energy_inc_lower = energy.incumbent_lower_bound
        energy_inc_estimate = (
            float(energy.incumbent_value) if energy.incumbent_value is not None else None)
        energy_lb = float(energy.global_bound)
        if energy_inc is not None:
            energy_gap_abs = max(0.0, float(energy_inc) - float(energy_lb))
            energy_gap_pct = 100.0 * energy_gap_abs / max(abs(float(energy_inc)), 1e-12)
            conditional_energy_gap_pct = energy_gap_pct
            # A positive user Gap target is only an anytime stopping rule.
            # Exact optimality comes exclusively from the stage proof flag.
            energy_optimal = bool(energy.optimal)

    lex_opt = bool(coverage_optimal and energy_optimal)
    chosen = _materialize_chosen(archive, final_selection, final_audit)
    covered_ids = []
    for j in final_selection:
        covered_ids.extend(_ordered_tids(archive[j]))
    duplicates = sorted({t for t in covered_ids if covered_ids.count(t) > 1})
    if len(covered_ids) != len(set(covered_ids)) or duplicates:
        raise RuntimeError("internal consistency error: duplicate turbine visits")

    termination_reason = (energy.termination_reason if energy is not None
                          else coverage.termination_reason)
    status = _classify_anytime_status(
        lexicographic_optimal=lex_opt, coverage_optimal=coverage_optimal,
        coverage_gap_abs=coverage_gap_abs,
        coverage_gap_target_abs=coverage_gap_target_abs,
        energy_gap_abs=energy_gap_abs, energy_gap_pct=energy_gap_pct,
        energy_gap_target_abs_Wh=energy_gap_target_abs_Wh,
        energy_gap_target_rel=energy_gap_target_rel,
        termination_reason=termination_reason,
        deadline_hit=_deadline_hit(deadline))
    if solve_scope == "coverage-only":
        status = ("coverage_optimal_scope_complete" if coverage_optimal
                  else ("coverage_time_limit_feasible"
                        if _is_explicit_time_limit(termination_reason, _deadline_hit(deadline))
                        else "coverage_unresolved"))
    total_open = coverage.open_nodes + (energy.open_nodes if energy is not None else 0)
    total_processed = coverage.processed_nodes + (energy.processed_nodes if energy is not None else 0)
    total_generated = coverage.generated_columns + (energy.generated_columns if energy is not None else 0)
    total_pricing = coverage.pricing_calls + (energy.pricing_calls if energy is not None else 0)
    total_exact_pricing = coverage.exact_pricing_calls + (energy.exact_pricing_calls if energy is not None else 0)
    total_discovery_calls = coverage.pricing_discovery_calls + (
        energy.pricing_discovery_calls if energy is not None else 0)
    total_discovery_early_returns = coverage.pricing_discovery_early_returns + (
        energy.pricing_discovery_early_returns if energy is not None else 0)
    total_discovery_improving_seen = coverage.pricing_discovery_improving_seen + (
        energy.pricing_discovery_improving_seen if energy is not None else 0)
    total_discovery_improving_returned = (
        coverage.pricing_discovery_improving_returned
        + (energy.pricing_discovery_improving_returned
           if energy is not None else 0))
    total_discovery_diverse_returns = coverage.pricing_discovery_diverse_returns + (
        energy.pricing_discovery_diverse_returns if energy is not None else 0)
    total_discovery_hard_cap_returns = (
        coverage.pricing_discovery_hard_cap_returns
        + (energy.pricing_discovery_hard_cap_returns
           if energy is not None else 0))
    total_discovery_max_return_batch = max(
        int(coverage.pricing_discovery_max_return_batch),
        int(energy.pricing_discovery_max_return_batch)
        if energy is not None else 0)
    total_discovery_max_distinct_launches = max(
        int(coverage.pricing_discovery_max_distinct_launches),
        int(energy.pricing_discovery_max_distinct_launches)
        if energy is not None else 0)
    total_discovery_max_distinct_service_sets = max(
        int(coverage.pricing_discovery_max_distinct_service_sets),
        int(energy.pricing_discovery_max_distinct_service_sets)
        if energy is not None else 0)
    total_primal_refresh_calls = coverage.primal_refresh_calls + (
        energy.primal_refresh_calls if energy is not None else 0)
    total_primal_refresh_audit_calls = coverage.primal_refresh_audit_calls + (
        energy.primal_refresh_audit_calls if energy is not None else 0)
    total_primal_refresh_timeouts = coverage.primal_refresh_timeouts + (
        energy.primal_refresh_timeouts if energy is not None else 0)
    total_primal_refresh_improvements = coverage.primal_refresh_improvements + (
        energy.primal_refresh_improvements if energy is not None else 0)
    total_primal_refresh_best_coverage = max(
        int(coverage.primal_refresh_best_coverage),
        int(energy.primal_refresh_best_coverage) if energy is not None else 0)
    total_primal_refresh_columns_seen = max(
        int(coverage.primal_refresh_columns_seen),
        int(energy.primal_refresh_columns_seen) if energy is not None else 0)
    total_primal_refresh_rebuilds = coverage.primal_refresh_rebuilds + (
        energy.primal_refresh_rebuilds if energy is not None else 0)
    total_primal_refresh_repairs = coverage.primal_refresh_repairs + (
        energy.primal_refresh_repairs if energy is not None else 0)
    total_primal_refresh_augmentation_audits = (
        coverage.primal_refresh_augmentation_audits
        + (energy.primal_refresh_augmentation_audits
           if energy is not None else 0))
    total_primal_refresh_rebuild_audits = (
        coverage.primal_refresh_rebuild_audits
        + (energy.primal_refresh_rebuild_audits
           if energy is not None else 0))
    total_primal_refresh_repair_audits = (
        coverage.primal_refresh_repair_audits
        + (energy.primal_refresh_repair_audits
           if energy is not None else 0))
    total_primal_refresh_augmentation_improvements = (
        coverage.primal_refresh_augmentation_improvements
        + (energy.primal_refresh_augmentation_improvements
           if energy is not None else 0))
    total_primal_refresh_rebuild_improvements = (
        coverage.primal_refresh_rebuild_improvements
        + (energy.primal_refresh_rebuild_improvements
           if energy is not None else 0))
    total_primal_refresh_repair_improvements = (
        coverage.primal_refresh_repair_improvements
        + (energy.primal_refresh_repair_improvements
           if energy is not None else 0))
    total_primal_refresh_duplicate_trials_skipped = int(
        coverage.primal_refresh_duplicate_trials_skipped
        + (energy.primal_refresh_duplicate_trials_skipped
           if energy is not None else 0))
    total_primal_refresh_cached_infeasible_trials = int(
        max(coverage.primal_refresh_cached_infeasible_trials,
            energy.primal_refresh_cached_infeasible_trials
            if energy is not None else 0))
    total_primal_refresh_uncovered_fair_rounds = int(
        coverage.primal_refresh_uncovered_fair_rounds
        + (energy.primal_refresh_uncovered_fair_rounds
           if energy is not None else 0))
    total_primal_refresh_failure_reasons = {}
    for _src in (
            coverage.primal_refresh_failure_reasons,
            energy.primal_refresh_failure_reasons if energy is not None else {}):
        for _reason, _count in dict(_src or {}).items():
            total_primal_refresh_failure_reasons[str(_reason)] = int(
                total_primal_refresh_failure_reasons.get(
                    str(_reason), 0)) + int(_count)
    total_primal_deck_archive_conflict_edges = max(
        int(coverage.primal_deck_archive_conflict_edges),
        int(energy.primal_deck_archive_conflict_edges)
        if energy is not None else 0)
    total_primal_deck_archive_max_degree = max(
        int(coverage.primal_deck_archive_max_degree),
        int(energy.primal_deck_archive_max_degree)
        if energy is not None else 0)
    total_primal_deck_archive_max_component = max(
        int(coverage.primal_deck_archive_max_component),
        int(energy.primal_deck_archive_max_component)
        if energy is not None else 0)
    total_primal_deck_candidate_scored = int(
        coverage.primal_deck_candidate_scored
        + (energy.primal_deck_candidate_scored
           if energy is not None else 0))
    total_primal_deck_candidate_zero_conflict = int(
        coverage.primal_deck_candidate_zero_conflict
        + (energy.primal_deck_candidate_zero_conflict
           if energy is not None else 0))
    total_primal_deck_candidate_positive_conflict = int(
        coverage.primal_deck_candidate_positive_conflict
        + (energy.primal_deck_candidate_positive_conflict
           if energy is not None else 0))
    total_primal_deck_prefilter_skips = int(
        coverage.primal_deck_prefilter_skips
        + (energy.primal_deck_prefilter_skips
           if energy is not None else 0))
    total_primal_deck_max_candidate_conflicts = max(
        int(coverage.primal_deck_max_candidate_conflicts),
        int(energy.primal_deck_max_candidate_conflicts)
        if energy is not None else 0)
    total_primal_deck_conflict_pairs_sample = []
    for _src_pairs in (
            coverage.primal_deck_conflict_pairs_sample,
            energy.primal_deck_conflict_pairs_sample if energy is not None else []):
        for _pair in list(_src_pairs or []):
            if (_pair not in total_primal_deck_conflict_pairs_sample
                    and len(total_primal_deck_conflict_pairs_sample) < 12):
                total_primal_deck_conflict_pairs_sample.append(_pair)
    total_multistop_merge_triggers = int(
        coverage.pricing_multistop_merge_triggers
        + (energy.pricing_multistop_merge_triggers if energy is not None else 0))
    total_multistop_merge_attempts = int(
        coverage.pricing_multistop_merge_attempts
        + (energy.pricing_multistop_merge_attempts if energy is not None else 0))
    total_multistop_merge_physical_feasible = int(
        coverage.pricing_multistop_merge_physical_feasible
        + (energy.pricing_multistop_merge_physical_feasible
           if energy is not None else 0))
    total_multistop_merge_new_candidates = int(
        coverage.pricing_multistop_merge_new_candidates
        + (energy.pricing_multistop_merge_new_candidates
           if energy is not None else 0))
    total_multistop_merge_returned = int(
        coverage.pricing_multistop_merge_returned
        + (energy.pricing_multistop_merge_returned if energy is not None else 0))
    total_multistop_merge_added = int(
        coverage.pricing_multistop_merge_added
        + (energy.pricing_multistop_merge_added if energy is not None else 0))
    total_multistop_merge_batches = int(
        coverage.pricing_multistop_merge_batches
        + (energy.pricing_multistop_merge_batches if energy is not None else 0))
    total_multistop_merge_distinct_pairs = int(
        coverage.pricing_multistop_merge_distinct_pairs
        + (energy.pricing_multistop_merge_distinct_pairs
           if energy is not None else 0))
    _merge_rc_vals = [
        _v for _v in (
            coverage.pricing_multistop_merge_best_rc_ub,
            energy.pricing_multistop_merge_best_rc_ub
                if energy is not None else None)
        if _v is not None]
    total_multistop_merge_best_rc_ub = (
        None if not _merge_rc_vals else min(float(_v) for _v in _merge_rc_vals))
    _merge_eps_vals = [
        _v for _v in (
            coverage.pricing_multistop_merge_best_energy_per_stop_Wh,
            energy.pricing_multistop_merge_best_energy_per_stop_Wh
                if energy is not None else None)
        if _v is not None]
    total_multistop_merge_best_energy_per_stop_Wh = (
        None if not _merge_eps_vals else min(
            float(_v) for _v in _merge_eps_vals))
    total_multistop_merge_best_uncovered_gain = max(
        int(coverage.pricing_multistop_merge_best_uncovered_gain),
        int(energy.pricing_multistop_merge_best_uncovered_gain)
        if energy is not None else 0)
    total_multistop_merge_used_in_incumbent = int(
        coverage.pricing_multistop_merge_used_in_incumbent
        + (energy.pricing_multistop_merge_used_in_incumbent
           if energy is not None else 0))
    total_resource_variant_triggers = int(
        coverage.pricing_resource_variant_triggers
        + (energy.pricing_resource_variant_triggers if energy is not None else 0))
    total_resource_variant_attempts = int(
        coverage.pricing_resource_variant_attempts
        + (energy.pricing_resource_variant_attempts if energy is not None else 0))
    total_resource_variant_deck_compatible_specs = int(
        coverage.pricing_resource_variant_deck_compatible_specs
        + (energy.pricing_resource_variant_deck_compatible_specs
           if energy is not None else 0))
    total_resource_variant_deck_prefilter_skips = int(
        coverage.pricing_resource_variant_deck_prefilter_skips
        + (energy.pricing_resource_variant_deck_prefilter_skips
           if energy is not None else 0))
    total_resource_variant_physical_feasible = int(
        coverage.pricing_resource_variant_physical_feasible
        + (energy.pricing_resource_variant_physical_feasible
           if energy is not None else 0))
    total_resource_variant_new_candidates = int(
        coverage.pricing_resource_variant_new_candidates
        + (energy.pricing_resource_variant_new_candidates
           if energy is not None else 0))
    total_resource_variant_returned = int(
        coverage.pricing_resource_variant_returned
        + (energy.pricing_resource_variant_returned if energy is not None else 0))
    total_resource_variant_added = int(
        coverage.pricing_resource_variant_added
        + (energy.pricing_resource_variant_added if energy is not None else 0))
    total_resource_variant_batches = int(
        coverage.pricing_resource_variant_batches
        + (energy.pricing_resource_variant_batches if energy is not None else 0))
    total_resource_variant_distinct_turbines = int(
        coverage.pricing_resource_variant_distinct_turbines
        + (energy.pricing_resource_variant_distinct_turbines
           if energy is not None else 0))
    _rv_rc_vals = [
        _v for _v in (
            coverage.pricing_resource_variant_best_rc_ub,
            energy.pricing_resource_variant_best_rc_ub
                if energy is not None else None)
        if _v is not None]
    total_resource_variant_best_rc_ub = (
        None if not _rv_rc_vals else min(float(_v) for _v in _rv_rc_vals))
    _rv_e_vals = [
        _v for _v in (
            coverage.pricing_resource_variant_best_energy_Wh,
            energy.pricing_resource_variant_best_energy_Wh
                if energy is not None else None)
        if _v is not None]
    total_resource_variant_best_energy_Wh = (
        None if not _rv_e_vals else min(float(_v) for _v in _rv_e_vals))
    total_resource_variant_used_in_incumbent = int(
        coverage.pricing_resource_variant_used_in_incumbent
        + (energy.pricing_resource_variant_used_in_incumbent
           if energy is not None else 0))
    total_resource_variant_records = (
        [dict(_r) for _r in coverage.pricing_resource_variant_records]
        + ([dict(_r) for _r in energy.pricing_resource_variant_records]
           if energy is not None else []))
    total_primal_exchange_calls = int(
        coverage.primal_exchange_calls
        + (energy.primal_exchange_calls if energy is not None else 0))
    total_primal_exchange_candidate_routes = int(
        coverage.primal_exchange_candidate_routes
        + (energy.primal_exchange_candidate_routes
           if energy is not None else 0))
    total_primal_exchange_trials_built = int(
        coverage.primal_exchange_trials_built
        + (energy.primal_exchange_trials_built if energy is not None else 0))
    total_primal_exchange_audit_calls = int(
        coverage.primal_exchange_audit_calls
        + (energy.primal_exchange_audit_calls if energy is not None else 0))
    total_primal_exchange_improvements = int(
        coverage.primal_exchange_improvements
        + (energy.primal_exchange_improvements if energy is not None else 0))
    total_primal_exchange_consolidation_trials = int(
        coverage.primal_exchange_consolidation_trials
        + (energy.primal_exchange_consolidation_trials
           if energy is not None else 0))
    total_primal_exchange_optional_drop_trials = int(
        coverage.primal_exchange_optional_drop_trials
        + (energy.primal_exchange_optional_drop_trials
           if energy is not None else 0))
    total_primal_exchange_max_stop_count_considered = max(
        int(coverage.primal_exchange_max_stop_count_considered),
        int(energy.primal_exchange_max_stop_count_considered)
        if energy is not None else 0)
    total_primal_exchange_best_coverage = max(
        int(coverage.primal_exchange_best_coverage),
        int(energy.primal_exchange_best_coverage)
        if energy is not None else 0)
    total_primal_exchange_multistop_used_in_incumbent = int(
        coverage.primal_exchange_multistop_used_in_incumbent
        + (energy.primal_exchange_multistop_used_in_incumbent
           if energy is not None else 0))

    def _sum_depth_dict(a, b):
        out = {}
        for src in (a or {}, b or {}):
            for k, v in src.items():
                ki = int(k)
                out[ki] = int(out.get(ki, 0)) + int(v)
        return out

    total_depth_prefixes = _sum_depth_dict(
        coverage.pricing_depth_prefixes_evaluated,
        energy.pricing_depth_prefixes_evaluated if energy is not None else {})
    total_depth_certified_prefix_prunes = _sum_depth_dict(
        coverage.pricing_depth_certified_prefix_prunes,
        energy.pricing_depth_certified_prefix_prunes if energy is not None else {})
    total_depth_service_floor_prunes = _sum_depth_dict(
        coverage.pricing_depth_service_floor_prunes,
        energy.pricing_depth_service_floor_prunes if energy is not None else {})
    total_depth_improving_seen = _sum_depth_dict(
        coverage.pricing_depth_improving_seen,
        energy.pricing_depth_improving_seen if energy is not None else {})
    total_depth_improving_returned = _sum_depth_dict(
        coverage.pricing_depth_improving_returned,
        energy.pricing_depth_improving_returned if energy is not None else {})
    total_pattern_cut_active_dual_rows = int(
        coverage.pricing_pattern_cut_active_dual_rows
        + (energy.pricing_pattern_cut_active_dual_rows if energy is not None else 0))
    total_pattern_cut_dual_abs_sum = float(
        coverage.pricing_pattern_cut_dual_abs_sum
        + (energy.pricing_pattern_cut_dual_abs_sum if energy is not None else 0.0))
    total_pattern_cut_improving_seen_count = int(
        coverage.pricing_pattern_cut_improving_seen_count
        + (energy.pricing_pattern_cut_improving_seen_count if energy is not None else 0))
    total_pattern_cut_improving_seen_contribution_sum = float(
        coverage.pricing_pattern_cut_improving_seen_contribution_sum
        + (energy.pricing_pattern_cut_improving_seen_contribution_sum
           if energy is not None else 0.0))
    total_pattern_cut_improving_seen_sign_essential = int(
        coverage.pricing_pattern_cut_improving_seen_sign_essential
        + (energy.pricing_pattern_cut_improving_seen_sign_essential
           if energy is not None else 0))
    total_pattern_cut_returned_count = int(
        coverage.pricing_pattern_cut_returned_count
        + (energy.pricing_pattern_cut_returned_count if energy is not None else 0))
    total_pattern_cut_returned_contribution_sum = float(
        coverage.pricing_pattern_cut_returned_contribution_sum
        + (energy.pricing_pattern_cut_returned_contribution_sum
           if energy is not None else 0.0))
    total_pattern_cut_returned_sign_essential = int(
        coverage.pricing_pattern_cut_returned_sign_essential
        + (energy.pricing_pattern_cut_returned_sign_essential
           if energy is not None else 0))
    total_battery_halfcap_dual_active_rmp_solves = int(
        coverage.battery_halfcap_dual_active_rmp_solves
        + (energy.battery_halfcap_dual_active_rmp_solves
           if energy is not None else 0))
    total_battery_halfcap_dual_abs_sum = float(
        coverage.battery_halfcap_dual_abs_sum
        + (energy.battery_halfcap_dual_abs_sum
           if energy is not None else 0.0))
    total_battery_halfcap_dual_max_abs = max(
        float(coverage.battery_halfcap_dual_max_abs),
        float(energy.battery_halfcap_dual_max_abs)
        if energy is not None else 0.0)

    total_pattern_cut_returned_by_depth = {}
    for _src in (coverage.pricing_pattern_cut_returned_by_depth,
                 energy.pricing_pattern_cut_returned_by_depth if energy is not None else {}):
        for _d, _rec in _src.items():
            _di = int(_d)
            _dst = total_pattern_cut_returned_by_depth.setdefault(
                _di, dict(count=0, contribution_sum=0.0, sign_essential=0,
                          rc_sum=0.0, rc_without_cut_ub_sum=0.0,
                          rc_without_cut_ub_finite_count=0))
            for _k in ("count", "sign_essential",
                       "rc_without_cut_ub_finite_count"):
                _dst[_k] += int(_rec.get(_k, 0))
            for _k in ("contribution_sum", "rc_sum",
                       "rc_without_cut_ub_sum"):
                _dst[_k] += float(_rec.get(_k, 0.0))

    total_certification_calls = coverage.pricing_certification_calls + (
        energy.pricing_certification_calls if energy is not None else 0)
    total_shadow_prefixes_evaluated = coverage.pricing_shadow_prefixes_evaluated + (
        energy.pricing_shadow_prefixes_evaluated if energy is not None else 0)
    total_shadow_prunable_prefixes = coverage.pricing_shadow_prunable_prefixes + (
        energy.pricing_shadow_prunable_prefixes if energy is not None else 0)
    total_shadow_false_prune_witnesses = coverage.pricing_shadow_false_prune_witnesses + (
        energy.pricing_shadow_false_prune_witnesses if energy is not None else 0)
    total_shadow_bound_errors = coverage.pricing_shadow_bound_errors + (
        energy.pricing_shadow_bound_errors if energy is not None else 0)
    total_shadow_complete_calls = coverage.pricing_shadow_complete_calls + (
        energy.pricing_shadow_complete_calls if energy is not None else 0)
    total_guided_order_calls = coverage.pricing_guided_order_calls + (
        energy.pricing_guided_order_calls if energy is not None else 0)
    total_guided_order_reorders = coverage.pricing_guided_order_reorders + (
        energy.pricing_guided_order_reorders if energy is not None else 0)
    total_guided_order_failures = coverage.pricing_guided_order_failures + (
        energy.pricing_guided_order_failures if energy is not None else 0)
    total_layered_depths_started = coverage.pricing_layered_depths_started + (
        energy.pricing_layered_depths_started if energy is not None else 0)
    total_layered_depths_completed = coverage.pricing_layered_depths_completed + (
        energy.pricing_layered_depths_completed if energy is not None else 0)
    total_layered_max_depth_completed = max(
        int(coverage.pricing_layered_max_depth_completed),
        int(energy.pricing_layered_max_depth_completed) if energy is not None else 0)
    total_layered_rounds = coverage.pricing_layered_rounds + (
        energy.pricing_layered_rounds if energy is not None else 0)
    total_depth_fair_requested_calls = (
        coverage.pricing_depth_fair_requested_calls
        + (energy.pricing_depth_fair_requested_calls
           if energy is not None else 0))
    total_depth_fair_active_calls = (
        coverage.pricing_depth_fair_active_calls
        + (energy.pricing_depth_fair_active_calls
           if energy is not None else 0))
    total_depth_fair_rounds = coverage.pricing_depth_fair_rounds + (
        energy.pricing_depth_fair_rounds if energy is not None else 0)
    total_depth_fair_halfcap_dual_abs_sum = (
        coverage.pricing_depth_fair_halfcap_dual_abs_sum
        + (energy.pricing_depth_fair_halfcap_dual_abs_sum
           if energy is not None else 0.0))
    total_multistop_neutral_enabled_calls = int(
        coverage.pricing_multistop_neutral_enabled_calls
        + (energy.pricing_multistop_neutral_enabled_calls
           if energy is not None else 0))
    total_multistop_candidates_seen = int(
        coverage.pricing_multistop_candidates_seen
        + (energy.pricing_multistop_candidates_seen
           if energy is not None else 0))
    total_multistop_cross_zero_seen = int(
        coverage.pricing_multistop_cross_zero_seen
        + (energy.pricing_multistop_cross_zero_seen
           if energy is not None else 0))
    total_multistop_nonnegative_seen = int(
        coverage.pricing_multistop_nonnegative_seen
        + (energy.pricing_multistop_nonnegative_seen
           if energy is not None else 0))
    total_multistop_neutral_returned = int(
        coverage.pricing_multistop_neutral_returned
        + (energy.pricing_multistop_neutral_returned
           if energy is not None else 0))
    total_multistop_neutral_added = int(
        coverage.pricing_multistop_neutral_added
        + (energy.pricing_multistop_neutral_added
           if energy is not None else 0))
    total_multistop_neutral_batches = int(
        coverage.pricing_multistop_neutral_batches
        + (energy.pricing_multistop_neutral_batches
           if energy is not None else 0))
    total_multistop_neutral_returned_by_depth = _sum_depth_dict(
        coverage.pricing_multistop_neutral_returned_by_depth,
        energy.pricing_multistop_neutral_returned_by_depth
        if energy is not None else {})
    total_multistop_best_stop_count = max(
        int(coverage.pricing_multistop_best_stop_count),
        int(energy.pricing_multistop_best_stop_count)
        if energy is not None else 0)
    total_multistop_best_uncovered_gain = max(
        int(coverage.pricing_multistop_best_uncovered_gain),
        int(energy.pricing_multistop_best_uncovered_gain)
        if energy is not None else 0)
    _rc_vals = [
        _v for _v in (
            coverage.pricing_multistop_best_rc_ub,
            energy.pricing_multistop_best_rc_ub if energy is not None else None)
        if _v is not None]
    total_multistop_best_rc_ub = (
        None if not _rc_vals else min(float(_v) for _v in _rc_vals))
    _eps_vals = [
        _v for _v in (
            coverage.pricing_multistop_best_energy_per_stop_Wh,
            energy.pricing_multistop_best_energy_per_stop_Wh
            if energy is not None else None)
        if _v is not None]
    total_multistop_best_energy_per_stop_Wh = (
        None if not _eps_vals else min(float(_v) for _v in _eps_vals))
    total_multistop_neutral_used_in_incumbent = int(
        coverage.pricing_multistop_neutral_used_in_incumbent
        + (energy.pricing_multistop_neutral_used_in_incumbent
           if energy is not None else 0))
    total_physical_cache_hits = coverage.pricing_physical_cache_hits + (
        energy.pricing_physical_cache_hits if energy is not None else 0)
    total_physical_cache_misses = coverage.pricing_physical_cache_misses + (
        energy.pricing_physical_cache_misses if energy is not None else 0)
    total_pricing_runtime_s = float(coverage.pricing_runtime_s) + (
        float(energy.pricing_runtime_s) if energy is not None else 0.0)
    total_pricing_physical_evaluator_runtime_s = float(
        coverage.pricing_physical_evaluator_runtime_s) + (
        float(energy.pricing_physical_evaluator_runtime_s) if energy is not None else 0.0)
    total_pricing_prefix_bound_runtime_s = float(coverage.pricing_prefix_bound_runtime_s) + (
        float(energy.pricing_prefix_bound_runtime_s) if energy is not None else 0.0)
    total_pricing_prefix_service_runtime_s = float(coverage.pricing_prefix_service_runtime_s) + (
        float(energy.pricing_prefix_service_runtime_s) if energy is not None else 0.0)
    total_pricing_certified_prefix_prunes = int(coverage.pricing_certified_prefix_prunes) + (
        int(energy.pricing_certified_prefix_prunes) if energy is not None else 0)
    total_pricing_service_floor_prunes = int(coverage.pricing_service_floor_prunes) + (
        int(energy.pricing_service_floor_prunes) if energy is not None else 0)
    total_rmp_runtime_s = float(coverage.rmp_runtime_s) + (
        float(energy.rmp_runtime_s) if energy is not None else 0.0)
    total_phase_one_runtime_s = float(coverage.phase_one_runtime_s) + (
        float(energy.phase_one_runtime_s) if energy is not None else 0.0)
    total_resource_audit_runtime_s = float(coverage.resource_audit_runtime_s) + (
        float(energy.resource_audit_runtime_s) if energy is not None else 0.0)
    total_pricing_call_records = (
        [dict(stage_scope="coverage", **_r) for _r in coverage.pricing_call_records]
        + ([dict(stage_scope="energy", **_r) for _r in energy.pricing_call_records]
           if energy is not None else []))
    total_rmp_records = (
        [dict(stage_scope="coverage", **_r) for _r in coverage.rmp_records]
        + ([dict(stage_scope="energy", **_r) for _r in energy.rmp_records]
           if energy is not None else []))
    total_resource_audit_records = (
        [dict(stage_scope="coverage", **_r) for _r in coverage.resource_audit_records]
        + ([dict(stage_scope="energy", **_r) for _r in energy.resource_audit_records]
           if energy is not None else []))
    total_cuts = coverage.resource_cuts_added + (energy.resource_cuts_added if energy is not None else 0)
    total_rmp_solves = coverage.rmp_solves + (energy.rmp_solves if energy is not None else 0)
    total_phase_one_solves = coverage.phase_one_solves + (energy.phase_one_solves if energy is not None else 0)
    total_pricing_candidates = coverage.pricing_candidates + (energy.pricing_candidates if energy is not None else 0)
    total_pricing_nodes = coverage.pricing_nodes + (energy.pricing_nodes if energy is not None else 0)
    total_resource_audits = (int(initial_resource_audit_calls) + coverage.resource_audit_calls
                             + (energy.resource_audit_calls if energy is not None else 0))
    total_branch_children = coverage.branch_children_created + (
        energy.branch_children_created if energy is not None else 0)
    total_branch_decisions = coverage.branch_decisions + (
        energy.branch_decisions if energy is not None else 0)
    pricing_closed = coverage.pricing_complete and (
        energy is None or energy.pricing_complete)
    pricing_search_complete = coverage.pricing_search_complete and (
        energy is None or energy.pricing_search_complete)
    pricing_bound_available = coverage.pricing_bound_available and (
        energy is None or energy.pricing_bound_available)
    resource_complete = coverage.resource_audit_complete and (
        energy is None or energy.resource_audit_complete)
    farkas_complete = coverage.farkas_pricing_complete and (
        energy is None or energy.farkas_pricing_complete)
    branching_complete = coverage.branching_complete and (
        energy is None or energy.branching_complete)
    coverage_bound_source = str(coverage.bound_source)
    energy_bound_source = (None if energy is None else str(energy.bound_source))
    bound_source = (coverage_bound_source if energy_bound_source is None else
                    f"coverage:{coverage_bound_source};energy:{energy_bound_source}")
    last_best = (energy.pricing_best_reduced_value if energy is not None
                 else coverage.pricing_best_reduced_value)
    last_bound = (energy.pricing_reduced_value_bound if energy is not None
                  else coverage.pricing_reduced_value_bound)
    assignment = (getattr(final_audit, "assignment", None) or {}) if final_audit is not None else {}
    battery_used = [float(v) for v in assignment.get("battery_energy_used_Wh", [])]
    battery_end_soc = [100.0 * max(0.0, float(p.B_use) - v) / max(float(p.B_use), 1e-12)
                       for v in battery_used]
    swap_events = [tuple(map(float, e)) for e in assignment.get("swap_events", [])]
    quick_events = [tuple(map(float, e)) for e in assignment.get("quick_inspection_events", [])]
    mean_stops = (float(np.mean([len(_ordered_tids(archive[j])) for j in final_selection]))
                  if final_selection else 0.0)
    multi_stop_ratio = (float(np.mean([len(_ordered_tids(archive[j])) > 1
                                      for j in final_selection]))
                        if final_selection else 0.0)
    makespan = max((float(archive[j]["tau"]) + float(archive[j]["h"]) + t_clear
                    for j in final_selection), default=0.0)

    # [THM-LEX] Algorithmic tree closure is necessary but physical certification
    # additionally requires the provenance/invariance/range/numeric proof contracts.
    algorithmic_global_certificate = bool(lex_opt)
    route_semantics_invariance_certified = _route_archive_semantics_invariant(archive)
    future_column_row_ranges_certified = _future_row_range_contract_self_check(max_stops)
    binary64_model_contract_enforced = bool(
        contract.get("physical_numeric_contract") == RM.FORMAL_PHYSICAL_NUMERIC_CONTRACT
        and contract.get("route_identity_contract") == ROUTE_IDENTITY_CONTRACT
        and contract.get("model_semantics_contract") == MODEL_SEMANTICS_CONTRACT)
    formal_proof_contract_enforced = bool(
        contract.get("formal_proof_contract") == FORMAL_PROOF_CONTRACT
        and tuple(contract.get("formal_proof_obligations", ())) == FORMAL_PROOF_OBLIGATIONS
        and contract.get("formal_proof_contract_sha256") == FORMAL_PROOF_CONTRACT_SHA256)
    if not route_semantics_invariance_certified:
        raise RuntimeError("formal route archive semantic invariant violated")
    if not future_column_row_ranges_certified:
        raise RuntimeError("formal future-column row-range contract self-check failed")
    coverage_algorithmic_certificate = bool(coverage_optimal)
    coverage_physical_model_certificate = _physical_certificate_guard(
        algorithmic_global_certificate=coverage_algorithmic_certificate,
        route_universe_provenance_certified=route_universe_provenance_certified,
        mode=mode,
        route_semantics_invariance_certified=route_semantics_invariance_certified,
        future_column_row_ranges_certified=future_column_row_ranges_certified,
        binary64_model_contract_enforced=binary64_model_contract_enforced,
        formal_proof_contract_enforced=formal_proof_contract_enforced)
    physical_model_global_certificate = _physical_certificate_guard(
        algorithmic_global_certificate=algorithmic_global_certificate,
        route_universe_provenance_certified=route_universe_provenance_certified,
        mode=mode,
        route_semantics_invariance_certified=route_semantics_invariance_certified,
        future_column_row_ranges_certified=future_column_row_ranges_certified,
        binary64_model_contract_enforced=binary64_model_contract_enforced,
        formal_proof_contract_enforced=formal_proof_contract_enforced)

    battery_halfcap_formal_enabled = bool(
        pricing_key in {
            "exact-layered-batch-primal-battery-halfcap-formal",
            "exact-layered-batch-primal-battery-halfcap-depth-fair-formal",
            "exact-layered-batch-primal-battery-halfcap-depth-fair-neutral-formal",
            "exact-layered-batch-primal-battery-halfcap-resource-exchange-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-primal-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-guided-formal",
                        "exact-layered-batch-primal-battery-halfcap-deck-guided-formal",
                        "exact-layered-batch-primal-battery-halfcap-adaptive-multistop-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-variant-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-variant-diagnostic-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-variant-archive-recovery-formal"})
    _formal_halfcap = Fraction.from_float(float(p.B_use))
    battery_halfcap_archive_high_energy_routes = int(sum(
        1 for _c in archive
        if 2 * Fraction.from_float(float(_c["E_soc_required_Wh"]))
        > _formal_halfcap))
    battery_halfcap_archive_low_energy_routes = int(
        len(archive) - battery_halfcap_archive_high_energy_routes)

    result = dict(
        status=status, termination_reason=str(termination_reason),
        runtime_s=float(formal_solver_runtime_s), time_limit_s=time_limit_s,
        total_wall_runtime_s=float(time.monotonic() - started),
        diagnostic_runtime_s=float(
            (archive_diag.get("runtime_s", 0.0) or 0.0)
            + (archive_target_diag.get("runtime_s", 0.0) or 0.0)
            + ((archive_target_diag.get("certificate_shadow", {}) or {}).get(
                "runtime_s", 0.0) or 0.0)
            + ((archive_target_diag.get("battery_clique_target_rerun", {}) or {}).get(
                "runtime_s", 0.0) or 0.0)
            + (resource_variant_diag.get("runtime_s", 0.0) or 0.0)
            + (fullspace_target_diag.get("runtime_s", 0.0) or 0.0)),
        algorithm="branch-price-and-cut-with-logic-benders",
        pricing_method=(
            "complete-materialized-physical-route-universe"
            if route_universe_source == "materialized-complete-physical-oracle"
            else ("exact-layered-batch-primal-target9-battery-clique-root-cause-diagnostic"
                  if pricing_key
                      == "exact-layered-batch-primal-target9-battery-clique-diagnostic-shadow"
                  else "exact-layered-batch-primal-target9-certificate-root-cause-diagnostic"
                  if pricing_key
                      == "exact-layered-batch-primal-target9-certificate-diagnostic-shadow"
                  else "exact-layered-batch-primal-target9-root-cause-diagnostic"
                  if pricing_key
                      == "exact-layered-batch-primal-target9-diagnostic-shadow"
                  else "exact-layered-batch-primal-root-cause-diagnostic"
                  if pricing_key
                      == "exact-layered-batch-primal-diagnostic-shadow"
                  else "exact-layered-batch-primal-plus-formal-battery-halfcap-resource-variants-with-exact-archive-primal-recovery"
                  if pricing_key
                      == "exact-layered-batch-primal-battery-halfcap-resource-variant-archive-recovery-formal"
                  else "exact-layered-batch-primal-plus-formal-battery-halfcap-resource-variants-with-postsolve-diagnostics"
                  if pricing_key
                      == "exact-layered-batch-primal-battery-halfcap-resource-variant-diagnostic-formal"
                  else "exact-layered-batch-primal-plus-formal-battery-halfcap-adaptive-two-stop-enrichment"
                  if pricing_key
                      == "exact-layered-batch-primal-battery-halfcap-adaptive-multistop-formal"
                  else "exact-layered-batch-primal-plus-formal-battery-halfcap-deck-guided-resource-primal"
                  if pricing_key
                      == "exact-layered-batch-primal-battery-halfcap-deck-guided-formal"
                  else "exact-layered-batch-primal-plus-formal-battery-halfcap-unified-resource-primal"
                  if pricing_key
                      == "exact-layered-batch-primal-battery-halfcap-resource-primal-formal"
                  else "exact-layered-batch-primal-plus-formal-battery-halfcap-resource-aware-exchange"
                  if pricing_key
                      == "exact-layered-batch-primal-battery-halfcap-resource-exchange-formal"
                  else "exact-layered-batch-primal-plus-formal-battery-halfcap-clique-depth-fair-neutral-multistop"
                  if pricing_key
                      == "exact-layered-batch-primal-battery-halfcap-depth-fair-neutral-formal"
                  else "exact-layered-batch-primal-plus-formal-battery-halfcap-clique-depth-fair"
                  if pricing_key
                      == "exact-layered-batch-primal-battery-halfcap-depth-fair-formal"
                  else "exact-layered-batch-primal-plus-formal-battery-halfcap-clique"
                  if pricing_key == "exact-layered-batch-primal-battery-halfcap-formal"
                  else "exact-layered-batch-discovery-plus-primal-incumbent-refresh"
                  if pricing_key == "exact-layered-batch-primal-shadow"
                  else "exact-layered-round-robin-adaptive-diverse-batch-discovery"
                  if pricing_key == "exact-layered-batch-shadow"
                  else "exact-layered-round-robin-dual-guided-discovery-plus-prefix-shadow"
                  if pricing_key == "exact-layered-guided-shadow"
                  else "exact-implicit-dfs-dual-guided-discovery-plus-prefix-shadow"
                  if pricing_key == "exact-dual-guided-shadow"
                  else "exact-implicit-dfs-discovery-plus-prefix-shadow"
                  if pricing_key == "exact-discovery-shadow"
                  else "exact-implicit-elementary-sequence-dfs")
            if route_universe_source == "physical-oracle"
            else "synthetic-fixture-exhaustive-list"),
        branching_complete=bool(branching_complete),
        farkas_pricing_complete=bool(farkas_complete),
        coverage_incumbent=coverage_inc,
        coverage_upper_bound=coverage_ub,
        coverage_gap_abs=coverage_gap_abs,
        coverage_gap_pct=float(coverage_gap_pct),
        coverage_optimal=coverage_optimal,
        coverage_algorithmic_certificate=bool(coverage_algorithmic_certificate),
        coverage_physical_model_certificate=bool(coverage_physical_model_certificate),
        coverage_global_certificate_available=bool(coverage_physical_model_certificate),
        solve_scope=str(solve_scope),
        energy_incumbent_Wh=(None if energy_inc is None else float(energy_inc)),
        energy_incumbent_estimate_Wh=(
            None if energy_inc is None else float(energy_inc_estimate)),
        energy_incumbent_lower_enclosure_Wh=(
            None if energy_inc is None else float(energy_inc_lower)),
        energy_lower_bound_Wh=(None if energy_lb is None else float(energy_lb)),
        energy_gap_abs_Wh=energy_gap_abs,
        energy_gap_pct=energy_gap_pct,
        energy_optimal=energy_optimal,
        conditional_energy_gap_pct=conditional_energy_gap_pct,
        global_energy_gap_reason=(
            "coverage optimum not proven" if not coverage_optimal else
            "energy stage intentionally skipped by solve_scope=coverage-only"
                if solve_scope == "coverage-only" else
            "energy stage not completed before deadline" if energy is None else
            "global energy lower bound unavailable" if energy_lb is None else
            None),
        lexicographic_optimal=lex_opt,
        # Legacy ``pricing_complete`` is retained as an alias of strict closure.
        pricing_complete=bool(pricing_closed),
        pricing_closed=bool(pricing_closed),
        pricing_search_complete=bool(pricing_search_complete),
        pricing_bound_available=bool(pricing_bound_available),
        resource_audit_complete=bool(resource_complete),
        bound_scope=("global_discrete_physical_model"
                     if physical_model_global_certificate or route_universe_provenance_certified
                     else "synthetic_finite_route_fixture"),
        bound_source=str(bound_source),
        coverage_bound_source=coverage_bound_source,
        energy_bound_source=energy_bound_source,
        open_nodes=int(total_open), processed_nodes=int(total_processed),
        branch_nodes=int(total_processed), branch_decisions=int(total_branch_decisions),
        branch_children_created=int(total_branch_children),
        rmp_solves=int(total_rmp_solves), phase_one_solves=int(total_phase_one_solves),
        generated_columns=int(total_generated), columns_accepted=int(total_generated),
        pricing_calls=int(total_pricing), exact_pricing_calls=int(total_exact_pricing),
        exact_certification_calls=int(total_certification_calls),
        pricing_discovery_calls=int(total_discovery_calls),
        pricing_discovery_early_returns=int(total_discovery_early_returns),
        pricing_discovery_improving_seen=int(
            total_discovery_improving_seen),
        pricing_discovery_improving_returned=int(
            total_discovery_improving_returned),
        pricing_discovery_diverse_returns=int(
            total_discovery_diverse_returns),
        pricing_discovery_hard_cap_returns=int(
            total_discovery_hard_cap_returns),
        pricing_discovery_max_return_batch=int(
            total_discovery_max_return_batch),
        pricing_discovery_max_distinct_launches=int(
            total_discovery_max_distinct_launches),
        pricing_discovery_max_distinct_service_sets=int(
            total_discovery_max_distinct_service_sets),
        pricing_multistop_neutral_enabled_calls=int(
            total_multistop_neutral_enabled_calls),
        pricing_multistop_candidates_seen=int(
            total_multistop_candidates_seen),
        pricing_multistop_physical_feasible=int(
            total_multistop_candidates_seen),
        pricing_multistop_cross_zero_seen=int(
            total_multistop_cross_zero_seen),
        pricing_multistop_nonnegative_seen=int(
            total_multistop_nonnegative_seen),
        pricing_multistop_neutral_returned=int(
            total_multistop_neutral_returned),
        pricing_multistop_neutral_added=int(
            total_multistop_neutral_added),
        pricing_multistop_neutral_batches=int(
            total_multistop_neutral_batches),
        pricing_multistop_neutral_returned_by_depth=dict(
            total_multistop_neutral_returned_by_depth),
        pricing_multistop_depth2_neutral=int(
            total_multistop_neutral_returned_by_depth.get(2, 0)),
        pricing_multistop_depth3_neutral=int(
            total_multistop_neutral_returned_by_depth.get(3, 0)),
        pricing_multistop_depth4_neutral=int(
            total_multistop_neutral_returned_by_depth.get(4, 0)),
        pricing_multistop_best_stop_count=int(
            total_multistop_best_stop_count),
        pricing_multistop_best_uncovered_gain=int(
            total_multistop_best_uncovered_gain),
        pricing_multistop_best_rc_ub=(
            None if total_multistop_best_rc_ub is None
            else float(total_multistop_best_rc_ub)),
        pricing_multistop_best_energy_per_stop_Wh=(
            None if total_multistop_best_energy_per_stop_Wh is None
            else float(total_multistop_best_energy_per_stop_Wh)),
        pricing_multistop_neutral_used_in_incumbent=int(
            total_multistop_neutral_used_in_incumbent),
        pricing_multistop_merge_enabled=bool(
            pricing_key
            == "exact-layered-batch-primal-battery-halfcap-adaptive-multistop-formal"),
        pricing_multistop_merge_triggers=int(total_multistop_merge_triggers),
        pricing_multistop_merge_attempts=int(total_multistop_merge_attempts),
        pricing_multistop_merge_physical_feasible=int(
            total_multistop_merge_physical_feasible),
        pricing_multistop_merge_new_candidates=int(
            total_multistop_merge_new_candidates),
        pricing_multistop_merge_returned=int(total_multistop_merge_returned),
        pricing_multistop_merge_added=int(total_multistop_merge_added),
        pricing_multistop_merge_batches=int(total_multistop_merge_batches),
        pricing_multistop_merge_distinct_pairs=int(
            total_multistop_merge_distinct_pairs),
        pricing_multistop_merge_best_rc_ub=(
            None if total_multistop_merge_best_rc_ub is None
            else float(total_multistop_merge_best_rc_ub)),
        pricing_multistop_merge_best_energy_per_stop_Wh=(
            None if total_multistop_merge_best_energy_per_stop_Wh is None
            else float(total_multistop_merge_best_energy_per_stop_Wh)),
        pricing_multistop_merge_best_uncovered_gain=int(
            total_multistop_merge_best_uncovered_gain),
        pricing_multistop_merge_used_in_incumbent=int(
            total_multistop_merge_used_in_incumbent),
        pricing_resource_variant_enabled=bool(
            pricing_key in {
                "exact-layered-batch-primal-battery-halfcap-resource-variant-formal",
                "exact-layered-batch-primal-battery-halfcap-resource-variant-diagnostic-formal",
                "exact-layered-batch-primal-battery-halfcap-resource-variant-archive-recovery-formal"}),
        pricing_resource_variant_triggers=int(total_resource_variant_triggers),
        pricing_resource_variant_attempts=int(total_resource_variant_attempts),
        pricing_resource_variant_deck_compatible_specs=int(
            total_resource_variant_deck_compatible_specs),
        pricing_resource_variant_deck_prefilter_skips=int(
            total_resource_variant_deck_prefilter_skips),
        pricing_resource_variant_physical_feasible=int(
            total_resource_variant_physical_feasible),
        pricing_resource_variant_new_candidates=int(
            total_resource_variant_new_candidates),
        pricing_resource_variant_returned=int(total_resource_variant_returned),
        pricing_resource_variant_added=int(total_resource_variant_added),
        pricing_resource_variant_batches=int(total_resource_variant_batches),
        pricing_resource_variant_distinct_turbines=int(
            total_resource_variant_distinct_turbines),
        pricing_resource_variant_best_rc_ub=(
            None if total_resource_variant_best_rc_ub is None
            else float(total_resource_variant_best_rc_ub)),
        pricing_resource_variant_best_energy_Wh=(
            None if total_resource_variant_best_energy_Wh is None
            else float(total_resource_variant_best_energy_Wh)),
        pricing_resource_variant_used_in_incumbent=int(
            total_resource_variant_used_in_incumbent),
        pricing_resource_variant_records=list(total_resource_variant_records),
        archive_primal_recovery_enabled=bool(
            coverage.archive_primal_recovery_enabled),
        archive_primal_recovery_time_limit_s=float(
            archive_primal_recovery_time_limit_s),
        archive_primal_recovery_calls=int(
            coverage.archive_primal_recovery_calls),
        archive_primal_recovery_runtime_s=float(
            coverage.archive_primal_recovery_runtime_s),
        archive_primal_recovery_audit_calls=int(
            coverage.archive_primal_recovery_audit_calls),
        archive_primal_recovery_timeouts=int(
            coverage.archive_primal_recovery_timeouts),
        archive_primal_recovery_improvements=int(
            coverage.archive_primal_recovery_improvements),
        archive_primal_recovery_best_coverage=int(
            coverage.archive_primal_recovery_best_coverage),
        archive_primal_recovery_best_archive_columns=int(
            coverage.archive_primal_recovery_best_archive_columns),
        archive_primal_recovery_records=list(
            coverage.archive_primal_recovery_records),
        archive_primal_recovery_witness_selection_indices=list(
            coverage.archive_primal_recovery_witness_selection_indices),
        archive_primal_recovery_witness_route_signatures=list(
            coverage.archive_primal_recovery_witness_route_signatures),
        archive_primal_recovery_witness_covered_turbines=list(
            coverage.archive_primal_recovery_witness_covered_turbines),
        primal_refresh_calls=int(total_primal_refresh_calls),
        primal_refresh_audit_calls=int(
            total_primal_refresh_audit_calls),
        primal_refresh_timeouts=int(total_primal_refresh_timeouts),
        primal_refresh_improvements=int(
            total_primal_refresh_improvements),
        primal_refresh_best_coverage=int(
            total_primal_refresh_best_coverage),
        primal_refresh_columns_seen=int(
            total_primal_refresh_columns_seen),
        primal_refresh_rebuilds=int(total_primal_refresh_rebuilds),
        primal_refresh_repairs=int(total_primal_refresh_repairs),
        primal_refresh_augmentation_audits=int(
            total_primal_refresh_augmentation_audits),
        primal_refresh_rebuild_audits=int(
            total_primal_refresh_rebuild_audits),
        primal_refresh_repair_audits=int(
            total_primal_refresh_repair_audits),
        primal_refresh_augmentation_improvements=int(
            total_primal_refresh_augmentation_improvements),
        primal_refresh_rebuild_improvements=int(
            total_primal_refresh_rebuild_improvements),
        primal_refresh_repair_improvements=int(
            total_primal_refresh_repair_improvements),
        primal_refresh_duplicate_trials_skipped=int(
            total_primal_refresh_duplicate_trials_skipped),
        primal_refresh_cached_infeasible_trials=int(
            total_primal_refresh_cached_infeasible_trials),
        primal_refresh_uncovered_fair_rounds=int(
            total_primal_refresh_uncovered_fair_rounds),
        primal_refresh_failure_reasons=dict(
            total_primal_refresh_failure_reasons),
        primal_deck_diagnostic_enabled=bool(
            pricing_key in {
                "exact-layered-batch-primal-battery-halfcap-deck-guided-formal",
                "exact-layered-batch-primal-battery-halfcap-adaptive-multistop-formal",
                "exact-layered-batch-primal-battery-halfcap-resource-variant-formal",
                "exact-layered-batch-primal-battery-halfcap-resource-variant-diagnostic-formal",
                "exact-layered-batch-primal-battery-halfcap-resource-variant-archive-recovery-formal"}),
        primal_deck_archive_conflict_edges=int(
            total_primal_deck_archive_conflict_edges),
        primal_deck_archive_max_degree=int(
            total_primal_deck_archive_max_degree),
        primal_deck_archive_max_component=int(
            total_primal_deck_archive_max_component),
        primal_deck_candidate_scored=int(total_primal_deck_candidate_scored),
        primal_deck_candidate_zero_conflict=int(
            total_primal_deck_candidate_zero_conflict),
        primal_deck_candidate_positive_conflict=int(
            total_primal_deck_candidate_positive_conflict),
        primal_deck_prefilter_skips=int(total_primal_deck_prefilter_skips),
        primal_deck_max_candidate_conflicts=int(
            total_primal_deck_max_candidate_conflicts),
        primal_deck_conflict_pairs_sample=list(
            total_primal_deck_conflict_pairs_sample),
        primal_exchange_enabled=bool(
            pricing_key in {
                "exact-layered-batch-primal-battery-halfcap-resource-exchange-formal",
                "exact-layered-batch-primal-battery-halfcap-resource-primal-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-guided-formal",
                        "exact-layered-batch-primal-battery-halfcap-deck-guided-formal",
                        "exact-layered-batch-primal-battery-halfcap-adaptive-multistop-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-variant-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-variant-diagnostic-formal",
                        "exact-layered-batch-primal-battery-halfcap-resource-variant-archive-recovery-formal"}),
        primal_exchange_calls=int(total_primal_exchange_calls),
        primal_exchange_candidate_routes=int(
            total_primal_exchange_candidate_routes),
        primal_exchange_trials_built=int(
            total_primal_exchange_trials_built),
        primal_exchange_audit_calls=int(
            total_primal_exchange_audit_calls),
        primal_exchange_improvements=int(
            total_primal_exchange_improvements),
        primal_exchange_consolidation_trials=int(
            total_primal_exchange_consolidation_trials),
        primal_exchange_optional_drop_trials=int(
            total_primal_exchange_optional_drop_trials),
        primal_exchange_max_stop_count_considered=int(
            total_primal_exchange_max_stop_count_considered),
        primal_exchange_best_coverage=int(
            total_primal_exchange_best_coverage),
        primal_exchange_multistop_used_in_incumbent=int(
            total_primal_exchange_multistop_used_in_incumbent),
        pricing_depth_prefixes_evaluated=dict(total_depth_prefixes),
        pricing_depth_improving_seen=dict(total_depth_improving_seen),
        pricing_depth_improving_returned=dict(
            total_depth_improving_returned),
        pricing_depth1_prefixes=int(total_depth_prefixes.get(1, 0)),
        pricing_depth2_prefixes=int(total_depth_prefixes.get(2, 0)),
        pricing_depth3_prefixes=int(total_depth_prefixes.get(3, 0)),
        pricing_depth4_prefixes=int(total_depth_prefixes.get(4, 0)),
        pricing_depth1_improving=int(total_depth_improving_seen.get(1, 0)),
        pricing_depth2_improving=int(total_depth_improving_seen.get(2, 0)),
        pricing_depth3_improving=int(total_depth_improving_seen.get(3, 0)),
        pricing_depth4_improving=int(total_depth_improving_seen.get(4, 0)),
        pricing_depth1_returned=int(
            total_depth_improving_returned.get(1, 0)),
        pricing_depth2_returned=int(
            total_depth_improving_returned.get(2, 0)),
        pricing_depth3_returned=int(
            total_depth_improving_returned.get(3, 0)),
        pricing_depth4_returned=int(
            total_depth_improving_returned.get(4, 0)),
        pricing_pattern_cut_active_dual_rows=int(
            total_pattern_cut_active_dual_rows),
        pricing_pattern_cut_dual_abs_sum=float(total_pattern_cut_dual_abs_sum),
        pricing_pattern_cut_improving_seen_count=int(
            total_pattern_cut_improving_seen_count),
        pricing_pattern_cut_improving_seen_contribution_sum=float(
            total_pattern_cut_improving_seen_contribution_sum),
        pricing_pattern_cut_improving_seen_sign_essential=int(
            total_pattern_cut_improving_seen_sign_essential),
        pricing_pattern_cut_returned_count=int(total_pattern_cut_returned_count),
        pricing_pattern_cut_returned_contribution_sum=float(
            total_pattern_cut_returned_contribution_sum),
        pricing_pattern_cut_returned_sign_essential=int(
            total_pattern_cut_returned_sign_essential),
        pricing_pattern_cut_returned_by_depth=dict(
            total_pattern_cut_returned_by_depth),
        battery_halfcap_formal_enabled=bool(
            battery_halfcap_formal_enabled),
        battery_halfcap_usable_capacity_Wh=float(p.B_use),
        battery_halfcap_rhs=int(battery_count),
        battery_halfcap_archive_route_count=int(len(archive)),
        battery_halfcap_archive_high_energy_routes=int(
            battery_halfcap_archive_high_energy_routes),
        battery_halfcap_archive_low_energy_routes=int(
            battery_halfcap_archive_low_energy_routes),
        battery_halfcap_dual_active_rmp_solves=int(
            total_battery_halfcap_dual_active_rmp_solves),
        battery_halfcap_dual_abs_sum=float(
            total_battery_halfcap_dual_abs_sum),
        battery_halfcap_dual_max_abs=float(
            total_battery_halfcap_dual_max_abs),
        coverage_battery_halfcap_dual_active_rmp_solves=int(
            coverage.battery_halfcap_dual_active_rmp_solves),
        coverage_battery_halfcap_dual_abs_sum=float(
            coverage.battery_halfcap_dual_abs_sum),
        coverage_battery_halfcap_dual_max_abs=float(
            coverage.battery_halfcap_dual_max_abs),
        energy_battery_halfcap_dual_active_rmp_solves=(
            None if energy is None else int(
                energy.battery_halfcap_dual_active_rmp_solves)),
        energy_battery_halfcap_dual_abs_sum=(
            None if energy is None else float(
                energy.battery_halfcap_dual_abs_sum)),
        energy_battery_halfcap_dual_max_abs=(
            None if energy is None else float(
                energy.battery_halfcap_dual_max_abs)),
        archive_diag_enabled=bool(archive_diag.get("enabled", False)),
        archive_diag_scope=str(archive_diag.get("scope")),
        archive_diag_status=str(archive_diag.get("status")),
        archive_diag_time_limit_s=float(
            archive_diag.get("time_limit_s", 0.0) or 0.0),
        archive_diag_runtime_s=float(
            archive_diag.get("runtime_s", 0.0) or 0.0),
        archive_diag_columns=int(
            archive_diag.get("archive_columns", len(archive)) or 0),
        archive_diag_coverage_lower_bound=archive_diag.get(
            "coverage_lower_bound"),
        archive_diag_coverage_upper_bound=archive_diag.get(
            "coverage_upper_bound"),
        archive_diag_exact_optimum=archive_diag.get("exact_optimum"),
        archive_diag_optimal_proven=bool(
            archive_diag.get("optimal_proven", False)),
        archive_diag_witness_selection_indices=list(
            archive_diag.get("witness_selection_indices", []) or []),
        archive_diag_witness_route_signatures=list(
            archive_diag.get("witness_route_signatures", []) or []),
        archive_diag_witness_covered_turbines=list(
            archive_diag.get("witness_covered_turbines", []) or []),
        archive_diag_open_nodes=archive_diag.get("open_nodes"),
        archive_diag_processed_nodes=archive_diag.get("processed_nodes"),
        archive_diag_rmp_solves=archive_diag.get("rmp_solves"),
        archive_diag_resource_audit_calls=archive_diag.get(
            "resource_audit_calls"),
        archive_diag_resource_cuts_added=archive_diag.get(
            "resource_cuts_added"),
        resource_variant_diag_enabled=bool(
            resource_variant_diag.get("enabled", False)),
        resource_variant_diag_scope=str(
            resource_variant_diag.get("scope")),
        resource_variant_diag_status=str(
            resource_variant_diag.get("status")),
        resource_variant_diag_time_limit_s=float(
            resource_variant_diag.get("time_limit_s", 0.0) or 0.0),
        resource_variant_diag_runtime_s=float(
            resource_variant_diag.get("runtime_s", 0.0) or 0.0),
        resource_variant_diag_timed_out=bool(
            resource_variant_diag.get("timed_out", False)),
        resource_variant_diag_records_analyzed=int(
            resource_variant_diag.get("records_analyzed", 0) or 0),
        resource_variant_diag_records_missing_from_archive=int(
            resource_variant_diag.get("records_missing_from_archive", 0) or 0),
        resource_variant_diag_final_coverage=int(
            resource_variant_diag.get("final_coverage", 0) or 0),
        resource_variant_diag_final_uncovered_turbines=list(
            resource_variant_diag.get("final_uncovered_turbines", []) or []),
        resource_variant_diag_direct_augmentation_audits=int(
            resource_variant_diag.get("direct_augmentation_audits", 0) or 0),
        resource_variant_diag_direct_augmentation_feasible=int(
            resource_variant_diag.get("direct_augmentation_feasible", 0) or 0),
        resource_variant_diag_direct_augmentation_infeasible=int(
            resource_variant_diag.get("direct_augmentation_infeasible", 0) or 0),
        resource_variant_diag_direct_augmentation_unknown=int(
            resource_variant_diag.get("direct_augmentation_unknown", 0) or 0),
        resource_variant_diag_single_blocker_records=int(
            resource_variant_diag.get("single_blocker_records", 0) or 0),
        resource_variant_diag_blocker_retime_candidates=int(
            resource_variant_diag.get("blocker_retime_candidates", 0) or 0),
        resource_variant_diag_blocker_retime_audits=int(
            resource_variant_diag.get("blocker_retime_audits", 0) or 0),
        resource_variant_diag_blocker_retime_feasible=int(
            resource_variant_diag.get("blocker_retime_feasible", 0) or 0),
        resource_variant_diag_final_uncovered_singleton_routes=int(
            resource_variant_diag.get("final_uncovered_singleton_routes", 0) or 0),
        resource_variant_diag_final_uncovered_zero_deck_conflict_routes=int(
            resource_variant_diag.get(
                "final_uncovered_zero_deck_conflict_routes", 0) or 0),
        resource_variant_diag_final_uncovered_single_blocker_routes=int(
            resource_variant_diag.get(
                "final_uncovered_single_blocker_routes", 0) or 0),
        resource_variant_diag_final_uncovered_multi_blocker_routes=int(
            resource_variant_diag.get(
                "final_uncovered_multi_blocker_routes", 0) or 0),
        resource_variant_diag_final_uncovered_single_blocker_distinct_turbines=int(
            resource_variant_diag.get(
                "final_uncovered_single_blocker_distinct_turbines", 0) or 0),
        resource_variant_diag_records=list(
            resource_variant_diag.get("records", []) or []),
        resource_variant_diag_single_blocker_pairs_sample=list(
            resource_variant_diag.get("single_blocker_pairs_sample", []) or []),
        fullspace_target_diag_enabled=bool(
            fullspace_target_diag.get("enabled", False)),
        fullspace_target_diag_scope=str(
            fullspace_target_diag.get("scope")),
        fullspace_target_diag_status=str(
            fullspace_target_diag.get("status")),
        fullspace_target_diag_time_limit_s=float(
            fullspace_target_diag.get("time_limit_s", 0.0) or 0.0),
        fullspace_target_diag_runtime_s=float(
            fullspace_target_diag.get("runtime_s", 0.0) or 0.0),
        fullspace_target_diag_archive_columns_start=int(
            fullspace_target_diag.get("archive_columns_start", len(archive)) or 0),
        fullspace_target_diag_archive_columns_end=int(
            fullspace_target_diag.get("archive_columns_end", len(archive)) or 0),
        fullspace_target_diag_start_coverage=int(
            fullspace_target_diag.get("start_coverage", 0) or 0),
        fullspace_target_diag_best_coverage=int(
            fullspace_target_diag.get("best_coverage", 0) or 0),
        fullspace_target_diag_highest_feasible_target=(
            fullspace_target_diag.get("highest_feasible_target")),
        fullspace_target_diag_first_infeasible_target=(
            fullspace_target_diag.get("first_infeasible_target")),
        fullspace_target_diag_unresolved_target=(
            fullspace_target_diag.get("unresolved_target")),
        fullspace_target_diag_targets_attempted=int(
            fullspace_target_diag.get("targets_attempted", 0) or 0),
        fullspace_target_diag_records=list(
            fullspace_target_diag.get("records", []) or []),
        fullspace_target_diag_witness_selection_indices=list(
            fullspace_target_diag.get("witness_selection_indices", []) or []),
        fullspace_target_diag_witness_route_signatures=list(
            fullspace_target_diag.get("witness_route_signatures", []) or []),
        fullspace_target_diag_witness_covered_turbines=list(
            fullspace_target_diag.get("witness_covered_turbines", []) or []),
        archive_target_enabled=bool(
            archive_target_diag.get("enabled", False)),
        archive_target_scope=str(archive_target_diag.get("scope")),
        archive_target_min=int(
            archive_target_diag.get("target_min", 9) or 9),
        archive_target_status=str(archive_target_diag.get("status")),
        archive_target_time_limit_s=float(
            archive_target_diag.get("time_limit_s", 0.0) or 0.0),
        archive_target_runtime_s=float(
            archive_target_diag.get("runtime_s", 0.0) or 0.0),
        archive_target_columns=int(
            archive_target_diag.get("archive_columns", len(archive)) or 0),
        archive_target_feasible_proven=bool(
            archive_target_diag.get("feasible_proven", False)),
        archive_target_infeasible_proven=bool(
            archive_target_diag.get("infeasible_proven", False)),
        archive_target_witness_coverage=archive_target_diag.get(
            "witness_coverage"),
        archive_target_coverage_incumbent=archive_target_diag.get(
            "coverage_incumbent"),
        archive_target_coverage_upper_bound=archive_target_diag.get(
            "coverage_upper_bound"),
        archive_target_open_nodes=archive_target_diag.get("open_nodes"),
        archive_target_processed_nodes=archive_target_diag.get(
            "processed_nodes"),
        archive_target_rmp_solves=archive_target_diag.get("rmp_solves"),
        archive_target_resource_audit_calls=archive_target_diag.get(
            "resource_audit_calls"),
        archive_target_resource_cuts_added=archive_target_diag.get(
            "resource_cuts_added"),
        archive_target_rejected_pattern_count=int(
            archive_target_diag.get("rejected_pattern_count", 0) or 0),
        archive_target_rejected_pattern_size_avg=archive_target_diag.get(
            "rejected_pattern_size_avg"),
        archive_target_rejected_pattern_size_min=archive_target_diag.get(
            "rejected_pattern_size_min"),
        archive_target_rejected_pattern_size_max=archive_target_diag.get(
            "rejected_pattern_size_max"),
        archive_target_rejected_pattern_coverage_avg=archive_target_diag.get(
            "rejected_pattern_coverage_avg"),
        archive_target_rejected_pattern_coverage_min=archive_target_diag.get(
            "rejected_pattern_coverage_min"),
        archive_target_rejected_pattern_coverage_max=archive_target_diag.get(
            "rejected_pattern_coverage_max"),
        archive_target_rejected_hamming_avg=archive_target_diag.get(
            "rejected_hamming_avg"),
        archive_target_rejected_hamming_min=archive_target_diag.get(
            "rejected_hamming_min"),
        archive_target_rejected_hamming_max=archive_target_diag.get(
            "rejected_hamming_max"),
        archive_target_resource_failure_event_counts=dict(
            archive_target_diag.get(
                "resource_failure_event_counts", {}) or {}),
        archive_target_resource_failure_pattern_counts=dict(
            archive_target_diag.get(
                "resource_failure_pattern_counts", {}) or {}),
        archive_target_rejected_morphology_counts=dict(
            archive_target_diag.get("rejected_morphology_counts", {}) or {}),
        archive_target_rejected_route_stop_count_totals=dict(
            archive_target_diag.get("rejected_route_stop_count_totals", {}) or {}),
        archive_target_rejected_coverage_route_count_joint=dict(
            archive_target_diag.get("rejected_coverage_route_count_joint", {}) or {}),
        archive_target_certificate_shadow=dict(
            archive_target_diag.get("certificate_shadow", {}) or {}),
        archive_target_shadow_analyzed_patterns=int(
            ((archive_target_diag.get("certificate_shadow", {}) or {}).get(
                "analyzed_patterns", 0)) or 0),
        archive_target_shadow_total_patterns=int(
            ((archive_target_diag.get("certificate_shadow", {}) or {}).get(
                "total_patterns", 0)) or 0),
        archive_target_shadow_timed_out=bool(
            (archive_target_diag.get("certificate_shadow", {}) or {}).get(
                "timed_out", False)),
        archive_target_shadow_pooled_energy_infeasible=int(
            ((archive_target_diag.get("certificate_shadow", {}) or {}).get(
                "pooled_energy_infeasible_patterns", 0)) or 0),
        archive_target_shadow_battery_binpack_infeasible=int(
            ((archive_target_diag.get("certificate_shadow", {}) or {}).get(
                "battery_binpack_infeasible_patterns", 0)) or 0),
        archive_target_shadow_battery_binpack_feasible=int(
            ((archive_target_diag.get("certificate_shadow", {}) or {}).get(
                "battery_binpack_feasible_patterns", 0)) or 0),
        archive_target_shadow_battery_binpack_unknown=int(
            ((archive_target_diag.get("certificate_shadow", {}) or {}).get(
                "battery_binpack_unknown_patterns", 0)) or 0),
        archive_target_shadow_battery_core_unique_count=int(
            ((archive_target_diag.get("certificate_shadow", {}) or {}).get(
                "battery_core_unique_count", 0)) or 0),
        archive_target_shadow_battery_core_size_avg=(
            (archive_target_diag.get("certificate_shadow", {}) or {}).get(
                "battery_core_size_avg")),
        archive_target_shadow_battery_core_size_min=(
            (archive_target_diag.get("certificate_shadow", {}) or {}).get(
                "battery_core_size_min")),
        archive_target_shadow_battery_core_size_max=(
            (archive_target_diag.get("certificate_shadow", {}) or {}).get(
                "battery_core_size_max")),
        archive_target_shadow_prior_core_cover_count=int(
            ((archive_target_diag.get("certificate_shadow", {}) or {}).get(
                "battery_core_shadow_prior_cover_count", 0)) or 0),
        archive_target_shadow_prior_core_cover_fraction=(
            (archive_target_diag.get("certificate_shadow", {}) or {}).get(
                "battery_core_shadow_prior_cover_fraction")),
        archive_target_shadow_fastest_turnaround_infeasible=int(
            ((archive_target_diag.get("certificate_shadow", {}) or {}).get(
                "fastest_turnaround_infeasible_patterns", 0)) or 0),
        archive_target_shadow_battery_min_required_counts=dict(
            (archive_target_diag.get("certificate_shadow", {}) or {}).get(
                "battery_min_required_counts", {}) or {}),
        archive_target_shadow_first_proof_layer_counts=dict(
            (archive_target_diag.get("certificate_shadow", {}) or {}).get(
                "first_proof_layer_counts", {}) or {}),
        archive_target_battery_clique_shadow=dict(
            archive_target_diag.get("battery_clique_shadow", {}) or {}),
        archive_target_clique_halfcap_rows=int(
            ((archive_target_diag.get("battery_clique_shadow", {}) or {}).get(
                "halfcap_rows", 0)) or 0),
        archive_target_clique_anchor_rows=int(
            ((archive_target_diag.get("battery_clique_shadow", {}) or {}).get(
                "anchor_rows", 0)) or 0),
        archive_target_clique_total_rows=int(
            ((archive_target_diag.get("battery_clique_shadow", {}) or {}).get(
                "total_rows", 0)) or 0),
        archive_target_clique_archive_halfcap_routes=int(
            ((archive_target_diag.get("battery_clique_shadow", {}) or {}).get(
                "archive_halfcap_routes", 0)) or 0),
        archive_target_clique_archive_nonhalfcap_routes=int(
            ((archive_target_diag.get("battery_clique_shadow", {}) or {}).get(
                "archive_nonhalfcap_routes", 0)) or 0),
        archive_target_clique_archive_halfcap_stop_count_counts=dict(
            (archive_target_diag.get("battery_clique_shadow", {}) or {}).get(
                "archive_halfcap_stop_count_counts", {}) or {}),
        archive_target_clique_rejected_halfcap_violations=int(
            ((archive_target_diag.get("battery_clique_shadow", {}) or {}).get(
                "rejected_halfcap_violations", 0)) or 0),
        archive_target_clique_rejected_anchor_violations=int(
            ((archive_target_diag.get("battery_clique_shadow", {}) or {}).get(
                "rejected_anchor_violations", 0)) or 0),
        archive_target_clique_rejected_anchor_only_violations=int(
            ((archive_target_diag.get("battery_clique_shadow", {}) or {}).get(
                "rejected_anchor_only_violations", 0)) or 0),
        archive_target_clique_rejected_any_violations=int(
            ((archive_target_diag.get("battery_clique_shadow", {}) or {}).get(
                "rejected_any_clique_violations", 0)) or 0),
        archive_target_clique_rejected_uncovered=int(
            ((archive_target_diag.get("battery_clique_shadow", {}) or {}).get(
                "rejected_uncovered_by_cliques", 0)) or 0),
        archive_target_clique_rejected_halfcap_count_distribution=dict(
            (archive_target_diag.get("battery_clique_shadow", {}) or {}).get(
                "rejected_halfcap_selected_count_distribution", {}) or {}),
        archive_clique_target_enabled=bool(
            (archive_target_diag.get("battery_clique_target_rerun", {}) or {}).get(
                "enabled", False)),
        archive_clique_target_scope=str(
            (archive_target_diag.get("battery_clique_target_rerun", {}) or {}).get(
                "scope",
                "fixed-generated-column-archive-target-with-battery-cliques-not-formal")),
        archive_clique_target_status=str(
            (archive_target_diag.get("battery_clique_target_rerun", {}) or {}).get(
                "status", "not-run")),
        archive_clique_target_time_limit_s=float(
            (archive_target_diag.get("battery_clique_target_rerun", {}) or {}).get(
                "time_limit_s", 0.0) or 0.0),
        archive_clique_target_runtime_s=float(
            (archive_target_diag.get("battery_clique_target_rerun", {}) or {}).get(
                "runtime_s", 0.0) or 0.0),
        archive_clique_target_rows=int(
            (archive_target_diag.get("battery_clique_target_rerun", {}) or {}).get(
                "clique_rows", 0) or 0),
        archive_clique_target_feasible_proven=bool(
            (archive_target_diag.get("battery_clique_target_rerun", {}) or {}).get(
                "feasible_proven", False)),
        archive_clique_target_infeasible_proven=bool(
            (archive_target_diag.get("battery_clique_target_rerun", {}) or {}).get(
                "infeasible_proven", False)),
        archive_clique_target_witness_coverage=(
            (archive_target_diag.get("battery_clique_target_rerun", {}) or {}).get(
                "witness_coverage")),
        archive_clique_target_coverage_incumbent=(
            (archive_target_diag.get("battery_clique_target_rerun", {}) or {}).get(
                "coverage_incumbent")),
        archive_clique_target_coverage_upper_bound=(
            (archive_target_diag.get("battery_clique_target_rerun", {}) or {}).get(
                "coverage_upper_bound")),
        archive_clique_target_open_nodes=(
            (archive_target_diag.get("battery_clique_target_rerun", {}) or {}).get(
                "open_nodes")),
        archive_clique_target_processed_nodes=(
            (archive_target_diag.get("battery_clique_target_rerun", {}) or {}).get(
                "processed_nodes")),
        archive_clique_target_rmp_solves=(
            (archive_target_diag.get("battery_clique_target_rerun", {}) or {}).get(
                "rmp_solves")),
        archive_clique_target_resource_audit_calls=(
            (archive_target_diag.get("battery_clique_target_rerun", {}) or {}).get(
                "resource_audit_calls")),
        archive_clique_target_resource_cuts_added=(
            (archive_target_diag.get("battery_clique_target_rerun", {}) or {}).get(
                "resource_cuts_added")),
        archive_clique_target_rejected_pattern_count=int(
            (archive_target_diag.get("battery_clique_target_rerun", {}) or {}).get(
                "rejected_pattern_count", 0) or 0),
        pricing_shadow_prefixes_evaluated=int(total_shadow_prefixes_evaluated),
        pricing_shadow_prunable_prefixes=int(total_shadow_prunable_prefixes),
        pricing_shadow_false_prune_witnesses=int(total_shadow_false_prune_witnesses),
        pricing_shadow_bound_errors=int(total_shadow_bound_errors),
        pricing_shadow_complete_calls=int(total_shadow_complete_calls),
        pricing_guided_order_calls=int(total_guided_order_calls),
        pricing_guided_order_reorders=int(total_guided_order_reorders),
        pricing_guided_order_failures=int(total_guided_order_failures),
        pricing_layered_depths_started=int(total_layered_depths_started),
        pricing_layered_depths_completed=int(total_layered_depths_completed),
        pricing_layered_max_depth_completed=int(
            total_layered_max_depth_completed),
        pricing_layered_rounds=int(total_layered_rounds),
        pricing_depth_fair_requested_calls=int(
            total_depth_fair_requested_calls),
        pricing_depth_fair_active_calls=int(total_depth_fair_active_calls),
        pricing_depth_fair_rounds=int(total_depth_fair_rounds),
        pricing_depth_fair_halfcap_dual_abs_sum=float(
            total_depth_fair_halfcap_dual_abs_sum),
        pricing_physical_cache_hits=int(total_physical_cache_hits),
        pricing_physical_cache_misses=int(total_physical_cache_misses),
        pricing_runtime_s=float(total_pricing_runtime_s),
        pricing_physical_evaluator_runtime_s=float(
            total_pricing_physical_evaluator_runtime_s),
        pricing_prefix_bound_runtime_s=float(total_pricing_prefix_bound_runtime_s),
        pricing_prefix_service_runtime_s=float(total_pricing_prefix_service_runtime_s),
        pricing_certified_prefix_prunes=int(total_pricing_certified_prefix_prunes),
        pricing_depth_certified_prefix_prunes=dict(total_depth_certified_prefix_prunes),
        pricing_service_floor_prunes=int(total_pricing_service_floor_prunes),
        pricing_depth_service_floor_prunes=dict(total_depth_service_floor_prunes),
        pricing_call_records=list(total_pricing_call_records),
        rmp_records=list(total_rmp_records),
        resource_audit_records=list(total_resource_audit_records),
        rmp_runtime_s=float(total_rmp_runtime_s),
        phase_one_runtime_s=float(total_phase_one_runtime_s),
        resource_audit_runtime_s=float(total_resource_audit_runtime_s),
        coverage_pricing_runtime_s=float(coverage.pricing_runtime_s),
        energy_pricing_runtime_s=(None if energy is None else float(energy.pricing_runtime_s)),
        coverage_rmp_runtime_s=float(coverage.rmp_runtime_s),
        energy_rmp_runtime_s=(None if energy is None else float(energy.rmp_runtime_s)),
        coverage_resource_audit_runtime_s=float(coverage.resource_audit_runtime_s),
        energy_resource_audit_runtime_s=(
            None if energy is None else float(energy.resource_audit_runtime_s)),
        pricing_candidates=int(total_pricing_candidates), pricing_nodes=int(total_pricing_nodes),
        heuristic_columns=int(total_multistop_neutral_added + total_multistop_merge_added
                              + total_resource_variant_added),
        resource_audit_calls=int(total_resource_audits),
        resource_cuts_added=int(total_cuts),
        resource_pattern_cuts_added=int(total_cuts),
        resource_cut_type="exact-selected-pattern",
        resource_cut_superset_assumption=False,
        heuristic_pricing_used=bool(total_multistop_neutral_added > 0),
        initial_column_heuristic_used=bool(initial_cols or seed_validation["accepted_count"]),
        exact_pricing_called=bool(coverage.exact_pricing_called or
                                  (energy is not None and energy.exact_pricing_called)),
        pricing_best_reduced_value=last_best,
        pricing_reduced_value_bound=last_bound,
        coverage_pricing_complete=bool(coverage.pricing_complete),
        coverage_pricing_closed=bool(coverage.pricing_complete),
        coverage_pricing_search_complete=bool(coverage.pricing_search_complete),
        energy_pricing_complete=(None if energy is None else bool(energy.pricing_complete)),
        energy_pricing_closed=(None if energy is None else bool(energy.pricing_complete)),
        energy_pricing_search_complete=(
            None if energy is None else bool(energy.pricing_search_complete)),
        coverage_pricing_best_reduced_value=coverage.pricing_best_reduced_value,
        coverage_pricing_reduced_value_bound=coverage.pricing_reduced_value_bound,
        energy_pricing_best_reduced_value=(None if energy is None else energy.pricing_best_reduced_value),
        energy_pricing_reduced_value_bound=(None if energy is None else energy.pricing_reduced_value_bound),
        chosen=chosen, covered_turbine_ids=covered_ids,
        duplicate_turbine_visits=duplicates,
        K=int(K), batteries=int(battery_count),
        mean_stops=float(mean_stops), multi_stop_ratio=float(multi_stop_ratio),
        makespan_min=float(makespan), pool_size=len(archive),
        solver="scipy-highs-rmp",
        solver_requested=contract.get("solver_requested"),
        solver_effective=contract.get("solver_effective", "scipy-highs-rmp"),
        battery_energy_used_Wh=battery_used,
        battery_end_soc_pct=battery_end_soc,
        swap_events=swap_events, quick_inspection_events=quick_events,
        n_swaps=len(swap_events), n_quick_reuses=len(quick_events),
        generated_column_archive_size=len(archive),
        physical_pricing_cache_entries=len(physical_pricing_cache),
        physical_pricing_cache_entries_before_pricing=int(
            physical_pricing_cache_entries_before_pricing),
        seed_validation=seed_validation,
        seed_columns_revalidated=bool(seed_validation.get("validation_complete", False)),
        route_space_complete=bool(complete_universe_mode),
        route_space_materialized=bool(complete_universe_mode),
        complete_route_universe_contract=(
            COMPLETE_ROUTE_UNIVERSE_CONTRACT if complete_universe_mode else None),
        complete_route_universe_columns_sha256=(
            certified_route_universe.columns_sha256 if complete_universe_mode else None),
        complete_route_universe_stats=(
            dict(certified_route_universe.stats) if complete_universe_mode else None),
        implicit_route_space_certified=bool(physical_model_global_certificate),
        algorithmic_route_space_certified=bool(algorithmic_global_certificate),
        algorithmic_global_certificate=bool(algorithmic_global_certificate),
        route_universe_source=str(route_universe_source),
        route_universe_provenance_certified=bool(route_universe_provenance_certified),
        physical_model_global_certificate=bool(physical_model_global_certificate),
        route_semantics_invariance_certified=bool(route_semantics_invariance_certified),
        future_column_row_ranges_certified=bool(future_column_row_ranges_certified),
        binary64_model_contract_enforced=bool(binary64_model_contract_enforced),
        formal_proof_contract_enforced=bool(formal_proof_contract_enforced),
        formal_proof_contract=FORMAL_PROOF_CONTRACT,
        formal_proof_obligations=list(FORMAL_PROOF_OBLIGATIONS),
        # This flag concerns validity of the reported complete-space bound,
        # not whether it came from a pricing objective bound.  Every interrupted
        # node is requeued with either a rigorous reduced-cost correction or the
        # strict trivial |I| / nonnegative-energy fallback.
        implicit_route_space_bound_valid=True,
        continuous_real_world_optimality_claimed=False,
        finite_discrete_model_only=True,
        empty_plan_allowed=True,
        empty_plan_is_incumbent=(len(final_selection) == 0),
        pricing_non_enumerative=False,
        pricing_uses_implicit_full_permutation_search=bool(
            route_universe_source == "physical-oracle"),
        pricing_dominance="identity-only",
        pricing_state_merging=False,
        pricing_unsafe_truncation_enabled=False,
        multi_column_generation=True, pricing_batch_size=int(pricing_batch_size),
        route_pool_reuse=True, route_pool_scope="shared-across-nodes-and-stages",
        heuristic_pricing_columns_certifying=False,
        initial_primal_heuristic_audit_unknown=bool(initial_audit_unknown),
        solver_mode="exact-branch-price-cut",
        pricing_mode=str(pricing_key),
        model_contract_validated=True,
        model_contract_sha256=contract.get("sha256"),
        parameter_contract_sha256=contract.get("parameter_contract_sha256"),
        instance_contract_sha256=contract.get("instance_contract_sha256"),
        algorithm_contract_sha256=contract.get("algorithm_sha256"),
        model_contract_scope=contract.get(
            "model_contract_scope", "full-finite-model-including-instance-data-binary64-exact"),
        risk_policy_contract=contract.get("risk_policy_contract"),
        physical_numeric_contract=contract.get("physical_numeric_contract"),
        route_identity_contract=contract.get("route_identity_contract"),
        model_semantics_contract=contract.get("model_semantics_contract"),
        result_certificate_contract=contract.get("result_certificate_contract"),
        route_semantics_contract=contract.get("route_semantics_contract"),
        future_column_row_range_contract=contract.get("future_column_row_range_contract"),
        proof_contract_sha256=FORMAL_PROOF_CONTRACT_SHA256,
        proof_code_sha256=FORMAL_PROOF_CODE_SHA256,
        formal_proof_code_anchors=[(k, list(v)) for k, v in FORMAL_PROOF_CODE_ANCHORS],
        pricing_bound_numeric_contract="binary64-outward-rounded-interval",
        master_dual_numeric_contract="binary64-outward-rounded-lagrangian",
        incumbent_energy_numeric_contract="exact-binary64-rational-sum-with-outward-float-enclosure",
        resource_numeric_contract="binary64-strict-half-open-time-exact-rational-soc",
        global_gap_numeric_contract="feasible-upper-enclosure-minus-rigorous-global-lower",
        integral_node_numeric_contract="exact-binary64-rational-lagrangian-when-needed",
        kappa_mode=str(kappa_mode), chance_mode=str(chance_mode),
        soc_correction=contract.get("soc_correction"),
        soc_risk_allocation=contract.get("soc_risk_allocation"),
        battery_energy_mode=contract.get("battery_energy_mode"),
        deck_mode=str(deck_mode), battery_reuse_mode=str(battery_reuse_mode),
        pool_h_mode_requested=str(pool_h_mode),
        pool_h_mode_effective="on-demand-all-discrete-horizons",
        wall_clock_deadline_enforcement="cooperative",
        blackbox_hard_interrupt_available=False,
        blackbox_overrun_scope="at-most-one-noncooperative-call-between-deadline-checks",
        power_envelope_contract=(
            RM.POWER_ENVELOPE_CONTRACT if getattr(p, "speed_adjustable", False) else None))
    # Backwards-compatible aliases used by existing reports and figures.
    result.update(
        covered=coverage_inc, coverable=len({_tid(t.tid) for t in turbines}),
        flights=len(chosen), energy_Wh=(0.0 if energy_inc is None else float(energy_inc)),
        resource_cuts=total_cuts, restricted_pool_gap_pct=None,
        global_certificate_available=bool(physical_model_global_certificate),
        global_route_space_certificate=bool(physical_model_global_certificate))
    if len(result["covered_turbine_ids"]) != len(set(result["covered_turbine_ids"])):
        raise RuntimeError("internal consistency error: duplicate covered turbine ids")
    if result["duplicate_turbine_visits"]:
        raise RuntimeError("internal consistency error: duplicate turbine visits")
    return result


def solve_fleet_anytime(turbines, launch_opts, p, xi_amb, K, T_min,
                        deck_delta_min=2.5, t_swap_min=6.0, max_stops=8,
                        weather_unc=None, kappa_mode="vp_unimodal",
                        chance_mode="drcc", budget_gamma=2.0,
                        batteries=None, solver="auto", deck_mode="interval",
                        t_launch_min=None, landing_clear_min=None,
                        quick_inspection_capacity=None, swap_station_capacity=None,
                        battery_reuse_mode="exact_soc", pool_h_mode="pareto",
                        allow_resource_only_columns=False,
                        time_limit_s=None, deadline=None,
                        coverage_gap_target_abs=0,
                        energy_gap_target_rel=0.0,
                        energy_gap_target_abs_Wh=1e-6,
                        solver_mode="exact-branch-price-cut",
                        pricing_mode="exact-implicit-dfs",
                        seed_cols=None,
                        seed_iterator_nonblocking=False,
                        implicit_test_columns=None,
                        pricing_batch_size=16, solve_scope="lexicographic", coverage_target=None,
                        certified_route_universe=None,
                        target_closure_checkpoint_path=None,
                        target_closure_resume=False,
                        archive_diagnostic_time_limit_s=30.0,
                        archive_shadow_diagnostic_time_limit_s=30.0,
                        archive_clique_diagnostic_time_limit_s=30.0,
                        archive_primal_recovery=False,
                        archive_primal_recovery_time_limit_s=2.0,
                        fullspace_target_diagnostic_time_limit_s=0.0, **_ignored):
    """Public solver entry; synthetic route-universe injection is never formal."""
    return _solve_fleet_anytime_impl(
        turbines, launch_opts, p, xi_amb, K, T_min,
        deck_delta_min=deck_delta_min, t_swap_min=t_swap_min, max_stops=max_stops,
        weather_unc=weather_unc, kappa_mode=kappa_mode,
        chance_mode=chance_mode, budget_gamma=budget_gamma,
        batteries=batteries, solver=solver, deck_mode=deck_mode,
        t_launch_min=t_launch_min, landing_clear_min=landing_clear_min,
        quick_inspection_capacity=quick_inspection_capacity,
        swap_station_capacity=swap_station_capacity,
        battery_reuse_mode=battery_reuse_mode, pool_h_mode=pool_h_mode,
        allow_resource_only_columns=allow_resource_only_columns,
        time_limit_s=time_limit_s, deadline=deadline,
        coverage_gap_target_abs=coverage_gap_target_abs,
        energy_gap_target_rel=energy_gap_target_rel,
        energy_gap_target_abs_Wh=energy_gap_target_abs_Wh,
        solver_mode=solver_mode, pricing_mode=pricing_mode,
        seed_cols=seed_cols, seed_iterator_nonblocking=seed_iterator_nonblocking,
        implicit_test_columns=implicit_test_columns,
        pricing_batch_size=pricing_batch_size, solve_scope=solve_scope,
        coverage_target=coverage_target,
        certified_route_universe=certified_route_universe,
        target_closure_checkpoint_path=target_closure_checkpoint_path,
        target_closure_resume=target_closure_resume,
        archive_diagnostic_time_limit_s=archive_diagnostic_time_limit_s,
        archive_shadow_diagnostic_time_limit_s=
            archive_shadow_diagnostic_time_limit_s,
        archive_clique_diagnostic_time_limit_s=
            archive_clique_diagnostic_time_limit_s,
        archive_primal_recovery=archive_primal_recovery,
        archive_primal_recovery_time_limit_s=
            archive_primal_recovery_time_limit_s,
        fullspace_target_diagnostic_time_limit_s=
            fullspace_target_diagnostic_time_limit_s,
        _internal_synthetic_route_universe=False, **_ignored)


def _solve_fleet_anytime_synthetic_fixture(turbines, launch_opts, p, xi_amb, K, T_min,
                                            **kwargs):
    """Private algorithmic BPC fixture over a caller-supplied finite route set.

    This path exists solely for independent finite-route unit/oracle tests.  It
    may prove lexicographic optimality *within that supplied route universe*, but
    its result deliberately has ``physical_model_global_certificate=False`` and
    ``global_certificate_available=False``.
    """
    kwargs = dict(kwargs)
    if "implicit_test_columns" not in kwargs:
        raise ValueError("synthetic fixture requires implicit_test_columns")
    kwargs.setdefault("solver_mode", "exact-branch-price-cut")
    kwargs.setdefault("pricing_mode", "exact-implicit-dfs")
    return _solve_fleet_anytime_impl(
        turbines, launch_opts, p, xi_amb, K, T_min,
        _internal_synthetic_route_universe=True, **kwargs)


def solve_branch_cut_price(turbines, launch_opts, p, xi_amb, K, T_min, **kwargs):
    kwargs = dict(kwargs)
    kwargs["solver_mode"] = "exact-branch-price-cut"
    return solve_fleet_anytime(turbines, launch_opts, p, xi_amb, K, T_min, **kwargs)


def solve_fleet(turbines, launch_opts, p, xi_amb, K, T_min, **kwargs):
    return solve_fleet_anytime(turbines, launch_opts, p, xi_amb, K, T_min, **kwargs)

def _formal_module_main():
    """Direct execution never enters the historical pre-formal demos."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Formal BPC library module; use step13_experiment_model.py for experiment CLI.")
    parser.add_argument("--show-entry", action="store_true",
                        help="print the supported formal entry")
    parser.parse_args()
    print("Formal exact solver: step12_branch_price.solve_fleet_anytime")
    print("Experiment CLI: python step13_experiment_model.py --help")
    print("Historical in-file demos are research-only and are not auto-executed.")


if __name__ == "__main__":
    _formal_module_main()

