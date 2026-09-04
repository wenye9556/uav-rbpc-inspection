# K=1 profiling diagnostics

Provenance: cProfile instrumentation of the K=1 accelerated run
(E1_k1_v31, 10 h budget), recorded 2026-08-27/28.

* `k1_v31*.prof` — raw cProfile binaries (authoritative source).
* `k1_pstats_top.txt` / `k1_pstats_callers.txt` / `k1_pstats_repr_fp.txt` —
  custom text summaries exported from the same binaries (2026-08-28).
* `verify_k1_pstats.py` — one-command machine check of the paper's
  "19.4 million string-rendering calls / about 9,000 s" claim.

Note: some entries in the raw profile carry negative call counts
(cProfile bookkeeping quirk on generator/C-level re-entrant frames);
the entries are present in the binary itself and are reproduced verbatim
in the text summaries.
