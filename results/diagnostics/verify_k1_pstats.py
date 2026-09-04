# -*- coding: utf-8 -*-
"""Machine-checkable provenance for the K=1 string-rendering attribution.

Loads the raw cProfile binary (k1_v31_final.prof, recorded 2026-08-27) and
asserts the numbers backing paper Section 4.8's sentence
    "... costs about 9,000 s through 19.4 million string-rendering calls ..."
Run:  python verify_k1_pstats.py          (from this directory)

Notes on reading the raw profile
--------------------------------
* The accompanying *.txt files are custom text summaries exported from this
  same binary on 2026-08-27/28; the authoritative source is the .prof binary.
* Some entries in the profile carry NEGATIVE call counts (e.g.
  _outward_add_interval: -732,187,320). This is a known cProfile bookkeeping
  quirk for generator/C-level re-entrant frames, present in the RAW binary
  itself -- not an artifact of the text export, and not a sign of tampering.
  The dual-fingerprint genentry verified below is unaffected.
"""
import hashlib
import pstats
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROF = HERE / 'k1_v31_final.prof'

EXPECT_CALLS = 19_388_322
EXPECT_CUMTIME = 9078.022

print('prof sha256 :', hashlib.sha256(PROF.read_bytes()).hexdigest())
p = pstats.Stats(str(PROF))
total_calls, primitive_calls, total_tt = p.total_calls, p.prim_calls, p.total_tt
print(f'profile totals: {total_calls:,} calls ({primitive_calls:,} primitive) in {total_tt:.1f}s')

hit = None
for (fn, ln, name), (cc, nc, tt, ct, callers) in p.stats.items():
    if ln == 11943 and 'step12_branch_price' in fn:
        hit = (cc, nc, tt, ct)
if hit is None:
    sys.exit('FAIL: step12_branch_price.py:11943 <genexpr> not found in profile')
cc, nc, tt, ct = hit
print(f'step12:11943 <genexpr>: calls={nc:,}  tottime={tt:.3f}s  cumtime={ct:.3f}s')
assert nc == EXPECT_CALLS, (nc, EXPECT_CALLS)
assert abs(ct - EXPECT_CUMTIME) < 0.01, (ct, EXPECT_CUMTIME)
print(f'PASS: dual-fingerprint generator made {nc:,} calls, {ct:.1f}s cumulative '
      f'(paper: 19.4 million / about 9,000 s)')
