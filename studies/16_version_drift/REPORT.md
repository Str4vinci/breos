# Study 16 — Do results drift across Python and dependency versions?

**Question.** BREOS supports Python ≥ 3.11.5 and ranges of numpy, pandas, pvlib and scipy. The gallery
results are stored from one environment. How far do results move in another?

**Answer.** Not at all across Python versions: with the locked dependencies, Python 3.11, 3.12 and
3.13 give the same numbers to 2.6e-14 in every case. pandas 2.3.3 and pvlib 0.14 change nothing
either. numpy is what moves results: numpy 2.2 and the lowest allowed versions shift energy and money
values by at most 1.55 € (one in a thousand), which the tool's `--cross-machine` tolerance doesn't
always accept. They also send the NSGA-II search down a different path. The returned front has
32–35 of 40 designs different, but the same end points and 99.8 % of the stored front's hypervolume.
No discrete result (replacement count, payback year) changed anywhere.

## Environments

| Env | Python | numpy | pandas | pvlib | scipy | numba |
|---|---|---|---|---|---|---|
| lock-py311/312/313 | 3.11.15 / 3.12.13 / 3.13.14 | 2.4.6 | 3.0.5 | 0.15.2 | 1.17.1 | 0.66.0 |
| pandas233-py313 | 3.13 | 2.4.6 | 2.3.3 | 0.15.2 | 1.17.1 | 0.66.0 |
| pvlib014-py313 | 3.13 | 2.4.6 | 3.0.5 | 0.14.0 | 1.17.1 | 0.66.0 |
| numpy22-py313 | 3.13 | 2.2.6 | 3.0.5 | 0.15.2 | 1.17.1 | 0.66.0 |
| lowest-py311 | 3.11 | 2.0.0 | 2.3.3 | 0.14.0 | 1.14.0 | 0.63.1 |
| highest-py313 | 3.13 | 2.5.5 | 3.1.1 | 0.15.2 | 1.18.1 | 0.68.1 |

All runs used the tool's numpy X86_V3 kernel limit (AVX2), as the stored results do.

## Results

| Env | Cases identical (1e-9) | Accepted by --cross-machine | Largest change outside NSGA-II |
|---|---|---|---|
| lock-py311/312/313, pandas 2.3.3, pvlib 0.14 | 14/14 | 14/14 | 2.6e-14 relative |
| highest | 11/14 | 13/14 | 0.01 € (clipping, east-west) |
| numpy 2.2 | 6/14 | 11/14 | 0.06 kWh / 0.03 € |
| lowest | 1/14 | 5/14 | 1.55 € (replacement_timing), 0.2 % SOC |

NSGA-II front (gallery `nsga2_front`): identical designs in the six environments with numpy 2.4.6 or
2.5.5. With numpy 2.2 or 2.0, 32–35 of 40 designs differ, the extreme NPV and grid-independence
points agree within 3 €, and the front's hypervolume is 0.998 of the stored one. A row-by-row
comparison reports these as 52× changes, which misreads a different design in the same row as drift.

## Takeaways for BREOS

1. The cent-level gallery drift seen earlier on this machine was not a Python-version effect. With
   locked dependencies, 3.11–3.13 are bit-identical. Different numpy versions cause it.
2. CI's lowest-direct job runs only the test suite, not the gallery comparison, so nothing
   fails today. If a gallery or golden check is ever added against older numpy, it needs
   tolerances wider than `--cross-machine` (about 1e-3 relative for money and energy) and
   set-level checks (hypervolume, extremes) for the NSGA-II front rather than row by row.
3. pandas and pvlib across their supported ranges don't change any number in the gallery.
