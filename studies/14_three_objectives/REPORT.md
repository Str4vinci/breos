# Study 14 — Would BREOS's NSGA-II cope with a third objective?

**Question.** The optimizer has two objectives (grid dependence, −NPV). Users ask for emissions or
investment cost as well. If a third objective were added with today's settings, would NSGA-II still
find the front?

**Answer.** Only with the opt-in front of every evaluated design. With the shipped defaults, the
returned front loses about 1.5 % of hypervolume with three objectives (median HV ratio 0.984,
5th percentile 0.971), against 0.4 % with two. The cause is structural: the exact 3D fronts have
85–252 points, and a population of 40 cannot hold them. The front of every evaluated design
(`pareto_front = "all_evaluated"`, 0.8.1 opt-in) gets back to a median of 0.998 at the same 820
simulations. A larger population and more generations reach 0.999+.

## Setup

The az180 design space of the NSGA-II TOU study (modules 1–20 × battery 0–30 kWh × tilt, south
facing, 6,820 designs, budget and area constraints) for 5 cases × 2 tariffs, enumerated exactly
with every metric. NSGA-II is replayed on the tables with BREOS's own operators, repair and
termination, so a run is identical to a real optimizer run (verified in the earlier study) but costs
no simulations. Objective sets: the shipped pair; plus −CO2 avoided over the project; or plus
initial cost. Budgets: shipped (pop 40, 20 offspring, 40 generations, ftol stop), shipped without
the stop, 200 generations, and pop 100 for 200 generations. 50 seeds each. Fronts normalised to
the exact front with a 10 % margin.

## Results

| Objectives | Exact front size | Returned front, shipped budget (median / p5) | All evaluated, shipped budget (median / p5 / min) | All evaluated, pop 100 × 200 gen |
|---|---|---|---|---|
| 2 (shipped) | 37–175 | 0.9955 / 0.984 | 0.9988 / 0.984 / 0.961 | 1.0000 |
| + CO2 avoided | 67–205 | 0.9847 / 0.974 | 0.9987 / 0.986 / 0.973 | 1.0000 |
| + initial cost | 85–252 | 0.9840 / 0.971 | 0.9983 / 0.993 / 0.972 | 1.0000 |

- The ftol early stop never triggered within the 40-generation cap. The shipped and no-stop budgets
  are identical.
- 200 generations at pop 40 doesn't fix the returned front for three objectives (0.984–0.986). The
  population size is the limit, not the run length.
- As with two objectives, the high-load household is hardest (all-evaluated median 0.990, p5 0.976
  with CO2). Its front is small and sits at the edge of the budget constraint.

## Takeaways for BREOS

1. Adding a third objective is feasible with the current algorithm, but `pareto_front =
   "all_evaluated"` should then be the default for it, or the population should grow with the
   number of objectives.
2. CO2 avoided as an objective mostly rewards more modules. It adds front points without making
   the search harder than initial cost does.
3. The two-objective result repeats the earlier NSGA-II study (0.991–0.997) on the current code.
