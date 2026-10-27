"""
Packages arrive at random (Poisson process) and get a destination. N sorters
(the servers) sort them one at a time, first come, first served. Every
destination has a vehicle that leaves at fixed times. A package is sorted in
time if sorting finishes BEFORE the next departure of its destination.
Otherwise it has MISSED that departure.

All sorting-time distributions have the same mean. Only the spread changes,
measured by cv2 = variance / mean^2
    cv2 = 0 constant sorting time
    cv2 = 1 exponential (the standard M/M/c case)
    cv2 > 1 very uneven sorting times

Theory used for comparison
--------------------------
- M/M/c  : Erlang C formula (exact, for cv2 = 1)
- M/G/c  : Allen-Cunneen approximation, Wq(M/G/c) ~ Wq(M/M/c) * (1 + cv2) / 2
- Missed departures: our own approximation, see theory_missed_pct()

Time unit: minutes.

HOW TO RUN
-----------
Go into the folder where this file is and run:
    pip install -r requirements.txt
Then:
    python warehouse_sim.py
"""
import math
import random
import statistics

import matplotlib.pyplot as plt
import simpy

# ---------------------------------------------------------------- PARAMETERS
# TODO: change these and think about WHY the results change.
N_SORTERS = 3
MEAN_SORT_TIME = 2.0         # minutes, the same for every distribution

SIM_TIME = 24 * 60           # packages arrive for 24 hours ...
WARMUP = 2 * 60              # ... but the first 2 hours are not measured
EXTRA_TIME = 2 * 60          # extra time so the last packages can be sorted

# destination -> minutes between departures
DEPARTURE_INTERVALS = {"North": 60, "South": 90, "East": 120, "West": 180}
# destination -> share of packages going there
DEST_WEIGHTS = {"North": 0.4, "South": 0.3, "East": 0.2, "West": 0.1}

N_REPLICATIONS = 30          # runs per setting (different random seeds)
CV2_VALUES = [0, 0.5, 1, 2, 4]
MAIN_RHO = 0.85              # main experiment: utilization of the sorters
RHO_LEVELS = [0.6, 0.75, 0.9]  # extra experiment: different loads

# t-values for 95 % confidence intervals, keyed by degrees of freedom
T_CRIT_95 = {9: 2.262, 14: 2.145, 19: 2.093, 24: 2.064, 29: 2.045, 49: 2.010}


# ---------------------------------------------------------------- MODEL
def next_departure(t, interval):
    """Time of the first departure strictly after time t."""
    return (int(t // interval) + 1) * interval


def sample_sort_time(rng, mean, cv2):
    """Random sorting time with the given mean and cv2 (variance / mean^2)."""
    if cv2 == 0:
        return mean                                  # constant
    if cv2 == 1:
        return rng.expovariate(1 / mean)             # exponential
    # gamma distribution: mean = shape * scale, cv2 = 1 / shape
    return rng.gammavariate(1 / cv2, mean * cv2)


def run_once(cv2, rho, seed, n_sorters=N_SORTERS):
    """One simulation run. Returns the measured metrics.

    rho is the target utilization of the sorters; it decides the arrival
    rate through  lambda = rho * n_sorters / MEAN_SORT_TIME.
    Only packages that ARRIVE after the warm-up are measured.
    """
    rng = random.Random(seed)
    lam = rho * n_sorters / MEAN_SORT_TIME           # packages per minute
    env = simpy.Environment()
    sorters = simpy.Resource(env, capacity=n_sorters)

    waits, missed = [], []         # one entry per measured, sorted package
    busy_samples = []
    counts = {"arrived": 0}
    destinations = list(DEST_WEIGHTS)
    weights = list(DEST_WEIGHTS.values())

    def package(dest):
        """Life of one package: wait for a sorter -> get sorted."""
        arrival = env.now
        deadline = next_departure(arrival, DEPARTURE_INTERVALS[dest])
        with sorters.request() as req:
            yield req                                # queue for a sorter
            wait = env.now - arrival
            yield env.timeout(sample_sort_time(rng, MEAN_SORT_TIME, cv2))
        if arrival >= WARMUP:                        # only measure after warm-up
            waits.append(wait)
            missed.append(env.now > deadline)        # too late for the vehicle?

    def arrivals():
        while True:
            yield env.timeout(rng.expovariate(lam))
            if env.now >= SIM_TIME:
                break
            if env.now >= WARMUP:
                counts["arrived"] += 1
            env.process(package(rng.choices(destinations, weights)[0]))

    def monitor():
        """Sample the number of busy sorters every minute (after warm-up)."""
        while env.now < SIM_TIME:
            if env.now >= WARMUP:
                busy_samples.append(sorters.count)
            yield env.timeout(1)

    env.process(arrivals())
    env.process(monitor())
    env.run(until=SIM_TIME + EXTRA_TIME)

    return {
        "wq": statistics.mean(waits),
        "pct_missed": 100 * sum(missed) / len(missed),
        "utilization": statistics.mean(busy_samples) / n_sorters,
        "unfinished": counts["arrived"] - len(missed),   # should be 0
    }


# ---------------------------------------------------------------- THEORY
def erlang_c(c, lam):
    """Probability that a package has to wait in an M/M/c queue."""
    a = lam * MEAN_SORT_TIME                         # offered load
    rho = a / c
    if rho >= 1:
        return None                                  # unstable queue
    top = a ** c / math.factorial(c) / (1 - rho)
    bottom = sum(a ** k / math.factorial(k) for k in range(c)) + top
    return top / bottom


def theory_wq(c, lam, cv2):
    """Mean wait in the queue.
    M/M/c: Wq = C / (c*mu - lambda).
    M/G/c: multiply by (1 + cv2) / 2 (Allen-Cunneen approximation)."""
    mu = 1 / MEAN_SORT_TIME
    C = erlang_c(c, lam)
    return None if C is None else C / (c * mu - lam) * (1 + cv2) / 2


def theory_missed_pct(c, lam, cv2):
    """Approximate % of missed departures.

    A package arrives at a random moment, so the time U until the next
    departure of its destination is uniform on (0, I), I = the interval.
    It misses the departure if its time in the sorting system
    T (= wait + sorting) is larger than U:
        P(miss) = E[min(T, I)] / I   which is about   E[T] / I
    when T is much smaller than I. Then average over the destinations.
    TODO: explain in your own words why this holds and when it breaks down.
    """
    wq = theory_wq(c, lam, cv2)
    if wq is None:
        return None
    mean_T = wq + MEAN_SORT_TIME
    return 100 * mean_T * sum(
        DEST_WEIGHTS[d] / DEPARTURE_INTERVALS[d] for d in DEST_WEIGHTS)


# ---------------------------------------------------------------- HELPERS
def mean_ci(values):
    """Mean and half-width of the 95 % confidence interval."""
    n = len(values)
    half = T_CRIT_95.get(n - 1, 1.96) * statistics.stdev(values) / math.sqrt(n)
    return statistics.mean(values), half


def replicate(cv2, rho):
    """Run N_REPLICATIONS runs and summarize each metric."""
    runs = [run_once(cv2, rho, seed) for seed in range(N_REPLICATIONS)]
    out = {m: mean_ci([r[m] for r in runs]) for m in ("wq", "pct_missed", "utilization")}
    out["unfinished"] = sum(r["unfinished"] for r in runs)
    return out


def lam_from_rho(rho):
    return rho * N_SORTERS / MEAN_SORT_TIME


# ---------------------------------------------------------------- EXPERIMENTS
def validation():
    """Check the simulation against exact M/M/c theory (cv2 = 1)."""
    print("1) VALIDATION: simulation vs Erlang C (exponential sorting times)")
    print(f"{'rho':>5} {'util sim':>9} {'Wq theory':>10} {'Wq sim':>14}")
    for rho in (0.5, 0.7, 0.85):
        r = replicate(1, rho)
        print(f"{rho:5.2f} {r['utilization'][0]:9.2f} "
              f"{theory_wq(N_SORTERS, lam_from_rho(rho), 1):10.2f} "
              f"{r['wq'][0]:8.2f} +-{r['wq'][1]:.2f}")


def main_experiment():
    """Main result: vary cv2 at a fixed load."""
    print(f"\n2) MAIN EXPERIMENT: rho = {MAIN_RHO}, {N_SORTERS} sorters")
    print(f"{'cv2':>4} {'Wq theory':>10} {'Wq sim':>14} "
          f"{'missed% theory':>15} {'missed% sim':>15}")
    lam = lam_from_rho(MAIN_RHO)
    rows = []
    for cv2 in CV2_VALUES:
        r = replicate(cv2, MAIN_RHO)
        t_wq, t_miss = theory_wq(N_SORTERS, lam, cv2), theory_missed_pct(N_SORTERS, lam, cv2)
        rows.append((cv2, r, t_wq, t_miss))
        print(f"{cv2:4} {t_wq:10.2f} {r['wq'][0]:8.2f} +-{r['wq'][1]:.2f} "
              f"{t_miss:15.1f} {r['pct_missed'][0]:9.1f} +-{r['pct_missed'][1]:.1f}")
        if r["unfinished"]:
            print(f"     warning: {r['unfinished']} packages were not sorted")
    return rows


def load_experiment():
    """Extra: does the effect of cv2 depend on the load?"""
    print("\n3) LOAD EXPERIMENT: missed departures (%) for different loads")
    print(f"{'rho':>5} " + " ".join(f"cv2={c:<5}" for c in CV2_VALUES))
    results = {}
    for rho in RHO_LEVELS:
        results[rho] = [replicate(cv2, rho)["pct_missed"] for cv2 in CV2_VALUES]
        print(f"{rho:5.2f} " + " ".join(f"{m:9.1f}" for m, _ in results[rho]))
    return results


# ---------------------------------------------------------------- FIGURES
def figure_main(rows, metric, theory_index, ylabel, filename):
    xs = [r[0] for r in rows]
    sim = [r[1][metric] for r in rows]
    plt.figure(figsize=(5, 3.5))
    plt.errorbar(xs, [m for m, _ in sim], yerr=[h for _, h in sim],
                 marker="o", capsize=3, label="Simulation (95 % CI)")
    plt.plot(xs, [r[theory_index] for r in rows], "s--", label="Theory")
    plt.xlabel("Squared coefficient of variation of sorting time, cv$^2$")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def figure_load(results, filename):
    plt.figure(figsize=(5, 3.5))
    for rho, vals in results.items():
        plt.errorbar(CV2_VALUES, [m for m, _ in vals], yerr=[h for _, h in vals],
                     marker="o", capsize=3, label=f"rho = {rho}")
    plt.xlabel("Squared coefficient of variation of sorting time, cv$^2$")
    plt.ylabel("Missed departures (%)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


if __name__ == "__main__":
    validation()
    rows = main_experiment()
    load_results = load_experiment()
    figure_main(rows, "wq", 2, "Mean wait in queue (min)", "wait_vs_cv2.png")
    figure_main(rows, "pct_missed", 3, "Missed departures (%)", "missed_vs_cv2.png")
    figure_load(load_results, "missed_vs_load.png")
    print("\nSaved: wait_vs_cv2.png, missed_vs_cv2.png, missed_vs_load.png")