"""
Warehouse sorting simulation (SimPy) + cost analysis - standalone version.
 
Research question
-----------------
At what staffing level does the marginal benefit of an extra sorter
stop justifying its cost?
 
Scenario
--------
Packages arrive at random, each with a destination. A limited number of
sorters (the servers) sort them one at a time. Every destination has a
vehicle that leaves at fixed times. A package that is sorted BEFORE the next
departure leaves on that vehicle; otherwise it waits for the following one
(= "missed departure").
 
Cost model (per 8-hour shift)
-----------------------------
    TotalCost(n) = n * WAGE_PER_HOUR * SHIFT_HOURS  +  MISS_PENALTY * Missed(n)
 
    Marginal cost of sorter n+1    = WAGE_PER_HOUR * SHIFT_HOURS
    Marginal benefit of sorter n+1 = MISS_PENALTY * (Missed(n) - Missed(n+1))
    Break-even penalty p*          = marginal cost / misses saved
 
Time unit: minutes.
Run:  python Cost_sims.py          (needs: pip install simpy)
Plot is saved only if matplotlib is installed (optional).
"""
import random
import statistics
 
import simpy
 
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
 
# ---------------------------------------------------------------- MODEL PARAMETERS
# Same values as the shared base model.
SIM_TIME = 8 * 60            # packages arrive during 8 hours
DRAIN_TIME = 4 * 60          # extra time so the last packages can be shipped
MEAN_INTERARRIVAL = 1.0      # on average 1 package per minute (Poisson arrivals)
MEAN_SORT_TIME = 2.0         # average sorting time per package (exponential)
 
# destination -> minutes between departures
DEPARTURE_INTERVALS = {"North": 60, "South": 90, "East": 120, "West": 180}
# destination -> share of packages going there
DEST_WEIGHTS = {"North": 0.4, "South": 0.3, "East": 0.2, "West": 0.1}
 
# ---------------------------------------------------------------- COST ASSUMPTIONS
# State these clearly in the report - the result depends on them.
SHIFT_HOURS = SIM_TIME / 60             # 8 h
WAGE_PER_HOUR = 300                     # SEK/h incl. employer fees (assumption)
MISS_PENALTIES = [25, 100, 400]         # SEK per missed departure (scenarios)
POLICY = "FIFO"                         # "FIFO" or "EDF"
N_REPLICATIONS = 30                     # runs per setting (different seeds)
SORTERS = list(range(3, 7))             # 2 sorters is unstable (rho = 1), skipped
T_95 = 2.045                            # t-value, 95 % CI, 29 degrees of freedom
 
 
# ---------------------------------------------------------------- MODEL
def next_departure(t, interval):
    """Time of the first departure strictly after time t."""
    return (int(t // interval) + 1) * interval
 
 
def run_once(n_sorters, policy, seed):
    """Run one simulation and return a dict with the measured metrics."""
    random.seed(seed)
    env = simpy.Environment()
 
    if policy == "FIFO":
        sorters = simpy.Resource(env, capacity=n_sorters)
    else:  # EDF: lower priority number is served first
        sorters = simpy.PriorityResource(env, capacity=n_sorters)
 
    ready = {d: [] for d in DEPARTURE_INTERVALS}   # sorted, waiting for vehicle
    queue_waits, dwell_times = [], []
    counts = {"sorted": 0, "missed": 0, "shipped": 0}
    busy_samples, queue_samples = [], []
 
    destinations = list(DEST_WEIGHTS)
    weights = list(DEST_WEIGHTS.values())
 
    def package(dest):
        """Life of one package: queue -> sorting -> wait for vehicle."""
        arrival = env.now
        deadline = next_departure(arrival, DEPARTURE_INTERVALS[dest])
 
        if policy == "FIFO":
            req = sorters.request()
        else:
            req = sorters.request(priority=deadline)
 
        with req:
            yield req                                    # wait for a free sorter
            queue_waits.append(env.now - arrival)
            yield env.timeout(random.expovariate(1 / MEAN_SORT_TIME))
 
        counts["sorted"] += 1
        if env.now > deadline:                           # too late for that vehicle
            counts["missed"] += 1
        ready[dest].append(arrival)
 
    def arrivals():
        while env.now < SIM_TIME:
            yield env.timeout(random.expovariate(1 / MEAN_INTERARRIVAL))
            dest = random.choices(destinations, weights)[0]
            env.process(package(dest))
 
    def vehicle(dest):
        """Leaves every `interval` minutes and takes all sorted packages."""
        interval = DEPARTURE_INTERVALS[dest]
        while True:
            yield env.timeout(next_departure(env.now, interval) - env.now)
            for arrival in ready[dest]:
                dwell_times.append(env.now - arrival)
                counts["shipped"] += 1
            ready[dest].clear()
 
    def monitor():
        """Sample queue length and number of busy sorters every minute."""
        while env.now < SIM_TIME:
            queue_samples.append(len(sorters.queue))
            busy_samples.append(sorters.count)
            yield env.timeout(1)
 
    env.process(arrivals())
    env.process(monitor())
    for d in DEPARTURE_INTERVALS:
        env.process(vehicle(d))
    env.run(until=SIM_TIME + DRAIN_TIME)
 
    return {
        "avg_queue_wait": statistics.mean(queue_waits),
        "avg_dwell": statistics.mean(dwell_times),
        "utilization": statistics.mean(busy_samples) / n_sorters,
        "pct_missed": 100 * counts["missed"] / counts["sorted"],
        "n_missed": counts["missed"],
    }
 
 
# ---------------------------------------------------------------- COST ANALYSIS
def ci(values):
    """Mean and half-width of a 95 % confidence interval."""
    return (statistics.mean(values),
            T_95 * statistics.stdev(values) / len(values) ** 0.5)
 
 
def main():
    # Run every staffing level with the same seeds (paired comparison)
    runs = {n: [run_once(n, POLICY, seed) for seed in range(N_REPLICATIONS)]
            for n in SORTERS}
    missed = {n: [r["n_missed"] for r in runs[n]] for n in SORTERS}
 
    labour = WAGE_PER_HOUR * SHIFT_HOURS     # cost of one extra sorter per shift
 
    print(f"Policy {POLICY}, {N_REPLICATIONS} replications, "
          f"one sorter costs {labour:.0f} SEK/shift\n")
 
    # ---------------- basic performance
    print(f"{'sorters':>7} {'queue wait':>11} {'util':>6} {'missed %':>9} "
          f"{'missed/shift (95% CI)':>24}")
    for n in SORTERS:
        wait = statistics.mean(r["avg_queue_wait"] for r in runs[n])
        util = statistics.mean(r["utilization"] for r in runs[n])
        pct = statistics.mean(r["pct_missed"] for r in runs[n])
        m, h = ci(missed[n])
        print(f"{n:7d} {wait:11.2f} {util:6.2f} {pct:9.1f} {m:13.1f} +/- {h:.1f}")
 
    # ---------------- marginal benefit and break-even penalty
    print(f"\n{'step':>7} {'missed saved/shift (95% CI)':>30} {'break-even penalty':>20}")
    for n in SORTERS[:-1]:
        diffs = [a - b for a, b in zip(missed[n], missed[n + 1])]
        mean, half = ci(diffs)
        p_star = f"{labour / mean:,.0f} SEK" if mean > 0 else "never pays off"
        print(f"{n}->{n + 1:<4} {mean:16.1f} +/- {half:<10.1f} {p_star:>20}")
 
    # ---------------- total cost per penalty scenario
    print(f"\n{'sorters':>7}", end="")
    for p in MISS_PENALTIES:
        print(f"{'cost @' + str(p):>12}", end="")
    print()
    best = {}
    for n in SORTERS:
        m = statistics.mean(missed[n])
        print(f"{n:7d}", end="")
        for p in MISS_PENALTIES:
            cost = n * labour + p * m
            print(f"{cost:12,.0f}", end="")
            if p not in best or cost < best[p][1]:
                best[p] = (n, cost)
        print()
    print()
    for p, (n, _) in best.items():
        print(f"Penalty {p} SEK per missed package -> optimal: {n} sorters")
 
    # ---------------- plot (optional)
    if plt is None:
        print("\nmatplotlib not installed - skipping plot.")
        return
    plt.figure(figsize=(5, 3.5))
    for p in MISS_PENALTIES:
        costs = [n * labour + p * statistics.mean(missed[n]) for n in SORTERS]
        plt.plot(SORTERS, costs, marker="o", label=f"{p} SEK / missed")
    plt.xlabel("Number of sorters")
    plt.ylabel("Total cost per shift (SEK)")
    plt.xticks(SORTERS)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("total_cost.png", dpi=200)
    print("\nSaved: total_cost.png")
 
 
if __name__ == "__main__":
    main()