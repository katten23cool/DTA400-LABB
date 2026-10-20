"""
Warehouse sorting simulation (SimPy) - starting skeleton.

Scenario
--------
Packages arrive at random, each with a destination. A limited number of
sorters (the servers) sort them one at a time. Every destination has a
vehicle that leaves at fixed times. A package that is sorted BEFORE the next
departure leaves on that vehicle; otherwise it waits for the following one
(= "missed departure").

Time unit: minutes.

Run:  python warehouse_sim.py
"""
import random
import statistics

import matplotlib.pyplot as plt
import simpy

# ---------------------------------------------------------------- PARAMETERS
# TODO: change these and think about WHY the results change.
SIM_TIME = 8 * 60            # packages arrive during 8 hours
DRAIN_TIME = 4 * 60          # extra time so the last packages can be shipped
MEAN_INTERARRIVAL = 1.0      # on average 1 package per minute (Poisson arrivals)
MEAN_SORT_TIME = 2.0         # average sorting time per package (exponential)

# destination -> minutes between departures
DEPARTURE_INTERVALS = {"North": 60, "South": 90, "East": 120, "West": 180}
# destination -> share of packages going there
DEST_WEIGHTS = {"North": 0.4, "South": 0.3, "East": 0.2, "West": 0.1}

N_SORTERS_RANGE = range(2, 7)   # compare 2..6 sorters
POLICIES = ["FIFO", "EDF"]      # FIFO = first in first out
                                # EDF  = earliest departure first
N_REPLICATIONS = 10             # runs per setting (different random seeds)


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
        "avg_queue_len": statistics.mean(queue_samples),
        "utilization": statistics.mean(busy_samples) / n_sorters,
        "pct_missed": 100 * counts["missed"] / counts["sorted"],
        "throughput_per_h": counts["shipped"] / (SIM_TIME / 60),
    }


# ---------------------------------------------------------------- EXPERIMENT
def experiment():
    """Run all settings, average over the replications."""
    results = {}   # (policy, n_sorters) -> {metric: (mean, stdev)}
    for policy in POLICIES:
        for n in N_SORTERS_RANGE:
            runs = [run_once(n, policy, seed) for seed in range(N_REPLICATIONS)]
            results[(policy, n)] = {
                m: (statistics.mean(r[m] for r in runs),
                    statistics.stdev(r[m] for r in runs))
                for m in runs[0]
            }
    return results


def print_table(results):
    print(f"{'policy':6} {'sorters':>7} {'queue wait':>10} {'queue len':>9} "
          f"{'util':>6} {'missed %':>9} {'dwell':>7} {'pkg/h':>6}")
    for (policy, n), m in results.items():
        print(f"{policy:6} {n:7d} {m['avg_queue_wait'][0]:10.1f} "
              f"{m['avg_queue_len'][0]:9.1f} {m['utilization'][0]:6.2f} "
              f"{m['pct_missed'][0]:9.1f} {m['avg_dwell'][0]:7.1f} "
              f"{m['throughput_per_h'][0]:6.1f}")


def plot(results, metric, ylabel, filename):
    plt.figure(figsize=(5, 3.5))
    for policy in POLICIES:
        xs = list(N_SORTERS_RANGE)
        means = [results[(policy, n)][metric][0] for n in xs]
        sds = [results[(policy, n)][metric][1] for n in xs]
        plt.errorbar(xs, means, yerr=sds, marker="o", capsize=3, label=policy)
    plt.xlabel("Number of sorters")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


if __name__ == "__main__":
    res = experiment()
    print_table(res)
    plot(res, "pct_missed", "Missed departures (%)", "missed_departures.png")
    plot(res, "avg_queue_wait", "Avg. queue wait (min)", "queue_wait.png")
    print("\nSaved: missed_departures.png, queue_wait.png")