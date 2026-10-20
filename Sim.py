
import csv
import math
import random
from dataclasses import dataclass

# =====================================
# SIMULATION SETTINGS
# =====================================

SHIFT_MINUTES = 480          # 8-hour shift
ARRIVAL_RATE = 0.70          # Average packages arriving per minute
DESTINATIONS = ["A", "B", "C"]

SORT_MIN = 4.0               # Minimum sorting time
SORT_MAX = 8.0               # Maximum sorting time

FIRST_DEPARTURE = 120.0      # First truck leaves after 2 hours
DEPARTURE_INTERVAL = 120.0  # Trucks leave every 2 hours
CUTOFF_MINUTES = 10.0        # Package must be ready 10 min before

REPLICATIONS = 30            # Repeat simulation 30 times
WORKER_OPTIONS = range(1, 9) # Test 1 to 8 workers
SEED = 2026


# =====================================
# PACKAGE CLASS
# =====================================

@dataclass
class Package:
    id: int
    destination: str
    arrival: float
    sort_duration: float

    sorting_start: float = 0.0
    sorting_end: float = 0.0
    planned_departure: float = 0.0
    actual_departure: float = 0.0
    missed_first_departure: bool = False


# =====================================
# TRUCK DEPARTURES
# =====================================

def next_departure(after_time):

    if after_time < FIRST_DEPARTURE:
        return FIRST_DEPARTURE

    cycles = math.floor(
        (after_time - FIRST_DEPARTURE) / DEPARTURE_INTERVAL
    ) + 1

    return FIRST_DEPARTURE + cycles * DEPARTURE_INTERVAL


# =====================================
# GENERATE RANDOM PACKAGES
# =====================================

def generate_packages(seed):

    rng = random.Random(seed)
    packages = []
    minute = 0.0

    while True:

        # Exponential gaps create Poisson arrivals
        minute += rng.expovariate(ARRIVAL_RATE)

        if minute >= SHIFT_MINUTES:
            break

        package = Package(
            id=len(packages) + 1,
            destination=rng.choice(DESTINATIONS),
            arrival=minute,
            sort_duration=rng.uniform(SORT_MIN, SORT_MAX)
        )

        packages.append(package)

    return packages


# =====================================
# WAREHOUSE SIMULATION
# =====================================

def simulate(num_workers, seed):

    packages = generate_packages(seed)

    # When each worker becomes available
    free_at = [0.0] * num_workers

    # Events for measuring queue length
    queue_events = []

    for package in packages:

        # Find worker who becomes available first
        worker = min(
            range(num_workers),
            key=lambda i: free_at[i]
        )

        # Start sorting when worker is free
        package.sorting_start = max(
            package.arrival,
            free_at[worker]
        )

        package.sorting_end = (
            package.sorting_start + package.sort_duration
        )

        free_at[worker] = package.sorting_end

        # First truck after package arrival
        package.planned_departure = next_departure(
            package.arrival
        )

        planned_cutoff = (
            package.planned_departure - CUTOFF_MINUTES
        )

        # Did sorting finish too late?
        package.missed_first_departure = (
            package.sorting_end > planned_cutoff
        )

        # Actual truck the package can catch
        package.actual_departure = next_departure(
            max(
                package.arrival,
                package.sorting_end + CUTOFF_MINUTES
            ) - 1e-9
        )

        # Record time spent in queue
        if package.sorting_start > package.arrival:

            queue_events.append(
                (package.arrival, 1)
            )

            queue_events.append(
                (package.sorting_start, -1)
            )

    # =====================================
    # QUEUE STATISTICS
    # =====================================

    queue_events.sort()

    last_time = 0.0
    queue_size = 0
    queue_area = 0.0
    max_queue = 0

    for time, change in queue_events:

        clipped = min(time, SHIFT_MINUTES)

        if clipped > last_time:
            queue_area += queue_size * (clipped - last_time)
            last_time = clipped

        if time >= SHIFT_MINUTES:
            break

        queue_size += change
        max_queue = max(max_queue, queue_size)

    queue_area += queue_size * (SHIFT_MINUTES - last_time)

    # =====================================
    # CALCULATE RESULTS
    # =====================================

    n = len(packages)

    missed = sum(
        p.missed_first_departure for p in packages
    )

    sorted_during_shift = sum(
        p.sorting_end <= SHIFT_MINUTES for p in packages
    )

    shipped_during_shift = sum(
        p.actual_departure <= SHIFT_MINUTES for p in packages
    )

    busy_during_shift = sum(
        max(
            0.0,
            min(p.sorting_end, SHIFT_MINUTES)
            - p.sorting_start
        )
        for p in packages
    )

    results = {
        "workers": num_workers,
        "arrived": n,
        "sorted_during_shift": sorted_during_shift,
        "shipped_during_shift": shipped_during_shift,

        "average_wait_min": (
            sum(
                p.sorting_start - p.arrival
                for p in packages
            ) / n if n else 0
        ),

        "average_sort_min": (
            sum(p.sort_duration for p in packages) / n
            if n else 0
        ),

        "average_queue_length": queue_area / SHIFT_MINUTES,
        "maximum_queue_length": max_queue,

        "worker_utilization_pct": (
            100 * busy_during_shift /
            (num_workers * SHIFT_MINUTES)
        ),

        "missed_count": missed,
        "missed_pct": 100 * missed / n if n else 0,

        "throughput_per_hour": (
            shipped_during_shift / (SHIFT_MINUTES / 60)
        ),
    }

    return results, packages


# =====================================
# RUN EXPERIMENTS
# =====================================

def run_experiment():

    print("WAREHOUSE SIMULATION")
    print("====================")

    print(f"Shift: {SHIFT_MINUTES} minutes")
    print(f"Arrival rate: {ARRIVAL_RATE * 60:.1f} packages/hour")
    print(f"Sorting time: {SORT_MIN}-{SORT_MAX} minutes")
    print(f"Truck interval: {DEPARTURE_INTERVAL} minutes")
    print(f"Repetitions: {REPLICATIONS}")
    print()

    print(
        f"{'Workers':>7} "
        f"{'Arrivals':>9} "
        f"{'Wait(min)':>11} "
        f"{'Queue':>8} "
        f"{'Util %':>8} "
        f"{'Missed %':>9} "
        f"{'Shipped/hr':>11}"
    )

    print("-" * 73)

    rows = []

    for workers in WORKER_OPTIONS:

        runs = []

        for run in range(REPLICATIONS):

            results, packages = simulate(
                workers,
                SEED + run
            )

            runs.append(results)

        averaged = {}

        for key in runs[0]:

            if key != "workers":

                averaged[key] = (
                    sum(r[key] for r in runs) / len(runs)
                )

        averaged["workers"] = workers
        rows.append(averaged)

        print(
            f"{workers:7d} "
            f"{averaged['arrived']:9.1f} "
            f"{averaged['average_wait_min']:11.2f} "
            f"{averaged['average_queue_length']:8.2f} "
            f"{averaged['worker_utilization_pct']:8.1f} "
            f"{averaged['missed_pct']:9.2f} "
            f"{averaged['throughput_per_hour']:11.2f}"
        )

    # =====================================
    # RESEARCH QUESTION
    # =====================================

    successful = [
        r for r in rows
        if r["missed_pct"] < 5.0
    ]

    if successful:

        best = successful[0]

        print(
            f"\nMinimum workers needed: {best['workers']}"
        )

    else:

        print("\nNo tested worker count achieved under 5%.")
        print(
            "Some packages arrive too late to catch "
            "their first truck."
        )

    # =====================================
    # SAVE RESULTS TO CSV
    # =====================================

    with open(
        "warehouse_results.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0])
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nResults saved to warehouse_results.csv")


# =====================================
# START PROGRAM
# =====================================

if __name__ == "__main__":
    run_experiment()
