"""Benchmarking script for the pandas truncation utilities.

Times :func:`~tmlt.core.utils.pandas_truncation.truncate_large_groups`,
:func:`~tmlt.core.utils.pandas_truncation.drop_large_groups`, and
:func:`~tmlt.core.utils.pandas_truncation.limit_keys_per_group` against their
Spark counterparts in :mod:`tmlt.core.utils.truncation`, over frames whose only
variable is the group-size distribution:

* ``worst-case``: ~10 rows per group, threshold 3, so nearly every row is in
  an oversized group. Pure hash-and-sort throughput; a fast path that skips
  rows in small groups cannot help here.
* ``realistic``: geometric group sizes (mean ~3.5), threshold 5, the shape
  differentially private inputs usually have.
* ``all-under``: ~10 rows per group, threshold 100, so no group is oversized.
* ``wide-pairs`` (``limit_keys_per_group`` only): ~4 rows per (group, key)
  pair, threshold 2, the shape where hashing once per pair rather than once
  per row pays.

Every timing is printed next to the measured fraction of rows in oversized
groups, without which the numbers are uninterpretable.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2026

import argparse
import cProfile
import platform
import pstats
import sys
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from benchmarking_utils import Timer, write_as_csv, write_as_html

from tmlt.core.utils import pandas_truncation

SEED = 20260809

SIZES = (1_000, 10_000, 100_000, 1_000_000)

DISTRIBUTIONS = ("worst-case", "realistic", "all-under", "wide-pairs")

FUNCTIONS = ("truncate_large_groups", "drop_large_groups", "limit_keys_per_group")

#: The truncation threshold used with each distribution.
THRESHOLDS = {
    "worst-case": 3,
    "realistic": 5,
    "all-under": 100,
    "wide-pairs": 2,
}


def make_frame(distribution: str, n_rows: int) -> pd.DataFrame:
    """Returns the benchmark frame for one distribution and size.

    Every frame has the same three columns -- ``id`` int64 (the grouping
    column), ``key`` int64 with 50 distinct values, and ``s``, an object
    column of strings with 1,000 distinct values -- so that the only variable
    is the group shape.

    Args:
        distribution: One of :data:`DISTRIBUTIONS`.
        n_rows: The number of rows.

    Returns:
        The generated frame.
    """
    rng = np.random.default_rng(SEED)
    if distribution in ("worst-case", "all-under"):
        ids = rng.integers(0, max(n_rows // 10, 1), size=n_rows, dtype=np.int64)
        n_keys = 50
    elif distribution == "realistic":
        sizes = 1 + rng.geometric(p=0.4, size=n_rows)
        ids = np.repeat(np.arange(len(sizes), dtype=np.int64), sizes.astype(np.int64))[
            :n_rows
        ]
        n_keys = 50
    elif distribution == "wide-pairs":
        n_ids = max(n_rows // 20, 1)
        ids = rng.integers(0, n_ids, size=n_rows, dtype=np.int64)
        n_keys = 5
    else:
        raise ValueError(f"Unknown distribution {distribution}")
    keys = rng.integers(0, n_keys, size=n_rows, dtype=np.int64)
    strings = pd.Series(
        [f"v{i}" for i in rng.integers(0, 1000, size=n_rows)], dtype=object
    )
    return pd.DataFrame({"id": ids, "key": keys, "s": strings})


def oversized_fraction(df: pd.DataFrame, function: str, threshold: int) -> float:
    """Returns the fraction of rows belonging to an oversized group.

    For ``limit_keys_per_group`` a group is oversized when it has more than
    ``threshold`` distinct keys; for the other functions, when it has more
    than ``threshold`` rows.

    Args:
        df: The benchmark frame.
        function: The function being benchmarked.
        threshold: The truncation threshold.

    Returns:
        The oversized-row fraction, in [0, 1].
    """
    if len(df) == 0:
        return 0.0
    if function == "limit_keys_per_group":
        keys_per_group = df.groupby("id")["key"].nunique()
        oversized = df["id"].map(keys_per_group) > threshold
    else:
        oversized = df.groupby("id")["id"].transform("size") > threshold
    return float(oversized.mean())


def make_runner(implementation: Any, function: str, threshold: int) -> Callable:
    """Returns a callable running one truncation function of an implementation.

    This is the single dispatch point over the benchmarked function names;
    :func:`spark_runner` below only adds Spark's plumbing.

    Args:
        implementation: The module carrying the three truncation functions.
        function: The function to run.
        threshold: The truncation threshold.

    Returns:
        A callable taking the input frame and returning the truncated frame.
    """
    if function == "truncate_large_groups":
        return lambda df: implementation.truncate_large_groups(df, ["id"], threshold)
    if function == "drop_large_groups":
        return lambda df: implementation.drop_large_groups(df, ["id"], threshold)
    if function == "limit_keys_per_group":
        return lambda df: implementation.limit_keys_per_group(
            df, ["id"], ["key"], threshold
        )
    raise ValueError(f"Unknown truncation function {function}")


def spark_prepare(spark: Any, df: pd.DataFrame) -> Any:
    """Builds and materializes a cached Spark frame outside the timed region.

    The frame only depends on the input data, so one prepared frame is shared
    by every function benchmarked against it.

    Args:
        spark: The Spark session.
        df: The pandas frame to convert.

    Returns:
        The cached Spark frame, forced with ``count()``.
    """
    sdf = spark.createDataFrame(df).cache()
    sdf.count()
    return sdf


def spark_runner(function: str, threshold: int) -> Callable[[Any], Any]:
    """Returns a callable running one Spark truncation function.

    The runner performs the truncation and forces it with ``count()`` (not
    ``collect()``), so the measurement is compute rather than driver transfer.

    Args:
        function: The function to run.
        threshold: The truncation threshold.

    Returns:
        A callable taking the prepared Spark frame.
    """
    from tmlt.core.utils import truncation  # noqa: PLC0415

    run = make_runner(truncation, function, threshold)
    return lambda sdf: run(sdf).count()


def build_spark() -> Any:
    """Returns a local Spark session matching the motivating benchmark.

    Returns:
        A ``local[4]`` session with 4 shuffle partitions.
    """
    from pyspark.sql import SparkSession  # noqa: PLC0415

    return (
        SparkSession.builder.master("local[4]")
        .config("spark.sql.shuffle.partitions", "4")
        .appName("pandas_truncation_benchmark")
        .getOrCreate()
    )


def best_of(action: Callable[[], None], repeats: int) -> float:
    """Returns the fastest wall-clock time over ``repeats`` timed runs.

    One untimed warm-up run happens first.

    Args:
        action: The callable to time.
        repeats: The number of timed runs.

    Returns:
        The best time, in seconds.
    """
    action()  # warm-up
    times = []
    for _ in range(repeats):
        with Timer() as timer:
            action()
        times.append(timer.elapsed)
    return min(times)


def profile_call(action: Callable[[], None], label: str) -> None:
    """Profiles one call and prints the top functions by cumulative time.

    Args:
        action: The callable to profile.
        label: A label naming the profiled configuration.
    """
    print(f"\n--- cProfile: {label} ---")
    profiler = cProfile.Profile()
    profiler.enable()
    action()
    profiler.disable()
    pstats.Stats(profiler).sort_stats("cumulative").print_stats(30)


def environment_description() -> str:
    """Returns a one-line description of the measurement environment.

    Returns:
        Python, pandas, and numpy versions plus the platform.
    """
    parts = [
        f"python {sys.version.split()[0]}",
        f"pandas {pd.__version__}",
        f"numpy {np.__version__}",
        platform.platform(),
    ]
    try:
        import pyspark  # noqa: PLC0415

        parts.insert(3, f"pyspark {pyspark.__version__}")
    except ImportError:
        pass
    return ", ".join(parts)


def main() -> None:
    """Runs the benchmark and writes the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trial_name",
        default="",
        nargs="?",
        help="Optional label appended to output file names and columns.",
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=list(SIZES), help="Row counts to run."
    )
    parser.add_argument(
        "--distributions",
        nargs="+",
        choices=DISTRIBUTIONS,
        default=list(DISTRIBUTIONS),
        help="Group-size distributions to run.",
    )
    parser.add_argument(
        "--functions",
        nargs="+",
        choices=FUNCTIONS,
        default=list(FUNCTIONS),
        help="Truncation functions to run.",
    )
    parser.add_argument(
        "--backends",
        choices=("pandas", "spark", "both"),
        default="both",
        help="Which implementations to time.",
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="Timed runs per configuration."
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="cProfile one pandas call per configuration at the largest size.",
    )
    args = parser.parse_args()

    backends = ["pandas", "spark"] if args.backends == "both" else [args.backends]
    print(environment_description())

    spark = build_spark() if "spark" in backends else None
    suffix = f"_{args.trial_name}" if args.trial_name else ""
    rows: List[Dict[str, Any]] = []
    for distribution in args.distributions:
        threshold = THRESHOLDS[distribution]
        for size in args.sizes:
            df = make_frame(distribution, size)
            # The Spark frame is prepared once, on first use, and shared by
            # every function timed against it.
            sdf = None
            for function in args.functions:
                if distribution == "wide-pairs" and (
                    function != "limit_keys_per_group"
                ):
                    continue
                oversized = oversized_fraction(df, function, threshold)
                record: Dict[str, Any] = {
                    "function": function,
                    "distribution": distribution,
                    "rows": size,
                    "threshold": threshold,
                    "oversized": round(oversized, 4),
                }
                for backend in backends:
                    if backend == "spark":
                        if sdf is None:
                            sdf = spark_prepare(spark, df)
                        run = spark_runner(function, threshold)
                        data = sdf
                    else:
                        run = make_runner(pandas_truncation, function, threshold)
                        data = df
                    seconds = best_of(lambda: run(data), args.repeats)
                    record[f"{backend}{suffix} (s)"] = round(seconds, 4)
                    print(
                        f"{backend} {function} {distribution} rows={size} "
                        f"oversized={oversized:.1%} best={seconds:.4f}s"
                    )
                    if backend == "pandas" and args.profile and size == max(args.sizes):
                        profile_call(
                            lambda: run(data),
                            f"{function} {distribution} rows={size}",
                        )
                rows.append(record)
            if sdf is not None:
                sdf.unpersist()

    result = pd.DataFrame(rows)
    write_as_csv(result, f"pandas_truncation{suffix}.csv")
    write_as_html(result, f"pandas_truncation{suffix}.html")
    print()
    print(markdown_table(result))


def markdown_table(result: pd.DataFrame) -> str:
    """Returns the result frame as a markdown table for the PR description.

    Args:
        result: The benchmark results.

    Returns:
        The markdown rendering.
    """
    columns = list(result.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for _, row in result.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
