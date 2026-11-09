#!/usr/bin/env python3
"""Compare two release binaries on the same input, including spectrum validation."""

import argparse
import json
import math
import statistics
import subprocess
import tempfile
import time
from pathlib import Path


def read_power(path):
    with path.open() as stream:
        return [
            tuple(map(float, line.split()))
            for line in stream
            if line.strip() and not line.startswith("#")
        ]


def compare_spectra(before, after):
    old = read_power(before)
    new = read_power(after)
    if len(old) != len(new) or not old:
        raise RuntimeError("Spectrum channel counts differ or output is empty")
    max_relative = 0.0
    for (old_freq, old_power), (new_freq, new_power) in zip(old, new):
        if not all(map(math.isfinite, (old_freq, old_power, new_freq, new_power))):
            raise RuntimeError("Non-finite spectrum value")
        if old_freq != new_freq:
            raise RuntimeError("Spectrum frequency grids differ")
        error = abs(old_power - new_power)
        scale = max(abs(old_power), abs(new_power))
        max_relative = max(max_relative, error / scale if scale else 0.0)
        if error > 1e-20 + scale * 1e-5:
            raise RuntimeError(f"Spectrum differs at {old_freq} MHz: {old_power} vs {new_power}")
    return max_relative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--vdif", type=Path, required=True)
    parser.add_argument("--fft", type=int, nargs="+", default=[4096, 16384])
    parser.add_argument("--cpu", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--length", type=float, default=0.524288)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--vsrec", action="store_true")
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.repeat < 1 or args.length <= 0:
        parser.error("--repeat and --length must be positive")
    directory = args.directory or Path(tempfile.mkdtemp(prefix="vdif2spec-bench-"))
    directory.mkdir(parents=True, exist_ok=True)
    binaries = {"before": args.before.resolve(), "after": args.after.resolve()}
    results = []
    report = {
        "input": str(args.vdif.resolve()),
        "length_seconds": args.length,
        "vsrec": args.vsrec,
        "binaries": {name: str(path) for name, path in binaries.items()},
        "results": results,
    }
    print(f"Results: {directory}", flush=True)
    for fft in args.fft:
        for cpu in args.cpu:
            times = {name: [] for name in binaries}
            outputs = {name: directory / f"{name}-{fft}-{cpu}.txt" for name in binaries}

            def run(name):
                command = [
                    str(binaries[name]), "--vdif", str(args.vdif.resolve()),
                    "--fft", str(fft), "--cpu", str(cpu), "--length", str(args.length),
                    "--no-plot", "--output", str(outputs[name]),
                ]
                if args.vsrec:
                    command.append("--vsrec")
                start = time.perf_counter()
                result = subprocess.run(command, capture_output=True, text=True)
                elapsed = time.perf_counter() - start
                if result.returncode:
                    raise RuntimeError(f"{' '.join(command)}\n{result.stdout}\n{result.stderr}")
                return elapsed

            # Warm the file cache and alternate order to reduce timing bias.
            for name in binaries:
                run(name)
            for repeat in range(args.repeat):
                order = ("before", "after") if repeat % 2 == 0 else ("after", "before")
                for name in order:
                    times[name].append(run(name))
            before = statistics.median(times["before"])
            after = statistics.median(times["after"])
            relative_error = compare_spectra(outputs["before"], outputs["after"])
            results.append({
                "fft": fft, "cpu": cpu, "times_seconds": times,
                "before_median_seconds": before, "after_median_seconds": after,
                "speedup": before / after, "max_relative_power_error": relative_error,
            })
            (directory / "results.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"FFT={fft:7d} CPU={cpu:2d}: {before:.4f}s -> {after:.4f}s "
                f"({before / after:.2f}x), max relative power error={relative_error:.3g}",
                flush=True,
            )


if __name__ == "__main__":
    main()
