from pathlib import Path
import re

import fabio
import matplotlib
matplotlib.use("Agg")  # Save plots without opening a GUI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter


# Configuration
DATA_DIR = Path("data")
OUTPUT_DIR = Path("results/intensity_analysis")

SERIES = "130cm_detPos1"
STRIDE = 1  # Analyze every 10th image initially
SPOT_SIZE = 16


def load_positions(log_path):
    """Read measurement numbers and hexapod positions."""

    lines = log_path.read_text().splitlines()

    header = next(
        line for line in lines
        if line.startswith("Date")
    )

    columns = re.split(r"\s{2,}", header.strip())[4:]

    rows = []

    for line in lines:
        if not re.match(r"^\d{4}/\d{2}/\d{2}\s", line):
            continue

        date, time, measurement, sample_time, values = (
            line.split(maxsplit=4)
        )

        numbers = re.findall(r"-?\d+\.\d{6}", values)

        if len(numbers) != len(columns):
            raise ValueError(
                f"Cannot parse measurement {measurement}"
            )

        record = dict(zip(columns, map(float, numbers)))

        rows.append({
            "measurement": int(measurement),
            "x": record["hexapod x"],
            "z": record["hexapod z"],
        })

    return pd.DataFrame(rows)


def calculate_intensities(image_path):
    """Calculate three intensity metrics for one CBF image."""

    data = fabio.open(str(image_path)).data

    # Ignore masked pixels (-1, -2, etc.)
    valid = np.maximum(data, 0).astype(np.float32)

    maximum = float(valid.max())
    total = float(valid.sum(dtype=np.float64))

    # Find the 16x16 region with the highest summed intensity
    local_average = uniform_filter(
        valid,
        size=SPOT_SIZE,
        mode="constant",
    )

    local_maximum = float(
        local_average.max() * SPOT_SIZE**2
    )

    return maximum, total, local_maximum


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    matches = list(DATA_DIR.rglob(f"*{SERIES}/log_all.txt"))

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one log_all.txt for {SERIES}, "
            f"found {len(matches)}"
        )

    log_path = matches[0]
    series_dir = log_path.parent

    df = load_positions(log_path)

    # Start with a subset to speed up exploration
    df = df.iloc[::STRIDE].copy()

    print(f"Series: {series_dir.name}")
    print(f"Images to analyze: {len(df)}", flush=True)

    results = []

    for i, row in enumerate(df.itertuples(index=False), start=1):
        image_path = series_dir / f"image_{row.measurement:07d}.cbf"

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        maximum, total, local = calculate_intensities(image_path)

        results.append({
            "measurement": row.measurement,
            "x": row.x,
            "z": row.z,
            "max_intensity": maximum,
            "total_intensity": total,
            "local_intensity": local,
        })

        if i % 100 == 0:
            print(f"Processed {i}/{len(df)}", flush=True)

    results = pd.DataFrame(results)

    csv_path = OUTPUT_DIR / f"{SERIES}_stride{STRIDE}.csv"
    results.to_csv(csv_path, index=False)

    metrics = [
        ("max_intensity", "Brightest pixel"),
        ("total_intensity", "Total intensity"),
        ("local_intensity", "Strongest 16x16 region"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (column, title) in zip(axes, metrics):

        scatter = ax.scatter(
            results["x"],
            results["z"],
            c=results[column],
            cmap="viridis",
            s=12,
            marker="s",
            linewidths=0,
        )

        best = results.loc[results[column].idxmax()]

        ax.scatter(
            best["x"],
            best["z"],
            color="red",
            marker="x",
            s=100,
            linewidths=2,
        )

        ax.set_title(title)
        ax.set_xlabel("Hexapod X")
        ax.set_ylabel("Hexapod Z")
        ax.set_aspect("equal")

        fig.colorbar(scatter, ax=ax)

        print(f"\n{title}")
        print(f"  Image: {int(best['measurement'])}")
        print(f"  X: {best['x']:.6f}")
        print(f"  Z: {best['z']:.6f}")
        print(f"  Intensity: {best[column]:.2f}")

    plt.tight_layout()

    plot_path = OUTPUT_DIR / f"{SERIES}_stride{STRIDE}.png"
    plt.savefig(plot_path, dpi=180)
    plt.close()

    print(f"\nSaved plot: {plot_path}")
    print(f"Saved data: {csv_path}")


if __name__ == "__main__":
    main()