"""Plot within- and between-power distances saved by the ensemble comparison."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def run(directory, powers):
    within = read_rows(directory / 'within8_distances.csv')
    between = read_rows(directory / 'between8_distances.csv')
    config = json.loads((directory / 'configuration.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
    rows, ratios = [], []
    labels = [f'Within {p}' for p in powers] + [f'{powers[0]} vs {powers[1]}']
    for ax, metric in zip(axes, ('mean', 'covariance')):
        samples = [np.array([float(r['distance']) for r in within
                             if r['power'] == p and r['metric'] == metric])
                   for p in powers]
        samples.append(np.array([float(r['distance']) for r in between
                                 if {r['power_a'], r['power_b']} == set(powers)
                                 and r['metric'] == metric]))
        if any(len(s) == 0 or not np.isfinite(s).all() for s in samples):
            raise ValueError(f'Missing or nonfinite distances for {metric}.')
        baseline = (samples[0].mean() + samples[1].mean()) / 2
        if baseline <= 0:
            raise ValueError(f'Cannot normalize zero within-power distance: {metric}.')
        ratio = samples[2].mean() / baseline
        ratios.append(ratio)
        scaled = [s / baseline for s in samples]
        bins = np.linspace(min(s.min() for s in scaled),
                           max(s.max() for s in scaled), 19)
        for values, raw, label, color in zip(scaled, samples, labels,
                                             ('#0072B2', '#E69F00', '#CC79A7')):
            ax.hist(values, bins=bins, weights=np.full(len(values), 100 / len(values)),
                    histtype='step', linewidth=2, color=color, label=label)
            lo, med, hi = np.quantile(values, [.05, .5, .95])
            rows.append([metric, label, len(raw), float(raw.mean()),
                         float(values.mean()), float(lo), float(med), float(hi)])
        ax.set(title=f'Ensemble {metric}\nBetween / within = {ratio:.3f}',
               xlabel='Distance / average within-power distance',
               ylabel='Comparisons per bin [%]')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle(f'{powers[0]} vs {powers[1]}: eight-repetition ensembles\n'
                 f'Rank {config["rank"]}; {config["time_samples"]} matched times; '
                 f'{config["disjoint_partitions"]} splits (reusing repetitions)', fontsize=11)
    for extension in ('png', 'pdf'):
        fig.savefig(directory / f'distance_distributions.{extension}', dpi=180)
    plt.close(fig)
    with (directory / 'distance_distribution_summary.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['metric', 'comparison', 'count', 'mean_distance',
                         'normalized_mean', 'normalized_p05', 'normalized_median',
                         'normalized_p95'])
        writer.writerows(rows)
    lines = [f'# {powers[0]} versus {powers[1]}', '',
             'Eight-member ensembles; 100 distinct complementary splits by default. '
             'Splits reuse repetitions and are descriptive, not independent trials.', '',
             f'Rank {config["rank"]}; {config["time_samples"]} matched sampled times; '
             'physical-space mean and covariance distances with both mean baselines removed.', '',
             '| Metric | Comparison | Mean / within baseline | 5–95% range / baseline |',
             '|---|---|---:|---:|']
    lines += [f'| {r[0]} | {r[1]} | {r[4]:.3f} | {r[5]:.3f}–{r[7]:.3f} |' for r in rows]
    lines += ['', 'The baseline is the average of the two within-power mean distances. '
              'Distances compare full spatial statistics over time, including amplitude differences. '
              'A ratio near one indicates little extra average separation between powers. '
              'It does not establish equality of the underlying statistics.', '',
              'This run uses one POD rank; rank sensitivity and amplitude-normalized shape '
              'comparisons are not assessed.']
    (directory / 'PAIR_SUMMARY.md').write_text('\n'.join(lines) + '\n')
    print(f'Mean ratio: {ratios[0]:.6f}; covariance ratio: {ratios[1]:.6f}')
    print(f'Plot: {directory.resolve() / "distance_distributions.png"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='Existing ensemble comparison output directory.')
    parser.add_argument('--powers', nargs=2, default=['0p20', '0p35'])
    args = parser.parse_args()
    run(args.directory, args.powers)
