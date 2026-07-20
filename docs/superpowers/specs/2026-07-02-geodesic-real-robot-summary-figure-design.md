# Geodesic Real-Robot Summary Figure Design

## Core conclusion

Geodesic optimization preserves the scanning task while reducing force-controller correction motion and tangential interaction.

## Figure contract

- Archetype: quantitative grid.
- Backend: Python/matplotlib only.
- Output: PDF, SVG, and 600 dpi PNG.
- Match the simulation boxplot geometry exactly.
- Individual boxplot size: 3.25 x 2.55 in.
- Individual process-panel size: 6.50 x 2.55 in.
- Combined figure size: 13.00 x 5.10 in.

## Top row: best-performing process

Use Group 7. Its smoothed tangential-force curve stays below Original for about 71.5% of scan progress, while tangential-force P95 and tangential-torque P95 are reduced by about 16.3% and 22.5%, respectively. This avoids the long unfavorable intervals visible in Group 5.

Plot Original and Geodesic against normalized scan progress in two panels:

1. Tangential force.
2. Tangential torque.

Show raw signals faintly and fixed Savitzky-Golay smoothing results prominently. Compute all statistics from unsmoothed measurements.

## Bottom row: paired Group 2-8 statistics

Show all four methods with boxplots, all seven paired case points, and within-case connecting lines:

1. Offset total variation.
2. Cumulative outward correction motion.
3. Tangential-force P95.
4. Tangential-torque P95.

## Reviewer safeguards

- Call Group 3 a best-performing illustrative case, not a representative case.
- Display all seven cases in the bottom row.
- Do not smooth data used for statistics.
- Keep path-coverage quantification outside this figure.
- Export the combined figure and each of the six panels separately.
