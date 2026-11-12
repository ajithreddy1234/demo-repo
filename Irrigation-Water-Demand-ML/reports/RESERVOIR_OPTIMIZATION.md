# Reservoir-Constrained Allocation Upgrade

## Why this was added
The ML model estimates crop irrigation demand. This optimization layer converts that demand into an operational release schedule when the seasonal irrigation water budget is limited.

## Optimization objective
Minimize the maximum relative daily irrigation deficit across the season, subject to:
- release on a day cannot exceed that day's gross irrigation demand,
- release cannot exceed the configured daily canal/release capacity,
- cumulative release cannot exceed the available seasonal irrigation-water budget.

A second optimization stage maximizes total delivered water while preserving the best achievable fairness level.

## Demonstration scenario
- Irrigated maize area: 1000 ha
- Overall delivery efficiency: 0.70
- Available seasonal water: 85.0% of full gross requirement
- Daily release-cap factor: 1.40 x mean gross daily demand

These values are an explicit stress-test scenario, not claimed field measurements.

## Results
- Full gross seasonal requirement: 2258198.07 m3
- Available seasonal water: 1919468.36 m3
- Optimized total release: 1469151.68 m3
- Total gross deficit: 789046.39 m3
- Minimum daily demand met: 34.89%
- Mean daily demand met: 88.72%
- Maximum relative deficit: 65.11%
- Final unused allocation budget: 450316.67 m3

## Interpretation
This layer changes the project from demand prediction alone into reservoir decision support. When water is sufficient, the optimizer can meet predicted demand. Under shortage, it distributes the constrained water budget while minimizing the worst relative daily deficit rather than applying an arbitrary release rule.

## Operational use
For a real reservoir deployment, the same optimizer accepts actual irrigated area, delivery efficiency, available irrigation allocation and daily release/canal capacity. Those inputs should come from the operating authority rather than being invented by the model.