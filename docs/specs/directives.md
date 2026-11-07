# Spec: knee point and directives

Synthetic data only. Not a medical device. A planner decides; the tool only suggests.

- Knee point: min-max scale each objective, pick the row with the smallest L2 norm. A constant objective gets range 1 (no divide by zero).
- Directives: MOVE (gain above `move_cmi_threshold` x CMI), HOLD (vacant anchor-ward bed), ALERT (patient cannot be clustered).
- `ActionPlan` counts equal the number of actions of each type.
- No feasible solution: `ValueError("Optimiser returned no feasible solutions.")`.
- Higher `move_cmi_threshold` gives equal or fewer MOVEs.
