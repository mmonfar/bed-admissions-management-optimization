# Spec: objectives and constraint

Synthetic data only. Not a medical device.

Encoding: integer vector x, x[i] = index of the bed for patient i. No bed twice.

| Output | Rule | Hand case (tests/test_engine.py::test_objectives_by_hand) |
|---|---|---|
| f1 clustering | per patient: different floor from anchor = 10, same floor other wing = 5, else 1; summed | 6.0 and 11.0 |
| f2 vacancy | non-overflow beds not used | 0.0 and 1.0 |
| f3 stability | sum of CMI x 2.5 for patients whose bed differs from a known current bed | 0.0 and 5.0 |
| g0 | moves minus transfer cap; feasible when <= 0 | -1.0 and 0.0 |

Penalty values (1/5/10) and the CMI cost (2.5) are demo weights, not from the literature.
