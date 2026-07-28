# Simulated Pendulum Dataset

Controlled toy physics videos generated from pendulum equations and rendered with OpenCV.

Scenes:

- `single_pendulum`: one bob attached to one rod.
- `double_pendulum`: two bobs attached by two rods.

Labels:

- `correct`: normal gravity and light damping.
- `wrong`: manually injected physical inconsistency such as energy gain, reversed gravity, or velocity kicks.

The `metadata.csv` file stores scene labels, wrong-physics type, seed, and video paths.
