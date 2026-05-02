"""Gym envs shared between training and inference.

Importing this submodule pulls in gymnasium and gym-pybullet-drones, which
are heavy. The drone-agent's `noop` policy must NOT import this package —
only `ppo` / training does.
"""
