# Locked MLP/C&S versus PPO official-test results

Original C&S: 84.2069% ± 0.0720% across ten base seeds.
PPO: 81.7399% ± 0.1985% across ten base seeds (five actors averaged within each).
Validation-selected global C&S: 83.1221%. Supervised selector: 81.6205%.
Paired mean gain: -2.4669 percentage points.

The published reference is 84.18% ± 0.07%. Numerical differences in spectral solver, runtime and seed settings are disclosed in MLP_PROTOCOL.md. Positive point estimates alone do not establish generalization to other graphs.

Published Products references and earlier unrelated Products experiments were known; this does not erase that history.

Frozen record: /var/tmp/mshanthappa/rl-cs/mlp/final-test
