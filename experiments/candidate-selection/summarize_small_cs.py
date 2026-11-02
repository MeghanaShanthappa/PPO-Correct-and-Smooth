"""Report experiment 1 using validation results only."""
import json
from pathlib import Path
import numpy as np

root = Path('/var/tmp/mshanthappa/rl-cs/small-cs-v1')
records = [json.loads(p.read_text()) for p in sorted(root.glob('policy-*/summary.json'))]
report = {'experiment': 'small_cs_v1', 'planned_base_seeds': 3,
          'completed_base_seeds': len(records), 'official_test_scored': False,
          'scope': 'exploratory full official validation; used for checkpoint selection',
          'per_seed': records}
if records:
    report['fixed_cs_mean'] = float(np.mean([r['global_actions']['fixed_cs'] for r in records]))
    report['methods'] = {method: {
        'mean_validation_accuracy': float(np.mean([r['methods'][method]['mean_accuracy'] for r in records])),
        'mean_gain_pp': float(np.mean([r['methods'][method]['gain_over_fixed_cs_pp'] for r in records]))}
        for method in ['warm_start', 'supervised', 'expected_reward', 'ppo']}
(root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'per_seed'}, indent=2))
