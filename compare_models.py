import os
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

import config

res_dir_lstm = os.path.join(config.RESULTS_DIR, 'lstm')
res_dir_baseline = os.path.join(config.RESULTS_DIR, 'baseline')
labels = np.load(os.path.join(res_dir_lstm, "labels.npy"))
preds_lstm = np.load(os.path.join(res_dir_lstm, "preds.npy"))
preds_baseline = np.load(os.path.join(res_dir_baseline, "labels.npy"))

correct_lstm = (preds_lstm == labels)
correct_baseline = (preds_baseline == labels)

a = np.sum(correct_lstm & correct_baseline)
b = np.sum(correct_lstm & ~correct_baseline)
c = np.sum(~correct_lstm & correct_baseline)
d = np.sum(~correct_lstm & ~correct_baseline)

table = [[a, b],
         [c, d]]

result = mcnemar(table, exact=True)

print("McNemar Test")
print("============")
print(f"Contingency table: {table}")
print(f"Statistic: {result.statistic}")
print(f"p-value: {result.pvalue}")
