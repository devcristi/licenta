import re
from pathlib import Path
p = Path(r"d:\study\licenta\creier\checkpoints\train_log.txt")
text = p.read_text(encoding='utf-8')
lines = text.splitlines()

best_mean_dice = (None, -1.0, '')
best_wt_tc_et = (None, -1.0, '')

for line in lines:
    m = re.search(r"Epoca\s+(\d+)\s+-\s+Mean Dice:\s*([0-9.]+)", line)
    if m:
        epoch = int(m.group(1))
        val = float(m.group(2))
        if val > best_mean_dice[1]:
            best_mean_dice = (epoch, val, line)
    m2 = re.search(r"Epoca\s+(\d+)\s+-\s+Mean WT/TC/ET:\s*([0-9.]+)", line)
    if m2:
        epoch = int(m2.group(1))
        val = float(m2.group(2))
        if val > best_wt_tc_et[1]:
            best_wt_tc_et = (epoch, val, line)

print('Best Mean Dice epoch:', best_mean_dice[0], 'value:', best_mean_dice[1])
print(best_mean_dice[2])
print('Best Mean WT/TC/ET epoch:', best_wt_tc_et[0], 'value:', best_wt_tc_et[1])
print(best_wt_tc_et[2])
