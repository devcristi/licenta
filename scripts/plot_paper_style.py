import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import savgol_filter

def get_data():
    # Datele extrase din log-ul tau (Epocile 21-71)
    data_str = """
    Epoca 21 - WT: 0.7736, TC: 0.6173, ET: 0.5960
    Epoca 22 - WT: 0.7879, TC: 0.6456, ET: 0.6271
    Epoca 23 - WT: 0.7926, TC: 0.6550, ET: 0.6395
    Epoca 24 - WT: 0.7792, TC: 0.6404, ET: 0.6177
    Epoca 25 - WT: 0.7650, TC: 0.6142, ET: 0.5887
    Epoca 26 - WT: 0.7809, TC: 0.6443, ET: 0.6256
    Epoca 27 - WT: 0.7699, TC: 0.6190, ET: 0.6062
    Epoca 28 - WT: 0.7720, TC: 0.6271, ET: 0.6016
    Epoca 29 - WT: 0.7918, TC: 0.6559, ET: 0.6441
    Epoca 30 - WT: 0.7889, TC: 0.6668, ET: 0.6389
    Epoca 31 - WT: 0.7840, TC: 0.6568, ET: 0.6253
    Epoca 32 - WT: 0.7875, TC: 0.6502, ET: 0.6368
    Epoca 33 - WT: 0.7925, TC: 0.6454, ET: 0.6323
    Epoca 34 - WT: 0.7870, TC: 0.6650, ET: 0.6457
    Epoca 35 - WT: 0.7889, TC: 0.6501, ET: 0.6167
    Epoca 36 - WT: 0.7865, TC: 0.6597, ET: 0.6387
    Epoca 37 - WT: 0.7762, TC: 0.6484, ET: 0.6300
    Epoca 38 - WT: 0.7892, TC: 0.6727, ET: 0.6590
    Epoca 39 - WT: 0.7911, TC: 0.6684, ET: 0.6354
    Epoca 40 - WT: 0.7741, TC: 0.6347, ET: 0.6207
    Epoca 41 - WT: 0.7844, TC: 0.6717, ET: 0.6514
    Epoca 42 - WT: 0.7834, TC: 0.6573, ET: 0.6433
    Epoca 43 - WT: 0.7869, TC: 0.6665, ET: 0.6494
    Epoca 44 - WT: 0.7946, TC: 0.6821, ET: 0.6622
    Epoca 45 - WT: 0.7755, TC: 0.6166, ET: 0.5989
    Epoca 46 - WT: 0.7884, TC: 0.6789, ET: 0.6539
    Epoca 47 - WT: 0.7997, TC: 0.6834, ET: 0.6640
    Epoca 48 - WT: 0.7857, TC: 0.6477, ET: 0.6268
    Epoca 49 - WT: 0.7959, TC: 0.6711, ET: 0.6442
    Epoca 50 - WT: 0.7967, TC: 0.6801, ET: 0.6677
    Epoca 51 - WT: 0.7878, TC: 0.6576, ET: 0.6344
    Epoca 52 - WT: 0.7980, TC: 0.6920, ET: 0.6737
    Epoca 53 - WT: 0.8035, TC: 0.6890, ET: 0.6611
    Epoca 54 - WT: 0.8205, TC: 0.7280, ET: 0.6942
    Epoca 55 - WT: 0.8025, TC: 0.6756, ET: 0.6577
    Epoca 56 - WT: 0.8023, TC: 0.6432, ET: 0.6290
    Epoca 57 - WT: 0.8058, TC: 0.6901, ET: 0.6705
    Epoca 58 - WT: 0.8034, TC: 0.6825, ET: 0.6556
    Epoca 59 - WT: 0.8034, TC: 0.6930, ET: 0.6703
    Epoca 60 - WT: 0.7996, TC: 0.6912, ET: 0.6664
    Epoca 61 - WT: 0.8091, TC: 0.6973, ET: 0.6737
    Epoca 62 - WT: 0.8123, TC: 0.7086, ET: 0.6651
    Epoca 63 - WT: 0.7987, TC: 0.6862, ET: 0.6643
    Epoca 64 - WT: 0.8072, TC: 0.6935, ET: 0.6730
    Epoca 65 - WT: 0.8102, TC: 0.7198, ET: 0.7001
    Epoca 66 - WT: 0.8081, TC: 0.6972, ET: 0.6803
    Epoca 67 - WT: 0.8118, TC: 0.7092, ET: 0.6867
    Epoca 68 - WT: 0.8051, TC: 0.6873, ET: 0.6670
    Epoca 69 - WT: 0.8070, TC: 0.6952, ET: 0.6719
    Epoca 70 - WT: 0.8147, TC: 0.7045, ET: 0.6832
    Epoca 71 - WT: 0.8105, TC: 0.7018, ET: 0.6783
    """
    epochs, wt, tc, et = [], [], [], []
    for line in data_str.strip().split('\n'):
        parts = line.split(':')
        epochs.append(int(parts[0].split()[1]))
        wt.append(float(parts[1].split(',')[0]))
        tc.append(float(parts[2].split(',')[0]))
        et.append(float(parts[3]))
    return epochs, wt, tc, et

def set_paper_style():
    sns.set_context("paper", font_scale=1.4)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.3,
        "figure.dpi": 300
    })

def plot_class_dice(epochs, vals, label, color, filename):
    set_paper_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    # Raw data (faint line)
    ax.plot(epochs, vals, alpha=0.3, color=color, linewidth=1.5, label=f'Raw {label}')
    
    # Smoothed trend (Savitzky-Golay)
    # window_length must be odd, polyorder must be less than window_length
    smoothed = savgol_filter(vals, window_length=11, polyorder=2)
    ax.plot(epochs, smoothed, color=color, linewidth=3, label=f'Trend {label}')
    
    ax.set_title(f'Segmentation Performance: {label} (Dice Coefficient)', pad=20)
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Dice Score')
    ax.set_ylim(min(vals)-0.05, max(vals)+0.05)
    ax.legend(frameon=True, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

# Execution
epochs, wt, tc, et = get_data()
classes = [ (wt, "Whole Tumor (WT)", "#e41a1c"), 
            (tc, "Tumor Core (TC)", "#377eb8"), 
            (et, "Enhancing Tumor (ET)", "#4daf4a") ]

for vals, label, color in classes:
    fname = f"dice_plot_{label.split('(')[1][:2]}.png"
    plot_class_dice(epochs, vals, label, color, fname)