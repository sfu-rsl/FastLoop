import matplotlib.pyplot as plt
import numpy as np

# داده‌ها
data = [
    [13.183, 15.268, 38.103],
    [14.957, 527.118, 570.002],
    [43.406, 271.606, 399.928],
    [56.4, 658.9, 948.2],
]
data = np.array(data)

loop_fusion = data[:, 0]
graph_opt = data[:, 1]
correct_loop = data[:, 2]
others = correct_loop - (loop_fusion + graph_opt)

datasets = ['Room3', 'Corridor1', 'Magistrale2', 'Outdoors5']
x = np.arange(len(datasets))

# اندازه شکل مناسب IEEE (عرض کمتر)
plt.figure(figsize=(4, 3))  # عرض کوچک‌تر و ارتفاع کم

# رنگ‌های استاندارد IEEE
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# رسم بارها
plt.bar(x, loop_fusion, width=0.4, color=colors[0], label='Loop Fusion')
plt.bar(x, graph_opt, bottom=loop_fusion, width=0.4, color=colors[1], label='Graph Optimization')
plt.bar(x, others, bottom=loop_fusion + graph_opt, width=0.4, color=colors[2], label='Others')

# برچسب‌ها و فونت‌ها
plt.xticks(x, datasets, fontsize=8)
plt.yticks(fontsize=8)
plt.ylabel('Time (ms)', fontsize=9)
plt.title('Loop Correction Time Breakdown', fontsize=10)
plt.legend(fontsize=8, frameon=False)  # بدون کادر مثل IEEE

plt.tight_layout()
plt.show()
