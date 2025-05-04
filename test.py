import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Mapping of original names to simplified ones
name_map = {
    'lines_15pm_lights_off_window_open': 'D1',
    'lines_15pm_lights_on_window_open': 'D2',
    'lines_15pm_lights_on_window_closed': 'D3',
}

# Updated raw data
raw_data = [
    ('lines_15pm_lights_off_window_open', 'lines_15pm_lights_off_window_open', 0.9863636363636363),
    ('lines_15pm_lights_off_window_open', 'lines_15pm_lights_on_window_open', 0.9941348973607038),
    ('lines_15pm_lights_off_window_open', 'lines_15pm_lights_on_window_closed', 0.9940652818991098),
    ('lines_15pm_lights_on_window_open', 'lines_15pm_lights_off_window_open', 0.959409594095941),
    ('lines_15pm_lights_on_window_open', 'lines_15pm_lights_on_window_open', 0.9852216748768473),
    ('lines_15pm_lights_on_window_open', 'lines_15pm_lights_on_window_closed', 0.9738903394255874),
    ('lines_15pm_lights_on_window_closed', 'lines_15pm_lights_off_window_open', 0.9923371647509579),
    ('lines_15pm_lights_on_window_closed', 'lines_15pm_lights_on_window_open', 0.9976019184652278),
    ('lines_15pm_lights_on_window_closed', 'lines_15pm_lights_on_window_closed', 0.987012987012987),
]

# Scenarios in order
scenarios = ['D1', 'D2', 'D3']

# Build 3x3 matrix
matrix = np.zeros((3, 3))
for r, c, val in raw_data:
    i = scenarios.index(name_map[r])
    j = scenarios.index(name_map[c])
    matrix[i, j] = val

font_size = 15

# Plotting heatmap
plt.figure(figsize=(5.5, 5))
ax = sns.heatmap(
    matrix,
    annot=True,
    fmt=".3f",
    annot_kws={"size": font_size},  # <- increase font size here
    xticklabels=scenarios,
    yticklabels=scenarios,
    cmap="coolwarm",
    cbar=True,
    square=True
)

# Increase axis tick label font sizes
ax.set_xticklabels(scenarios, fontsize=font_size)
ax.set_yticklabels(scenarios, fontsize=font_size)

# Increase colorbar font size
colorbar = ax.collections[0].colorbar
colorbar.ax.tick_params(labelsize=font_size)

# Add axis labels and title
plt.title("Precision in Different Illuminations", fontsize=font_size, pad=10)
plt.xlabel("Evaluation", fontsize=font_size)
plt.ylabel("Thresholds Trained On", fontsize=font_size)
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()
