import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap



# Data Entry
GAMES = 2500
AGENTS = ['NATE_QMIX', 'RANDOM', 'NATE_STACKED_PPO', 'RICHARD_LSTM-10M', 'CEIA', 'NATE_PPO', 'RICHARD_LSTM-15M', 'SHOURIK_SHAPED']
N = len(AGENTS)
______ = 0.0
WINS = np.array([
    [______, 1275.5, 2270.0, 2431.0, 2466.5, 2475.0, 2469.0, 2487.0],
    [1224.5, ______, 2219.5, 2447.5, 2440.5, 2456.0, 2460.0, 2480.0],
    [0230.0, 0280.5, ______, 2164.0, 2207.0, 2328.0, 2394.0, 2420.0],
    [0069.0, 0052.5, 0336.0, ______, 1279.5, 1917.5, 2139.0, 2284.0],
    [0033.5, 0059.5, 0293.0, 1220.5, ______, 1684.0, 1999.0, 2128.0],
    [0025.0, 0044.0, 0172.0, 0582.5, 0816.0, ______, 1643.0, 1838.0],
    [0031.0, 0040.0, 0106.0, 0361.0, 0501.0, 0857.0, ______, 1527.0],
    [0013.0, 0020.0, 0080.0, 0216.0, 0372.0, 0662.0, 0973.0, ______]
]) # collected these values from `python -m soccer_twos.evaluate -m1 AGENT -m2 AGENT -e 2500`
MARKED_CELLS = {(1, 6), (6, 1), (4, 6), (6, 4), (5, 6), (6, 5)}



# Colors
cmap = LinearSegmentedColormap.from_list(
    'winrate',
    [
        (0.00, "#7a0000"),
        (0.50, "#d0d0d0"),
        (1.00, "#004d00"),
    ],
)
def text_color(val):
    """White text on strongly-colored cells, dark on pale cells near 50%."""
    return 'white' if abs(val - 0.5) > 0.22 else "#1a1a1a"



# Figure
fig, ax = plt.subplots(figsize=(11, 10))
ax.set_aspect('equal')
for i in range(N): # row = "this agent"
    for j in range(N): # col = "against"
        y = N - 1 - i # flip so row 0 is at the top

        # Diagonal cells
        if i == j:
            ax.add_patch(mpatches.Rectangle((j, y), 1, 1, color='black', zorder=1))
            continue

        # Normal cells
        wins = WINS[i][j]
        winrate = wins / GAMES
        ax.add_patch(mpatches.Rectangle((j, y), 1, 1, color=cmap(winrate), zorder=1))
        tc = text_color(winrate)
        ax.text(
            j + 0.5, y + 0.58,
            f'{round(winrate * 100):d}%',
            ha='center', va='center',
            fontsize=13, fontweight='bold',
            color=tc, zorder=3,
        )
        if wins % 1 != 0:
            wintext = f'{wins:.1f}/{GAMES}'
        else:
            wintext = f'{int(wins):d}/{GAMES}'
        ax.text(
            j + 0.5, y + 0.3,
            wintext,
            ha='center', va='center',
            fontsize=7.5,
            color=tc, zorder=3,
        )

        # Cheating matchup cells
        if (i, j) in MARKED_CELLS:
            ax.add_patch(
                mpatches.Rectangle(
                    (j, y), 1, 1,
                    fill=False,
                    hatch='////',
                    edgecolor="#888888",
                    linewidth=1,
                    zorder=2,
                )
            )



# Labeling
ax.set_xlim(0, N)
ax.set_ylim(0, N)

ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
ax.set_xticks(np.arange(N) + 0.5)
ax.set_xticklabels(AGENTS, rotation=-45, ha='right', fontsize=10)
ax.set_xlabel('Scoring Rate of These Agents ↓', labelpad=12, fontsize=11)

ax.set_yticks(np.arange(N) + 0.5)
ax.set_yticklabels(reversed(AGENTS), rotation=-45, ha='right', fontsize=10, rotation_mode='anchor')
ax.set_ylabel('Against These Agents ↓', labelpad=12, fontsize=11)

hatch_patch = mpatches.Patch(
    facecolor="#d0d0d0", edgecolor="#888888", hatch='////',
    label="Unfair matchup (trained on opponent)"
)
ax.legend(
    handles=[hatch_patch],
    loc='lower right',
    bbox_to_anchor=(1.0, -0.08),
    frameon=False,
    fontsize=9,
)



# Save
plt.tight_layout()
plt.savefig('winrate_heatmap.png', dpi=150, bbox_inches='tight')
print('Done')