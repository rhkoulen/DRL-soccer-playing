import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(input('Relative path to checkpoint progress.csv: '))

plt.figure(figsize=(10, 5))
plt.plot(df['timesteps_total'], df['episode_reward_mean'], label='Mean Reward', color='blue')
plt.fill_between(df['timesteps_total'], df['episode_reward_min'], df['episode_reward_max'], alpha=0.2, color='blue', label='Min/Max Range')
plt.xlabel('Timesteps (10M)')
plt.ylabel('Reward')
plt.title(input('Title: ') + ' — Training Curve')
plt.legend()
plt.tight_layout()
plt.savefig('training_curve.png', dpi=150)
print('Done')