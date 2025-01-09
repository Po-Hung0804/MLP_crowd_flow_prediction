import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


file_path = "filtered_data.csv"  
df = pd.read_csv(file_path)

print(df.head())

df_filtered = df[(df['x'] >= 0) & (df['x'] <= 200) & (df['y'] >= 0) & (df['y'] <= 200)]

df['t_hour'] = df['t'] // 2


grid_counts = (
    df.groupby(['d', 't_hour', 'x', 'y'])['uid']
    .nunique()
    .reset_index(name='user_count') 
)


output_file = "grid_total.csv"
grid_counts.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"已將每個網格的總人數存入 {output_file}")


plt.figure(figsize=(10, 10))
plt.scatter(df_filtered['x'], df_filtered['y'], s=1, alpha=0.5, c='blue')
plt.title("City Grid Visualization (0~200 Range)")
plt.xlabel("X Coordinate (500m per unit)")
plt.ylabel("Y Coordinate (500m per unit)")
plt.grid(True)
plt.show()
