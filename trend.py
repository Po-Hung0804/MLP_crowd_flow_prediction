import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "city_grid_total_user_counts.csv"  
grid_totals = pd.read_csv(file_path)


plt.figure(figsize=(10, 8))


heatmap_data = grid_totals.pivot(index='y', columns='x', values='total_user_count')


sns.heatmap(heatmap_data, cmap='viridis', cbar_kws={'label': 'Total User Count'})
plt.title('Total User Count per Grid (75 Days)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.gca().invert_yaxis() 
plt.show()


top_grids = grid_totals.nlargest(5, 'total_user_count')

print("Top 5 grids with highest total user count:")
print(top_grids)

plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, cmap='viridis', cbar_kws={'label': 'Total User Count'})
for _, row in top_grids.iterrows():
    plt.scatter(row['x'] + 0.5, row['y'] + 0.5, color='red', s=100, label=f"Grid ({row['x']}, {row['y']})")
plt.title('Total User Count per Grid with Top Grids Highlighted')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.gca().invert_yaxis()
plt.legend()
plt.show()
