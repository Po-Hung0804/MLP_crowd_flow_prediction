import pandas as pd
import matplotlib.pyplot as plt

file_path = 'grid_total.csv'  
grid_totals = pd.read_csv(file_path)

filtered_data = grid_totals[(grid_totals['x'] == 79) & (grid_totals['y'] == 92)]

filtered_data['time_slot'] = filtered_data['d'] * 24 + filtered_data['t_hour']
plt.figure(figsize=(10, 6))
plt.plot(filtered_data['time_slot'], filtered_data['user_count'], marker='o', linestyle='-', color='b')
plt.xlim(min(filtered_data['time_slot']) - 1, max(filtered_data['time_slot']) + 1)
plt.ylim(min(filtered_data['user_count']) - 1, max(filtered_data['user_count']) + 1)
plt.title('Total User Count for Grid (79, 92) over Time')
plt.xlabel('Time Slot (d*24 + t_hour)')
plt.ylabel('Total User Count')
plt.tight_layout()
plt.show()


average_users = filtered_data.groupby('d')['user_count'].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.plot(average_users['d'], average_users['user_count'], marker='o', label='Average User Count', color='g')

plt.title('Daily Average User Count for Grid (79, 92)')
plt.xlabel('Day (d)')
plt.ylabel('Average User Count')
plt.legend()

plt.tight_layout()
plt.show()
time_user_data = filtered_data[['time_slot', 'user_count']]

output_path = 'grid_79_92.csv'  
time_user_data.to_csv(output_path, index=False)


