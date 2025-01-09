import pandas as pd

file_path = "hiroshima_challengedata.csv"
df = pd.read_csv(file_path)

df_filtered_days = df[(df['d'] >= 60) & (df['d'] <= 75)]

uids_to_exclude = df_filtered_days[
    (df_filtered_days['x'] > 200) | (df_filtered_days['y'] > 200)
]['uid'].unique()

df_valid = df[~df['uid'].isin(uids_to_exclude)]

output_file = "filtered_data.csv"
df_valid.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"篩選完成，結果保存到 {output_file}")
