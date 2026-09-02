import pandas as pd

# Load both CSV files into pandas DataFrames
file1_df = pd.read_csv(r"D:\simulationMM\python Proximity\generatedFiles\2024PicColPro_rot_old_20240311173938_1.csv")
file2_df = pd.read_csv(r"D:\simulationMM\python Proximity\2024_output_rot_old\merged\pedestrian_counts_2024-03-11_17-30-26.csv")

# Remove the 'merged_' prefix from the 'File Name' column in file2_df
file2_df['File Name'] = file2_df['File Name'].str.replace('merged_', '')

# Merge the DataFrames based on the 'File Name' and 'Image' columns
merged_df = pd.merge(file1_df, file2_df, left_on='Image', right_on='File Name', how='left')

# Rename the 'Pedestrian Count' column from file2_df to 'pedestrian_count'
merged_df.rename(columns={'Pedestrian Count': 'pedestrian_count'}, inplace=True)

# Drop the duplicate 'File Name' column from file2_df
merged_df.drop(columns=['File Name'], inplace=True)


# Save the merged DataFrame to a new CSV file
merged_file_path = r"D:\simulationMM\python Proximity\generatedFiles\2024PicColPro_rot_old_20240311173938_1_merged.csv"
merged_df.to_csv(merged_file_path, index=False)

print("Merged data saved to:", merged_file_path)
