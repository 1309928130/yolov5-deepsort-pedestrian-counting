import pandas as pd

def add_coordinates_to_csv(csv_file, values):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Add the specified values to the specified columns for each row
    for col in ['Cx1', 'Cy1', 'Cx2', 'Cy2']:
        df[col] = values[col]

    # Save the modified DataFrame back to the CSV file
    df.to_csv(csv_file, index=False)

# Specify the CSV file path and the values to add
csv_file_path = r'F:\minyu\brussels\videobrussels_enshan_ped\video_list.csv'
values_to_add = {'Cx1': 0.33, 'Cy1': 0, 'Cx2': 0.33, 'Cy2': 1}

# Add the values to the CSV file
add_coordinates_to_csv(csv_file_path, values_to_add)
