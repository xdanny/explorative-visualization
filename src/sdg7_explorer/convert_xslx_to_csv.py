import os
import pandas as pd
from pathlib import Path

def convert_excel_to_csv(input_folder, output_folder):
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.xlsx'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace('.xlsx', '.csv'))

            # Read the Excel file
            df = pd.read_excel(input_path)

            # Write to CSV
            df.to_csv(output_path, index=False)
            print(f"Converted {filename} to CSV")

if __name__ == "__main__":
    input_folder = "data"
    output_folder = "output"
    convert_excel_to_csv(input_folder, output_folder)