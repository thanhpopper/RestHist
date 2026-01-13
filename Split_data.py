import os
import shutil
import numpy as np
import pandas as pd
import numpy as np

## Count the number of images for each patient and magnitude and make a dataframe
count_data = pd.read_csv("C:\\Users\\nguyendaithanh\\Desktop\\K_breast\\archive\\BreaKHis_v1\\BreaKHis_v1\\histology_slides\\breast\\count_data.csv")
count_data["label"] = count_data["Patient"].str[4]

## Split data by ratio 

import pandas as pd
import numpy as np

count_data.set_index("Patient", inplace=True)

### Divide into train, test and valid set
def split_column_to_ratio(df, ratio=(0.7, 0.3), random_state=None):
    
    # Initialize empty dataframes for part1 and part2
    part1 = pd.DataFrame()
    part2 = pd.DataFrame()

    for col in ["40X", "100X", "200X", "400X"]:
        # Shuffle the DataFrame
        shuffled_df = df[[col]].sample(frac=1, random_state=random_state)
        
        total_sum = shuffled_df[col].sum()
        target_sum1 = total_sum * ratio[0]
        target_sum2 = total_sum * ratio[1]
        
        current_sum1 = 0
        current_sum2 = 0

        part1_indices = []
        part2_indices = []

        for index, value in shuffled_df[col].items():
            if current_sum1 + value <= target_sum1 and index not in part2_indices:
                part1_indices.append(index)
                current_sum1 += value
            elif index not in part1_indices:
                part2_indices.append(index)
                current_sum2 += value

        # Create Series for part1 and part2
        part1[col] = pd.Series(shuffled_df.loc[part1_indices, col])
        part2[col] = pd.Series(shuffled_df.loc[part2_indices, col])

    # Drop rows with all NaN values (if any)
    part1.dropna(how='all', inplace=True)
    part2.dropna(how='all', inplace=True)

    # Reset index to include the original index column
    part1.reset_index(inplace=True)
    part2.reset_index(inplace=True)

    return (list(part1["Patient"]), list(part2["Patient"]))

# Divide data into train and test set with ratio 7:3
train_list, test_list = split_column_to_ratio(count_data, random_state = 42)

train_df = count_data.loc[train_list]
test_df = count_data.loc[test_list]


# Save to csv file
train_df.to_csv("C:\\Users\\nguyendaithanh\\Desktop\\K_breast\\Output_split\\train.csv")
test_df.to_csv("C:\\Users\\nguyendaithanh\\Desktop\\K_breast\\Output_split\\test.csv")

## Make directory for train and test for each magnitude
base_dir = "C:\\Users\\nguyendaithanh\\Desktop\\K_breast\\Data"
raw_directory = "C:\\Users\\nguyendaithanh\\Desktop\\K_breast\\archive\\BreaKHis_v1\\BreaKHis_v1\\histology_slides\\breast"
categories = ['benign', 'malignant']
#magnitudes = ['40X', '100X', '200X', '400X']

train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

for category in categories:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

# Function to split data to train and test
sub =[x[0] for x in os.walk(raw_directory)] # all subfolder
train_sources = [path for path in sub for patient in train_df.index if os.path.split(path)[-1] == patient] # all subfolder are patients name
test_sources = [path for path in sub for patient in test_df.index if os.path.split(path)[-1] == patient]

def move_folder(SOURCES, dir):
    for source in SOURCES:
        if "malignant" in source:
            shutil.copytree(source, os.path.join(dir, "malignant"), dirs_exist_ok= True)
        else:
            shutil.copytree(source, os.path.join(dir, "benign"), dirs_exist_ok= True)


# Move all folders
move_folder(train_sources, train_dir)
move_folder(test_sources, test_dir)

i = []
for path in sub:
    for patient in test_df.index:
        if os.path.split(path)[-1] == patient:
            i.append(patient)
        