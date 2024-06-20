import scipy.io
import pandas as pd
import os
import time

start_time = time.time()

# Specify the main directory
main_directory = '05_ErgebnisseSimulationen'

# List of specific file names
specific_files = ['SpannungLeiterLeiter', 'Strangstrom', 'Drehmoment', 'i_abc']

column_list = ['VM9.V [V]', 'VM7.V [V]', 'VM8.V [V]', 'i_a', 'i_b', 'i_c', 'Current(PhaseU) [A]', 'Current(PhaseV) [A]',
               'Current(PhaseW) [A]', 'Moving1.Torque [NewtonMeter]']

dataset = pd.DataFrame()

# Iterate through each subdirectory in the main directory
for root, dirs, files in os.walk(main_directory):
    for subdir in dirs:
        subdir_path = os.path.join(root, subdir)
        #print(f'Accessing subdirectory: {subdir_path}')

        datafile = pd.DataFrame()

        # Iterate through each file in the subdirectory
        for file_name in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, file_name)

            # Get the file
            file_base_name = os.path.splitext(file_name)[0]

            # Check if it's a file and process based on file type
            if os.path.isfile(file_path) and file_base_name in specific_files:
                if file_name.endswith('.csv'):
                    #print(f'Reading CSV file: {file_path}')
                    df_csv = pd.read_csv(file_path)
                    #print(df_csv.head())
                    datafile = pd.concat([datafile, df_csv], axis=1)

                elif file_name.endswith('.mat'):
                    #print(f'Reading MAT file: {file_path}')
                    mat_data = scipy.io.loadmat(file_path)
                    #print(mat_data.keys())
                    data_key = 'i_abc'
                    if data_key in mat_data:
                        data = mat_data[data_key]
                    else:
                        raise KeyError(f"Key '{data_key}' not found in the .mat file.")

                    # Convert the data to a pandas DataFrame
                    df_mat = pd.DataFrame(data)
                    df_mat.columns = ['i_a', 'i_b', 'i_c']
                    datafile = pd.concat([datafile, df_mat], axis=1)
                    #print(df_mat.head())

        dataset = pd.concat([dataset, datafile], axis=0, ignore_index=True)

dataset = dataset[column_list]
#print(dataset)
#print(dataset.keys())
#dataset.to_csv("05_ErgebnisseSimulationen/dataset.csv")
dataset.to_hdf("05_ErgebnisseSimulationen/dataset.h5", key='table')

end_time = time.time()
print(f"Time taken: {end_time - start_time:.6f} seconds")

# import matplotlib.pyplot as plt
# dataset[['Moving1.Torque [NewtonMeter]']].plot()
# plt.show()
# dataset[['VM9.V [V]', 'VM7.V [V]', 'VM8.V [V]']].plot()
# plt.show()
