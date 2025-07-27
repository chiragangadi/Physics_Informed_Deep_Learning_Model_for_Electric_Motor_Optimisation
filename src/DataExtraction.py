import scipy.io
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def averaging(df, averageing_column, idx_start, idx_end, averaged_column):
    average = df[averageing_column][idx_start:idx_end].mean()
    df[averaged_column] = average
    return df, average

def addspeed(df):
    speed = (df['Moving1.Position [deg]'][10] - df['Moving1.Position [deg]'][9]) / (
                df['Time [ms]'][10] - df['Time [ms]'][9]) * (1000 / 6)
    df['Speed'] = speed
    return df, speed

class DataExtract:
    def __init__(self, main_directory, specific_files, column_list):
        self.main_directory = main_directory
        self.column_list = column_list
        self.specific_files = specific_files
        self.dataset = pd.DataFrame()
        self.datafile = pd.DataFrame()
        self.average_speed = []
        self.average_torque = []
        self.average_id = []
        self.average_iq = []
        self.hysteresisloss_rotor = []
        self.hysteresisloss_stator = []
        self.strandedloss = []
        self.eddycurrentloss_rotor = []
        self.eddycurrentloss_stator = []
        self.p = 6


    def generate(self):
        # Iterate through each subdirectory in the main directory
        global speed
        self.main_directory.sort()
        for root, dirs, file in os.walk(self.main_directory):
            dirs.sort()
            for subdir in dirs:
                subdir_path = os.path.join(root, subdir)

                self.datafile = pd.DataFrame()

                # Iterate through each file in the subdirectory
                for file_name in self.specific_files:
                    file_path = os.path.join(subdir_path, file_name)

                    # Check if it's a file and process based on file type
                    if os.path.isfile(file_path):
                        if file_name.endswith('.csv'):
                            #print(f'Reading CSV file: {file_path}')
                            if file_name == 'Winkel_el_mech.csv':
                                df_csv = pd.read_csv(file_path)
                                df_csv = df_csv[df_csv.columns][1:]
                                df_csv, speed = addspeed(df_csv)
                                self.average_speed.append(speed)
                            # elif file_name == 'Hystereseverluste.csv':
                            #     df_csv = pd.read_csv(file_path)
                            #     df_csv = df_csv[df_csv.columns][1:]
                            #     df_csv, hysloss_rotor, hysloss_stator = losses(df_csv, speed, 'Hystersis')
                            #     self.hysteresisloss_rotor.append(hysloss_rotor)
                            #     self.hysteresisloss_stator.append(hysloss_stator)
                            # elif file_name == 'Wicklungsverluste.csv':
                            #     df_csv = pd.read_csv(file_path)
                            #     df_csv = df_csv[df_csv.columns][1:]
                            #     df_csv, stranded_loss = losses(df_csv, speed, 'Stranded')
                            #     self.strandedloss.append(stranded_loss)
                            # elif file_name == 'Wirbelstromverluste.csv':
                            #     df_csv = pd.read_csv(file_path)
                            #     df_csv = df_csv[df_csv.columns][1:]
                            #     df_csv, eddyloss_rotor, eddyloss_stator = losses(df_csv, speed, 'Eddy')
                            #     self.eddycurrentloss_rotor.append(eddyloss_rotor)
                            #     self.eddycurrentloss_stator.append(eddyloss_stator)
                            elif file_name == 'Drehmoment.csv':
                                df_csv = pd.read_csv(file_path)
                                df_csv = df_csv[df_csv.columns][1:]
                                df_csv, torque = averaging(df_csv, 'Moving1.Torque [NewtonMeter]', 0, 40000,'Average Torque')
                                self.average_torque.append(torque)
                            else:
                                df_csv = pd.read_csv(file_path)
                                df_csv = df_csv[df_csv.columns][1:]

                            self.datafile = pd.concat([self.datafile, df_csv], axis=1)

                        elif file_name.endswith('.mat'):
                            #print(f'Reading MAT file: {file_path}')
                            mat_data = scipy.io.loadmat(file_path)
                            df_mat = pd.DataFrame()
                            #print(mat_data.keys())
                            if file_name == 'i_abc.mat':
                                data = mat_data['i_abc']
                                # Convert the data to a pandas DataFrame
                                df_mat = pd.DataFrame(data)
                                df_mat.columns = ['i_a', 'i_b', 'i_c']
                                df_mat = df_mat[df_mat.columns][1:]
                            elif file_name == 'i_dq.mat':
                                data = mat_data['i_dq']
                                # Convert the data to a pandas DataFrame
                                df_mat = pd.DataFrame(data)
                                df_mat.columns = ['i_d', 'i_q']
                                df_mat = df_mat[df_mat.columns][1:]
                                df_mat, i_d_avg = averaging(df_mat, 'i_d', 0, 40000, 'i_d_average')
                                self.average_id.append(i_d_avg)
                                df_mat, i_q_avg = averaging(df_mat, 'i_q', 0, 40000, 'i_q_average')
                                self.average_iq.append(i_q_avg)

                            self.datafile = pd.concat([self.datafile, df_mat], axis=1)
                self.dataset = pd.concat([self.dataset, self.datafile], axis=0, ignore_index=True)

        self.dataset['sin(Phi_el [])'] = np.sin(np.deg2rad(self.dataset['Phi_el []']))
        self.dataset['cos(Phi_el [])'] = np.cos(np.deg2rad(self.dataset['Phi_el []']))
        self.dataset['Electrical_frequency(Hz)'] = self.dataset['Speed']*(self.p/120)


    def visualisation(self, data):
        _ , axs = plt.subplots(1, 2, figsize = (12,4))
        axs[0].scatter(self.average_speed, self.average_torque)
        axs[0].set_xlabel("Rotational speed")
        axs[0].set_ylabel("Torque")
        axs[0].set_title(data) 
        
        axs[1].scatter(self.average_iq, self.average_id)
        axs[1].set_xlabel(f"i_q")
        axs[1].set_ylabel(f"i_d")
        axs[1].set_title(data)
        
        plt.show()

    def save(self, path, file_name1, file_name2):
        self.dataset = self.dataset[self.column_list]
        self.dataset = self.dataset.loc[:, ~self.dataset.columns.duplicated()]
        # full_path = path + file_name1
        # self.dataset.to_excel(full_path)
        full_path = path + file_name2
        self.dataset.to_hdf(full_path, key="table")


# List of specific file names
files = ['SpannungLeiterLeiter.csv', 'Strangstrom.csv', 'Drehmoment.csv', 'i_abc.mat', 'i_dq.mat', 'Winkel_el_mech.csv']

columns = ['Time [ms]', 'VM9.V [V]', 'VM7.V [V]', 'VM8.V [V]','Phi_el []','sin(Phi_el [])','cos(Phi_el [])','frq_el [Hz]','Electrical_frequency(Hz)','Speed', 
           'i_a', 'i_b', 'i_c', 'i_d', 'i_q','i_d_average', 'i_q_average','Moving1.Torque [NewtonMeter]', 'Average Torque', 'Moving1.Position [deg]']

# Specify the main directory
train_path = '/Users/chiragangadi/Uni Siegen/WHB_IAS/02_Data/ErgebnisseSimulationen'

train_data = DataExtract(train_path, files, columns)
train_data.generate()
train_data.save(train_path, "/train_dataset.xlsx", "/train_dataset.h5")
train_data.visualisation("Train data")