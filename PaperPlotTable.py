import os
import pandas as pd
import orloge as ol
import copy
import yaml
from utils import config_decoder


def best_parameter_distribution(dir_path, file_condition='best_params_', output_file_name='best_params.csv',
                                save_output=True, noise_level=0):
    file_list = [f_name for f_name in os.listdir(dir_path) if f_name.startswith(file_condition)]
    path_list = [[os.path.join(dir_path, f_name), f_name] for f_name in file_list
                 if os.path.isfile(os.path.join(dir_path, f_name))]
    combined_csv = []
    for f in path_list:
        temp_csv = pd.read_csv(f[0], index_col=0)
        temp_info = pd.DataFrame(dict(zip(['N', 'group_size', 'm', 's', 'seed'], f[1].strip('.csv').split('_')[2:])), index=[0])
        temp_info['noise_level'] = noise_level
        temp_csv = pd.concat([temp_info, temp_csv], axis=1)
        combined_csv.append(copy.copy(temp_csv))
    combined_csv = pd.concat(combined_csv).reset_index().drop(['index'], axis=1)

    if save_output:
        combined_csv.to_csv(os.path.join(dir_path, output_file_name), index=False, encoding='utf-8-sig')
    else:
        return combined_csv

def cplex_log_files(dir_path, file_condition='log_', output_file_name='parsed_log.csv',
                                save_output=True):
    file_list = [f_name for f_name in os.listdir(dir_path) if f_name.startswith(file_condition)]
    path_list = [[os.path.join(dir_path, f_name), f_name] for f_name in file_list
                 if os.path.isfile(os.path.join(dir_path, f_name))]
    combined_csv = []
    for f in path_list:
        temp_info = pd.DataFrame(dict(zip(['N', 'group_size', 'm', 's', 'seed'], f[1].strip('.txt').split('_')[1:])), index=[0])
        flist = open(os.path.join(f[0])).readlines()
        print(flist[-1])
        temp_info['time(sec)'] = flist[-1].split(' ')[-4].strip('(')
        temp_info['ticks'] = flist[-1].split(' ')[-2].strip('(')
        combined_csv.append(copy.copy(temp_info))
    combined_csv = pd.concat(combined_csv).reset_index().drop(['index'], axis=1)
    if save_output:
        combined_csv.to_csv(os.path.join(dir_path, output_file_name), index=False, encoding='utf-8-sig')
    else:
        return combined_csv

# dir_path = '/Users/hoomanzabeti/Desktop/GTResults/Noisy/Threshold'
# dir_list = [i for i in os.walk(dir_path)][0][1]
#
# # print(pd.concat([best_parameter_distribution(os.path.join(dir_path,i,'Logs'),save_output=False) for i in dir_list]).reset_index())
# pd.concat([best_parameter_distribution(os.path.join(dir_path, i, 'Logs'), save_output=False)
#            for i in dir_list]).reset_index().to_csv(os.path.join(dir_path, 'best_params.csv'),
#                                                     index=False, encoding='utf-8-sig')

# dir_path = '/Users/hoomanzabeti/Desktop/GTResults/LP_IT'
# dir_list = [i for i in os.walk(dir_path)][0][1]
# pd.concat([cplex_log_files(os.path.join(dir_path, i, 'Logs'), save_output=False)
#            for i in dir_list]).reset_index().to_csv(os.path.join(dir_path, 'parsed_lp_logs.csv'),
#                                                     index=False, encoding='utf-8-sig')

dir_path = '/Users/hoomanzabeti/Desktop/GTResults/Noisy/Permutation/'
dir_list = [i for i in os.walk(dir_path)][0][1]
noise_level = []
for f in dir_list:
    with open(os.path.join(dir_path,f,'config.yml'), 'r') as config_file:
        config_dict = yaml.load(config_file, Loader=yaml.FullLoader)
    opts = config_decoder(config_dict['opts'])[0]
    noise_level.append(opts['permutation_noise_prob'])
    print(noise_level)
    print(dir_list)
# print(pd.concat([best_parameter_distribution(os.path.join(dir_path,i,'Logs'),save_output=False) for i in dir_list]).reset_index())
pd.concat([best_parameter_distribution(os.path.join(dir_path, i, 'Logs'), save_output=False,noise_level=noise_level[idx])
           for idx, i in enumerate(dir_list)]).reset_index().to_csv(os.path.join(dir_path, 'best_params.csv'),
                                                    index=False, encoding='utf-8-sig')