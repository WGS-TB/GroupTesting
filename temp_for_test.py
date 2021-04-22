import os
import pandas as pd
import orloge as ol
import copy
import yaml
from utils import config_decoder

path = '/Users/hoomanzabeti/Desktop/GTResults/Noisy/Noise_without_CV'
alist = [i for i in os.walk(path)]
#dir_list = [i for i in alist[0][1] if i.startswith('Jan_')]
dir_list = [i for i in alist[0][1]]
#dir_list = [os.path.join(path,i,'CM.csv') for i in dir_list if os.path.isfile(os.path.join(path,i,'CM.csv'))]
dir_list = [(os.path.join(path,i,'CM.csv'),os.path.join(path,i,'config.yml')) for i in dir_list if os.path.isfile(os.path.join(path,i,'CM.csv'))]
#combined_csv = pd.concat([pd.read_csv(f) for f in dir_list ])
combined_csv= []
for i in range(len(dir_list)):
    with open(dir_list[i][1], 'r') as config_file:
        config_dict = yaml.load(config_file, Loader=yaml.FullLoader)
    opts = config_decoder(config_dict['opts'])[0]
    params = config_decoder(config_dict['param'])[0]
    temp_csv = pd.read_csv(dir_list[i][0])
    temp_csv['Noise'] = opts['test_noise_methods'][0]
    if opts['test_noise_methods'][0] == 'permutation':
        temp_csv['permutation_noise_prob'] = opts['permutation_noise_prob']
        temp_csv['lambda_e'] = params ['lambda_e']
    combined_csv.append(copy.copy(temp_csv))
combined_csv = pd.concat(combined_csv)
combined_csv.to_csv( os.path.join(path,"Noise_{}_without_cv_CM.csv".format(opts['test_noise_methods'][0])), index=False, encoding='utf-8-sig')

# log_list = [j for i in dir_list for j in os.walk(os.path.join(path,i,'Logs/'))]
# #log_list = [i for j in log_list for i in j[2] if i.startswith('log')]
#
# key_list = ['N','group_size','m','s','seed']
# total_list = []
# missed_list = []
# for i in log_list:
#     for j in i[2]:
#         value_list = [int(k) for k in j.strip('log_.txt').split('_')]
#         temp_dict = dict(zip(key_list,value_list))
#         try:
#             temp_dict.update(ol.get_info_solver(os.path.join(i[0],j), 'CPLEX'))
#         except:
#             missed_list.append(copy.copy(temp_dict))
#         total_list.append(copy.copy(temp_dict))
# pd.DataFrame(total_list).to_csv(os.path.join(path,"complete_log.csv"), index=False, encoding='utf-8-sig')
