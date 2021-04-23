'''
The following code was copied from a stackoverflow post:
https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
'''
import re
import numpy as np
import itertools
import os
import datetime
import yaml
import sys
from shutil import copyfile
import inspect
import math
import warnings


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def config_decoder(config_inpt):
    '''
    TODO: I used the following stackoverflow post for the return part:
    https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    '''
    for keys, vals in config_inpt.items():
        if isinstance(vals, dict):
            # print(vals)
            if 'mode' in config_inpt[keys].keys():
                if config_inpt[keys]['mode'] == 'range':
                    config_inpt[keys] = np.arange(*config_inpt[keys]['values'])
                elif config_inpt[keys]['mode'] == 'list':
                    config_inpt[keys] = config_inpt[keys]['values']
            else:
                config_inpt[keys] = [config_inpt[keys]]
        else:
            config_inpt[keys] = [vals]
    return [dict(zip(config_inpt.keys(), vals)) for vals in itertools.product(*config_inpt.values())]


def config_input_or_params(current_dict, block_name, generate_label):
    if 'input' in current_dict.keys():
        current_setting = {'{}_input'.format(block_name): current_dict['input']}
        current_setting[generate_label] = 'input'
    else:
        current_setting = current_dict['params']
        current_setting[generate_label] = 'generate'
        if 'alternative_module' in current_dict.keys():
            current_setting[generate_label] = 'alternative_module'
            current_setting['{}_alternative_module'.format(block_name)] = current_dict['alternative_module']
    return current_setting


def config_reader(config_file_name):
    try:
        # Read config file
        with open(config_file_name, 'r') as config_file:
            config_dict = yaml.load(config_file, Loader=yaml.FullLoader)
    except:

        e = sys.exc_info()[0]
        print("config file can not be found!")
        print("Error:", e)
    design_param = {'generate_groups': False}
    decoder_param = {'decoding': False, 'decoder': False, 'lambda_selection': False, 'evaluation': False}
    # Load params
    if 'design' in config_dict.keys():
        assert 'groups' in config_dict['design'].keys(), \
            "You should define the 'groups' block in the config file!"
        generate_groups = config_input_or_params(config_dict['design']['groups'], 'groups', 'generate_groups')
        design_param.update(generate_groups)
        try:
            generate_individual_status = config_input_or_params(
                current_dict=config_dict['design']['individual_status'],
                block_name='individual_status',
                generate_label='generate_individual_status')
            generate_individual_status_keys = list(generate_individual_status.keys())
            for k in generate_individual_status_keys:
                if k in design_param.keys():
                    print('{} has been set before in the "group" block! The framework would continue with the initial'
                          ' value!'.format(k))
                    generate_individual_status.pop(k)
            design_param.update(generate_individual_status)
        except KeyError:
            print("Warning: 'individual_status' block is not found in the config file! Individual status is necessary"
                  " if the results need to be evaluated!")
            design_param['generate_individual_status'] = False
        try:
            generate_test_results = config_input_or_params(config_dict['design']['test_results'], 'test_results',
                                                           'generate_test_results')
            generate_test_results_keys = list(generate_test_results.keys())
            for k in generate_test_results_keys:
                if k in design_param.keys():
                    print('{} has been set before in the "group" or "individual_status" block! The framework '
                          'would continue with the initial value!'.format(k))
                    generate_test_results.pop(k)
            design_param.update(generate_test_results)
            design_param['test_results'] = True
        except KeyError:
            print("Warning: 'test_results' block is not found in the config file! Test results is necessary for"
                  " decoding!")
            design_param['generate_test_results'] = False
            design_param['test_results'] = False
    if 'decode' in config_dict.keys():
        assert design_param['test_results'], "It is not possible to decode without test results! Please define the " \
                                             "'test_results' block in the config file."
        decoder_param['decoding'] = True
        if 'decoder' in config_dict['decode'].keys():
            try:
                decode_param = config_input_or_params(config_dict['decode']['decoder'], 'decoder', 'decoder')
                # decoder_param['decoder'] = True
                decoder_param.update(decode_param)
            except:
                print("decoder format in the config file is not correct!")
                e = sys.exc_info()[0]
                print("Error:", e)
        if 'evaluation' in config_dict['decode'].keys():
            try:
                evaluation_param = config_dict['decode']['evaluation']
                decoder_param['evaluation'] = True
                decoder_param.update(evaluation_param)
            except:
                print("evaluation format in the config file is not correct!")
                e = sys.exc_info()[0]
                print("Error:", e)

    return config_decoder(design_param), config_decoder(decoder_param)


def path_generator(file_path, file_name, file_format, dir_name=None):
    currentDate = datetime.datetime.now()
    # TODO: change dir_name to unique code if you want to run multiple configs
    if dir_name is None:
        dir_name = currentDate.strftime("%b_%d_%Y_%H_%M")
    local_path = "./Results/{}".format(dir_name)
    path = os.getcwd()
    if not os.path.isdir(local_path):
        try:
            os.makedirs(path + local_path[1:])
        except OSError:
            print("Creation of the directory %s failed" % path + local_path[1:])
        else:
            print("Successfully created the directory %s" % path + local_path[1:])
    return path + local_path[1:] + "/{}.{}".format(file_name, file_format)


def report_file_path(report_path, report_label,report_extension, params):
    report_path = report_path + '/{}_N{}_g{}_m{}_s{}_seed{}.{}'.format(report_label, params['N'], params['group_size'],
                                                                        params['m'], params['s'], params['seed'],
                                                                       report_extension)
    return report_path


def result_path_generator(dir_name=None):
    current_path = os.getcwd()
    currentDate = datetime.datetime.now()
    if dir_name is None:
        dir_name = currentDate.strftime("%b_%d_%Y_%H_%M_%S")
    result_path = os.path.join(current_path, "Results/{}".format(dir_name))
    if not os.path.isdir(result_path):
        try:
            os.makedirs(result_path)
        except OSError:
            print("Creation of the directory %s failed" % result_path)
        else:
            print("Successfully created the directory %s " % result_path)
    # Copy config file
    copyfile('config.yml', os.path.join(result_path, 'config.yml'))
    return current_path, result_path


def inner_path_generator(current_path, inner_dir):
    inner_path = os.path.join(current_path, inner_dir)
    if not os.path.isdir(inner_path):
        try:
            os.makedirs(inner_path)
        except OSError:
            print("Creation of the directory %s failed" % inner_path)
        else:
            print("Successfully created the directory %s " % inner_path)
    return inner_path


def param_distributor(param_dictionary, function_name):
    passing_param = {k: param_dictionary[k] for k in inspect.signature(function_name).parameters if k in param_dictionary}
    remaining_param = {k: inspect.signature(function_name).parameters[k].default if
                       inspect.signature(function_name).parameters[k].default!= inspect._empty else None for k in
                       inspect.signature(function_name).parameters if k not in passing_param}
    return passing_param, remaining_param

    
def auto_group_size(N,s):
    group_size = round(math.log(0.5)/math.log(1-(int(s)/int(N))))
    if group_size > 32:
        group_size=32
    return group_size

if __name__ == '__main__':
    design_param, decoder_param = config_reader('config.yml')
    print(design_param, decoder_param)
    print(len(design_param), len(decoder_param))
