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
    """
    Based on a stackoverflow post:
    https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
    """
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def config_decoder(config_inpt):
    """
    Helper Function to parse the config file.
    Based on the stackoverflow post for the return part:
    https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists

    Parameters:

        config_inpt (dict): dictionary/sub-dictionary from a config file.

    Returns:

        List of all combinations available in config_inpt.
    """
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
    """
    Helper function to parse the config file.

    Parameters:

        current_dict (dict): The current sub-dictionary corresponding to modules of the framework
        block_name (str): Name of the sub-dictionary.
        generate_label(str): generate label.

    Returns:

        current_setting (dict): A dictionary that specifies status of the module corresponding to current_dict.

    """
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
    """
    Main function to parse config file.

    Parameters:

        config_file_name(path): Path to the config file.

    Returns:

        A dictionary of parameters for design and decoding modules.
    """
    try:
        # Read config file
        with open(config_file_name, 'r') as config_file:
            config_dict = yaml.load(config_file, Loader=yaml.FullLoader)
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
                        print(
                            '{} has been set before in the "group" block! The framework would continue with the initial'
                            ' value!'.format(k))
                        generate_individual_status.pop(k)
                design_param.update(generate_individual_status)
            except KeyError:
                print(
                    "Warning: 'individual_status' block is not found in the config file! Individual status is necessary"
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
    except FileNotFoundError:
        sys.exit("Config file '{}' cannot be found!".format(config_file_name))
    except:
        e = sys.exc_info()[0]
        sys.exit(e)


def path_generator(file_name, file_format, dir_name=None):
    """
    Function to create Results directory and its sub-directories and return saving file path

    Parameters:

        file_name (str): Name of the file.
        file_format (str): File extension.
        dir_name (str): sub-directory name.

    Returns:
        File path
    """
    currentDate = datetime.datetime.now()
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


def report_file_path(report_path, report_label, report_extension, params):
    """
    Function to create a path for report files.

    Parameters:

        report_path (path): Report file path
        report_label (str): Report file label
        report_extension (str): Report file extension. e.g. txt, csv
        params (dict): A dictionary of parameters of the problem that the report file is corresponds to.
         N, m, group_size, s, seed.

    Return:

        report_path: Path to the report file.
    """
    report_path = report_path + '/{}_N{}_g{}_m{}_s{}_seed{}.{}'.format(report_label, params['N'], params['group_size'],
                                                                       params['m'], params['s'], params['seed'],
                                                                       report_extension)
    return report_path


def dict_key_checker(current_dict, current_key):
    """
    Function to check if a dictionary contains a key.

    Parameters:

        current_dict (dict): The dictionary.
        current_key (str): They key.

    Returns:
        True if the dictionary contains the key. False otherwise.
    """
    if current_key in current_dict.keys():
        return True
    else:
        return False


def result_path_generator(args):
    """
    Function to create a result directory.

    Parameters:

        args (Namespace): Namespace object contains args from command-line including output directory.

    Returns:

        current_path: Current path
        result_path: Path of the result directory. If output directory is not specified output directory would be Results
        with timestamped inner subdirectory.
    """
    current_path = os.getcwd()
    currentDate = datetime.datetime.now()
    if args.output_path is None:
        dir_name = currentDate.strftime("%b_%d_%Y_%H_%M_%S")
        result_path = os.path.join(current_path, "Results/{}".format(dir_name))
    else:
        dir_name = args.output_path
        result_path = os.path.join(current_path, dir_name)
    if not os.path.isdir(result_path):
        try:
            os.makedirs(result_path)
        except OSError:
            print("Creation of the directory %s failed" % result_path)
        else:
            print("Successfully created the directory %s " % result_path)
    # Copy config file
    if os.path.isfile(args.config):
        copyfile(args.config, os.path.join(result_path, 'config.yml'))
    return current_path, result_path


def inner_path_generator(current_path, inner_dir):
    """
    Function to create inner subdirectories for intermediate files like design matrix.

    Parameters:

        current_path (path): Current path.
        inner_dir (str): Inner subdirectory name.

    Returns:

        inner_path (path): Inner subdirectory path.
    """
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
    """
    Function to distribute parameters obtained form the config file to the corresponding modules/functions.

    Parameters:

        param_dictionary (dict): Dictionary of parameters.
        function_name (str): Name of the corresponding function.

    Returns:

        passing_param (dict): Dictionary of parameter which is passing to the function.
        remaining_param (dict): Dictionary of parameter which is NOT passing to the function.

    """
    passing_param = {k: param_dictionary[k] for k in inspect.signature(function_name).parameters if
                     k in param_dictionary}
    remaining_param = {k: inspect.signature(function_name).parameters[k].default if
    inspect.signature(function_name).parameters[k].default != inspect._empty else None for k in
                       inspect.signature(function_name).parameters if k not in passing_param}
    return passing_param, remaining_param


def auto_group_size(N, s):
    """
    Function to generate a information-theoretic optimal group size.

    Parameters:

        N (int): Population size.
        s (int): Number of infected individuals.

    Returns:

        group_size: Information-theoretic optimal group size. It will return group size 32 if Information-theoretic
         optimal group size > 32.
    """

    group_size = round(math.log(0.5) / math.log(1 - (int(s) / int(N))))
    if group_size > 32:
        print('The optimal group size is {}, However the maximum group size cannot be greater than 32!'
              ' The group size is set to 32'.format(group_size))
        group_size = 32
    return group_size


if __name__ == '__main__':
    """
    Main method for testing config_reader
    """
    design_param, decoder_param = config_reader('config.yml')
    print(design_param, decoder_param)
    print(len(design_param), len(decoder_param))
