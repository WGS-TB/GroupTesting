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
    # TODO: check the yml file format -> exception handeling
    for keys, vals in config_inpt.items():
        if isinstance(vals, dict):
            print(vals)
            if config_inpt[keys]['mode'] == 'range':
                config_inpt[keys] = np.arange(*config_inpt[keys]['values'])
            elif config_inpt[keys]['mode'] == 'scalar':
                config_inpt[keys] = config_inpt[keys]['values']
            # TODO: remove mode = exact there is no need for it!
            elif config_inpt[keys]['mode'] == 'exact':
                config_inpt[keys].pop('mode')
                config_inpt[keys] = [config_inpt[keys]]
        else:
            config_inpt[keys] = [vals]
    return [dict(zip(config_inpt.keys(), vals)) for vals in itertools.product(*config_inpt.values())]


def config_reader(config_file_name):
    try:
        # Read config file
        with open(config_file_name, 'r') as config_file:
            config_dict = yaml.load(config_file, Loader=yaml.FullLoader)
    except:
        # TODO: this should be OSError
        e = sys.exc_info()[0]
        print("config file can not be found!")
        print("Error:", e)
    design_param = {'generate_groups': False, 'generate_individual_status': False, 'generate_test_results': False}
    decoder_param = {'decode': False, 'lambda_selection': False, 'evaluation': False}
    # Load params
    if 'design_param' in config_dict.keys():
        try:
            assert 'general' in config_dict['design_param'].keys(), "You should define general properties of design!"
            general_param = config_dict['design_param']['general']
            design_param.update(general_param)
            assert 'generate_groups' in config_dict['design_param'].keys(), \
                "You should define generate_group parameters!"
            generate_groups = config_dict['design_param']['generate_groups']
            design_param['generate_groups'] = True
            design_param.update(generate_groups)
        except:
            # TODO: this should be KeyError!
            e = sys.exc_info()[0]
            print("config file format is not correct!")
            print("Error:", e)
        if 'generate_individual_status' in config_dict['design_param'].keys():
            try:
                generate_individual_status = config_dict['design_param']['generate_individual_status']
                design_param['generate_individual_status'] = True
                design_param.update(generate_individual_status)
            except:
                # TODO: this should be KeyError!
                e = sys.exc_info()[0]
                print("Error:", e)
        if 'generate_test_results' in config_dict['design_param'].keys():
            try:
                generate_test_results = config_dict['design_param']['generate_test_results']
                design_param['generate_test_results'] = True
                design_param.update(generate_test_results)
            except:
                # TODO: this should be KeyError!
                e = sys.exc_info()[0]
                print("Error:", e)
    if 'decoder_param' in config_dict.keys():
        if 'decode' in config_dict['decoder_param'].keys():
            try:
                decode_param = config_dict['decoder_param']['decode']
                decoder_param['decode'] = True
                decoder_param.update(decode_param)
            except:
                # TODO: this should be KeyError
                e = sys.exc_info()[0]
                print("config file format is not correct!")
                print("Error1:", e)
        if 'lambda_selection' in config_dict['decoder_param'].keys():
            try:
                lambda_selection_param = config_dict['decoder_param']['lambda_selection']
                decoder_param['lambda_selection'] = True
                decoder_param.update(lambda_selection_param)
            except:
                # TODO: this should be KeyError
                e = sys.exc_info()[0]
                print("config file format is not correct!")
                print("Error2:", e)
        if 'evaluation' in config_dict['decoder_param'].keys():
            try:
                evaluation_param = config_dict['decoder_param']['evaluation']
                decoder_param['evaluation'] = True
                decoder_param.update(evaluation_param)
            except:
                # TODO: this should be KeyError
                e = sys.exc_info()[0]
                print("config file format is not correct!")
                print("Error3:", e)

    return config_decoder(design_param), config_decoder(decoder_param)


def path_generator(file_path, file_name, file_format):
    currentDate = datetime.datetime.now()
    # TODO: change dir_name to unique code if you want to run multiple configs
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


if __name__ == '__main__':
    design_param, decoder_param = config_reader('config.yml')
    print(design_param, decoder_param)