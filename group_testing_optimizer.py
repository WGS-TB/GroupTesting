import os

def GT_optimizer(file_path, param, name="cplex"):
    if name == "cplex":
        import cplex
        prob = cplex.Cplex()
        prob.read(file_path)
        log_stream_status = param['log_stream']
        error_stream_status = param['error_stream']
        warning_stream_status = param['warning_stream']
        result_stream_status = param['result_stream']

        prob.set_log_stream(log_stream_status)
        prob.set_error_stream(error_stream_status)
        prob.set_warning_stream(warning_stream_status)
        prob.set_results_stream(result_stream_status)

        # Solving the problem
        prob.solve()
        sln = [int(v[1:]) for v in prob.variables.get_names() if v[0] == 'w']

    return sln


if __name__ == '__main__':
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, r'problem.mps')

    param = {}
    param['file_path'] = file_path
    param['log_stream'] = None
    param['error_stream'] = None
    param['warning_stream'] = None
    param['result_stream'] = None

    GT_optimizer(file_path=file_path, param=param, name="cplex")
