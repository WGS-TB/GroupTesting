import os
import gurobipy as gp
from gurobipy import GRB


def GT_optimizer(file_path, param, name="cplex"):
    if name == "cplex":
        # TODO: cplex is imported here in case some users does not have the licence. In this case it would be called
        # TODO: when "cplex" is selected.
        import cplex
        prob = cplex.Cplex()
        prob.read(file_path)
        log_stream_status = param['log_stream']
        error_stream_status = param['error_stream']
        warning_stream_status = param['warning_stream']
        result_stream_status = param['result_stream']

        # prob.set_log_stream(log_stream_status)
        # prob.set_error_stream(error_stream_status)
        # prob.set_warning_stream(warning_stream_status)
        # prob.set_results_stream(result_stream_status)

        # Solving the problem
        prob.solve()
        groupTestingSln = {'w': [int(v[2:-1]) for v in prob.variables.get_names() if
                                 v[0] == 'w' and prob.solution.get_values(v) >= 0.5],
                           'Fn': [int(v[3:-1]) for v in prob.variables.get_names() if
                                  v[0:2] == 'ep' and prob.solution.get_values(v) >= 0.5],
                           'Fp': [int(v[3:-1]) for v in prob.variables.get_names() if
                                  v[0:2] == 'en' and prob.solution.get_values(v) >= 0.5]}
        #print(prob.solution.get_objective_value())
    elif name == "gurobi":
        import gurobipy as gp
        from gurobipy import GRB
        prob = gp.read(file_path)
        prob.optimize()
        groupTestingSln = {'w': [int(v.varName[2:-1]) for v in prob.getVars() if
                                 v.varName[0] == 'w' and v.x >= 0.5],
                           'Fn': [int(v.varName[3:-1]) for v in prob.getVars() if
                                  v.varName[0:2] == 'ep' and v.x >= 0.5],
                           'Fp': [int(v.varName[3:-1]) for v in prob.getVars() if
                                  v.varName[0:2] == 'en' and v.x >= 0.5]}
        #print(prob.objVal)

    return groupTestingSln


if __name__ == '__main__':
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, r'problem.mps')

    param = {}
    param['file_path'] = file_path
    param['log_stream'] = None
    param['error_stream'] = None
    param['warning_stream'] = None
    param['result_stream'] = None

    sln = GT_optimizer(file_path=file_path, param=param, name="cplex")
    #print(sln)
