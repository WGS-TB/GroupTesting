import pandas as pd

def decoder_reporter(ev_result):
    pd.DataFrame.from_dict(ev_result, orient='index').to_csv('Results/CM.csv')