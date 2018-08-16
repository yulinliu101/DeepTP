import os
import re
import numpy as np
def read_log_file(root_dir, file_name_list):
    losses = []
    for file_name in file_name_list:
        i = 0
        with open(os.path.join(root_dir, file_name), 'r') as log_file:
            for line in log_file:
                match_cost = re.findall('train_cost: [-+]?[0-9]*\.?[0-9]*', line)
                if len(match_cost) > 0:
                    loss = float(match_cost[0].split(": ")[1])
                    losses.append(loss)
    return np.array(losses)
                    
# losses = read_log_file('log', ['log_train_20180421-142055.log', 'log_train_20180423-154608.log', 'log_train_20180424-112057.log', 'log_train_20180425-001738.log'])