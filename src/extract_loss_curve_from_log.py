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
                    
losses = read_log_file('log', 
                        ['log_Lite_train_20181011-213601.log',
                         'log_Lite__train_20181012-103955.log'])
# print(np.argmin(losses[2000:]), losses[2000:].min())

import matplotlib.pyplot as plt
plt.plot(np.arange(losses.shape[0]), losses)
plt.show()