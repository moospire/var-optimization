import os
import time
import numpy as np

from variational_optimization.ops import ops
#from variational_optimization.ops import plot
from variational_optimization import ROOT_DIR

class PrintLogger:
    def __init__(self, log_id=None, num_print_steps=20):
        self.log_id = log_id
        self.num_print_steps = num_print_steps

    def process_iter(self, n, result, objective, iters=None):
        print('Time: {} Log Optim: {} - Iteration: {}/{} - Cost: {}'.format(time.time(), self.log_id, n, iters, result))
        print(objective.model)

    def set_log_id(self, log_id):
        self.log_id = log_id


class SaveResultsLogger(PrintLogger):
    def __init__(self, save_dir, save_key='test', log_id=None, save_vis=False):
        SimplePrintLogger.__init__(self, log_id=log_id)
        self.save_key = save_key
        self.save_dir = save_dir
        self.save_vis = save_vis
        self.file_num_ = 0

    # maybe accept optim cost results
    def process_iter(self, n, result, objective, iters=None):
        mode = objective.parameter_interface.sampling_distribution.mode
        # add parameters to log (for checking convergence), parameters should know how to parameterize things
        extra_params = objective.parameter_interface.extra_distribution_params
        update_string = '{},{},{},{},{},{},{}'.format(self.log_id, time.time(), n, iters, result, objective.parameter_interface, extra_params)
        print(update_string)

        # if a pool is provided, calculate activations for logging
        save_dir_name = '{}-out'.format(self.save_key)
        file_path = os.path.join(ROOT_DIR, '..', self.save_dir, save_dir_name)
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        # append the string to a file
        with open(os.path.join(file_path, 'inter.csv'), 'a') as file_handle:
            file_handle.write('{}\n'.format(update_string))
        self.file_num_ += 1
        # need to fix
        if self.save_vis:
            params = objective.parameter_interface.get_params_mode()
            act_workers = objective.activations_at([params])
            act_res = act_workers[0]
            act_arr = np.array(act_res[1])
            for w,act in enumerate(act_workers):
                print('worker {} norm params: {}'.format(w, act[2]))
            # save activation to numbered png in file_path
            # get the current line count
            # if this is the first file, let's save the target as well
            if self.file_num_ == 1:
                ops.imsave(os.path.join(file_path, 'target.png'), act_arr[0])
            file_name = '{:05d}.png'.format(self.file_num_)
            # just print the first in the act_arr for now
            ops.imsave(os.path.join(file_path, file_name), act_arr[1])
