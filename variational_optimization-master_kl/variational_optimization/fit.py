from variational_optimization.core import *
from variational_optimization.ops import ops
from variational_optimization.rpc.pubsub import DockerRedisInterface
from variational_optimization.sampler import NormalSampler
from variational_optimization.logger import SimplePrintLogger
from variational_optimization.worker import WorkerManager
from variational_optimization.cost_definitions import docker_blender_test, docker_matyas, sphere

# these (like the logger), should be factories to avoid passing the args
class DefaultConfig:
    cost_manager_cls = CostFunctionManager
    sampler_cls = NormalSampler
    logger = SimplePrintLogger()
    updater = SimpleGradientDescentUpdater(0.01)
    worker_f = sphere
    sampler_init = [5., 5., 0., 1.6, 1.6]
    num_samples = 10
    num_print_steps = 30

class DefaultRemoteConfig(DefaultConfig):
    cost_manager_cls = RemoteCostFunctionManager
    worker_manager_cls = WorkerManager
    remote_interface = DockerRedisInterface()
    # matyas is scaled a bit different... prob should stardardize test cost magnitudes
    updater = SimpleGradientDescentUpdater(0.1)
    worker_f = docker_matyas
    target = None
    worker_params = {}
    num_workers = 10


class Fit:
    def __init__(self, config):
        # don't do much on init
        self.is_remote = hasattr(config, 'remote_interface')
        self.config = config

    def run_update_loop(self, iters, cost_manager):
        parameters = self.config.parameter_cls(self.config.param_config, self.config.sampler_cls)
        objective = Objective(parameters, cost_manager)
        optim = Optimizer(objective, self.config.updater, num_samples=self.config.num_samples)
        log_optim = LoggingOptimizer(optim, self.config.logger, self.config.num_print_steps)
        result = log_optim.run(iters)
        return result, optim.objective.parameter_interface

    # here we actually run the problem
    def fit(self, iters=100):
        if self.is_remote:
            remote_interface = self.config.remote_interface
            remote_interface.register()
            result = -1.
            with self.config.worker_manager_cls(self.config.num_workers, self.config.worker_f, remote_interface) as worker_manager:
                cost_manager = self.config.cost_manager_cls(self.config.param_config, self.config.request_cls, worker_manager)
                result, parameters = self.run_update_loop(iters, cost_manager)
            return result, parameters
        else:
            cost_manager = self.config.cost_manager_cls(self.config.worker_f)
            return self.run_update_loop(iters, cost_manager)
