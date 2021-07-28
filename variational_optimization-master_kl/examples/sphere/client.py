import numpy as np
import grpc

import concurrent.futures

import examples.sphere.remote.sphere_pb2 as sphere_pb
import examples.sphere.remote.sphere_pb2_grpc as sphere_rpc

import examples.sphere.sphere as sphere

from variational_optimization.core import Tasker
from variational_optimization.core import Solver
from variational_optimization.core import SimpleGradientDescentUpdater, ExternalUpdater
from variational_optimization.sampler import NormalSampler

from variational_optimization.logger import PrintLogger


def run():
    # pass a model to the optim code (fit)
    model = sphere.SphereModel(mu=[5., 1., 3.],vars=[10., 20., 2.])
    lr = 0.03
    channel = grpc.insecure_channel('127.0.0.1:50051')
    stub = sphere_rpc.SphereStub(channel)

    def task_function(sample):
        proto = sample.as_proto(sphere_pb.Request)
        return stub.GetVal(proto)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        updater = SimpleGradientDescentUpdater(lr)
        tasker = Tasker(task_function=task_function, executor=executor)
        logger = PrintLogger('sphere-remote-example', num_print_steps=5)
        solver = Solver(model, updater, num_iters=200, num_samples=10, logger=logger, tasker=tasker)
        #log_optim = LoggingOptimizer(optim, self.config.logger, self.config.num_print_steps)
        result = solver.run()

    print("Converged Value (min is all zeros for first 3 params): {}".format(model.mode))
    return model.mode

if __name__ == "__main__":
    run()
