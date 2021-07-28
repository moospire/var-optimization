import numpy as np
import grpc
from scipy import optimize

import concurrent.futures

import examples.blender_shadow.remote.blender_shadow_pb2 as blender_shadow_pb
import examples.blender_shadow.remote.blender_shadow_pb2_grpc as blender_shadow_rpc

from variational_optimization.ops import ops
from variational_optimization.logger import PrintLogger

from variational_optimization.core import Tasker, Objective
from variational_optimization.core import Model, Solver, Variable
from variational_optimization.core import AdamUpdater, ExternalUpdater
from variational_optimization.sampler import NormalSampler


def norm_fit(result, target, alpha_affine = 1., beta_affine = 0., lr_affine=0.1, iters=1000, batch_sample=100):
    def norm(x):
        b,a = x
        return np.mean(np.square((result-b)*a - target))

    def jac_norm(x):
        b,a = x
        da = np.mean((2*b - 2*result)*(target - a*(result - b)))
        db = np.mean(2*a*(target - a*(result - b)))
        return np.array([db, da])

    result = optimize.minimize(norm,[beta_affine, alpha_affine], jac=jac_norm, method="BFGS")
    return result.x

class BlenderShadowModel(Model):
    def __init__(self, target_img, mu=[0., 0.], vars=[1., 1.]):
        Model.__init__(self)
        self.target_img = target_img
        self.light_position_sampler = NormalSampler(mu, vars)
        self.target_x = Variable(self.light_position_sampler, indices=[0])
        self.target_y = Variable(self.light_position_sampler, indices=[1])
        self.target_scale = 0.2
        self.target_shift = 0.
        self.best_cost = np.inf
        self.best_render = None
        self.best_params = None

    def loss(self, r):
        # return scalar after processing response message
        render = r.render
        target_shift, target_scale = norm_fit(render,
                                              self.target_img,
                                              alpha_affine = self.target_scale,
                                              beta_affine = self.target_shift,
                                              iters=5)
        a = 0.99
        self.target_shift = (a)*self.target_shift + (1-a)*target_shift
        self.target_scale = (a)*self.target_scale + (1-a)*target_scale
        cost_val = np.mean(np.abs((render-target_shift)*target_scale - self.target_img)**2)
        return cost_val

    def iteration_hook(self, results, samples, costs):
        for i, cost in enumerate(costs):
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_render = results[i].render
                self.best_params = [samples[i].target_x, samples[i].target_y]

def run():
    # pass a model to the optim code (fit)
    target_img = ops.load_blender_im('examples/blender_shadow/blender_test_1_3.png')
    model = BlenderShadowModel(target_img, mu=[0., 0.],vars=[.5, .5])

    lr = 0.1
    num_iters = 20
    num_samples = 10
    grpc_opts = [("grpc.lb_policy_name", "round_robin",)]
    channel = grpc.insecure_channel('127.0.0.1:50051', grpc_opts)
    stub = blender_shadow_rpc.BlenderShadowStub(channel)

    def task_function(sample):
        proto = sample.as_proto(blender_shadow_pb.Request)
        result = stub.GetVal(proto)
        return result

    updater = AdamUpdater(lr, beta1=0.9, beta2=0.99)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        logger = PrintLogger('blender-shadow-remote-example', num_print_steps=20)
        tasker = Tasker(task_function=task_function, executor=executor)
        objective = Objective(model, tasker)
        # solve this like we're using pytorch
        for n in range(num_iters):
            print('Iteration {}/{}'.format(n, num_iters))
            samples = objective.get_samples(num_samples)
            costs = objective.costs_at(samples)
            grad = objective.gradient(costs, samples)
            updater.step(objective, grad)
            print(model)


    ops.imsave('example_shadow_result.png', model.best_render)
    print("best sample params: x,y {}".format(model.best_params))

    print("Converged Value (GT is x: 1, y: 3): {}".format(model.mode))
    return model.mode

if __name__ == "__main__":
    run()
