import numpy as np

# remove sampling distribution class once functionality moves to Model
from variational_optimization.core import Model, Variable, Tasker, Objective
from variational_optimization.core import AdamUpdater
from variational_optimization.sampler import DiagNormalSampler


class SphereModel(Model):
    def __init__(self, mu=[0.], vars=[1.]):
        Model.__init__(self)
        #self.sampler1 = DiagNormalSampler(mu, vars)
        self.sampler = DiagNormalSampler(mu, vars)
        self.x = Variable(self.sampler, indices=list(range(len(mu))))
        # we can also store state here
        self.cost_calls = 0
        self.best_cost = np.inf
        self.best_val = np.array([])

    def loss(self, r):
        # return scalar after processing response message
        b = r.x
        return np.linalg.norm(b)

    def iteration_hook(self, r_arr, s_arr, c_arr):
        print('MEAN COST: {}'.format(1./len(c_arr) * sum(c_arr)))
        self.cost_calls += len(c_arr)
        print('Best COST: {}'.format(self.best_cost))
        for i,c in enumerate(c_arr):
            if c < self.best_cost:
                self.best_cost = c
                self.best_val = s_arr[i].x

def run():
    # "MNIST" test (simpler)
    num_dims = 3*3
    mu = np.ones(num_dims)
    vars = np.ones(num_dims)
    model = SphereModel(mu=mu, vars=vars)
    lr = 0.01
    updater = AdamUpdater(lr)
    tasker = Tasker()
    objective = Objective(model, tasker)
    # do one big update first
    samples = objective.get_samples(500)
    costs = objective.costs_at(samples)
    grad = objective.gradient(costs, samples)
    updater.step(objective, grad)
    num_iters = 100
    # then iterate with smaller steps
    for n in range(num_iters):
        print('Iteration {}/{}'.format(n, num_iters))
        samples = objective.get_samples(200)
        costs = objective.costs_at(samples)
        grad = objective.gradient(costs, samples)
        print(objective.update_gradient(model.mode))
        # with diff_gradient in updater:
        # diff_grad = objective.diff_gradient(model.mode)
        # maybe make updater accept a learning rate for this update?
        # updater.step(objective, diff_grad)
        print(model.mode)
        updater.step(objective, grad)

    converged = model.mode
    print("Converged Value (min is all zeros): {}".format(converged))
    return converged

if __name__ == "__main__":
    run()
