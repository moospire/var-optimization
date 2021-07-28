import pytest
from types import MethodType
from scipy import optimize
import numpy as np

from variational_optimization.core import Objective, Model, Solver, Tasker, Variable
from variational_optimization.core import Optimizer
from variational_optimization.core import SimpleGradientDescentUpdater, ExternalUpdater, AdamUpdater

from variational_optimization.sampler import NormalSampler, FiniteDiffSampler

def test_optim_interface_functional_integration():
    # minimal interace for external updater
    sampling_distribution = NormalSampler([5., 5.],[10., 10.])
    model = Model()
    model.sampler = sampling_distribution
    model.var_param = Variable(model.sampler, indices=[0, 1])
    model.counter = 0

    def loss(x):
        return np.linalg.norm(x.var_param)

    def iteration_hook(self, p, s, c):
        self.counter += 1

    model.loss = loss
    model.iteration_hook = MethodType(iteration_hook, model)

    tasker = Tasker()
    objective = Objective(model, tasker)
    # most methods seem to need at least 100 samples
    # there seems to be a tradeoff where more samples makes method more reliable
    # SLSQP seems to work well, 0.05-0.5s convergence time (sometimes fails to achieve good value)
    # maybe need to increase samples for this?
    # L-BFGS-B works well, seems more reliable, 1s convergence time (slightly faster than SGD)
    # prob need to cache cost eval to make more efficient
    # ^ above seem to always exit sucessfully
    # BFGS sometimes works but gives convergence errors
    # COBYLA sometimes works, more iterations, but runs very fast (0.05s)
    # need hess to try out others
    updater = ExternalUpdater(optimize.minimize, method='SLSQP')
    optim = Optimizer(objective, updater, num_samples=500)
    optim.run(100)
    # test the print __str__ on the final objective dist

    assert max([abs(m) for m in model.sampler.mu]) < 1.
    assert model.counter > 0

def test_solver_functional_integration():
    # minimal interace for external updater
    sampling_distribution = NormalSampler([5., 5.],[10., 10.])
    model = Model()
    model.sampler = sampling_distribution
    model.var_param = Variable(model.sampler, indices=[0, 1])
    def loss(x):
        return np.linalg.norm(x.var_param)

    model.loss = loss

    updater = SimpleGradientDescentUpdater(0.1)
    solver = Solver(model, updater, num_samples=100)
    solver.run(100)
    # test the print __str__ on the final objective dist

    assert max([abs(m) for m in model.sampler.mu]) < 1.

def test_finite_diff_functional_integration():
    # minimal interace for external updater
    sampling_distribution = FiniteDiffSampler(np.array([5., -5., 1., -1.]))
    #mixed_distribution = NormalSampler([5., 5.],[10., 10.])
    model = Model()
    model.sampler = sampling_distribution
    #model.sampler_other = mixed_distribution
    model.var_param = Variable(model.sampler, indices=[0, 1, 2, 3])
    model.counter = 0

    def loss(x):
        return np.linalg.norm(x.var_param)

    def iteration_hook(self, p, s, c):
        self.counter += 1

    model.loss = loss
    model.iteration_hook = MethodType(iteration_hook, model)

    tasker = Tasker()
    objective = Objective(model, tasker)
    samples = objective.get_samples(2)
    costs = objective.costs_at(samples)
    grad = objective.gradient(costs, samples)
    updater = AdamUpdater(0.5)
    updater.step(objective, grad)
    for n in range(100):
        # normally we want the number of samples to match the dimension of
        # unknowns for finite difference, but undersampling should still work.
        samples = objective.get_samples(2)
        costs = objective.costs_at(samples)
        grad = objective.gradient(costs, samples)
        updater.step(objective, grad)
    print(model.mode)
    assert max(model.mode.var_param) < 1.
