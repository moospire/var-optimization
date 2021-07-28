import numpy as np
import math
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
from numpy.linalg import inv

# remove sampling distribution class once functionality moves to Model
from variational_optimization.core import Model, Solver, Variable, Tasker, Objective
from variational_optimization.core import SimpleGradientDescentUpdater, ExternalUpdater, AdamUpdater
from variational_optimization.sampler import NormalSampler


# t_ab and z_aa
def calc_t(d, co1, co2, mean1, mean2):
    # helper function to calculate kl divergence
    sum = np.add(co1, co2)
    # print("pain", np.linalg.det(sum))
    # print(math.log(2 * math.pi, 2))
    # ADDED ABS
    ans = -float(d) / 2 * math.log(2 * math.pi, 2) - 0.5 * math.log(abs(np.linalg.det(sum)), 2)
    ans += -0.5 * np.matmul(np.matmul(np.array((mean2 - mean1)).transpose(), inv(sum)), (mean2 - mean1))
    print(ans)
    # this next variable becomes too large
    ans2 = math.pow(2, ans)
    return ans2


# kl divergence between two gaussians
def kl_gaussians(co1, co2, mean1, mean2):
    d = len(co1)
    inv(co2)
    mulmatrix = np.matmul(inv(np.array(co2)), np.matrix(co1))
    # ADDED ABS
    ans = 0.5 * math.log(abs(np.linalg.det(co2)) / abs(np.linalg.det(co1)), 2) + 0.5 * mulmatrix.trace()[0, 0]
    ans += 0.5 * np.matmul(np.matmul(np.array(mean1 - mean2).transpose(), inv(co2)), (mean1 - mean2))
    ans -= float(d) * 0.5
    return ans


class SphereModel(Model):
    def __init__(self, dgm, mu=[0.], vars=[1.], sigma=[[1.]], num_comp = 1, num_vars = 1, div = 5):
        Model.__init__(self)
        self.sampler1 = NormalSampler()
        self.num_comp = num_comp
        self.num_vars = num_vars
        self.div = div
        self.dgm = dgm
        # print("sigma", sigma)
        # temp = []
        # for i in mu:
        #     for k in i:
        #         temp.append(k)
        # mu = temp
        # temp = []
        # for i in sigma:
        #     for k in i:
        #         temp.append(k)
        # sigma = temp
        # print("testing", mu, sigma)
        new_parameters = NormalSampler.parameters_from_mu_Sigma(mu, sigma)
        self.sampler1.update_properties_from_parameters(new_parameters)
        self.x = Variable(self.sampler1, indices=list(range(len(mu))))

        # self.sampler1 = NormalSampler(mu, vars)
        # self.x = Variable(self.sampler1, indices=list(range(len(mu))))
        # these are just to demonstrate/test compositionality
        # not used in loss (see below)
        # self.sampler2 = NormalSampler([1., 2.],[3., 2.])
        # self.fixed_param = 6
        # self.not_used = 5
        # self.var_param_1 = Variable(self.sampler2, indices=[0])
        # self.var_param_2 = Variable(self.sampler2, indices=[1])
        # we can also store state on the model
        self.cost_calls = 0
        self.best_cost = np.inf
        self.best_x = np.array([])

    def loss(self, r):
        # return scalar after processing response message
        print("thisis r", r)
        b = r.x
        d = self.num_comp
        v = self.num_vars
        # breaks apart the thing
        mus1 = []
        co1 = []
        w1 = []
        for i in range(d):
            mus1.append(b[0:v])
            b = b[v:]
        for i in range(d):
            temp = []
            for k in range(v):
                temp.append(b[0:v])
                b = b[v:]
            co1.append(temp)
        w1 = b

        dgm = self.dgm
        # print("co1", co1)
        # print("w1", w1)
        # print(self.sampler1.mu)
        # print(self.sampler1.Sigma)
        # d = self.numcomponents
        # l = len(self.sampler1.mu) // d
        # mus1 = []
        # for i in range(d):
        #     mus1.append(self.sampler1.mu[i * l:i * l + l])
        # co1 = []
        # for i in range(d):
        #     co1.append(self.sampler1.Sigma[i * l:i * l + l])
        #
        ans = 0
        for f in range(d):
            inside_sum = 0
            first_sum = 0
            for p in range(d):
                first_sum += w1[p] * math.pow(math.e, -1 * kl_gaussians(co1[f], co1[p], mus1[f], mus1[p]))
            second_sum = 0
            for p in range(d):
                # ADD Z_AA
                second_sum += w1[p] * calc_t(d, co1[f], co1[p], mus1[f], mus1[p])
            third_sum = 0
            for p in range(d):
                # ADD T_AB
                third_sum += dgm.weights_[p] * calc_t(d, co1[f], dgm.covariances_[p], mus1[f], dgm.means_[p])
            fourth_sum = 0
            for p in range(d):
                # ADD KL_DIV
                fourth_sum += dgm.weights_[p] * math.pow(math.e, -1 * kl_gaussians(co1[f], dgm.covariances_[p],
                                                                                   mus1[f], dgm.means_[p]))
            # print("ajhh", first_sum, second_sum, third_sum, fourth_sum)
            inside_sum = math.log(first_sum, 2) + math.log(second_sum, 2) - math.log(third_sum, 2) - math.log(
                fourth_sum, 2)
            ans += inside_sum * w1[f]
        return ans * 0.5 - self.div

        # return np.linalg.norm(b)

    def iteration_hook(self, results, samples, costs):
        print('MEAN COST: %s' % (1. / len(costs) * sum(costs)))
        self.cost_calls += len(costs)
        print(self)
        print('BEST VAL: %s' % (self.best_x))
        for i, cost in enumerate(costs):
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_x = samples[i].x


def dummygaussian(num_comp, num_vars):
    # creates a dummy guassian sample to measure against
    file = open("breast-cancer-wisconsin.data")
    X = file.readlines()
    file.close()
    total = []
    for term in X:
        term = term[:-1]
        term = term.split(',')
        if '?' in term:
            continue
        temp = []
        for k in term:
            temp.append(int(k))
        total.append(temp[1:num_vars + 1])
    scaler = preprocessing.StandardScaler().fit(total)
    total_scaled = scaler.transform(total)

    gm = GaussianMixture(n_components = num_comp).fit(total_scaled)
    return gm



def run():
    # pass a model to the optim code (fit)
    # model = SphereModel(mu=[-2., 3., -1, 1.],vars=[2., 2., 2., 2.])
    # model = SphereModel(mu = [-0.2, 1.], sigma = [[2., 1.], [1., 4.]]
    num_comp = 2
    num_vars = 4
    total_vars = num_comp * num_vars * (num_vars + 1) + num_comp
    mu = [1.] * total_vars
    sigma = ([[1.] * total_vars]) * total_vars
    dgm = dummygaussian(num_comp, num_vars)
    model = SphereModel(mu=mu, sigma=sigma, num_comp = num_comp, num_vars = num_vars, div = 5, dgm = dgm)
    lr = 0.03
    updater = AdamUpdater(lr)
    tasker = Tasker()
    objective = Objective(model, tasker, importance_sample_buffer_size=20)
    # solve this like we're using pytorch
    # do one big update first
    samples = objective.get_samples(50)
    costs = objective.costs_at(samples)
    grad = objective.gradient(costs, samples)
    updater.step(objective, grad)
    num_iters = 4
    # then iterate with smaller steps
    for n in range(num_iters):
        print('Iteration {}/{}'.format(n, num_iters))
        samples = objective.get_samples(30)
        costs = objective.costs_at(samples)
        grad = objective.gradient(costs, samples)
        updater.step(objective, grad)
        print(model)

    converged = model.mode
    print("Converged Value (min is all zeros): {}".format(converged))
    return converged


if __name__ == "__main__":
    run()
