import numpy as np

from variational_optimization.ops.posdef import nearestPD
from variational_optimization.core import SamplingDistribution

from scipy.stats import multivariate_normal


class NormalSampler(SamplingDistribution):
    def __init__(self, mus=None, vars=None, scale_grad=True):
        SamplingDistribution.__init__(self)
        self.name = "MultivariateGaussian"
        self.dim = None
        self.mu = None
        self.Sigma = None
        self.inv_Sigma = None
        self.scale_grad = scale_grad
        # since dim = x + x*x = (x+1)*x for mu and sigma
        if mus is not None and vars is not None:
            self.parameters = self.params_from_means_vars(mus, vars)
            self.update_properties_from_parameters()

    def update_properties_from_parameters(self, parameters=None):
        if parameters is None:
            parameters = self.parameters
        self.dim = self.dim_from_parameters(parameters)
        self.mu = parameters[:self.dim]
        self.Sigma = self.Sigma_from_manifold_params(self.dim, parameters[self.dim:])
        # Warning to check Sigma explosion
        if np.linalg.cond(np.array(self.Sigma)) > 100.0:
            print("WARNING: Condition number high, is Sigma blowing up?")
            print(self.Sigma)
        # Consider lazily evaluating this for paramaters with many dimensions
        self.inv_Sigma = np.linalg.inv(np.array(self.Sigma)).tolist()

    # generator for skew matrix indices
    @staticmethod
    def skew_matrix_indices(dim):
        for m in range(dim - 1):
            for o in range(dim - m - 1):
                yield m, dim - 1 - o

    @staticmethod
    def Sigma_components_from_manifold_params(dim, manifold_params):
        # note the Identity and order of the exponentiation
        Gamma = np.exp(np.diag(manifold_params[-dim:])) * np.eye(dim)

        # for each row in A, fill from params list
        A = np.zeros((dim, dim))
        for i, (m, n) in enumerate(NormalSampler.skew_matrix_indices(dim)):
            A[m, n] = manifold_params[i]
        A = -A.T + A
        if np.linalg.cond(np.eye(dim) - A) > 100.0:
            print(
                "WARNING: Condition number high on manifold calc!"
            )
            print("Gamma: {}".format(Gamma))
        O = (np.eye(dim) + A) @ np.linalg.inv(np.eye(dim) - A)
        return Gamma, A, O

    @staticmethod
    def Sigma_from_manifold_params(dim, manifold_params):
        Gamma, A, O = NormalSampler.Sigma_components_from_manifold_params(
            dim, manifold_params
        )
        return O @ Gamma @ O.T

    def d_S_dgamma(self):
        if self.parameters is None:
            raise AttributeError("parameters must first be initialized!")
        manifold_params = self.parameters[self.dim :]
        Gamma, A, O = self.Sigma_components_from_manifold_params(
            self.dim, manifold_params
        )
        num_skew_params = int((self.dim ** 2 - self.dim) / 2)
        J = np.zeros((num_skew_params + self.dim, self.dim ** 2))
        I = np.eye(self.dim)
        for p in range(num_skew_params):
            A_i = np.zeros((self.dim, self.dim))
            for i, (m, n) in enumerate(NormalSampler.skew_matrix_indices(self.dim)):
                if i == p:
                    A_i[m, n] = 1
            A_i = -A_i.T + A_i
            dO_A_i = A_i @ np.linalg.inv(I - A) + (I + A) @ (
                np.linalg.inv(I - A) @ A_i @ np.linalg.inv(I - A)
            )
            J[p] = (O @ Gamma @ dO_A_i.T + dO_A_i @ Gamma @ O.T).flatten()
        for p in range(self.dim):
            d_eigen = np.zeros(self.dim)
            d_eigen[p] = np.exp(manifold_params[-self.dim:][p])
            J[num_skew_params + p] = (O @ np.diag(d_eigen) @ O.T).flatten()
        return J

    @property
    def mode(self):
        return self.mu

    @staticmethod
    def dim_from_parameters(parameters):
        p = 2
        n = 1
        while p < len(parameters):
            n += 1
            p += (n + 1)
        return n

    # this must be updated with new manifold projection
    @staticmethod
    def parameters_from_mu_Sigma(mu, Sigma):
        dim = len(mu)
        num_parameters = int(dim + (dim ** 2 + dim) / 2)
        parameters = [None] * num_parameters
        w, v = np.linalg.eig(Sigma)
        eig_gamma = np.log(w)
        I = np.eye(dim)
        skew_gamma_M = (I - v.T) @ np.linalg.inv(I + v.T)
        # unroll skew matrix
        skew_gamma = []
        for m, n in NormalSampler.skew_matrix_indices(dim):
            skew_gamma.append(skew_gamma_M[m, n])
        parameters[:dim] = mu
        parameters[-dim:] = eig_gamma
        parameters[dim:-dim] = skew_gamma
        assert None not in parameters
        return parameters

    @staticmethod
    def params_from_means_vars(means, variances):
        c_Sigma = []
        for i, variance in enumerate(variances):
            c_Sigma.append([0.0] * len(means))
            c_Sigma[-1][i] = variance
        return NormalSampler.parameters_from_mu_Sigma(means, c_Sigma)

    def sigma_scale(self, gradient):
        g = np.array([gradient[: self.dim]]).T
        g = g / np.linalg.norm(g)
        # scale based on covariance eigenvalues
        scale = g.T @ self.Sigma @ g
        return scale[0][0]

    # return a sampling distribution delta object
    def gradient(self, sample):
        num_parameters = int(self.dim + (self.dim ** 2 + self.dim) / 2)
        parameters = [None] * num_parameters

        JS_gamma = self.d_S_dgamma()
        dS = self.d_E_dS(sample)
        dgamma = JS_gamma @ np.array(dS).flatten()
        dmu = self.d_E_dmu(sample)

        if self.dim == 1:
            dmu = [dmu]

        parameters[: self.dim] = dmu
        parameters[self.dim :] = dgamma
        return self.gradient_cls(parameters)

    def sample(self, num_samples=1, seed=None):
        if seed is not None:
            np.random.seed(seed=np.array([hash(seed)]).astype(np.uint32))
        samples = np.random.multivariate_normal(self.mu, self.Sigma, num_samples)
        return samples.tolist()

    def prob_at(self, sample):
        return multivariate_normal.pdf(sample, mean=self.mu, cov=self.Sigma)

    # don't for manifold - override the parent class to increment for gaussian
    # def calc_parameters_updated(self, inc_parameters):
    #    inc_S = self.Sigma_from_parameters(inc_parameters)
    #    mu = self.update_mu(inc_mu)
    #    S = self.update_Sigma(inc_S)
    #    return self.parameters_from_mu_Sigma(mu, S)

    # to mirror S below - compute the difference update
    # return an updated mu
    def update_mu(self, dmu):
        return (np.array(dmu) + np.array(self.mu)).tolist()

    # return an updated Sigma
    # this is pure hacks
    # the new parameterization should be much better
    def update_Sigma(self, dS):
        # scale S so that diagonal entries don't go below zero...
        # (really need Jacobian w.r.t lie algebra SO(n) + eigenvalues >= 0)
        # if diag entries of diff_S are < 0, determine factor to correct
        np_S = np.array(self.Sigma)
        np_dS = np.array(dS)
        diff_S = (np_S - np_dS) * np.eye(np_S.shape[0])
        if np.min(diff_S) < 0.0:
            ind_min_diff_S = np.argmin(diff_S)
            a = np.ravel(np_S)[ind_min_diff_S] / (np.ravel(np_dS)[ind_min_diff_S])
            S_scaled = a * np_dS
        else:
            S_scaled = np_dS

        np_S += S_scaled
        # project to posdef matrix
        np_S = nearestPD(np_S)
        # note that S_inv blows up with a large condition number for S
        # add a small amount to diagonal entries to improve conditioning...
        # weight with scaling so that S stays simnilarly scaled...
        c = 0.01
        if np.linalg.cond(np_S) > 1.0 / c:
            # print('using condition number scaling')
            np_S = (1.0 - c) * np_S + c * np.eye(np_S.shape[0])
            np_S = nearestPD(np_S)
        return np_S.tolist()

    # from the matrix cookbook + sympy verification
    def d_E_dS(self, x):
        x = np.squeeze(x)
        cx = np.expand_dims(np.array(x), axis=0).T
        mu = np.expand_dims(np.array(self.mu), axis=0).T
        S_inv_np = np.array(self.inv_Sigma)
        dM = 0.5 * (S_inv_np @ ((cx - mu) @ (cx - mu).T) @ S_inv_np - S_inv_np)
        return dM.tolist()

    # from the matrix cookbook, note derivative w.r.t. mu
    def d_E_dmu(self, x):
        cx = np.expand_dims(np.array(x), axis=0).T
        mu = np.expand_dims(np.array(self.mu), axis=0).T
        dmu = np.array(self.inv_Sigma) @ (cx - mu)
        # scale according to Sigma... (justify this better?)
        if self.scale_grad:
            max_grad = dmu.T @ self.Sigma @ dmu
            norm = np.linalg.norm(dmu)
            if norm > 0.0:
                dmu = max_grad * (dmu / norm)
        return dmu.squeeze().tolist()

    def __str__(self):
        return "mu: \n{} \nSigma: \n{}\n".format(self.mu, self.Sigma)


class DiagNormalSampler(SamplingDistribution):
    def __init__(self, mus=None, vars=None, scale_grad=True):
        SamplingDistribution.__init__(self)
        self.dim = None
        self.mu = None
        self.cov = None
        self.inv_cov = None
        self.scale_grad = scale_grad
        if mus is not None and vars is not None:
            self.parameters = self.params_from_means_vars(mus, vars)
            self.update_properties_from_parameters()

    def update_properties_from_parameters(self, parameters=None):
        if parameters is None:
            parameters = self.parameters
        self.dim = int(len(parameters) / 2)
        self.mu = parameters[: self.dim]
        self.cov = self.cov_from_manifold_params(self.dim, parameters)
        self.inv_cov = 1.0 / self.cov


    @staticmethod
    def params_from_means_vars(means, variances):
        assert len(means) == len(variances)
        return np.concatenate([np.array(means), np.array(np.log(variances))])

    @staticmethod
    def cov_from_manifold_params(dim, manifold_params):
        cov = np.exp(manifold_params[dim :])
        return cov

    def d_cov_dgamma(self):
        if self.parameters is None:
            raise AttributeError("parameters must first be initialized!")
        Gamma = self.cov_from_manifold_params(self.dim, self.parameters)
        return Gamma

    # return a sampling distribution delta object
    def gradient(self, sample):
        num_parameters = int(2*self.dim)
        parameters = [None] * num_parameters

        Jcov_gamma = self.d_cov_dgamma()
        dcov = self.d_E_dS(sample)
        dgamma = Jcov_gamma * np.array(dcov).flatten()
        dmu = self.d_E_dmu(sample)

        if self.dim == 1:
            dmu = [dmu]

        parameters[: self.dim] = dmu
        parameters[self.dim :] = dgamma
        return self.gradient_cls(parameters)

    def sample(self, num_samples=1, seed=None):
        if seed is not None:
            np.random.seed(seed=np.array([hash(seed)]).astype(np.uint32))
        samples = []
        for n in range(num_samples):
            sample = [np.random.normal(m, np.sqrt(c)) for m, c in zip(self.mu, self.cov)]
            samples.append(sample)
        return samples

    def prob_at(self, sample):
        cumulative_log_prob = sum(
            [
                np.log(multivariate_normal.pdf(s, mean=[m], cov=[c]))
                for s, m, c in zip(sample, self.mu, self.cov)
            ]
        )
        return np.exp(cumulative_log_prob)

    @property
    def mode(self):
        return self.mu

    # assume independence and use NormalSampler as template
    def d_E_dS(self, x):
        x = np.squeeze(x)
        cx = np.array(x)
        mu = self.mu
        S_inv_np = self.inv_cov
        dM = 0.5 * (S_inv_np ** 2 * (cx - mu) ** 2 - S_inv_np)
        return dM

    # assume independence and use NormalSampler as template
    def d_E_dmu(self, x):
        cx = np.array(x)
        mu = self.mu
        dmu = self.inv_cov * (cx - mu)
        # scale according to Sigma... (justify this better?)
        if self.scale_grad:
            max_grad = (dmu * self.cov).dot(dmu)
            norm = np.linalg.norm(dmu)
            if norm > 0.0:
                dmu = max_grad * (dmu / norm)
        return dmu.squeeze()

    def __str__(self):
        return "mu: \n{} \ncovariance: \n{}\n".format(self.mu, self.cov)

class FiniteDiffSampler(SamplingDistribution):
    """
    Sampler that emulates a finite difference operation.

    Ensure that sample is called enough times for the number of dimensions to check.
    Otherwise, only m random dimensions will be sampled.

    Implements: (f(x0 + eps) - f(x0)) / eps

    Parameters
    ----------
    x0 : array_like
        The center of the sampling distribution
    epsilon : The amout to perturb each entry in f0 to calculate the finite difference

    Returns
    -------
    FiniteDiffSampler class

    """
    def __init__(self, x0, epsilon=0.0001):
        SamplingDistribution.__init__(self)
        self.x0 = x0
        self.dim = int(len(x0))
        self.parameters = x0
        self.epsilon = epsilon
        self.dim_index = 0
        self.shuffle_indices = list(range(self.dim))

    def update_properties_from_parameters(self, parameters=None):
        if parameters is None:
            parameters = self.parameters
        self.x0 = parameters
        self.dim = int(len(parameters))

    @property
    def mode(self):
        return self.x0

    def sample(self, num_samples=1):
        """
        sample a random m pertubations along dimenions

        sample may be called many times, so sampler keeps track of current index
        """
        samples = []
        for n in range(num_samples):
            # if dim_index is largest, reset self.dim_index
            if self.dim_index >= self.dim:
                self.dim_index = 0
            # shuffle indices if dim_index is 0
            if self.dim_index == 0:
                np.random.shuffle(self.shuffle_indices)
            index = self.shuffle_indices[self.dim_index]
            perturbation = np.zeros(self.dim)
            perturbation[index] = self.epsilon
            self.dim_index += 1
            sample = self.x0 + perturbation
            samples.append(sample)
        return samples

    def gradient(self, sample):
        parameters = (sample - self.x0) / self.epsilon**2
        return self.gradient_cls(parameters.tolist())

    def eps(self, sample):
        diff = sample - self.x0
        if max(diff) == 0.:
            return [None]
        else:
            return diff / self.epsilon**2

    # TODO: could do weighting here based on richardson interpolation?
    # for now return ones
    def prob_at(self, sample):
        return np.ones(len(sample))
