import time
from scipy import optimize

from variational_optimization.fit import DefaultRemoteConfig, Fit
from variational_optimization.logger import SaveResultsLogger
from variational_optimization.ops import ops
from variational_optimization.cost_definitions import docker_blender_test
from variational_optimization.core import SimpleGradientDescentUpdater, CovarianceGradientDescentUpdater
from variational_optimization.core import ExternalUpdater

class BlenderRemoteConfig(DefaultRemoteConfig):
    logger = SaveResultsLogger('results', save_key=int(time.time()), save_vis=True)
    updater = CovarianceGradientDescentUpdater(15.)
    sampler_init = [0., 0., 0., 1.1, 1.1]
    # L-BFGS-B
    #updater = ExternalUpdater(optimize.minimize, method='BFGS')
    #sampler_init = [0., 0., 0., 1.1, 1.1]
    worker_f = docker_blender_test
    # beta, alpha, x, y, --- a, b, c, d, e, f --- covbeta, covalpha, covx, covy
    #sampler_init = [0., .2, 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.2, 1.2]
    # fixed
    target = ops.load_blender_im('tests/blender_test_1_3.png')
    worker_params = {}
    num_workers = 20
    num_samples = 50
            
def fit(config, iters=100):
    config.logger.set_log_id('blender_shadow_example')
    blender_fit = Fit(config)
    result, distribution = blender_fit.fit(iters)
    return result, distribution

if __name__== "__main__":
    config = BlenderRemoteConfig
    result, distribution = fit(config, iters=100)
    print(result)
    print(distribution)
    