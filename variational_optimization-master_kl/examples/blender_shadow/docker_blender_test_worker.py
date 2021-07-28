
import redis
import sys
import json
import numpy as np
from scipy import optimize
import os, sys

#import grpc

# if in the future we don't want to have to load a provided blender file
# bpy.ops.wm.read_homefile(use_empty=True)

# this is hardcoded for testing... (only if you request to render test target)
GT_TEST_XY = (1, 3)

argv = sys.argv
try:
    index = argv.index("--") + 1
except ValueError:
    index = len(argv)

argv = argv[index:]

# testing grpc
#with grpc.insecure_channel('localhost:50051') as channel:
#    print(channel)

print(argv)


# startup.blend --background --python {}/generate-random-scene.py --
# '/home/tswedish/nlos/blender-git/build_linux/bin/blender'
# add paths relative to the location of the python file
# there must be a better solution here?
file_dir = os.path.dirname(__file__)
# add below when breaking out into another file
if not file_dir in sys.path:
    sys.path.append(file_dir)
    print('adding {} to sys path'.format(file_dir))
from blender_ops import Scene
blender_scene = Scene()

variational_optim_module_path = os.path.join('/',*file_dir.split('/')[:-2])
if not variational_optim_module_path in sys.path:
    sys.path.append(variational_optim_module_path)

from pubsub import RedisInterface

remote_interface = RedisInterface(host='redis')

worker_channel = argv[0]
worker_uid = argv[1]
worker_timeout = float(argv[2])
render_test_target = int(argv[3])

print('render test target is : {}'.format(render_test_target))

print(worker_uid)
print('worker created on channel {}'.format(worker_channel))

res_channel = '{}-response'.format(worker_channel)

target = None

class Norm:
    target_shift = 0.
    target_scale = 1.

if render_test_target == 1:
    print('Rendering test target')
    target = blender_scene.render(*GT_TEST_XY, quality=1)
    print('done.')


# other params not implemented yet
def norm_fit(result, target, alpha_affine = 1., beta_affine = 0., lr_affine=0.1, iters=1000, batch_sample=100):
    def norm(x):
        b,a = x
        return np.mean(np.square((result-b)*a - target))

    def jac_norm(x):
        b,a = x
        da = np.mean((2*b - 2*result)*(target - a*(result - b)))
        db = np.mean(2*a*(target - a*(result - b)))
        return np.array([db, da])

    # taking mean of entries is prob not ideal
    # may want to ensure is PSD?
    # DON'T USE THIS UNTIL VERIFIED
    def hess_norm(x):
        b,a =x
        daa = np.mean(-2*b*result + 2*b**2 + 2*result**2 - 2*result*b)
        dab = np.mean(2*target - 2*a*result + 4*b*a - 2*a*result)
        dba = np.mean(2*target - 4*a*result + 4*a*b)
        dbb = np.mean(2*a**2)
        hess = np.array([[daa, dab],[dba, dbb]])
        return hess

    result = optimize.minimize(norm,[beta_affine, alpha_affine], jac=jac_norm, method="BFGS")
    return result.x

# quick normalize fit using adam (build in explicity?)
# assum recon, meas are numpy arrays
# maybe should use line search instead of adam?
def norm_fit_adam(recon, meas, alpha_affine = 1., beta_affine = 0., lr_affine=0.1, iters=1000, batch_sample=100):
    m_alpha = 0.
    v_alpha = 0.
    m_beta = 0.
    v_beta = 0.
    for n in range(iters):
        # subsample so gradients based on subset of residuals (performance option)
        samp_indices = np.random.choice(meas.reshape(-1).shape[0], batch_sample)
        # derivative of cost: (meas - (recon-beta)*alpha)**2
        d_alpha_affine = np.mean((2*beta_affine - 2*recon.flat[samp_indices])*(meas.flat[samp_indices] - alpha_affine*(recon.flat[samp_indices] - beta_affine)))
        d_beta_affine = np.mean(2*alpha_affine*(meas.flat[samp_indices] - alpha_affine*(recon.flat[samp_indices] - beta_affine)))
        # adam
        m_alpha = 0.5*m_alpha + (1-0.5)*d_alpha_affine
        v_alpha = 0.9*v_alpha + (1-0.9)*(d_alpha_affine**2)
        alpha_affine += - lr_affine * m_alpha / (np.sqrt(v_alpha) + 1e-6)
        m_beta = 0.5*m_beta + (1-0.5)*d_beta_affine
        v_beta = 0.9*v_beta + (1-0.9)*(d_beta_affine**2)
        beta_affine += - lr_affine * m_beta / (np.sqrt(v_beta) + 1e-8)

    return beta_affine, alpha_affine

def cost(sample, return_vis=False, beta=0., alpha=1., norm_params=None):
    x, y = sample[:2]
    result = blender_scene.render(x, y, quality=0.)
    assert target is not None, 'target not defined!'
    # run a few steps of auto-normalization
    if norm_params is not None:
        target_shift, target_scale = norm_fit(result,
                                              target,
                                              alpha_affine = norm_params.target_scale,
                                              beta_affine = norm_params.target_shift,
                                              iters=5)
        norm_params.target_shift = target_shift
        norm_params.target_scale = target_scale
    else:
        target_shift = 0.
        target_scale = 1.
    # calculate the updated L1 costh
    cost_val = np.mean(np.abs((result-target_shift)*target_scale - target))
    if return_vis:
        vis_arr = np.zeros((2, target.shape[0], target.shape[1], target.shape[2]))
        vis_arr[0] = target
        vis_arr[1] = (result-target_shift)*target_scale
        return cost_val, vis_arr
    return cost_val

if __name__== "__main__":
    remote_interface.subscribe(worker_channel)
    remote_interface.res_worker_ready(res_channel, worker_uid)
    norm = Norm()
    while True:
        # res: len(2) -> update config, res[0] == target, res[1] == params_list
        #      len(3) -> compute cost, res[0] == params, res[1] == uid, res[2] == return activations
        # TODO: return objects, not tuples
        res = remote_interface.get_worker_job(timeout=worker_timeout)
        if res is not None:
            if len(res) == 2:
                target = np.array(res[0])
            elif target is not None:
                params = res[0]
                uid = res[1]
                print(res)
                if res[2]:
                    cost_val, vis = cost(params, return_vis=True, norm_params=norm)
                    remote_interface.res_worker_cost_and_vis(res_channel,
                                                             cost_val,
                                                             uid,
                                                             vis,
                                                             [norm.target_shift, norm.target_scale])
                else:
                    remote_interface.res_worker_cost(res_channel,
                                                     cost(params, norm_params=norm),
                                                     uid)
