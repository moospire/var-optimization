# Variational Optimization

This is a project for variational optimization.

All the instuctions below assume you are in the repo root directory. This has been tested
using Ubuntu 18.04, but may also work on macos.

```
git clone https://github.com/mitmedialab/variational_optimization.git
cd variational_optimization
INSTALL_DIR=$(pwd)
```

## Install

I recommend installing miniconda to install dependencies and keep track of the environment.

Use pip to install dependencies:

(grpc is only required for examples)

```
pip install -r requirements.txt
```

For developer mode (only need to run once):

`pip install -e .`

Full install (not tested):

`pip install .`

### Installing examples

Install [docker engine](https://www.docker.com/products/docker-engine).
There is a Community Edition (CE) that will work just fine.

You can now create the necessary docker containers:

```
cd ${INSTALL_DIR}/examples
cd blender-shadow/remote
docker build . -t blender-shadow-worker:latest
#(TODO) cd ${INSTALL_DIR}/examples
#(TODO) cd blender-corner/remote
#(TODO) docker build . -t blender-corner-worker:latest
```

When running docker workers, you'll need to bring them up. For docker swarm
sphere example (requires grpcio is installed):

```
cd ${INSTALL_DIR}/examples/sphere/remote
docker stack deploy -c docker-compose.yml sphere
```

(Optional) Compile example protobufs. Compiled protobufs are added to source
control so this step is not needed unless you modify the `.proto` file.

```
cd ${INSTALL_DIR}/examples
cd sphere/remote
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. sphere.proto
```

## Run Tests

Install pytest:

```
pip install -U pytest
```

To run basic unit and integration tests:

`pytest tests/`


## Run Examples

To run examples in test (may take a few mins) regression and full stack tests (and more verbose output):

`pytest examples/ -vs --durations=0 --remote_sphere`

For the remote examples, you'll need to deploy the docker stack corresponding to the examples.

Examples may create a `results/` directory and subdirectory.

### Sphere Cost Function

The `sphere` example minimizes the norm of vector. It also demonstrates ways to combine multiple
samplers into the same problem.

`python ${INSTALL_DIR}/examples/sphere/sphere.py`

### Logging Visualization

For logging examples:

The results file in `results/` is labeled with a timestamp
(e.g. `results/1547220773-out`). Use this to create a visualization of the optimization:

```
cd results
../make_video.sh 1547220773
```

## Create a custom task

Refer to the `examples/` to see how to use the library on a specific task. For a minimal
example refere to `tests/test_integration.py`.

Every example implements a class that inherits `Model`. This model class defines
the samplers and corresponding variables we are interested in optimizing for.
The class attributes are scanned by the `Model` parent class and are used to dynamically
generate gradients for the optimization.

The implemented `Model` must also implement a `loss` function that must return a scalar
from the result of the sample. The examples show ways to easily integrate with docker
containers and communicate over grpc by creating a `task_function` to pass to the
 `Tasker` class. The examples define a `.proto` file which is compiled and used by the
 corresponding `client.py` and `server.py` modules.
