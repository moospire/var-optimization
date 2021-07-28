import pytest
import numpy as np

from examples.sphere import sphere, client

from examples.blender_shadow import client as shadow_client

@pytest.mark.remote_sphere
def test_remote_sphere_fit():
    converged = client.run()
    assert np.max(np.abs(np.array(converged[:3]))) < 1.

@pytest.mark.remote_shadow
def test_remote_blender_shadow_fit():
    converged = shadow_client.run()

def test_sphere_fit():
    converged = sphere.run()
    assert np.max(np.abs(np.array(converged.x))) < 1.
