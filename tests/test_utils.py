import numpy as np

from triton_python_backend_utils import InferenceRequest, InferenceResponse, Tensor


def test_tensor():
    t = Tensor("t", np.array([1, 2, 3], dtype=np.int32))
    assert t.name() == "t"
    assert np.array_equal(t.as_numpy(), np.array([1, 2, 3], dtype=np.int32))


def test_inference_request():
    t1 = Tensor("t1", np.array([1, 2, 3], dtype=np.int32))
    t2 = Tensor("t2", np.array([4, 5, 6], dtype=np.int32))
    ir = InferenceRequest(inputs=[t1, t2], model_name="model")
    assert ir.inputs() == [t1, t2]


def test_inference_response():
    t1 = Tensor("t1", np.array([1, 2, 3], dtype=np.int32))
    t2 = Tensor("t2", np.array([4, 5, 6], dtype=np.int32))
    ir = InferenceResponse(output_tensors=[t1, t2])
    assert ir.output_tensors() == [t1, t2]
