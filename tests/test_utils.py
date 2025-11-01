import json
import struct

import numpy as np
import pytest

from triton_python_backend_utils import (
    TRITONSERVER_RESPONSE_COMPLETE_FINAL,
    InferenceRequest,
    InferenceResponse,
    ModelConfig,
    ResponseSender,
    Tensor,
    deserialize_bytes_tensor,
    get_input_config_by_name,
    get_input_tensor_by_name,
    get_output_config_by_name,
    get_output_tensor_by_name,
    numpy_to_triton_type,
    serialize_byte_tensor,
    triton_string_to_numpy,
    triton_to_numpy_type,
    using_decoupled_model_transaction_policy,
)


@pytest.fixture()
def base_model_config():
    return {
        "name": "toy_model",
        "max_batch_size": 2,
        "input": [
            {"name": "INPUT0", "data_type": "TYPE_INVALID", "dims": []},
        ],
        "output": [
            {"name": "OUTPUT0", "data_type": "TYPE_INVALID", "dims": []},
        ],
    }


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


def test_inference_request_response_sender():
    sender = ResponseSender()
    tensor = Tensor("t", np.array([1], dtype=np.int32))
    request = InferenceRequest(inputs=[tensor], response_sender=sender)
    assert request.get_response_sender() is sender


def test_serialize_byte_tensor_empty_returns_tuple():
    array = np.array([], dtype=np.object_)
    assert serialize_byte_tensor(array) == ()


def test_serialize_byte_tensor_mixed_object_values():
    array = np.array([b"abc", "xyz"], dtype=np.object_)
    expected = struct.pack("<I", 3) + b"abc" + struct.pack("<I", 3) + b"xyz"
    assert serialize_byte_tensor(array) == expected


def test_serialize_byte_tensor_non_object_returns_none():
    array = np.array([1, 2, 3], dtype=np.int32)
    assert serialize_byte_tensor(array) is None


def test_deserialize_bytes_tensor_round_trip():
    encoded = struct.pack("<I", 3) + b"one" + struct.pack("<I", 3) + b"two"
    decoded = deserialize_bytes_tensor(encoded)
    assert decoded.dtype == np.object_
    assert list(decoded) == [b"one", b"two"]


def test_get_input_tensor_by_name_returns_tensor():
    t1 = Tensor("input", np.array([1], dtype=np.int32))
    request = InferenceRequest(inputs=[t1])
    assert get_input_tensor_by_name(request, "input") is t1


def test_get_input_tensor_by_name_missing_returns_none():
    t1 = Tensor("input", np.array([1], dtype=np.int32))
    request = InferenceRequest(inputs=[t1])
    assert get_input_tensor_by_name(request, "missing") is None


def test_get_output_tensor_by_name():
    t1 = Tensor("a", np.array([1], dtype=np.int32))
    response = InferenceResponse(output_tensors=[t1])
    assert get_output_tensor_by_name(response, "a") is t1


def test_get_output_tensor_by_name_missing_returns_none():
    response = InferenceResponse(output_tensors=[])
    assert get_output_tensor_by_name(response, "missing") is None


def test_get_input_config_by_name():
    config = {"input": [{"name": "INPUT0", "dims": [1], "data_type": "TYPE_INT32"}]}
    result = get_input_config_by_name(config, "INPUT0")
    assert result == {"name": "INPUT0", "dims": [1], "data_type": "TYPE_INT32"}


def test_get_input_config_by_name_returns_none_when_missing():
    config = {"input": [{"name": "INPUT0", "dims": [1], "data_type": "TYPE_INT32"}]}
    assert get_input_config_by_name(config, "INPUT1") is None


def test_get_output_config_by_name():
    config = {"output": [{"name": "OUTPUT0", "dims": [1], "data_type": "TYPE_INT32"}]}
    result = get_output_config_by_name(config, "OUTPUT0")
    assert result == {"name": "OUTPUT0", "dims": [1], "data_type": "TYPE_INT32"}


def test_using_decoupled_model_transaction_policy():
    config = {"model_transaction_policy": {"decoupled": True}}
    assert using_decoupled_model_transaction_policy(config) is True
    config["model_transaction_policy"]["decoupled"] = False
    assert using_decoupled_model_transaction_policy(config) is False
    assert using_decoupled_model_transaction_policy({}) is False


@pytest.mark.parametrize(
    "triton_type, expected",
    [
        (1, np.bool_),
        (2, np.uint8),
        (3, np.uint16),
        (4, np.uint32),
        (5, np.uint64),
        (6, np.int8),
        (7, np.int16),
        (8, np.int32),
        (9, np.int64),
        (10, np.float16),
        (11, np.float32),
        (12, np.float64),
        (13, np.object_),
    ],
)
def test_triton_to_numpy_type(triton_type, expected):
    assert triton_to_numpy_type(triton_type) is expected


@pytest.mark.parametrize(
    "numpy_type, expected",
    [
        (np.bool_, 1),
        (np.uint8, 2),
        (np.uint16, 3),
        (np.uint32, 4),
        (np.uint64, 5),
        (np.int8, 6),
        (np.int16, 7),
        (np.int32, 8),
        (np.int64, 9),
        (np.float16, 10),
        (np.float32, 11),
        (np.float64, 12),
        (np.object_, 13),
        (np.bytes_, 13),
    ],
)
def test_numpy_to_triton_type(numpy_type, expected):
    assert numpy_to_triton_type(numpy_type) == expected


def test_triton_string_to_numpy():
    assert triton_string_to_numpy("TYPE_BOOL") is bool
    assert triton_string_to_numpy("TYPE_FP32") is np.float32


def test_model_config_set_max_batch_size(base_model_config):
    model_config = ModelConfig(json.dumps(base_model_config))
    model_config.set_max_batch_size(4)
    assert model_config.as_dict()["max_batch_size"] == 4


def test_model_config_set_max_batch_size_raises_with_lower_value(base_model_config):
    base_model_config["max_batch_size"] = 4
    model_config = ModelConfig(json.dumps(base_model_config))
    with pytest.raises(ValueError):
        model_config.set_max_batch_size(1)


def test_model_config_set_dynamic_batching_adds_when_missing(base_model_config):
    model_config = ModelConfig(json.dumps(base_model_config))
    model_config.set_dynamic_batching()
    assert "dynamic_batching" in model_config.as_dict()


def test_model_config_add_input_appends_new_entry(base_model_config):
    model_config = ModelConfig(json.dumps(base_model_config))
    model_config.add_input({"name": "INPUT1", "data_type": "TYPE_INT32", "dims": [1]})
    assert any(i["name"] == "INPUT1" for i in model_config.as_dict()["input"])


def test_model_config_add_input_updates_existing_invalid_entry(base_model_config):
    base_model_config["input"][0] = {"name": "INPUT0", "data_type": "TYPE_INVALID", "dims": []}
    model_config = ModelConfig(json.dumps(base_model_config))
    model_config.add_input({"name": "INPUT0", "data_type": "TYPE_FP32", "dims": [4]})
    updated = get_input_config_by_name(model_config.as_dict(), "INPUT0")
    assert updated["data_type"] == "TYPE_FP32"
    assert updated["dims"] == [4]


def test_model_config_add_input_requires_expected_fields(base_model_config):
    model_config = ModelConfig(json.dumps(base_model_config))
    with pytest.raises(ValueError):
        model_config.add_input({"name": "INPUT1", "data_type": "TYPE_INT32"})


def test_model_config_add_output_appends_new_entry(base_model_config):
    model_config = ModelConfig(json.dumps(base_model_config))
    model_config.add_output({"name": "OUTPUT1", "data_type": "TYPE_INT32", "dims": [1]})
    assert any(o["name"] == "OUTPUT1" for o in model_config.as_dict()["output"])


def test_response_sender_collects_values():
    sender = ResponseSender()
    response = InferenceResponse(output_tensors=[])
    sender.send(response=response)
    sender.send(flags=TRITONSERVER_RESPONSE_COMPLETE_FINAL)
    assert sender.values == [response, TRITONSERVER_RESPONSE_COMPLETE_FINAL]
