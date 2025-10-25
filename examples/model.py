# Model can be tested locally by running this file.

import json

import numpy as np

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT0")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            out_0 = in_0.as_numpy() * 2

            out_tensor_0 = pb_utils.Tensor("OUTPUT0", out_0.astype(self.output0_dtype))

            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(inference_response)

        return responses

    def finalize(self):
        print("Cleaning up...")


if __name__ == "__main__":
    # Run your model for local testing
    model = TritonPythonModel()

    # Example model configuration for testing
    model_config = {"output": [{"name": "OUTPUT0", "data_type": "TYPE_FP32"}]}
    args = {"model_config": json.dumps(model_config)}
    model.initialize(args)

    # Add code to create mock requests and call model.execute(requests)
    requests = [
        pb_utils.InferenceRequest(inputs=[pb_utils.Tensor("INPUT0", np.array([1.0, 2.0, 3.0], dtype=np.float32))])
    ]
    responses = model.execute(requests)

    assert len(responses) == 1
    output_tensor = responses[0].output_tensors()[0]
    print("Input Tensor:", requests[0].inputs()[0].as_numpy())
    print("Output Tensor:", output_tensor.as_numpy())
    assert np.array_equal(output_tensor.as_numpy(), np.array([2.0, 4.0, 6.0], dtype=np.float32))

    model.finalize()
