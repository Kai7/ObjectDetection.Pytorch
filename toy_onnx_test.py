import onnx
import caffe2.python.onnx.backend as backend
import numpy as np
import pdb

# Load the ONNX model
model = onnx.load('./alexnet.onnx')
print(type(model))

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
onnx.helper.printable_graph(model.graph)


rep = backend.prepare(model, device="CUDA:0") # or "CPU"
# For the Caffe2 backend:
#     rep.predict_net is the Caffe2 protobuf for the network
#     rep.workspace is the Caffe2 workspace for the network
#       (see the class caffe2.python.onnx.backend.Workspace)

input_tensor = np.random.randn(1, 3, 224, 224).astype(np.float32)
print(input_tensor)
print('Input.shape = {}'.format(str(input_tensor.shape)))
print('start inference')
outputs = rep.run(input_tensor)
# To run networks with more than one input, pass a tuple
# rather than a single numpy ndarray.
print('inference done.')

print(len(outputs))
print(outputs[0])
print('Output.shape = {}'.format(str(outputs[0].shape)))
print(type(outputs))

# pdb.set_trace()
print('Done')