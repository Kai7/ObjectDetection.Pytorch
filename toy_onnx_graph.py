import onnx
from onnx import shape_inference
import caffe2.python.onnx.backend as backend
import numpy as np
import pdb

# Load the ONNX model
model = onnx.load('./alexnet.onnx')
model = shape_inference.infer_shapes(model)
model_graph = model.graph
print(type(model))
print(type(model_graph))

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
onnx.helper.printable_graph(model.graph)


rep = backend.prepare(model, device="CUDA:0") # or "CPU"
# For the Caffe2 backend:
#     rep.predict_net is the Caffe2 protobuf for the network
#     rep.workspace is the Caffe2 workspace for the network
#       (see the class caffe2.python.onnx.backend.Workspace)
#outputs = rep.run(np.random.randn(10, 3, 224, 224).astype(np.float32))
# To run networks with more than one input, pass a tuple
# rather than a single numpy ndarray.

# print(outputs[0])
#print(type(outputs))

pdb.set_trace()
print('Done')