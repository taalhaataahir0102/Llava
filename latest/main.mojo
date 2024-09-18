from max.engine import InputSpec, InferenceSession
from python import Python, PythonObject
from utils.index import Index
from time import now
from max.graph import Graph, TensorType, Type, ops, Symbol
from max import engine
from max.tensor import Tensor, TensorShape
from max.engine import Model
from algorithm import sum
from utils.numerics import inf
from algorithm import parallelize
from memory import memcpy, memcmp, memset_zero


fn numpy_to_tensor(numpy_array: PythonObject) raises -> Tensor[DType.float32]:
    var tensor_shape = numpy_array.shape
    var tensor_rank = len(numpy_array.shape)
    var shape_list: List[Int]  = List[Int]()
    for i in range(tensor_rank):
        shape_list.append(tensor_shape[i].__int__())

    var tensor = Tensor[DType.float32] (shape_list)

    memcpy(tensor.unsafe_ptr(), numpy_array.__array_interface__['data'][0].unsafe_get_as_pointer[DType.float32](), tensor.num_elements())
        
    return tensor


fn tensor_to_numpy(tensor: Tensor[DType.float32]) raises -> PythonObject:
    var np = Python.import_module("numpy")
    var tensor_shape = tensor.shape()
    var tensor_rank = tensor.rank()

    var python_list = Python.evaluate("list()")
    for i in range(tensor_rank):
        _ = python_list.append(tensor_shape[i])

    var numpy_array:PythonObject = np.zeros(python_list, dtype=np.float32)
    memcpy(numpy_array.__array_interface__['data'][0].unsafe_get_as_pointer[DType.float32](), tensor.unsafe_ptr(), tensor.num_elements())
    return numpy_array^


fn main() raises:
    var mypython = Python.import_module("helper")
    var image_path = "download.jpeg"

    var preprocessed_image = mypython.image_preprocessing(image_path)
    print(preprocessed_image)
