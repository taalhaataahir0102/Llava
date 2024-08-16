from max.engine import InputSpec, InferenceSession
from python import Python
from utils.index import Index
from time import now
from max.graph import Graph, TensorType, Type, ops
from max import engine
from max.tensor import Tensor, TensorShape
from max.engine import Model
# from max.engine import transpose_matrix

@always_inline
fn numpy_data_pointer[
    type: DType
](numpy_array: PythonObject) raises -> DTypePointer[type]:
    return DTypePointer[type](
        address=int(numpy_array.__array_interface__["data"][0])
    )

@always_inline
fn tensor_to_numpy[
    type: DType
](tensor: Tensor[type]) raises -> PythonObject:
    var np = Python.import_module("numpy")
    var tensor_shape = tensor.shape()
    var tensor_rank = tensor.rank()

    var python_list = Python.evaluate("list()")
    for i in range(tensor_rank):
        _ = python_list.append(tensor_shape[i])

    var numpy_array:PythonObject = np.zeros(python_list, dtype=np.float32)
    var dst = numpy_data_pointer[type](numpy_array)
    var src = tensor.unsafe_ptr()
    var length = tensor.num_elements()
    memcpy(dst, src, length)

    return numpy_array

@always_inline
fn numpy_to_tensor(numpy_array: PythonObject) raises -> Tensor[DType.float32]:
    
    var tensor_shape = numpy_array.shape
    var tensor_rank = len(numpy_array.shape)

    var shape_list: List[Int]  = List[Int]()
    for i in range(tensor_rank):
        shape_list.append(tensor_shape[i].__int__())

    var tensor = Tensor[DType.float32] (shape_list)

    var src = numpy_data_pointer[DType.float32](numpy_array)
    var dst = tensor.unsafe_ptr()
    var length = tensor.num_elements()
    memcpy(dst, src, length)
    return tensor

struct MyPair:
    var W_K: Tensor[DType.float32]
    var W_Q: Tensor[DType.float32]
    var W_V: Tensor[DType.float32]
    var W_O: Tensor[DType.float32]
    var b_K: Tensor[DType.float32]
    var b_Q: Tensor[DType.float32]
    var b_V: Tensor[DType.float32]
    var b_O: Tensor[DType.float32]
    
    var embed_dim: Int
    var num_heads: Int
    var head_dim: Int

    fn __init__(inout self, W_K: Tensor[DType.float32], W_Q: Tensor[DType.float32],W_V: Tensor[DType.float32],W_O: Tensor[DType.float32],b_K: Tensor[DType.float32],b_Q: Tensor[DType.float32],b_V: Tensor[DType.float32],b_O: Tensor[DType.float32], embed_dim:Int, num_heads:Int):  
        self.W_K = W_K
        self.W_Q = W_Q
        self.W_V = W_V
        self.W_O = W_O
        self.b_K = b_K
        self.b_Q = b_Q
        self.b_V = b_V
        self.b_O = b_O
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

    fn forward(self, query:Tensor[DType.float32] , key: Tensor[DType.float32], value: Tensor[DType.float32], model:Model) raises:
        var Q:Tensor[DType.float32] = model.execute("input0", query, "input1", self.W_Q)
        var K:Tensor[DType.float32] = model.execute("input0", key, "input1", self.W_K)
        var V:Tensor[DType.float32] = model.execute("input0", value, "input1", self.W_V)
        print("WoW")
        # Q = Q.reshape()
        
        # return None


fn main() raises:

    var graph = Graph(in_types=List[Type](TensorType(DType.float32, "a","m"), TensorType(DType.float32, "m","n")))
    var transposed = ops.transpose_matrix(graph[1])
    var out = graph[0] @ transposed
    graph.output(out)
    graph.verify()
    var session = engine.InferenceSession()
    var multiplication = session.load(graph)

    print("Hello!")
    Python.add_to_path(".")
    var mypython = Python.import_module("main")
    var x: PythonObject = mypython.inputs_outputs()
    print(len(x))

    print("================================")

    var inputs = List[Tensor[DType.float32]]()
    var output = numpy_to_tensor(x[-1])
    for i in range(len(x)-1):
        inputs.append(numpy_to_tensor(x[i]))
    
    for i in range(len(inputs)):
        print(inputs[i].shape())
    print(output.shape())
