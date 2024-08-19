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

fn KQV_calculation(mat: Tensor[DType.float32], weight: Tensor[DType.float32], baises: Tensor[DType.float32], num_heads:Int,
                   head_dim:Int, multiplication:Model, transpose:Model, addition :Model, transpose_12:Model, 
                   transpose_01:Model) raises -> Tensor[DType.float32]:
    var results = transpose.execute("input0", weight)
    var Q_T = results.get[DType.float32]("output0")
    results = multiplication.execute("input0", mat, "input1", Q_T)
    var Q = results.get[DType.float32]("output0")
    results = addition.execute("input0", Q, "input1", baises)
    var Q_B = results.get[DType.float32]("output0")

    var known_product = 1 * num_heads * head_dim
    var inferred_dimension = Q_B.num_elements() // known_product
    var t:TensorShape = (inferred_dimension,1, num_heads, head_dim)
    Q_B = Q_B.reshape(t)
    results = transpose_12.execute("input0", Q_B)
    Q = results.get[DType.float32]("output0")
    
    t = (inferred_dimension, 1 * num_heads, head_dim)
    Q = Q.reshape(t)
    results = transpose_01.execute("input0", Q)
    Q = results.get[DType.float32]("output0")
    return Q

struct Attention:
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

    fn __init__(inout self, W_K: Tensor[DType.float32], W_Q: Tensor[DType.float32],W_V: Tensor[DType.float32],W_O: Tensor[DType.float32],
                b_K: Tensor[DType.float32],b_Q: Tensor[DType.float32],b_V: Tensor[DType.float32],b_O: Tensor[DType.float32], 
                embed_dim:Int, num_heads:Int):  
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

    fn forward(self, query:Tensor[DType.float32] , key: Tensor[DType.float32], value: Tensor[DType.float32], 
               multiplication:Model, transpose:Model, addition :Model, transpose_12:Model, transpose_01:Model, 
               transpose_21:Model, multiplication_3D:Model, division:Model) raises:
        var Q = KQV_calculation(query, self.W_Q, self.b_Q, self.num_heads, self.head_dim, multiplication, transpose, addition, 
                        transpose_12, transpose_01)
        var K = KQV_calculation(key, self.W_K, self.b_K, self.num_heads, self.head_dim, multiplication, transpose, addition, 
                        transpose_12, transpose_01)
        var V = KQV_calculation(value, self.W_V, self.b_V, self.num_heads, self.head_dim, multiplication, transpose, addition, 
                        transpose_12, transpose_01)
        

        print("Q.shape:", Q.shape())
        print("K.shape:", K.shape())
        print("V.shape:", V.shape())

        var results = transpose_21.execute("input0", K)
        K = results.get[DType.float32]("output0")

        print("K.shape: ",K.shape())

        results = multiplication_3D.execute("input0", Q, "input1", K)
        var attn_scores = results.get[DType.float32]("output0")

        var divisor = Tensor[DType.float32](1)
        divisor[0] = math.sqrt(self.head_dim)

        results = division.execute("input0", attn_scores, "input1", divisor)
        attn_scores = results.get[DType.float32]("output0")
        print("attn_scores.shape",attn_scores.shape())
        print(attn_scores)

        var attn_weights = Tensor[DType.float32] (attn_scores.shape())
        for i in range(attn_scores.shape()[0]):
            for j in range(attn_scores.shape()[1]):
                var score_slice = attn_scores[Index(i)]
                print(score_slice)
                print("===============================")





fn main() raises:

    var graph = Graph(in_types=List[Type](TensorType(DType.float32, "a","m", "n"), TensorType(DType.float32, "n","x")))
    var out = graph[0] @ graph[1]
    graph.output(out)
    graph.verify()
    var session = engine.InferenceSession()
    var multiplication = session.load(graph)

    var graph1 = Graph(in_types=List[Type](TensorType(DType.float32, "a","m")))
    var transposed = ops.transpose(graph1[0],-1,-2)
    graph1.output(transposed)
    graph1.verify()
    var transpose = session.load(graph1)

    var graph2 = Graph(in_types=List[Type](TensorType(DType.float32, "a","m", "n"), TensorType(DType.float32, "n")))
    var out2 = graph2[0] + graph2[1]
    graph2.output(out2)
    graph2.verify()
    var addition = session.load(graph2)

    var graph3 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "b", "c", "d")))
    transposed = ops.transpose(graph3[0],1,2)
    graph3.output(transposed)
    graph3.verify()
    var transpose_12 = session.load(graph3)

    var graph4 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "b", "c")))
    transposed = ops.transpose(graph4[0],0,1)
    graph4.output(transposed)
    graph4.verify()
    var transpose_01 = session.load(graph4)

    var graph5 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "b", "c")))
    transposed = ops.transpose(graph5[0],-2,-1)
    graph5.output(transposed)
    graph5.verify()
    var transpose_21 = session.load(graph5)

    var graph6 = Graph(in_types=List[Type](TensorType(DType.float32, "a","m", "n"), TensorType(DType.float32, "a","n", "m")))
    var out6 = graph6[0] @ graph6[1]
    graph6.output(out6)
    graph6.verify()
    var multiplication_3D = session.load(graph6)

    var graph7 = Graph(in_types=List[Type](TensorType(DType.float32, "a","m","n"), TensorType(DType.float32)))
    var div = graph7[0] / graph7[1]
    graph7.output(div)
    graph7.verify()
    var division = session.load(graph7)

    var graph8 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "b", "c")))
    var softmaxed = ops.softmax(graph8[0])
    graph8.output(softmaxed)
    graph8.verify()
    var softmax = session.load(graph8)

    Python.add_to_path(".")
    var mypython = Python.import_module("main")
    var x: PythonObject = mypython.inputs_outputs()

    var input_weights = List[Tensor[DType.float32]]()
    var output = numpy_to_tensor(x[-2])
    var input = numpy_to_tensor(x[-1])
    for i in range(len(x)-1):
        input_weights.append(numpy_to_tensor(x[i]))
    
    print("weights shape")
    for i in range(len(input_weights)):
        print(input_weights[i].shape())
    print("output shape:",output.shape())
    print("input shape:",input.shape())


    var layer1 = Attention(input_weights[0],input_weights[1],input_weights[2],input_weights[3],input_weights[4],
                           input_weights[5],input_weights[6],input_weights[7],512,8)

    var xd = layer1.forward(input,input,input,multiplication, transpose, addition, transpose_12, transpose_01, transpose_21,
                            multiplication_3D, division)