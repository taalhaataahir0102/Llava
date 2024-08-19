# var transposed = ops.transpose_matrix(graph1[0])
from max.engine import InputSpec, InferenceSession
from max.graph import Graph, TensorType, Type, ops, Symbol, SymbolicSlice
from max import engine
from max.tensor import Tensor, TensorShape
from max.engine import Model
from utils.index import Index

# fn main() raises:

#     var graph3 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "b", "c", "d")))
#     var transposed = ops.transpose(graph3[0],1,2)
#     graph3.output(transposed)
#     graph3.verify()
#     var session = engine.InferenceSession()
#     var transpose_12 = session.load(graph3)
    

#     var shape = TensorShape(3, 1, 2, 4)
#     var t = Tensor[DType.float32].rand(shape)
#     print(t)
#     var results = transpose_12.execute("input0", t)
#     var output = results.get[DType.float32]("output0")
#     print(output)

# fn main () raises:
#     var graph = Graph(in_types=List[Type](TensorType(DType.float32, "a","m", "n"), TensorType(DType.float32, "a","n", "m")))
#     var out = graph[0] @ graph[1]
#     graph.output(out)
#     graph.verify()
#     var session = engine.InferenceSession()
#     var multiplication = session.load(graph)

#     var shape1 = TensorShape(1,3,2)
#     var shape2 = TensorShape(1,2,3)
#     var t1 = Tensor[DType.float32].rand(shape1)
#     var t2 = Tensor[DType.float32].rand(shape2)

#     var results = multiplication.execute("input0", t1, "input1", t2)
#     var x = results.get[DType.float32]("output0")
#     print("t1.shape", t1.shape())
#     print("t2.shape", t2.shape())
#     print(t1)
#     print(t2)
#     print("attn_scores.shape",x.shape())

# fn main () raises:
#     var graph7 = Graph(in_types=List[Type](TensorType(DType.float32, "a","m","n"), TensorType(DType.float32)))
#     var div = graph7[0] / graph7[1]
#     graph7.output(div)
#     graph7.verify()
#     var session = engine.InferenceSession()
#     var division = session.load(graph7)

#     var shape1 = TensorShape(1,2,3)
#     var t1 = Tensor[DType.float32].rand(shape1)
#     var t2 = Tensor[DType.float32](1)
#     t2[0] = 0.5
    
#     print(t1)
#     print(t2)

#     var results = division.execute("input0", t1, "input1", t2)
#     var x = results.get[DType.float32]("output0")
#     print(x)

fn main () raises:
    var shape1 = TensorShape(8,3,3)
    var t1 = Tensor[DType.float32].rand(shape1)
    print(t1)
    var t2 = Tensor[DType.float32] (1) # Change this to rank 1 tensor
    t2[0] = 0 # This is the index where the slice should start
    print(t2)
    print(t2.rank())

    var graph1 = Graph(in_types=List[Type] (TensorType(DType.float32, "a","m", "n"), TensorType(DType.float32)))
    var x = ops.slicing.slice(graph1[0],graph1[1], axis=0)
    graph1.output(x)
    graph1.verify()
    var session = engine.InferenceSession()
    var slicing = session.load(graph1)

    # var results = slicing.execute(("input0", t1, "input1", t2))




