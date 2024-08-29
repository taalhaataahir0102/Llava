# var transposed = ops.transpose_matrix(graph1[0])
from max.engine import InputSpec, InferenceSession
from max.graph import Graph, TensorType, Type, ops, Symbol
from max import engine
from max.tensor import Tensor, TensorShape
from max.engine import Model
from utils.index import Index
from utils.numerics import inf
from max.engine.tensor_map import TensorMap

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

# fn main () raises:
#     var shape1 = TensorShape(8,3,3)
#     var t1 = Tensor[DType.float32].rand(shape1)
#     print(t1)

#     var s = Slice(1,4)


#     var graph1 = Graph(in_types=List[Type] (TensorType(DType.float32, "a","m", "m")))
#     var x = ops.slicing.slice(graph1[0],s)
#     # var x = ops.softmax(graph1[0])
#     graph1.output(x)
#     graph1.verify()
#     var session = engine.InferenceSession()
#     var slicing = session.load(graph1)

#     var results = slicing.execute("input0", t1)
#     var xd = results.get[DType.float32]("output0")
#     print(xd)

####################################NOT WORKING#############################
# fn main () raises:
#     var shape1 = TensorShape(8,3,3)
#     var t1 = Tensor[DType.float32].rand(shape1)
#     var idx = Tensor[DType.int32] (1)  # Create a tensor for the index
#     idx[0] = 2
#     print(t1)
#     print(idx)

#     var graph1 = Graph(in_types=List[Type] (TensorType(DType.float32, "a","m", "m"), TensorType(DType.int32, "b")))
#     var x = ops.slicing.slice(graph1[0], graph1[1], axis = 0)
#     graph1.output(x)
#     graph1.verify()
#     var session = engine.InferenceSession()
#     var slicing = session.load(graph1)

#     var results = slicing.execute("input0", t1, "input1", idx)  # Pass the inputs as a dictionary
#     var xd = results.get[DType.float32] ("output0")
#     print(xd)


# fn main () raises:
#     var graph8 = Graph(in_types=List[Type](TensorType(DType.float32, "a")))
#     var softmaxed = ops.softmax(graph8[0])
#     graph8.output(softmaxed)
#     graph8.verify()
#     var session = engine.InferenceSession()
#     var softmax = session.load(graph8)

#     var shape1 = TensorShape(8,3,3)
#     var t1 = Tensor[DType.float32].rand(shape1)
#     print(t1)
#     var attn_weights = Tensor[DType.float32] (t1.shape())
#     for i in range(t1.shape()[0]):
#             for j in range(t1.shape()[1]):
#                 var new_tens = Tensor[DType.float32] (t1.shape()[2])
#                 new_tens.store(0, t1.load[width = 3](i,j,0))
#                 var results = softmax.execute("input0", new_tens)
#                 var xd = results.get[DType.float32] ("output0")
#                 print(xd)
#                 attn_weights.store(Index(i,j,0), xd.load[width = 3](0))
    
#     print(attn_weights)


# fn add(x:Int, y:Int)->Int:
#     return x+y

# fn main():
#     var x = add(3,4)
#     var y = add(5,8)
#     var z = add(31,2)
#     print(x,y,z)

# fn main () raises:
#     var graph9 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "b", "c")))
#     var mean = ops.mean(graph9[0], axis = -1)
#     graph9.output(mean)
#     graph9.verify()
#     var session = engine.InferenceSession()
#     var softmax = session.load(graph9)

#     var shape1 = TensorShape(1,3,4)
#     var t1 = Tensor[DType.float32].rand(shape1)
#     print(t1)

#     var results = softmax.execute("input0", t1)
#     var xd = results.get[DType.float32] ("output0")
#     print(xd)

# fn main () raises:
#     var graph10 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "b", "c"),TensorType(DType.float32, "c"), TensorType(DType.float32, "c")))
#     var mean = ops.layer_norm(graph10[0],gamma = graph10[1], beta = graph10[2] , epsilon = 1e-5)
#     graph10.output(mean)
#     graph10.verify()
#     var session = engine.InferenceSession()
#     var norm = session.load(graph10)

#     var shape1 = TensorShape(1,3,4)
#     var t1 = Tensor[DType.float32].rand(shape1)

#     var shape2 = TensorShape(4)
#     var t2 = Tensor[DType.float32](shape2)
#     for i in range(t2.num_elements()):
#         t2[i] = 1

#     var t3 = Tensor[DType.float32](shape2)
#     print(t1)
#     print(t2)
#     print(t3)
#     var results = norm.execute("input0", t1, "input1", t2,"input2", t3)
#     var xd = results.get[DType.float32] ("output0")
#     print(xd)

# fn main() raises:
#     var shape = TensorShape(50257,512)
#     var emb_table = Tensor[DType.float32].rand(shape)
#     print(emb_table)

#     # print(emb_table[Index(0,1)])

#     var x = emb_table.load[width = 512](Index(15496))
#     # print(x.shape())
#     print(x)


# fn main():
#     print(math.sqrt(5))
#     var base = Float64(10)
#     var exponent = Float64(0.5)
#     var x = pow(base, exponent)
#     print(x)

# fn main():
#     var x = Tensor[DType.float32] (5)
#     for i in range(x.shape()[0]):
#         x[i] = inf[DType.float32] ()
#     print(x)

fn main() raises:
    var x = Tensor[DType.float32] (1,5,42)
    var y = Tensor[DType.float32] (1,5,42)
    var z = Tensor[DType.float32] (1,5,42)

    for i in range(1):
        for j in range(5):
            for k in range(42):
                x[Index(i,j,k)] = 1
                y[Index(i,j,k)] = 2
                z[Index(i,j,k)] = 3


    var in_types = List[Type] (TensorType(DType.float32, 1, 5, 42), TensorType(DType.float32, 1, 5, 42), TensorType(DType.float32, 1, 5, 42))
    var graph10 = Graph(in_types=in_types)
    var inputs = List[Symbol] (graph10[0], graph10[1], graph10[2])
    var con = ops.concat(inputs, -1)
    graph10.output(con)
    graph10.verify()
    var session = engine.InferenceSession()
    var concat = session.load(graph10)
    
    var tensorMap = session.new_tensor_map()
    
    tensorMap.borrow("input"+str(0), x)
    tensorMap.borrow("input"+str(1), y)
    tensorMap.borrow("input"+str(2), z)

    print("tensorMap:", tensorMap)
    var results = concat.execute(tensorMap)
    var xd = results.get[DType.float32] ("output0")

    print("x:",x)
    print("y:", y)
    print("z:",z)
    print("xd:",xd)