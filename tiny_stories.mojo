from max.engine import InputSpec, InferenceSession
from python import Python
from utils.index import Index
from time import now
from max.graph import Graph, TensorType, Type, ops, Symbol
from max import engine
from max.tensor import Tensor, TensorShape
from max.engine import Model
from algorithm import sum
from utils.numerics import inf


alias batch_size = 1
alias d_model = 512  # Embedding dimension
alias num_heads = 12
alias sequence_length = 5

alias vocab_size = 50257

alias head_dim = d_model // num_heads

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

fn embeddings(emb_table:Tensor[DType.float32], context:PythonObject) raises -> Tensor[DType.float32]:
    var emb = Tensor[DType.float32] (1,len(context),d_model)
    for i in range(len(context)):
        emb.store(Index(0,i,0),emb_table.load[width = d_model](Index(context[i].__int__(), 0)))
    return emb

fn positional_embeddings(pos_emb_table:Tensor[DType.float32], token_embeddings:Tensor[DType.float32]) raises -> Tensor[DType.float32]:
    var pos_emb = Tensor[DType.float32](token_embeddings.shape())
    for i in range(pos_emb.shape()[1]):
        pos_emb.store(Index(0,i,0),pos_emb_table.load[width = d_model](Index(i, 0)))
    return pos_emb + token_embeddings

fn KQV(input :Tensor[DType.float32],weights: Tensor[DType.float32] ,multiplication:Model, transpose:Model) raises -> Tensor[DType.float32]:
    var results = transpose.execute("input0", weights)
    var k_w_t = results.get[DType.float32]("output0")
    results = multiplication.execute("input0", input, "input1", k_w_t)
    var k = results.get[DType.float32]("output0")
    return k

fn mask(inout input: Tensor[DType.float32], inf_tensor: Tensor[DType.float32]):
    for i in range(input.shape()[1]):
        if i == 0:
            input.store(Index(0,i,1), inf_tensor.load[width = 4](Index(0)))
        if i == 1:
            input.store(Index(0,i,2), inf_tensor.load[width = 3](Index(0)))
        if i == 2:
            input.store(Index(0,i,3), inf_tensor.load[width = 2](Index(0)))
        if i == 3:
            input.store(Index(0,i,4), inf_tensor.load[width = 1](Index(0)))

        

fn head(input: Tensor[DType.float32],key_weight: Tensor[DType.float32],query_weight: Tensor[DType.float32],
       value_weight: Tensor[DType.float32], inf_tensor: Tensor[DType.float32], multiplication:Model, transpose:Model, 
       transpose_21: Model, multiplication_3D: Model, division: Model, softmax:Model) 
       raises -> Tensor[DType.float32]:
    
    var out = Tensor[DType.float32]()
    var k = KQV(input, key_weight, multiplication, transpose)
    var q = KQV(input, query_weight, multiplication, transpose)
    var v = KQV(input, value_weight, multiplication, transpose)
    
    var results = transpose_21.execute("input0", k)
    var k_t = results.get[DType.float32]("output0")
    results = multiplication_3D.execute("input0", q, "input1", k_t)
    var wei = results.get[DType.float32]("output0")

    var divisor = Tensor[DType.float32](1)
    var base = Float32(k.shape()[-1])
    var exponent = Float32(0.5)
    divisor[0] = pow(base, exponent)
    results = division.execute("input0", wei, "input1", divisor)
    wei = results.get[DType.float32]("output0")
    mask(wei, inf_tensor)


    var attn_weights = Tensor[DType.float32] (wei.shape())
    for i in range(wei.shape()[0]):
        for j in range(wei.shape()[1]):
            var new_tens = Tensor[DType.float32] (wei.shape()[2])
            new_tens.store(0, wei.load[width = sequence_length](i,j,0))         # Extract the last dimension for the current slice
            var results = softmax.execute("input0", new_tens)
            var xd = results.get[DType.float32] ("output0")
            attn_weights.store(Index(i,j,0), xd.load[width = sequence_length](0))

    results = multiplication_3D.execute("input0", attn_weights, "input1", v)
    out = results.get[DType.float32]("output0")

    return out

fn layer_norm(input: Tensor[DType.float32], gema: Tensor[DType.float32], beta: Tensor[DType.float32],
    norm: Model) raises -> Tensor[DType.float32]:
    var results = norm.execute("input0", input, "input1", gema, "input2", beta)
    var output = results.get[DType.float32]("output0")
    return output

fn linear(input:Tensor[DType.float32], weights: Tensor[DType.float32], bias:Tensor[DType.float32], multiplication:Model,
          transpose:Model, addition:Model) raises -> Tensor[DType.float32]:
    var results = transpose.execute("input0", weights)
    var w_t = results.get[DType.float32]("output0")
    results = multiplication.execute("input0", input, "input1", w_t)
    var k = results.get[DType.float32]("output0")

    results = addition.execute("input0", k, "input1", bias)
    var out = results.get[DType.float32]("output0")
    return out

fn feedforward(input:Tensor[DType.float32], weights: List[Tensor[DType.float32]], bias: List[Tensor[DType.float32]], 
               multiplication:Model, transpose:Model, addition:Model, relu:Model) raises -> Tensor[DType.float32]:
    var x1  = linear(input, weights[0], bias[0], multiplication, transpose, addition)
    var results = relu.execute("input0", x1)
    var rel = results.get[DType.float32]("output0")
    var x2  = linear(rel, weights[1], bias[1], multiplication, transpose, addition)
    return x2

fn logits_extraction(input:Tensor[DType.float32], softmax_2d:Model) raises -> Tensor[DType.float32]:
    var out = Tensor[DType.float32] (input.shape()[0],input.shape()[2])
    for i in range(input.shape()[0]):
            for k in range(input.shape()[2]):
                out[Index(i,k)] = input[Index(i,sequence_length-1, k)]
    var results = softmax_2d.execute("input0", out)
    var sft = results.get[DType.float32]("output0")
    return sft


fn main() raises:

    print("Compiling Graphs")
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

    var graph2 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "b", "c"),TensorType(DType.float32, "c"), TensorType(DType.float32, "c")))
    var mean = ops.layer_norm(graph2[0],gamma = graph2[1], beta = graph2[2] , epsilon = 1e-5)
    graph2.output(mean)
    graph2.verify()
    var norm = session.load(graph2)

    var graph3 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "b", "c")))
    transposed = ops.transpose(graph3[0],-2,-1)
    graph3.output(transposed)
    graph3.verify()
    var transpose_21 = session.load(graph3)

    var graph4 = Graph(in_types=List[Type](TensorType(DType.float32, "a","m", "n"), TensorType(DType.float32, "a","n", "x")))
    var out6 = graph4[0] @ graph4[1]
    graph4.output(out6)
    graph4.verify()
    var multiplication_3D = session.load(graph4)

    var graph5 = Graph(in_types=List[Type](TensorType(DType.float32, "a","m","n"), TensorType(DType.float32)))
    var div = graph5[0] / graph5[1]
    graph5.output(div)
    graph5.verify()
    var division = session.load(graph5)

    var graph6 = Graph(in_types=List[Type](TensorType(DType.float32, "a")))
    var softmaxed = ops.softmax(graph6[0])
    graph6.output(softmaxed)
    graph6.verify()
    var softmax = session.load(graph6)

    var in_types = List[Type] (TensorType(DType.float32, 1, 5, 42), TensorType(DType.float32, 1, 5, 42), TensorType(DType.float32, 1, 5, 42), 
                               TensorType(DType.float32, 1, 5, 42), TensorType(DType.float32, 1, 5, 42), TensorType(DType.float32, 1, 5, 42),
                               TensorType(DType.float32, 1, 5, 42), TensorType(DType.float32, 1, 5, 42), TensorType(DType.float32, 1, 5, 42),
                               TensorType(DType.float32, 1, 5, 42), TensorType(DType.float32, 1, 5, 42), TensorType(DType.float32, 1, 5, 42)
                               )
    var graph7 = Graph(in_types=in_types)
    var inputs = List[Symbol] (graph7[0], graph7[1], graph7[2], graph7[3], graph7[4], graph7[5], graph7[6], graph7[7], graph7[8], graph7[9], 
                               graph7[10], graph7[11])
    var con = ops.concat(inputs, -1)
    graph7.output(con)
    graph7.verify()
    var concat = session.load(graph7)

    var graph8 = Graph(in_types=List[Type](TensorType(DType.float32, "a","m", "n"), TensorType(DType.float32, "n")))
    var out2 = graph8[0] + graph8[1]
    graph8.output(out2)
    graph8.verify()
    var addition = session.load(graph8)

    var graph9 = Graph(in_types=List[Type](TensorType(DType.float32, "a","m", "n")))
    var rel = ops.relu(graph9[0])
    graph9.output(rel)
    graph9.verify()
    var relu = session.load(graph9)

    var graph10 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "b")))
    var softmaxed_2D = ops.softmax(graph10[0])
    graph10.output(softmaxed_2D)
    graph10.verify()
    var softmax_2d = session.load(graph10)

    var inf_tensor = Tensor[DType.float32] (sequence_length)
    for i in range(inf_tensor.shape()[0]):
        inf_tensor[i] = -inf[DType.float32] ()
    ###########################################################################################################


    var input = "Once upon a time there"
    Python.add_to_path(".")
    var mypython = Python.import_module("weights")
    var token: PythonObject = (mypython.tokenizer(input))

    var emb_table = numpy_to_tensor( mypython.layer_weights('token_embedding_table.weight'))
    var token_embedding = embeddings(emb_table, token)
    print("token_embedding.shape():",token_embedding.shape())

    var pos_emb_table = numpy_to_tensor( mypython.layer_weights('position_embedding_table.weight'))
    var pos_embedding = positional_embeddings(pos_emb_table, token_embedding)
    print("pos_embedding.shape():",pos_embedding.shape())

    var input_to_block = pos_embedding
    for w in range(num_heads):
        var gema = numpy_to_tensor(mypython.layer_weights('blocks.'+str(w)+'.ln1.weight'))
        var beta = numpy_to_tensor(mypython.layer_weights('blocks.'+str(w)+'.ln1.bias'))
        var norm_layer = layer_norm(input_to_block, gema, beta, norm)
        print("norm_layer.shape():", norm_layer.shape())

        var weights_keys = List[Tensor[DType.float32]]()
        var weights_queries = List[Tensor[DType.float32]]()
        var weights_values = List[Tensor[DType.float32]]()
        for i in range(num_heads):
            weights_keys.append(numpy_to_tensor( mypython.layer_weights('blocks.'+str(w)+'.sa.heads.'+str(i)+'.key.weight')))
            weights_queries.append(numpy_to_tensor( mypython.layer_weights('blocks.'+str(w)+'.sa.heads.'+str(i)+'.query.weight')))
            weights_values.append(numpy_to_tensor( mypython.layer_weights('blocks.'+str(w)+'.sa.heads.'+str(i)+'.value.weight')))
        print("len(weights_keys):", len(weights_keys))

        var heads = List[Tensor[DType.float32]]()
        for i in range(num_heads):
            heads.append(head(norm_layer,weights_keys[i], weights_queries[i], weights_values[i], inf_tensor ,multiplication, transpose, transpose_21, 
                        multiplication_3D, division, softmax))

        var heads_py = Python.dict()
        for i in range(num_heads):
            heads_py["input"+str(i)] = tensor_to_numpy(heads[i])
        print("len(heads_py):", len(heads_py))

        var results = concat.execute(heads_py)
        var concatinated = results.get[DType.float32] ("output0")
        print("concatinated.shape()", concatinated.shape())
        print(concatinated)

        var weight_proj = numpy_to_tensor( mypython.layer_weights('blocks.'+str(w)+'.sa.proj.weight'))
        var bias_proj = numpy_to_tensor( mypython.layer_weights('blocks.'+str(w)+'.sa.proj.bias'))
        var proj = linear(concatinated, weight_proj, bias_proj, multiplication, transpose, addition)+input_to_block

        gema = numpy_to_tensor(mypython.layer_weights('blocks.'+str(w)+'.ln2.weight'))
        beta = numpy_to_tensor(mypython.layer_weights('blocks.'+str(w)+'.ln2.bias'))
        var norm_layer2 = layer_norm(proj, gema, beta, norm)
        print("norm_layer2.shape():", norm_layer2.shape())

        var weight_ffwd = List[Tensor[DType.float32]]()
        var bias_ffwd = List[Tensor[DType.float32]]()
        weight_ffwd.append(numpy_to_tensor( mypython.layer_weights('blocks.'+str(w)+'.ffwd.net.0.weight')))
        weight_ffwd.append(numpy_to_tensor( mypython.layer_weights('blocks.'+str(w)+'.ffwd.net.2.weight')))
        bias_ffwd.append(numpy_to_tensor( mypython.layer_weights('blocks.'+str(w)+'.ffwd.net.0.bias')))
        bias_ffwd.append(numpy_to_tensor( mypython.layer_weights('blocks.'+str(w)+'.ffwd.net.2.bias')))
        var ffwd = feedforward(norm_layer2, weight_ffwd, bias_ffwd, multiplication, transpose, addition, relu) + proj
        print("ffwd.shape():", ffwd.shape())

        input_to_block = ffwd

    var gema = numpy_to_tensor(mypython.layer_weights('ln_f.weight'))
    var beta = numpy_to_tensor(mypython.layer_weights('ln_f.bias'))
    var norm_layer = layer_norm(input_to_block, gema, beta, norm)

    var lm_head_weight = numpy_to_tensor(mypython.layer_weights('lm_head.weight'))
    var lm_head_bias = numpy_to_tensor(mypython.layer_weights('lm_head.bias'))
    var logits = linear(norm_layer,lm_head_weight,lm_head_bias,multiplication, transpose, addition)
    print("logits.shape():", logits.shape())

    var probs = logits_extraction(logits,softmax_2d)
    print("probs.shape():", probs.shape())

    var next_token = mypython.output(tensor_to_numpy(probs))
    print(input, next_token)