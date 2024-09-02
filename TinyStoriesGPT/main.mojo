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
from algorithm import parallelize

alias batch_size = 1
alias d_model = 512  # Embedding dimension
alias num_heads = 12
alias sequence_length = 5

alias vocab_size = 50257
alias total_len = 500


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

struct tokenEmbeddings:
    var emb_table:Tensor[DType.float32]

    fn __init__(inout self, emb_table:Tensor[DType.float32]):
        self.emb_table = emb_table
    
    fn forward(self, context:PythonObject) raises -> Tensor[DType.float32]:
        var emb = Tensor[DType.float32] (1,len(context),d_model)
        for i in range(len(context)):
            emb.store(Index(0,i,0),self.emb_table.load[width = d_model](Index(context[i].__int__(), 0)))
        return emb

struct positionalEmbeddings:
    var pos_emb_table:Tensor[DType.float32]

    fn __init__(inout self, pos_emb_table:Tensor[DType.float32]):
        self.pos_emb_table = pos_emb_table

    fn forward(self, token_embeddings:Tensor[DType.float32]) raises -> Tensor[DType.float32]:
        var pos_emb = Tensor[DType.float32](token_embeddings.shape())
        for i in range(pos_emb.shape()[1]):
            pos_emb.store(Index(0,i,0),self.pos_emb_table.load[width = d_model](Index(i, 0)))
        return pos_emb + token_embeddings


fn mask(inout input: Tensor[DType.float32], inf_tensor: Tensor[DType.float32]):
    for i in range(input.shape()[1]):
        if i == 0:
            input.store(Index(0,i,1), inf_tensor.load[width = sequence_length - 1](Index(0)))
        if i == 1:
            input.store(Index(0,i,2), inf_tensor.load[width = sequence_length - 2](Index(0)))
        if i == 2:
            input.store(Index(0,i,3), inf_tensor.load[width = sequence_length - 3](Index(0)))
        if i == 3:
            input.store(Index(0,i,4), inf_tensor.load[width = sequence_length - 4](Index(0)))

struct head(CollectionElement):
    var key_weight: Tensor[DType.float32]
    var query_weight: Tensor[DType.float32]
    var value_weight: Tensor[DType.float32]
    var inf_tensor: Tensor[DType.float32]

    fn __init__(inout self, key_weight: Tensor[DType.float32],query_weight: Tensor[DType.float32],
                value_weight: Tensor[DType.float32], inf_tensor: Tensor[DType.float32]):
        self.key_weight = key_weight
        self.query_weight = query_weight
        self.value_weight = value_weight
        self.inf_tensor = inf_tensor

    fn __copyinit__(inout self, existing: Self):
        self.key_weight = existing.key_weight
        self.query_weight = existing.query_weight
        self.value_weight = existing.value_weight
        self.inf_tensor = existing.inf_tensor
    
    fn __moveinit__(inout self, owned existing: Self):
        self.key_weight = existing.key_weight^
        self.query_weight = existing.query_weight^
        self.value_weight = existing.value_weight^
        self.inf_tensor = existing.inf_tensor^

    fn KQV(self, input:Tensor[DType.float32], weights: Tensor[DType.float32], multiplication:Model, transpose:Model) 
           raises -> Tensor[DType.float32]:
        var results = transpose.execute("input0", weights)
        var k_w_t = results.get[DType.float32]("output0")
        results = multiplication.execute("input0", input, "input1", k_w_t)
        var k = results.get[DType.float32]("output0")
        return k

    fn forward(self, input:Tensor[DType.float32], multiplication:Model, transpose:Model, 
                transpose_21: Model, multiplication_3D: Model, division: Model, softmax:Model) raises -> Tensor[DType.float32]:
        var out = Tensor[DType.float32]()
        var k = self.KQV(input, self.key_weight, multiplication, transpose)
        var q = self.KQV(input, self.query_weight, multiplication, transpose)
        var v = self.KQV(input, self.value_weight, multiplication, transpose)
        
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
        mask(wei, self.inf_tensor)


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

struct layer_norm(CollectionElement):
    var gamma: Tensor[DType.float32]
    var beta: Tensor[DType.float32]

    fn __init__(inout self, gema: Tensor[DType.float32], beta: Tensor[DType.float32]):
        self.gamma = gema
        self.beta = beta

    fn __copyinit__(inout self, existing: Self):
        self.gamma = existing.gamma
        self.beta = existing.beta
    
    fn __moveinit__(inout self, owned existing: Self):
        self.gamma = existing.gamma^
        self.beta = existing.beta^
    
    fn forward(self, input: Tensor[DType.float32], norm: Model) raises -> Tensor[DType.float32]:
        var results = norm.execute("input0", input, "input1", self.gamma, "input2", self.beta)
        var output = results.get[DType.float32]("output0")
        return output

struct linear(CollectionElement):
    var weights: Tensor[DType.float32]
    var bias: Tensor[DType.float32]

    fn __init__(inout self, weights: Tensor[DType.float32], bias:Tensor[DType.float32]):
        self.weights = weights
        self.bias = bias

    fn __copyinit__(inout self, existing: Self):
        self.weights = existing.weights
        self.bias = existing.bias
    
    fn __moveinit__(inout self, owned existing: Self):
        self.weights = existing.weights^
        self.bias = existing.bias^
    
    fn forward(self, input: Tensor[DType.float32], addition:Model, multiplication: Model, transpose: Model)
               raises -> Tensor[DType.float32]:
        var results = transpose.execute("input0", self.weights)
        var w_t = results.get[DType.float32]("output0")
        results = multiplication.execute("input0", input, "input1", w_t)
        var k = results.get[DType.float32]("output0")

        results = addition.execute("input0", k, "input1", self.bias)
        var out = results.get[DType.float32]("output0")
        return out


struct feedForward(CollectionElement):
    var weights: List[Tensor[DType.float32]]
    var bias: List[Tensor[DType.float32]]

    fn __init__(inout self, weights: List[Tensor[DType.float32]], bias:List[Tensor[DType.float32]]):
        self.weights = weights
        self.bias = bias
    
    fn __copyinit__(inout self, existing: Self):
        self.weights = existing.weights
        self.bias = existing.bias
    
    fn __moveinit__(inout self, owned existing: Self):
        self.weights = existing.weights^
        self.bias = existing.bias^
    
    fn forward(self, input: Tensor[DType.float32], multiplication: Model, addition: Model, transpose: Model, relu: Model) 
              raises -> Tensor[DType.float32]:
        var lin1  = linear(self.weights[0], self.bias[0])
        var x1 = lin1.forward(input, addition, multiplication, transpose)
        var results = relu.execute("input0", x1)
        var rel = results.get[DType.float32]("output0")
        var lin2  = linear(self.weights[1], self.bias[1])
        var x2 = lin2.forward(rel, addition, multiplication, transpose)
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

    print("Compiling Graphs", end = " ")
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

    print(".", end = "")

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

    print(".", end = "")

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

    print(".", end = "")

    var graph10 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "b")))
    var softmaxed_2D = ops.softmax(graph10[0])
    graph10.output(softmaxed_2D)
    graph10.verify()
    var softmax_2d = session.load(graph10)

    var inf_tensor = Tensor[DType.float32] (sequence_length)
    for i in range(inf_tensor.shape()[0]):
        inf_tensor[i] = -inf[DType.float32] ()

    print()
    ###########################################################################################################

    ####################### STORING MODEL WEIGHTS ########################

    print("Compiling Model", end = " ")

    Python.add_to_path(".")
    var mypython = Python.import_module("helper")

    var emb_table = numpy_to_tensor( mypython.layer_weights('token_embedding_table.weight'))
    var token_embedding = tokenEmbeddings(emb_table)

    var pos_emb_table = numpy_to_tensor( mypython.layer_weights('position_embedding_table.weight'))
    var pos_embedding = positionalEmbeddings(pos_emb_table)
    

    var Layer_Norm_1_List = List[layer_norm] ()
    var Head_List = List[head] ()
    var Linear_List = List[linear] ()
    var Layer_Norm_2_List = List[layer_norm] ()
    var Feed_Forward_list = List[feedForward] ()

    for w in range(num_heads):
        var gema = numpy_to_tensor(mypython.layer_weights('blocks.'+str(w)+'.ln1.weight'))
        var beta = numpy_to_tensor(mypython.layer_weights('blocks.'+str(w)+'.ln1.bias'))
        Layer_Norm_1_List.append(layer_norm(gema, beta))

        for i in range(num_heads):
            var weights_keys = (numpy_to_tensor( mypython.layer_weights('blocks.'+str(w)+'.sa.heads.'+str(i)+'.key.weight')))
            var weights_queries = (numpy_to_tensor( mypython.layer_weights('blocks.'+str(w)+'.sa.heads.'+str(i)+'.query.weight')))
            var weights_values = (numpy_to_tensor( mypython.layer_weights('blocks.'+str(w)+'.sa.heads.'+str(i)+'.value.weight')))
            Head_List.append(head(weights_keys,weights_queries, weights_values,inf_tensor))


        var weight_proj = numpy_to_tensor( mypython.layer_weights('blocks.'+str(w)+'.sa.proj.weight'))
        var bias_proj = numpy_to_tensor( mypython.layer_weights('blocks.'+str(w)+'.sa.proj.bias'))
        Linear_List.append(linear(weight_proj,bias_proj))

        var gema1 = numpy_to_tensor(mypython.layer_weights('blocks.'+str(w)+'.ln2.weight'))
        var beta1 = numpy_to_tensor(mypython.layer_weights('blocks.'+str(w)+'.ln2.bias'))
        Layer_Norm_2_List.append(layer_norm(gema1, beta1))

        var weight_ffwd = List[Tensor[DType.float32]]()
        var bias_ffwd = List[Tensor[DType.float32]]()
        weight_ffwd.append(numpy_to_tensor( mypython.layer_weights('blocks.'+str(w)+'.ffwd.net.0.weight')))
        weight_ffwd.append(numpy_to_tensor( mypython.layer_weights('blocks.'+str(w)+'.ffwd.net.2.weight')))
        bias_ffwd.append(numpy_to_tensor( mypython.layer_weights('blocks.'+str(w)+'.ffwd.net.0.bias')))
        bias_ffwd.append(numpy_to_tensor( mypython.layer_weights('blocks.'+str(w)+'.ffwd.net.2.bias')))

        Feed_Forward_list.append(feedForward(weight_ffwd, bias_ffwd))

        print(".", end = "")
    print()
    var gema = numpy_to_tensor(mypython.layer_weights('ln_f.weight'))
    var beta = numpy_to_tensor(mypython.layer_weights('ln_f.bias'))
    var Final_Layer_Norm = layer_norm(gema, beta)

    var lm_head_weight = numpy_to_tensor(mypython.layer_weights('lm_head.weight'))
    var lm_head_bias = numpy_to_tensor(mypython.layer_weights('lm_head.bias'))
    var Final_Linear = linear(lm_head_weight, lm_head_bias)

    ###############################################################

    ####################### RUNNING MODEL ########################
    
    var input = "Red riding hood was a"

    var token:PythonObject = []
    token = mypython.tokenizer(input)
    var nxt:PythonObject = []
    
    print("Running Inference")
    print("Input text: ", input)
    print("predicted text: ", input, end = "")
    var start = now()
    for v in range(total_len):

        var g = mypython.combine_lists(nxt,token)
        token = g

        var tkn_emb = token_embedding.forward(token)
        var pos_emb = pos_embedding.forward(tkn_emb)
        var input_to_block = pos_emb

        for w in range(num_heads):
            var nrm_lyer1 = Layer_Norm_1_List[w].forward(input_to_block, norm)
 
            var heads = List[Tensor[DType.float32]]()
            for i in range(num_heads):
                heads.append(Head_List[w*num_heads+i].forward(nrm_lyer1,multiplication,transpose,transpose_21,multiplication_3D,division,softmax))
            # for b in range(num_heads):
            #     heads.append(Tensor[DType.float32](1,5,42))

            # @parameter
            # fn increment_wrapper(index: Int) capturing -> None:
            #     print("before trying")
            #     try:
            #         print("trying")
            #         for i in range(num_heads):
            #             if index == i:
            #                 var xyz = Head_List[w*num_heads+i].forward(nrm_lyer1,multiplication,transpose,transpose_21,multiplication_3D,division,softmax)
            #                 print(i, "xyz.shape()", xyz.shape())
            #                 heads[i] = xyz
            #     except:
            #         print("An error occurred")

            # print("before parallelize")
            # parallelize[increment_wrapper] (num_heads)
            # print("after parallelize")
            # print("len(heads): ", len(heads))
            # print("heads[0].shape(): ",heads[0].shape())

            var heads_py = Python.dict()
            for i in range(num_heads):
                heads_py["input"+str(i)] = tensor_to_numpy(heads[i])
            var results = concat.execute(heads_py)
            var concatinated = results.get[DType.float32] ("output0")
 
            var proj = Linear_List[w].forward(concatinated,addition,multiplication,transpose)+ input_to_block
            var nrm_lyer2 = Layer_Norm_2_List[w].forward(proj, norm)
 
            var ffwd = Feed_Forward_list[w].forward(nrm_lyer2,multiplication,addition,transpose,relu) + proj
 
            input_to_block = ffwd

        var norm_layer = Final_Layer_Norm.forward(input_to_block,norm)
        var logits = Final_Linear.forward(norm_layer,addition,multiplication,transpose)

        var probs = logits_extraction(logits,softmax_2d)
        var next_token = mypython.output(tensor_to_numpy(probs))

        nxt = next_token[1]

        print(next_token[0], end = "")
    
    print("\n")

    var end = now()
    print("total generation time: ",(end - start)/1000000000)
    print("time/token: ",((end - start)/1000000000)/total_len)