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
from max.graph.checkpoint import save, TensorDict, load


alias num_attention_heads = 32
alias hidden_size = 2048
alias head_dim = hidden_size // num_attention_heads
alias min_len = 700


fn numpy_to_tensor(numpy_array: PythonObject) raises -> Tensor[DType.float32]:
    var tensor_shape = numpy_array.shape
    var tensor_rank = len(numpy_array.shape)
    var shape_list: List[Int]  = List[Int]()
    for i in range(tensor_rank):
        shape_list.append(tensor_shape[i].__int__())

    var tensor = Tensor[DType.float32] (shape_list)

    memcpy(tensor.unsafe_ptr(), numpy_array.__array_interface__['data'][0].unsafe_get_as_pointer[DType.float32](), 
           tensor.num_elements())
        
    return tensor


fn tensor_to_numpy(tensor: Tensor[DType.float32]) raises -> PythonObject:
    var np = Python.import_module("numpy")
    var tensor_shape = tensor.shape()
    var tensor_rank = tensor.rank()

    var python_list = Python.evaluate("list()")
    for i in range(tensor_rank):
        _ = python_list.append(tensor_shape[i])

    var numpy_array:PythonObject = np.zeros(python_list, dtype=np.float32)
    memcpy(numpy_array.__array_interface__['data'][0].unsafe_get_as_pointer[DType.float32](), tensor.unsafe_ptr(), 
           tensor.num_elements())
    return numpy_array^

struct LayerNorm(CollectionElement):
    var w: Tensor[DType.float32]
    var b: Tensor[DType.float32]

    fn __init__(inout self, gema: Tensor[DType.float32], beta: Tensor[DType.float32]):
        self.w = gema
        self.b = beta

    fn __copyinit__(inout self, existing: Self):
        self.w = existing.w
        self.b = existing.b
    
    fn __moveinit__(inout self, owned existing: Self):
        self.w = existing.w^
        self.b = existing.b^
    
    fn forward(self, inputs_embeds: Tensor[DType.float32], mean: Model) raises -> Tensor[DType.float32]:
        results = mean.execute("input0", inputs_embeds)
        m = results.get[DType.float32]("output0")
        variance = Tensor[DType.float32] (m.shape())
        var epsilon:Float32 = 1e-5

        for b in range(inputs_embeds.shape()[0]):
            for i in range(inputs_embeds.shape()[1]):
                var sum_squared_diff:Float32 = 0.0
                for j in range(inputs_embeds.shape()[2]):
                    diff = inputs_embeds[Index(b, i, j)] - m[Index(b, i, 0)]
                    sum_squared_diff += diff ** 2

                variance[Index(b, i, 0)] = sum_squared_diff / inputs_embeds.shape()[2]

        x = Tensor[DType.float32] (inputs_embeds.shape())
        y = Tensor[DType.float32] (inputs_embeds.shape())
        for i in range(x.shape()[0]):
            for j in range(x.shape()[1]):
                for k in range(x.shape()[2]):
                    x[Index(i,j,k)] = inputs_embeds[Index(i,j,k)] - m[j]

        y = variance + epsilon

        var exponent = Float32(0.5)
        for i in range(y.shape()[0]):
            for j in range(y.shape()[1]):
                for k in range(y.shape()[2]):
                    var base = Float32(y[Index(i,j,k)])
                    y[Index(i,j,k)] =  pow(base, exponent)


        z = Tensor[DType.float32] (inputs_embeds.shape())
        for i in range(z.shape()[0]):
            for j in range(z.shape()[1]):
                for k in range(z.shape()[2]):
                    z[Index(i,j,k)] = y[Index(0,j,0)]
        
        normalized_states = x/z

        out = Tensor[DType.float32] (normalized_states.shape())

        for i in range(out.shape()[0]):
            for j in range(out.shape()[1]):
                for k in range(out.shape()[2]):
                    out[Index(i,j,k)] = normalized_states[Index(i,j,k)] * self.w[k] + self.b[k]
        return out

fn chunk_qkv(qkv: Tensor[DType.float32], inout q: Tensor[DType.float32], inout start_qkv: Int):
    var start_q = 0
    alias load_sizes = 2048
    var q_num_elements = q.num_elements()
    for i in range(0, q_num_elements, load_sizes):
        q.store(start_q, qkv.load[width=load_sizes](start_qkv))
        start_q += load_sizes
        start_qkv += load_sizes*3

fn apply_rotary_pos_emb(q:Tensor[DType.float32], k:Tensor[DType.float32], cos:Tensor[DType.float32], 
                        sin:Tensor[DType.float32], position_ids:Tensor[DType.float32]) raises ->List[Tensor[DType.float32]]:
    new_cos = Tensor[DType.float32] (position_ids.shape()[0], position_ids.shape()[1], cos.shape()[1])
    new_sin = Tensor[DType.float32] (position_ids.shape()[0], position_ids.shape()[1], cos.shape()[1])
    for i in range(new_cos.shape()[0]):
        for j in range(new_cos.shape()[1]):
            pos_id = position_ids[i, j]
            for k in range(new_cos.shape()[2]):
                new_cos[Index(i,j,k)] = cos[Index(pos_id, k)]
                new_sin[Index(i,j,k)] = sin[Index(pos_id, k)]
    
    new_cos = new_cos.reshape((1,position_ids.shape()[0], position_ids.shape()[1], cos.shape()[1]))
    new_sin = new_sin.reshape((1,position_ids.shape()[0], position_ids.shape()[1], cos.shape()[1]))

    rotate_half_q_x1 = Tensor[DType.float32] (q.shape()[0], q.shape()[1], q.shape()[2], int(q.shape()[3]//2))
    rotate_half_q_x2 = Tensor[DType.float32] (q.shape()[0], q.shape()[1], q.shape()[2], int(q.shape()[3]//2))

    rotate_half_k_x1 = Tensor[DType.float32] (k.shape()[0], k.shape()[1], k.shape()[2], int(k.shape()[3]//2))
    rotate_half_k_x2 = Tensor[DType.float32] (k.shape()[0], k.shape()[1], k.shape()[2], int(k.shape()[3]//2))

    for b in range(rotate_half_q_x1.shape()[0]):
        for h in range(rotate_half_q_x1.shape()[1]):
            for i in range(rotate_half_q_x1.shape()[2]):
                for j in range(rotate_half_q_x1.shape()[3]):
                    # First half remains the same
                    rotate_half_q_x1[Index(b, h, i, j)] = q[Index(b, h, i, j)]
                    rotate_half_k_x1[Index(b, h, i, j)] = k[Index(b, h, i, j)]
                    # Second half is negated and swapped
                    rotate_half_q_x2[Index(b, h, i, j)] = -1 * q[Index(b, h, i, j + 16)]
                    rotate_half_k_x2[Index(b, h, i, j)] = -1 * k[Index(b, h, i, j + 16)]

    
    rotate_half_q_out = Tensor[DType.float32] (rotate_half_q_x1.shape()[0],rotate_half_q_x1.shape()[1],
                                             rotate_half_q_x1.shape()[2], rotate_half_q_x1.shape()[3] + rotate_half_q_x2.shape()[3])
    rotate_half_k_out = Tensor[DType.float32] (rotate_half_k_x1.shape()[0],rotate_half_k_x1.shape()[1],
                                             rotate_half_k_x1.shape()[2], rotate_half_k_x1.shape()[3] + rotate_half_k_x2.shape()[3])
    
    for b in range(rotate_half_q_x1.shape()[0]):
        for h in range(rotate_half_q_x1.shape()[1]):
            for i in range(rotate_half_q_x1.shape()[2]):
                for j in range(rotate_half_q_x1.shape()[3]):
                    # First 16 elements: negated -x2
                    rotate_half_q_out[Index(b, h, i, j)] = rotate_half_q_x2[Index(b, h, i, j)]
                    rotate_half_k_out[Index(b, h, i, j)] = rotate_half_k_x2[Index(b, h, i, j)]
                    # Next 16 elements: x1
                    rotate_half_q_out[Index(b, h, i, j + rotate_half_q_x1.shape()[3])] = rotate_half_q_x1[Index(b, h, i, j)]
                    rotate_half_k_out[Index(b, h, i, j + rotate_half_k_x1.shape()[3])] = rotate_half_k_x1[Index(b, h, i, j)]
    
    new_cos = Tensor[DType.float32] (q.shape())
    new_sin = Tensor[DType.float32] (q.shape())

    for i in range(new_cos.shape()[0]):
        for j in range(new_cos.shape()[1]):
            for k in range(new_cos.shape()[2]):
                for l in range(new_cos.shape()[3]):
                    new_cos[Index(i,j,k,l)] = cos[Index(k,l)]
                    new_sin[Index(i,j,k,l)] = sin[Index(k,l)]
    
    q_embed = q*new_cos + (rotate_half_q_out* new_sin)
    k_embed = (k * new_cos) + (rotate_half_k_out * new_sin)
    embs = List[Tensor[DType.float32]] ()
    embs.append(q_embed)
    embs.append(k_embed)
    return embs

fn scaled_dot_product_attention(query:Tensor[DType.float32], key:Tensor[DType.float32], value:Tensor[DType.float32],
                                transpose_21:Model, multiplication_4D:Model, addition_42:Model ,softmax:Model, 
                                multiplication_4D_2:Model, is_causal:Bool = False) raises -> Tensor[DType.float32]:
    
    var base = Float32(query.shape()[-1])
    var exponent = Float32(0.5)
    var scale_factor:Float32 = 1 / pow(base, exponent)

    # Transpose key and compute attention weights
    var results = transpose_21.execute("input0", key)
    var key_transpose = results.get[DType.float32]("output0")
    results = multiplication_4D.execute("input0", query, "input1", key_transpose)
    var atten_weights = results.get[DType.float32]("output0")
    var attention_weights = atten_weights * scale_factor

    # Initialize attn_bias based on query and key dimensions
    var L = query.shape()[-2]  # Sequence length of query
    var S = key.shape()[-2]    # Sequence length of key
    var attn_bias = Tensor[DType.float32]((L, S))

    # Apply causal masking if is_causal is True
    if is_causal:
        for i in range(L):
            for j in range(S):
                if j > i:
                    attn_bias.store(Index(i, j), -inf[DType.float32]())  # Mask out future positions
    # Apply the attention bias to the attention weights
    # attention_weights += attn_bias
    results = addition_42.execute("input0", attention_weights, "input1", attn_bias)
    attention_weights = results.get[DType.float32]("output0")

    # Softmax operation over the last dimension of attention weights
    var attn_weights = Tensor[DType.float32](attention_weights.shape())
    for i in range(attn_weights.shape()[0]):  # Batch
        for j in range(attn_weights.shape()[1]):  # Number of heads
            for k in range(attn_weights.shape()[2]):  # Sequence length (L)
                # Collect the entire last dimension (attention scores for each head for one sequence element)
                var new_tens = Tensor[DType.float32](attn_weights.shape()[3])  # Head dimension (S)

                # Copy each element from attention_weights into new_tens
                for l in range(attn_weights.shape()[3]):  # Head dimension
                    new_tens[l] = attention_weights[Index(i, j, k, l)]

                # Perform softmax on the extracted last dimension
                var results = softmax.execute("input0", new_tens)
                var xd = results.get[DType.float32]("output0")

                # Store the softmaxed values back into the attention weights
                for l in range(attn_weights.shape()[3]):
                    attn_weights[Index(i, j, k, l)] = xd[l]

    results = multiplication_4D_2.execute("input0", attn_weights, "input1", value)
    var output = results.get[DType.float32]("output0")

    return output


fn main() raises:
    print("Compiling Graphs", end = " ")
    var session = engine.InferenceSession()

    var graph1 = Graph(in_types=List[Type](TensorType(DType.float32, "a","m")))
    var transposed = ops.transpose(graph1[0],-1,-2)
    graph1.output(transposed)
    graph1.verify()
    var transpose = session.load(graph1)
    print(".", end = " ")

    var graph2 = Graph(in_types=List[Type](TensorType(DType.float32, "a","m", "n"), TensorType(DType.float32, "n")))
    var out2 = graph2[0] + graph2[1]
    graph2.output(out2)
    graph2.verify()
    var addition = session.load(graph2)

    var graph3 = Graph(in_types=List[Type](TensorType(DType.float32, "a","m", "n"), TensorType(DType.float32, "n","x")))
    var out3 = graph3[0] @ graph3[1]
    graph3.output(out3)
    graph3.verify()
    var multiplication = session.load(graph3)
    print(".", end = " ")

    var graph5 = Graph(in_types=List[Type](TensorType(DType.float32, 1, "b", "c")))
    var hollo = ops.mean(graph5[0])
    graph5.output(hollo)
    graph5.verify()
    var mean = session.load(graph5)

    var graph6 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "b", "c", "d")))
    transposed = ops.transpose(graph6[0],1,2)
    graph6.output(transposed)
    graph6.verify()
    var transpose_12 = session.load(graph6)
    print(".", end = " ")

    var graph7 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "b", "c", "d")))
    transposed = ops.transpose(graph7[0],-2,-1)
    graph7.output(transposed)
    graph7.verify()
    var transpose_21 = session.load(graph7)
    print(".", end = " ")

    var graph8 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "b", "c", "d"), 
                                           TensorType(DType.float32, "a", "b", "d", "c")))
    var out8 = graph8[0] @ graph8[1]
    graph8.output(out8)
    graph8.verify()
    var multiplication_4D = session.load(graph8)
    print(".", end = " ")

    var graph9 = Graph(in_types=List[Type](TensorType(DType.float32, "a","m", "n", "n"), TensorType(DType.float32, "n", "n")))
    var out9 = graph9[0] + graph9[1]
    graph9.output(out9)
    graph9.verify()
    var addition_42 = session.load(graph9)
    print(".", end = " ")

    var graph10 = Graph(in_types=List[Type](TensorType(DType.float32, "a")))
    var softmaxed = ops.softmax(graph10[0])
    graph10.output(softmaxed)
    graph10.verify()
    var softmax = session.load(graph10)
    print(".", end = " ")

    var graph11 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "m", "n", "n"), 
                                           TensorType(DType.float32, "a", "m", "n", "x")))
    var out11 = graph11[0] @ graph11[1]
    graph11.output(out11)
    graph11.verify()
    var multiplication_4D_2 = session.load(graph11)
    print(".", end = " ")

    ###################################################################################################################
    print()
    print("Compiling Model", end = " ")
    
    var mypython = Python.import_module("helper")
    var tensors = load("encoder_output.maxckpt")
    var weights = load("text_model.maxckpt")

    var encoder_output = tensors.get[DType.float32]("x")
    print(".", end = " ")

    ln_weight = numpy_to_tensor(mypython.layer_weights_text("transformer.h.0.ln.weight"))
    ln_bias = numpy_to_tensor(mypython.layer_weights_text("transformer.h.0.ln.bias"))
    var ln = LayerNorm(ln_weight, ln_bias)
    print(".", end = " ")

    qkv_weight = numpy_to_tensor(mypython.layer_weights_text("transformer.h.0.mixer.Wqkv.weight"))
    qkv_bias = numpy_to_tensor(mypython.layer_weights_text("transformer.h.0.mixer.Wqkv.bias"))
    print(".", end = " ")

    cos_cache = numpy_to_tensor(mypython.cos_sin("cos_cached"))
    sin_cache = numpy_to_tensor(mypython.cos_sin("sin_cached"))
    print(".", end = " ")

    outproj_weight = numpy_to_tensor(mypython.layer_weights_text("transformer.h.0.mixer.out_proj.weight"))
    outproj_bias = numpy_to_tensor(mypython.layer_weights_text("transformer.h.0.mixer.out_proj.bias"))
    print(".", end = " ")

    print()
    print("Running model")

    var question = "\n\nQuestion: Explain the image\n\nAnswer:"

    var inputs_embeds:Tensor[DType.float32] = numpy_to_tensor(mypython.text_emb(question, tensor_to_numpy(encoder_output)))
    
    var out = ln.forward(inputs_embeds, mean)

    bsz = out.shape()[0]
    q_len = out.shape()[1]

    var results = transpose.execute("input0", qkv_weight)
    var PW_T = results.get[DType.float32]("output0")
    results = multiplication.execute("input0", out, "input1", PW_T)
    var atten_out = results.get[DType.float32]("output0")
    results = addition.execute("input0", atten_out, "input1", qkv_bias)
    qkv = results.get[DType.float32]("output0")

    var t1:TensorShape = (qkv.shape()[0],qkv.shape()[1],int(qkv.shape()[2]/3))
    var query_states = Tensor[DType.float32] (t1)
    var key_state = Tensor[DType.float32] (t1)
    var value_state = Tensor[DType.float32] (t1)

    var start_qkv_q = 0
    var start_qkv_k = 2048
    var start_qkv_v = 4096
    chunk_qkv(qkv, query_states, start_qkv_q)
    chunk_qkv(qkv, key_state, start_qkv_k)
    chunk_qkv(qkv, value_state, start_qkv_v)

    reshaped_q = query_states.reshape((bsz, q_len, num_attention_heads, head_dim))
    results = transpose_12.execute("input0", reshaped_q)
    query_states = results.get[DType.float32]("output0")

    reshaped_k = key_state.reshape((bsz, q_len, num_attention_heads, head_dim))
    results = transpose_12.execute("input0", reshaped_k)
    key_state = results.get[DType.float32]("output0")

    reshaped_v = value_state.reshape((bsz, q_len, num_attention_heads, head_dim))
    results = transpose_12.execute("input0", reshaped_v)
    value_state = results.get[DType.float32]("output0")

    kv_seq_len = key_state.shape()[-2]


    cos = Tensor[DType.float32] (kv_seq_len, num_attention_heads)
    sin = Tensor[DType.float32] (kv_seq_len, num_attention_heads)

    for i in range(cos.shape()[0]):
        for j in range(cos.shape()[1]):
            cos[Index(i,j)] = cos_cache[Index(i,j)]
            sin[Index(i,j)] = sin_cache[Index(i,j)]

    query_rot = Tensor[DType.float32] (query_states.shape()[0], query_states.shape()[1], query_states.shape()[2], num_attention_heads)

    for i in range(query_rot.shape()[0]):
        for j in range(query_rot.shape()[1]):
            for k in range(query_rot.shape()[2]):
                for l in range(query_rot.shape()[3]):
                    query_rot[Index(i,j,k,l)] = query_states[Index(i,j,k,l)]


    query_pass = Tensor[DType.float32] (query_states.shape()[0], query_states.shape()[1], query_states.shape()[2], num_attention_heads)

    for i in range(query_pass.shape()[0]):
        for j in range(query_pass.shape()[1]):
            for k in range(query_pass.shape()[2]):
                for l in range(query_pass.shape()[3]):
                    query_pass[Index(i,j,k,l)] = query_states[Index(i,j,k,l+num_attention_heads)]


    key_rot = Tensor[DType.float32] (key_state.shape()[0], key_state.shape()[1], key_state.shape()[2], num_attention_heads)

    for i in range(key_rot.shape()[0]):
        for j in range(key_rot.shape()[1]):
            for k in range(key_rot.shape()[2]):
                for l in range(key_rot.shape()[3]):
                    key_rot[Index(i,j,k,l)] = key_state[Index(i,j,k,l)]


    key_pass = Tensor[DType.float32] (key_state.shape()[0], key_state.shape()[1], key_state.shape()[2], num_attention_heads)

    for i in range(key_pass.shape()[0]):
        for j in range(key_pass.shape()[1]):
            for k in range(key_pass.shape()[2]):
                for l in range(key_pass.shape()[3]):
                    key_pass[Index(i,j,k,l)] = key_state[Index(i,j,k,l+num_attention_heads)]


    position_ids = Tensor[DType.float32] (1,kv_seq_len)
    count = 0
    for i in range(position_ids.shape()[0]):
        for j in range(position_ids.shape()[1]):
            position_ids[Index(i,j)] = count
            count +=1
    
    embs = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
    query_rot = embs[0]
    key_rot =embs[1]

    # print("query_rot:\n", query_rot)
    # print("key_rot:\n", key_rot)
    # print("query_pass:\n", query_pass)
    # print("key_pass:\n", key_pass)

    new_query_states = Tensor[DType.float32] (query_pass.shape()[0], query_pass.shape()[1], query_pass.shape()[2], 
                                              query_pass.shape()[3]+query_rot.shape()[3])
    new_key_states = Tensor[DType.float32] (key_pass.shape()[0], key_pass.shape()[1], key_pass.shape()[2], 
                                              key_pass.shape()[3]+key_rot.shape()[3])

    for i in range(query_rot.shape()[0]):
        for j in range(query_rot.shape()[1]):
            for k in range(query_rot.shape()[2]):
                for l in range(query_rot.shape()[3]):
                    new_query_states[Index(i,j,k,l)] = query_rot[Index(i, j, k, l)]
                    new_query_states[Index(i,j,k,l+query_rot.shape()[3])] = query_pass[Index(i, j, k, l)]

                    new_key_states[Index(i,j,k,l)] = key_rot[Index(i, j, k, l)]
                    new_key_states[Index(i,j,k,l+key_rot.shape()[3])] = key_pass[Index(i, j, k, l)]

    attn_output = scaled_dot_product_attention(new_query_states, new_key_states, value_state, transpose_21, multiplication_4D, 
                                      addition_42, softmax, multiplication_4D_2, True)


    results = transpose_12.execute("input0", attn_output)
    var attn_output_t = results.get[DType.float32]("output0")
    attn_output_r = attn_output_t.reshape((bsz, q_len, hidden_size))

    results = transpose.execute("input0", outproj_weight)
    PW_T = results.get[DType.float32]("output0")
    results = multiplication.execute("input0", attn_output_r, "input1", PW_T)
    attn_out = results.get[DType.float32]("output0")
    results = addition.execute("input0", attn_out, "input1", outproj_bias)
    attention_output = results.get[DType.float32]("output0")

    print("attention_output:\n", attention_output)


