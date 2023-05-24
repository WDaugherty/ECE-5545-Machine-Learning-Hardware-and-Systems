import os
import tvm
from tvm import te, autotvm

"""
Baseline(Unoptimized) Conv1D
"""
# def make_conv1d_cpu_scheduler(M, N):
#     A = te.placeholder((M,), name="A")
#     W = te.placeholder((N,), name="W")

#     k = te.reduce_axis((0, M + N - 1), "k")
#     B = te.compute(
#         (M + N - 1,),
#         lambda n: te.sum(tvm.tir.if_then_else(
#             tvm.tir.any(k < 0, k >= M, n - k < 0, n - k >= N),
#             tvm.tir.const(0.0, "float32"),
#             A[k] * W[n - k]), axis=k),
#         name="B",
#     )

#     s = te.create_schedule(B.op)

#     return s, A, W, B

"""
Optimized Conv1D utilizing:
NOTE: This is based on the readings provided and the use of chatgpt/github copilot plugins in VSCode
"""
def make_conv1d_cpu_scheduler(M, N):
    A = te.placeholder((M,), name="A")
    W = te.placeholder((N,), name="W")

    Pad_var = N - 1

    A_Pad = te.compute(
        (M + 2 * Pad_var,),
        lambda n: tvm.tir.if_then_else(
            tvm.tir.any(n < Pad_var, n >= (M + Pad_var)),
            tvm.tir.const(0.0, "float32"),
            A[n - Pad_var]),
        name="A_Pad",
    )

    k = te.reduce_axis((0, N), "k")
    B = te.compute(
        (M + N-1,),
        lambda n: te.sum(
            A_Pad[n + (N-1) - k] * W[k], axis=k
        ),
        name="B",
    )
    s = te.create_schedule(B.op)
    
    factor = 16
    x_out, x_in = s[B].split(B.op.axis[0], factor=factor)
    k_oout, k_in = s[B].split(k, factor=factor)
    s[B].reorder(x_out, k_oout, k_in, x_in)
    s[B].unroll(k_in)
    s[B].vectorize(x_in)

    #Caching 
    s[s.cache_read(A_Pad, "local", [B])].compute_at(s[B], k_oout)
    s[s.cache_read(W, "local", [B])].compute_at(s[B], k_oout)

    # # Fuse and parallelize the outer loops
    # fused_outer = s[B].fuse(x_out, k_oout)
    # s[B].parallel(fused_outer)
    
    return s, A, W, B

"""
AutoTVM Conv1D
"""
# @autotvm.template("make_conv1d_cpu_scheduler")
# def auto_make_conv1d_cpu_scheduler(M, N):
#     A = te.placeholder((M,), name="A")
#     W = te.placeholder((N,), name="W")

#     Pad_var = N - 1

#     A_Pad = te.compute(
#         (M + 2 * Pad_var,),
#         lambda n: tvm.tir.if_then_else(
#             tvm.tir.any(n < Pad_var, n >= (M + Pad_var)),
#             tvm.tir.const(0.0, "float32"),
#             A[n - Pad_var]),
#         name="A_Pad",
#     )

#     k = te.reduce_axis((0, N), "k")
#     B = te.compute(
#         (M + N-1,),
#         lambda n: te.sum(
#             A_Pad[n + (N-1) - k] * W[k], axis=k
#         ),
#         name="B",
#     )

#     s = te.create_schedule(B.op)
    
#     cfg = autotvm.get_config()
    
#     if cfg.is_fallback:
#         # Apply default schedule optimization
#     else:
#         # Apply the schedule optimization according to the best configuration
#         x_out, x_in = cfg["tile_x"].apply(s, B)
#         k_out, k_in = cfg["tile_k"].apply(s, B)
#         s[B].reorder(x_out, k_out, k_in, x_in)
#         s[B].unroll(k_in)
#         s[B].vectorize(x_in)

#         # Caching
#         s[s.cache_read(A_Pad, "local", [B])].compute_at(s[B], k_out)
#         s[s.cache_read(W, "local", [B])].compute_at(s[B], k_out)

#     return s, [A, W, B]



""""
Baseline(Unoptimized) GPU Conv1D
"""
# def make_unoptimized_conv1d_gpu_scheduler(M, N):
#     A = te.placeholder((M + 2 * (N - 1),), dtype="float32")
#     W = te.placeholder((N,), dtype="float32")

#     Apad = te.compute(
#         (M + 2 * (N - 1),),
#         lambda n: te.if_then_else(
#             te.all(n >= (N - 1), n < (M + N - 1)),
#             A[n - N + 1],
#             0.0,
#         ),
#         name="Apad",
#     )

#     k = te.reduce_axis((0, N), "k")
#     B = te.compute(
#         (M,),
#         lambda n: te.sum(Apad[n + (N - 1) - k] * W[k], axis=k),
#         name="B",
#     )

#     s = te.create_schedule(B.op)

#     return s, A, W, B





""""
Optimized GPU Scheduler utilizing:
Use specialized TVM operators: Instead of using the if_then_else operator in the reduction computation, you can use the tvm.tir.
Select operator which is specialized for conditional statements.
"""
def make_conv1d_gpu_scheduler(M, N):
    A = te.placeholder((M,), name="A")
    W = te.placeholder((N,), name="W")

    Apad = te.compute(
        (M + 2 * (N - 1),),
        lambda n: tvm.tir.if_then_else(
            tvm.tir.all(n >= (N - 1), n < (M + N - 1)),
            A[n - N + 1],
            tvm.tir.const(0.0, "float32"),
        ),
        name="Apad",
    )

    k = te.reduce_axis((0, N), "k")
    B = te.compute(
        (M + N - 1,),
        lambda n: te.sum(
            Apad[n + (N - 1) - k] * W[k], axis=k
        ),
        name="B",
    )

    s = te.create_schedule(B.op)

    # Parallelize over the outer axis of Apad and B
    s[Apad].parallel(Apad.op.axis[0])
    s[B].parallel(B.op.axis[0])

    # # Vectorize over the inner loop of B
    outer, inner = s[B].split(B.op.axis[0], factor=64)
    s[B].vectorize(inner)

    # Bind the outer axes of B to GPU threads
    #if B.op.axis[0] not in s[B].iter_var_attrs:
    # xo, xi = s[B].split(B.op.axis[0], nparts=te.thread_axis("blockIdx.x"))
    # s[B].bind(xi, te.thread_axis("threadIdx.x"))
    # s[B].bind(s[B].op.axis[1], te.thread_axis("vthread"))

    s[B].bind(outer, te.thread_axis("blockIdx.x"))
    s[B].bind(inner, te.thread_axis("threadIdx.x"))
    s[B].pragma(outer, "auto_unroll_max_step", 16)

    outerA, innerA = s[Apad].split(Apad.op.axis[0], factor=64)
    s[Apad].bind(outerA, te.thread_axis("blockIdx.x"))
    s[Apad].bind(innerA, te.thread_axis("threadIdx.x"))

    return s, A, W, B

""""
AutoTVM GPU Scheduler
"""
# @autotvm.template("make_conv1d_gpu_scheduler")
# def make_conv1d_gpu_scheduler(M, N):
#     A = te.placeholder((M,), name="A")
#     W = te.placeholder((N,), name="W")

#     Apad = te.compute(
#         (M + 2 * (N - 1),),
#         lambda n: tvm.tir.if_then_else(
#             tvm.tir.all(n >= (N - 1), n < (M + N - 1)),
#             A[n - N + 1],
#             tvm.tir.const(0.0, "float32"),
#         ),
#         name="Apad",
#     )

#     k = te.reduce_axis((0, N), "k")
#     B = te.compute(
#         (M + N - 1,),
#         lambda n: te.sum(
#             Apad[n + (N - 1) - k] * W[k], axis=k
#         ),
#         name="B",
#     )

#     s = te.create_schedule(B.op)

#     # Define the configuration space
#     cfg = autotvm.get_config()
#     cfg.define_split("split_B", B.op.axis[0], num_outputs=2, policy='factors')
#     cfg.define_split("split_Apad", Apad.op.axis[0], num_outputs=2, policy='factors')
#     cfg.define_knob("unroll", [0, 16])

#     # Split and bind axes for B
#     outer, inner = s[B].split(B.op.axis[0], cfg["split_B"].size[-1])
#     s[B].bind(outer, te.thread_axis("blockIdx.x"))
#     s[B].bind(inner, te.thread_axis("threadIdx.x"))
#     s[B].pragma(outer, "auto_unroll_max_step", cfg["unroll"].val)

#     # Split and bind axes for Apad
#     outerA, innerA = s[Apad].split(Apad.op.axis[0], cfg["split_Apad"].size[-1])
#     s[Apad].bind(outerA, te.thread_axis("blockIdx.x"))
#     s[Apad].bind(innerA, te.thread_axis("threadIdx.x"))

#     return s, [A, W, B]




"""
Baseline (Unoptimized) CPU GEMM Scheduler
"""
# def make_gemm_gpu_scheduler(M, K, N):
#     A = te.placeholder((M, K), name="A")
#     B = te.placeholder((K, N), name="B")

#     # TVM Matrix Multiplication using TE
#     k = te.reduce_axis((0, K), "k")
#     A = te.placeholder((M, K), name="A")
#     B = te.placeholder((K, N), name="B")
#     C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")
#     # Default schedule
#     s = te.create_schedule(C.op)

#     # the i-th block is indexed by blockIdx.x.
#     # the number of threads in each block is blockDim.x
#     # and the i-th thread within a block is indexed by threadIdx.x
#     # overall index of a thread can be calculated as
#     # 𝑖=blockIdx.x×blockDim.x+threadIdx.x
#     block_x = te.thread_axis("blockIdx.y")
#     block_y = te.thread_axis("blockIdx.x")

#     x, y = s[C].op.axis
#     (k,) = s[C].op.reduce_axis
#     s[C].bind(y, block_y)
#     s[C].bind(x, block_x)

#     return s, A, B, C


""""
Optimized CPU GEMM Scheduler utilizing:
Use specialized TVM operators: Instead of using the if_then_else operator in the reduction computation, you can use the tvm.tir.
Select operator which is specialized for conditional statements.
"""

def make_gemm_gpu_scheduler(M, K, N):
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")

    s = te.create_schedule(C.op)

    block_size = 32
    thread_x = te.thread_axis((0, block_size), "threadIdx.x")
    thread_y = te.thread_axis((0, block_size), "threadIdx.y")
    block_x = te.thread_axis("blockIdx.y")
    block_y = te.thread_axis("blockIdx.x")

    x, y = s[C].op.axis
    yo, xo, yi, xi = s[C].tile(x, y, block_size, block_size)
    s[C].parallel(yo)
    s[C].parallel(xo)
    s[C].bind(yo, block_y)
    s[C].bind(xo, block_x)

    # Split the workloads
    k, = s[C].op.reduce_axis
    ko, ki = s[C].split(k, factor=block_size)
    xio, xi = s[C].split(xi, nparts=block_size)
    yio, yi = s[C].split(yi, nparts=block_size)

    s[C].reorder(ko, ki, xi, yi, xio, yio)
    s[C].parallel(xio)
    s[C].parallel(yio)
    s[C].bind(xio, thread_x)
    s[C].bind(yio, thread_y)

    # Use local memory for A and B tiles
    AA = s.cache_read(A, "local", [C])
    BB = s.cache_read(B, "local", [C])
    s[AA].compute_at(s[C], ko)
    s[BB].compute_at(s[C], ko)

    return s, A, B, C


@autotvm.template("gemm_autotvm_template")
def gemm_autotvm_template(M, K, N, dtype):
    A = te.placeholder((M, K), dtype=dtype, name="A")
    B = te.placeholder((K, N), dtype=dtype, name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")

    s = te.create_schedule(C.op)
    x, y = s[C].op.axis
    k, = s[C].op.reduce_axis

    cfg = autotvm.get_config()
    cfg.define_split("tile_y", y, num_outputs=2)
    cfg.define_split("tile_x", x, num_outputs=2)
    cfg.define_split("tile_k", k, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])

    yo, yi = cfg["tile_y"].apply(s, C, y)
    xo, xi = cfg["tile_x"].apply(s, C, x)
    ko, ki = cfg["tile_k"].apply(s, C, k)

    s[C].reorder(yo, xo, yi, xi, ko, ki)

    # Bind the threads
    s[C].bind(yo, te.thread_axis("blockIdx.y"))
    s[C].bind(xo, te.thread_axis("blockIdx.x"))
    s[C].bind(yi, te.thread_axis("threadIdx.y"))
    s[C].bind(xi, te.thread_axis("threadIdx.x"))

    return s, [A, B, C]


"""
Baseline Version(Unoptimized) make_dwsp_conv2d_gpu_scheduler
"""
# def make_dwsp_conv2d_gpu_scheduler(B, C, H, W, K):
#     assert K % 2 == 1
#     inp = te.placeholder((B, C, H, W), name="A")
#     ker = te.placeholder((C, 1, K, K), name="W")

#     # TODO: fill-in start
    
    
#     # TODO: fill-in end

#     return s, inp, ker, out


"""
Optimized make_dwsp_conv2d_gpu_scheduler version 2:
"""


def make_dwsp_conv2d_gpu_scheduler(B, C, H, W, K):
    #Made with ChatGPT
    assert K % 2 == 1
    inp = te.placeholder((B, C, H, W), name="A")
    ker = te.placeholder((C, 1, K, K), name="W")

    rkh = te.reduce_axis((0, K), name='rkh')
    rkw = te.reduce_axis((0, K), name='rkw')
    pkh = (K-1)//2
    pkw = pkh

    #TODO: fill-in start

    inp_pad = te.compute((B, C, H + K - 1, W + K - 1),
        lambda *i: tvm.tir.if_then_else(te.any(i[-2] < pkh, i[-2] >= H + pkh, i[-1] < pkw, i[-1]>= W + pkw),
         tvm.tir.const(0.0,"float32"),
        inp[i[:-2] +(i[-2] - pkh, i[-1] - pkw)]),
        name="inp_pad",
    )
    
   
    out = te.compute(
        (B, C, H, W),
        lambda b, c, i, j: te.sum(
            (inp_pad[b, c, i+rkh, j+rkw] * ker[c, 0, rkh, rkw]),
            axis=[rkh, rkw]), name='Y')
    
    s = te.create_schedule(out.op)
    C_Out, C_In = s[out].split(out.op.axis[2], factor=4)
    s[out].parallel(C_Out)
    s[out].parallel(C_In)
    s[out].bind(C_Out, te.thread_axis("blockIdx.x"))
    s[out].bind(C_In, te.thread_axis("threadIdx.x"))
    C_Out, C_In = s[out].split(out.op.axis[3], factor=4)
    s[out].parallel(C_Out)
    s[out].parallel(C_In)
    s[out].bind(C_Out, te.thread_axis("blockIdx.z"))
    s[out].bind(C_In, te.thread_axis("threadIdx.z"))
    outer_inp_pad, inner_inp_pad = s[inp_pad].split(inp_pad.op.axis[2], factor=4)
    s[out].parallel(C_Out)
    s[out].parallel(C_In)
    s[inp_pad].bind(outer_inp_pad, te.thread_axis("blockIdx.y"))
    s[inp_pad].bind(inner_inp_pad, te.thread_axis("threadIdx.y"))
    return s, inp, ker, out