import os
import vta
from tvm import rpc

def make_conv1d_fpga_scheduler(M,  N):

    env = vta.get_env()

    # We read the Pynq RPC host IP address and port number from the OS environment
    host = os.environ.get("VTA_PYNQ_RPC_HOST", "192.168.2.99")
    port = int(os.environ.get("VTA_PYNQ_RPC_PORT", "9091"))

    # We configure both the bitstream and the runtime system on the Pynq
    # to match the VTA configuration specified by the vta_config.json file.
    if env.TARGET == "pynq":

        # Make sure that TVM was compiled with RPC=1
        assert tvm.module.enabled("rpc")
        remote = rpc.connect(host, port)

        # Reconfigure the JIT runtime
        vta.reconfig_runtime(remote)

        # Program the FPGA with a pre-compiled VTA bitstream.
        # You can program the FPGA with your own custom bitstream
        # by passing the path to the bitstream file instead of None.
        vta.program_fpga(remote, bitstream=None)

    # In simulation mode, host the RPC server locally.
    elif env.TARGET == "sim":
        remote = rpc.LocalSession()

    # TODO: fill-in start
    # TODO: compute scheduler [s] and operator [C] for FPGA
    A = None
    B = None
    C = None
    s = None
    # TODO: fill-in end

    return {
        "scheduler": s,
        "input_A": A,
        "input_B": B,
        "output_C": C,
        "remote": remote,
        "env": env,
        'M': M,
        'N': N
    }


def make_conv1d_fpga_function(scheduler_info):
    """
    Create a function that takes two numpy arrays A, B of
    size (N) and (M) correspondingly, and output a numpy
    array that represents the output C of the function.

    :param scheduler_info:
    :return:
    """
    # extract scheduler and operator information
    s = scheduler_info["scheduler"]
    A = scheduler_info["input_A"]
    B = scheduler_info["input_B"]
    C = scheduler_info["output_C"]
    remote = scheduler_info["remote"]
    env = scheduler_info["env"]
    M = scheduler_info['M']
    N = scheduler_info['N']

    # create relay Constant for A, B, and C
    A = tvm.nd.array(A, device=env.target)
    B = tvm.nd.array(B, device=env.target)
    C = tvm.nd.array(C, device=env.target)

    def func(a_numpy, b_numpy):
        # create relay Constant for input A and B numpy arrays
        a = tvm.nd.array(a_numpy, device=env.target)
        b = tvm.nd.array(b_numpy, device=env.target)

        # create a relay function that takes A and B as inputs, and returns C as output
        a_const = tvm.relay.const(a.numpy(), dtype=a.dtype)
        b_const = tvm.relay.const(b.numpy(), dtype=b.dtype)
        c_const = tvm.relay.const(C.numpy(), dtype=C.dtype)
        conv = tvm.relay.nn.conv1d(a_const, b_const, kernel_size=M)
        func = tvm.relay.Function(conv.params, conv.body)
        with tvm.transform.PassContext(opt_level=3):
            func = tvm.relay.build(func, target=env.target, params={})

        # execute the relay function on the VTA FPGA
        ctx = remote.context(env.target, 0)
        a_nd = tvm.nd.array(a_numpy, ctx)
        b_nd = tvm.nd.array(b_numpy, ctx)
        c_nd = tvm.nd.empty((N-M+1,), C.dtype, ctx)
        func(a_nd, b_nd, c_nd)

        # return the output as a numpy array
        return c_nd.numpy()
        # TODO: fill-in end
    return func