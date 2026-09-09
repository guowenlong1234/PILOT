"""Frozen NWM compilation with native ATen arithmetic and bounded CUDA graphs."""
from collections import OrderedDict

import torch
from torch.utils._pytree import tree_map


class NativeCudaGraphBackend:
    """Compile Dynamo graphs to CUDA replay without Inductor arithmetic rewrites.

    Dynamic Dynamo dimensions avoid one recompile per candidate count. CUDA
    replay itself needs concrete shapes, so only a bounded set is retained.
    Outputs own their storage: a later sampler step must not overwrite them.
    """
    def __init__(self, model, max_graphs=4):
        if max_graphs < 1:
            raise ValueError('max_graphs must be positive')
        self.max_graphs = max_graphs
        self.name = 'native'
        self.graphs = OrderedDict()
        self.static_ptrs = {v.data_ptr() for v in list(model.parameters()) + list(model.buffers())}
        self.stats = dict(compilations=0, captures=0, replays=0, evictions=0)

    def __call__(self, gm, example_inputs):
        self.stats['compilations'] += 1
        graph_id = self.stats['compilations']

        def run(*args):
            if torch.is_grad_enabled():
                raise RuntimeError('NWM CUDA graph compilation is inference-only')
            tensors = [x for x in args if isinstance(x, torch.Tensor)]
            if not tensors or any(x.device.type != 'cuda' for x in tensors):
                raise ValueError('NWM CUDA graph inputs must be CUDA tensors')
            signature = tuple((tuple(x.shape), tuple(x.stride()), x.dtype, x.device)
                if isinstance(x, torch.Tensor) else (type(x), x) for x in args)
            key = (graph_id, signature, torch.is_autocast_enabled(),
                   torch.get_autocast_gpu_dtype(), torch.backends.cuda.matmul.allow_tf32,
                   torch.backends.cudnn.allow_tf32)
            if key not in self.graphs:
                if len(self.graphs) >= self.max_graphs:
                    self.graphs.popitem(last=False)
                    self.stats['evictions'] += 1
                with torch.cuda.device(tensors[0].device):
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        gm(*args)
                    torch.cuda.current_stream().wait_stream(stream)
                    static_args = [x if not isinstance(x, torch.Tensor) or x.data_ptr() in self.static_ptrs
                                   else x.clone() for x in args]
                    copied = [i for i,(a,b) in enumerate(zip(args,static_args))
                              if isinstance(a, torch.Tensor) and a is not b]
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        outputs = gm(*static_args)
                    self.graphs[key] = (graph, static_args, copied, outputs)
                    self.stats['captures'] += 1
            graph, static_args, copied, outputs = self.graphs[key]
            self.graphs.move_to_end(key)
            for i in copied:
                static_args[i].copy_(args[i])
            graph.replay()
            self.stats['replays'] += 1
            return tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, outputs)
        return run


class InductorBackend:
    """Track dynamic graph compilation; retain standard Inductor numerics."""
    name = 'inductor'

    def __init__(self):
        self.stats = dict(compilations=0)
        self.graphs = {}

    def __call__(self, gm, example_inputs):
        self.stats['compilations'] += 1
        return torch._inductor.compile(gm, example_inputs, options={'triton.cudagraphs': False})


def compile_frozen_world_model(model, max_graphs=4, backend_name='native'):
    if backend_name not in ('native', 'inductor'):
        raise ValueError('Unknown NWM compiler backend: '+str(backend_name))
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Only a frozen eval world model can be compiled')
    if next(model.parameters()).device.type != 'cuda':
        raise ValueError('World-model CUDA graph compilation requires CUDA')
    backend = (NativeCudaGraphBackend(model, max_graphs=max_graphs)
               if backend_name == 'native' else InductorBackend())
    compiled = torch.compile(model, backend=backend, dynamic=True, fullgraph=True)
    return compiled, backend
