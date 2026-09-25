"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The iPhone's Neural Engine preparation, in Python, as the reference the
Swift one (OnnxPrepare.swift) is checked against by check_prepare.py.

The Mac server's Neural Engine build runs the policy's LayerNorms in fp32,
which the Neural Engine cannot do, and leaves two Expand ops CoreML will not
take. On an M1 Pro, MLComputePlan put the whole policy on the GPU as a
second CoreML model. That costs little on a Mac's GPU and a great deal on a
phone's. These passes keep the model on the Neural Engine in one piece:

  expand_to_tile        Expand(x, [1, 1, 32, 1]) on a [1, 9, 1, 512] x is
                        Tile(x, [1, 1, 32, 1]): the same values, an op the
                        CoreML provider accepts, so the graph stays one
                        partition.
  prescale_layernorm    LayerNorm(x / 8) is LayerNorm(x) up to epsilon, and
                        keeps the fp16 arithmetic in range: the policy's
                        inputs reach 1189, whose square overflows fp16 while
                        the row sums of (x / 8)^2 stay under 34,000. Epsilon
                        is left as it is. Measured over 32 recurrent frames
                        of the 766 MB model, 2026-09-24.
  heads_in_fp32         the small MLPs between the end of the vision trunk
                        and the output (the summarizer and hydra heads that
                        make road_transform, pose, lane_lines_prob and the
                        rest) in fp32, which puts them on the GPU or CPU.
                        In fp16 on an iPhone 17 Pro's Neural Engine,
                        road_transform failed the parity gate. 2026-09-25.

On the M1 Pro: 99.5% of the estimated cost on the Neural Engine (was the
policy on the GPU), the parity gate passes, 30 ms back to back.

The preparation is _prepared_model(for_ane=False, for_coreml=True) with
these three after it: no fp32 LayerNorms, the Gemm and Gather rewrites as
before.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from jetlink.onnx_patch import _static_info, vision_nodes
from jetlink.server.backends.ort import _prepared_model

LAYERNORM_PRESCALE = 8
# The ops heads_in_fp32 moves into fp32: the small MLPs and linear layers that
# end the vision trunk. A reduction, a reshape or anything else ends the region.
HEAD_OPS = {'Gemm', 'MatMul', 'LayerNormalization', 'Gelu', 'Add', 'Sub', 'Mul', 'Div', 'Relu', 'Sigmoid', 'Tanh'}
HEAD_MAX_NODES = 64


def expand_to_tile(model) -> int:
  """Rewrite an Expand with a constant shape of the input's rank, which only
  repeats size-1 axes, as the equivalent Tile. In place; returns how many."""
  g = model.graph
  init = {t.name: t for t in g.initializer}
  dims, _ = _static_info(model, {n.input[0] for n in g.node if n.op_type == 'Expand'})
  done = 0
  for n in g.node:
    if n.op_type != 'Expand' or len(n.input) != 2 or n.input[1] not in init:
      continue
    shape = dims.get(n.input[0])
    target = [int(v) for v in numpy_helper.to_array(init[n.input[1]]).reshape(-1)]
    if shape is None or len(shape) != len(target) or any(d <= 0 for d in shape):
      continue
    repeats = []
    for have, want in zip(shape, target, strict=True):
      if want in (1, have):
        repeats.append(1)
      elif have == 1:
        repeats.append(want)
      else:
        repeats = None
        break
    if repeats is None:
      continue
    name = f"{n.output[0]}__repeats"
    g.initializer.append(numpy_helper.from_array(np.array(repeats, np.int64), name))
    n.op_type = 'Tile'
    n.input[1] = name
    done += 1
  return done


def prescale_layernorm(model, only: set[str], k: int = LAYERNORM_PRESCALE) -> int:
  """Feed the fp16 LayerNorms named in `only` their input times 1/k, one Mul
  per distinct input. In place; returns how many."""
  g = model.graph
  const = f"__layernorm_prescale_{k}"
  _, dtypes = _static_info(model, {n.input[0] for n in g.node if n.op_type == 'LayerNormalization'})
  new, scaled, done = [], {}, 0
  for n in g.node:
    if n.op_type == 'LayerNormalization' and n.name in only and dtypes.get(n.input[0]) == TensorProto.FLOAT16:
      x = n.input[0]
      if x not in scaled:
        scaled[x] = f"{x}__scaled"
        new.append(helper.make_node('Mul', [x, const], [scaled[x]], name=f"{n.name}__prescale"))
      n.input[0] = scaled[x]
      done += 1
    new.append(n)
  if done:
    g.initializer.append(numpy_helper.from_array(np.array(1.0 / k, np.float16), const))
    del g.node[:]
    g.node.extend(new)
  return done


def vision_heads(model, vision: set[str]) -> list[int]:
  """The indices, in graph order, of the heads that end the vision trunk:
  the largest set of vision nodes with ops in HEAD_OPS whose outputs are all
  read, and read only, by each other or by a Concat that makes a graph
  output. Empty when there is no such Concat or the set would exceed
  HEAD_MAX_NODES."""
  g = model.graph
  outputs = {o.name for o in g.output}
  ends = {i for i, n in enumerate(g.node) if n.op_type == 'Concat' and any(o in outputs for o in n.output)}
  if not ends:
    return []
  readers: dict[str, set[int]] = {}
  for i, n in enumerate(g.node):
    for x in n.input:
      readers.setdefault(x, set()).add(i)
  region: set[int] = set()
  grew = True
  while grew:
    grew = False
    for i in reversed(range(len(g.node))):
      n = g.node[i]
      if i in region or n.name not in vision or n.op_type not in HEAD_OPS or any(o in outputs for o in n.output):
        continue
      read = [readers.get(o, set()) for o in n.output]
      if all(r and r <= region | ends for r in read):
        region.add(i)
        grew = True
  return sorted(region) if len(region) <= HEAD_MAX_NODES else []


def heads_in_fp32(model, vision: set[str]) -> int:
  """Run vision_heads in fp32: cast what they read from the trunk up, their
  fp16 weights to fp32 copies, and what they hand the output Concat back
  down. The Neural Engine cannot run fp32, so CoreML places them on the GPU
  or CPU. In place; returns how many nodes moved.

  The heads are small (24 nodes and 4 MB of weights in the 766 MB chestnut
  model), but in fp16 on the Neural Engine their LayerNorm, Gelu and 1024-wide
  Gemms lose enough that road_transform fails the parity gate on an iPhone
  17 Pro (worst column 0.9989). Computed exactly from the Neural Engine's own
  trunk output, every column is 0.9996 or better. Measured 2026-09-25."""
  g = model.graph
  index = vision_heads(model, vision)
  if not index:
    return 0
  heads = set(index)
  init = {t.name: t for t in g.initializer}
  produced = {o for i in index for o in g.node[i].output}
  entries = {x for i in index for x in g.node[i].input if x and x not in init and x not in produced}
  _, dtypes = _static_info(model, entries)
  if any(dtypes.get(x) != TensorProto.FLOAT16 for x in entries):
    return 0
  weights = {x for i in index for x in g.node[i].input if x in init}
  if any(init[w].data_type not in (TensorProto.FLOAT16, TensorProto.FLOAT) for w in weights):
    return 0
  read_outside = {x for j, n in enumerate(g.node) if j not in heads for x in n.input}
  exits = produced & read_outside
  wide = {}
  for w in sorted(weights):
    if init[w].data_type == TensorProto.FLOAT16:
      wide[w] = f"{w}__fp32"
      g.initializer.append(numpy_helper.from_array(numpy_helper.to_array(init[w]).astype(np.float32), wide[w]))
  new, cast = [], set()
  for j, n in enumerate(g.node):
    if j not in heads:
      new.append(n)
      continue
    for i, x in enumerate(n.input):
      if x in entries:
        if x not in cast:
          cast.add(x)
          new.append(helper.make_node('Cast', [x], [f"{x}__fp32"], name=f"{x}__cast_fp32", to=TensorProto.FLOAT))
        n.input[i] = f"{x}__fp32"
      elif x in exits:
        n.input[i] = f"{x}__fp32"
      elif x in wide:
        n.input[i] = wide[x]
    back = [o for o in n.output if o in exits]
    for i, o in enumerate(n.output):
      if o in exits:
        n.output[i] = f"{o}__fp32"
    new.append(n)
    for o in back:
      new.append(helper.make_node('Cast', [f"{o}__fp32"], [o], name=f"{o}__cast_fp16", to=TensorProto.FLOAT16))
  del g.node[:]
  g.node.extend(new)
  # The fp16 originals, unless something outside the heads reads them too.
  keep = [t for t in g.initializer if t.name not in wide or t.name in read_outside]
  del g.initializer[:]
  g.initializer.extend(keep)
  # What the heads compute inside is fp32 now; the value infos said fp16.
  keep_vi = [v for v in g.value_info if v.name not in produced - exits]
  del g.value_info[:]
  g.value_info.extend(keep_vi)
  return len(index)


def ane_prepared_model(onnx_path: Path):
  """The model the iPhone's Neural Engine build hands onnxruntime."""
  model = _prepared_model(Path(onnx_path), for_ane=False, for_coreml=True)
  expand_to_tile(model)
  vision = vision_nodes(model)
  policy = {n.name for n in model.graph.node} - vision
  prescale_layernorm(model, policy)
  heads_in_fp32(model, vision)
  return model
