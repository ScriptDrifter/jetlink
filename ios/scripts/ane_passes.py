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

On the M1 Pro: 99.5% of the estimated cost on the Neural Engine (was the
policy on the GPU), the parity gate passes, 30 ms back to back.

The preparation is _prepared_model(for_ane=False, for_coreml=True) with
these two after it: no fp32 LayerNorms, the Gemm and Gather rewrites as
before.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from jetlink.onnx_patch import _static_info, vision_nodes
from jetlink.server.backends.ort import _prepared_model

LAYERNORM_PRESCALE = 8


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


def ane_prepared_model(onnx_path: Path):
  """The model the iPhone's Neural Engine build hands onnxruntime."""
  model = _prepared_model(Path(onnx_path), for_ane=False, for_coreml=True)
  expand_to_tile(model)
  policy = {n.name for n in model.graph.node} - vision_nodes(model)
  prescale_layernorm(model, policy)
  return model
