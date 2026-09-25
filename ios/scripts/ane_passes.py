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

# The passes live in the package, where the Mac server's `ane` device uses them
# too; this module is the iPhone build's reference that check_prepare.py runs.
from jetlink.onnx_patch import (  # noqa: F401
  HEAD_MAX_NODES,
  HEAD_OPS,
  LAYERNORM_PRESCALE,
  expand_to_tile,
  heads_in_fp32,
  prescale_layernorm,
  vision_heads,
  vision_nodes,
)
from jetlink.server.backends.ort import _prepared_model


def ane_prepared_model(onnx_path: Path):
  """The model the iPhone's Neural Engine build hands onnxruntime."""
  model = _prepared_model(Path(onnx_path), for_ane=False, for_coreml=True)
  expand_to_tile(model)
  vision = vision_nodes(model)
  policy = {n.name for n in model.graph.node} - vision
  prescale_layernorm(model, policy)
  heads_in_fp32(model, vision)
  return model
