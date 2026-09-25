#!/usr/bin/env python3
"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

check_prepare.py over small models built to hit every branch of the
preparation, not just the ones one real driving model happens to take:

  comma layout      Concat(img, big_img) -> Cast, a passthrough feeding the
                    graph output, negative Gather indices (a shared scalar on
                    axes of different sizes, an int32 vector), LayerNorms in
                    the vision trunk and the policy (shared input, no bias,
                    one already fp32), MatMul+Add at rank 2, 3 and 4, and the
                    MatMuls that must be left alone (no Add, a small weight,
                    an Add that is not a bias); for the Neural Engine, an
                    Expand that becomes a Tile and one that must not
  sunnypilot layout tests/tiny_model.py: a Cast per image input, a
                    passthrough in the middle

    swift build -c release --package-path ios/JetlinkKit
    python3 ios/scripts/prepare_variants.py --swift ios/JetlinkKit/.build/release/jetlink-swift
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_prepare import KEY, first_difference, python_prepared  # noqa: E402

rng = np.random.default_rng(3)


def f16(shape, scale=0.1):
  return (rng.standard_normal(shape) * scale).astype(np.float16)


def comma_layout() -> onnx.ModelProto:
  T = TensorProto
  inits = {
    'vshape': np.array([1, 24, 32], np.int64),
    'vs': f16([32], 1), 'vb': f16([32]),
    'W1': f16([32, 64]), 'b1': f16([64]),
    's': f16([64], 1), 'b': f16([64]), 's32': (rng.standard_normal(64)).astype(np.float32),
    'idx': np.array(-1, np.int64),
    'idxv': np.array([0, -1, -5], np.int32),
    'W2': f16([64, 32]), 'b2': f16([32]),
    'W3': f16([64, 32]),
    'Ws': f16([64, 8]), 'bs': f16([8]),
    'W4': f16([64, 32]),
    'r4': np.array([1, 2, 15, 64], np.int64),
    'W5': f16([64, 16]), 'b5': f16([16]),
    'eshape': np.array([1, 1, 4, 1], np.int64),
    'uaxis': np.array([2], np.int64),
    'eshape_up': np.array([2, 1, 64], np.int64),
  }
  nodes = [
    helper.make_node('Concat', ['img', 'big_img'], ['cat'], axis=1, name='cat'),
    helper.make_node('Cast', ['cat'], ['catf'], to=T.FLOAT16, name='cast'),
    # the vision trunk: reads the images only
    helper.make_node('Reshape', ['catf', 'vshape'], ['v'], name='vreshape'),
    helper.make_node('LayerNormalization', ['v', 'vs', 'vb'], ['vln'], axis=-1, epsilon=1e-5, name='vln'),
    helper.make_node('MatMul', ['vln', 'W1'], ['vmm'], name='vmm'),
    helper.make_node('Add', ['vmm', 'b1'], ['vout'], name='vadd'),
    # the policy: reads features too
    helper.make_node('Concat', ['vout', 'feat'], ['p'], axis=1, name='pcat'),
    helper.make_node('LayerNormalization', ['p', 's', 'b'], ['ln1'], axis=-1, epsilon=1e-5, name='ln1'),
    helper.make_node('LayerNormalization', ['p', 's', 'b'], ['ln2'], axis=-1, epsilon=1e-5, name='ln2'),
    helper.make_node('LayerNormalization', ['p', 's'], ['ln3'], axis=-1, epsilon=1e-5, name='ln3'),
    helper.make_node('Cast', ['p'], ['p32'], to=T.FLOAT, name='p32'),
    helper.make_node('LayerNormalization', ['p32', 's32'], ['ln4'], axis=-1, epsilon=1e-5, name='ln4'),
    helper.make_node('Cast', ['ln4'], ['ln4h'], to=T.FLOAT16, name='ln4h'),
    helper.make_node('Add', ['ln1', 'ln2'], ['q0'], name='q0'),
    helper.make_node('Add', ['q0', 'ln3'], ['q1'], name='q1'),
    helper.make_node('Add', ['q1', 'ln4h'], ['q'], name='q'),
    helper.make_node('Gather', ['q', 'idx'], ['g1'], axis=1, name='g1'),
    helper.make_node('Gather', ['q', 'idx'], ['g2'], axis=2, name='g2'),
    helper.make_node('Gather', ['q', 'idxv'], ['g3'], axis=1, name='g3'),
    helper.make_node('MatMul', ['g1', 'W2'], ['m2mm'], name='m2mm'),
    helper.make_node('Add', ['m2mm', 'b2'], ['m2'], name='m2'),
    helper.make_node('MatMul', ['g3', 'W3'], ['m3'], name='m3'),
    helper.make_node('MatMul', ['g1', 'Ws'], ['smm'], name='smm'),
    helper.make_node('Add', ['smm', 'bs'], ['small'], name='small'),
    helper.make_node('MatMul', ['g1', 'W4'], ['nb'], name='nbmm'),
    helper.make_node('Add', ['nb', 'm2'], ['notbias'], name='notbias'),
    helper.make_node('Reshape', ['q', 'r4'], ['q4'], name='q4'),
    helper.make_node('MatMul', ['q4', 'W5'], ['m5mm'], name='m5mm'),
    helper.make_node('Add', ['m5mm', 'b5'], ['m5'], name='m5'),
    # an Expand that only repeats a size-1 axis becomes a Tile...
    helper.make_node('Unsqueeze', ['g3', 'uaxis'], ['g3u'], name='g3u'),
    helper.make_node('Expand', ['g3u', 'eshape'], ['ex'], name='ex'),
    # ...one that broadcasts a lower-rank input stays an Expand
    helper.make_node('Expand', ['g1', 'eshape_up'], ['ex2'], name='ex2'),
  ]
  flat = []
  for name in ('m2', 'm3', 'small', 'notbias', 'm5', 'g2', 'ex', 'ex2'):
    nodes.append(helper.make_node('Flatten', [name], [f'{name}_f'], axis=1, name=f'{name}_flat'))
    flat.append(f'{name}_f')
  nodes.append(helper.make_node('Concat', flat, ['pre'], axis=1, name='pre'))
  # a layout hint feeding the graph output: the output has to keep its name
  nodes.append(helper.make_node('Contiguous', ['pre'], ['outputs'], domain='org.tinygrad', name='hint'))
  n_out = 32 + 3 * 32 + 8 + 32 + 2 * 15 * 16 + 30 + 3 * 4 * 64 + 2 * 64
  graph = helper.make_graph(
    nodes, 'variants',
    [helper.make_tensor_value_info('img', T.UINT8, [1, 12, 4, 8]),
     helper.make_tensor_value_info('big_img', T.UINT8, [1, 12, 4, 8]),
     helper.make_tensor_value_info('feat', T.FLOAT16, [1, 6, 64])],
    [helper.make_tensor_value_info('outputs', T.FLOAT16, [1, n_out])],
    initializer=[numpy_helper.from_array(v, k) for k, v in inits.items()])
  model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 20), helper.make_opsetid('org.tinygrad', 1)])
  model.ir_version = 10
  # Record every intermediate shape, as the driving models' exporter does;
  # the unknown-domain op at the end is simply left without one.
  inferred = onnx.shape_inference.infer_shapes(model)
  model.graph.value_info.extend(v for v in inferred.graph.value_info if v.name != 'outputs')
  return model


def sunnypilot_layout(path: Path) -> None:
  sys.path.insert(0, str(ROOT / 'tests'))
  from tiny_model import write
  write(path)
  model = onnx.load(str(path))
  inferred = onnx.shape_inference.infer_shapes(model)
  model.graph.value_info.extend(inferred.graph.value_info)
  onnx.save(model, str(path))


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--swift', required=True, type=Path)
  args = p.parse_args()
  failed = 0
  with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    comma = tmp / 'comma.onnx'
    onnx.save(comma_layout(), str(comma))
    sunny = tmp / 'sunnypilot.onnx'
    sunnypilot_layout(sunny)
    for src in (comma, sunny):
      for device in ('coreml', 'ane', 'cpu'):
        py, sw = tmp / f'{src.stem}.{device}.py.onnx', tmp / f'{src.stem}.{device}.swift.onnx'
        python_prepared(src, device, py)
        run = subprocess.run([str(args.swift), 'prepare', str(src), str(sw), '--device', device, '--cache-key', KEY],
                             capture_output=True, text=True)
        if run.returncode != 0:
          print(f"{src.stem} {device}: swift failed: {run.stderr.strip()}")
          failed += 1
          continue
        a, b = onnx.load(str(py)), onnx.load(str(sw))
        onnx.checker.check_model(b)
        diff = first_difference(a, b)
        print(f"{src.stem:10} {device:6} {run.stdout.strip().removeprefix('prepared in ')}: "
              f"{'identical' if diff is None else 'DIFFERENT'}")
        if diff is not None:
          print(diff)
          failed += 1
  return 1 if failed else 0


if __name__ == '__main__':
  sys.exit(main())
