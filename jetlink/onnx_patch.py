"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Make an openpilot driving model acceptable to TensorRT's ONNX parser.

TensorRT 10.3 rejects UINT8 graph inputs ("Found unsupported input type of
UINT8"), so the image inputs are declared FP16 and the head Cast deleted.
Feeding 0..255 as fp16 is exact and free; the weights are FP16 already.

Runs on the Jetson at build time. The shipped model is never modified in place.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

IMG_INPUTS = ('img', 'big_img')

# tinygrad's exporter leaves layout hints in its own domain, and TensorRT rejects
# any op in a domain it does not know. Contiguous is about tinygrad's buffers,
# not the arithmetic, so bypassing it is safe. Whether a model has one varies
# with how it was exported.
TINYGRAD_DOMAIN = 'org.tinygrad'
PASSTHROUGH_OPS = ('Contiguous',)


def needs_patch(model: onnx.ModelProto) -> bool:
  return any(vi.name in IMG_INPUTS and vi.type.tensor_type.elem_type == TensorProto.UINT8
             for vi in model.graph.input)


def strip_tinygrad_ops(model: onnx.ModelProto) -> int:
  """Bypass tinygrad's layout-hint nodes. In place, returns how many went.

  Refuses anything it has not been told is a passthrough rather than guessing:
  silently dropping an op that did something would change what the car sees.
  """
  g = model.graph
  graph_outputs = {o.name for o in g.output}
  removed = 0

  for node in [n for n in g.node if n.domain == TINYGRAD_DOMAIN]:
    if node.op_type not in PASSTHROUGH_OPS:
      raise ValueError(f"unknown {TINYGRAD_DOMAIN} op {node.op_type!r}; it may not "
                       "be a no-op, so dropping it is not safe")
    if len(node.input) != 1 or len(node.output) != 1 or node.attribute:
      raise ValueError(f"{node.op_type} is not a plain one-in one-out passthrough")

    source, produced = node.input[0], node.output[0]
    if produced in graph_outputs:
      # The output keeps its name: output_slices and the parity tools address
      # it. So the producer takes the name over; renaming both ends leaves the
      # output with no producer at all.
      for n in g.node:
        for i, name in enumerate(n.output):
          if name == source:
            n.output[i] = produced
      for n in g.node:
        for i, name in enumerate(n.input):
          if name == source:
            n.input[i] = produced
    else:
      for n in g.node:
        for i, name in enumerate(n.input):
          if name == produced:
            n.input[i] = source
    g.node.remove(node)
    removed += 1

  if removed:
    for i, opset in enumerate(model.opset_import):
      if opset.domain == TINYGRAD_DOMAIN:
        del model.opset_import[i]
        break
    live = {n for node in g.node for n in list(node.input) + list(node.output)}
    for i in reversed(range(len(g.value_info))):
      if g.value_info[i].name not in live:
        del g.value_info[i]
  return removed


def normalize_gather_indices(model: onnx.ModelProto) -> int:
  """Rewrite Gather nodes whose constant index is negative to the positive
  equivalent. In place, returns how many were rewritten.

  Apple's Neural Engine returns garbage for Gather with a scalar index of -1
  on a large axis: measured on the driving model's `select_2` (add_53[:, -1]
  over 288 tokens), correlation 0.03 against the CPU, exact with the index
  written as 287. ONNX defines a negative index as counting from the end, so
  the rewrite changes nothing about the graph's meaning; it needs the data's
  static size along the axis, which the exporter's value_info carries, and
  leaves any node whose size it cannot see alone rather than guess.

  Each rewritten node gets its own index initializer: the exporter shares one
  constant between Gathers on axes of different sizes.
  """
  g = model.graph
  init = {t.name: t for t in g.initializer}
  gathers = [n for n in g.node if n.op_type == 'Gather' and len(n.input) > 1 and n.input[1] in init]
  dims, _ = _static_info(model, {n.input[0] for n in gathers})
  rewritten = 0
  for node in gathers:
    index = numpy_helper.to_array(init[node.input[1]])
    if not np.issubdtype(index.dtype, np.integer) or index.size == 0 or index.min() >= 0:
      continue
    axis = next((helper.get_attribute_value(a) for a in node.attribute if a.name == 'axis'), 0)
    shape = dims.get(node.input[0])
    if shape is None or axis >= len(shape) or shape[axis] <= 0:
      continue
    size = shape[axis]
    fixed = np.where(index < 0, index + size, index).astype(index.dtype)
    if (fixed < 0).any() or (fixed >= size).any():
      raise ValueError(f"{node.name}: Gather index {index.tolist()} out of range for axis {axis} of size {size}")
    name = f"{node.output[0]}__index"
    g.initializer.append(numpy_helper.from_array(fixed, name))
    node.input[1] = name
    rewritten += 1
  return rewritten


def vision_nodes(model: onnx.ModelProto) -> set[str]:
  """Names of the nodes that depend on the image inputs alone: the vision
  trunk and the heads that hang off it. Found by dataflow: a node is vision
  when every tensor it reads is an image input, an initializer, or another
  vision node's output. Nodes reading only initializers count as neither."""
  g = model.graph
  init = {t.name for t in g.initializer}
  image_inputs = {vi.name for vi in g.input if vi.name in IMG_INPUTS}
  if not image_inputs:
    raise ValueError(f"model has none of {IMG_INPUTS} as graph inputs")
  vision_tensors = set(image_inputs)
  names: set[str] = set()
  for node in g.node:
    data = [x for x in node.input if x and x not in init]
    if data and all(x in vision_tensors for x in data):
      names.add(node.name)
      vision_tensors.update(node.output)
  return names


def layernorm_in_fp32(model: onnx.ModelProto, only: set[str] | None = None) -> int:
  """Run LayerNormalization in fp32: cast its input up, its scale and bias to
  fp32, its output back down. Every node, or only the names in `only`. In
  place, returns how many.

  Apple's Neural Engine computes LayerNormalization in fp16, and on this
  model's residual stream, values in the hundreds, that loses enough that
  the small heads fail the parity gate (policy output error 0.0021 against
  0.0003 on the GPU). In fp32 the Neural Engine cannot run the node, so
  CoreML places it elsewhere, and the policy came out as precise as on the
  GPU and faster (8.9 ms against 14.1). Not every node, though: each fp32
  node is a compute-unit switch, and with the vision trunk's 41 included the
  frame went from 28 ms to 70. The trunk is precise enough in fp16, so the
  backend passes the policy's nodes only. Measured 2026-09-08,
  docs/platforms.md.

  Nothing else changes: a LayerNormalization that was fp32 already is left
  alone, and an input that is not fp16 is not cast.
  """
  g = model.graph
  init = {t.name: t for t in g.initializer}
  _, dtypes = _static_info(model, {n.input[0] for n in g.node if n.op_type == 'LayerNormalization'})
  done = 0
  new_nodes = []
  cast_up: dict[str, str] = {}   # one up-cast per input: the head MLPs share theirs
  for node in g.node:
    if (node.op_type != 'LayerNormalization' or dtypes.get(node.input[0]) != TensorProto.FLOAT16
        or (only is not None and node.name not in only)):
      new_nodes.append(node)
      continue
    x = node.input[0]
    if x not in cast_up:
      cast_up[x] = f"{x}__fp32"
      new_nodes.append(helper.make_node('Cast', [x], [cast_up[x]], to=TensorProto.FLOAT, name=f"{node.name}__cast_in"))
    node.input[0] = cast_up[x]
    for i in range(1, len(node.input)):
      name = node.input[i]
      if name in init and init[name].data_type == TensorProto.FLOAT16:
        wide = numpy_helper.from_array(numpy_helper.to_array(init[name]).astype(np.float32), f"{name}__fp32")
        if wide.name not in init:
          g.initializer.append(wide)
          init[wide.name] = wide
        node.input[i] = wide.name
    out = node.output[0]
    node.output[0] = f"{out}__fp32"
    new_nodes.append(node)
    new_nodes.append(helper.make_node('Cast', [node.output[0]], [out], to=TensorProto.FLOAT16, name=f"{node.name}__cast_out"))
    done += 1
  if done:
    del g.node[:]
    g.node.extend(new_nodes)
    # the file's own shape record for the retyped output is stale; drop it
    # rather than leave a lie the checker would trip on
    stale = {n.input[0] for n in g.node if n.op_type == 'Cast' and n.input[0].endswith('__fp32')}
    for i in reversed(range(len(g.value_info))):
      if g.value_info[i].name in stale:
        del g.value_info[i]
  return done


# -- the Neural Engine passes: the iPhone build (ios/) and the Mac's `ane` device --

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
      # layernorm_in_fp32 names its copies the same way; one is enough
      if wide[w] not in init:
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


def _static_info(model: onnx.ModelProto, wanted: set[str]) -> tuple[dict[str, tuple[int, ...]], dict[str, int]]:
  """Static shapes and element types of the graph's tensors, from what the
  file carries; the shape inferrer only when one of `wanted` is missing.
  Fails open: a tensor still unknown afterwards is simply absent, and the
  caller decides what it cannot do without it."""
  g = model.graph
  dims: dict[str, tuple[int, ...]] = {}
  dtypes: dict[str, int] = {}

  def take(values):
    for vi in values:
      tt = vi.type.tensor_type
      if tt.elem_type:
        dtypes[vi.name] = tt.elem_type
      if tt.HasField('shape'):
        dims[vi.name] = tuple(d.dim_value if d.HasField('dim_value') else -1 for d in tt.shape.dim)
  take(g.input)
  take(g.value_info)
  take(g.output)
  if wanted - {n for n in dims if n in dtypes}:
    try:
      take(onnx.shape_inference.infer_shapes(model).graph.value_info)
    except Exception:
      pass
  return dims, dtypes


def patch_uint8_inputs(model: onnx.ModelProto) -> onnx.ModelProto:
  """Retype the uint8 image inputs to fp16 and drop the head Cast. In place."""
  g = model.graph

  img_inputs = [vi for vi in g.input if vi.name in IMG_INPUTS]
  if not img_inputs:
    raise ValueError(f"model has none of {IMG_INPUTS} as graph inputs")

  # Two graph shapes in the wild, same fix either way: the casts are all that
  # stands between a uint8 input and the fp16 the model wants.
  #
  #   comma      Concat(img, big_img) -> cat -> Cast(fp16)
  #   sunnypilot Cast(img), Cast(big_img) -> Concat -> cat
  casts = _head_casts(g)
  if not casts:
    raise ValueError("could not find the head Cast on the image inputs; "
                      "neither a Cast per input nor one after their Concat")

  for cast in casts:
    to = next(onnx.helper.get_attribute_value(a) for a in cast.attribute if a.name == 'to')
    if to != TensorProto.FLOAT16:
      raise ValueError(f"head Cast targets {to}, expected FLOAT16 ({TensorProto.FLOAT16})")

  for vi in img_inputs:
    if vi.type.tensor_type.elem_type != TensorProto.UINT8:
      raise ValueError(f"{vi.name} is not uint8; model already patched?")
    vi.type.tensor_type.elem_type = TensorProto.FLOAT16

  retyped = set()
  for cast in casts:
    # Everything downstream reads the cast's source directly now.
    source, produced = cast.input[0], cast.output[0]
    for n in g.node:
      for i, name in enumerate(n.input):
        if name == produced:
          n.input[i] = source
    g.node.remove(cast)
    retyped.add(source)

  # TensorRT tolerates a stale uint8 value_info; onnxruntime rejects the model.
  for vi in g.value_info:
    if vi.name in retyped:
      vi.type.tensor_type.elem_type = TensorProto.FLOAT16

  return model


# A weight smaller than this stays as it is. onnxruntime writes an initializer
# of ten elements or more to the weight file, but rewriting a graph is only
# worth it where the MIL text would be large, and a bias-sized constant costs
# a few hundred bytes either way.
BLOB_MIN_ELEMENTS = 1024


def gemm_with_transposed_weight(model: onnx.ModelProto) -> int:
  """Rewrite `MatMul(x, W)` followed by `Add(b)` as `Gemm(x, W.T, b,
  transB=1)`. In place, returns how many were rewritten.

  onnxruntime's MatMulAddFusion already does this fusion at optimization
  level 1, but it emits `transB=0` and leaves W as it was. The CoreML EP's
  Gemm builder then transposes W on the host and adds it through
  `AddConstant`, which is always an immediate, so the weight is written into
  `model.mil` as hex float text: 6.06 bytes per fp16 value measured on an M1
  Pro, which is how the trunk's MIL reached 4.1 GB against a 47 MB
  weight.bin. Every load parses all of it. Handed W already transposed with
  `transB=1`, the builder passes the initializer through as a TensorProto and
  it lands in the weight file instead. The MIL op is `linear` either way, so
  this moves bytes and changes no arithmetic.

  Doing it here rather than leaving it to onnxruntime leaves the fusion
  nothing to fuse. ORT flattens a rank-3 or rank-4 A to 2-D around the Gemm
  and reshapes the result back, because ONNX Gemm is 2-D only; this inserts
  the same pair, so the graph onnxruntime receives is the one its own fusion
  would have built. `session.disable_specified_optimizers` was tried first
  and does not reach this transformer in onnxruntime 1.29.0: the 106 fused
  nodes keep their MatMulAddFusion names under every spelling of it.
  """
  g = model.graph
  init = {t.name: t for t in g.initializer}
  consumers: dict[str, list] = {}
  for node in g.node:
    for name in node.input:
      consumers.setdefault(name, []).append(node)
  outputs = {vi.name for vi in g.output}

  candidates = [n for n in g.node if n.op_type == 'MatMul' and len(n.input) == 2
                and n.input[1] in init and n.output[0] not in outputs]
  dims, _ = _static_info(model, {n.input[0] for n in candidates})

  replacements: dict[int, list] = {}
  drop: set[int] = set()
  order = {id(n): i for i, n in enumerate(g.node)}
  rewritten = 0
  for node in candidates:
    weight = init[node.input[1]]
    if len(weight.dims) != 2 or weight.dims[0] * weight.dims[1] < BLOB_MIN_ELEMENTS:
      continue
    after = consumers.get(node.output[0], [])
    if len(after) != 1 or after[0].op_type != 'Add':
      continue
    add = after[0]
    bias = next((i for i in add.input if i in init), None)
    # The bias has to be the one that broadcasts over the output's last axis;
    # anything else is a real elementwise Add and not a Gemm's C.
    if bias is None or list(init[bias].dims) != [weight.dims[1]]:
      continue
    shape = dims.get(node.input[0])
    if shape is None or len(shape) < 2 or shape[-1] != weight.dims[0] or any(d <= 0 for d in shape):
      continue

    stem = node.output[0]
    array = numpy_helper.to_array(weight)
    transposed = f"{stem}__wt"
    g.initializer.append(numpy_helper.from_array(np.ascontiguousarray(array.T), transposed))
    del array

    new: list = []
    a_name = node.input[0]
    if len(shape) > 2:
      flat = f"{stem}__flat_shape"
      g.initializer.append(numpy_helper.from_array(
        np.array([-1, weight.dims[0]], dtype=np.int64), flat))
      a_name = f"{stem}__flat"
      new.append(helper.make_node('Reshape', [node.input[0], flat], [a_name],
                                  name=f"{stem}__reshape_in"))

    gemm_out = add.output[0] if len(shape) == 2 else f"{stem}__gemm"
    new.append(helper.make_node('Gemm', [a_name, transposed, bias], [gemm_out],
                                name=f"{stem}__gemm", transB=1))
    if len(shape) > 2:
      back = f"{stem}__out_shape"
      g.initializer.append(numpy_helper.from_array(
        np.array(list(shape[:-1]) + [weight.dims[1]], dtype=np.int64), back))
      new.append(helper.make_node('Reshape', [gemm_out, back], [add.output[0]],
                                  name=f"{stem}__reshape_out"))

    replacements[order[id(node)]] = new
    drop.add(order[id(add)])
    rewritten += 1

  if not rewritten:
    return 0

  rebuilt = []
  for i, node in enumerate(g.node):
    if i in drop:
      continue
    rebuilt.extend(replacements.get(i, [node]))
  del g.node[:]
  g.node.extend(rebuilt)
  _drop_unused_initializers(model)
  return rewritten


def _drop_unused_initializers(model: onnx.ModelProto) -> int:
  """The weights the rewrite left behind. A transposed copy replaces the
  original, and the model would otherwise carry 671 MB of both."""
  g = model.graph
  used = {name for node in g.node for name in node.input}
  stale = [t for t in g.initializer if t.name not in used]
  for t in stale:
    g.initializer.remove(t)
  return len(stale)


def _head_casts(g) -> list:
  """The Cast nodes turning the uint8 image inputs into fp16, in either shape."""
  per_input = [n for n in g.node
               if n.op_type == 'Cast' and len(n.input) == 1 and n.input[0] in IMG_INPUTS]
  if per_input:
    return per_input

  concat = next((n for n in g.node if n.op_type == 'Concat'
                 and all(i in IMG_INPUTS for i in n.input)), None)
  if concat is None:
    return []
  cat = concat.output[0]
  cast = next((n for n in g.node if n.op_type == 'Cast' and list(n.input) == [cat]), None)
  return [cast] if cast is not None else []


def patch_file(src: str, dst: str, check: bool = True) -> str:
  model = onnx.load(src)
  strip_tinygrad_ops(model)
  if needs_patch(model):
    patch_uint8_inputs(model)
  if check:
    onnx.checker.check_model(model, full_check=False)
  onnx.save(model, dst)
  return dst


if __name__ == '__main__':
  import sys
  print(patch_file(sys.argv[1], sys.argv[2]))
