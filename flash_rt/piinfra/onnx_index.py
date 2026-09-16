from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from flash_rt.piinfra.hierarchy import model_path_from_name
from flash_rt.piinfra.schema import OnnxNodeIndex


def index_onnx(onnx_path: str | Path, *, load_weights: bool = False) -> dict[str, Any]:
    import onnx

    path = Path(onnx_path)
    model = onnx.load(str(path), load_external_data=bool(load_weights))
    value_info = {v.name: v for v in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)}

    nodes: list[OnnxNodeIndex] = []
    producer: dict[str, int] = {}
    children: dict[int, list[int]] = defaultdict(list)
    parents: dict[int, list[int]] = defaultdict(list)
    ops = Counter()
    module_ops: dict[str, Counter] = defaultdict(Counter)

    for i, n in enumerate(model.graph.node):
        name = n.name or f"node_{i}"
        mp = model_path_from_name(name)
        item = OnnxNodeIndex(
            node_id=i,
            name=name,
            op_type=n.op_type,
            inputs=list(n.input),
            outputs=list(n.output),
            model_path=mp,
        )
        nodes.append(item)
        ops[n.op_type] += 1
        module_ops[mp.split(".")[0]][n.op_type] += 1
        for o in n.output:
            producer[o] = i

    for i, n in enumerate(nodes):
        for inp in n.inputs:
            pid = producer.get(inp)
            if pid is None:
                continue
            parents[i].append(pid)
            children[pid].append(i)

    shapes: dict[str, list[int] | None] = {}
    for name, vi in value_info.items():
        try:
            dims = []
            for d in vi.type.tensor_type.shape.dim:
                dims.append(d.dim_value if d.dim_value else None)
            shapes[name] = dims
        except Exception:
            shapes[name] = None

    hierarchy_counts = Counter(n.model_path for n in nodes)
    return {
        "path": str(path),
        "nodes": len(nodes),
        "initializers": len(model.graph.initializer),
        "inputs": [i.name for i in model.graph.input],
        "outputs": [o.name for o in model.graph.output],
        "ops": dict(ops.most_common()),
        "module_ops": {k: dict(v.most_common()) for k, v in module_ops.items()},
        "hierarchy_counts": dict(hierarchy_counts.most_common()),
        "index": [
            {
                "node_id": n.node_id,
                "name": n.name,
                "op_type": n.op_type,
                "inputs": n.inputs,
                "outputs": n.outputs,
                "model_path": n.model_path,
                "parents": parents.get(n.node_id, []),
                "children": children.get(n.node_id, []),
            }
            for n in nodes
        ],
        "shapes": shapes,
    }


def save_index(index: dict[str, Any], out: str | Path) -> Path:
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in index.items() if k != "index"}
    outp.write_text(json.dumps(slim, indent=2))
    full = outp.with_suffix(outp.suffix + ".full.json") if outp.suffix else Path(str(outp) + ".full.json")
    full.write_text(json.dumps(index))
    return outp
