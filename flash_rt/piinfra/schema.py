from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class OnnxNodeIndex:
    node_id: int
    name: str
    op_type: str
    inputs: list[str]
    outputs: list[str]
    model_path: str = ""


@dataclass
class TrtLayer:
    name: str
    time_ms: float = 0.0
    pct: float = 0.0
    layer_type: str = ""
    model_path: str = ""


@dataclass
class KernelLaunch:
    name: str
    time_ms: float = 0.0
    pct: float = 0.0
    instances: int = 0
    avg_us: float = 0.0
    category: str = ""
    model_path: str = ""
    trt_layer: str = ""
    ncu: dict[str, Any] = field(default_factory=dict)


@dataclass
class HierarchyNode:
    path: str
    kind: str
    time_ms: float = 0.0
    pct: float = 0.0
    children: list["HierarchyNode"] = field(default_factory=list)
    trt_layers: list[str] = field(default_factory=list)
    kernels: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "time_ms": self.time_ms,
            "pct": self.pct,
            "flags": self.flags,
            "trt_layers": self.trt_layers,
            "kernels": self.kernels,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class CorrelationRecord:
    model_node: str
    trt_layer: str
    kernel: str
    duration_us: float = 0.0
    nsys: dict[str, Any] = field(default_factory=dict)
    ncu: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaxBucket:
    name: str
    time_ms: float
    pct: float
    count: int = 0
    severity: str = ""
    diagnosis: str = ""
    recommendations: list[str] = field(default_factory=list)


@dataclass
class ProfileDB:
    source: str = ""
    total_ms: float = 0.0
    onnx_ops: dict[str, int] = field(default_factory=dict)
    onnx_nodes: int = 0
    trt_layers: list[TrtLayer] = field(default_factory=list)
    kernels: list[KernelLaunch] = field(default_factory=list)
    hierarchy: HierarchyNode | None = None
    correlations: list[CorrelationRecord] = field(default_factory=list)
    taxes: list[TaxBucket] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "total_ms": self.total_ms,
            "onnx_nodes": self.onnx_nodes,
            "onnx_ops": self.onnx_ops,
            "trt_layers": [asdict(x) for x in self.trt_layers],
            "kernels": [asdict(x) for x in self.kernels],
            "hierarchy": None if self.hierarchy is None else self.hierarchy.to_dict(),
            "correlations": [c.to_dict() for c in self.correlations],
            "taxes": [asdict(t) for t in self.taxes],
            "meta": self.meta,
        }
