from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

DEFAULT_EXPORT = "/home/nvidia/workspace/pace_sculptor/artifacts/exports/sculptor_0911"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="piinfra",
        description="FlashRT Pi0.5 hierarchical profiler (ckpt + CUDA graphs, not ONNX/TRT)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("spec", help="Hierarchy from export ckpt (5-cam sculptor_0911)")
    s.add_argument("export", nargs="?", default=DEFAULT_EXPORT)

    s = sub.add_parser("demo", help="FlashRT infer demo on calibration/ samples")
    s.add_argument("--export", default=DEFAULT_EXPORT)
    s.add_argument("--calib", default=None, help="default: $EXPORT/calibration")
    s.add_argument("--num-samples", type=int, default=4)
    s.add_argument("--warmup", type=int, default=5)
    s.add_argument("--repeats", type=int, default=20)
    s.add_argument("--no-fa4", action="store_true")
    s.add_argument("--no-profile", action="store_true")
    s.add_argument("-o", "--out", default=None)

    s = sub.add_parser("layers", help="Per-layer ms + icons; pick NCU targets")
    s.add_argument("--export", default=DEFAULT_EXPORT)
    s.add_argument("--calib", default=None)
    s.add_argument("--num-samples", type=int, default=2)
    s.add_argument("--warmup", type=int, default=2)
    s.add_argument("--repeats", type=int, default=5)
    s.add_argument("--no-fa4", action="store_true")
    s.add_argument("--module", default=None, choices=["vision", "paligemma", "action_expert"])
    s.add_argument("--all", action="store_true", help="print every layer")
    s.add_argument("--detail", action="store_true", help="AE as step×block (not summed)")
    s.add_argument("--top-ncu", type=int, default=8)
    s.add_argument("-o", "--out", default=None)

    s = sub.add_parser("ncu-layer", help="Emit/run NCU; writes .ncu-rep for GUI")
    s.add_argument("layer", help="e.g. paligemma.block5 or action_expert.block3")
    s.add_argument("--export", default=DEFAULT_EXPORT)
    s.add_argument("--app", default="", help="override app cmdline")
    s.add_argument("--out", default=None, help="output stem (writes <out>.ncu-rep)")
    s.add_argument("--set", dest="ncu_set", default="detailed", help="ncu section set: detailed|full|basic|roofline")
    s.add_argument("--run", action="store_true", help="actually run ncu (not just print)")

    s = sub.add_parser("analyze", help="Tax report from nsys kern-sum (FlashRT run)")
    s.add_argument("--export", default=DEFAULT_EXPORT)
    s.add_argument("--nsys", required=True, help=".nsys-rep / kern_sum.csv / json")
    s.add_argument("--layers", default=None, help="flashrt_layers.json to fill per-block ms")
    s.add_argument("-o", "--out", default="piinfra_db.json")
    s.add_argument("--top", type=int, default=15)

    s = sub.add_parser("drilldown", help="Zoom into cast/layout/gemm/vision/...")
    s.add_argument("query")
    s.add_argument("--db", default="piinfra_db.json")
    s.add_argument("--top", type=int, default=20)

    s = sub.add_parser("ncu", help="Emit NCU cmds for top kernels in a tax bucket")
    s.add_argument("query")
    s.add_argument("--db", default="piinfra_db.json")
    s.add_argument("--top", type=int, default=5)
    s.add_argument("--app", default="", help='e.g. --app "python -m flash_rt.piinfra demo"')
    s.add_argument("--run", action="store_true")
    s.add_argument("--out", default="piinfra_ncu")
    s.add_argument("--import-csv", default=None)

    s = sub.add_parser("show", help="Re-print analyze/demo report from db")
    s.add_argument("--db", default="piinfra_db.json")

    args = p.parse_args(argv)

    if args.cmd == "spec":
        from flash_rt.piinfra.model_spec import load_export_spec

        spec = load_export_spec(args.export)
        print(spec.hierarchy_text())
        print(json.dumps(spec.to_dict(), indent=2))
        return 0

    if args.cmd == "demo":
        from flash_rt.piinfra.flashrt_run import run_demo
        from flash_rt.piinfra.report import format_flashrt_demo
        from flash_rt.piinfra.store import save_db

        result = run_demo(
            export_dir=args.export,
            calib_dir=args.calib,
            num_samples=args.num_samples,
            warmup=args.warmup,
            repeats=args.repeats,
            use_fa4=not args.no_fa4,
            profile=not args.no_profile,
        )
        out = args.out or str(Path(args.export) / "benchmark/piinfra/flashrt_demo.json")
        save_db(result["db"], out)
        print(format_flashrt_demo(result))
        print(f"wrote {out}")
        return 0

    if args.cmd == "layers":
        from flash_rt.piinfra.calib_io import load_calibration_samples
        from flash_rt.piinfra.flashrt_run import build_pipe
        from flash_rt.piinfra.layer_profile import (
            format_layer_report,
            run_layer_profile,
            score_layers,
        )
        from flash_rt.piinfra.model_spec import load_export_spec

        spec = load_export_spec(args.export)
        calib = args.calib or str(Path(args.export) / "calibration")
        samples = load_calibration_samples(
            calib,
            image_keys=spec.image_keys,
            num_samples=args.num_samples,
            ckpt_dir=str(Path(args.export) / "ckpt"),
            state_dim=spec.action_dim,
        )
        pipe = build_pipe(
            Path(args.export) / "ckpt",
            num_views=spec.num_views,
            use_fa4=not args.no_fa4,
        )
        s0 = samples[0]
        pipe.set_prompt(s0["prompt"], state=s0["state"])
        pipe.calibrate([{"images": s["images"]} for s in samples[: min(4, len(samples))]], percentile=99.9)
        result = run_layer_profile(
            pipe,
            {"images": s0["images"]},
            repeats=args.repeats,
            warmup=args.warmup,
        )
        src = result["raw"] if args.detail else result["agg"]
        stats = score_layers(src, top_ncu=args.top_ncu)
        print(spec.hierarchy_text())
        print(format_layer_report(stats, module_filter=args.module, show_all=args.all))
        out = args.out or str(Path(args.export) / "benchmark/piinfra/flashrt_layers.json")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(
            json.dumps(
                {
                    "total_ms": result["total_ms"],
                    "layers": [
                        {
                            "path": s.path,
                            "module": s.module,
                            "time_ms": s.time_ms,
                            "pct": s.pct,
                            "icon": s.icon,
                            "ncu": s.ncu,
                            "note": s.note,
                        }
                        for s in stats
                    ],
                },
                indent=2,
            )
        )
        print(f"wrote {out}")
        return 0

    if args.cmd == "ncu-layer":
        from flash_rt.piinfra.ncu import build_ncu_command, format_ncu_command, run_ncu

        layer = args.layer
        # Push/Pop NVTX: trailing ']' = range at top of stack (required by ncu)
        nvtx = f"layer.{layer}]"
        out = args.out
        if not out:
            out = str(Path(args.export) / "benchmark/piinfra" / f"ncu_{layer.replace('.', '_')}")
        Path(out).parent.mkdir(parents=True, exist_ok=True)

        module = layer.split(".", 1)[0] if "." in layer else ""
        app = shlex.split(args.app) if args.app else [
            "python", "-m", "flash_rt.piinfra", "layers",
            "--export", args.export,
            "--repeats", "3", "--warmup", "1",
        ]
        if module and "--module" not in app:
            app.extend(["--module", module])

        # application replay: reliable but slow; kernel replay works if NVTX ] filter matches
        cmd = build_ncu_command(
            ".*",
            app,
            out=out,
            set_name=args.ncu_set,
            replay_mode="kernel",
            extra=["--nvtx", "--nvtx-include", nvtx],
        )
        print(f"# NVTX filter: {nvtx}")
        print(f"# GUI open: {out}.ncu-rep")
        print(format_ncu_command(cmd))
        if args.run:
            print(f"running ncu → {out}.ncu-rep …", flush=True)
            run_ncu(cmd)
            rep = Path(f"{out}.ncu-rep")
            if not rep.is_file() and Path(out).suffix == ".ncu-rep":
                rep = Path(out)
            if rep.is_file() and rep.stat().st_size > 0:
                print(f"wrote {rep} ({rep.stat().st_size} bytes)")
            else:
                print(f"error: expected report missing or empty: {out}.ncu-rep", file=sys.stderr)
                return 2
        else:
            print("# add --run to execute and save the .ncu-rep")
        return 0

    if args.cmd == "analyze":
        from flash_rt.piinfra.analyze_build import assemble_analyze_db, load_layers_json
        from flash_rt.piinfra.model_spec import load_export_spec
        from flash_rt.piinfra.nsys import (
            nsys_stats_kern_sum,
            nsys_stats_nvtx_sum,
            parse_nsys_kern_sum_csv,
            parse_nsys_kern_sum_text,
        )
        from flash_rt.piinfra.report import format_analyze
        from flash_rt.piinfra.schema import KernelLaunch
        from flash_rt.piinfra.store import save_db

        spec = load_export_spec(args.export)
        inp = Path(args.nsys)
        nvtx_rows: list[dict] = []
        if inp.suffix == ".json":
            raw = json.loads(inp.read_text())
            if "layers" in raw and "kernels" not in raw:
                print(
                    "error: this looks like flashrt_layers.json from `piinfra layers`.\n"
                    "  Use: --nsys flashrt.nsys-rep --layers flashrt_layers.json",
                    file=sys.stderr,
                )
                return 2
            kernels = [KernelLaunch(**k) for k in raw.get("kernels", [])]
            nvtx_rows = list(raw.get("nvtx") or [])
        elif inp.suffix == ".csv":
            kernels = parse_nsys_kern_sum_csv(inp)
        elif str(inp).endswith(".nsys-rep") or inp.suffix == ".sqlite":
            kernels = nsys_stats_kern_sum(inp)
            try:
                nvtx_rows = nsys_stats_nvtx_sum(inp)
            except Exception as e:
                print(f"warn: nvtx_sum failed: {e}", file=sys.stderr)
        else:
            kernels = parse_nsys_kern_sum_text(inp.read_text(errors="ignore"))

        if not kernels:
            print(f"error: no CUDA kernels parsed from {inp}", file=sys.stderr)
            return 2

        layers_path = args.layers
        if layers_path is None:
            cand = Path(args.export) / "benchmark/piinfra/flashrt_layers.json"
            if cand.is_file():
                layers_path = str(cand)
            elif Path(args.out).parent.joinpath("flashrt_layers.json").is_file():
                layers_path = str(Path(args.out).parent / "flashrt_layers.json")
        layer_rows = load_layers_json(layers_path)

        db = assemble_analyze_db(
            spec=spec,
            kernels=kernels,
            nvtx_rows=nvtx_rows,
            layer_rows=layer_rows,
            source=f"flashrt+nsys:{args.export}",
        )
        save_db(db, args.out)
        print(spec.hierarchy_text())
        print(format_analyze(db, top_k=args.top))
        if layers_path:
            print(f"layers: {layers_path}")
        print(f"wrote {args.out}")
        return 0

    if args.cmd == "drilldown":
        from flash_rt.piinfra.report import format_drilldown
        from flash_rt.piinfra.store import load_db

        print(format_drilldown(_db_from_raw(load_db(args.db)), args.query, top=args.top))
        return 0

    if args.cmd == "ncu":
        from flash_rt.piinfra.ncu import (
            attach_ncu,
            build_ncu_command,
            format_ncu_command,
            parse_ncu_csv,
            run_ncu,
            top_kernel_regexes,
        )
        from flash_rt.piinfra.store import load_db, save_db
        from flash_rt.piinfra.tax import analyze_taxes

        db = _db_from_raw(load_db(args.db))
        if args.import_csv:
            attach_ncu(db.kernels, parse_ncu_csv(args.import_csv))
            db.taxes = analyze_taxes(db.kernels, db.trt_layers)
            save_db(db, args.db)
            from flash_rt.piinfra.report import format_drilldown

            print(format_drilldown(db, args.query, top=args.top))
            return 0
        regs = top_kernel_regexes(
            db.kernels,
            category=None if args.query == "top" else args.query,
            top=args.top,
        ) or [args.query]
        app = shlex.split(args.app) if args.app else []
        if not app:
            print('# provide --app "python -m flash_rt.piinfra demo --export ..."')
        for i, rgx in enumerate(regs):
            cmd = build_ncu_command(rgx, app or ["#APP#"], out=f"{args.out}_{i}")
            print(format_ncu_command(cmd))
            if args.run and app:
                run_ncu(cmd)
        return 0

    if args.cmd == "show":
        from flash_rt.piinfra.report import format_analyze
        from flash_rt.piinfra.store import load_db

        print(format_analyze(_db_from_raw(load_db(args.db))))
        return 0

    return 2


def _db_from_raw(raw: dict):
    from flash_rt.piinfra.schema import (
        CorrelationRecord,
        HierarchyNode,
        KernelLaunch,
        ProfileDB,
        TaxBucket,
        TrtLayer,
    )

    def hier(d):
        if not d:
            return None
        n = HierarchyNode(
            path=d["path"],
            kind=d["kind"],
            time_ms=d.get("time_ms", 0.0),
            pct=d.get("pct", 0.0),
            trt_layers=d.get("trt_layers") or [],
            kernels=d.get("kernels") or [],
            flags=d.get("flags") or [],
        )
        n.children = [hier(c) for c in d.get("children") or []]
        return n

    db = ProfileDB(
        source=raw.get("source", ""),
        total_ms=raw.get("total_ms", 0.0),
        onnx_ops=raw.get("onnx_ops") or {},
        onnx_nodes=raw.get("onnx_nodes") or 0,
        trt_layers=[TrtLayer(**x) for x in raw.get("trt_layers") or []],
        kernels=[KernelLaunch(**x) for x in raw.get("kernels") or []],
        correlations=[CorrelationRecord(**x) for x in raw.get("correlations") or []],
        taxes=[TaxBucket(**x) for x in raw.get("taxes") or []],
        meta=raw.get("meta") or {},
    )
    db.hierarchy = hier(raw.get("hierarchy"))
    return db


if __name__ == "__main__":
    raise SystemExit(main())
