"""Import layers of the bs3 package (refactoring plan of 3.1, stage 9).

The import graph is read from the source with `ast`, nothing is imported: every `import` and `from … import` at module
level and inside functions, relative imports resolved («from . import caveats» is an edge to bs3.caveats, «from . import
MODEL_TITLES» an edge to the package bs3 itself).

Layers, bottom to top: leaf, data, text, render, analysis, top. A module imports modules of its own layer or below.
The rules of the plan:
  (a) no import cycle;
  (b) a leaf imports only leaves; (c) data imports only leaf and data; (d) text imports no render or analysis module;
      a render module imports no analysis module either, except that the app module imports pipeline;
  (e) the PDF modules never import the web modules; only the app module imports the PDF build and pipeline;
  (f) outside the analysis layer nothing imports torch, cv2, transformers, pandas or librosa at module level.
ALLOW holds the violations the code still has, each with the stage of the plan that removes it. The test fails on a
violation that is not in ALLOW and on an ALLOW entry that is no longer a violation, so every later stage shrinks ALLOW
in its own commit; after stage 22 it is empty. A new module must be given a layer in LAYERS.

`python tests/test_layers.py` prints the violations and their lines.
"""
from __future__ import annotations

import ast
from fnmatch import fnmatchcase
from pathlib import Path

BS3 = Path(__file__).resolve().parents[1] / "bs3"

LEAF, DATA, TEXT, RENDER, ANALYSIS, TOP = range(6)
LAYER_NAMES = ("leaf", "data", "text", "render", "analysis", "top")
# module names relative to the package ("bs3" is bs3/__init__.py); names that do not exist yet are ignored
LAYERS = {
    LEAF: ("bs3", "settings", "palette", "norms", "media", "ollama", "textfmt", "labels", "bands", "frame_phrase",
           "errors", "jobfiles"),
    DATA: ("scores", "mbti", "report", "segments"),
    TEXT: ("caveats", "characterization", "narrative", "facts", "analyses_text", "ru_texts", "translate", "words",
           "frame_captions"),
    RENDER: ("journal", "jobview", "web", "web.*", "pdf", "pdf.*"),
    ANALYSIS: ("pipeline", "longvideo", "backend_*", "analyses", "analyses.*", "mm", "mm.*"),
    TOP: ("cli",),
}
APP = ("web.app",)                                       # the Gradio app module (web.app since the web split)
PDF_BUILD = ("pdf", "pdf.build")                         # builds the PDF: build_pdf, export_pdf
PDF_MODULES = ("pdf", "pdf.*")
WEB_MODULES = ("web", "web.*")
HEAVY = ("torch", "cv2", "transformers", "pandas", "librosa")
# modules whose docstring promises no bs3 import at all (palette is loaded standalone by scripts/check_palette.py)
NO_BS3_IMPORTS = ("palette", "settings", "jobfiles", "textfmt", "frame_phrase", "bands")

# (importer, imported, the stage of the refactoring plan that removes the edge)
# Empty since stage 20 moved AnalysisCancelled into bs3/errors.py: web.app no longer imports longvideo (and torch) to
# build the page. Every layer edge is now clean.
ALLOW: set = set()


def _name(path: Path) -> str:
    parts = list(path.relative_to(BS3).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) or "bs3"


def modules() -> dict[str, Path]:
    return {_name(p): p for p in sorted(BS3.rglob("*.py")) if "__pycache__" not in p.parts}


def layer(name: str) -> int | None:
    found = [lv for lv, pats in LAYERS.items() if any(fnmatchcase(name, p) for p in pats)]
    return found[0] if len(found) == 1 else None


def _is(name: str, patterns) -> bool:
    return any(fnmatchcase(name, p) for p in patterns)


def _targets(importer: str, is_pkg: bool, node, mods) -> list[str]:
    """What one import statement imports: a bs3 module relative to the package ("bs3" for the package itself), or
    the top-level name of an outside package."""
    def inner(dotted: str) -> str | None:
        if dotted == "bs3":
            return "bs3"
        return dotted[4:] if dotted.startswith("bs3.") else None

    out = []
    if isinstance(node, ast.Import):
        for a in node.names:
            n = inner(a.name)
            out.append(n if n is not None else a.name.split(".")[0])
        return out
    if node.level:
        base = [] if importer == "bs3" else importer.split(".")
        if not is_pkg:
            base = base[:-1]
        if node.level > 1:
            base = base[:len(base) - (node.level - 1)]
        pkg = ".".join(base + ([node.module] if node.module else []))
    else:
        n = inner(node.module or "")
        if n is None:
            return [(node.module or "").split(".")[0]]
        pkg = "" if n == "bs3" else n
    for a in node.names:
        sub = f"{pkg}.{a.name}" if pkg else a.name
        out.append(sub if sub in mods else (pkg or "bs3"))
    return out


def graph() -> dict[tuple[str, str], list[tuple[int, bool]]]:
    """{(importer, imported): [(line, at module level)]}; imported is a bs3 module name or an outside package."""
    mods = modules()
    edges: dict[tuple[str, str], list[tuple[int, bool]]] = {}

    def visit(name, is_pkg, node, in_func):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                if isinstance(child, ast.ImportFrom) and child.module == "__future__":
                    continue
                for t in _targets(name, is_pkg, child, mods):
                    if t != name:
                        edges.setdefault((name, t), []).append((child.lineno, not in_func))
            visit(name, is_pkg, child,
                  in_func or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)))

    for name, path in mods.items():
        visit(name, path.name == "__init__.py", ast.parse(path.read_text(encoding="utf-8")), False)
    return edges


def _adjacency(edges, skip=()) -> dict[str, set[str]]:
    mods = modules()
    adj: dict[str, set[str]] = {}
    for a, b in edges:
        if b in mods and (a, b) not in skip:
            adj.setdefault(a, set()).add(b)
    return adj


def _reaches(src: str, dst: str, adj: dict[str, set[str]]) -> bool:
    seen, todo = {src}, [src]
    while todo:
        for m in adj.get(todo.pop(), ()):
            if m == dst:
                return True
            if m not in seen:
                seen.add(m)
                todo.append(m)
    return False


def _cycle(start: str, adj: dict[str, set[str]]) -> list[str] | None:
    """The shortest import path start -> … -> start, or None."""
    parent: dict[str, str] = {}
    todo = [start]
    while todo:
        nxt = []
        for n in todo:
            for m in sorted(adj.get(n, ())):
                if m == start:
                    path = [n]
                    while path[-1] != start:
                        path.append(parent[path[-1]])
                    return path[::-1] + [start]
                if m not in parent:
                    parent[m] = n
                    nxt.append(m)
        todo = nxt
    return None


def violations(edges=None) -> dict[tuple[str, str], list[str]]:
    """{(importer, imported): [reasons]} for every edge that breaks rules (b)–(f) of the module docstring. Cycles (a)
    are found on the whole graph by test_no_import_cycle."""
    edges = graph() if edges is None else edges
    mods = modules()
    out: dict[tuple[str, str], list[str]] = {}
    for (a, b), where in edges.items():
        reasons = []
        if b in mods:
            la, lb = layer(a), layer(b)
            if la is not None and lb is not None and lb > la and not (_is(a, APP) and b == "pipeline"):
                reasons.append(f"the {LAYER_NAMES[la]} layer imports the {LAYER_NAMES[lb]} layer")
            if _is(a, PDF_MODULES) and _is(b, WEB_MODULES):
                reasons.append("a PDF module imports a web module")
            if b in PDF_BUILD and not _is(a, APP) and not (a == "pdf" and b == "pdf.build"):
                reasons.append("only the app module imports the PDF build")
            if b == "pipeline" and not _is(a, APP):
                reasons.append("only the app module imports pipeline")
        elif b in HEAVY and layer(a) != ANALYSIS and any(top for _, top in where):
            reasons.append(f"{b} imported at module level outside the analysis layer")
        if reasons:
            out[(a, b)] = reasons
    return out


def _report(items, edges) -> str:
    def lines(e) -> str:
        return ", ".join(str(ln) for ln in sorted({ln for ln, _ in edges[e]}))
    return "\n".join(f"  {a} -> {b} (line {lines((a, b))}): {'; '.join(why)}" for (a, b), why in sorted(items))


def test_every_module_has_one_layer():
    names = sorted(modules())
    assert {"bs3", "textfmt", "scores", "pipeline", "mm.explain", "cli"} <= set(names), names
    missing = [n for n in names if layer(n) is None]
    assert not missing, f"give these modules one layer in LAYERS: {missing}"


def test_the_graph_reads_every_kind_of_import():
    edges = graph()
    expected = {
        ("pipeline", "report"): True,                  # from .report import build_report
        ("pipeline", "jobfiles"): True,                # from . import …, jobfiles (a submodule)
        ("caveats", "bs3"): True,                      # from . import MODEL_TITLES, PRODUCT (the package)
        ("caveats", "textfmt"): True,
        ("cli", "backend_mm"): False,                  # inside a function
        ("analyses.emotions_voice", "mm.extractors"): True,   # from ..mm.extractors
        ("backend_mm", "torch"): True,                 # an outside package
    }
    for edge, top in expected.items():
        assert edge in edges, edge
        assert any(t == top for _, t in edges[edge]), (edge, edges[edge])
    assert not any(b == "__future__" for _, b in edges)


def test_no_new_layer_violation():
    edges = graph()
    allowed = {(a, b) for a, b, _ in ALLOW}
    new = [(e, why) for e, why in violations(edges).items() if e not in allowed]
    assert not new, "imports that break the layers of tests/test_layers.py:\n" + _report(new, edges)


def test_no_import_cycle():
    """(a), on the graph without the ALLOW entries (module-level and function-level imports alike)."""
    adj = _adjacency(graph(), skip={(a, b) for a, b, _ in ALLOW})
    cycles = [c for c in (_cycle(n, adj) for n in sorted(adj)) if c]
    assert not cycles, "import cycles:\n" + "\n".join("  " + " -> ".join(c) for c in cycles)


def test_allow_holds_only_current_violations():
    """Every ALLOW entry is an import the code still has and that still breaks a rule or closes a cycle: a stage
    that removes an import removes its entry in the same commit."""
    edges = graph()
    found = violations(edges)
    adj = _adjacency(edges)
    assert len({(a, b) for a, b, _ in ALLOW}) == len(ALLOW)
    stale = sorted((a, b, st) for a, b, st in ALLOW
                   if (a, b) not in edges or ((a, b) not in found and not _reaches(b, a, adj)))
    assert not stale, f"remove these ALLOW entries, the import is gone or no longer breaks a rule: {stale}"
    assert all(isinstance(st, int) and 10 <= st <= 22 for _, _, st in ALLOW), ALLOW


def test_modules_without_bs3_imports():
    edges = graph()
    mods = modules()
    for name in NO_BS3_IMPORTS:
        assert name in mods, name
        own = sorted(b for a, b in edges if a == name and b in mods)
        assert not own, f"bs3/{name}.py imports {own}"


if __name__ == "__main__":
    g = graph()
    v = violations(g)
    allowed = {(a, b) for a, b, _ in ALLOW}
    print("violations (* = in ALLOW):")
    print("\n".join(("* " if e in allowed else "  ") + _report([(e, why)], g).strip() for e, why in sorted(v.items())))
