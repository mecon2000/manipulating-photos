"""Session step graph: append-only nodes + an undo pointer.

Nodes are never destroyed — undo moves `head` to the parent, redo moves it to
the most recently created child. Editing a mid-graph step creates a sibling
branch (new node + clones of the old descendants re-parented onto it), so the
old chain stays reachable and upstream cache entries keep hitting.
"""
import json
import time
import uuid
from pathlib import Path

from .paths import SESSIONS_DIR, ensure_dirs, trash


def _now() -> float:
    return time.time()


class Session:
    def __init__(self, data: dict):
        self.data = data

    # -- persistence --------------------------------------------------------
    @property
    def path(self) -> Path:
        return SESSIONS_DIR / self.data["id"] / "graph.json"

    def save(self) -> None:
        ensure_dirs()
        self.data["updated"] = _now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)

    @classmethod
    def create(cls, source_path: str, source_ref: str) -> "Session":
        sid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        s = cls({
            "id": sid,
            "source_path": str(source_path),
            "source_ref": source_ref,
            "created": _now(),
            "updated": _now(),
            "nodes": {},
            "order": [],
            "head": None,
        })
        s.save()
        return s

    @classmethod
    def load(cls, session_id: str) -> "Session":
        if "/" in session_id or ".." in session_id:
            raise ValueError("bad session id")
        p = SESSIONS_DIR / session_id / "graph.json"
        return cls(json.loads(p.read_text()))

    @classmethod
    def list_ids(cls) -> list[str]:
        if not SESSIONS_DIR.exists():
            return []
        return sorted(d.name for d in SESSIONS_DIR.iterdir()
                      if (d / "graph.json").exists())

    def clear(self) -> None:
        """Trash the whole session dir (never rm)."""
        trash(self.path.parent)

    # -- graph ops ----------------------------------------------------------
    @property
    def nodes(self) -> dict:
        return self.data["nodes"]

    def add_step(self, tool: str, params: dict, seed=None, preview=True,
                 flags=None, parent: str | None = "HEAD") -> dict:
        if parent == "HEAD":
            parent = self.data["head"]
        if parent is not None and parent not in self.nodes:
            raise KeyError(f"unknown parent node {parent}")
        node = {
            "id": uuid.uuid4().hex[:10],
            "parent": parent,
            "tool": tool,
            "params": params or {},
            "flags": flags or [],
            "seed": seed,
            "preview": bool(preview),
            "created": _now(),
        }
        self.nodes[node["id"]] = node
        self.data["order"].append(node["id"])
        self.data["head"] = node["id"]
        self.save()
        return node

    def chain(self, node_id: str | None = None) -> list[dict]:
        """Nodes from root to node_id (default: head), in execution order."""
        nid = self.data["head"] if node_id is None else node_id
        out = []
        while nid is not None:
            node = self.nodes[nid]
            out.append(node)
            nid = node["parent"]
        return list(reversed(out))

    def children(self, node_id: str | None) -> list[dict]:
        return sorted((n for n in self.nodes.values() if n["parent"] == node_id),
                      key=lambda n: n["created"])

    def set_head(self, node_id: str | None) -> str | None:
        """Jump head to any node (variant switch / step-strip selection)."""
        if node_id is not None and node_id not in self.nodes:
            raise KeyError(f"unknown node {node_id}")
        self.data["head"] = node_id
        self.save()
        return node_id

    def variant_group(self, node_id: str) -> list[dict]:
        """Sibling variants of a node: same parent, tool and params, differing
        only by seed. Includes the node itself, ordered by creation."""
        n = self.nodes[node_id]
        return sorted((c for c in self.nodes.values()
                       if c["parent"] == n["parent"] and c["tool"] == n["tool"]
                       and c["params"] == n["params"]),
                      key=lambda c: c["created"])

    def undo(self) -> str | None:
        head = self.data["head"]
        if head is None:
            return None
        self.data["head"] = self.nodes[head]["parent"]
        self.save()
        return self.data["head"]

    def redo(self) -> str | None:
        kids = self.children(self.data["head"])
        if not kids:
            return None
        self.data["head"] = kids[-1]["id"]
        self.save()
        return self.data["head"]

    def edit_step(self, node_id: str, params: dict | None = None, seed="KEEP",
                  flags=None) -> dict:
        """Re-do a mid-graph step with new params: creates a sibling branch and
        re-parents clones of the active chain's downstream nodes onto it.
        Returns the new branch's tail node (new head)."""
        active = self.chain()
        idx = next((i for i, n in enumerate(active) if n["id"] == node_id), None)
        if idx is None:
            raise KeyError(f"node {node_id} is not on the active chain")
        old = active[idx]
        replacement = self.add_step(
            tool=old["tool"],
            params=params if params is not None else old["params"],
            seed=old["seed"] if seed == "KEEP" else seed,
            preview=old["preview"],
            flags=flags if flags is not None else old["flags"],
            parent=old["parent"],
        )
        tail = replacement
        for down in active[idx + 1:]:
            tail = self.add_step(tool=down["tool"], params=down["params"],
                                 seed=down["seed"], preview=down["preview"],
                                 flags=down["flags"], parent=tail["id"])
        return tail
