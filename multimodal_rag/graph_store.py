"""Persist undirected chunk adjacency for graph-expanded retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from langchain_core.documents import Document

EDGE_SAME_ELEMENT_NEXT = "same_element_next"
EDGE_ADJACENT_ELEMENT = "adjacent_element"


def graph_json_path(persist_directory: Path | str) -> Path:
    return Path(persist_directory).expanduser().resolve() / "chunk_graph.json"


@dataclass
class ChunkGraph:
    """Undirected neighbor map plus optional typed edges for debugging."""

    neighbors: dict[str, list[str]]
    edge_types: dict[str, str]

    @classmethod
    def empty(cls) -> ChunkGraph:
        return cls(neighbors={}, edge_types={})

    @classmethod
    def load(cls, path: Path) -> ChunkGraph:
        if not path.is_file():
            return cls.empty()
        data = json.loads(path.read_text(encoding="utf-8"))
        neighbors = {k: list(v) for k, v in (data.get("neighbors") or {}).items()}
        edge_types = dict(data.get("edge_types") or {})
        return cls(neighbors=neighbors, edge_types=edge_types)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "neighbors": {k: sorted(set(v)) for k, v in self.neighbors.items()},
            "edge_types": dict(sorted(self.edge_types.items())),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def merge_undirected_edge(self, a: str, b: str, rel: str) -> None:
        if a == b:
            return
        self.neighbors.setdefault(a, [])
        self.neighbors.setdefault(b, [])
        if b not in self.neighbors[a]:
            self.neighbors[a].append(b)
        if a not in self.neighbors[b]:
            self.neighbors[b].append(a)
        key = _edge_key(a, b)
        self.edge_types[key] = rel

    def merge_from_local(self, local: ChunkGraph) -> None:
        for a, nbrs in local.neighbors.items():
            for b in nbrs:
                rel = local.edge_types.get(_edge_key(a, b), EDGE_ADJACENT_ELEMENT)
                self.merge_undirected_edge(a, b, rel)


def _edge_key(a: str, b: str) -> str:
    return f"{a}|{b}" if a < b else f"{b}|{a}"


def build_local_graph_from_ordered_docs(docs: list[Document]) -> ChunkGraph:
    """Build neighbor edges from document order and element_index metadata."""
    g = ChunkGraph.empty()
    if len(docs) < 2:
        return g
    for i in range(len(docs) - 1):
        m0 = docs[i].metadata
        m1 = docs[i + 1].metadata
        a = m0.get("chunk_id")
        b = m1.get("chunk_id")
        if not a or not b:
            continue
        ei0 = m0.get("element_index")
        ei1 = m1.get("element_index")
        rel = (
            EDGE_SAME_ELEMENT_NEXT
            if ei0 is not None and ei0 == ei1
            else EDGE_ADJACENT_ELEMENT
        )
        g.merge_undirected_edge(str(a), str(b), rel)
    return g


def expand_chunk_ids(
    seed_ids: list[str],
    graph: ChunkGraph,
    *,
    max_hops: int,
    expand_k: int,
) -> list[str]:
    """BFS by hop; collect up to expand_k new chunk ids not in seeds."""
    if expand_k <= 0 or max_hops <= 0:
        return []
    seed_set = set(seed_ids)
    collected: list[str] = []
    seen_new: set[str] = set()
    frontier = list(seed_ids)
    for _ in range(max_hops):
        next_frontier: list[str] = []
        for cid in frontier:
            for nbr in graph.neighbors.get(cid, []):
                if nbr in seed_set or nbr in seen_new:
                    continue
                seen_new.add(nbr)
                collected.append(nbr)
                next_frontier.append(nbr)
                if len(collected) >= expand_k:
                    return collected[:expand_k]
        frontier = next_frontier
        if not frontier:
            break
    return collected[:expand_k]
