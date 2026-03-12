#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build and query causal KG with NetworkX.

Input default: data/my/kg_from_re/triples.jsonl
Output default: data/my/kg_from_re/nx_graph.gpickle

Examples:
  python nx_kg.py build
  python nx_kg.py stats
  python nx_kg.py query --keyword 水位 --hops 1 --limit 20
  python nx_kg.py path --src 水位超过警戒水位 --dst 启动应急响应措施 --max_hops 3
"""

import argparse
import json
import os
import pickle
from collections import Counter

import networkx as nx


def load_triples(path):
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            head = obj.get("head")
            rel = obj.get("relation")
            tail = obj.get("tail")
            if not head or not rel or not tail:
                continue
            triples.append(
                {
                    "head": head,
                    "relation": rel,
                    "tail": tail,
                    "evidence": obj.get("evidence", ""),
                    "confidence": float(obj.get("confidence", 1.0)),
                }
            )
    return triples


def build_graph(triples):
    g = nx.MultiDiGraph()
    for t in triples:
        h, r, ta = t["head"], t["relation"], t["tail"]
        conf = float(t["confidence"])
        g.add_node(h, label="Entity")
        g.add_node(ta, label="Entity")
        g.add_edge(
            h,
            ta,
            relation=r,
            evidence=t["evidence"],
            confidence=conf,
            weight=conf,
        )
    return g


def save_graph(g, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(g, f)


def load_graph(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def print_stats(g):
    rel_counter = Counter()
    for _, _, data in g.edges(data=True):
        rel_counter[data.get("relation", "UNKNOWN")] += 1

    print(f"nodes={g.number_of_nodes()}")
    print(f"edges={g.number_of_edges()}")
    print("relation_distribution:")
    for rel, cnt in rel_counter.most_common():
        print(f"  {rel}: {cnt}")


def query_keyword(g, keyword, hops=1, limit=20):
    matches = [n for n in g.nodes if keyword in n]
    results = []

    for src in matches:
        if hops == 1:
            for _, v, k, data in g.out_edges(src, keys=True, data=True):
                results.append((src, data.get("relation"), v, data.get("evidence", "")))
        else:
            # up to 2 hops for practical QA retrieval
            for _, mid, k1, d1 in g.out_edges(src, keys=True, data=True):
                results.append((src, d1.get("relation"), mid, d1.get("evidence", "")))
                if hops >= 2:
                    for _, dst, k2, d2 in g.out_edges(mid, keys=True, data=True):
                        chain = f"{src} -[{d1.get('relation')}]-> {mid} -[{d2.get('relation')}]-> {dst}"
                        ev = d2.get("evidence", "")
                        results.append(("2hop", "PATH", chain, ev))

    print(f"matched_nodes={len(matches)}")
    for row in results[:limit]:
        print(row)


def query_path(g, src, dst, max_hops=3):
    # convert to simple digraph for path search
    dg = nx.DiGraph()
    for u, v in g.edges():
        dg.add_edge(u, v)

    if src not in dg or dst not in dg:
        print("src_or_dst_not_found")
        return

    found = 0
    for path in nx.all_simple_paths(dg, source=src, target=dst, cutoff=max_hops):
        print(" -> ".join(path))
        found += 1
        if found >= 20:
            break
    if found == 0:
        print("no_path")


def main():
    parser = argparse.ArgumentParser(description="NetworkX KG build/query")
    parser.add_argument("action", choices=["build", "stats", "query", "path"])
    parser.add_argument("--triples", default="data/my/kg_from_re/triples.jsonl")
    parser.add_argument("--graph", default="data/my/kg_from_re/nx_graph.gpickle")
    parser.add_argument("--keyword", default="水位")
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--src", default="")
    parser.add_argument("--dst", default="")
    parser.add_argument("--max_hops", type=int, default=3)
    args = parser.parse_args()

    if args.action == "build":
        triples = load_triples(args.triples)
        g = build_graph(triples)
        save_graph(g, args.graph)
        print(f"built_graph nodes={g.number_of_nodes()} edges={g.number_of_edges()} saved={args.graph}")
        return

    g = load_graph(args.graph)

    if args.action == "stats":
        print_stats(g)
    elif args.action == "query":
        query_keyword(g, args.keyword, hops=args.hops, limit=args.limit)
    elif args.action == "path":
        query_path(g, args.src, args.dst, max_hops=args.max_hops)


if __name__ == "__main__":
    main()
