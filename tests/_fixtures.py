"""Shared fixtures: one loaded graph and one warm index for the whole suite."""

from __future__ import annotations

import functools

from tcm_kg import load_kg
from tcm_kg.index import KGRetriever


@functools.lru_cache(maxsize=1)
def graph():
    return load_kg()


@functools.lru_cache(maxsize=1)
def retriever():
    r = KGRetriever(graph())
    r.warm()
    return r
