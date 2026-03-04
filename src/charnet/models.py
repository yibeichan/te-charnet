"""Data models for the charnet pipeline."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Utterance:
    """A single speaker utterance from the transcript."""
    speaker: str
    start: float
    end: float
    text: str
    index: int = 0  # original position in transcript


@dataclass
class Shot:
    """A single continuous camera take from PySceneDetect output."""
    shot_id: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Scene:
    """A meaningful narrative unit, composed of merged shots."""
    scene_id: int
    start: float
    end: float
    speakers: list[str] = field(default_factory=list)
    n_shots: int = 0
    n_utterances: int = 0
    utterance_indices: list[int] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "start": self.start,
            "end": self.end,
            "speakers": self.speakers,
            "n_shots": self.n_shots,
            "n_utterances": self.n_utterances,
            "utterance_indices": self.utterance_indices,
        }


@dataclass
class EdgeData:
    """Weighted interaction edge between two characters."""
    source: str
    target: str
    weight: float
    adjacency: float
    proximity: float
    copresence: float

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "adjacency": self.adjacency,
            "proximity": self.proximity,
            "copresence": self.copresence,
        }


@dataclass
class SceneGraph:
    """Interaction graph for a single scene."""
    scene_id: int
    start: float
    end: float
    nodes: list[str] = field(default_factory=list)
    edges: list[EdgeData] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "start": self.start,
            "end": self.end,
            "nodes": self.nodes,
            "edges": [e.to_dict() for e in self.edges],
        }
