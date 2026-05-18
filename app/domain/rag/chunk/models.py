from dataclasses import dataclass, field


@dataclass
class Chunk:
    content: str
    doc_id: str = ""
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)
