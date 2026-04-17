from dataclasses import asdict, dataclass, field
import json
from typing import Any, List, Optional


@dataclass
class Explanation:
    name: str
    value: Optional[Any] = None
    reasons: List[str] = field(default_factory=list)
    children: List["Explanation"] = field(default_factory=list)

    def add_child(self, child: "Explanation") -> None:
        self.children.append(child)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)