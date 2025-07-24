import uuid
from typing import Optional, Dict, Text, Any


class UserInterfaceDocument:

    def __init__(self, id: Optional[str] = None, filepath: str = '', annotation: Optional[Dict[Text, Any]] = None, text: Optional[Text] = None):
        self.id = id
        self.filepath = filepath
        self.annotation = annotation
        self.text = text

    def to_dict(self) -> Dict[Text, Any]:
        return {'id': self.id, 'filepath': self.filepath, 'annotation': self.annotation, 'text': self.text}

    @classmethod
    def from_dict(cls, d: Dict[Text, Any]) -> 'UserInterfaceDocument':
        return cls(
            id=d.get('id'),
            filepath=d.get('filepath', ''),
            annotation=d.get('annotation'),
            text=d.get('text')
        )

    def __eq__(self, other):
        if not isinstance(other, UserInterfaceDocument):
            return False
        return self.id == other.id and self.filepath == other.filepath and self.text == other.text

    def __hash__(self):
        return hash((self.id, self.filepath, self.text))

    def __str__(self):
        return f"UserInterfaceDocument(id={self.id}, filepath={self.filepath}, text={self.text})"

    def __repr__(self):
        return f"UserInterfaceDocument(id={repr(self.id)}, filepath={repr(self.filepath)}, text={repr(self.text)})"
