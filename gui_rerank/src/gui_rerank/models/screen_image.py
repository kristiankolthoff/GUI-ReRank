from typing import Optional, Dict, Text, Any

class ScreenImage:
    
    def __init__(self, id: str, filelocation: str):
        self.id = id
        self.filelocation = filelocation

    def to_dict(self) -> Dict[Text, Any]:
        return {'id': self.id, 'filelocation': self.filelocation}

    def __eq__(self, other):
        if not isinstance(other, ScreenImage):
            return False
        return self.id == other.id and self.filelocation == other.filelocation

    def __hash__(self):
        return hash((self.id, self.filelocation))

    def __str__(self):
        return f"ScreenImage(id={self.id}, filelocation={self.filelocation})"

    def __repr__(self):
        return f"ScreenImage(id={repr(self.id)}, filelocation={repr(self.filelocation)})"
