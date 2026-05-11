# DocumentType枚举类
from enum import Enum


class DocumentType(Enum):
    FAQ = "faq"
    MANUAL = "manual"
    PAPER = "paper"
    API_DOC = "api_doc"
    OTHER = "other"


class DocumentClassifier:

    def classify(
            self,
    ) -> DocumentType:
        pass
