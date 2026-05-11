# 知识块切分流程
class Chunker:

    def __init__(self, minio_storage):
        self.minio_storage = minio_storage

    def split(self):
        print("Splitting knowledge blocks")
