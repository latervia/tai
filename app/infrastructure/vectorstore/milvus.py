import logging
from typing import List, Optional, Dict, Any

from pymilvus import (
    DataType,
    MilvusException,
    MilvusClient, Function, FunctionType,
)

from app.shared.config import settings

logger = logging.getLogger(__name__)


class MilvusManager:
    """优化后的 Milvus 客户端：统一使用 MilvusClient (v2.4+)"""

    _instance: Optional["MilvusManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.config = settings.milvus
        self._dim = self.config.dim
        # 统一使用 MilvusClient
        self.client = self._connect()
        self._init_collection()
        self._initialized = True

    def _connect(self) -> MilvusClient:
        try:
            client = MilvusClient(
                uri=f"http://{self.config.host}:{self.config.port}",
                user=self.config.user,
                password=self.config.password,
                db_name=self.config.database,
            )
            logger.info(f"Successfully connected to Milvus at {self.config.host}:{self.config.port}")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def _init_collection(self):
        """初始化集合：判断是否存在，不存在则创建。"""
        try:
            if self.client.has_collection(self.config.collection):
                # 自动同步维度信息
                desc = self.client.describe_collection(self.config.collection)
                self._sync_dim_from_desc(desc)
                logger.info(f"Collection '{self.config.collection}' loaded. Current Dim: {self._dim}")
            else:
                self._create_collection()
        except MilvusException as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise

    def _sync_dim_from_desc(self, desc: Dict[str, Any]):
        """从集合描述中提取 dense_vector 的维度。"""
        for field in desc.get("fields", []):
            if field.get("name") == "dense_vector":
                params = field.get("params", {})
                self._dim = int(params.get("dim", self._dim))

    def _create_collection(self):
        """使用 MilvusClient 的方式创建集合和索引。"""
        try:
            schema = self.client.create_schema(
                auto_id=True,
                enable_dynamic_field=False,
                description="Knowledge base vector collection",
            )

            # 添加字段
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=512)
            schema.add_field(field_name="chunk_index", datatype=DataType.INT32)
            schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=16384,
                             enable_analyzer=True, enable_match=True)
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self._dim)
            schema.add_field(field_name="metadata", datatype=DataType.JSON)
            schema.add_field(field_name="tenant_id", datatype=DataType.VARCHAR, max_length=128, is_partition_key=True)

            schema.add_function(
                Function(name="bm25_fn",
                         input_field_names=["content"],
                         output_field_names=["sparse_vector"],
                         function_type=FunctionType.BM25))

            # 配置索引参数
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="dense_vector",
                index_type="HNSW",
                metric_type="IP",
                params={"M": 16, "efConstruction": 200}
            )
            index_params.add_index(
                field_name="sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={"drop_ratio_build": 0.2}
            )

            # 一键创建集合（包含 Schema 和 Index）
            self.client.create_collection(
                collection_name=self.config.collection,
                schema=schema,
                index_params=index_params
            )
            logger.info(f"Collection '{self.config.collection}' created with HNSW and Sparse indexes.")

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise

    # === 数据操作 ===

    def insert_vectors(
            self,
            doc_ids: List[str],
            chunk_indices: List[int],
            documents: List[str],
            dense_vectors: List[List[float]],
            sparse_vectors: Optional[List[Dict[int, float]]] = None,
            metadata_list: Optional[List[Dict[str, Any]]] = None,
            tenant_ids: Optional[List[str]] = None,
    ) -> List[Any]:
        """批量插入数据。Client 模式下推荐使用 List[Dict] 格式，更直观且不易错位。"""
        n = len(doc_ids)
        if sparse_vectors is None: sparse_vectors = [{}] * n
        if metadata_list is None: metadata_list = [{}] * n
        if tenant_ids is None: tenant_ids = ["default"] * n

        rows = []
        for i in range(n):
            rows.append({
                "doc_id": doc_ids[i],
                "chunk_index": chunk_indices[i],
                "content": documents[i],
                # "sparse_vector": sparse_vectors[i],
                "dense_vector": dense_vectors[i],
                "metadata": metadata_list[i],
                "tenant_id": tenant_ids[i],
            })

        try:
            res = self.client.insert(collection_name=self.config.collection, data=rows)
            return res.get("primary_keys", [])
        except Exception as e:
            logger.error(f"Insert failed: {e}")
            raise

    # === 检索 ===

    def search(
            self,
            query_vector: List[float],
            limit: int = 10,
            output_fields: Optional[List[str]] = None,
            tenant_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """使用 Client 进行搜索。"""
        filter_expr = f'tenant_id == "{tenant_id}"' if tenant_id else ""

        results = self.client.search(
            collection_name=self.config.collection,
            data=[query_vector],
            limit=limit,
            filter=filter_expr,
            output_fields=output_fields or ["doc_id", "content", "metadata"],
            search_params={"metric_type": "IP", "params": {"ef": 64}}
        )
        # Client 的返回格式通常直接就是 List[List[Dict]]
        return results[0] if results else []

    def hybrid_search(
            self,
            dense_query_vector: List[float],
            sparse_query_vector: Dict[int, float],
            tenant_id: str,
            limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """混合检索。"""
        # 注意：MilvusClient 的混合检索语法在不同小版本中有所演进
        # 这里展示标准的高级检索调用
        from pymilvus import AnnSearchRequest, RRFRanker

        reqs = [
            AnnSearchRequest([dense_query_vector], "dense_vector", {"metric_type": "IP", "params": {"ef": 64}}, limit),
            AnnSearchRequest([sparse_query_vector], "sparse_vector", {"metric_type": "BM25"}, limit)
        ]

        # 混合搜索目前在某些 client 版本中仍需通过 Collection 或特定的内部方法
        # 但推荐尝试 client.hybrid_search (如果 PyMilvus >= 2.4.0)
        res = self.client.hybrid_search(
            collection_name=self.config.collection,
            reqs=reqs,
            ranker=RRFRanker(),
            limit=limit,
            output_fields=["doc_id", "content", "metadata"]
        )
        return res[0] if res else []

    # === 工具方法 ===

    def delete_by_doc_id(self, doc_id: str):
        self.client.delete(
            collection_name=self.config.collection,
            filter=f'doc_id == "{doc_id}"'
        )

    def drop_collection(self):
        if self.client.has_collection(self.config.collection):
            self.client.drop_collection(self.config.collection)
            logger.info(f"Dropped collection: {self.config.collection}")

    def disconnect(self):
        self.client.close()

    @property
    def dim(self) -> int:
        return self._dim

    # === 内部方法 ===

    @staticmethod
    def _format_results(results) -> List[Dict[str, Any]]:
        """将 pymilvus 搜索结果转为统一的字典列表格式。"""
        formatted = []
        if results and len(results) > 0:
            for hit in results[0]:
                item = {"id": hit.id, "distance": hit.distance}
                for field_name in hit.fields:
                    try:
                        item[field_name] = hit.get(field_name)
                    except Exception:
                        pass
                formatted.append(item)
        return formatted


# === 全局访问 ===

def get_milvus_client() -> MilvusManager:
    return MilvusManager()
