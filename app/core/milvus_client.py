import logging
from typing import List, Optional, Dict, Any
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    MilvusException, RRFRanker, AnnSearchRequest,
)

from app.core.config import settings

logger = logging.getLogger(__name__)


class MilvusClient:
    """
    Milvus 向量数据库客户端
    提供向量存储、检索、更新、删除等操作
    """

    _instance = None
    _collection: Optional[Collection] = None

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化 Milvus 客户端"""
        if self._initialized:
            return

        self.config = settings.milvus
        self._connect()
        self._init_collection()
        self._initialized = True

    def _connect(self):
        """连接到 Milvus 服务"""
        try:
            connections.connect(
                alias="default",
                host=self.config.host,
                port=self.config.port,
                user=self.config.user,
                password=self.config.password,
                db_name=self.config.database,
                pool_size=10,
            )
            logger.info(
                f"成功连接到 Milvus: {self.config.host}:{self.config.port}"
            )
        except MilvusException as e:
            logger.error(f"Milvus 连接失败: {e}")
            raise

    def _init_collection(self):
        """初始化或获取集合"""
        try:
            # 检查集合是否存在
            if connections.has_collection(self.config.collection, using="default"):
                self._collection = Collection(
                    self.config.collection, using="default"
                )
                logger.info(f"成功加载现有集合: {self.config.collection}")
            else:
                self._create_collection()
        except MilvusException as e:
            logger.error(f"初始化集合失败: {e}")
            raise

    def _create_collection(self):
        """创建新集合"""
        try:
            # 定义字段schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="chunk_index", dtype=DataType.INT32),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
                FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
                FieldSchema(name="metadata", dtype=DataType.JSON),
                # 根据租户自动分区 大幅提升性能
                FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_len=128, is_partition_key=True),
            ]

            # 创建collection schema
            schema = CollectionSchema(
                fields=fields,
                description="知识库向量集合",
                enable_dynamic_field=False,
            )

            # 创建collection
            self._collection = Collection(
                name=self.config.collection,
                schema=schema,
                using="default",
            )

            # 创建索引以加速查询
            self._create_index()

            logger.info(f"成功创建新集合: {self.config.collection}")

        except MilvusException as e:
            logger.error(f"创建集合失败: {e}")
            raise

    def _create_index(self):
        """为向量字段创建索引"""
        try:
            # 稠密向量索引 (Dense Vector) 策略
            # 场景 A: 数据量 < 100万，追求极速且内存充足 -> 选 HNSW
            # 场景 B: 数据量 > 1000万，内存有限 -> 选 IVF_PQ (压缩存储)
            # 场景 C: 通用开发、中等规模 -> 选 IVF_FLAT
            dense_index_params = {
                "index_type": "HNSW",  # 策略：HNSW 是目前综合性能最强的索引, 支持极高并发检索
                "metric_type": "IP",  # 策略：L2(欧氏距离) 适合图片/无偏数据；IP(内积) 适合文本 Embedding
                "params": {
                    "M": 16,  # 策略：每个节点的邻居数，范围 4-64，越大精度越高但内存消耗越多
                    "efConstruction": 200  # 策略：构建时搜索范围，越大索引质量越高，构建越慢
                },
            }

            self._collection.create_index(
                field_name="dense_vector",
                index_params=dense_index_params,
            )

            # 稀疏向量索引 (Sparse Vector) 策略
            # 稀疏向量主要用于关键词匹配（BM25 逻辑）
            sparse_index_params = {
                "index_type": "SPARSE_INVERTED_INDEX",  # 策略：标准倒排索引，适合绝大多数稀疏匹配场景
                "metric_type": "BM25",  # IP(内积)使用点积衡量相似性 / BM25通常用于全文搜索，侧重于文本相似性
                "params": {
                    "drop_ratio_build": 0.2  # 策略：忽略权重最低的 20% 词，能极大减小索引体积并加速，且几乎不损精度
                },
            }

            # self._collection.create_index(
            #     field_name="sparse_vector",
            #     index_params=sparse_index_params,
            # )

            # 3. 加载到内存（必须步骤）
            self._collection.load()
            logger.info("混合索引创建并加载成功")

            logger.info("成功创建向量索引")

        except MilvusException as e:
            logger.error(f"创建索引失败: {e}")
            # 如果索引已存在，不抛异常
            if "index already exists" not in str(e):
                raise

    # === 操作 ===

    def insert_vectors(
            self,
            doc_ids: List[str],
            chunk_indices: List[int],
            documents: List[str],
            # sparse_vectors: List[Dict[int, float]],
            dense_vectors: List[List[float]],
            metadata_list: Optional[List[Dict[str, Any]]] = None,
            tenant_ids: List[str] = None,
    ) -> List[int]:
        """
        将混合向量数据插入 Milvus
        """
        # 最佳实践：在组织数据前进行长度检查，防止插入失败
        data_length = len(doc_ids)
        if not (len(chunk_indices) == len(documents) == len(dense_vectors) == data_length):
            raise ValueError("所有输入列表的长度必须一致")

        # 组织成 Milvus 要求的列表格式（按字段顺序）
        data = [
            doc_ids,
            chunk_indices,
            documents,
            dense_vectors,
            metadata_list if metadata_list else [{} for _ in range(data_length)],
            tenant_ids
        ]

        try:
            mr = self._collection.insert(data)
            # 返回插入数据的自增 ID 列表
            return mr.primary_keys
        except Exception as e:
            logger.error(f"插入数据到 Milvus 失败: {e}")
            raise

    def search(
            self,
            query_vector: List[float],
            limit: int = 10,
            output_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        向量相似度搜索

        Args:
            query_vector: 查询向量
            limit: 返回结果数量
            output_fields: 输出字段列表

        Returns:
            搜索结果列表
        """
        if output_fields is None:
            output_fields = ["doc_id", "chunk_index", "content", "metadata"]

        try:
            self._collection.load()

            search_params = {
                "metric_type": "IP",
                "params": {"nprobe": 10},
            }

            results = self._collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=limit,
                output_fields=output_fields,
            )

            # 格式化返回结果
            formatted_results = []
            if results and len(results) > 0:
                for hit in results[0]:
                    result_dict = {
                        "id": hit.id,
                        "distance": hit.distance,
                    }
                    if output_fields:
                        for field in output_fields:
                            if field in hit.entity:
                                result_dict[field] = hit.entity[field]
                    formatted_results.append(result_dict)

            logger.info(f"搜索返回 {len(formatted_results)} 条结果")
            return formatted_results

        except MilvusException as e:
            logger.error(f"搜索失败: {e}")
            raise

    def hybrid_search(self, dense_query_vector, sparse_query_vector, tenant_id, limit=100):
        # 1. 稠密向量查询请求
        dense_req = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"ef": 64}},  # 对应 HNSW 的参数
            limit=limit
        )

        # 2. 稀疏向量查询请求
        sparse_req = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field="sparse_vector",
            param={"metric_type": "BM25", "params": {"drop_ratio_search": 0.2}},
            limit=limit
        )

        # 混合检索与重排序 (Rerank)
        reranker = RRFRanker(k=60)  # WeightedRanker

        results = self._collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=reranker,
            limit=limit,
            # 核心：必须带上租户过滤，否则 partition key 就白设了
            partition_names=[tenant_id],  # 直接指定分区，性能最佳
            output_fields=["doc_id", "chunk_index", "content", "metadata"]
        )
        return results[0]

    def batch_search(
            self,
            query_vectors: List[List[float]],
            limit: int = 10,
            output_fields: Optional[List[str]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        批量搜索

        Args:
            query_vectors: 查询向量列表
            limit: 每次搜索返回结果数量
            output_fields: 输出字段列表

        Returns:
            搜索结果列表的列表
        """
        if output_fields is None:
            output_fields = ["doc_id", "chunk_index", "content", "metadata"]

        try:
            self._collection.load()

            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10},
            }

            results = self._collection.search(
                data=query_vectors,
                anns_field="vector",
                param=search_params,
                limit=limit,
                output_fields=output_fields,
            )

            # 格式化返回结果
            formatted_results = []
            for result_group in results:
                group_results = []
                for hit in result_group:
                    result_dict = {
                        "id": hit.id,
                        "distance": hit.distance,
                    }
                    if output_fields:
                        for field in output_fields:
                            if field in hit.entity:
                                result_dict[field] = hit.entity[field]
                    group_results.append(result_dict)
                formatted_results.append(group_results)

            logger.info(
                f"批量搜索完成，共处理 {len(query_vectors)} 个查询"
            )
            return formatted_results

        except MilvusException as e:
            logger.error(f"批量搜索失败: {e}")
            raise

    def delete_by_doc_id(self, doc_id: str) -> int:
        """
        删除指定文档ID的所有向量

        Args:
            doc_id: 文档ID

        Returns:
            删除的记录数
        """
        try:
            expr = f'doc_id == "{doc_id}"'
            result = self._collection.delete(expr)
            self._collection.flush()

            logger.info(f"成功删除文档 {doc_id} 的所有向量")
            return result.delete_count

        except MilvusException as e:
            logger.error(f"删除失败: {e}")
            raise

    def update_vectors(
            self,
            vectors: List[List[float]],
            documents: List[str],
            doc_ids: List[str],
            chunk_indices: List[int],
            metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """
        更新向量数据（先删除后插入）

        Args:
            vectors: 新的向量列表
            documents: 新的文档内容列表
            doc_ids: 文档ID列表
            chunk_indices: 文本块索引列表
            metadata_list: 元数据列表 (可选)

        Returns:
            插入的记录数
        """
        try:
            # 删除旧数据
            unique_doc_ids = set(doc_ids)
            for doc_id in unique_doc_ids:
                self.delete_by_doc_id(doc_id)

            # 插入新数据
            primary_keys = self.insert_vectors(
                vectors, documents, doc_ids, chunk_indices, metadata_list
            )

            logger.info(f"成功更新 {len(primary_keys)} 条数据")
            return len(primary_keys)

        except MilvusException as e:
            logger.error(f"更新失败: {e}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息

        Returns:
            包含集合统计信息的字典
        """
        try:
            num_entities = self._collection.num_entities
            return {
                "collection_name": self.config.collection,
                "num_entities": num_entities,
                "num_shards": self._collection.num_shards,
            }
        except MilvusException as e:
            logger.error(f"获取统计信息失败: {e}")
            raise

    def drop_collection(self):
        """删除集合"""
        try:
            if connections.has_collection(
                    self.config.collection, using="default"
            ):
                Collection.drop(self.config.collection, using="default")
                logger.info(f"成功删除集合: {self.config.collection}")
        except MilvusException as e:
            logger.error(f"删除集合失败: {e}")
            raise

    def disconnect(self):
        """断开连接"""
        try:
            connections.disconnect(alias="default")
            logger.info("已断开 Milvus 连接")
        except MilvusException as e:
            logger.error(f"断开连接失败: {e}")


# 全局单例实例
milvus_client: Optional[MilvusClient] = None


def get_milvus_client() -> MilvusClient:
    """
    获取 Milvus 客户端单例

    Returns:
        MilvusClient 实例
    """
    global milvus_client
    if milvus_client is None:
        milvus_client = MilvusClient()
    return milvus_client
