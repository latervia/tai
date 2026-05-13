from app.core.model.model_factory import ModelProvider, ModelFactory


class ModelDispatcher:
    def __init__(self, primary_provider: ModelProvider):
        self.primary_llm = ModelFactory.get_model(primary_provider)
        self.backup_llm = ModelFactory.get_model(ModelProvider.OLLAMA)  # 备用

    def chat(self, messages, use_backup=False):
        llm = self.backup_llm if use_backup else self.primary_llm
        try:
            # 这里可以统一处理 LangSmith 的 tags 或 metadata
            return llm.invoke(messages)
        except Exception as e:
            # 统一的错误日志处理
            print(f"Error calling {llm.model_name}: {e}")
            raise e
