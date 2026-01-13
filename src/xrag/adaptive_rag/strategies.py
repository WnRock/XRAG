from ..llms import get_llm
from ..config import Config
from .utils import load_prompt, extract_token_usage
from ..utils import get_metrics_logger
from llama_index.core import QueryBundle
from llama_index.core.query_engine import RetrieverQueryEngine
from ..retrievers.retriever import get_retriver, response_synthesizer


class SimpleResponse:
    def __init__(self, text: str, raw=None):
        self.response = text
        self.source_nodes = []
        self.raw = raw


class AdaptiveStrategy:
    """
    Base class for adaptive RAG strategies.
    """

    def execute(self, query: str, **kwargs) -> str:
        """
        Execute the strategy.

        Params:
            query (str): Input query
            **kwargs: Additional parameters

        Returns:
            Generated response
        """
        raise NotImplementedError


class DirectGenerationStrategy(AdaptiveStrategy):
    """
    Direct LLM generation without retrieval.
    """

    def __init__(self, llm=None):
        """
        Initialize direct generation strategy.

        Params:
            llm (optional): LLM instance
        """
        if llm is not None:
            self.llm = llm
        else:
            config = Config()
            self.llm = get_llm(config.llm)

    def execute(self, query: str, **kwargs) -> SimpleResponse:
        """
        Generate response directly using LLM.

        Params:
            query (str): Input query
            **kwargs: Additional parameters

        Returns:
            Generated response
        """
        metrics = get_metrics_logger()
        metrics.start_timer()
        for retry in range(3):
            try:
                resp = self.llm.complete(query)
                break
            except Exception:
                if retry == 2:
                    raise
        gen_time = metrics.stop_timer()
        metrics.log_generation(gen_time)

        token_count = extract_token_usage(resp)
        if token_count is not None:
            metrics.log_tokens(token_count)
        
        return SimpleResponse(resp.text, raw=resp)


class SingleRetrievalStrategy(AdaptiveStrategy):
    """
    Single-turn retrieval followed by generation.
    """

    def __init__(self, index, llm=None, retriever_type="Vector"):
        """
        Initialize single retrieval strategy.

        Params:
            index: Document index for retrieval
            llm (optional): LLM instance
            retriever_type (str, optional): Type of retriever to use
        """
        self.index = index
        self.config = Config()
        self.adaptive_rag_config = self.config.config.get("adaptive_rag", {})

        self.llm = llm if llm is not None else get_llm(self.config.llm)

        self.retriever_type = retriever_type or self.adaptive_rag_config.get(
            "retriever_type", "Vector"
        )

    def execute(self, query: str, **kwargs):
        """
        Execute single retrieval and generation.

        Params:
            query (str): Input query
            **kwargs: Additional parameters

        Returns:
            Generated response
        """
        metrics = get_metrics_logger()
        
        retriever = get_retriver(self.retriever_type, self.index, cfg=self.config)
        synthesizer = response_synthesizer(self.config.responce_synthsizer)
        
        metrics.start_timer()
        nodes = retriever.retrieve(query)
        retrieval_time = metrics.stop_timer()
        metrics.log_retrieval(retrieval_time)
        
        metrics.start_timer()
        result = synthesizer.synthesize(query, nodes)
        gen_time = metrics.stop_timer()
        metrics.log_generation(gen_time)
        
        token_count = extract_token_usage(result)
        if token_count is not None:
            metrics.log_tokens(token_count)
        
        return result


class IterativeRetrievalStrategy(AdaptiveStrategy):
    """
    Multi-turn iterative retrieval strategy.
    """

    def __init__(self, index, llm=None, max_iterations=3, retriever_type="Vector"):
        """
        Initialize iterative retrieval strategy.

        Params:
            index: Document index for retrieval
            llm (optional): LLM instance
            max_iterations (int, optional): Maximum number of retrieval iterations
            retriever_type (str, optional): Type of retriever to use
        """
        self.index = index
        self.config = Config()
        self.adaptive_rag_config = self.config.config.get("adaptive_rag", {})

        self.llm = llm if llm is not None else get_llm(self.config.llm)

        self.max_iterations = max_iterations or self.adaptive_rag_config.get(
            "max_iterations", 3
        )
        self.retriever_type = retriever_type or self.adaptive_rag_config.get(
            "retriever_type", "Vector"
        )

    def _is_information_sufficient(self, query: str, context: str):
        metrics = get_metrics_logger()
        sufficiency_prompt = load_prompt("information_sufficiency")

        prompt = sufficiency_prompt.format(query=query, context=context)
        
        metrics.start_timer()
        for retry in range(3):
            try:
                resp = self.llm.complete(prompt)
                break
            except Exception:
                if retry == 2:
                    raise
        critic_time = metrics.stop_timer()
        metrics.log_critic(critic_time)

        token_count = extract_token_usage(resp)
        if token_count is not None:
            metrics.log_tokens(token_count)
        
        text = resp.text.strip()
        decision = "YES" in text.upper()
        additional_info = text.replace("YES", "").replace("NO", "").strip()
        return decision, additional_info

    def execute(self, query: str, **kwargs):
        metrics = get_metrics_logger()
        retriever = get_retriver(self.retriever_type, self.index, cfg=self.config)
        all_context = []
        current_query = query
        for _ in range(self.max_iterations):
            metrics.start_timer()
            nodes = retriever.retrieve(QueryBundle(query_str=current_query))
            retrieval_time = metrics.stop_timer()
            metrics.log_retrieval(retrieval_time)
            
            ctx_texts = [n.node.get_content() for n in nodes]
            all_context.extend(ctx_texts)
            combined = "\n".join(all_context)
            sufficient, info = self._is_information_sufficient(query, combined)
            if sufficient:
                break
            current_query = f"{query}\nAdditional information needed: {info}"
        synthesizer = response_synthesizer(self.config.responce_synthsizer)
        combined = "\n".join(all_context)
        context_query = f"Query: {query}\n\nRelevant Context:\n{combined}"
        
        metrics.start_timer()
        result = synthesizer.synthesize(context_query, nodes)
        gen_time = metrics.stop_timer()
        metrics.log_generation(gen_time)
        
        token_count = extract_token_usage(result)
        if token_count is not None:
            metrics.log_tokens(token_count)
        
        return result
