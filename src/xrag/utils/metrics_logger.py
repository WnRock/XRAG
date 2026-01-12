import time
from ..utils import get_module_logger
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = get_module_logger(__name__)


@dataclass
class MetricsLogger:
    """Tracks timing and token usage metrics for RAG operations."""

    index_build_time: float = 0.0
    classifier_times: List[float] = field(default_factory=list)
    critic_times: List[float] = field(default_factory=list)
    retrieval_times: List[float] = field(default_factory=list)
    generation_times: List[float] = field(default_factory=list)
    token_usages: List[int] = field(default_factory=list)

    _current_start: float = field(default=0.0, repr=False)

    def start_timer(self) -> None:
        self._current_start = time.perf_counter()

    def stop_timer(self) -> float:
        elapsed = time.perf_counter() - self._current_start
        return elapsed

    def log_index_build(self, elapsed: float) -> None:
        self.index_build_time = elapsed
        logger.info(f"[METRICS] Index build time: {elapsed:.4f}s")

    def log_classifier(self, elapsed: float) -> None:
        self.classifier_times.append(elapsed)
        logger.info(
            f"[METRICS] Classifier round {len(self.classifier_times)} time: {elapsed:.4f}s"
        )

    def log_critic(self, elapsed: float) -> None:
        self.critic_times.append(elapsed)
        logger.info(
            f"[METRICS] Critic round {len(self.critic_times)} time: {elapsed:.4f}s"
        )

    def log_retrieval(self, elapsed: float) -> None:
        self.retrieval_times.append(elapsed)
        logger.info(
            f"[METRICS] Retrieval round {len(self.retrieval_times)} time: {elapsed:.4f}s"
        )

    def log_generation(self, elapsed: float) -> None:
        self.generation_times.append(elapsed)
        logger.info(
            f"[METRICS] Generation round {len(self.generation_times)} time: {elapsed:.4f}s"
        )

    def log_tokens(self, token_count: int) -> None:
        self.token_usages.append(token_count)
        logger.info(
            f"[METRICS] Token usage for entry {len(self.token_usages)}: {token_count}"
        )

    def _calc_avg(self, values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _calc_total(self, values: List[float]) -> float:
        return sum(values)

    def get_classifier_avg(self) -> float:
        return self._calc_avg(self.classifier_times)

    def get_classifier_total(self) -> float:
        return self._calc_total(self.classifier_times)

    def get_critic_avg(self) -> float:
        return self._calc_avg(self.critic_times)

    def get_critic_total(self) -> float:
        return self._calc_total(self.critic_times)

    def get_retrieval_avg(self) -> float:
        return self._calc_avg(self.retrieval_times)

    def get_retrieval_total(self) -> float:
        return self._calc_total(self.retrieval_times)

    def get_generation_avg(self) -> float:
        return self._calc_avg(self.generation_times)

    def get_generation_total(self) -> float:
        return self._calc_total(self.generation_times)

    def get_token_avg(self) -> float:
        return self._calc_avg([float(t) for t in self.token_usages])

    def get_token_total(self) -> int:
        return sum(self.token_usages)

    def log_summary(self) -> None:
        logger.info("=" * 60)
        logger.info("[METRICS SUMMARY]")
        if self.index_build_time > 0:
            logger.info(f"  Index build time: {self.index_build_time:.4f}s")
        if self.classifier_times:
            logger.info(
                f"  Classifier - count: {len(self.classifier_times)}, total: {self.get_classifier_total():.4f}s, avg: {self.get_classifier_avg():.4f}s"
            )
        if self.critic_times:
            logger.info(
                f"  Critic - count: {len(self.critic_times)}, total: {self.get_critic_total():.4f}s, avg: {self.get_critic_avg():.4f}s"
            )
        if self.retrieval_times:
            logger.info(
                f"  Retrieval - count: {len(self.retrieval_times)}, total: {self.get_retrieval_total():.4f}s, avg: {self.get_retrieval_avg():.4f}s"
            )
        if self.generation_times:
            logger.info(
                f"  Generation - count: {len(self.generation_times)}, total: {self.get_generation_total():.4f}s, avg: {self.get_generation_avg():.4f}s"
            )
        if self.token_usages:
            logger.info(
                f"  Token usage - count: {len(self.token_usages)}, total: {self.get_token_total()}, avg: {self.get_token_avg():.2f}"
            )
        logger.info("=" * 60)

    def reset(self) -> None:
        self.index_build_time = 0.0
        self.classifier_times.clear()
        self.critic_times.clear()
        self.retrieval_times.clear()
        self.generation_times.clear()
        self.token_usages.clear()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index_build_time": self.index_build_time,
            "classifier_times": self.classifier_times.copy(),
            "classifier_total": self.get_classifier_total(),
            "classifier_avg": self.get_classifier_avg(),
            "critic_times": self.critic_times.copy(),
            "critic_total": self.get_critic_total(),
            "critic_avg": self.get_critic_avg(),
            "retrieval_times": self.retrieval_times.copy(),
            "retrieval_total": self.get_retrieval_total(),
            "retrieval_avg": self.get_retrieval_avg(),
            "generation_times": self.generation_times.copy(),
            "generation_total": self.get_generation_total(),
            "generation_avg": self.get_generation_avg(),
            "token_usages": self.token_usages.copy(),
            "token_total": self.get_token_total(),
            "token_avg": self.get_token_avg(),
        }


_global_metrics: Optional[MetricsLogger] = None


def get_metrics_logger() -> MetricsLogger:
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsLogger()
    return _global_metrics


def reset_metrics_logger() -> MetricsLogger:
    global _global_metrics
    _global_metrics = MetricsLogger()
    return _global_metrics
