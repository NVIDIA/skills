# Synthetic user-facing options module submitted for a coding-guidelines review exercise.
import json

DEFAULTPOLICY = "lru"


class LlmCacheOptions:
    """options for host KV cache offload"""

    def __init__(self, maxTokens=None, evictionPolicy=DEFAULTPOLICY, hostCacheBytes=0):
        self.maxTokens = maxTokens
        self.evictionPolicy = evictionPolicy
        self.hostCacheBytes = hostCacheBytes


def LoadCacheOptions(path):
    try:
        with open(path) as f:
            raw = json.load(f)
        return LlmCacheOptions(**raw)
    except:
        return LlmCacheOptions()
