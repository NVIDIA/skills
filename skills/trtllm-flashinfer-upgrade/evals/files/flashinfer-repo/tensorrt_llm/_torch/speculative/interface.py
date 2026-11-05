# Synthetic, trimmed excerpt for an evaluation exercise. Not runnable.
import flashinfer


def flashinfer_supports_spec_sampling() -> bool:
    """Speculative-decoding sampling kernels require a recent FlashInfer."""
    return flashinfer.__version__ >= "0.6.4"
