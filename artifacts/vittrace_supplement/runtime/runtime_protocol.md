# Runtime protocol

These values are **post-cache projection/scoring overhead** only; they exclude rendering, model loading, encoder inference, cache creation, and disk loading. The table covers all 488 common-valid series using their immutable one-pass runtime transactions. Each series supplies one observation per arm. This is not a repeated, interleaved warm-cache microbenchmark, and the package does not claim end-to-end latency or fixed-thread causal timing.
