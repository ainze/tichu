"""Patience-based early-stop helper for the value-baseline fit.

The streaming value-baseline fit (`fit_value_baseline_streaming`)
otherwise iterates the dataset to exhaustion. On the corpus that's
~9h for one Phase 2 pass even after the running-mean MSE has visibly
plateaued — wasting wall-clock and yielding no further fit.

This helper signals "stop" once `running_mse` has failed to improve
by at least `min_delta` across `patience` consecutive chunks. Pure
Python, no torch dependency, fully unit-testable.

Disabled by setting `patience <= 0` (the default in
`fit_value_baseline_streaming`). Opt-in only — existing runs are
byte-for-byte unaffected.
"""


class EarlyStop:
    """Track per-chunk `running_mse` and signal when to stop.

    Stops when `patience` consecutive chunks fail to improve on the
    best `running_mse` seen so far by at least `min_delta`. The
    counter resets whenever a chunk does improve.

    `patience <= 0` disables the stopper — `update` is a no-op and
    `should_stop` always returns False.
    """

    def __init__(self, *, patience: int, min_delta: float = 0.0) -> None:
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta!r}")
        self.patience = patience
        self.min_delta = min_delta
        self.best_mse: float = float("inf")
        self.bad_chunks: int = 0
        self._stopped: bool = False

    def update(self, running_mse: float) -> None:
        """Feed the latest `running_mse`. Once `patience` consecutive
        non-improving chunks have been seen, the stopper latches —
        `should_stop` returns True until a fresh instance is built.
        """
        if self.patience <= 0:
            return
        if self._stopped:
            return
        if running_mse < self.best_mse - self.min_delta:
            self.best_mse = running_mse
            self.bad_chunks = 0
        else:
            self.bad_chunks += 1
            if self.bad_chunks >= self.patience:
                self._stopped = True

    def should_stop(self) -> bool:
        return self._stopped
