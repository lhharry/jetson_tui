"""SoftVoter — temporal aggregation of frame-level predictions into stable decisions.

A frame-level prediction is noisy: sensor noise, transient poses and genuinely ambiguous points
in the gait cycle all produce single-frame misclassifications. A downstream consumer (a controller
adjusting assistance) neither needs nor should react to every raw frame — doing so makes its
behaviour jittery. So predictions are collected over a short window and distilled into one robust
decision, trading a little latency for a large gain in stability. Same principle as the majority
vote in the exosuit literature, except the windows here are fixed-length in *frames*: no gait
phase or step segmentation signal is available, the prediction stream is all there is.

At the shipped settings (10 Hz in, ``window = emit_every = 5``) that is one decision per 500 ms.

**Soft, not hard.** Probability vectors are averaged element-wise and the argmax of the average
is the decision, rather than counting predicted labels. Two reasons this matters at five votes:

* Hard votes tie. Over 3+ classes five votes can split 2:2:1, forcing an invented tie-break;
  averaged floats are never exactly equal in practice.
* Hard votes discard confidence. A hesitant 51% frame counts the same as a 99% one, so two
  low-confidence errors outvote one confident correct frame. Averaging lets the confident frame win.

Standalone by design: no numpy, no torch, no threading, no I/O, and no knowledge of the sensor
source or the web server. ``ClsService`` injects an instance and only ever calls ``push`` and
``reset``, so this can be swapped for another aggregation scheme or tested on its own.

    voter = SoftVoter(window=5, emit_every=5)
    for probs in stream:                 # 10 Hz
        d = voter.push(probs)
        if d is not None:                # 2 Hz
            send(d.index)
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    """One aggregated decision. ``probs`` is the full averaged distribution, so a consumer that
    wants more than the argmax (a confidence gate, an offline evaluation) has everything."""

    index: int          # argmax of the averaged distribution, or the held class
    confidence: float   # averaged probability of the class actually emitted
    probs: list[float]  # the averaged distribution
    n_frames: int       # frames that went into it
    held: bool          # True if hysteresis suppressed a switch this window


class SoftVoter:
    """Averages the last ``window`` probability vectors and emits a decision every
    ``emit_every`` pushes.

    The two integers cover both regimes the design calls for, with no separate code path — the
    buffer is a bounded deque, so it rolls over on its own:

    * ``window == emit_every`` — **tumbling**: each decision is built from an entirely fresh set
      of frames (the baseline: 5 and 5).
    * ``window > emit_every`` — **sliding**: decisions still come at the same rate but average a
      longer history, which steadies the output across class boundaries at the cost of latency
      (e.g. 10 and 5).
    * ``window == emit_every == 1`` — exact passthrough, i.e. aggregation disabled.

    (``emit_every > window`` is allowed and means "average the most recent ``window`` frames every
    ``emit_every`` frames", skipping the ones in between.)
    """

    def __init__(self, window: int = 5, emit_every: int = 5, hysteresis: int = 0) -> None:
        self._window = max(1, int(window))
        self._emit_every = max(1, int(emit_every))
        # 0 and 1 both mean "switch as soon as the vote says so"; n > 1 requires n consecutive
        # windows to agree on the new class before the output follows.
        self._hysteresis = max(0, int(hysteresis))

        self._buf: deque[list[float]] = deque(maxlen=self._window)
        self._since_emit = 0
        self._committed: int | None = None   # class currently being emitted
        self._pending: int | None = None     # challenger accumulating consecutive wins
        self._streak = 0

    @property
    def config(self) -> dict[str, int]:
        """The three knobs, for ``/cls`` and the UI tooltip."""
        return {
            "window": self._window,
            "emit_every": self._emit_every,
            "hysteresis": self._hysteresis,
        }

    def reset(self) -> None:
        """Drop all state, including the committed class.

        Called on any discontinuity in the prediction stream (sensor stall, CLS pause, a source
        switch). Clearing ``_committed`` too means the first decision after a reset commits
        immediately instead of paying the hysteresis delay against a stale class."""
        self._buf.clear()
        self._since_emit = 0
        self._committed = None
        self._pending = None
        self._streak = 0

    def push(self, probs: Sequence[float]) -> Decision | None:
        """Feed one frame's probability distribution.

        Returns a ``Decision`` when one is due, else None. A partial buffer never produces a
        decision, so the first one waits for ``window`` frames however long that takes."""
        vec = [float(p) for p in probs]
        if self._buf and len(vec) != len(self._buf[0]):
            self.reset()  # class count changed under us — never average ragged vectors
        self._buf.append(vec)
        self._since_emit += 1
        if self._since_emit < self._emit_every or len(self._buf) < self._window:
            return None
        self._since_emit = 0
        return self._decide()

    def _decide(self) -> Decision:
        n = len(self._buf)
        avg = [sum(col) / n for col in zip(*self._buf)]
        # First index wins a tie, matching np.argmax; exact ties are essentially unreachable
        # with averaged floats, which is the point of soft voting.
        cand = max(range(len(avg)), key=avg.__getitem__)

        held = False
        if self._committed is None or self._hysteresis <= 1 or cand == self._committed:
            # No hysteresis, nothing committed yet, or the vote agrees — follow it.
            self._committed = cand
            self._pending = None
            self._streak = 0
        else:
            # A challenger must win `hysteresis` *consecutive* windows. Any window that agrees
            # with the committed class lands in the branch above and clears the streak, so an
            # alternating A/B flicker never switches the output.
            self._streak = self._streak + 1 if cand == self._pending else 1
            self._pending = cand
            if self._streak >= self._hysteresis:
                self._committed = cand
                self._pending = None
                self._streak = 0
            else:
                held = True

        idx = self._committed
        return Decision(
            index=idx,
            confidence=avg[idx],
            probs=avg,
            n_frames=n,
            held=held,
        )
