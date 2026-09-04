"""Ground truth: what the subject was actually doing, set by hand while a recording runs.

``cls.csv`` and ``cls_vote.csv`` hold the model's opinion. Nothing in a recording says what the
activity really was, so a session can be plotted but not scored and not used as training data.
This module is the answer key: the operator presses a button, and every CSV row from that instant
carries that class name until the next press.

The state is a **step function of host-monotonic time**, stored as a list of transitions rather
than as "the current label" alone. Two consumers need different views of the same thing and both
are exact reads of that list:

* ``Recorder`` asks ``label_at(sample_t)`` -- which label was in force at the instant the device
  sent this sample. Per sample, not per drain batch: the drain wakes at ``DRAIN_HZ`` and a press
  lands between two wake-ups as often as not, so snapshotting once per batch would round every
  transition to the nearest 10 ms for no reason.
* The web payload asks ``spans(t_from)`` -- the labelled intervals overlapping the plot window, so
  the browser can tint them behind the traces.

Label strings are ``cls.model.CLASSES`` verbatim, which is what makes ``gt_label.csv`` join to
``cls.csv`` row for row with no mapping table in between. ``NO_LABEL`` is a *value*, not a gap: a
recording made with no button ever pressed is a column of "No Label", which is information, where
an absent file would only be an absent file.

Timestamps are ``time.monotonic()`` -- the same clock the ring buffer stamps samples with and the
same one ``/data`` reports as ``t``. Never wall clock: the recorder converts to time of day once,
from its own reference, so a label and the sample it describes cannot drift apart.
"""

from __future__ import annotations

import bisect
import threading
import time

from jetson_imu_tui.cls.model import CLASSES

# What an unlabelled row says. A literal value rather than an empty cell: empty means "the device
# had no reading" everywhere else in a recording, and "nobody had pressed a button yet" is a
# different and equally real statement.
NO_LABEL = "No Label"

# The classes a label may take. Re-exported from the model rather than re-listed: CLASSES is also
# the serial return-channel byte encoding, so a second copy that drifted would be silent.
LABELS: tuple[str, ...] = tuple(CLASSES)

# Slack added when two transitions land on the same monotonic reading, in seconds. The clock is
# coarse on Windows (~15.6 ms), so two fast presses can share a value; the list must stay strictly
# ascending for bisect to be well defined. Same trick as the sample nudge in serial_service.
_TIE_EPS = 1e-6


class GroundTruth:
    """The live ground-truth label, as a timeline of transitions.

    Thread-safe on its own lock, held only for the few list operations below. It never calls back
    into its caller, so it can be locked from inside ``ServerState._lock`` (which is not
    reentrant) without any ordering risk.

    The transition list is not pruned. It grows by button presses, not by elapsed time -- a
    session is tens of entries and a day of use is a few thousand, tens of kilobytes. Dropping old
    entries would make ``label_at`` quietly wrong for any earlier timestamp, which is a real cost
    for no real saving.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Parallel arrays rather than a list of tuples: bisect searches _t directly.
        self._t: list[float] = []       # monotonic instant of each change, strictly ascending
        self._label: list[str] = []     # label in force from _t[i] until _t[i + 1]
        self._current: str | None = None

    @property
    def current(self) -> str | None:
        """The label in force right now, or None while nothing is labelled.

        Returns: str (one of ``LABELS``) | None.
        """
        with self._lock:
            return self._current

    def set(self, label: str | None) -> str | None:
        """Start ``label`` now, or end the current one when it is None/empty.

        Args:    label: str | None. A name from ``LABELS``, or None / "" / ``NO_LABEL`` to clear.
        Returns: str | None, the label now in force (None when cleared).
        Raises:  ValueError if ``label`` is a non-empty string that is not one of ``LABELS``.

        Idempotent: setting the label already in force appends nothing, so a double-click cannot
        produce a zero-length span the charts would draw as a seam.
        """
        lab = NO_LABEL if label is None or label == "" else str(label)
        if lab != NO_LABEL and lab not in LABELS:
            raise ValueError("unknown ground-truth label: " + repr(label))
        with self._lock:
            if self._label and self._label[-1] == lab:
                return self._current
            t = time.monotonic()
            if self._t and t <= self._t[-1]:
                t = self._t[-1] + _TIE_EPS
            self._t.append(t)
            self._label.append(lab)
            self._current = None if lab == NO_LABEL else lab
            return self._current

    def toggle(self, label: str) -> str | None:
        """Start ``label``, or end it if it is already the one in force.

        Args:    label: str, a name from ``LABELS``.
        Returns: str | None, the label now in force.

        One button that starts and stops, matching ``/record`` and ``/zero``. The alternative -- a
        separate stop button per class -- is eleven more things to hit mid-experiment.
        """
        with self._lock:
            active = self._current
        return self.set(None) if active == label else self.set(label)

    def label_at(self, t: float) -> str:
        """The label in force at monotonic instant ``t``.

        Args:    t: float, a host-monotonic timestamp (a sample's own ``t``).
        Returns: str -- a name from ``LABELS``, or ``NO_LABEL`` before the first press.

        A press at exactly ``t`` counts as in force at ``t``: the operator pressing the button as
        the activity begins means that sample onward, not the one after.
        """
        with self._lock:
            i = bisect.bisect_right(self._t, t) - 1
            return self._label[i] if i >= 0 else NO_LABEL

    def spans(self, t_from: float) -> list[dict]:
        """Labelled intervals that have not finished before ``t_from``.

        Args:    t_from: float, monotonic; anything wholly older is dropped.
        Returns: list[dict], oldest first, each ``{"t0": float, "t1": float | None,
                 "label": str}``. ``t1`` is None for the interval still running.

        ``NO_LABEL`` stretches are **omitted rather than emitted as spans**. That absence is what
        makes the charts leave an unlabelled region alone, showing the page's own colours, and it
        keeps the payload to the handful of intervals a human can actually produce.
        """
        out: list[dict] = []
        with self._lock:
            n = len(self._t)
            for i in range(n):
                lab = self._label[i]
                if lab == NO_LABEL:
                    continue
                t1 = self._t[i + 1] if i + 1 < n else None
                if t1 is not None and t1 < t_from:
                    continue
                out.append({"t0": self._t[i], "t1": t1, "label": lab})
        return out

    def reset(self) -> None:
        """Forget every transition. Called when a recording starts or stops.

        A label belongs to the session it was pressed in: carrying one across a stop would let the
        next recording open on a stale activity nobody re-confirmed, and would leave a coloured
        band on screen for a session that is over.
        """
        with self._lock:
            self._t.clear()
            self._label.clear()
            self._current = None
