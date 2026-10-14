"""Terminal progress for training and fitting.

The trainer and the deterministic-floor harness report through the small
:class:`Reporter` interface below. :class:`TerminalReporter` renders it: on a TTY as
one line that is rewritten in place, and when stdout is redirected (a log file,
``nohup``, a scheduler) as bounded plain-text lines, at most one every
``interval`` seconds plus one at every epoch and phase boundary. Rendering never
depends on run journals or on the output directory.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, TextIO


class Reporter:
    """No-op progress sink; every method may be overridden independently."""

    @contextmanager
    def phase(self, name: str, **fields: Any) -> Iterator[None]:
        yield

    def epoch_start(self, epoch: int, total_epochs: int, total_batches: int) -> None:
        pass

    def step(self, batch: int, total_batches: int, **fields: Any) -> None:
        pass

    def progress(self, label: str, done: int, total: int, **fields: Any) -> None:
        """A labelled counter for any long pass (evaluation, preprocessing)."""

    def validation(self, epoch: int, **fields: Any) -> None:
        pass

    def epoch_end(self, epoch: int, **fields: Any) -> None:
        pass

    def message(self, text: str) -> None:
        pass


def _duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _number(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        if value == 0:
            return "0"
        magnitude = abs(value)
        if magnitude >= 1000 or magnitude < 1e-3:
            return f"{value:.3e}"
        return f"{value:.4f}"
    return str(value)


#: "Last render" before anything was rendered. A monotonic clock can read less than
#: the interval (it starts at boot), so 0.0 would suppress the first update.
_NEVER_RENDERED = float("-inf")


class TerminalReporter(Reporter):
    """Render progress to a stream.

    ``mode`` is ``"auto"`` (bar on a TTY, plain otherwise), ``"bar"`` or ``"plain"``.
    In bar mode the current line is redrawn at most ten times a second and finished
    with a newline at every epoch or phase boundary, so a log captured from a TTY
    still reads line by line. In plain mode ``interval`` bounds the output volume.
    """

    def __init__(
        self, stream: TextIO | None = None, *, mode: str = "auto", interval: float = 5.0
    ) -> None:
        if mode not in {"auto", "bar", "plain"}:
            raise ValueError("progress mode must be auto, bar or plain")
        self.stream = stream if stream is not None else sys.stdout
        if mode == "auto":
            mode = "bar" if getattr(self.stream, "isatty", lambda: False)() else "plain"
        self.mode = mode
        self.interval = float(interval)
        self._line_open = False
        self._last_render = _NEVER_RENDERED
        self._phase_label = ""
        self._progress_label = ""
        self._progress_started = 0.0
        self._epoch = 0
        self._total_epochs = 0
        self._epoch_started = 0.0

    # -- low-level output ----------------------------------------------------

    def _write_line(self, text: str) -> None:
        self._close_line()
        self.stream.write(text + "\n")
        self.stream.flush()

    def _redraw(self, text: str, *, force: bool = False) -> None:
        now = time.monotonic()
        if self.mode == "bar":
            if not force and now - self._last_render < 0.1:
                return
            self.stream.write("\r\x1b[2K" + text)
            self.stream.flush()
            self._line_open = True
        else:
            if not force and now - self._last_render < self.interval:
                return
            self.stream.write(text + "\n")
            self.stream.flush()
        self._last_render = now

    def _close_line(self) -> None:
        if self._line_open:
            self.stream.write("\n")
            self.stream.flush()
            self._line_open = False

    # -- Reporter interface --------------------------------------------------

    @contextmanager
    def phase(self, name: str, **fields: Any) -> Iterator[None]:
        previous = self._phase_label
        self._phase_label = name
        started = time.monotonic()
        detail = " ".join(f"{key}={_number(value)}" for key, value in fields.items())
        self._write_line(f"[{name}] start" + (f" {detail}" if detail else ""))
        try:
            yield
        finally:
            self._write_line(f"[{name}] done in {_duration(time.monotonic() - started)}")
            self._phase_label = previous

    def epoch_start(self, epoch: int, total_epochs: int, total_batches: int) -> None:
        self._epoch, self._total_epochs = epoch, total_epochs
        self._epoch_started = time.monotonic()
        self._last_render = _NEVER_RENDERED

    def step(self, batch: int, total_batches: int, **fields: Any) -> None:
        self._counter(
            f"[train] epoch {self._epoch}/{self._total_epochs}", batch, total_batches,
            self._epoch_started, fields,
        )

    def progress(self, label: str, done: int, total: int, **fields: Any) -> None:
        if done == 1 or self._progress_label != label:
            self._progress_label = label
            self._progress_started = time.monotonic()
            self._last_render = _NEVER_RENDERED
        self._counter(f"[{label}]", done, total, self._progress_started, fields)

    def _counter(self, prefix: str, done: int, total: int, started: float,
                 fields: dict[str, Any]) -> None:
        elapsed = time.monotonic() - started
        eta = (elapsed / done) * (total - done) if done and total else None
        if self.mode == "bar" and total:
            width = 20
            filled = int(width * min(done, total) / total)
            bar = "|" + "#" * filled + "-" * (width - filled) + "| "
        else:
            bar = ""
        detail = " ".join(f"{key} {_number(value)}" for key, value in fields.items())
        text = (
            f"{prefix} {bar}{done}/{total or '?'} {detail} elapsed {_duration(elapsed)} "
            f"eta {_duration(eta)}"
        ).replace("  ", " ")
        self._redraw(text, force=done == total)

    def validation(self, epoch: int, **fields: Any) -> None:
        detail = " ".join(f"{key} {_number(value)}" for key, value in fields.items())
        self._write_line(f"[validation] epoch {epoch} {detail}")

    def epoch_end(self, epoch: int, **fields: Any) -> None:
        detail = " ".join(f"{key} {_number(value)}" for key, value in fields.items())
        self._write_line(
            f"[epoch] {epoch}/{self._total_epochs} {detail} "
            f"took {_duration(time.monotonic() - self._epoch_started)}"
        )

    def message(self, text: str) -> None:
        self._write_line(text)
