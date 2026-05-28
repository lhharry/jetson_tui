"""Modal to set the recording frequency."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.validation import Integer
from textual.widgets import Button, Input, Static


class FrequencyModal(ModalScreen[int | None]):
    DEFAULT_CSS = """
    FrequencyModal {
        align: center middle;
    }
    FrequencyModal > Vertical {
        background: $panel;
        border: round $primary;
        padding: 1 2;
        width: 50;
        height: auto;
    }
    """

    def __init__(self, current: int) -> None:
        super().__init__()
        self._current = current

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("Recording frequency (1-200 Hz):"),
            Input(
                value=str(self._current),
                id="freq-input",
                validators=[Integer(minimum=1, maximum=200)],
            ),
            Static("", id="freq-error"),
            Button("OK", id="ok", variant="primary"),
            Button("Cancel", id="cancel"),
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        self._submit()

    def on_input_submitted(self, _event: Input.Submitted) -> None:
        self._submit()

    def _submit(self) -> None:
        inp = self.query_one("#freq-input", Input)
        result = inp.validate(inp.value)
        if result is not None and not result.is_valid:
            self.query_one("#freq-error", Static).update("Enter an integer between 1 and 200")
            return
        try:
            self.dismiss(int(inp.value))
        except ValueError:
            self.query_one("#freq-error", Static).update("Enter an integer between 1 and 200")
