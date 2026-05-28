"""Modal to pick a folder for log output."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Input, Static


class FolderModal(ModalScreen[Path | None]):
    DEFAULT_CSS = """
    FolderModal {
        align: center middle;
    }
    FolderModal > Vertical {
        background: $panel;
        border: round $primary;
        padding: 1 2;
        width: 80%;
        height: 80%;
    }
    FolderModal DirectoryTree { height: 1fr; }
    FolderModal Horizontal { height: auto; }
    """

    def __init__(self, current: Path) -> None:
        super().__init__()
        self._current = current.expanduser().resolve() if current else Path.home()

    def compose(self) -> ComposeResult:
        start = self._current if self._current.exists() else Path.home()
        yield Vertical(
            Static("Pick the log directory (selects highlighted folder, or type a path):"),
            DirectoryTree(str(start), id="tree"),
            Input(value=str(start), id="path-input"),
            Horizontal(
                Button("OK", id="ok", variant="primary"),
                Button("Cancel", id="cancel"),
            ),
        )

    def on_directory_tree_directory_selected(
        self, event: DirectoryTree.DirectorySelected
    ) -> None:
        self.query_one("#path-input", Input).value = str(event.path)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        raw = self.query_one("#path-input", Input).value.strip()
        if not raw:
            self.dismiss(None)
            return
        self.dismiss(Path(raw).expanduser())
