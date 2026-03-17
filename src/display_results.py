import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import (QAction, QColor, QKeySequence, QShortcut, QTextCharFormat, QTextCursor, QTextDocument)
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton, QScrollArea, QTextBrowser, QTextEdit, QVBoxLayout, QWidget)


class EventsManager:
    def __init__(self, scored_events_file: str | Path):
        self.scored_events_file = Path(scored_events_file)
        self.events_data = self._load_events()

    def _load_events(self) -> Dict[str, List[Dict[str, Any]]]:
      """
      Read a json file and:
      1. Scan these arrays if present:
        - incr_sentences
        - incr_lines
        - incr_paragraphs
        - incr_chunks
      2. For each entry like:
          {"text": "...", "events": [...]}
        if it has at least one event, extract each event and attach:
          "matched_text": <entry text> | None
      3. Group all extracted events by event["chemical"].

      Returns:
          {
              "CHEMICAL_A": [event1, event2, ...],
              "CHEMICAL_B": [event3, event4, ...],
              ...
          }
      """
      with self.scored_events_file.open("r", encoding="utf-8") as f:
          data = json.load(f)

      sections = [
          "incr_sentences",
          "incr_lines",
          "incr_paragraphs",
          "incr_chunks",
      ]

      all_events: List[Dict[str, Any]] = []

      for section in sections:
          found_in_section = 0
          entries = data.get(section, [])
          if not isinstance(entries, list):
              continue

          for entry in entries:
              if not isinstance(entry, dict):
                  continue

              text = entry.get("text", "")
              events = entry.get("events", [])

              if not events or not isinstance(events, list):
                  continue

              for event in events:
                  if not isinstance(event, dict):
                      continue

                  extracted_event = {
                      **event,
                      "matched_text": text,
                  }
                  all_events.append(extracted_event)
                  found_in_section += 1

          print(f"Found {found_in_section} events in section {section}")

      events_by_chemical: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

      for event in all_events:
          chemical = event.get("chemical")
          if chemical is None:
              continue
          events_by_chemical[str(chemical)].append(event)

      unmatched_events = data.get("unmatched_events_incr", [])
      found_unmatched = len(unmatched_events) if isinstance(unmatched_events, list) else 0
      print(f"Remaining {found_unmatched} unmatched events")
      if isinstance(unmatched_events, list):
          for event in unmatched_events:
              if not isinstance(event, dict):
                  continue

              chemical = event.get("chemical")
              if chemical is None:
                  event.setdefault("chemical", "UNKNOWN")
                  extracted_event = {
                      **event,
                      "matched_text": None,
                  }
              else:
                extracted_event = {
                    **event,
                    "matched_text": None,
                }

              events_by_chemical[str(chemical)].append(extracted_event)

      return dict(events_by_chemical)
        
class CollapsibleChemicalBox(QFrame):
    def __init__(self, chemical_name: str, events: list[dict], on_event_click, parent=None):
        super().__init__(parent)
        self.chemical_name = chemical_name
        self.events = events
        self.on_event_click = on_event_click
        self.expanded = True

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("chemicalBox")

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Header
        self.header_button = QPushButton()
        self.header_button.setCheckable(False)
        self.header_button.clicked.connect(self.toggle)

        self.header_button.setStyleSheet(
            """
            QPushButton {
                text-align: left;
                padding: 12px 14px;
                font-size: 15px;
                font-weight: 600;
                color: black;
                background: #eef2f6;
                border: none;
            }
            QPushButton:hover {
                background: #e3e8ef;
            }
            """
        )

        # Content container
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(10, 10, 10, 10)
        self.content_layout.setSpacing(8)

        for ev in events:
            event_box = ClickableEventBox(ev)
            event_box.clicked.connect(self.on_event_click)
            self.content_layout.addWidget(event_box)

        self.content_layout.addStretch()

        root_layout.addWidget(self.header_button)
        root_layout.addWidget(self.content_widget)

        self.setStyleSheet(
            """
            QFrame#chemicalBox {
                background: #f8fafc;
                border: 1px solid #cbd5e1;
                border-radius: 10px;
            }
            """
        )

        self._refresh_header()

    def toggle(self):
        self.expanded = not self.expanded
        self.content_widget.setVisible(self.expanded)
        self._refresh_header()

    def _refresh_header(self):
        arrow = "▲" if self.expanded else "▼"
        self.header_button.setText(f"{self.chemical_name}    {arrow}")

class ClickableEventBox(QFrame):
    clicked = Signal(dict)

    def __init__(self, event_data: dict, parent=None):
        super().__init__(parent)
        self.event_data = event_data
        self.has_matched_text = self._has_matched_text(event_data)

        self.setCursor(
          Qt.CursorShape.PointingHandCursor
          if self.has_matched_text
          else Qt.CursorShape.ArrowCursor
        )
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("eventBox")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        # Top row: event_type + short description
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(8)

        event_type = str(event_data.get("event_type", "")).upper()
        short_desc = str(event_data.get("event_description_short", ""))
        long_desc = str(event_data.get("event_description_long", ""))
        score = float(event_data.get("score", 0.0))
        chemical_found = bool(event_data.get("chemical_found", False))

        type_label = QLabel(event_type)
        type_label.setTextFormat(Qt.TextFormat.RichText)
        type_label.setText(f"<b>{event_type}</b>")
        type_label.setStyleSheet(f"color: {self._event_type_color(event_type)};")

        short_label = QLabel(f"<b>{short_desc}</b>")
        short_label.setTextFormat(Qt.TextFormat.RichText)
        short_label.setWordWrap(True)
        short_label.setStyleSheet("color: black;")

        top_row.addWidget(type_label, 0)
        top_row.addWidget(short_label, 1)

        long_label = QLabel(long_desc)
        long_label.setWordWrap(True)
        long_label.setStyleSheet("color: #666666;")

        score_color = self._score_color(score)
        chemical_found_text = "True" if chemical_found else "False"
        chemical_found_color = "#1b7f3b" if chemical_found else "#b42318"

        score_line = QLabel(
            f'Score: <span style="color:{score_color}; font-weight:600;">{score:.1f}%</span> '
            f'- Chemical found in text: <span style="color:{chemical_found_color}; font-weight:600;">{chemical_found_text}</span>'
        )
        score_line.setTextFormat(Qt.TextFormat.RichText)
        score_line.setWordWrap(True)
        score_line.setStyleSheet("color: black;")

        layout.addLayout(top_row)
        layout.addWidget(long_label)
        layout.addWidget(score_line)

        if not self.has_matched_text:
            missing_label = QLabel("No matched text available")
            missing_label.setStyleSheet("color: #991b1b; font-style: italic;")
            layout.addWidget(missing_label)

        self.setStyleSheet(self._build_stylesheet())

    @staticmethod
    def _has_matched_text(event_data: dict) -> bool:
      value = event_data.get("matched_text")
      return isinstance(value, str) and value.strip() != ""

    def _event_type_color(self, event_type: str) -> str:
        if event_type == "MIE":
            return "#166534"   # dark green
        if event_type == "KE":
            return "#a16207"   # dark yellow
        if event_type == "AO":
            return "#991b1b"   # dark red
        return "#111111"

    def _score_color(self, score: float) -> str:
        if score > 75:
            return "#15803d"
        if score > 50:
            return "#a16207"
        return "#b91c1c"

    def _build_stylesheet(self) -> str:
        if self.has_matched_text:
            return """
            QFrame#eventBox {
                background: white;
                border: 1px solid #d0d5dd;
                border-radius: 8px;
            }
            QFrame#eventBox:hover {
                background: #f9fafb;
                border: 1px solid #98a2b3;
            }
            """
        return """
        QFrame#eventBox {
            background: #fef2f2;
            border: 1px solid #fecaca;
            border-radius: 8px;
        }
        """

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.has_matched_text:
            self.clicked.emit(self.event_data)
        super().mousePressEvent(event)

class MarkdownViewer(QTextBrowser):
    """
    Read-only markdown viewer with:
    - markdown rendering
    - showMatch(sentence): find one sentence, scroll to center, strong highlight
    - search_text(query): highlight all matches
    - next_result / previous_result navigation
    """

    def __init__(self, markdown_path: str, parent=None):
        super().__init__(parent)
        self.markdown_path = Path(markdown_path)

        self.setReadOnly(True)
        self.setOpenExternalLinks(True)
        self.setFrameShape(QFrame.Shape.NoFrame)

        self.search_results: list[QTextCursor] = []
        self.current_search_index: int = -1
        self.current_query: str = ""

        self._load_markdown()

    def _load_markdown(self):
        if self.markdown_path.exists():
            content = self.markdown_path.read_text(encoding="utf-8")
        else:
            content = (
                "# Demo Markdown\n\n"
                "This pane renders markdown, not raw plain text.\n\n"
                "## First section\n\n"
                "Here is a line containing test so the button can find it.\n\n"
                "Another sentence with test appears here too.\n\n"
                "## Second section\n\n"
                "Searching test should find multiple matches.\n"
            )
            self.markdown_path.write_text(content, encoding="utf-8")

        self.setMarkdown(content)
        self.document().setDefaultStyleSheet(
            """
            body { color: black; }
            h1 { color: black; }
            h2 { color: black; }
            p  { color: black; }
            """
        )

    def clear_search(self):
        self.search_results = []
        self.current_search_index = -1
        self.current_query = ""
        self.setExtraSelections([])

    def showMatch(self, sentence: str) -> bool:
        """
        External jump-to-text function.
        Clears any previous search highlights and highlights only this match.
        """
        self.clear_search()

        if not sentence:
            return False

        cursor = self.document().find(sentence)
        if cursor.isNull():
            return False

        self.search_results = [QTextCursor(cursor)]
        self.current_search_index = 0
        self.current_query = sentence
        self._apply_highlights(single_mode=True)
        self.setTextCursor(cursor)
        QTimer.singleShot(0, lambda: self._scroll_cursor_to_center(cursor))
        return True

    def search_text(self, query: str) -> int:
        """
        Find all occurrences of query in the rendered text.
        Returns number of matches.
        """
        self.clear_search()

        query = query.strip()
        if not query:
            return 0

        self.current_query = query

        cursor = QTextCursor(self.document())
        found_results: list[QTextCursor] = []

        while True:
            cursor = self.document().find(query, cursor, QTextDocument.FindFlag(0))
            if cursor.isNull():
                break
            found_results.append(QTextCursor(cursor))

        self.search_results = found_results

        if self.search_results:
            self.current_search_index = 0
            self._apply_highlights()
            current = self.search_results[self.current_search_index]
            self.setTextCursor(current)
            QTimer.singleShot(0, lambda: self._scroll_cursor_to_center(current))
        else:
            self.setExtraSelections([])

        return len(self.search_results)

    def next_result(self) -> bool:
        if not self.search_results:
            return False

        self.current_search_index = (self.current_search_index + 1) % len(self.search_results)
        self._focus_current_result()
        return True

    def previous_result(self) -> bool:
        if not self.search_results:
            return False

        self.current_search_index = (self.current_search_index - 1) % len(self.search_results)
        self._focus_current_result()
        return True

    def get_search_status(self) -> tuple[int, int]:
        """
        Returns (current_index_1_based, total_results)
        """
        total = len(self.search_results)
        if total == 0 or self.current_search_index < 0:
            return 0, 0
        return self.current_search_index + 1, total

    def _focus_current_result(self):
        self._apply_highlights()
        current = self.search_results[self.current_search_index]
        self.setTextCursor(current)
        QTimer.singleShot(0, lambda: self._scroll_cursor_to_center(current))

    def _apply_highlights(self, single_mode: bool = False):
        """
        Draw visual-only highlights using ExtraSelections.
        - single_mode=True: only current result highlighted strongly
        - normal search: all results highlighted lightly, current one strongly
        """
        selections: list[QTextEdit.ExtraSelection] = []

        if not self.search_results:
            self.setExtraSelections([])
            return

        all_match_bg = QColor("#fff3a3")
        current_match_bg = QColor("#ffbf69")

        if not single_mode:
            for cursor in self.search_results:
                sel = QTextEdit.ExtraSelection()
                sel.cursor = QTextCursor(cursor)
                fmt = QTextCharFormat()
                fmt.setBackground(all_match_bg)
                fmt.setForeground(QColor("black"))
                sel.format = fmt
                selections.append(sel)

        if 0 <= self.current_search_index < len(self.search_results):
            current_sel = QTextEdit.ExtraSelection()
            current_sel.cursor = QTextCursor(self.search_results[self.current_search_index])
            fmt = QTextCharFormat()
            fmt.setBackground(current_match_bg)
            fmt.setForeground(QColor("black"))
            current_sel.format = fmt
            selections.append(current_sel)

        self.setExtraSelections(selections)

    def _scroll_cursor_to_center(self, cursor: QTextCursor):
        rect = self.cursorRect(cursor)
        scrollbar = self.verticalScrollBar()
        target_value = scrollbar.value() + rect.center().y() - (self.viewport().height() // 2)
        scrollbar.setValue(target_value)

class FileCard(QFrame):
    clicked = Signal(object)

    def __init__(self, file_info: dict, parent=None):
        super().__init__(parent)
        self.file_info = file_info
        self.setObjectName("fileCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        title = QLabel(f"<b>{file_info['filename']}</b>")
        title.setTextFormat(Qt.TextFormat.RichText)
        title.setStyleSheet("color: black; font-size: 15px;")

        md_label = QLabel(f"Markdown file path: {file_info['markdown_path']}")
        md_label.setWordWrap(True)
        md_label.setStyleSheet("color: #555;")

        ev_label = QLabel(f"Extracted events file path: {file_info['events_path']}")
        ev_label.setWordWrap(True)
        ev_label.setStyleSheet("color: #555;")

        layout.addWidget(title)
        layout.addWidget(md_label)
        layout.addWidget(ev_label)

        self.setStyleSheet(
            """
            QFrame#fileCard {
                background: white;
                border: 1px solid #d0d5dd;
                border-radius: 10px;
            }
            QFrame#fileCard:hover {
                background: #f8fafc;
                border: 1px solid #98a2b3;
            }
            """
        )

    def mousePressEvent(self, mouse_event):
        if mouse_event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.file_info)
        super().mousePressEvent(mouse_event)

class FileSelectionPage(QWidget):
    file_selected = Signal(object)

    def __init__(self, matched_files: list[dict], parent=None):
        super().__init__(parent)
        self.matched_files = matched_files

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(24, 24, 24, 24)
        root_layout.setSpacing(16)

        title = QLabel("Select file to check extractions")
        title.setStyleSheet("font-size: 26px; font-weight: 700; color: black;")

        subtitle = QLabel(f"{len(matched_files)} matching file pair(s) found")
        subtitle.setStyleSheet("color: #666;")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(12)

        for file_info in matched_files:
            card = FileCard(file_info)
            card.clicked.connect(self.file_selected.emit)
            container_layout.addWidget(card)

        container_layout.addStretch()
        scroll.setWidget(container)

        root_layout.addWidget(title)
        root_layout.addWidget(subtitle)
        root_layout.addWidget(scroll, 1)

        self.setStyleSheet("background: #f5f6f8;")

class MainWindow(QMainWindow):
    def __init__(self, markdown_path: str, events_path: str):
        super().__init__()
        self.setWindowTitle("Split View Markdown Matcher")

        root = QWidget()
        self.setCentralWidget(root)

        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # LEFT PANE
        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(12)

        left_title = QLabel("Chemicals")
        left_title.setStyleSheet("font-size: 24px; font-weight: 600; color: black;")

        self.result_label = QLabel("")
        self.result_label.setStyleSheet("color: #666;")

        # Scroll area for chemicals/events
        self.chemical_scroll = QScrollArea()
        self.chemical_scroll.setWidgetResizable(True)
        self.chemical_scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.chemical_container = QWidget()
        self.chemical_layout = QVBoxLayout(self.chemical_container)
        self.chemical_layout.setContentsMargins(0, 0, 0, 0)
        self.chemical_layout.setSpacing(10)

        self.chemical_scroll.setWidget(self.chemical_container)

        left_layout.addWidget(left_title)
        left_layout.addWidget(self.result_label)
        left_layout.addWidget(self.chemical_scroll, 1)

        # RIGHT PANE CONTAINER
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # SEARCH BAR
        search_bar = QWidget()
        search_bar_layout = QHBoxLayout(search_bar)
        search_bar_layout.setContentsMargins(12, 12, 12, 12)
        search_bar_layout.setSpacing(8)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search in markdown...")

        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.clear_button = QPushButton("Clear")
        self.match_label = QLabel("0 / 0")
        self.match_label.setStyleSheet("color: black; min-width: 60px;")

        search_bar_layout.addWidget(self.search_input, 1)
        search_bar_layout.addWidget(self.prev_button)
        search_bar_layout.addWidget(self.next_button)
        search_bar_layout.addWidget(self.clear_button)
        search_bar_layout.addWidget(self.match_label)

        # MARKDOWN VIEWER
        self.md_viewer = MarkdownViewer(markdown_path)

        right_layout.addWidget(search_bar)
        right_layout.addWidget(self.md_viewer, 1)

        # Add panes
        main_layout.addWidget(left_pane, 1)
        main_layout.addWidget(right_container, 1)

        left_pane.setStyleSheet(
            """
            QWidget {
                background: #f7f7f7;
                border-right: 1px solid #dcdcdc;
            }
            QPushButton {
                font-size: 14px;
                padding: 8px 14px;
            }
            """
        )

        search_bar.setStyleSheet(
            """
            QWidget {
                background: #f2f2f2;
                border-bottom: 1px solid #dcdcdc;
            }
            QLineEdit {
                padding: 8px;
                font-size: 14px;
                color: black;
                background: white;
                border: 1px solid #bbb;
                border-radius: 6px;
            }
            QPushButton {
                padding: 8px 12px;
                font-size: 14px;
                color: black;
            }
            """
        )

        self.md_viewer.setStyleSheet(
            """
            QTextBrowser {
                background: white;
                color: black;
                padding: 24px;
                font-size: 15px;
                line-height: 1.5;
                border: none;
            }
            """
        )

        # Signals
        self.search_input.textChanged.connect(self._on_search_changed)
        self.search_input.returnPressed.connect(self._on_return_pressed)
        self.next_button.clicked.connect(self._on_next_clicked)
        self.prev_button.clicked.connect(self._on_prev_clicked)
        self.clear_button.clicked.connect(self._on_clear_clicked)

        # Shortcuts
        self.shortcut_find = QShortcut(QKeySequence("Ctrl+F"), self)
        self.shortcut_find.activated.connect(self._focus_search)

        self.shortcut_next = QShortcut(QKeySequence("F3"), self)
        self.shortcut_next.activated.connect(self._on_next_clicked)

        self.shortcut_prev = QShortcut(QKeySequence("Shift+F3"), self)
        self.shortcut_prev.activated.connect(self._on_prev_clicked)

        self.shortcut_escape = QShortcut(QKeySequence("Escape"), self)
        self.shortcut_escape.activated.connect(self._on_escape)

        # Optional menu action too
        find_action = QAction("Find", self)
        find_action.setShortcut(QKeySequence("Ctrl+F"))
        find_action.triggered.connect(self._focus_search)
        self.addAction(find_action)

        # Events
        chemical_events = EventsManager(events_path).events_data
        self.populate_chemical_list(chemical_events)

    def _on_test_clicked(self):
        found = self.md_viewer.showMatch("test")
        if found:
            self.result_label.setText("Found and highlighted: test")
        else:
            self.result_label.setText("Text not found: test")
        self._update_match_label()

    def _on_search_changed(self, text: str):
        count = self.md_viewer.search_text(text)
        self.result_label.setText("" if text.strip() else self.result_label.text())

        if text.strip():
            if count:
                self.result_label.setText(f'Found {count} match(es) for "{text}"')
            else:
                self.result_label.setText(f'No matches for "{text}"')
        self._update_match_label()

    def _on_return_pressed(self):
        modifiers = QApplication.keyboardModifiers()
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            self._on_prev_clicked()
        else:
            self._on_next_clicked()

    def _on_next_clicked(self):
        if self.md_viewer.next_result():
            self._update_match_label()

    def _on_prev_clicked(self):
        if self.md_viewer.previous_result():
            self._update_match_label()

    def _on_clear_clicked(self):
        self.search_input.clear()
        self.md_viewer.clear_search()
        self.result_label.setText("")
        self._update_match_label()

    def _on_escape(self):
        if self.search_input.hasFocus() or self.search_input.text():
            self._on_clear_clicked()

    def _focus_search(self):
        self.search_input.setFocus()
        self.search_input.selectAll()

    def _update_match_label(self):
        current, total = self.md_viewer.get_search_status()
        self.match_label.setText(f"{current} / {total}")

    def populate_chemical_list(self, chemical_dict: dict[str, list[dict]]):
      # clear current layout
      while self.chemical_layout.count():
          item = self.chemical_layout.takeAt(0)
          if item is not None:
            widget = item.widget()
            if widget is not None:
              widget.deleteLater()

      for chemical_name, events in chemical_dict.items():
          chemical_box = CollapsibleChemicalBox(
              chemical_name=chemical_name,
              events=events,
              on_event_click=self._on_event_clicked,
          )
          self.chemical_layout.addWidget(chemical_box)

      self.chemical_layout.addStretch()

    def _on_event_clicked(self, event: dict):
      matched_text = str(event.get("matched_text", "")).strip()
      short_desc = str(event.get("event_description_short", "")).strip()

      if not matched_text:
          self.result_label.setText(f'No matched_text for event: {short_desc}')
          print(f"Event clicked without matched_text: {event}")
          return

      found = self.md_viewer.showMatch(matched_text)
      if found:
          self.result_label.setText(f'Jumped to: {short_desc}')
          self.result_label.setStyleSheet("color: #1b7f3b;")
      else:
          self.result_label.setText(f'Could not find matched text for: {short_desc}')
          self.result_label.setStyleSheet("color: #b42318;")

      self._update_match_label()


def main():
    app = QApplication(sys.argv)

    markdown_file = r"C:\Users\Davide.Lugli\Code\tesi\PaperDataExtraction\test_data\processed\cleaned_markdown\paper_0001.md"
    events_file = r"C:\Users\Davide.Lugli\Code\tesi\PaperDataExtraction\test_data\labels\scored\paper_0001_events.json"

    window = MainWindow(markdown_file, events_file)
    window.showMaximized()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()