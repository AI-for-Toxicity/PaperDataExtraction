import os
import json
import tkinter as tk
from tkinter import ttk, messagebox

# === CONFIG ===
PAPERS_DIR = "./data/raw/pdfs"             # folder containing your paper_* files
JSON_PATH = "./data/labels/annotations.json"    # path to the JSON store


class EventRow:
    def __init__(self, parent, index, remove_callback):
        """
        One event row inside a paragraph:
        chemical, event_type, event_short, event_long, order (read-only), delete button.
        """
        self.parent = parent
        self.frame = tk.Frame(parent, bd=1, relief=tk.RIDGE, padx=5, pady=5)
        self.remove_callback = remove_callback

        # Order (read-only label)
        self.order_label = tk.Label(self.frame, width=3, anchor="center")
        self.order_label.grid(row=0, column=0, rowspan=2, sticky="n")

        # chemical
        tk.Label(self.frame, text="Chemical").grid(row=0, column=1, sticky="w")
        self.chemical_entry = tk.Entry(self.frame, width=20)
        self.chemical_entry.grid(row=0, column=2, sticky="w")

        # event_type selector
        tk.Label(self.frame, text="Type").grid(row=0, column=3, sticky="w")
        self.event_type_var = tk.StringVar(value="ME")
        self.event_type_combo = ttk.Combobox(
            self.frame,
            textvariable=self.event_type_var,
            values=["ME", "KE", "AO"],
            width=5,
            state="readonly",
        )
        self.event_type_combo.grid(row=0, column=4, sticky="w")

        # event_short
        tk.Label(self.frame, text="Short").grid(row=0, column=5, sticky="w")
        self.event_short_entry = tk.Entry(self.frame, width=40)
        self.event_short_entry.grid(row=0, column=6, sticky="w")

        # Delete button
        self.delete_btn = tk.Button(self.frame, text="X", command=self._on_delete)
        self.delete_btn.grid(row=0, column=7, rowspan=2, sticky="ne")

        # event_long
        tk.Label(self.frame, text="Long").grid(row=1, column=1, sticky="nw")
        self.event_long_text = tk.Text(self.frame, width=60, height=3)
        self.event_long_text.grid(row=1, column=2, columnspan=5, sticky="w")

        # Set initial order
        self.set_order(index)

    def grid(self, **kwargs):
        self.frame.grid(**kwargs)

    def destroy(self):
        self.frame.destroy()

    def set_order(self, idx: int):
        self.order_label.config(text=str(idx))

    def _on_delete(self):
        self.remove_callback(self)

    def to_dict(self, idx: int):
        chemical = self.chemical_entry.get().strip()
        event_type = self.event_type_var.get()
        event_short = self.event_short_entry.get().strip()
        event_long = self.event_long_text.get("1.0", "end").strip()

        return {
            "order": idx,
            "chemical": chemical,
            "event_type": event_type,
            "event_short": event_short,
            "event_long": event_long,
        }


class ParagraphBlock:
    def __init__(self, parent, index, remove_callback):
        """
        One paragraph block:
        paragraph_id, paragraph_text, events list (+ button, X for each event).
        """
        self.parent = parent
        self.frame = tk.Frame(parent, bd=2, relief=tk.GROOVE, padx=5, pady=5)
        self.remove_callback = remove_callback

        # Header row: label + delete paragraph button
        header = tk.Frame(self.frame)
        header.pack(fill="x")

        self.title_label = tk.Label(header, text=f"Paragraph {index}", font=("TkDefaultFont", 10, "bold"))
        self.title_label.pack(side="left")

        self.delete_paragraph_btn = tk.Button(header, text="Delete paragraph", command=self._on_delete)
        self.delete_paragraph_btn.pack(side="right")

        # Paragraph fields
        fields = tk.Frame(self.frame)
        fields.pack(fill="x", pady=3)

        tk.Label(fields, text="Paragraph ID").grid(row=0, column=0, sticky="w")
        self.paragraph_id_entry = tk.Entry(fields, width=8)
        self.paragraph_id_entry.grid(row=0, column=1, sticky="w", padx=3)

        tk.Label(fields, text="Paragraph text").grid(row=1, column=0, sticky="nw")
        self.paragraph_text = tk.Text(fields, width=80, height=4)
        self.paragraph_text.grid(row=1, column=1, columnspan=5, sticky="w", pady=2)

        # Events header
        events_header = tk.Frame(self.frame)
        events_header.pack(fill="x", pady=(5, 0))

        tk.Label(events_header, text="Events").pack(side="left")
        self.add_event_btn = tk.Button(events_header, text="+ Add event", command=self.add_event)
        self.add_event_btn.pack(side="right")

        # Events container
        self.events_container = tk.Frame(self.frame)
        self.events_container.pack(fill="both", expand=True, pady=(2, 0))

        self.event_rows = []
        self.set_index(index)

    def grid(self, **kwargs):
        self.frame.grid(**kwargs)

    def destroy(self):
        self.frame.destroy()

    def set_index(self, idx: int):
        self.title_label.config(text=f"Paragraph {idx}")

    def _on_delete(self):
        self.remove_callback(self)

    def add_event(self):
        idx = len(self.event_rows)
        row = EventRow(self.events_container, idx, self.remove_event)
        row.grid(row=idx, column=0, sticky="ew", pady=2)
        self.event_rows.append(row)

    def remove_event(self, row: EventRow):
        row.destroy()
        self.event_rows.remove(row)
        self._renumber_events()

    def _renumber_events(self):
        for idx, row in enumerate(self.event_rows):
            row.set_order(idx)
            row.frame.grid(row=idx, column=0, sticky="ew", pady=2)

    def to_dict(self):
        # paragraph_id can be null
        pid_raw = self.paragraph_id_entry.get().strip()
        if pid_raw == "":
            paragraph_id = None
        else:
            try:
                paragraph_id = int(pid_raw)
            except ValueError:
                raise ValueError(f"Invalid paragraph_id '{pid_raw}' (must be int or empty)")

        paragraph_text = self.paragraph_text.get("1.0", "end").strip()

        events = []
        for idx, row in enumerate(self.event_rows):
            events.append(row.to_dict(idx))

        return {
            "paragraph_id": paragraph_id,
            "paragraph_text": paragraph_text,
            "events": events,
        }


class App:
    def __init__(self, root):
        self.root = root
        root.title("AOP Annotation Form (paragraph + events)")

        self.paper_ids = self._load_paper_ids()
        if not self.paper_ids:
            messagebox.showerror("Error", f"No files found in {PAPERS_DIR}")
            root.destroy()
            return

        # Top: paper selector
        top_frame = tk.Frame(root, padx=10, pady=10)
        top_frame.pack(fill="x")

        tk.Label(top_frame, text="paper_id:").pack(side="left")

        self.paper_var = tk.StringVar()
        self.paper_combo = ttk.Combobox(
            top_frame,
            textvariable=self.paper_var,
            values=self.paper_ids,
            state="readonly",
            width=40,
        )
        self.paper_combo.pack(side="left", padx=5)
        self.paper_combo.current(0)

        # Paragraphs area
        paragraphs_frame = tk.Frame(root, padx=10, pady=10)
        paragraphs_frame.pack(fill="both", expand=True)

        header_frame = tk.Frame(paragraphs_frame)
        header_frame.pack(fill="x")

        tk.Label(header_frame, text="Paragraphs").pack(side="left")

        self.add_paragraph_btn = tk.Button(header_frame, text="+ Add paragraph", command=self.add_paragraph)
        self.add_paragraph_btn.pack(side="right")

        # Scrollable area
        canvas = tk.Canvas(paragraphs_frame)
        scrollbar = tk.Scrollbar(paragraphs_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.paragraphs_container = tk.Frame(canvas)
        canvas.create_window((0, 0), window=self.paragraphs_container, anchor="nw")

        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        self.paragraphs_container.bind("<Configure>", _on_frame_configure)

        def _on_mousewheel(event):
            # Windows/Linux
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Confirm button
        bottom_frame = tk.Frame(root, padx=10, pady=10)
        bottom_frame.pack(fill="x")

        self.confirm_btn = tk.Button(bottom_frame, text="CONFIRM", command=self.on_confirm)
        self.confirm_btn.pack(side="right")

        # Internal list of ParagraphBlock
        self.paragraph_blocks = []

    def _load_paper_ids(self):
        if not os.path.isdir(PAPERS_DIR):
            return []
        files = [
            f for f in os.listdir(PAPERS_DIR)
            if os.path.isfile(os.path.join(PAPERS_DIR, f))
        ]
        stems = sorted({os.path.splitext(f)[0] for f in files})
        return stems

    def add_paragraph(self):
        idx = len(self.paragraph_blocks)
        block = ParagraphBlock(self.paragraphs_container, idx, self.remove_paragraph)
        block.grid(row=idx, column=0, sticky="ew", pady=5)
        self.paragraph_blocks.append(block)

    def remove_paragraph(self, block: ParagraphBlock):
        block.destroy()
        self.paragraph_blocks.remove(block)
        self._renumber_paragraphs()

    def _renumber_paragraphs(self):
        for idx, block in enumerate(self.paragraph_blocks):
            block.set_index(idx)
            block.frame.grid(row=idx, column=0, sticky="ew", pady=5)

    def clear_form(self):
        for block in self.paragraph_blocks:
            block.destroy()
        self.paragraph_blocks.clear()

    def on_confirm(self):
        if not self.paper_ids:
            messagebox.showerror("Error", "No paper IDs available.")
            return

        paper_id = self.paper_var.get().strip()
        if paper_id == "":
            messagebox.showerror("Error", "paper_id is empty.")
            return

        if not self.paragraph_blocks:
            res = messagebox.askyesno(
                "No paragraphs",
                "No paragraphs added. Save empty paragraph list for this paper_id?"
            )
            if not res:
                return

        paragraphs = []
        try:
            for block in self.paragraph_blocks:
                paragraphs.append(block.to_dict())
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return

        form_content = {
            "paper_id": paper_id.split("_")[0] + "_" + paper_id.split("_")[1],
            "paragraphs": paragraphs,
        }

        # Load existing JSON (if any)
        data = {}
        if os.path.isfile(JSON_PATH):
            try:
                with open(JSON_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError("Root of JSON must be an object")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read {JSON_PATH}:\n{e}")
                return

        # Overwrite or insert for this paper_id
        data[paper_id] = form_content

        # Write back
        try:
            with open(JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to write {JSON_PATH}:\n{e}")
            return

        # After saving:
        self.clear_form()
        self._advance_paper_selection(paper_id)

        messagebox.showinfo("Saved", f"Data for {paper_id} saved to {JSON_PATH}.")

    def _advance_paper_selection(self, current_id):
        if current_id not in self.paper_ids:
            return
        idx = self.paper_ids.index(current_id)
        if idx + 1 < len(self.paper_ids):
            next_id = self.paper_ids[idx + 1]
            self.paper_var.set(next_id)
        # If it's the last one, you’re out of papers. Touch grass or something.


def main():
    root = tk.Tk()
    app = App(root)
    if app.paper_ids:
        root.mainloop()


if __name__ == "__main__":
    main()
