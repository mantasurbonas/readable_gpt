"""Debug tracer for printing intermediate tensors during the GPT forward pass."""

import numpy as np

_PREVIEW_COUNT = 3
_DECIMAL_PLACES = 2


class Tracer:
    """Print formatted vectors and matrices when enabled; no-op when off."""

    def __init__(self):
        self._enabled = False

    def on(self):
        self._enabled = True
        return self

    def off(self):
        self._enabled = False
        return self

    def trace(self, explanation, value):
        if not self._enabled:
            return

        array = np.asarray(value)

        if array.ndim == 1:
            print(self._format_vector(explanation, array))
        elif array.ndim == 2:
            print(self._format_matrix(explanation, array))
        else:
            print(f"{explanation}: (unsupported shape {array.shape})")

    def _format_number(self, value):
        value = float(value)
        # Causal-mask cells are about -1e10. Print them as the mask is written,
        # instead of a ten-billion-digit float.
        if abs(value) >= 1e9:
            return "-1e10" if value < 0 else "1e10"
        return f"{value:.{_DECIMAL_PLACES}f}"

    def trace_tokens(self, explanation, token_ids, texts):
        """Print each token id beside the text piece it stands for."""
        if not self._enabled:
            return

        lines = [f"{explanation}:"]
        for token_id, text in zip(token_ids, texts):
            lines.append(f"  {int(token_id):>6}  {text!r}")
        print("\n".join(lines))

    def trace_token_scores(self, explanation, token_ids, texts, scores, best_token_id):
        """Print a few vocabulary rows and mark the winning logit."""
        if not self._enabled:
            return

        rendered_texts = [repr(text) for text in texts]
        text_width = max(len(rendered) for rendered in rendered_texts)
        lines = [f"{explanation}:"]

        for token_id, rendered, score in zip(token_ids, rendered_texts, scores):
            marker = ""
            if int(token_id) == int(best_token_id):
                marker = "  <- largest"
            formatted_score = self._format_number(score).rjust(8)
            lines.append(f"  {int(token_id):>6}  {rendered:<{text_width}}  {formatted_score}{marker}")

        print("\n".join(lines))

    def _format_vector(self, explanation, array):
        total_count = array.shape[0]
        preview_count = min(_PREVIEW_COUNT, total_count)
        preview_values = array[:preview_count]

        formatted_values = [self._format_number(value) for value in preview_values]
        value_text = ", ".join(formatted_values)

        if total_count > _PREVIEW_COUNT:
            value_text = f"{value_text}, ... ({total_count} in total)"
        else:
            value_text = f"{value_text} ({total_count} in total)"

        return f"{explanation}:\n  [{value_text}]"

    def _format_matrix(self, explanation, array):
        row_count, column_count = array.shape
        preview_row_count = min(_PREVIEW_COUNT, row_count)
        preview_column_count = min(_PREVIEW_COUNT, column_count)

        preview = array[:preview_row_count, :preview_column_count]
        column_widths = [
            max(len(self._format_number(preview[row_index, column_index])) for row_index in range(preview_row_count))
            for column_index in range(preview_column_count)
        ]

        lines = [f"{explanation}:"]

        for row_index in range(preview_row_count):
            formatted_cells = [
                self._format_number(preview[row_index, column_index]).rjust(column_widths[column_index])
                for column_index in range(preview_column_count)
            ]
            row_text = ", ".join(formatted_cells)

            if column_count > _PREVIEW_COUNT:
                row_text = f"{row_text}, ... ({column_count} in total)"
            else:
                row_text = f"{row_text} ({column_count} in total)"

            lines.append(f"   [{row_text}]")

        if row_count > _PREVIEW_COUNT:
            lines.append(f"   ... ({row_count} rows in total)")

        return "\n".join(lines)
