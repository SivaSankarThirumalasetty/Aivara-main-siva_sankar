"""
export/chart_images.py — Plotly figure -> static PNG bytes (kaleido), sized
for slide placement.

This module never touches the network and never rebuilds a chart with a
different library — it re-renders the exact same Plotly figure object that
was already built for the on-screen dashboard (ui/components.py), so the
deck matches what the user saw.
"""

from __future__ import annotations

import io

# Default size tuned for a half/third-width placement on a 13.33" x 7.5"
# (widescreen) slide at scale=2 for crispness.
DEFAULT_WIDTH = 900
DEFAULT_HEIGHT = 500
DEFAULT_SCALE = 2


def figure_to_png_bytes(
    fig, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT, scale: int = DEFAULT_SCALE
) -> bytes:
    """Render a Plotly figure to PNG bytes via the kaleido engine, fully
    offline. Raises RuntimeError with a clear message if kaleido isn't
    available, rather than a cryptic import error deep in export flow."""
    try:
        png_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
    except Exception as exc:
        raise RuntimeError(
            "Could not export chart to PNG — is the 'kaleido' package installed? "
            f"Original error: {exc}"
        ) from exc
    return png_bytes


def figure_to_png_stream(fig, **kwargs) -> io.BytesIO:
    return io.BytesIO(figure_to_png_bytes(fig, **kwargs))
