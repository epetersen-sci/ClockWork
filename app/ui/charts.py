"""One place that renders a plotly figure, so its download button behaves.

Every chart in the app is drawn through :func:`plotly_chart` rather than
``st.plotly_chart`` directly. The only thing it adds is the modebar camera
button's configuration, and the reason it is worth a wrapper is that the
alternative is repeating the same config dict at thirty-odd call sites and
watching it drift.

**What the camera button can and cannot do.** It is a browser download, so the
file lands in the browser's download folder and nothing on this side can change
that — a page cannot choose a download directory. What plotly does let us set is
the name, the format and the pixel scale, so a figure arrives as
``exp8_8_18_26_sleep_profile.png`` at twice the on-screen resolution instead of
``newplot.png`` at one. Anyone who wants figures beside their data uses the
"Save … as PNG" buttons in ``export_helpers``, which write into the working
folder like every other export.

The filename comes from the figure's own title, so a page gets a sensible one
without passing anything, and the experiment's name is prefixed when a dataset is
loaded — a downloads folder collects figures from every run anyone has open.
"""

import re

import streamlit as st

import export_helpers

#: Twice the on-screen pixel dimensions. The default of 1 produces a figure that
#: is visibly soft the moment it goes into a slide or a figure panel.
DEFAULT_SCALE = 2

#: Used when a figure has no title and the caller named nothing. Still better than
#: plotly's own "newplot", which says nothing about where the file came from.
FALLBACK_NAME = "clockwork_figure"


def _slug(text, limit=80):
    """``'Sleep profile (ZT)<br>DD'`` -> ``'Sleep_profile_ZT_DD'``.

    Plotly titles carry markup — ``<br>`` for a second line, ``<b>``/``<i>`` for
    emphasis, ``<sub>`` for units — which has to come out before the text can be a
    filename, or the tags end up in it as literal characters.
    """
    text = re.sub(r"<[^>]+>", " ", str(text or ""))
    return "_".join(p for p in re.split(r"[^A-Za-z0-9]+", text) if p)[:limit]


def _figure_title(fig):
    """The figure's title text, or ``''``. Tolerates a plain dict figure and a
    figure whose layout has no title at all."""
    try:
        title = fig.layout.title.text
    except AttributeError:
        title = (fig or {}).get("layout", {}).get("title", {}).get("text")
    return title or ""


def _experiment_prefix():
    """The loaded experiment's name, for the front of a downloaded filename.

    Read from the dataset rather than passed in, so no call site has to hand a
    dataset to a function that only draws a chart. Absent before anything is
    loaded, which is fine — the title alone still names the file.
    """
    ds = st.session_state.get("dataset")
    attrs = getattr(ds, "attrs", None) if ds is not None else None
    return _slug(attrs.get("experiment_name")) if attrs else ""


def png_filename(fig, filename=None):
    """The name the camera button should save under, without its extension."""
    stem = _slug(filename) if filename else _slug(_figure_title(fig))
    prefix = _experiment_prefix()
    if stem and prefix:
        return f"{prefix}_{stem}"
    return stem or prefix or FALLBACK_NAME


def png_config(fig, filename=None, *, scale=DEFAULT_SCALE):
    """The plotly config dict that configures the modebar camera button.

    Merged by plotly with its own defaults, so this changes the download and
    nothing else about the chart.
    """
    return {
        "toImageButtonOptions": {
            "format": "png",
            "filename": png_filename(fig, filename),
            "scale": scale,
        }
    }


#: Figures drawn during the current rerun, as ``(filename, figure)``. Reset by
#: :func:`begin_run` from the entry point, which runs before every page body.
_DRAWN_KEY = "_figures_drawn_this_run"


def begin_run():
    """Start a fresh figure collection for this rerun.

    Called by the entry point rather than by each page, because the entry point is
    the one piece of code that runs exactly once per rerun before any page body. A
    page that forgot to call it would quietly accumulate duplicates of every figure
    across reruns, which is the kind of bug nobody notices until an export writes
    forty files.
    """
    st.session_state[_DRAWN_KEY] = []


def drawn_figures():
    """The figures drawn so far this rerun.

    Only what actually rendered: a chart inside an unselected tab was never drawn,
    so it is not here and will not be exported. That is the honest answer — what
    you can see is what you get.
    """
    return list(st.session_state.get(_DRAWN_KEY, []))


def _record(fig, name):
    """Remember a drawn figure under a filename unique within this run.

    Two figures on a page can legitimately share a title (the same plot for LD and
    DD, say). Left alone they would write to one filename and the second would
    overwrite the first, so the duplicate gets a numeric suffix.
    """
    drawn = st.session_state.setdefault(_DRAWN_KEY, [])
    taken = {existing for existing, _ in drawn}
    filename, n = f"{name}.png", 1
    while filename in taken:
        n += 1
        filename = f"{name}_{n}.png"
    drawn.append((filename, fig))


def save_figures_button(ds=None, *, key="save_page_figures", subfolder=None):
    """Render "Save the N figures on this page" — or nothing, if none were drawn.

    Rendered by the entry point after the page body, so every page offers it in the
    same place without each one having to remember. Writes into the same
    ``Graph Exports_<experiment>/`` folder as the CSV buttons, rather than the
    browser's download folder, which is the one thing the modebar's camera button
    cannot do.
    """
    figures = drawn_figures()
    if not figures:
        return None
    ds = ds if ds is not None else st.session_state.get("dataset")
    if ds is None:
        return None

    n = len(figures)
    return export_helpers.save_figures_png_button(
        f"Save {n} figure{'s' if n != 1 else ''} on this page as PNG",
        figures,
        ds,
        key,
        subfolder=subfolder,
        help="Writes them into the working folder beside your data, at twice screen "
        "resolution. The camera icon on a single chart downloads just that one, to "
        "your browser's download folder.",
    )


def plotly_chart(fig, *, filename=None, scale=DEFAULT_SCALE, **kwargs):
    """``st.plotly_chart`` with the camera button configured.

    ``filename`` overrides the name derived from the figure's title — pass it when
    a page draws several figures that share a title, or when the title makes a poor
    filename. Every other argument goes straight to ``st.plotly_chart``, including
    ``key`` and ``on_select``, and its return value comes back unchanged.
    """
    _record(fig, png_filename(fig, filename))
    return st.plotly_chart(fig, config=png_config(fig, filename, scale=scale), **kwargs)
