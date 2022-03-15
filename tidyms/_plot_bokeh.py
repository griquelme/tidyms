import numpy as np
import bokeh.plotting
from bokeh.palettes import all_palettes
from bokeh.models import ColumnDataSource, Segment
from .peaks import Peak
from .utils import SETTINGS
from typing import Dict, Generator, List, Optional

BOKEH_SETTINGS = SETTINGS["bokeh"]


def get_theme_params() -> dict:
    return BOKEH_SETTINGS["theme"]


def get_line_params() -> dict:
    return BOKEH_SETTINGS["line"]


def get_chromatogram_figure_params() -> dict:
    return BOKEH_SETTINGS["chromatogram"]["figure"]


def get_spectrum_figure_params() -> dict:
    return BOKEH_SETTINGS["spectrum"]["figure"]


def get_varea_params() -> dict:
    return BOKEH_SETTINGS["varea"]


def get_palette() -> List[str]:
    palette_params = BOKEH_SETTINGS["palette"]
    return find_palette(**palette_params)


def make_figure(fig_params: Optional[dict]):
    if fig_params is None:
        fig_params = dict()
    return bokeh.plotting.figure(**fig_params)


def find_palette(name: str, size: Optional[int] = None) -> List[str]:
    try:
        palette = bokeh.palettes.all_palettes[name]
        # by default get the palette with the largest size
        if size is None:
            size = max(list(palette.keys()))
        palette = palette[size]
    except KeyError:
        link = "https://docs.bokeh.org/en/latest/docs/reference/palettes.html"
        msg = "Palette not found. Refer to the list of prebuilt palettes at {}"
        raise ValueError(msg.format(link))
    return palette


def palette_cycler(palette: List[str]) -> Generator[str, None, None]:
    ind = 0
    size = len(palette)
    while True:
        yield palette[ind]
        ind = (ind + 1) % size


def add_line(
    figure: bokeh.plotting.Figure,
    x: np.ndarray,
    y: np.ndarray,
    line_params: Optional[dict] = None
):
    """
    Plots a line.

    Parameters
    ----------
    figure : bokeh.plotting.Figure
        key-value parameters to pass into bokeh figure function.
    x : array
    y : array
    line_params : dict
        key-value parameters to pass into bokeh line function.

    """
    default_line_params = get_line_params()
    if line_params:
        default_line_params.update(line_params)
    line_params = default_line_params
    figure.line(x, y, **line_params)


def set_chromatogram_axis_params(fig: bokeh.plotting.Figure):
    xaxis_params = BOKEH_SETTINGS["chromatogram"]["xaxis"]
    yaxis_params = BOKEH_SETTINGS["chromatogram"]["yaxis"]
    fig.xaxis.update(**xaxis_params)
    fig.yaxis.update(**yaxis_params)


def set_ms_spectrum_axis_params(fig: bokeh.plotting.Figure):
    xaxis_params = BOKEH_SETTINGS["spectrum"]["xaxis"]
    yaxis_params = BOKEH_SETTINGS["spectrum"]["yaxis"]
    fig.xaxis.update(**xaxis_params)
    fig.yaxis.update(**yaxis_params)


def fill_peaks(
    figure: bokeh.plotting.Figure,
    x: np.ndarray,
    y: np.ndarray,
    peaks: List[Peak],
    palette: Optional[str] = None,
    varea_params: Optional[Dict] = None,
):
    default_varea_params = get_varea_params()
    if varea_params:
        default_varea_params.update(varea_params)
    varea_params = default_varea_params

    if palette is None:
        palette = get_palette()
    colors = palette_cycler(palette)
    for p in peaks:
        color = next(colors)
        xp = x[p.start:p.end]
        yp = y[p.start:p.end]
        figure.varea(xp, yp, 0, fill_color=color, **varea_params)


def add_stems(
    fig: bokeh.plotting.Figure,
    x: np.ndarray,
    y: np.ndarray,
    line_params: Optional[Dict] = None
):
    default_line_params = get_line_params()
    if line_params:
        default_line_params.update(line_params)
    line_params = default_line_params
    x0 = x
    y0 = np.zeros_like(y)
    source = ColumnDataSource(dict(x0=x0, x1=x, y0=y0, y1=y))
    stems = Segment(x0="x0", x1="x1", y0="y0", y1="y1", **line_params)
    fig.add_glyph(source, stems)
