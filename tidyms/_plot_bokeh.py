import numpy as np
import bokeh.plotting
from bokeh.palettes import all_palettes
from bokeh.models import ColumnDataSource, Segment
from .utils import get_settings
from . import _constants as c
from typing import Dict, Generator, List, Optional


def get_bokeh_settings():
    return get_settings()["bokeh"]


def get_theme_params() -> dict:
    return get_bokeh_settings()["theme"]


def get_line_params() -> dict:
    return get_bokeh_settings()["line"]


def get_chromatogram_figure_params() -> dict:
    return get_bokeh_settings()["chromatogram"]["figure"]


def get_spectrum_figure_params() -> dict:
    return get_bokeh_settings()["spectrum"]["figure"]


def get_varea_params() -> dict:
    return get_bokeh_settings()["varea"]


def get_palette() -> List[str]:
    palette_params = get_bokeh_settings()["palette"]
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
    bokeh_settings = get_bokeh_settings()
    xaxis_params = bokeh_settings["chromatogram"]["xaxis"]
    yaxis_params = bokeh_settings["chromatogram"]["yaxis"]
    fig.xaxis.update(**xaxis_params)
    fig.yaxis.update(**yaxis_params)


def set_ms_spectrum_axis_params(fig: bokeh.plotting.Figure):
    bokeh_settings = get_bokeh_settings()
    xaxis_params = bokeh_settings["spectrum"]["xaxis"]
    yaxis_params = bokeh_settings["spectrum"]["yaxis"]
    fig.xaxis.update(**xaxis_params)
    fig.yaxis.update(**yaxis_params)


def fill_area(
    figure: bokeh.plotting.Figure,
    x: np.ndarray,
    y: np.ndarray,
    start: int,
    end: int,
    color: str,
    **varea_params,
):
    default_varea_params = get_varea_params()
    if varea_params:
        default_varea_params.update(varea_params)
    varea_params = default_varea_params

    xp = x[start:end]
    yp = y[start:end]
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


class _LCAssayPlotter:     # pragma: no cover
    """
    Methods to plot data from an Assay. Generates Bokeh Figures.

    Methods
    -------
    roi(sample: str) :
        m/z vs Rt view of the ROI and features in a sample.
    stacked_chromatogram(feature: int) :
        Overlapped chromatographic peaks for a feature in all samples

    """
    def __init__(self, assay):
        self.assay = assay
        self.roi_index = None
        self.ft_index = None

    def _build_roi_index_table(self):
        ft_table = self.assay.feature_table.copy()
        ft_table = ft_table[ft_table[c.LABEL] > -1]
        self.roi_index = (
            ft_table.pivot(index=c.SAMPLE, columns=c.LABEL, values=c.ROI_INDEX)
            .fillna(-1)
            .astype(int)
        )

    def _build_peak_index_table(self):
        ft_table = self.assay.feature_table.copy()
        ft_table = ft_table[ft_table[c.LABEL] > -1]
        self.ft_index = (
            ft_table.pivot(index=c.SAMPLE, columns=c.LABEL, values=c.FT_INDEX)
            .fillna(-1)
            .astype(int)
        )

    def roi(self, sample: str, show: bool = True) -> bokeh.plotting.Figure:
        """
        Plots m/z vs time dispersion of the ROI in a sample. Detected features
        are highlighted using circles.

        Parameters
        ----------
        sample : str
            sample used in the Assay.
        show : bool, default=True
            If True calls ``bokeh.plotting.show`` on the Figure.

        Returns
        -------
        bokeh Figure
        """
        roi = self.assay.load_roi_list(sample)

        TOOLTIPS = [
            ("m/z", "@{}".format(c.MZ)),
            ("Rt", "@{}".format(c.RT)),
            ("area", "@{}".format(c.AREA)),
            ("height", "@{}".format(c.HEIGHT)),
            ("width", "@{}".format(c.WIDTH)),
            ("SNR", "@{}".format(c.SNR)),
            ("roi index", "@{}".format(c.ROI_INDEX)),
            ("feature index", "@{}".format(c.FT_INDEX))
        ]
        fig = bokeh.plotting.figure(tooltips=TOOLTIPS)

        rt_list = list()
        mz_list = list()
        for r in roi:
            rt_list.append(r.time)
            mz_list.append(r.mz)
        line_source = bokeh.plotting.ColumnDataSource(
            dict(xs=rt_list, ys=mz_list)
        )
        line_params = get_line_params()
        fig.multi_line(xs="xs", ys="ys", source=line_source, **line_params)

        try:
            ft = self.assay.load_features(sample)
            source = bokeh.plotting.ColumnDataSource(ft)
            fig.circle('rt', 'mz', size=5, source=source)
        except ValueError:
            pass
        fig.xaxis.update(axis_label="Rt [s]")
        fig.yaxis.update(axis_label="m/z")
        if show:
            bokeh.plotting.show(fig)
        return fig

    def stacked_chromatogram(
        self,
        cluster: int,
        include_classes: Optional[List[str]] = None,
        show: bool = True
    ) -> bokeh.plotting.Figure:
        """
        Plots chromatograms of a feature detected across different samples.

        Parameters
        ----------
        cluster : int
            cluster value obtained from feature correspondence.
        include_classes : List[str] or None, default=None
            List of classes to plot. If None is used, samples from all classes
            are plotted.
        show : bool, default=True
            If True calls ``bokeh.plotting.show`` on the Figure.

        Returns
        -------
        bokeh Figure

        """
        if not self.assay.manager.check_step("match_features"):
            msg = "This plot only can be generated after feature matching"
            raise ValueError(msg)
        else:
            if self.ft_index is None:
                self._build_peak_index_table()

            if self.roi_index is None:
                self._build_roi_index_table()

        fig_params = get_chromatogram_figure_params()
        fig = bokeh.plotting.figure(**fig_params)
        roi_index = self.roi_index[cluster].to_numpy()
        ft_index = self.ft_index[cluster].to_numpy()
        samples = self.roi_index.index
        # TODO: fix after refactoring DataContainers
        classes = self.assay.get_sample_metadata()["class"]
        palette = get_palette()
        if include_classes is not None:
            class_to_color = dict()
            for k, cl in enumerate(include_classes):
                class_to_color[cl] = palette[k]

        iterator = zip(samples, roi_index, ft_index, classes)
        for sample, roi_index, ft_index, class_ in iterator:
            check_draw = (
                (roi_index > -1) and
                ((include_classes is None) or (class_ in include_classes))
            )
            if check_draw:
                if include_classes is None:
                    color = palette[0]
                else:
                    color = class_to_color[class_]
                r = self.assay.load_roi(sample, roi_index)
                ft = r.features[ft_index]
                add_line(fig, r.time, r.spint)
                fill_area(
                    fig, r.time, r.spint, ft.start, ft.end, color, alpha=0.2)
        set_chromatogram_axis_params(fig)
        if show:
            bokeh.plotting.show(fig)
        return fig
