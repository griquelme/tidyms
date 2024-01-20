"""Spectra plotting utilities"""


from ..base.spectrum import MSSpectrum


# TODO: complete.
# def plot(
#     self,
#     fig_params: dict | None = None,
#     line_params: dict | None = None,
#     show: bool = True,
# ) -> bokeh.plotting.figure:  # pragma: no cover
#     """
#     Plot the spectrum using Bokeh.

#     Parameters
#     ----------
#     fig_params : dict or None, default=None
#         key-value parameters to pass to ``bokeh.plotting.figure``.
#     line_params : dict, or None, default=None
#         key-value parameters to pass to ``bokeh.plotting.figure.line``.
#     show : bool, default=True
#         If True calls ``bokeh.plotting.show`` on the Figure.

#     Returns
#     -------
#     bokeh.plotting.figure

#     """
#     default_fig_params = _plot_bokeh.get_spectrum_figure_params()
#     if fig_params:
#         default_fig_params.update(fig_params)
#         fig_params = default_fig_params
#     else:
#         fig_params = default_fig_params
#     fig = bokeh.plotting.figure(**fig_params)

#     if self.is_centroid:
#         plotter = _plot_bokeh.add_stems
#     else:
#         plotter = _plot_bokeh.add_line
#     plotter(fig, self.mz, self.spint, line_params=line_params)
#     _plot_bokeh.set_ms_spectrum_axis_params(fig)
#     if show:
#         bokeh.plotting.show(fig)
#     return fig
