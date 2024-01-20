import numpy as np
from . import constants as c


class MSSpectrum:
    """
    Representation of a Mass Spectrum.

    Manages conversion to centroid and plotting of data.

    Attributes
    ----------
    mz : array
        m/z data
    spint : array
        Intensity data
    time : float or None
        Time at which the spectrum was acquired
    ms_level : int
        MS level of the scan
    polarity : int or None
        Polarity used to acquire the data.
    instrument : {"qtof", "orbitrap"}, default="qtof"
        MS instrument type. Used to set default values in methods.
    is_centroid : bool
        True if the data is in centroid mode.

    """

    def __init__(
        self,
        mz: np.ndarray,
        spint: np.ndarray,
        time: float = 0.0,
        ms_level: int = 1,
        polarity: c.Polarity = c.Polarity.POSITIVE,
        instrument: c.MSInstrument = c.MSInstrument.QTOF,
        is_centroid: bool = True,
    ):
        self.mz = mz
        self.spint = spint
        self.time = time
        self.ms_level = ms_level
        self.polarity = polarity
        self.instrument = instrument
        self.is_centroid = is_centroid

    @property
    def instrument(self) -> str:
        """Get the instrument type used to measure the data."""
        return self._instrument

    @instrument.setter
    def instrument(self, value):
        valid_values = c.MS_INSTRUMENTS
        if value in valid_values:
            self._instrument = value
        else:
            msg = "{} is not a valid instrument. Valid values are: {}."
            raise ValueError(msg.format(value, c.MS_INSTRUMENTS))

    # TODO: move method to raw_data_utilities
    # def find_centroids(
    #     self, min_snr: float = 10.0, min_distance: float | None = None
    # ) -> tuple[np.ndarray, np.ndarray]:
    #     r"""
    #     Find centroids in the spectrum.

    #     Parameters
    #     ----------
    #     min_snr : positive number, default=10.0
    #         Minimum signal-to-noise ratio of the peaks.
    #     min_distance : positive number or None, default=None
    #         Minimum distance between consecutive peaks. If ``None``, the value
    #         is set to 0.01 if ``self.instrument`` is ``"qtof"`` or to 0.005 if
    #         ``self.instrument`` is ``"orbitrap"``.

    #     Returns
    #     -------
    #     centroid : array
    #         m/z centroids. If ``self.is_centroid`` is ``True``, returns
    #         ``self.mz``.
    #     area : array
    #         peak area. If ``self.is_centroid`` is ``True``, returns
    #         ``self.spint``.

    #     """
    #     if self.is_centroid:
    #         centroid, area = self.mz, self.spint
    #     else:
    #         params = get_find_centroid_params(self.instrument)
    #         if min_distance is not None:
    #             params["min_distance"] = min_distance

    #         if min_snr is not None:
    #             params["min_snr"] = min_snr

    #         centroid, area = peaks.find_centroids(self.mz, self.spint, **params)
    #         ord = np.argsort(centroid)
    #         centroid = centroid[ord]
    #         area = area[ord]
    #     return centroid, area


def get_find_centroid_params(instrument: c.MSInstrument) -> dict[str, float]:
    """
    Retrieve default parameters to find_centroid method using instrument information.

    Parameters
    ----------
    instrument : MSInstrument

    Returns
    -------
    params : dict

    """
    params = {"min_snr": 10.0}
    if instrument == c.MSInstrument.QTOF:
        md = 0.01
    else:  # orbitrap
        md = 0.005
    params["min_distance"] = md
    return params
