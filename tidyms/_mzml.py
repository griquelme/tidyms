"""
Functions to parse mzML files, optimized for low-memory usage.

Currently, it retrieves spectrum and chromatogram data.

build_offset_list : Finds offsets with the location of spectra and chromatograms
get_spectrum : Extracts data from a spectrum using offsets.
get_chromatogram : Extracts data from a chromatogram using offsets.

"""

import base64
import numpy as np
import re
import zlib
from os import SEEK_END
from os.path import getsize
from typing import Dict, List, Tuple
from xml.etree.ElementTree import fromstring
from xml.etree.ElementTree import Element

# Rationale for the implementation:
# mzML files can have typical sizes that span from 50 MiB to more than 1000 MiB.
# mzML files are basically xml files with the acquired data and metadata
# associated with the acquisition conditions. The data is inside <spectrum> and
# <chromatogram> elements.  From the expected file sizes, it is clear that
# loading the entire file in memory is not desirable.
# There are tools to parse xml files in an incremental way (such as iterparse
# inside the xml module), but the problem with this approach is that accessing a
# specific scan in the file turns out to be rather slow.
# Indexed mzML files have an <indexedmzML> tag that allows to search in a fast
# way the location of spectrum and chromatogram tags.
# Taking advantage of this information, this module search spectrum and
# chromatogram elements and only loads the part associated with the selected
# spectrum or chromatogram.
# The function build_offset_list creates the list of offsets associated with
# chromatograms and spectra using the information in <indexedmzML>. In the
# case of non-indexed files, the offset list is built from scratch.
# The functions get_spectrum and get_chromatogram takes the offset information
# and extracts the relevant data from each spectrum/chromatogram.


# mzml specification:
# https://www.psidev.info/mzML
# terms defined here:
# https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo
UNSUPPORTED_COMPRESSION = {
    "MS:1002312",
    "MS:1002313",
    "MS:1002314",
    "MS:1002746",
    "MS:1002747",
    "MS:1002748",
    "MS:1003088",
    "MS:1003089",
    "MS:1003090",
}
ZLIB = "MS:1000574"
MZ_ARRAY = "MS:1000514"
INT_ARRAY = "MS:1000515"
TIME_ARRAY = "MS:1000595"
MS_LEVEL = "MS:1000511"
TIME = "MS:1000016"  # time value in scans
NEGATIVE_POLARITY = "MS:1000129"
POSITIVE_POLARITY = "MS:1000130"
# time units
SECONDS = "UO:0000010"
MINUTES = "UO:0000031"

# data types
FLOAT16 = "MS:1000521"
FLOAT32 = "MS:1000518"
FLOAT64 = "MS:1000523"
INT32 = "MS:1000519"
INT64 = "MS:1000522"
DATA_TYPES = {
    FLOAT16: np.float16,
    FLOAT32: np.float32,
    FLOAT64: np.float64,
    INT32: np.int32,
    INT64: np.int64,
}


def build_offset_list(filename: str) -> Tuple[List[int], List[int], int]:
    """
    Finds the offset values in the file where Spectrum or Chromatogram elements
    start.

    Parameters
    ----------
    filename : path to a mzML file

    Returns
    -------
    spectra_offset : list
        Offsets where spectrum element start.
    chromatogram_offset : list
        Offsets where chromatogram elements start.
    index_offset : int
        Offset where the index starts. If the file is not indexed, return the
        file size.

    """
    if is_indexed(filename):
        index_offset = _get_index_offset(filename)
        spectra_offset, chromatogram_offset = _build_offset_list_indexed(
            filename, index_offset
        )
    else:
        spectra_offset, chromatogram_offset = _build_offset_list_non_indexed(
            filename
        )
        index_offset = getsize(filename)
    return spectra_offset, chromatogram_offset, index_offset


def get_spectrum(
    filename: str,
    spectra_offset: List[int],
    chromatogram_offset: List[int],
    index_offset: int,
    n: int,
) -> Dict:
    """
    Extracts m/z, intensity, polarity , time, and ms level from the nth scan.
    Parameters
    ----------
    filename : str
        path to mzML file.
    spectra_offset : list
        Offset list obtained from `_build_offset_list`.
    chromatogram_offset : list
        Offset list obtained from `_build_offset_list`.
    index_offset : int
        Offset obtained from `_build_offset_list`.
    n : int
        scan number to select.

    Returns
    -------
    spectrum : dictionary with spectrum data.

    """
    xml_str = _get_xml_data(
        filename,
        spectra_offset,
        chromatogram_offset,
        index_offset,
        n,
        "spectrum"
    )
    elements = list(fromstring(xml_str))
    spectrum = dict()
    for el in elements:
        tag = el.tag
        if tag == "cvParam":
            accession = el.attrib.get("accession")
            if accession == MS_LEVEL:
                spectrum["ms_level"] = int(el.attrib.get("value"))
            elif accession == NEGATIVE_POLARITY:
                spectrum["polarity"] = -1
            elif accession == POSITIVE_POLARITY:
                spectrum["polarity"] = 1
        elif tag == "scanList":
            spectrum["time"] = _get_time(el)
        elif tag == "binaryDataArrayList":
            sp_data = _parse_binary_data_list(el)
            spectrum.update(sp_data)
    return spectrum


def get_chromatogram(
    filename: str,
    spectra_offset: List[int],
    chromatogram_offset: List[int],
    index_offset: int,
    n: int,
) -> Dict:
    """
    Extracts time and intensity from xml chunk.

    Parameters
    ----------
    filename : path to mzML file
    spectra_offset : offset list obtained from _build_offset_list
    chromatogram_offset : offset list obtained from _build_offset_list
    index_offset : offset obtained from _build_offset_list
    n : chromatogram number

    Returns
    -------
    chromatogram: dict

    """
    xml_str = _get_xml_data(
        filename,
        spectra_offset,
        chromatogram_offset,
        index_offset,
        n,
        "chromatogram"
    )
    elements = fromstring(xml_str)
    name = elements.attrib.get("id")
    chromatogram = dict(name=name)
    for el in list(elements):
        tag = el.tag
        if tag == "binaryDataArrayList":
            chrom_data = _parse_binary_data_list(el)
            chromatogram.update(chrom_data)
    return chromatogram


class ReverseReader:
    """
    Reads file objects starting from the EOF.

    """

    def __init__(self, filename: str, buffer_size: int, **kwargs):
        """
        Constructor object

        Parameters
        ----------
        filename : str,
            Path to the file
        buffer_size : int
            size of the chunk to get when using the `read_chunk` method
        **kwargs :
            keyword arguments to pass to the open function
        """

        self.file = open(filename, **kwargs)
        self.buffer_size = buffer_size
        self._size = self.file.seek(0, SEEK_END)
        self.offset = self._size
        self._is_complete = False

    @property
    def offset(self) -> int:
        return self._offset

    @offset.setter
    def offset(self, value: int):
        if value < 0:
            value = 0
            self._is_complete = True
        self._offset = value
        self.file.seek(value)

    def __enter__(self):
        return self

    def __exit__(self, t, value, traceback):
        self.file.close()

    def read_chunk(self) -> str:
        """
        Reads a chunk of data from the file, starting from the end. If the
        beginning of the file has been reached, it returns None.

        Returns
        -------
        chunk : str
        """
        if self._is_complete:
            res = None
        else:
            self.offset = self.offset - self.buffer_size
            res = self.file.read(self.buffer_size)
        return res

    def reset(self):
        """
        Set the position of the reader to the EOF.
        """
        self.offset = self._size
        self._is_complete = False


def _read_binary_data_array(element: Element) -> Tuple[np.ndarray, str]:
    """
    Extracts the binary data and data kind from a binaryArray element.

    Parameters
    ----------
    element: Element

    Returns
    -------
    data : array
    kind : can be one of {"mz", "spint", "time"}

    """
    has_zlib_compression = False
    data = None
    kind = None
    units = None
    dtype = None
    for e in element:
        tag = e.tag
        if tag == "binary":
            data = e.text
        else:
            accession = e.attrib.get("accession")
            if accession in UNSUPPORTED_COMPRESSION:
                msg = "Currently only zlib compression is supported."
                raise NotImplementedError(msg)
            elif accession == ZLIB:
                has_zlib_compression = True
            elif accession == INT_ARRAY:
                kind = "spint"
            elif accession == MZ_ARRAY:
                kind = "mz"
            elif accession == TIME_ARRAY:
                kind = "time"
                units = e.attrib.get("unitAccession")
            if accession in DATA_TYPES:
                dtype = DATA_TYPES[accession]
    if data:
        data = base64.b64decode(data)
        if has_zlib_compression:
            data = zlib.decompress(data)
        data = np.frombuffer(data, dtype=dtype).copy()
        if kind == "time":
            data = _time_to_seconds(data, units)
    else:
        data = np.array([])
    return data, kind


def _parse_binary_data_list(element: Element) -> Dict[str, np.ndarray]:
    """
    Extracts the data from a binaryDataArrayList.

    Parameters
    ----------
    element: Element

    Returns
    -------
    dictionary that maps data kind to its correspondent data array.

    """
    res = dict()
    for e in element:
        data, data_type = _read_binary_data_array(e)
        res[data_type] = data
    return res


def is_indexed(filename: str) -> bool:
    """
    Checks if a mzML file is indexed.

    Parameters
    ----------
    filename: str

    Returns
    -------
    bool

    Notes
    -----
    This function assumes that the mzML file is validated. It looks up the last
    closing tag, that should be </indexedmzML> if the file is indexed.

    """
    with ReverseReader(filename, 1024, mode="r") as fin:
        end_tag = "</indexedmzML>"
        chunk = fin.read_chunk()
        res = chunk.find(end_tag) != -1
    return res


def _get_index_offset(filename: str) -> int:
    """
    Search the byte offset where the indexListOffset element starts.

    Parameters
    ----------
    filename : str

    Returns
    -------
    index_offset : int

    """
    tag = "<indexListOffset>"
    # reads mzml backwards until the tag is found
    # we exploit the fact that according to the mzML schema the
    # indexListOffset should have neither attributes nor subelements
    with ReverseReader(filename, 1024, mode="r") as fin:
        xml = ""
        ind = -1
        while ind == -1:
            chunk = fin.read_chunk()
            xml = chunk + xml
            ind = chunk.find(tag)
        # starts at the beginning of the text tag
        start = ind + len(tag)
        xml = xml[start:]
        end = xml.find("<")
    index_offset = int(xml[:end])
    return index_offset


def _build_offset_list_non_indexed(
        filename: str
) -> Tuple[List[int], List[int]]:
    """
    Builds manually the indices for non-indexed mzML files.

    Parameters
    ----------
    filename : str

    Returns
    -------

    """
    # indices are build by finding the offset where spectrum or chromatogram
    # elements starts.
    spectrum_regex = re.compile("<spectrum .[^(><.)]+>")
    chromatogram_regex = re.compile("<chromatogram .[^(><.)]+>")
    ind = 0
    spectrum_offset_list = list()
    chromatogram_offset_list = list()
    with open(filename) as fin:
        while True:
            line = fin.readline()
            if line == "":
                break
            spectrum_offset = _find_spectrum_tag_offset(line, spectrum_regex)
            if spectrum_offset:
                spectrum_offset_list.append(ind + spectrum_offset)
            chromatogram_offset = _find_chromatogram_tag_offset(
                line, chromatogram_regex
            )
            if chromatogram_offset:
                chromatogram_offset_list.append(ind + chromatogram_offset)
            ind += len(line)
    return spectrum_offset_list, chromatogram_offset_list


def _find_spectrum_tag_offset(line: str, regex: re.Pattern) -> int:
    if line.lstrip().startswith("<spectrum"):
        match = regex.search(line)
        if match:
            start, end = match.span()
        else:
            start = None
        return start


def _find_chromatogram_tag_offset(line: str, regex: re.Pattern) -> int:
    if line.lstrip().startswith("<chromatogram"):
        match = regex.search(line)
        if match:
            start, end = match.span()
        else:
            start = None
        return start


def _build_offset_list_indexed(
    filename: str,
    index_offset: int
) -> Tuple[List[int], List[int]]:
    """
    Builds a list of offsets where spectra and chromatograms are stored.

    Parameters
    ----------
    filename : str
    index_offset : int
        offset obtained from _get_index_offset

    Returns
    -------
    spectra_offset : List
        offset where spectra are stored
    chromatogram_offset : List
        offset where chromatograms are stored

    """
    end_tag = "</indexList>"
    with open(filename, "r") as fin:
        fin.seek(index_offset)
        index_xml = fin.read()
        end = index_xml.find(end_tag)
        index_xml = index_xml[: end + len(end_tag)]
    index_xml = fromstring(index_xml)

    spectra_offset = list()
    chromatogram_offset = list()
    for index in index_xml:
        for offset in index:
            value = int(offset.text)
            if index.attrib["name"] == "spectrum":
                spectra_offset.append(value)
            elif index.attrib["name"] == "chromatogram":
                chromatogram_offset.append(value)
    return spectra_offset, chromatogram_offset


def _get_xml_data(
    filename: str,
    spectra_offset: List[int],
    chromatogram_offset: List[int],
    index_offset: int,
    n: int,
    kind: str,
) -> str:
    """
    Get the xml string associated with a spectrum or chromatogram.

    Parameters
    ----------
    filename : str
    spectra_offset : list
        offsets obtained from _build_offset_list
    chromatogram_offset : list
        offsets obtained from _build_offset_list
    index_offset : int
        offset obtained from _get_index_offset
    n : int
        number of spectrum/chromatogram to select
    kind : {"spectrum", "chromatogram"}

    Returns
    -------
    str

    """
    if kind == "spectrum":
        l, other = spectra_offset, chromatogram_offset
        end_tag = "</spectrum>"
    elif kind == "chromatogram":
        l, other = chromatogram_offset, spectra_offset
        end_tag = "</chromatogram>"
    else:
        raise ValueError("Kind must be `spectrum` or `chromatogram`")

    start = l[n]
    # Here we search the closest offset to start such that the complete data
    # is contained in the text
    try:
        end = l[n + 1]
    except IndexError:
        try:
            end = other[0]
        except IndexError:
            end = index_offset

    if end < start:
        end = index_offset

    with open(filename, "r") as fin:
        fin.seek(start)
        chunk = fin.read(end - start)
    end = chunk.find(end_tag)
    return chunk[: end + len(end_tag)]


def _get_time(element):
    for e in list(element):
        tag = e.tag
        if tag == "scan":
            for ee in list(e):
                accession = ee.attrib.get("accession")
                if accession == TIME:
                    value = float(ee.attrib.get("value"))
                    units = ee.attrib.get("unitAccession")
                    value = _time_to_seconds(value, units)
                    return value


def _time_to_seconds(value: float, units: str):
    if units == MINUTES:
        value = value * 60
    return value
