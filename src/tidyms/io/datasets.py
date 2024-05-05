"""Download example datasets."""

import pathlib
from typing import Any

import pydantic
import requests

from ..core.exceptions import ExampleDatasetNotFound
from ..utils import get_tidyms_path


class DatasetFileContent(pydantic.BaseModel):
    """Store a dataset file content."""

    name: str
    """The file name"""

    path: str
    """The file path in the GitHub repo"""

    content: str | None = None


def download_dataset(name: str, download_dir: pathlib.Path | None = None):
    """
    Download a list of files from the `data repository <https://github.com/griquelme/tidyms-data>`_.

    Parameters
    ----------
    name : str
        Name of the data directory
    files : List[str]
        List of files inside the data directory.
    download_dir : str or None, default=None
        String representation of a path to download the data. If None, downloads
        the data to the `.tidyms` directory

    Examples
    --------
    Download the `data.csv` file from the `reference-materials` directory into
    the current directory:

    >>> import tidyms as ms
    >>> dataset = "reference-materials"
    >>> file_list = ["data.csv"]
    >>> ms.fileio.download_tidyms_data(dataset, file_list, download_dir=".")

    See Also
    --------
    download_dataset
    load_dataset

    """
    owner = "griquelme"
    repo = "tidyms-data"

    dataset_files = list_dataset_files(owner, repo, name)

    if download_dir is None:
        download_dir = get_tidyms_path()

    download_dir = pathlib.Path(download_dir) / name
    download_dir.mkdir(exist_ok=True, parents=True)

    for file_content in dataset_files:
        file_path = download_dir / file_content.name
        if file_path.is_file():
            continue

        if file_content.content is None:
            file_content.content = get_github_repo_contents(owner, repo, file_content.path)["content"]

        file_path.write_text(file_content.content)


def get_dataset_file(dataset: str, file: str, download_path: pathlib.Path | None = None) -> pathlib.Path:
    """Get the path to an example dataset file."""
    if download_path is None:
        download_path = get_tidyms_path()

    download_dataset(dataset, download_path)

    path = download_path / dataset / file

    if not path.is_file():
        msg = f"File {file} not found in dataset {dataset}."
        raise ExampleDatasetNotFound(msg)

    return path


def list_dataset_files(owner: str, repo: str, dataset: str) -> list[DatasetFileContent]:
    """List the files in a directory in a GitHub repository."""
    try:
        contents = get_github_repo_contents(owner, repo, dataset)
    except (requests.HTTPError, ValueError):
        raise ExampleDatasetNotFound(dataset)

    if not isinstance(contents, list):
        raise ExampleDatasetNotFound(dataset)

    return [DatasetFileContent(**x) for x in contents]


def get_github_repo_contents(owner: str, repo: str, path: str) -> Any:
    """
    Get the API response from endpoint `/repos/{owner}/{repo}/contents/{path}`.

    Parameters
    ----------
    owner : str
        the repository owner.
    repo : str
        the repository name
    path : str
        The repository path to list

    Returns
    -------
    dict
        the JSON response

    """
    endpoint = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"

    response = requests.get(endpoint)

    if not response.ok:
        response.raise_for_status()
    return response.json()
