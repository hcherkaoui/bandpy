""" Define all downloader utilities. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import os
import zipfile

try:
    import wget
except ImportError:
    pass


HOME_DIR = os.path.expanduser("~")
DEFAULT_DATADIR = os.path.join(HOME_DIR, "bandpy_data")

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
YAHOO_URL = "https://webscope.sandbox.yahoo.com/catalog.php?datatype=c"


def download_movielens_dataset(data_dir=DEFAULT_DATADIR):
    """Download Movie Lens dataset."""

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    zip_name = os.path.join(data_dir, "ml-latest-small.zip")
    dir_name = os.path.join(data_dir, "ml-latest-small/")

    if not os.path.exists(zip_name) or not os.path.exists(dir_name):
        wget.download(url=MOVIELENS_URL, out=data_dir)

    if not os.path.isdir(dir_name):
        with zipfile.ZipFile(zip_name, "r") as zip_ref:
            zip_ref.extractall(data_dir)

    return dir_name


# mock function
def download_yahoo_dataset(data_dir=DEFAULT_DATADIR):
    """Download Yahoo dataset."""

    yahoo_dir = os.path.join(data_dir, "yahoo", "ltrc_yahoo")

    if not os.path.isdir(yahoo_dir):
        raise ValueError(
            f"Yahoo dataset not found under {yahoo_dir}, follow "
            f"the process to download the dataset from "
            f"{YAHOO_URL}"
        )

    return yahoo_dir
