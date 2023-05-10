""" Define all downloader utilities. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import os
import wget
import zipfile


HOME_DIR = os.path.expanduser('~')
DEFAULT_DATADIR = os.path.join(HOME_DIR, 'bandpy_data')

MOVIELENS_URL = ("https://files.grouplens.org/datasets/movielens/"
                 "ml-latest-small.zip")
YAHOO_URL = "https://webscope.sandbox.yahoo.com/catalog.php?datatype=c"


def download_movielens_dataset(data_dir=DEFAULT_DATADIR):
    """Download Movie Lens dataset."""

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(os.path.join(data_dir, 'ml-latest-small.zip')):
        wget.download(url=MOVIELENS_URL, out=data_dir)

    if not os.path.isdir(os.path.join(data_dir, 'ml-latest-small')):

        with zipfile.ZipFile(os.path.join(data_dir, 'ml-latest-small.zip'), 'r') as zip_ref:  # noqa
            zip_ref.extractall(data_dir)

    return os.path.join(data_dir, 'ml-latest-small')


# mock function
def download_yahoo_dataset(data_dir=DEFAULT_DATADIR):
    """Download Yahoo dataset."""

    yahoo_dir = os.path.join(data_dir, 'yahoo', 'ltrc_yahoo')

    if not os.path.isdir(yahoo_dir):
        raise ValueError(f"Yahoo dataset not found under {yahoo_dir}, follow "
                         f"the process to download the dataset from "
                         f"{YAHOO_URL}")

    return yahoo_dir
