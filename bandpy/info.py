""" Info module: required package version and utilities for checking to help
install Bandpy package. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.fr>

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
# X.Y
# X.Y.Z # For bugfix releases
#
# Admissible pre-release markers:
# X.YaN # Alpha release
# X.YbN # Beta release
# X.YrcN # Release Candidate
# X.Y # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'

from distutils.version import LooseVersion


__version__ = '0.1.dev'

_BANDPY_GITHUB_REPO_URL = ('https://rnd-gitlab-eu.huawei.com/Noahs-Ark/'
                           'libraries/bandpy')
_BANDPY_INSTALL_MSG = (f"See {_BANDPY_GITHUB_REPO_URL} for installation"
                       f" information.")

# This is a tuple to preserve order, so that dependencies are checked
#   in some meaningful order (more => less 'core').
REQUIRED_MODULE_METADATA = (
    ('matplotlib', {
        'min_version': '3.0.0',
        'required_at_installation': True,
        'install_info': _BANDPY_INSTALL_MSG}),
    ('numpy', {
        'min_version': '1.10.0',
        'required_at_installation': True,
        'install_info': _BANDPY_INSTALL_MSG}),
    ('pandas', {
        'min_version': '1.4.1',
        'required_at_installation': True,
        'install_info': _BANDPY_INSTALL_MSG}),
    ('scipy', {
        'min_version': '1.8.0',
        'required_at_installation': True,
        'install_info': _BANDPY_INSTALL_MSG}),
    ('joblib', {
        'min_version': '0.16.0',
        'required_at_installation': True,
        'install_info': _BANDPY_INSTALL_MSG}),
    ('scikit-learn', {
        'min_version': '1.0.2',
        'import_name': 'sklearn',
        'required_at_installation': True,
        'install_info': _BANDPY_INSTALL_MSG}),
    ('networkx', {
        'min_version': '2.8.6',
        'required_at_installation': True,
        'install_info': _BANDPY_INSTALL_MSG}),
    ('gym', {
        'min_version': '0.23.1',
        'required_at_installation': True,
        'install_info': _BANDPY_INSTALL_MSG}),
    ('matrix_factorization', {
        'min_version': '0.0.0',  # no version avalaible
        'required_at_installation': True,
        'install_info': _BANDPY_INSTALL_MSG}),
    )


def _import_module_with_version_check(module_name, minimum_version,
                                      install_info=None):
    """Private helper, check that module is installed with a recent enough
    version.
    Parameters
    ----------
    module_name : str, module name
    minimum_version : str, minimum version required
    install_info : str or None, (default=None), message to install it if
        installation failed
    Return
    ------
    module : Python module, the imported module
    """
    try:
        module = __import__(module_name)
    except ImportError as exc:
        msg = 'Please install it properly to use Bandpy.'
        user_friendly_info = (f"Module '{module_name}' could not be found. "
                              f"{install_info or msg}")
        exc.args += (user_friendly_info,)
        raise

    # Avoid choking on modules with no __version__ attribute
    module_version = getattr(module, '__version__', '0.0.0')

    version_too_old = (not LooseVersion(module_version) >=
                       LooseVersion(minimum_version))

    if version_too_old:
        message = (f"A {module_name} version of at least {minimum_version} "
                   f"is required to use Bandpy. {module_version} was "
                   f"found. Please upgrade {module_name}")

        raise ImportError(message)

    return module


def _check_module_dependencies(is_bandpy_installing=False):
    """Throw an exception if Bandpy dependencies are not installed.
    Parameters
    ----------
    is_bandpy_installing: boolean
        if True, only error on missing packages that cannot be auto-installed.
        if False, error on any missing package.
    Throws
    ------
    ImportError : if a dependencie is not installed.
    """

    for (module_name, module_metadata) in REQUIRED_MODULE_METADATA:
        if not (is_bandpy_installing and
           not module_metadata['required_at_installation']):
            # Skip check only when installing and it's a module that
            # will be auto-installed.
            if 'import_name' in module_metadata.keys():
                module_name = module_metadata['import_name']
            _import_module_with_version_check(
                module_name=module_name,
                minimum_version=module_metadata['min_version'],
                install_info=module_metadata.get('install_info'))
