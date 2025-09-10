from setuptools import setup, find_packages

# Build package list so 'src' becomes the top-level package name and its
# subpackages are installed under it. This lets code import `src.api...`.
subpackages = find_packages(where='src')
packages = ['src'] + [f'src.{p}' for p in subpackages]

setup(
    name='remorph-backend',
    version='0.0.1',
    packages=packages,
    package_dir={'src': 'src'},
    include_package_data=True,
)
