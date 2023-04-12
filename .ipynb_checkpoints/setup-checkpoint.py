import setuptools

setuptools.setup(
    name = 'spikewaveformfilter',
    version = '0.0.2',
    author = 'RaisoLiu',
    author_email = 'raisoliu@gmail.com',
    description = 'Eliminate the units with waveforms that resemble fake waveforms from Kilosort2.',
    packages = setuptools.find_packages(),
    classifiers = [
    ],
    install_requires = [
        "numpy",
        "torch",
    ],
    python_requires = ">=3.7",
    include_package_data=True,
    package_data={'': ['model_pt/*.pt']},

)