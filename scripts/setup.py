from setuptools import setup, find_packages

setup(
    name='TSLG',
    packages=find_packages(),
    install_requires=[
        'pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@stable',
        'torchaudio',
        'transformers[torch]',
        'matplotlib',
        'matplotlib-inline',
        'ipython',
        'ipykernel',
        'ffmpeg-python',
        'opencv-python',
        'pycocotools',
        'grpcio',
        'scipy==1.6.0',
        'pyyaml>=5.3',
    ]
)