from setuptools import setup, find_packages

setup(
    name='tlayers',
    packages=find_packages(exclude=[]),
    version='0.0.1',
    license='MIT',
    description='Most common neural layers on PyTorch',
    author='Daniil Korolev',
    url='https://github.com/Danielto1404/tlayers',
    keywords=[
        'Artificial Intelligence',
        'Deep Learning',
        'PyTorch',
        'Computer Vision',
        'NLP'
    ],
    install_requires=[
        'torch',
        'einops'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3.9',
    ],
)
