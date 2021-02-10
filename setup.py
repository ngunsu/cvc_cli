import setuptools


setuptools.setup(
    name="cv-cli",
    version=0.1,
    author="Cristhian Aguilera",
    author_email="svarogcl@gmail.com",
    url="stemx.cl",
    scripts=['scripts/cv-convert'],
    packages=['cv_cli'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'click>=7.1.2',
        'joblib>=1.0',
        'opencv-python>=4.0'
    ]
)
