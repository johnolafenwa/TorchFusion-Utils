from setuptools import setup,find_packages

setup(name="torchfusion-utils",
      version='0.1.0',
      description='A pytorch helper library for Mixed Precision Training, Initialization, Metrics and More Utilities to simplify training of deep learning models',
      url="https://github.com/johnolafenwa/TorchFusion-Utils",
      author='John Olafenwa and Moses Olafenwa (DeepQuest AI)',
      license='MIT',
      packages= find_packages(),
      zip_safe=False,
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
      )
