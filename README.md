
<div align="center">
        <picture>
                <source media="(prefers-color-scheme: dark)" srcset="./docs/assets/logo.svg">
                <source media="(prefers-color-scheme: light)" srcset="./docs/assets/logo.svg">
                <img alt="Library Banner" src="./docs/assets/logo.svg">
        </picture>
</div>

<br>

<div align="center">
  <a href="#">
        <img src="https://img.shields.io/badge/Python-%E2%89%A53.9-efefef">
    </a>
    <a href="https://github.com/ducoffeM/jacobinet/actions/workflows/python-tests.yml">
        <img alt="Tox" src="https://github.com/ducoffeM/jacobinet/actions/workflows/python-tests.yml/badge.svg">
    </a>
    <a href="https://github.com/ducoffeM/jacobinet/actions/workflows/python-linters.yml">
        <img alt="Lint" src="https://github.com/ducoffeM/jacobinet/actions/workflows/python-linters.yml/badge.svg">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/License-MIT-efefef">
    </a>
    <br>
    <a href="https://ducoffeM.github.io/jacobinet/"><strong>Explore Jacobinet docs »</strong></a>
</div>
<br>


## 👋 Welcome to jacobinet documentation!
***JacobiNet*** is a Python library built on Keras that constructs the backward pass as a Keras model using the chain rule to compute Jacobians. It provides a clear, modular framework for understanding and analyzing gradient propagation, making it a valuable tool for researchers and educators in deep learning.
It enables researchers and developers to efficiently analyze, manipulate, and utilize the Jacobian structure of neural models in downstream tasks.


## 📚 Table of contents

- [📚 Table of contents](#-table-of-contents)
- [🔥 Tutorials](#-tutorials)
- [🚀 Quick Start](#-quick-start)
- [📦 What's Included](#-whats-included)
- [👍 Contributing](#-contributing)
- [🙏 Acknowledgments](#-acknowledgments)
- [📝 License](#-license)

## 🚀 Quick Start

You can install ``jacobinet`` directly from pypi:

```python
pip install jacobinet
```

In order to use ``jacobinet``, you also need a [valid Keras
installation](https://keras.io/getting_started/). ``jacobinet``
supports Keras versions 3.x.

## 🔥 Tutorials

| **Tutorial Name**           | Notebook                                                                                                                                                           |
| :-------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Getting Started 1 - Computing and Visualizing Gradients Using Backward Models in Keras | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ducoffeM/jacobinet/blob/main/tutorials/PlottingBackward.ipynb)            |
| Getting Started 2 - Implementing Custom Backward Pass for Non-Native Keras Operators with Jacobinet | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ducoffeM/jacobinet/blob/main/tutorials/CustomOp.ipynb)            |
| Estimating the Local Lipschitz Constant of a Neural Network Using Jacobinet | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ducoffeM/jacobinet/blob/main/tutorials/LipschitzConstant.ipynb) |
| Training Neural Networks with Sparse Input Decision Using Jacobinet | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ducoffeM/jacobinet/blob/main/tutorials/TrainWithSparsity.ipynb) |
| Robust Training with Jacobinet and Adversarial Attacks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ducoffeM/jacobinet/blob/main/tutorials/AdversarialTraining.ipynb) |
| A Complete Guide to Jacobinet Backward Model Serialization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ducoffeM/jacobinet/blob/main/tutorials/Serialization_Export.ipynb) |


## 📦 What's Included

* Neural Jacobian Encoding: Parse a neural network and construct a neural network that represents its Jacobian.

* Seamless Integration with Keras: Built on Keras, ensuring compatibility with TensorFlow and Keras models.

* Customizable Layers: Allows users to specify which layers to include when computing the Jacobian.

* Efficient Computation: Optimized for performance, leveraging GPU acceleration for large-scale models.

* Extensibility: Easy to integrate with existing workflows and extend to custom use cases.

Documentation is available [**online**](https://ducoffeM.github.io/jacobinet/index.html).



## 👍 Contributing

#To contribute, you can open an
#[issue](https://github.com/deel-ai/deel-lip/issues), or fork this
#repository and then submit changes through a
#[pull-request](https://github.com/deel-ai/deel-lip/pulls).
We use [black](https://pypi.org/project/black/) to format the code and follow PEP-8 convention.
To check that your code will pass the lint-checks, you can run:

```python
tox -e py36-lint
```

You need [`tox`](https://tox.readthedocs.io/en/latest/) in order to
run this. You can install it via `pip`:

```python
pip install tox
```


## 🙏 Acknowledgments

<div align="right">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://share.deel.ai/apps/theming/image/logo?useSvg=1&v=10"  width="25%" align="right">
    <source media="(prefers-color-scheme: light)" srcset="https://www.deel.ai/wp-content/uploads/2021/05/logo-DEEL.png"  width="25%" align="right">
    <img alt="DEEL Logo" src="https://www.deel.ai/wp-content/uploads/2021/05/logo-DEEL.png" width="25%" align="right">
  </picture>
</div>
This project received funding from the French program within the <a href="https://aniti.univ-toulouse.fr/">Artificial and Natural Intelligence Toulouse Institute (ANITI)</a>. The authors gratefully acknowledge the support of the <a href="https://www.deel.ai/"> DEEL </a> project.


## 🗞️ Citation



## 📝 License

The package is released under <a href="https://choosealicense.com/licenses/mit"> MIT license</a>.
