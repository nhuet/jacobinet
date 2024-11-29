

<!-- Banner -->
<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/banner_light.png">
  <img src="docs/assets/banner_light.png" alt="Puncc" width="90%" align="right">
</picture>
</div>
<br>@inproceedings{},
  title={},
  author={},
  booktitle={},
  year={},
  organization={}
}
<!-- Badges -->
<div align="center">
  <a href="#">
    <img src="https://img.shields.io/badge/Python-3.8 +-efefef">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/License-MIT-efefef">
  </a>
  <a href="https://github.com/deel-ai/puncc/actions/workflows/linter.yml">
    <img alt="PyLint" src="https://github.com/deel-ai/puncc/actions/workflows/linter.yml/badge.svg">
  </a>
  <a href="https://github.com/deel-ai/puncc/actions/workflows/tests.yml">
    <img alt="Tox" src="https://github.com/deel-ai/puncc/actions/workflows/tests.yml/badge.svg">
  </a>
</div>
<br>

# JacobiNet
***JacobiNet*** is a Python library built on Keras that constructs the backward pass as a Keras model using the chain rule to compute Jacobians. It provides a clear, modular framework for understanding and analyzing gradient propagation, making it a valuable tool for researchers and educators in deep learning.
It enables researchers and developers to efficiently analyze, manipulate, and utilize the Jacobian structure of neural models in downstream tasks.

**Features**
-Neural Jacobian Encoding: Parse a neural network and construct a neural network that represents its Jacobian.
-Seamless Integration with Keras: Built on Keras, ensuring compatibility with TensorFlow and Keras models.
-Customizable Layers: Allows users to specify which layers to include when computing the Jacobian.
-Efficient Computation: Optimized for performance, leveraging GPU acceleration for large-scale models.
-Extensibility: Easy to integrate with existing workflows and extend to custom use cases.

Documentation is available [**online**](https://ducoffeM.github.io/jacobinet/index.html).

## 📚 Table of contents

- [Installation](#-installation)
- [Documentation](#-documentation)
- [Tutorials](#-tutorials)
- [QuickStart](#-quickstart)
- [Citation](#-citation)
- [Contributing](#-contributing)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)



## 🐾 Installation

*puncc* requires a version of python higher than 3.8 and several libraries including Scikit-learn and Numpy. It is recommended to install *puncc* in a virtual environment to not mess with your system's dependencies.

You can directly install the library using pip:

```bash
pip install puncc
```


## Documentation

For comprehensive documentation, we encourage you to visit the [**official documentation page**](https://ducoffeM.github.io/jacobinet/index.html).

## Tutorials


We highly recommend following the introductory tutorials to get familiar with the library and its API.

| Tutorial | Description | Link |
|----------|-------------|------|
| **Introduction Tutorial** | Get started with the basics of *puncc*. | [![Open In Github](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](docs/puncc_intro.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/deel-ai/puncc/blob/main/docs/puncc_intro.ipynb) |
| **API Tutorial** | Learn about *puncc*'s API. | [![Open In Github](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](docs/api_intro.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/deel-ai/puncc/blob/main/docs/api_intro.ipynb) |
| **Tutorial on CP with PyTorch** | Learn how to use *puncc* with PyTorch. | [![Open In Github](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](docs/puncc_pytorch.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/deel-ai/puncc/blob/main/docs/puncc_pytorch.ipynb) |
| **Conformal Object Detection** | Learn to conformalize an object detector. | [![Open In Github](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](docs/puncc_cod.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/deel-ai/puncc/blob/main/docs/puncc_cod.ipynb) |
| **Architecture Overview** | Detailed overview of *puncc*'s architecture. | [![Open In Github](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](docs/puncc_architecture.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/deel-ai/puncc/blob/main/docs/puncc_architecture.ipynb) |

## Quickstart

Conformal prediction enables to transform point predictions into interval predictions with high probability of coverage. The figure below shows the result of applying the split conformal algorithm on a linear regressor.

<figure style="text-align:center">
<img src="docs/assets/cp_process.png"/>
</figure>

Many conformal prediction algorithms can easily be applied using *puncc*.  The code snippet below shows the example of split conformal prediction with a pretrained linear model:

 ```python
 from deel.puncc.api.prediction import BasePredictor
from deel.puncc.regression import SplitCP

# Load calibration and test data
# ...

# Pretrained regression model
# trained_linear_model = ...

# Wrap the model to enable interoperability with different ML libraries
trained_predictor =  BasePredictor(trained_linear_model)

# Instanciate the split conformal wrapper for the linear model.
# Train argument is set to False because we do not want to retrain the model
split_cp = SplitCP(trained_predictor, train=False)

# With a calibration dataset, compute (and store) nonconformity scores
split_cp.fit(X_calib=X_calib, y_calib=y_calib)

# Obtain the model's point prediction y_pred and prediction interval
# PI = [y_pred_lower, y_pred_upper] for a target coverage of 90% (1-alpha).
y_pred, y_pred_lower, y_pred_upper = split_cp.predict(X_test, alpha=0.1)
```


The library provides several metrics (`deel.puncc.metrics`) and plotting capabilities (`deel.puncc.plotting`) to evaluate and visualize the results of a conformal procedure. For a target error rate of $\alpha = 0.1$, the marginal coverage reached in this example on the test set is higher than $90$% (see [**Introduction tutorial**](docs/puncc_intro.ipynb)):
<div align="center">
<figure style="text-align:center">
<img src="docs/assets/results_quickstart_split_cp_pi.png" alt="90% Prediction Interval with the Split Conformal Prediction Method" width="70%"/>
<div align=center>90% Prediction Interval with Split Conformal Prediction.</div>
</figure>
</div>
<br>



## 📚 Citation

If you use our library for your work, please cite our paper:

```
@inproceedings{mendil2023puncc,
  title={PUNCC: a Python Library for Predictive Uncertainty Calibration and Conformalization},
  author={Mendil, Mouhcine and Mossina, Luca and Vigouroux, David},
  booktitle={???},
  year={???},
  organization={???}
}
```

*Jacobinet* has been used to support the following works

```
@inproceedings{},
  title={},
  author={},
  booktitle={},
  year={},
  organization={}
}


```


## Acknowledgments

<img align="right" src="https://www.deel.ai/wp-content/uploads/2021/05/logo-DEEL.png" width="25%">
This project received funding from the French ”Investing for the Future – PIA3” program within the Artificial and Natural Intelligence Toulouse Institute (ANITI). The authors gratefully acknowledge the support of the <a href="https://www.deel.ai/"> DEEL </a> project.


## Contributing

Contributions are welcome! Feel free to report an issue or open a pull
request. Take a look at our guidelines [here](CONTRIBUTING.md).

## License

The package is released under [MIT](LICENSES/headers/MIT-Clause.txt) license.
