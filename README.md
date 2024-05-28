# CF-OPT

This is the official GitHub repository of the ICML'24 article __CF-OPT: Counterfactual Explanations for Structured Prediction__. 

Many experimental settings and parts of the code are adapted from the PyEPO repository https://github.com/khalil-research/PyEPO by __Bo Tang__, __Elias B.__ and __Khalil__.

## Presentation
Optimization layers in deep neural networks have gained popularity in structured learning, significantly improving the state of the art across various applications. However, these pipelines often lack interpretability due to their composition of two opaque layers: a highly non-linear prediction model (such as a deep neural network) and an optimization layer (typically a complex black-box solver).

Our primary goal is to enhance the transparency of such methods by providing counterfactual explanations of their outputs. To achieve this, we leverage variational autoencoders (VAEs) to obtain meaningful counterfactuals. By operating in the latent space, we establish a natural measure of plausibility for explanations.

![pipeline_vae_last](https://github.com/GermainVivierArdisson/CF-OPT/assets/102970346/ceb76aae-5f09-4513-8b7b-9a324ff74931)
*Full pipeline for generating counterfactual explanations of the shortest paths on Warcraft maps structured learning pipeline.*

## Usage

The notebooks `Workflow_Warcraft`, `Workflow_SPG` and `Workflow_Knapsack` explain in detail how to run CF-OPT, and allow for easy reproduction of all of the numerical experiments presented in the article.

## Installation

First, create your conda environment from the `envCFOPT.yml` file by running:

```
conda env create -f envCFOPT.yml
```

Then, activate it:

```
conda activate envCFOPT
```

Then, install `PyEPO` by running:

```
git clone -b main --depth 1 https://github.com/khalil-research/PyEPO.git
pip install PyEPO/pkg/.
```

Finally, clone this repository and run the provided example notebooks to explore counterfactual explanations for structured prediction pipelines.
