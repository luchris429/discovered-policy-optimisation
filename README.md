# Discovered Policy Optimisation (NeurIPS 2022)

Code for Discovered Policy Optimisation (NeurIPS 2022)

Lu, Chris, Jakub Kuba, Alistair Letcher, Luke Metz, Christian Schroeder de Witt, and Jakob Foerster. "Discovered policy optimisation." Advances in Neural Information Processing Systems 35 (2022): 16455-16468.

[Paper](https://arxiv.org/abs/2210.05639)

[Tweet](https://twitter.com/_chris_lu_/status/1595388750330155010)

[Related Blog](https://chrislu.page/blog/meta-disco/)

Due to the rapid development of JAX's ecosystem it can be difficult for users to precisely set up the environment. We *highly* recommend instead using the [PureJaxRL repository](https://github.com/luchris429/purejaxrl/tree/main) to perform related research. We plan to upload a clean re-implementation of this work there. This repository is for reproducing the original results in the paper.

PureJaxRL is similar to this repository in that it contains end-to-end Jax-vectorised PPO implementations. However, it differs from this repository in many ways -- it uses newer libraries that did not exist at the time that the bulk of this research was performed. Notably, we *already* have an implementation of [DPO there](https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/dpo_continuous_action.py). Interestingly, the underlying PPO implementations (and subsequent Brax environment versions) differ significantly, yet DPO still outperforms PPO.

# Installation

`pip install -r requirements.txt`

Notably, newer versions of `numpy` can break this older version of Jax.

To install Jax with cuda, run

`pip install jax==0.3.0 -f "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"`

`pip install jaxlib==0.3.0 -f "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"`

# Usage

To train DPO on Ant, run (replacing `save-dir`):

`python3 main_drift_brax.py --env="ant" --ppo-init --end-only --save-dir=SAVE_DIR_HERE`

#  Citation

```
@article{lu2022discovered,
    title={Discovered policy optimisation},
    author={Lu, Chris and Kuba, Jakub and Letcher, Alistair and Metz, Luke and Schroeder de Witt, Christian and Foerster, Jakob},
    journal={Advances in Neural Information Processing Systems},
    volume={35},
    pages={16455--16468},
    year={2022}
}
```
