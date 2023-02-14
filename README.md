## Personalized Federated Learning using Hypernetworks [ICML 2021]
This is an official implementation of ***Personalized Federated Learning using Hypernetworks*** paper. [[Link]](https://arxiv.org/abs/2103.04628)

![](resources/pfedhn_arch.png)

#### Installation
- Create a virtual environment with conda/virtualenv
- Clone the repo
- Run: ```cd <PATH_TO_THE_CLONED_REPO>```
- Run: ```pip install -e .``` to install necessary packages and path links.

---------

#### Reproduce Paper Results

---------
##### PfedHN Results on CIFAR10
- Run: ```cd experiments/pfedhn```
- Run: ```python trainer.py```

---------

##### PfedHN-PC Results on CIFAR10
- Run: ```cd experiments/pfedhn_pc```
- Run: ```python trainer.py```

#### Citation

If you find pFedHN to be useful in your own research, please consider citing the following paper:

```bib
@inproceedings{shamsian2021personalized,
  title={Personalized federated learning using hypernetworks},
  author={Shamsian, Aviv and Navon, Aviv and Fetaya, Ethan and Chechik, Gal},
  booktitle={International Conference on Machine Learning},
  pages={9489--9502},
  year={2021},
  organization={PMLR}
}
```
