torchGB
=======

[![ReadTheDocs](https://readthedocs.org/projects/torchGB/badge/?version=latest)](https://torchGB.readthedocs.io/) [![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)



What is **torchGB**?
================

**torchGB** is a highly parallel PyTorch implementation of the [genomic bottleneck](https://www.pnas.org/doi/abs/10.1073/pnas.2409160121)
for many different architectures such as CNNs and Transformers. The genomic 
bottleneck is in essence an input- **independent** hypernetwork that predicts the
parameters/weights of the phenotype network, i.e. the model we want to compress.


Quick Start
===========


Running a Program
=================

**torchGB** relies on a highly parallelized implementation that distributes the
g-nets evenly across the available hardware using the ``torch.distributed`` 
library. Thus, launching the model requires the use of the ``torchrun`` binary
instead of the usual ``python`` binary. Here is an example launch:

<pre>
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_llm_small.py \
    --gpus 1,2,3,4 --seed 42 --language en --batchsize 36 \
    --name test --no_commit --log_level DEBUG
```
</pre>

Note that it is imperative to use the ``--nproc_per_node=4`` argument to enable
the proper distribution of the workload. To learn more about ``torch.distributed``,
look [here](https://pytorch.org/docs/stable/distributed.html).
For ``torchrun`` specifically, check out [this link](https://pytorch.org/docs/stable/elastic/run.html).



Installation
============

There us no PyPI_ package available yet. The project has to be installed with
``pip`` directly from source using:

<pre>
```python
pip install git+https://github.com/jamielohoff/torchGB.git
```
</pre>


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.

