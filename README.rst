.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/torchGB.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/torchGB
    .. image:: https://img.shields.io/coveralls/github/<USER>/torchGB/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/torchGB
    .. image:: https://img.shields.io/pypi/v/torchGB.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/torchGB/
    .. image:: https://img.shields.io/conda/vn/conda-forge/torchGB.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/torchGB
    .. image:: https://pepy.tech/badge/torchGB/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/torchGB
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/torchGB


=======
torchGB
=======

.. image:: https://readthedocs.org/projects/torchGB/badge/?version=latest
    :alt: ReadTheDocs
    :target: https://torchGB.readthedocs.io/

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/



What is **torchGB**?
================

**torchGB** is a highly parallel PyTorch implementation of the genomic bottleneck
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

.. code-block:: bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_llm_small.py \
    --gpus 1,2,3,4 --seed 42 --language en --batchsize 36 \
    --name test --no_commit --log_level DEBUG

Note that it is imperative to use the ``--nproc_per_node=4`` argument to enable
the proper distribution of the workload. To learn more about ``torch.distributed``,
look `here <https://pytorch.org/docs/stable/distributed.html>`.
For ``torchrun`` specifically, check out `this link <https://pytorch.org/docs/stable/elastic/run.html>`.



Installation
============

There no **PyPI** package available yet. The project has to be installed with
``pip`` directly from source using:

.. code-block:: python
    pip install git+https://github.com/jamielohoff/torchGB.git


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.

