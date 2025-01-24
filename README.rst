.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly
    .. image:: https://img.shields.io/pypi/v/torchGB.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/torchGB/
    .. image:: https://pepy.tech/badge/torchGB/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/torchGB


=======
torchGB
=======

.. image:: https://readthedocs.org/projects/torchGB/badge/?version=latest
    :alt: ReadTheDocs
    :target: https://torchGB.readthedocs.io/en/latest

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/


What is **torchGB**?
====================

**torchGB** is a highly parallel PyTorch implementation of the genomic bottleneck
for many different architectures such as CNNs and Transformers. The genomic 
bottleneck is in essence an input- **independent** hypernetwork that predicts the
parameters/weights of the phenotype network, i.e. the model we want to compress.


Terminology
===========
The **p-net** or phenotype network is the model that we intend to compress. A
rough analogy in neuroscience is the brain of an animal or human. The **g-net**
or genomic network is the model or rather an assortment of models that compress
the weights of the p-net, i.e. the hypernetwork. You can think of it as the 
genome of a animal or human which roughly encodes the base wiring patterns.


Quick Start
===========
**torchGB** is straight-forward to use in common scenarios. It also supports 
distributed data parallelism out of the box with ``torch.distributed`` library.
It is however sufficient to implement your model so that it works on a single 
node. **torchGB** takes care of the rest, but it requires some adjustments to the
training scripts. We start by additionally importing the parallelization library:

..  code-block:: python
    :caption: Required imports

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torchGB import GenomicBottleneck


Then we need to initialize the multiprocessing environment, e.g. **nccl** or **gloo**:

..  code-block:: python
    :caption: Setting up the multiprocessing

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()


Then we set up our model and wrap it with the genomic bottleneck.
Note that it is important to move the model to the
specific device rank after creation to guarantee proper parallelization. This is
done using the ``.to(rank)`` call as is default in PyTorch. Then we have to wrap
it with a ``DistributedDataParallel`` class that takes care of the parallelization
for us. Finally, we create the genomic networks from the model by wrapping the
model in the ``GenomicBottleneck`` class:

..  code-block:: python
    :caption: Wrapping the model in a genomic bottleneck

    model = GPT(**experiment_config["model"]).to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)
    gnets = GenomicBottleneck(model, num_batches, **experiment_config["gnets"])


The ``gnets`` object contains an assortment of MLPs or similar architectures.
Since out goal is to train them all at once, we need to call them in the same way
we would do for a normal PyTorch model. To avoid total chaos when managing all
of these small network and to parallelize training across devices, **torchGB**
reimplements it's own ``.zero_grad()``, ``.backward()``, ``.step()`` etc. that
take care of all that under the hood. It is imperative to also call the respective
methods also on the ``model`` so that we compute the gradients of the p-net first
so that we can propagate them to the g-nets. Here is some pseudo-code for a common
training setup:

..  code-block:: python
    :caption: Common training setup 

    #...
    for data, labels in data_loader:
        model.train()
        data = data.to(rank) # Important so that the data is moved to the correct device
        labels = labels.to(rank)
        
        # Zeroing out the gradients in the p-net and g-net optimizers
        optimizer.zero_grad()
        gnets.zero_grad()
        gnets.predict_weights() # implicitly updates the model weights with g-nets!

        output = model(data)
        loss = criterion(output, labels)

        # Backpropagate the error through the p-net and then through the g-nets
        loss.backward()
        gnets.backward()
        
        # Do a gradient-descent step with the p-nets and then the g-nets
        optimizer.step()
        gnets.step()
    # ...


The key here is the ``.predict_weights()`` method which automatically uses the 
g-nets to compute the weights of the p-net, which then is used to make predictions,
compute errors and then backpropagate the errors through both p-net and g-nets.


Running a Program
=================
**torchGB** relies on a highly parallelized implementation that distributes the
g-nets evenly across the available hardware using the ``torch.distributed`` 
library. Thus, launching the model requires the use of the ``torchrun`` binary
instead of the usual ``python`` binary. Here is an example launch:

..  code-block:: bash
    :caption: Example run command with torchrun

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_llm_gnet_small.py \
    --gpus 1,2,3,4 --seed 42 --language en --batchsize 36 \
    --name test --no_commit --log_level DEBUG

Note that it is imperative to use the ``--nproc_per_node=4`` argument to enable
the proper distribution of the workload. To learn more about ``torch.distributed``,
look `here <https://pytorch.org/docs/stable/distributed.html>`_.
For ``torchrun`` specifically, check out `this link <https://pytorch.org/docs/stable/elastic/run.html>`_.


Installation
============

There no **PyPI** package available yet. The project has to be installed with
``pip`` directly from source using:

.. code-block:: python
    :caption: Installation of the package with pip directly from GitHub

    pip install git+https://github.com/jamielohoff/torchGB.git


Clearly the project also needs the most recent version of PyTorch installed. You
can find it `here <https://pytorch.org>`_ and install it with ``pip``. 


Reproducibility
===============
To reproduce the results in the paper and run the scripts in the ``experiments``
folder, you additionally need to install the following packages:

+-------------+---------+
|package      |version  |
+-------------+---------+
|torch        |>= 2.5.1 |
+-------------+---------+
|seaborn      |>= 0.13.2|
+-------------+---------+
|matplotlib   |>= 3.10.0|
+-------------+---------+
|wandb        |>= 0.19.4|
+-------------+---------+
|tqdm         |>= 4.67.1|
+-------------+---------+
|transformers |>= 4.48.1|
+-------------+---------+
|datasets     |>= 3.2.0 |
+-------------+---------+
|loguru       |>= 0.7.3 |
+-------------+---------+
|torchdata    |>= 0.10.1|
+-------------+---------+

.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.

