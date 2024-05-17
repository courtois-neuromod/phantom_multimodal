phantom_multimodal
==============================

Training multimodal neural networks on the CNeuroMod datasets

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for this project.
    ├── checkpoints        <- Where to save model checkpoints
    ├── data               <- Where to save raw and processed datasets
    │    ├── fmri
    │    │     ├── friends.fmriprep     <- CNeuroMod fMRI datasets processed with fmriprep
    │    │     └── friends.timeseries   <- Timeseries extracted from CNeuroMod fMRI datasets
    │    └── stimuli
    │           ├── friends.stimuli     <- movies as .mkv files
    │           └── friends.features    <- features derived from raw input stimuli, e.g., language annotations
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── phantom_multimodal      <- Source code for this project.
    │   ├── __init__.py         <- Makes phantom_multimodal a Python module
    │   ├── run.py              <- Main script to run a task
    │   │
    │   ├── data           <- Datamodule scripts
    │   │     └── datamodule.py
    │   │
    │   ├── losses         <- Loss functions
    │   │     └── loss.py
    │   │
    │   ├── models          <- Model architectures
    │   │     ├── litmodule.py
    │   │     └── modules   <- Model components
    │   │           ├── encoders   <- Unimodal encoders
    │   │           │      ├── audio
    │   │           │      ├── brain
    │   │           │      ├── language
    │   │           │      └── vision
    │   │           └── fusion     <- Fusion layers
    │   │
    │   ├── optimizers
    │   │
    │   ├── tasks       <- .yaml config files to specify tasks
    │   │     ├── conf
    │   │     └── base.yaml
    │   │
    │   └── utils                   <- supporting scripts
    │         ├── lightning.py
    │         ├── misc.py
    │         ├── runner.py         <- Task runner class
    │         └── wandb.py          <- Weights & Biases scripts
    └── wandb
          └── WANDB_KEY.txt    <- save your Weights & Biases key here

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
