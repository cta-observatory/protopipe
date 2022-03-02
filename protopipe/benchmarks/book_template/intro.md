# CTA performance using the _protopipe_ pipeline

This document presents the performance of [protopipe](https://github.com/cta-observatory/protopipe) (vX.Y.Z) for the following simulation settings:

- **Observation site:** 
- **Simulation production:** 
- **Array:** 
- **Zenith angle:** 
- **Azimuth angle:** 
- **Type of signal:** 
- **Observation time(s):** 

## Analysis metadata

The analysis metadata is the following,

```{literalinclude} ../analysis_metadata.yaml
:language: yaml
```

It can be used to retrieve all processed data related to this analysis on the DIRAC Grid File Catalog and reproduce its results.

## Software used
<!-- 
Protopipe version couldbe sufficient since versions are fixed at each release,
but better specifiy 
-->
- [protopipe](https://github.com/cta-observatory/protopipe) vX.Y.Z, 
- [ctapipe](https://github.com/cta-observatory/ctapipe) vX.Y.Z,
- [pyirf](https://github.com/cta-observatory/pyirf) vX.Y.Z,
- ...

## Analysis workflow and management of the simulation dataset

<!-- This section is an example (only this analysis workflow has been tested up to now) -->

The entire simulated production is composed by 3 lists files, one for each particle species (gammas, protons and electrons).

Each of them has been split with `protopipe-SPLIT_DATASET` using the default analysis workflow:

- part of the gamma rays is used to train a machine-learning model for energy estimation,
- part of the gamma rays and part of the protons are used to train a a machine-learning model for particle-type classification (this model depends on the estimated energy, so the gamma rays used to train it have this quantity),
- the rest of the simulation dataset is analyzed completely (both energy and particle-type estimation)
- the rest of the dataset is used to compute the performance of the selected array.

The following table shows precisely how the datasets has been split.
Each sub-dataset is referenced with a particle species and a number describing the analysis step in which it has been used (e.g. `gamma1` or `proton2`).

|Analysis step |   gammas   |  protons  |  electrons  |
| --------- | --------- | --------- | -----------|
| Training for energy model|     10%    |     -     |      -      |
| Training for particle classification model|     10%    |    40%    |      -      |
| Performance calculation |     80%    |    60%    |     100%    |

### Corrupted runs

Among simtel runs in the production, there are some
that couldn't be processed by _ctapipe_ for different reasons (corrupted or incomplete simtel file).

They are listed here for reference.

```{admonition} Warning
:class: warning

Simulation datasets could migrate to tape with time. 
```

#### TRAINING CLASSIFICATION PROTONS
<!-- 

These sections are supposed to be filled with lists of LFNs like,

/vo.cta.in2p3.fr/MC/PROD3/LaPalma/proton/simtel/1605/Data/001xxx/proton_20deg_180deg_run1659___cta-prod3-demo-2147m-LaPalma-baseline.simtel.gz -->

#### DL2 PROTONS


#### DL2 gammas


## Analysis settings


### Image extraction


### Image cleaning and parametrization


### Direction reconstruction


### Energy model


### Particle classification model


### Model estimation


#### Energy


#### Classification


### Cuts optimization and performance estimation

