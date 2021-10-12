This folder contains cachers for different modules of the front-end. The cachers are designed following the [decorator
pattern](https://en.wikipedia.org/wiki/Decorator_pattern), which allows us to add the caching functionality to the
various implementations with minimal code changes in other classes.

## Why?
- In the deep-learning based front-end configuration for GTSFM, neural-nets are used for both these modules. Executing 
these on a CPU slows down the pipeline by a significant factor and leads to slower iteration.
- Having a cache will speed up local testing, as well as CI runs.

## Design Philosophy
- For developers and users not using cache, they should not encounter this code path anywhere. We do not want to bloat
classes like `SceneOptimizer` and `FeatureExtractor` with cache related code.

## How?
- We use the decorator design pattern for caching objects for both these modules. That is, for `DetectorDescriptor`, we
create a new class which is the subclass of `DetectorDescriptor` and has an underlying actual detector-descriptor, which
it uses in case of cache miss. We went ahead with this design as it involves the least number of changes in other 
classes. No other class except the utility functions are changed.
- There is a config change which needs to happen to use this cache functionality.
- On the CI, we use Github action's cache to reuse data between different runs.


## How to use the cache?
- Enabled in the CI runs by this PR.
- When run locally, the cache will be generated when the user first runs GTSFM. To use precomputed cache for certain 
datasets (door-12, skydio-8, skydio-32, palace-of-fine-arts-281, notre-dame-20, 
2011205_rc3), download the repo [gtsfm-cache](https://github.com/ayushbaid/gtsfm-cache) to a top level folder called 
cache. This repo contains the front-end cache of all the benchmarks in the CI.

## How to regenerate cache?
- CI: Change the cache env variable in `benchmark.yaml`
- Local testing: delete the files manually.

## How to add new data to cache?
- CI: Will be automatically generated, provided we stay below 5 GB total cache limit. Run the benchmarks on master manually.
- Local testing: commit to the gtsfm-cache repo.