# The Lifecycle of a TwoViewResult

This document provides a concise overview of how [`TwoViewResult`](../products/two_view_result.py) objects flow through the GTSFM pipeline.

## Overview

A `TwoViewResult` is created for each image pair during two-view estimation and contains:
- Relative pose estimates (rotation, translation)
- Verified correspondences 
- Bundle adjustment results
- Processing reports and metrics

## Data Flow

```mermaid
graph TD
    A[CorrespondenceGenerator] --> B(putative_correspondences)
    B --> C[TwoViewEstimator.run_2view]
    C --> D(TwoViewResult)
    
    D --> E[SceneOptimizer]
    E --> F(AnnotatedGraph&lt;TwoViewResult&gt;)
    
    F --> G[ViewGraphEstimator]
    G --> H(filtered_two_view_results)
    
    H --> I{Data Extraction}
    I --> J(i2Ri1_dict)
    I --> K(i2Ui1_dict)
    I --> L(verified_correspondences)
    
    J --> M[RotationAveraging]
    M --> N(global_rotations)
    
    K --> O[TranslationAveraging]
    N --> O
    O --> P(global_translations)
    
    L --> Q[DataAssociation]
    N --> Q
    P --> Q
    Q --> R(tracks_3d)
    
    N --> S[GlobalBundleAdjustment]
    P --> S
    R --> S
    
    S --> T(optimized_poses)
    S --> U(optimized_tracks)
    
    classDef processBox fill:#e3f2fd,stroke:#1976d2,stroke-width:3px,font-size:12px
    classDef productBox fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,font-size:10px
    classDef collectionBox fill:#fff3e0,stroke:#f57c00,stroke-width:2px,font-size:11px
    
    class A,C,E,G,I,M,O,Q,S processBox
    class B,D,H,J,K,L,N,P,R,T,U productBox
    class F collectionBox
```

## Pipeline Dependencies

1. **[`RotationAveraging.run_rotation_averaging()`](../averaging/rotation/rotation_averaging_base.py)** takes `i2Ri1_dict` and produces `wRi_list`
2. **[`TranslationAveraging.run_translation_averaging()`](../averaging/translation/translation_averaging_base.py)** takes `i2Ui1_dict` AND `wRi_list` from rotation averaging
3. **[`DataAssociation`](../data_association/data_association_base.py)** takes `verified_correspondences`, `wRi_list`, AND `wti_list` from translation averaging
