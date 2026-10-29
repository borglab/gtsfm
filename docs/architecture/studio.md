# GTSFM Studio: application flow before and after the UI refactor

**Before** refers to [PR #1166](https://github.com/borglab/gtsfm/pull/1166); **after** refers to [PR #1168](https://github.com/borglab/gtsfm/pull/1168).

## Application context

The browser renders the 3D scene on the client device. The server serves the application, manages jobs, and runs reconstruction. Optional remote/Modal execution is omitted for clarity.

**Color key:** blue = browser; amber = server; red = failure path. Labeled boundaries also identify where each component runs.

```mermaid
flowchart TB
    subgraph Browser["Browser"]
        UI["React interface"]
        View["Babylon renderer / WebGL"]
        UI <-->|"Commands and view state"| View
    end

    subgraph Server["Server"]
        API["FastAPI workspace"]
        Jobs["JobManager"]
        Runner["Reconstruction or splat-only process"]
        Files[("Uploaded datasets, results, logs and job history")]

        API -->|"Start, retry, stop"| Jobs
        Jobs -->|"Launch and monitor"| Runner
        Runner -->|"Write results and progress"| Files
        Jobs <-->|"Persist and restore history"| Files
        API -->|"Discover and serve files"| Files
    end

    UI <-->|"HTTP requests and WebSocket updates"| API
    API -->|"Geometry, splats and photos"| View

    classDef browser fill:#e8f1ff,stroke:#2563eb,color:#123765,stroke-width:2px;
    classDef server fill:#fff0d9,stroke:#b45309,color:#78350f,stroke-width:2px;
    class UI,View browser;
    class API,Jobs,Runner,Files server;
    style Browser fill:#f4f8ff,stroke:#2563eb,color:#123765,stroke-width:2px
    style Server fill:#fff8ed,stroke:#b45309,color:#78350f,stroke-width:2px
```

HTTP and WebSocket connections link the browser and server. Reconstruction compute stays on the server; browser rendering uses the client device's graphics hardware. PR #1168 changes the browser's internal organization, not this deployment model.

## Before: React and viewer.js share interface ownership

```mermaid
flowchart TB
    subgraph Browser["Browser"]
        Start["HTML bootstrap loads React, Babylon and viewer.js"]
        React["React App: forms, activity and viewer markup"]
        Boot["viewer.js boot: create Babylon viewer"]
        DOM["Shared controls, statistics and results DOM"]
        Outcome{"Results request succeeds?"}
        Bind["Build results tree and attach control handlers"]
        Stop["Return early: controls remain unwired"]
        Render["Render geometry and update interface DOM"]
        ListWS["Job-list WebSocket client: reconnects"]
        JobWS["Selected-job WebSocket client: no reconnect"]
    end

    subgraph Server["Server"]
        Scenes["GET /api/scenes"]
        Events["Job-list and selected-job WebSocket endpoints"]
    end

    Start --> React
    Start --> Boot
    React -->|"Creates elements"| DOM
    Boot --> Scenes --> Outcome
    Outcome -->|"Yes"| Bind
    Outcome -->|"No"| Stop
    Bind -->|"Mutates elements and binds events"| DOM
    DOM -->|"Viewer handlers"| Render
    Render -->|"Writes labels, classes and progress"| DOM
    React <-->|"Global viewer reference and custom events"| Render
    Events --> ListWS --> React
    Events --> JobWS --> React

    classDef browser fill:#e8f1ff,stroke:#2563eb,color:#123765,stroke-width:2px;
    classDef server fill:#fff0d9,stroke:#b45309,color:#78350f,stroke-width:2px;
    classDef failure fill:#fee2e2,stroke:#dc2626,color:#7f1d1d,stroke-width:2px;
    class Start,React,Boot,DOM,Outcome,Bind,Render,ListWS,JobWS browser;
    class Scenes,Events server;
    class Stop failure;
    style Browser fill:#f4f8ff,stroke:#2563eb,color:#123765,stroke-width:2px
    style Server fill:#fff8ed,stroke:#b45309,color:#78350f,stroke-width:2px
```

The first 3D frame could appear before result discovery finished, but the controls were attached only afterward. Result discovery ran once per page, and parsed geometry caches used file URLs without a file revision. React and the renderer communicated through a global viewer reference and custom events.

## After: React owns the interface; Babylon owns rendering

```mermaid
flowchart TB
    subgraph Browser["Browser"]
        Start["HTML bootstrap loads interface and engine separately"]
        subgraph ReactUI["React owns interface state and DOM"]
            Canvas["ViewerCanvas: create and dispose renderer"]
            Controls["Controls, statistics, loading and photos"]
            Results["ResultsBrowser: catalog, filter, tree and selection"]
            Activity["App: run configuration, activity and live progress"]
            Context["ViewerProvider: typed API and state subscriptions"]
            Streams["useJobStream: shared recovery and cleanup"]
        end
        Renderer["ColmapViewer / Babylon: geometry, camera, caches and rendering"]
    end

    subgraph Server["Server"]
        Scenes["GET /api/scenes: catalog plus file revisions"]
        FileAPI["Geometry, splat and photo endpoints"]
        Events["Job-list and selected-job WebSocket endpoints"]
    end

    Start --> Canvas
    Start --> Activity
    Canvas -->|"Create on mount; dispose on unmount"| Renderer
    Canvas -->|"Register renderer"| Context
    Controls -->|"Typed commands"| Context
    Results -->|"Open versioned resource URLs"| Context
    Activity -->|"Load live or final splat"| Context
    Context -->|"Viewer API calls"| Renderer
    Renderer -->|"State and photo subscriptions"| Context
    Context -->|"Render updated UI"| Controls
    Context -->|"Busy transitions only"| Results

    Results <-->|"Initial load, manual refresh, run completion"| Scenes
    Renderer <-->|"Fetch resources"| FileAPI
    Events --> Streams --> Activity
    Activity -->|"Finished-run state changes"| Results

    classDef browser fill:#e8f1ff,stroke:#2563eb,color:#123765,stroke-width:2px;
    classDef server fill:#fff0d9,stroke:#b45309,color:#78350f,stroke-width:2px;
    class Start,Canvas,Controls,Results,Activity,Context,Streams,Renderer browser;
    class Scenes,FileAPI,Events server;
    style Browser fill:#f4f8ff,stroke:#2563eb,color:#123765,stroke-width:2px
    style ReactUI fill:#dbeafe,stroke:#60a5fa,color:#123765
    style Server fill:#fff8ed,stroke:#b45309,color:#78350f,stroke-width:2px
```

Viewer initialization and control handlers no longer depend on the results request. Catalog failures stay in the results panel and can be retried. Changed file revisions produce new cache keys when a result is reopened; refreshing the catalog does not automatically replace the scene being inspected.

Both WebSocket connections use the same reconnection hook and report interruptions. Frequent renderer updates are confined to the viewer UI: the results browser subscribes only to busy-state changes, and the run form does not subscribe to streamed point counts.

## Loading order: structure becomes usable before photos

This progressive loading behavior existed before the refactor and is preserved.

```mermaid
flowchart TB
    subgraph Browser["Browser"]
        Open["Select a reconstruction"]
        Load["Renderer starts scene load"]
        Points["Receive point stream"]
        Batches["Parse batches and grow point buffers"]
        Display["Display points progressively with fly-in animation"]
        Ready["Finish point stream: structure ready"]
        Metadata["Receive camera poses and intrinsics"]
        Cameras["Attach camera geometry when metadata and points are ready"]
        Photos["Load visible thumbnails and textures"]
    end

    subgraph Server["Server"]
        PointFile["points3D.txt"]
        CameraFiles["images.txt and cameras.txt"]
        PhotoAPI["Photo manifest and image endpoints"]
    end

    Open --> Load
    Load -->|"Request points"| PointFile
    PointFile --> Points --> Batches --> Display --> Ready
    Load -->|"Request camera metadata in parallel"| CameraFiles
    CameraFiles --> Metadata --> Cameras
    Ready --> Cameras
    Cameras -->|"Request photos"| PhotoAPI
    PhotoAPI --> Photos

    classDef browser fill:#e8f1ff,stroke:#2563eb,color:#123765,stroke-width:2px;
    classDef server fill:#fff0d9,stroke:#b45309,color:#78350f,stroke-width:2px;
    class Open,Load,Points,Batches,Display,Ready,Metadata,Cameras,Photos browser;
    class PointFile,CameraFiles,PhotoAPI server;
    style Browser fill:#f4f8ff,stroke:#2563eb,color:#123765,stroke-width:2px
    style Server fill:#fff8ed,stroke:#b45309,color:#78350f,stroke-width:2px
```

Neither camera metadata nor image downloads block the point stream. Splat files follow a separate loading path; when a live splat updates, the existing splat stays visible while its replacement loads.

## What changed, and where

| Responsibility | Before | After |
| --- | --- | --- |
| Viewer controls and status | React markup, imperative viewer handlers and DOM updates | React components in `viewer-controls.tsx` |
| Renderer lifetime and UI boundary | Global viewer created by `boot()` | `ViewerCanvas` and typed context in `viewer-context.tsx` |
| Result catalog and selection | One-time fetch and HTML tree inside `viewer.js` | Refreshable React `ResultsBrowser` |
| Geometry cache freshness | URL alone | URL plus result-file revision |
| Live connection recovery | Separate socket implementations | Shared `useJobStream` hook |
| Manual GPU selection | Recommendation effect overwrote the selection | Explicit choice persists; recommendation remains available |
| Browser regression coverage | Geometry tests without normal browser startup | Production-bundle browser tests and dedicated Studio CI |

The backend still polls job state and scans result files to produce WebSocket updates. This PR does **not** introduce an event-driven backend or redesign `JobManager`. The larger run-form extraction and backend monitoring optimization remain follow-up work.
