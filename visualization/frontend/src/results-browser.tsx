import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { RefreshCw } from "lucide-react";
import { useViewerApi, useViewerBusy } from "./viewer-context";

export interface ResultItem {
  kind: "scene" | "splat";
  rel_path: string;
  label: string;
  modified_at?: number;
  revision?: string;
  points: string | null;
  images: string | null;
  cameras?: string | null;
  splats: string | null;
  point_count?: number;
  camera_count?: number;
  splat_count?: number;
  can_splatify?: boolean;
  splat_rel_path?: string;
}
interface Catalog {
  base_dir: string;
  items: ResultItem[];
  directory_modified_at: Record<string, number>;
}
interface ResultNode {
  name: string;
  path: string;
  modified: number;
  item?: ResultItem;
  children: Map<string, ResultNode>;
}

export function resultsForRun(items: ResultItem[], baseDir: string, runRoot: string) {
  const normalize = (path: string) => path.replaceAll("\\", "/").replace(/\/+$/, "");
  const base = normalize(baseDir),
    root = normalize(runRoot);
  if (root !== base && !root.startsWith(`${base}/`)) return [];
  const relative = root === base ? "" : root.slice(base.length + 1);
  return items.filter((item) => !relative || item.rel_path === relative || item.rel_path.startsWith(`${relative}/`));
}

export function preferredRunResult(items: ResultItem[]) {
  const stages = ["merged_retriangulated", "merged", "ba_output", "vggt", "anysplat"];
  const rank = (item: ResultItem) => {
    if (item.kind === "splat") return 0;
    const parts = item.rel_path.split("/");
    const stage = stages.indexOf(parts.at(-1) || "");
    return stage < 0 ? Infinity : 1 + parts.length * stages.length + stage;
  };
  return items.filter((item) => Number.isFinite(rank(item))).sort((a, b) => rank(a) - rank(b))[0];
}

export function resultTree(items: ResultItem[], modified: Record<string, number>) {
  const root = new Map<string, ResultNode>();
  for (const item of items) {
    let children = root;
    const parts = item.rel_path.split("/");
    parts.forEach((name, index) => {
      const path = parts.slice(0, index + 1).join("/");
      let node = children.get(name);
      if (!node) {
        node = { name, path, modified: 0, children: new Map() };
        children.set(name, node);
      }
      node.modified = Math.max(node.modified, modified[path] ?? item.modified_at ?? 0);
      if (index === parts.length - 1) node.item = item;
      children = node.children;
    });
  }
  return root;
}

function sortedNodes(nodes: Map<string, ResultNode>) {
  return [...nodes.values()].sort(
    (a, b) =>
      Number(Boolean(b.item)) - Number(Boolean(a.item)) ||
      b.modified - a.modified ||
      (a.name < b.name ? -1 : a.name > b.name ? 1 : 0),
  );
}

function ModifiedTime({ seconds }: { seconds: number }) {
  const date = new Date(seconds * 1000);
  return seconds && Number.isFinite(date.getTime()) ? (
    <time className="result-modified" dateTime={date.toISOString()} title={date.toString()}>
      Modified{" "}
      {date.toLocaleString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
        second: "2-digit",
      })}
    </time>
  ) : (
    <small className="result-modified">Modification time unavailable</small>
  );
}

/** A revision includes geometry, camera metadata, and attached splats. */
function versioned(url: string | null | undefined, item: ResultItem) {
  if (!url) return null;
  const revision = item.revision ?? String(item.modified_at ?? "");
  return `${url}${url.includes("?") ? "&" : "?"}v=${encodeURIComponent(revision)}`;
}

export function ResultsBrowser({
  onSplatify,
  refreshKey,
}: {
  onSplatify: (item: ResultItem) => void;
  refreshKey: string;
}) {
  const { viewer } = useViewerApi();
  const busy = useViewerBusy();
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [loadError, setLoadError] = useState("");
  const [revision, setRevision] = useState(0);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState("");
  const [collapsed, setCollapsed] = useState(new Set<string>());
  const autoOpened = useRef(false);
  const search = useMemo(() => new URLSearchParams(location.search), []);
  const runRoot = search.get("run");

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    void fetch("/api/scenes", { signal: controller.signal, cache: "no-store" })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Could not load results (${response.status}).`);
        const data = (await response.json()) as Catalog;
        if (!controller.signal.aborted) setCatalog(data);
      })
      .catch((reason) => {
        if (!controller.signal.aborted) setError(String(reason));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [revision, refreshKey]);

  const items = useMemo(
    () => (catalog ? (runRoot ? resultsForRun(catalog.items, catalog.base_dir, runRoot) : catalog.items) : []),
    [catalog, runRoot],
  );
  const tree = useMemo(
    () =>
      resultTree(
        items.filter((item) => item.label.toLowerCase().includes(query.trim().toLowerCase())),
        catalog?.directory_modified_at ?? {},
      ),
    [items, query, catalog],
  );
  const open = useCallback(
    async (item: ResultItem) => {
      if (!viewer || viewer.isBusy()) return;
      setActive(item.rel_path);
      setLoadError("");
      try {
        const loaded =
          item.kind === "splat"
            ? await viewer.loadSplatsFile({ splatsUrl: versioned(item.splats, item)!, label: item.label })
            : await viewer.loadScene({
                pointsUrl: versioned(item.points, item)!,
                imagesUrl: versioned(item.images, item),
                camerasUrl: versioned(item.cameras, item),
                scenePath: item.rel_path,
                splatsUrl: versioned(item.splats, item),
              });
        if (loaded === false) setLoadError(`Could not open ${item.label}. Select it again to retry.`);
      } catch (reason) {
        setLoadError(String(reason));
      }
    },
    [viewer],
  );

  useEffect(() => {
    if (autoOpened.current || !viewer || busy || !catalog || search.get("view") !== "results") return;
    const firstScene = (nodes: Map<string, ResultNode>): ResultItem | undefined => {
      for (const node of sortedNodes(nodes)) {
        if (node.item?.kind !== "splat" && node.item) return node.item;
        const child = firstScene(node.children);
        if (child) return child;
      }
    };
    const item = runRoot
      ? preferredRunResult(items)
      : (firstScene(resultTree(items, catalog.directory_modified_at)) ?? items[0]);
    if (item) {
      autoOpened.current = true;
      void open(item);
    }
  }, [viewer, busy, catalog, search, runRoot, items, open]);

  const renderNodes = (nodes: Map<string, ResultNode>): React.ReactNode => (
    <ul>
      {sortedNodes(nodes).map((node) => {
        const item = node.item;
        if (item)
          return (
            <li key={node.path}>
              <div className={`item${active === item.rel_path ? " active" : ""}`} data-kind={item.kind}>
                <button
                  type="button"
                  className="result-open"
                  disabled={!viewer || busy}
                  title={item.rel_path}
                  onClick={() => {
                    autoOpened.current = true;
                    void open(item);
                  }}
                >
                  <span>{node.name}</span>
                  <small className="result-counts">
                    {item.kind === "splat"
                      ? item.splat_count == null
                        ? "Gaussian splats"
                        : `${item.splat_count.toLocaleString()} splats`
                      : `${(item.point_count ?? 0).toLocaleString()} points · ${(item.camera_count ?? 0).toLocaleString()} cameras`}
                  </small>
                  <ModifiedTime seconds={node.modified} />
                </button>
                {item.can_splatify && (
                  <button type="button" className="text-button result-splatify" onClick={() => onSplatify(item)}>
                    Splatify
                  </button>
                )}
                {item.splat_rel_path && (
                  <div className="result-splat-download">
                    <span className="splat-format-label">.PLY</span>
                    <a href={`/api/splats/export?path=${encodeURIComponent(item.splat_rel_path)}&format=ply`} download>
                      Download
                    </a>
                  </div>
                )}
              </div>
              {node.children.size > 0 && renderNodes(node.children)}
            </li>
          );
        const closed = !query.trim() && collapsed.has(node.path);
        return (
          <li key={node.path} className={`directory-branch${closed ? " is-collapsed" : ""}`}>
            <button
              className="directory-node"
              type="button"
              aria-expanded={!closed}
              onClick={() =>
                setCollapsed((current) => {
                  const next = new Set(current);
                  if (next.has(node.path)) next.delete(node.path);
                  else next.add(node.path);
                  return next;
                })
              }
            >
              <span className="directory-chevron" aria-hidden="true" />
              <span className="directory-label">
                <span>{node.name}</span>
                <ModifiedTime seconds={node.modified} />
              </span>
            </button>
            {renderNodes(node.children)}
          </li>
        );
      })}
    </ul>
  );

  return (
    <>
      <button
        type="button"
        className="text-button"
        disabled={loading}
        onClick={() => setRevision((current) => current + 1)}
      >
        <RefreshCw size={12} /> {loading ? "Refreshing results…" : "Refresh results"}
      </button>
      <div id="info" className={`info-summary${error ? " info-error" : ""}`} role="status">
        <div className="info-pill" data-role="count">
          {error ? "!" : catalog ? items.length.toLocaleString() : "…"}
        </div>
        <div className="info-details">
          <div className="info-title" data-role="title">
            {error ||
              (loading
                ? "Loading reconstructions…"
                : items.length
                  ? `${items.length.toLocaleString()} items ready to explore`
                  : "No results found")}
          </div>
          <div className="info-path" data-role="path">
            {runRoot || catalog?.base_dir || "Fetching data from server…"}
          </div>
        </div>
      </div>
      {loadError && (
        <p className="job-error" role="alert">
          {loadError}
        </p>
      )}
      <input
        type="text"
        id="filter"
        placeholder="Filter scenes…"
        aria-label="Filter scenes"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
      />
      <div id="sceneList" aria-busy={loading}>
        {tree.size ? renderNodes(tree) : !loading && !error && <p>No items found.</p>}
      </div>
    </>
  );
}
