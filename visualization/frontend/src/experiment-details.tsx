import { type ReactNode, useEffect, useState } from "react";
import { Download, RefreshCw, Terminal, Square, Copy } from "lucide-react";
import {
  loadResult,
  resultStageLabel,
  waitForViewerIdle,
  type ResultItem,
  type ResultRepresentation,
} from "./result-view";
import { useViewerApi } from "./viewer-context";

type ExperimentJob = { id: string; name: string; status: string; error?: string; created_at?: string; history_note?: string | null; remote?: {cancel_status?: string} };
type Details = {
  job: ExperimentJob;
  provenance: { settings: Record<string, unknown>; source: {commit?: string; dirty?: boolean; fingerprint?: string} | null; dataset: {manifest_sha256?: string; files: unknown[]} | null; note?: string };
  configuration: unknown;
  metrics: Record<string, Record<string, number>>;
  results: ResultItem[];
  preferred_result: ResultItem | null;
};
const number = (value: number) => Number.isInteger(value) ? value.toLocaleString() : value.toLocaleString(undefined, {maximumFractionDigits: 4});
async function readDetails(id: string, signal?: AbortSignal): Promise<Details> {
  const response = await fetch(`/api/experiments/${encodeURIComponent(id)}`, { signal, cache: "no-store" });
  if (!response.ok) throw new Error(`Unable to load experiment (${response.status}).`);
  return response.json();
}
export function ExperimentDetails({job, jobs, onReuse, onCancel, onLogs, onBack, live, children, enabled = true}: {
  children?: ReactNode; enabled?: boolean; job: ExperimentJob | null; jobs: ExperimentJob[]; onReuse: (settings: Record<string, unknown>) => void;
  live?: {message?: string; stage?: string; progress?: number} | null; onBack: () => void; onCancel: (id: string) => Promise<void>; onLogs: () => void;
}) {
  const {viewer} = useViewerApi();
  const jobId = job?.id;
  const [fetched, setFetched] = useState<Details | null>(null);
  const [detailError, setDetailError] = useState("");
  const [resultError, setResultError] = useState("");
  const [comparisonError, setComparisonError] = useState("");
  const [revision, setRevision] = useState(0);
  const [selectedStage, setSelectedStage] = useState("");
  const [representation, setRepresentation] = useState<ResultRepresentation>("scene");
  const [compareId, setCompareId] = useState("");
  const [comparison, setComparison] = useState<Details | null>(null);

  const details = fetched?.job.id === jobId ? fetched : null;
  const result = details?.results.find((item) => item.rel_path === selectedStage) ?? details?.preferred_result;
  const stage = result?.rel_path ?? "";

  useEffect(() => {
    setSelectedStage("");
    setRepresentation("scene");
    setCompareId("");
    setComparison(null);
  }, [jobId]);

  useEffect(() => {
    setDetailError("");
    if (!jobId || !enabled) return;
    const controller = new AbortController();
    void readDetails(jobId, controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setFetched(data);
      })
      .catch((reason) => {
        if (!controller.signal.aborted) setDetailError(String(reason));
      });
    return () => controller.abort();
  }, [jobId, job?.status, revision, enabled]);

  useEffect(() => {
    setComparison(null);
    setComparisonError("");
    if (!compareId || !enabled) return;
    const controller = new AbortController();
    void readDetails(compareId, controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setComparison(data);
      })
      .catch((reason) => {
        if (!controller.signal.aborted) setComparisonError(String(reason));
      });
    return () => controller.abort();
  }, [compareId, enabled]);

  useEffect(() => {
    setResultError("");
    if (!enabled || !viewer || !result) return;
    const controller = new AbortController();
    const { signal } = controller;
    const open = async () => {
      try {
        await waitForViewerIdle(viewer, signal);
        if (signal.aborted) return;
        const mode = result.kind === "splat" || (result.splats && representation === "splat") ? "splat" : "scene";
        const success = await loadResult(viewer, result, mode);
        if (!signal.aborted && !success) {
          setResultError("The reconstruction could not be opened. Retry loading this experiment.");
        }
      } catch (reason) {
        if (!signal.aborted) setResultError(String(reason));
      }
    };
    void open();
    return () => controller.abort();
  }, [viewer, result, representation, enabled, jobId]);
  if (!job) return <>{enabled && <section className="experiment-detail empty-experiment"><h2>Select an experiment</h2><p>Its settings, measured results, and reconstruction will appear here.</p></section>}{children}</>;
  const source = details?.provenance.source;
  const settings = details?.provenance.settings;
  const metricRows = Object.entries(details?.metrics || {}).flatMap(([group, values]) => Object.entries(values).map(([name, value]) => ({group, name, value})));
  const sameDataset = Boolean(details?.provenance.dataset?.manifest_sha256 && details.provenance.dataset.manifest_sha256 === comparison?.provenance.dataset?.manifest_sha256);
  return <>
    {enabled && <section className="experiment-detail" aria-label="Selected experiment">
      <button type="button" className="text-button back-experiments" onClick={onBack}>← All experiments</button>
      <div className="experiment-heading"><div className="experiment-title"><h1>{job.name}</h1><span className={`job-status ${job.status}`}>{job.status[0].toUpperCase() + job.status.slice(1)}</span></div><div className="experiment-actions"><button type="button" className="icon-button" aria-label="Refresh experiment" onClick={() => setRevision((value) => value + 1)}><RefreshCw size={19}/></button><button type="button" className="secondary-action" disabled={!settings} onClick={() => settings && onReuse(settings)}><Copy size={18}/>Use these settings</button></div></div>
      <p className="experiment-context">{String(settings?.config_name || "Loading preset…")} · {settings?.execution_target === "remote" ? settings?.remote_connection === "ssh" ? "SSH machine" : `Modal ${String(settings?.modal_gpu || "cloud")}` : "This computer"} · {source?.commit ? `source ${source.commit.slice(0, 8)}${source.dirty ? " + local changes" : ""}` : "Source not recorded"}</p>
      {["queued", "running"].includes(job.status) && <div className="experiment-progress" role="status"><strong>{job.status === "queued" ? "Waiting for compute" : live?.message || live?.stage?.replaceAll("_", " ") || "Running reconstruction"}</strong>{typeof live?.progress === "number" && <progress max="1" value={live.progress}/>}</div>}
      {job.status === "cancelled" && job.remote?.cancel_status && <p className={job.remote.cancel_status === "failed" ? "job-error" : "field-help"} role="status">{job.remote.cancel_status === "confirmed" ? "Remote cancellation acknowledged." : job.remote.cancel_status === "not_submitted" ? "Stopped before remote submission." : job.remote.cancel_status === "failed" ? "Remote cancellation could not be confirmed. Inspect the provider before starting another run." : "Stop requested. Waiting for remote confirmation."}</p>}
      {job.error && <p className="job-error" role="alert">{job.error}</p>}
      {detailError && <p className="job-error" role="alert">{detailError}</p>}
      {resultError && <p className="job-error" role="alert">{resultError}</p>}
      {!details && !detailError && <p role="status">Loading experiment…</p>}
      {details && <div className="experiment-result-bar">{result ? <><strong>{(result.camera_count || 0).toLocaleString()} cameras · {(result.point_count || result.splat_count || 0).toLocaleString()} {result.kind === "splat" ? "splats" : "points"}</strong><label>Result stage<select aria-label="Result stage" value={stage} onChange={(event) => {setSelectedStage(event.target.value); setRepresentation("scene");}}>{details.results.map((item) => <option key={item.rel_path} value={item.rel_path}>{resultStageLabel(item)}</option>)}</select></label>{result.kind !== "splat" && result.splats && <label>View<select aria-label="Reconstruction view" value={representation} onChange={(event) => setRepresentation(event.target.value === "splat" ? "splat" : "scene")}><option value="scene">Cameras and points</option><option value="splat">Gaussian splats</option></select></label>}</> : <p>{job.status === "completed" ? "No viewable reconstruction was saved. Inspect the logs and measured results below." : ["running", "queued"].includes(job.status) ? "The reconstruction will appear here when it is ready." : "No reconstruction is available for this experiment."}</p>}{["running", "queued"].includes(job.status) && <button type="button" className="secondary-action danger" onClick={() => void onCancel(job.id)}><Square size={16}/>Stop experiment</button>}</div>}
    </section>}
    {children}
    {enabled && details && <section className="experiment-record" aria-label="Experiment evidence">
      <div className="experiment-links"><a className="text-button" href={`/api/experiments/${job.id}/export`} download><Download size={18}/>Experiment record</a>{job.status === "completed" && <a className="text-button" href={`/api/jobs/${job.id}/results`} download><Download size={18}/>Result files</a>}<button type="button" className="text-button" onClick={onLogs}><Terminal size={18}/>Logs</button></div>
      <details className="experiment-evidence"><summary>Metrics, configuration & comparison</summary><div className="evidence-content"><label>Compare with<select aria-label="Compare with" value={compareId} onChange={(event) => setCompareId(event.target.value)}><option value="">Choose an experiment</option>{jobs.filter((other) => other.id !== job.id).map((other) => <option key={other.id} value={other.id}>{other.name} · {other.status}</option>)}</select></label>{comparisonError && <p className="job-error" role="alert">{comparisonError}</p>}{comparison && <p className="comparison-note">{sameDataset ? "Dataset fingerprints match. Review configuration differences before interpreting metric changes." : "Dataset identity differs or was not recorded. These runs are not a matched benchmark."}</p>}
        {metricRows.length ? <table><thead><tr><th>Measured metric</th><th>This experiment</th>{comparison && <th>{comparison.job.name}</th>}</tr></thead><tbody>{metricRows.map(({group, name, value}) => <tr key={`${group}:${name}`}><td>{group.replaceAll("_", " ")} / {name.replaceAll("_", " ")}</td><td>{number(value)}</td>{comparison && <td>{typeof comparison.metrics[group]?.[name] === "number" ? number(comparison.metrics[group][name]) : "Not recorded"}</td>}</tr>)}</tbody></table> : <p>No scalar metrics were saved for this experiment.</p>}
        <details><summary>Resolved pipeline configuration</summary>{details.configuration ? <pre>{JSON.stringify(details.configuration, null, 2)}</pre> : <p>This run did not save a resolved configuration.</p>}</details>
      </div></details>
      <details className="experiment-evidence"><summary>Dataset & source provenance</summary><div className="evidence-content"><p>{details.provenance.note || "SHA-256 fingerprints identify the input files and source tree. Reuse applies settings; a new run uses the currently installed code."}</p>{job.history_note && <p>{job.history_note}</p>}<pre>{JSON.stringify({source: details.provenance.source, dataset: details.provenance.dataset}, null, 2)}</pre></div></details>
    </section>}
  </>;
}
