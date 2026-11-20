import type { ViewerApi } from "./viewer-context";

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

export type ResultRepresentation = "scene" | "splat";

const STAGE_LABELS: Record<string, string> = {
  merged_retriangulated: "Final reconstruction",
  merged: "Merged reconstruction",
  ba_output: "Bundle adjustment",
  vggt_pre_ba: "Before optimization",
  vggt: "VGGT reconstruction",
};

export function resultStageLabel(item: ResultItem, fallback = item.label): string {
  return STAGE_LABELS[item.rel_path.split("/").at(-1) ?? ""] ?? fallback;
}

function versioned(url: string | null | undefined, item: ResultItem): string | null {
  const revision = item.revision ?? item.modified_at;
  if (!url || revision == null) return url ?? null;
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}v=${encodeURIComponent(revision)}`;
}

export async function loadResult(
  viewer: ViewerApi,
  item: ResultItem,
  representation: ResultRepresentation = item.kind,
): Promise<boolean> {
  const splatsUrl = versioned(item.splats, item);
  if (representation === "splat") {
    return !!splatsUrl && (await viewer.loadSplatsFile({ splatsUrl, label: item.label })) !== false;
  }
  const pointsUrl = versioned(item.points, item);
  return !!pointsUrl && (await viewer.loadScene({
    pointsUrl,
    imagesUrl: versioned(item.images, item),
    camerasUrl: versioned(item.cameras, item),
    scenePath: item.rel_path,
    splatsUrl,
  })) !== false;
}

export async function waitForViewerIdle(viewer: ViewerApi, signal: AbortSignal): Promise<void> {
  if (signal.aborted || !viewer.isBusy()) return;
  await new Promise<void>((resolve) => {
    const finish = () => {
      unsubscribe();
      signal.removeEventListener("abort", finish);
      resolve();
    };
    const unsubscribe = viewer.subscribe(() => {
      if (!viewer.isBusy()) finish();
    });
    signal.addEventListener("abort", finish, { once: true });
    if (signal.aborted || !viewer.isBusy()) finish();
  });
}
