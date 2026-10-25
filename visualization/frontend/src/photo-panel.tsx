import React, { useEffect, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, Image as ImageIcon, Maximize2, X } from "lucide-react";

export interface ScenePhoto {
  name: string;
  thumbnail: string | null;
  preview: string | null;
}

export interface PhotoState {
  items: ScenePhoto[];
  selected: number;
  loading: boolean;
  error?: string;
}

const ROW_HEIGHT = 164;
const EMPTY: PhotoState = { items: [], selected: 0, loading: false };

export function PhotoPanel() {
  const [state, setState] = useState<PhotoState>(EMPTY);
  const [collapsed, setCollapsed] = useState(false);
  const [scrollTop, setScrollTop] = useState(0);
  const [height, setHeight] = useState(600);
  const [preview, setPreview] = useState<ScenePhoto | null>(null);
  const list = useRef<HTMLDivElement>(null);
  const dialog = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const update = (event: Event) => setState((event as CustomEvent<PhotoState>).detail);
    window.addEventListener("gtsfm-photo-state", update);
    const initial = window.gtsfmViewer?.getPhotoState();
    if (initial) setState(initial);
    return () => window.removeEventListener("gtsfm-photo-state", update);
  }, []);

  useEffect(() => {
    const element = list.current;
    if (!element) return;
    const observer = new ResizeObserver(() => setHeight(element.clientHeight));
    observer.observe(element);
    return () => observer.disconnect();
  }, [collapsed, state.items.length]);

  useEffect(() => {
    const element = list.current;
    if (!element) return;
    const top = state.selected * ROW_HEIGHT;
    if (top < element.scrollTop || top + ROW_HEIGHT > element.scrollTop + element.clientHeight) {
      element.scrollTop = top;
    }
    setScrollTop(element.scrollTop);
  }, [state.selected, state.items, collapsed]);

  useEffect(() => { setPreview(null); }, [state.items]);
  useEffect(() => {
    const element = dialog.current;
    if (!preview || !element) return;
    element.showModal();
    return () => element.close();
  }, [preview]);

  if (!state.items.length) return null;
  // Only the visible rows and two neighbors exist in the DOM, even for large runs.
  const start = Math.max(0, Math.min(state.items.length - 1, Math.floor(scrollTop / ROW_HEIGHT) - 2));
  const end = Math.min(state.items.length, start + Math.ceil(height / ROW_HEIGHT) + 4);
  const select = (index: number) => window.gtsfmViewer?.selectCamera(index);
  return <>
    <aside className={`photo-panel${collapsed ? " photo-panel-collapsed" : ""}`} aria-label="Reconstruction photos">
      <header className="photo-panel-header">
        <button type="button" aria-label={collapsed ? "Expand photos" : "Collapse photos"} aria-expanded={!collapsed} onClick={() => setCollapsed(!collapsed)}>{collapsed ? <ChevronLeft size={16}/> : <ChevronRight size={16}/>}</button>
        {!collapsed && <><strong>Photos</strong><span aria-live="polite">{state.selected + 1} / {state.items.length}</span></>}
      </header>
      {collapsed ? <ImageIcon className="photo-panel-symbol" size={18}/> : <>
        <p className="photo-panel-hint">{state.loading ? "Finding source images…" : state.error || "Select a photo to view from its camera."}</p>
        <div ref={list} className="photo-list" onScroll={(event) => setScrollTop(event.currentTarget.scrollTop)} aria-label="Camera thumbnails" onKeyDown={(event) => {
          if (event.key === "ArrowDown" || event.key === "ArrowUp") {
            event.preventDefault();
            select((state.selected + (event.key === "ArrowDown" ? 1 : -1) + state.items.length) % state.items.length);
          }
        }}>
          <div style={{ height: state.items.length * ROW_HEIGHT, position: "relative" }}>
            {state.items.slice(start, end).map((photo, offset) => {
              const index = start + offset;
              return <div className="photo-row" key={`${index}:${photo.name}`} style={{ top: index * ROW_HEIGHT, height: ROW_HEIGHT }}>
                <button type="button" className={`photo-thumbnail${index === state.selected ? " selected" : ""}`} aria-label={`View camera ${index + 1}: ${photo.name}`} aria-current={index === state.selected ? "true" : undefined} onClick={() => select(index)}>
                  <span className="photo-image">{photo.thumbnail ? <img src={photo.thumbnail} alt={photo.name} decoding="async" loading="lazy" onError={(event) => { event.currentTarget.style.display = "none"; event.currentTarget.parentElement?.classList.add("photo-unavailable"); }}/> : <span className="photo-missing"><ImageIcon size={22}/>{state.loading ? "Loading…" : "Image unavailable"}</span>}</span>
                  <span className="photo-caption"><span>{index + 1}</span><span title={photo.name}>{photo.name}</span></span>
                </button>
                {photo.preview && <button type="button" className="photo-enlarge" aria-label={`Enlarge ${photo.name}`} onClick={() => setPreview(photo)}><Maximize2 size={13}/></button>}
              </div>;
            })}
          </div>
        </div>
      </>}
    </aside>
    {preview?.preview && <dialog ref={dialog} className="photo-lightbox" aria-label={preview.name} onCancel={() => setPreview(null)} onClick={(event) => { if (event.target === event.currentTarget) setPreview(null); }}>
      <button type="button" autoFocus aria-label="Close photo" onClick={() => setPreview(null)}><X size={20}/></button>
      <img src={preview.preview} alt={preview.name} onClick={(event) => event.stopPropagation()}/><span>{preview.name}</span>
    </dialog>}
  </>;
}
