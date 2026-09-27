"use client";

import { useCallback, useEffect, useRef, useState } from "react";

interface ComparisonSliderProps {
  beforeSrc: string;
  afterSrc: string;
  beforeAlt: string;
  afterAlt: string;
}

/**
 * Interactive before/after comparison slider.
 * - Pointer drag + touch support via pointer events
 * - Keyboard accessible: role="slider", arrow keys / Home / End
 * - Works even when the two images have different intrinsic dimensions,
 *   because both are rendered into the same aspect-ratio box with
 *   object-fit: cover on the container and contain on each layer.
 */
export function ComparisonSlider({
  beforeSrc,
  afterSrc,
  beforeAlt,
  afterAlt,
}: ComparisonSliderProps) {
  const [position, setPosition] = useState(50); // percent
  const containerRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef(false);

  const updateFromClientX = useCallback((clientX: number) => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setPosition(Math.min(100, Math.max(0, pct)));
  }, []);

  const onPointerDown = useCallback(
    (event: React.PointerEvent) => {
      draggingRef.current = true;
      (event.target as HTMLElement).setPointerCapture?.(event.pointerId);
      updateFromClientX(event.clientX);
    },
    [updateFromClientX],
  );

  const onPointerMove = useCallback(
    (event: React.PointerEvent) => {
      if (!draggingRef.current) return;
      updateFromClientX(event.clientX);
    },
    [updateFromClientX],
  );

  const stopDragging = useCallback(() => {
    draggingRef.current = false;
  }, []);

  const onKeyDown = useCallback(
    (event: React.KeyboardEvent) => {
      const step = event.shiftKey ? 10 : 2;
      switch (event.key) {
        case "ArrowLeft":
          event.preventDefault();
          setPosition((p) => Math.max(0, p - step));
          break;
        case "ArrowRight":
          event.preventDefault();
          setPosition((p) => Math.min(100, p + step));
          break;
        case "Home":
          event.preventDefault();
          setPosition(0);
          break;
        case "End":
          event.preventDefault();
          setPosition(100);
          break;
      }
    },
    [],
  );

  // Prevent scroll-jacking on touch while dragging inside the slider.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const prevent = (e: TouchEvent) => {
      if (draggingRef.current) e.preventDefault();
    };
    el.addEventListener("touchmove", prevent, { passive: false });
    return () => el.removeEventListener("touchmove", prevent);
  }, []);

  return (
    <div className="rounded-xl border border-line bg-surface p-3 shadow-card">
      <p className="mb-2 text-sm font-semibold">Before / after comparison</p>
      <div
        ref={containerRef}
        className="relative aspect-[4/3] w-full select-none overflow-hidden rounded-lg bg-canvas sm:aspect-auto sm:h-[460px]"
        style={{ touchAction: "pan-y" }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={stopDragging}
        onPointerLeave={stopDragging}
        onPointerCancel={stopDragging}
      >
        {/* After image forms the base layer */}
        <img
          src={afterSrc}
          alt={afterAlt}
          draggable={false}
          className="absolute inset-0 h-full w-full object-contain"
        />
        {/* Before image clipped to the left of the handle */}
        <div
          className="absolute inset-0"
          style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
        >
          <img
            src={beforeSrc}
            alt={beforeAlt}
            draggable={false}
            className="absolute inset-0 h-full w-full object-contain"
          />
        </div>
        {/* Divider line */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-y-0 w-0.5 bg-white/90 shadow-[0_0_6px_rgba(0,0,0,0.45)]"
          style={{ left: `${position}%`, transform: "translateX(-50%)" }}
        />
        {/* Handle — keyboard accessible slider */}
        <div
          role="slider"
          tabIndex={0}
          aria-label="Comparison position between original and colorized image"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(position)}
          aria-orientation="horizontal"
          onKeyDown={onKeyDown}
          onPointerDown={(e) => e.stopPropagation()}
          className="absolute top-1/2 z-10 flex h-10 w-10 -translate-x-1/2 -translate-y-1/2 cursor-ew-resize items-center justify-center rounded-full border-2 border-white bg-accent text-white shadow-md"
          style={{ left: `${position}%` }}
        >
          <svg aria-hidden width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
            <path d="M8 7l-5 5 5 5" />
            <path d="M16 7l5 5-5 5" />
          </svg>
        </div>
        <span className="pointer-events-none absolute bottom-2 left-2 rounded bg-black/55 px-2 py-0.5 text-[11px] font-medium uppercase tracking-wide text-white">
          Original
        </span>
        <span className="pointer-events-none absolute bottom-2 right-2 rounded bg-black/55 px-2 py-0.5 text-[11px] font-medium uppercase tracking-wide text-white">
          Colorized
        </span>
      </div>
      <p className="mt-2 text-xs text-muted">
        Drag the handle, use touch, or focus it and press the arrow keys to compare.
      </p>
    </div>
  );
}
