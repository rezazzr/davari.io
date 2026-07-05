"use client";

import { useEffect } from "react";

interface PrefetchImagesProps {
  /** absolute /public paths to warm in the browser cache */
  srcs: string[];
}

/**
 * Warms the browser cache with the given images during idle time so they
 * appear instantly when the user later navigates to the page that shows them.
 * Renders nothing. No-op when `srcs` is empty (e.g. before real photos exist).
 */
export default function PrefetchImages({ srcs }: PrefetchImagesProps) {
  useEffect(() => {
    if (!srcs.length) return;

    const run = () => {
      for (const src of srcs) {
        const img = new window.Image();
        img.src = src;
      }
    };

    const w = window as Window & {
      requestIdleCallback?: (cb: () => void) => number;
      cancelIdleCallback?: (id: number) => void;
    };

    if (typeof w.requestIdleCallback === "function") {
      const id = w.requestIdleCallback(run);
      return () => w.cancelIdleCallback?.(id);
    }

    const t = setTimeout(run, 1200);
    return () => clearTimeout(t);
  }, [srcs]);

  return null;
}
