"use client";

import { useSyncExternalStore, useCallback } from "react";

let listeners: Array<() => void> = [];
let observer: MutationObserver | null = null;

function notifyListeners() {
  listeners.forEach((l) => l());
}

function subscribe(listener: () => void) {
  listeners.push(listener);

  // Create a single shared MutationObserver for all subscribers
  if (listeners.length === 1) {
    observer = new MutationObserver(notifyListeners);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });
  }

  return () => {
    listeners = listeners.filter((l) => l !== listener);
    if (listeners.length === 0 && observer) {
      observer.disconnect();
      observer = null;
    }
  };
}

function getSnapshot(): boolean {
  return document.documentElement.classList.contains("dark");
}

function getServerSnapshot(): boolean {
  return false;
}

export function useTheme() {
  const isDark = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  const toggle = useCallback(() => {
    const next = !document.documentElement.classList.contains("dark");
    document.documentElement.classList.toggle("dark", next);
    localStorage.setItem("theme", next ? "dark" : "light");
    // MutationObserver will fire and notify listeners automatically
  }, []);

  return { isDark, toggle };
}
