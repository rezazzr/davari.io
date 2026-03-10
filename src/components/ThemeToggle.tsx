/**
 * ThemeToggle — dark/light mode toggle button.
 *
 * KEY CONCEPT: Client-side state + localStorage persistence
 * This component needs browser APIs (localStorage, document.classList)
 * so it must be a Client Component. The theme preference persists
 * across page reloads via localStorage.
 *
 * KEY CONCEPT: Hydration mismatch prevention
 * The server doesn't know the user's theme preference, so we render
 * a placeholder until the component mounts on the client. This avoids
 * React's "server/client content mismatch" warning.
 */

"use client";

import { useEffect, useState } from "react";
import { FaSun, FaMoon } from "react-icons/fa";

export default function ThemeToggle() {
  const [isDark, setIsDark] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    setIsDark(document.documentElement.classList.contains("dark"));
  }, []);

  const toggle = () => {
    const next = !isDark;
    setIsDark(next);
    document.documentElement.classList.toggle("dark", next);
    localStorage.setItem("theme", next ? "dark" : "light");
  };

  // Render a fixed-size placeholder until mounted to prevent layout shift
  if (!mounted) return <div className="h-8 w-8" />;

  return (
    <button
      onClick={toggle}
      aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
      className="rounded-lg p-2 text-text-muted transition-colors hover:bg-black/5 hover:text-text dark:hover:bg-white/5"
    >
      {isDark ? <FaSun size={16} /> : <FaMoon size={16} />}
    </button>
  );
}
