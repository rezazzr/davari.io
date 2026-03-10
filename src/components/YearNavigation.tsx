"use client";

import { useState, useEffect } from "react";

interface YearNavigationProps {
  years: number[];
}

export default function YearNavigation({ years }: YearNavigationProps) {
  const [activeYear, setActiveYear] = useState<number | null>(null);

  useEffect(() => {
    // Create Intersection Observer to track which year section is in view
    const observers = years.map((year) => {
      const element = document.getElementById(`year-${year}`);
      if (!element) return null;

      const observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              setActiveYear(year);
            }
          });
        },
        {
          rootMargin: "-50px 0px -66% 0px",
        }
      );

      observer.observe(element);
      return observer;
    });

    return () => {
      observers.forEach((observer) => {
        if (observer) observer.disconnect();
      });
    };
  }, [years]);

  return (
    <aside className="hidden lg:flex flex-col gap-4 sticky top-8 h-fit">
      <div className="text-xs font-semibold text-text-muted uppercase tracking-wider">
        Years
      </div>
      <nav className="flex flex-col gap-1">
        {years.map((year) => (
          <a
            key={year}
            href={`#year-${year}`}
            className={`py-1.5 px-3 rounded-md text-sm transition-colors ${
              activeYear === year
                ? "bg-primary/10 text-primary font-semibold"
                : "text-text-muted hover:text-text hover:bg-black/5 dark:hover:bg-white/5"
            }`}
          >
            {year}
          </a>
        ))}
      </nav>
    </aside>
  );
}
