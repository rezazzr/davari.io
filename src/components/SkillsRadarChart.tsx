/**
 * SkillsRadarChart — radar/spider chart of technical skills.
 *
 * KEY CONCEPT: 'use client' is required here because Chart.js
 * needs the browser's <canvas> API to render. Server Components
 * can't use browser APIs — they run at build time on Node.js.
 *
 * KEY CONCEPT: Third-party library integration
 * react-chartjs-2 is a React wrapper around Chart.js. We import
 * and register the specific Chart.js modules we need (tree-shaking).
 */

"use client";

import { useEffect, useState } from "react";
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
} from "chart.js";
import { Radar } from "react-chartjs-2";
import { skills } from "@/data/skills";

// Register the Chart.js modules we need (tree-shaking: only include what we use)
ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, Tooltip);

export default function SkillsRadarChart() {
  // Chart.js uses Canvas (not CSS), so we detect dark mode in JS
  // and use a MutationObserver to re-render when the theme changes.
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    const check = () =>
      setIsDark(document.documentElement.classList.contains("dark"));
    check();
    const observer = new MutationObserver(check);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });
    return () => observer.disconnect();
  }, []);

  const textColor = isDark ? "rgba(255,255,255,0.7)" : "rgba(0,0,0,0.6)";
  const gridColor = isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)";

  const data = {
    labels: [...skills.aspects],
    datasets: [
      {
        label: skills.label,
        data: [...skills.percentages],
        backgroundColor: "rgba(66, 185, 131, 0.15)",
        borderColor: "#42b983",
        pointBackgroundColor: "#42b983",
        pointBorderColor: isDark ? "#1e293b" : "#fff",
        pointHoverBackgroundColor: "#42b983",
        pointHoverBorderColor: "#42b983",
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: true,
    scales: {
      r: {
        min: 0,
        max: 100,
        ticks: { display: false },
        pointLabels: { font: { size: 12 }, color: textColor },
        grid: { color: gridColor },
        angleLines: { color: gridColor },
      },
    },
    plugins: {
      legend: { display: false },
    },
  };

  return (
    <div className="mx-auto max-w-md">
      <h2 className="mb-4 text-xl font-bold">Skills</h2>
      <Radar data={data} options={options} />
    </div>
  );
}
