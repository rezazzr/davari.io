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

ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, Tooltip);

export default function SkillsRadarChart() {
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
      <div role="img" aria-label={`Skills radar chart: ${skills.aspects.map((s, i) => `${s} ${skills.percentages[i]}%`).join(", ")}`}>
        <Radar data={data} options={options} />
      </div>
    </div>
  );
}
