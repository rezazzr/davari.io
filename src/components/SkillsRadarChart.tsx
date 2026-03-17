"use client";

import { useEffect, useMemo, useState } from "react";
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
import { useTheme } from "@/hooks/useTheme";
import { useRevealOnScroll } from "@/hooks/useRevealOnScroll";

ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, Tooltip);

export default function SkillsRadarChart() {
  const { isDark } = useTheme();
  const { ref, isVisible } = useRevealOnScroll({ threshold: 0.3 });

  const [labelSize, setLabelSize] = useState(12);

  useEffect(() => {
    const update = () => setLabelSize(window.innerWidth < 400 ? 10 : 12);
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);

  const textColor = isDark ? "rgba(255,255,255,0.7)" : "rgba(0,0,0,0.6)";
  const gridColor = isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)";

  const data = useMemo(
    () => ({
      labels: [...skills.aspects],
      datasets: [
        {
          label: skills.label,
          data: isVisible ? skills.percentages : skills.percentages.map(() => 0),
          backgroundColor: "rgba(66, 185, 131, 0.15)",
          borderColor: "#42b983",
          pointBackgroundColor: "#42b983",
          pointBorderColor: isDark ? "#1e293b" : "#fff",
          pointHoverBackgroundColor: "#42b983",
          pointHoverBorderColor: "#42b983",
        },
      ],
    }),
    [isDark, isVisible]
  );

  const options = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: true,
      animation: {
        duration: isVisible ? 1200 : 0,
        easing: "easeOutQuart" as const,
      },
      scales: {
        r: {
          min: 0,
          max: 100,
          ticks: { display: false },
          pointLabels: { font: { size: labelSize }, color: textColor },
          grid: { color: gridColor },
          angleLines: { color: gridColor },
        },
      },
      plugins: {
        legend: { display: false },
      },
    }),
    [textColor, gridColor, isVisible, labelSize]
  );

  const ariaLabel = useMemo(
    () => `Skills radar chart: ${skills.aspects.map((s, i) => `${s} ${skills.percentages[i]}%`).join(", ")}`,
    []
  );

  return (
    <div ref={ref} className="mx-auto max-w-md">
      <h2 className="mb-4 text-xl font-bold">Skills</h2>
      <div role="img" aria-label={ariaLabel}>
        <Radar data={data} options={options} />
      </div>
    </div>
  );
}
