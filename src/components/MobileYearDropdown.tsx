"use client";

import { useState } from "react";

interface MobileYearDropdownProps {
  years: number[];
}

export default function MobileYearDropdown({ years }: MobileYearDropdownProps) {
  const [isOpen, setIsOpen] = useState(false);

  const handleSelect = (year: number) => {
    const element = document.getElementById(`year-${year}`);
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
      setIsOpen(false);
    }
  };

  return (
    <div className="relative lg:hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="px-4 py-2 rounded-full text-sm font-medium transition-all duration-200 border"
        style={{
          borderColor: "#42b983",
          color: "#42b983",
        }}
      >
        Jump to year ▼
      </button>

      {isOpen && (
        <div
          className="absolute top-full mt-2 left-0 bg-surface border rounded-lg shadow-lg z-20"
          style={{ borderColor: "#42b983" }}
        >
          {years.map((year) => (
            <button
              key={year}
              onClick={() => handleSelect(year)}
              className="w-full text-left px-4 py-2 text-sm hover:bg-primary/10 first:rounded-t-lg last:rounded-b-lg transition-colors"
              style={{ color: "#42b983" }}
            >
              {year}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
