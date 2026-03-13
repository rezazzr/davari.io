"use client";

import { useState, useEffect } from "react";
import { siteConfig } from "@/data/site-config";

export default function MobileSubtitle() {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    const handleScroll = () => {
      // Hide when scrolled more than 20px
      if (window.scrollY > 20) {
        setIsVisible(false);
      } else {
        setIsVisible(true);
      }
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <div
      className={`sticky top-16 z-40 bg-surface px-4 py-3 border-b border-black/10 dark:border-white/10 dark:bg-surface md:hidden transition-opacity duration-200 ${
        isVisible ? "opacity-100" : "opacity-0 pointer-events-none"
      }`}
    >
      <p className="text-xs text-text-muted leading-relaxed">
        {siteConfig.owner.bio}
      </p>
    </div>
  );
}
