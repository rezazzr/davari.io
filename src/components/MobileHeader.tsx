"use client";

import { FaFileDownload } from "react-icons/fa";
import { siteConfig } from "@/data/site-config";
import MobileNavMenu from "./MobileNavMenu";
import ThemeToggle from "./ThemeToggle";

export default function MobileHeader() {
  return (
    <header className="sticky top-0 z-50 flex h-16 items-center justify-between gap-4 border-b border-black/10 bg-surface px-3 py-4 dark:border-white/10 dark:bg-surface md:hidden">
      {/* Hamburger Menu - Left */}
      <MobileNavMenu />

      {/* Name/Title - Center */}
      <div className="flex-1 min-w-0">
        <div className="text-xs font-bold text-heading truncate">
          {siteConfig.owner.name}
        </div>
        <div className="text-xs text-text-muted truncate">
          Senior Applied Scientist @ Microsoft
        </div>
      </div>

      {/* Action Buttons - Right */}
      <div className="flex items-center gap-1 shrink-0">
        <a
          href="/assets/Reza_Davari_CV.pdf"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1 rounded-lg bg-primary/10 px-2 py-2 text-xs font-medium text-primary transition-colors hover:bg-primary/20"
        >
          <FaFileDownload size={12} />
          <span>CV</span>
        </a>
        <ThemeToggle />
      </div>
    </header>
  );
}
