"use client";

import { FaFileDownload } from "react-icons/fa";
import { siteConfig } from "@/data/site-config";
import MobileNavMenu from "./MobileNavMenu";
import ThemeToggle from "./ThemeToggle";

export default function MobileHeader() {
  return (
    <header className="sticky top-0 z-50 flex h-16 items-center justify-between border-b border-black/10 bg-surface px-4 dark:border-white/10 dark:bg-surface md:hidden">
      {/* Logo/Name */}
      <div className="flex-1 text-sm font-bold text-heading">
        {siteConfig.owner.name.split(" ")[0]}
      </div>

      {/* Navigation Menu */}
      <div className="flex-1 flex justify-center">
        <MobileNavMenu />
      </div>

      {/* Action Buttons */}
      <div className="flex-1 flex items-center justify-end gap-2">
        <a
          href="/assets/Reza_Davari_CV.pdf"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="Download CV"
          className="rounded-lg p-2 text-text-muted transition-colors hover:bg-black/5 dark:hover:bg-white/5 hover:text-text"
        >
          <FaFileDownload size={16} />
        </a>
        <ThemeToggle />
      </div>
    </header>
  );
}
