"use client";

import Image from "next/image";
import { FaFileDownload } from "react-icons/fa";
import { siteConfig } from "@/data/site-config";
import MobileNavMenu from "./MobileNavMenu";
import ThemeToggle from "./ThemeToggle";

export default function MobileHeader() {
  return (
    <header className="sticky top-0 z-50 flex h-16 items-center justify-between gap-2 border-b border-black/10 bg-surface px-2 py-2 dark:border-white/10 dark:bg-surface md:hidden">
      {/* Hamburger Menu - Left */}
      <MobileNavMenu />

      {/* Avatar - Center Left */}
      <Image
        src={siteConfig.owner.avatar}
        alt={siteConfig.owner.name}
        width={40}
        height={40}
        className="h-10 w-10 rounded-full shrink-0"
      />

      {/* Name/Title - Center */}
      <div className="flex-1 min-w-0">
        <div className="text-xs font-bold text-heading truncate">
          {siteConfig.owner.name}
        </div>
        <div
          className="text-xs truncate"
          dangerouslySetInnerHTML={{ __html: siteConfig.owner.job }}
        />
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
