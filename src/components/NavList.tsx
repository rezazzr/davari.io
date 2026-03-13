"use client";

import { useRef, useState, useEffect, useCallback } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { navItems } from "@/data/nav";

export default function NavList() {
  const pathname = usePathname();
  const navRef = useRef<HTMLElement>(null);
  const [indicator, setIndicator] = useState<{ top: number; height: number } | null>(null);

  const updateIndicator = useCallback(() => {
    const nav = navRef.current;
    if (!nav) return;

    const activeLink = nav.querySelector<HTMLElement>("[data-active='true']");
    if (activeLink) {
      setIndicator({
        top: activeLink.offsetTop,
        height: activeLink.offsetHeight,
      });
    } else {
      setIndicator(null);
    }
  }, []);

  useEffect(() => {
    updateIndicator();
  }, [pathname, updateIndicator]);

  return (
    <nav ref={navRef} className="relative flex flex-col gap-1">
      {indicator && (
        <div
          className="absolute left-0 w-full rounded-lg bg-primary/10 transition-all duration-300 ease-out"
          style={{
            top: indicator.top,
            height: indicator.height,
          }}
        />
      )}
      {navItems.map((item) => {
        const isActive =
          pathname === item.path ||
          (item.path !== "/" && pathname.startsWith(item.path));

        return (
          <Link
            key={item.path}
            href={item.path}
            data-active={isActive}
            className={`relative z-10 rounded-lg px-3 py-2 text-sm font-medium transition-all active:scale-95 ${
              isActive
                ? "text-primary"
                : "text-text-muted hover:bg-black/5 dark:hover:bg-white/5 hover:text-text"
            }`}
          >
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}
