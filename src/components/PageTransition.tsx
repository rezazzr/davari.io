"use client";

import { usePathname } from "next/navigation";
import { useEffect, useRef, type ReactNode } from "react";

export default function PageTransition({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const prevPathname = useRef(pathname);

  useEffect(() => {
    if (pathname === prevPathname.current) return;
    prevPathname.current = pathname;

    if (!document.startViewTransition) return;

    // The transition has already happened via Next.js routing,
    // so we trigger a minimal crossfade on the main content area.
    document.startViewTransition(() => Promise.resolve());
  }, [pathname]);

  return <div style={{ viewTransitionName: "main-content" }}>{children}</div>;
}
