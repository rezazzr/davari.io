/**
 * NavList — navigation link list.
 *
 * KEY CONCEPT: next/link
 * Unlike a regular <a> tag that causes a full page reload,
 * Next.js <Link> performs client-side navigation — it only fetches
 * the new page's content and swaps it in, keeping the layout mounted.
 * This makes navigation feel instant.
 */

"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { navItems } from "@/data/nav";

export default function NavList() {
  // usePathname() returns the current URL path (e.g., "/publication").
  // We use it to highlight the active nav item.
  const pathname = usePathname();

  return (
    <nav className="flex flex-col gap-1">
      {navItems.map((item) => {
        const isActive =
          pathname === item.path ||
          (item.path !== "/" && pathname.startsWith(item.path));

        return (
          <Link
            key={item.path}
            href={item.path}
            className={`rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
              isActive
                ? "bg-primary/10 text-primary"
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
