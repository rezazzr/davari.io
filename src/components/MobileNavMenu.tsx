"use client";

import { useState, useRef, useEffect } from "react";
import { FaBars } from "react-icons/fa";
import NavList from "./NavList";
import SocialLinks from "./SocialLinks";

export default function MobileNavMenu() {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close menu when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
      return () => document.removeEventListener("mousedown", handleClickOutside);
    }
  }, [isOpen]);

  // Close menu when navigating
  const handleNavClick = () => {
    setIsOpen(false);
  };

  return (
    <div ref={menuRef} className="relative shrink-0">
      <button
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Toggle menu"
        aria-expanded={isOpen}
        className="rounded-lg p-2 text-text-muted transition-colors hover:bg-black/5 dark:hover:bg-white/5"
      >
        <FaBars size={18} />
      </button>

      {isOpen && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 top-16 z-40 bg-black/20 dark:bg-black/40"
            onClick={() => setIsOpen(false)}
          />

          {/* Dropdown Menu */}
          <nav className="absolute left-0 top-full z-50 mt-2 w-36 rounded-lg border border-black/10 bg-surface dark:border-white/10 dark:bg-surface shadow-xl">
            <div className="space-y-1 p-2" onClick={handleNavClick}>
              <NavList />
            </div>
            <div className="border-t border-black/10 px-3 py-3 dark:border-white/10">
              <SocialLinks size={16} />
            </div>
          </nav>
        </>
      )}
    </div>
  );
}
