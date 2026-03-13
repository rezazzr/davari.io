"use client";

import { useState, useRef, useEffect } from "react";
import { FaBars } from "react-icons/fa";
import NavList from "./NavList";

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
    <div ref={menuRef} className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Toggle menu"
        aria-expanded={isOpen}
        className="rounded-lg p-2 text-text-muted transition-colors hover:bg-black/5 dark:hover:bg-white/5"
      >
        <FaBars size={20} />
      </button>

      {isOpen && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 top-16 z-40 bg-black/20 dark:bg-black/40"
            onClick={() => setIsOpen(false)}
          />

          {/* Dropdown Menu */}
          <div className="absolute left-0 right-0 top-full z-50 mt-2 rounded-lg border border-black/10 bg-surface dark:border-white/10 dark:bg-surface shadow-lg">
            <div className="px-2 py-2" onClick={handleNavClick}>
              <NavList />
            </div>
          </div>
        </>
      )}
    </div>
  );
}
