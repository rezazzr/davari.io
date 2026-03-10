/**
 * Root Layout — the outermost shell of the entire application.
 *
 * KEY CONCEPT: In Next.js App Router, layout.tsx wraps ALL pages.
 * When you navigate between pages, the layout stays mounted (doesn't re-render),
 * while only the page content inside {children} changes. This is why the sidebar
 * persists across navigation without flickering.
 */

import type { Metadata } from "next";
import { Inter, Roboto_Mono } from "next/font/google";
import "./globals.css";
import Sidebar from "@/components/Sidebar";
import Footer from "@/components/Footer";
import ScrollToTop from "@/components/ScrollToTop";

// next/font automatically optimizes fonts:
// - Downloads at build time (no render-blocking requests to Google)
// - Self-hosts them
// - Applies CSS font-display: swap for fast rendering
const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const robotoMono = Roboto_Mono({
  variable: "--font-roboto-mono",
  subsets: ["latin"],
});

// Metadata export — Next.js uses this to generate <head> tags.
// Each page can also export its own metadata to override these defaults.
export const metadata: Metadata = {
  title: {
    default: "Reza Davari",
    template: "%s | Reza Davari", // e.g. "Publications | Reza Davari"
  },
  description: "Reza's notes and projects.",
  keywords: [
    "Mohammad Reza Davari",
    "Reza Davari",
    "Machine Learning",
    "NLP",
    "AI",
    "Artificial Intelligence",
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${inter.variable} ${robotoMono.variable} font-sans antialiased`}
      >
        {/*
         * Anti-flash script: reads the saved theme from localStorage BEFORE
         * React hydrates, so the page renders in the correct mode immediately.
         * Without this, dark mode users would see a flash of light mode.
         * suppressHydrationWarning on <html> prevents React from warning about
         * the class mismatch between server (no "dark") and client (has "dark").
         */}
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var t=localStorage.getItem("theme");if(t==="dark"||(!t&&matchMedia("(prefers-color-scheme:dark)").matches)){document.documentElement.classList.add("dark")}}catch(e){}})()`,
          }}
        />

        {/* Two-column layout: sidebar (fixed on desktop) + scrollable content */}
        <div className="flex min-h-screen flex-col md:flex-row">
          {/* Sidebar — stacks on top on mobile, fixed left panel on desktop */}
          <div className="w-full shrink-0 border-b border-black/10 bg-surface dark:border-white/10 md:fixed md:inset-y-0 md:left-0 md:w-64 md:overflow-y-auto md:border-b-0 md:border-r">
            <Sidebar />
          </div>

          {/* Main content — offset by sidebar width (md:ml-64) on desktop */}
          <div className="flex min-h-screen flex-1 flex-col md:ml-64">
            <main className="flex-1 px-6 py-8 md:px-12 md:py-12">
              {children}
            </main>
            <Footer />
          </div>
        </div>

        {/* Scroll to top button — mobile only */}
        <ScrollToTop />
      </body>
    </html>
  );
}
