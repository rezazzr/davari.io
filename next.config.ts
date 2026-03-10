import type { NextConfig } from "next";

/**
 * Next.js configuration file.
 *
 * Key settings:
 * - output: 'export' — generates a fully static site (HTML/CSS/JS files)
 *   that can be hosted on GitHub Pages without a Node.js server.
 *   This is the key setting that makes GitHub Pages deployment possible.
 *
 * - images.unoptimized: true — required for static export because Next.js's
 *   built-in image optimization needs a server. We use plain <img> tags instead.
 *
 * When you're ready to go full-stack (e.g., deploy to Vercel), remove
 * 'output: export' to unlock server-side rendering and API routes.
 */
const nextConfig: NextConfig = {
  output: "export",
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
