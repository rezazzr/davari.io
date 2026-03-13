"use client";

import { useEffect, useRef } from "react";
import { useTheme } from "@/hooks/useTheme";

interface RGB {
  r: number;
  g: number;
  b: number;
}

interface Particle {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  color: RGB;
  cellX: number;
  cellY: number;
}

interface ParticleNetworkProps {
  imageSrc?: string;
}

const CONNECTION_DISTANCE = 130;
const PARTICLE_COUNT = 24;

export default function ParticleNetwork({ imageSrc = "/assets/img/reza_profile.png" }: ParticleNetworkProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const { isDark } = useTheme();
  const isDarkRef = useRef(isDark);
  isDarkRef.current = isDark;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let particles: Particle[] = [];
    let imageData: ImageData | null = null;
    let imgW = 0;
    let imgH = 0;

    // Spatial hash grid
    const cellSize = CONNECTION_DISTANCE;
    let gridCols = 0;
    let gridRows = 0;
    const grid: Map<number, Particle[]> = new Map();

    const resize = () => {
      const parent = canvas.parentElement;
      if (!parent) return;
      canvas.width = parent.clientWidth;
      canvas.height = parent.clientHeight;
      gridCols = Math.ceil(canvas.width / cellSize) + 1;
      gridRows = Math.ceil(canvas.height / cellSize) + 1;
    };

    const samplePixel = (imgX: number, imgY: number): RGB => {
      if (!imageData) return { r: 120, g: 120, b: 120 };
      const cx = Math.max(0, Math.min(Math.floor(imgX), imgW - 1));
      const cy = Math.max(0, Math.min(Math.floor(imgY), imgH - 1));
      const i = (cy * imgW + cx) * 4;
      return { r: imageData.data[i], g: imageData.data[i + 1], b: imageData.data[i + 2] };
    };

    // Ring band: particles live in an annular region around the image edge
    const INNER_RATIO = 0.35; // how far inside the image half-size
    const OUTER_RATIO = 0.30; // how far outside the image half-size

    let ringInner = 0;
    let ringOuter = 0;
    let cx = 0;
    let cy = 0;

    const updateRing = () => {
      cx = canvas.width / 2;
      cy = canvas.height / 2;
      const halfSize = Math.min(canvas.width, canvas.height) / 2;
      ringInner = halfSize * (1 - INNER_RATIO);
      ringOuter = halfSize * (1 + OUTER_RATIO);
    };

    const createParticles = () => {
      particles = [];
      updateRing();
      const total = PARTICLE_COUNT;
      const scale = canvas.height / imgH;

      for (let i = 0; i < total; i++) {
        const angle = Math.random() * Math.PI * 2;
        const radius = ringInner + Math.random() * (ringOuter - ringInner);
        const px = cx + Math.cos(angle) * radius;
        const py = cy + Math.sin(angle) * radius;

        // Sample color from image at this position
        const imgOffsetX = (canvas.width - imgW * scale) / 2;
        const srcX = (px - imgOffsetX) / scale;
        const srcY = py / scale;
        const color = samplePixel(srcX, srcY);

        particles.push({
          id: i,
          x: px,
          y: py,
          vx: (Math.random() - 0.5) * 0.4,
          vy: (Math.random() - 0.5) * 0.4,
          color,
          cellX: 0,
          cellY: 0,
        });
      }
    };

    const buildGrid = () => {
      grid.clear();
      for (const p of particles) {
        p.cellX = Math.floor(p.x / cellSize);
        p.cellY = Math.floor(p.y / cellSize);
        const key = p.cellY * gridCols + p.cellX;
        let cell = grid.get(key);
        if (!cell) {
          cell = [];
          grid.set(key, cell);
        }
        cell.push(p);
      }
    };

    const themeAdjust = (c: RGB): RGB => {
      if (isDarkRef.current) return c;
      return { r: 255 - c.r, g: 255 - c.g, b: 255 - c.b };
    };

    const avgColor = (a: RGB, b: RGB): RGB => ({
      r: (a.r + b.r) >> 1,
      g: (a.g + b.g) >> 1,
      b: (a.b + b.b) >> 1,
    });

    // Pre-allocate pair-visited bitfield (reused across frames)
    let visitedPairs: Uint8Array | null = null;

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const baseOpacity = isDarkRef.current ? 1 : 0.7;
      const connDistSq = CONNECTION_DISTANCE * CONNECTION_DISTANCE;
      const n = particles.length;

      // Update positions — constrain to ring band
      for (const p of particles) {
        p.x += p.vx;
        p.y += p.vy;
        const dx = p.x - cx;
        const dy = p.y - cy;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < ringInner || dist > ringOuter) {
          // Reflect velocity radially
          const nx = dx / dist;
          const ny = dy / dist;
          const dot = p.vx * nx + p.vy * ny;
          p.vx -= 2 * dot * nx;
          p.vy -= 2 * dot * ny;
          // Push back inside the ring
          const clampedDist = Math.max(ringInner, Math.min(dist, ringOuter));
          p.x = cx + nx * clampedDist;
          p.y = cy + ny * clampedDist;
        }
      }

      // Build spatial hash
      buildGrid();

      // Reset pair-visited bitfield (allocate/resize lazily)
      const pairCount = (n * (n - 1)) / 2;
      if (!visitedPairs || visitedPairs.length < pairCount) {
        visitedPairs = new Uint8Array(pairCount);
      } else {
        visitedPairs.fill(0);
      }

      // Draw connections using spatial hash (only check neighboring cells)
      for (const p of particles) {
        for (let dx = -1; dx <= 1; dx++) {
          for (let dy = -1; dy <= 1; dy++) {
            const nx = p.cellX + dx;
            const ny = p.cellY + dy;
            if (nx < 0 || ny < 0 || nx >= gridCols || ny >= gridRows) continue;
            const cell = grid.get(ny * gridCols + nx);
            if (!cell) continue;
            for (const q of cell) {
              if (q.id <= p.id) continue;
              // Deduplicate using stable particle ids
              const pairIdx = p.id * n - ((p.id * (p.id + 1)) >> 1) + (q.id - p.id - 1);
              if (visitedPairs[pairIdx]) continue;
              visitedPairs[pairIdx] = 1;

              const ddx = p.x - q.x;
              const ddy = p.y - q.y;
              const distSq = ddx * ddx + ddy * ddy;
              if (distSq < connDistSq) {
                const dist = Math.sqrt(distSq);
                const edgeColor = themeAdjust(avgColor(p.color, q.color));
                const opacity = (1 - dist / CONNECTION_DISTANCE) * 0.2 * baseOpacity;
                ctx.strokeStyle = `rgba(${edgeColor.r},${edgeColor.g},${edgeColor.b},${opacity})`;
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(p.x, p.y);
                ctx.lineTo(q.x, q.y);
                ctx.stroke();
              }
            }
          }
        }
      }

      // Draw particles
      for (const p of particles) {
        const dc = themeAdjust(p.color);
        ctx.fillStyle = `rgba(${dc.r},${dc.g},${dc.b},${0.5 * baseOpacity})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }

      animRef.current = requestAnimationFrame(animate);
    };

    const loadImageAndInit = () => {
      resize();
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        const offscreen = document.createElement("canvas");
        offscreen.width = img.naturalWidth;
        offscreen.height = img.naturalHeight;
        imgW = img.naturalWidth;
        imgH = img.naturalHeight;
        const offCtx = offscreen.getContext("2d");
        if (offCtx) {
          offCtx.drawImage(img, 0, 0);
          imageData = offCtx.getImageData(0, 0, imgW, imgH);
        }
        createParticles();
        animate();
      };
      img.onerror = () => {
        createParticles();
        animate();
      };
      img.src = imageSrc;
    };

    loadImageAndInit();

    const resizeObserver = new ResizeObserver(() => {
      resize();
      createParticles();
    });
    resizeObserver.observe(canvas.parentElement!);

    return () => {
      cancelAnimationFrame(animRef.current);
      resizeObserver.disconnect();
    };
  }, [imageSrc]);

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none absolute inset-0 z-10"
      aria-hidden="true"
    />
  );
}
