"use client";

import { useRef, type ReactNode, type MouseEvent } from "react";

const CARD_CLASSES = "rounded-xl border border-black/5 dark:border-white/5 bg-surface shadow-sm transition-shadow hover:shadow-md";

interface TiltCardProps {
  children: ReactNode;
  className?: string;
}

export { CARD_CLASSES };

export default function TiltCard({ children, className }: TiltCardProps) {
  const cardRef = useRef<HTMLDivElement>(null);

  const handleMouseEnter = () => {
    const card = cardRef.current;
    if (card) card.style.willChange = "transform";
  };

  const handleMouseMove = (e: MouseEvent) => {
    const card = cardRef.current;
    if (!card) return;

    const rect = card.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const centerX = rect.width / 2;
    const centerY = rect.height / 2;

    const rotateX = ((y - centerY) / centerY) * -4;
    const rotateY = ((x - centerX) / centerX) * 4;

    card.style.transform = `perspective(800px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.01, 1.01, 1.01)`;
  };

  const handleMouseLeave = () => {
    const card = cardRef.current;
    if (!card) return;
    card.style.transform = "perspective(800px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)";
    card.style.willChange = "auto";
  };

  return (
    <div
      ref={cardRef}
      onMouseEnter={handleMouseEnter}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      className={className ?? CARD_CLASSES}
      style={{ transition: "transform 0.15s ease-out" }}
    >
      {children}
    </div>
  );
}
