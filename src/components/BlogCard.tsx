"use client";

import Link from "next/link";
import { useRevealOnScroll } from "@/hooks/useRevealOnScroll";

interface BlogCardProps {
  slug: string;
  title: string;
  date: string;
  tags: string[];
  excerpt?: string;
  delay?: number;
}

export default function BlogCard({ slug, title, date, tags, excerpt, delay = 0 }: BlogCardProps) {
  const { ref, isVisible } = useRevealOnScroll();

  return (
    <div
      ref={ref}
      style={{
        opacity: isVisible ? 1 : 0,
        transform: isVisible ? "translateY(0)" : "translateY(16px)",
        transition: `opacity 0.5s ease-out ${delay}ms, transform 0.5s ease-out ${delay}ms`,
      }}
    >
      <Link
        href={`/blog/${slug}`}
        className="group block rounded-xl border border-black/5 dark:border-white/5 bg-surface p-6 transition-shadow hover:shadow-md"
      >
        <h2 className="text-lg font-semibold">{title}</h2>
        <p className="mt-1 text-sm text-text-muted">
          {new Date(date).toLocaleDateString("en-US", {
            year: "numeric",
            month: "long",
            day: "numeric",
          })}
        </p>
        {tags.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-2">
            {tags.map((tag) => (
              <span
                key={tag}
                className="rounded-full bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary"
              >
                {tag}
              </span>
            ))}
          </div>
        )}
        {excerpt && (
          <div className="mt-2 grid grid-rows-[0fr] transition-[grid-template-rows] duration-300 ease-out group-hover:grid-rows-[1fr]">
            <p className="overflow-hidden text-sm leading-relaxed text-text-muted">
              {excerpt}
              <span className="ml-1 font-medium text-primary">... Read more</span>
            </p>
          </div>
        )}
      </Link>
    </div>
  );
}
