"use client";

import { useEffect, useRef } from "react";
import Image from "next/image";

export interface TimelineItem {
  name: string;
  name2?: string;
  link: string;
  link2?: string;
  date: string;
  job?: string;
  descr: string;
  logoFile: string;
  logoFile2?: string;
}

interface TimelineProps {
  title: string;
  items: TimelineItem[];
  animated?: boolean;
}

export default function Timeline({ title, items, animated = false }: TimelineProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!animated) return;
    const container = containerRef.current;
    if (!container) return;

    const entries = container.querySelectorAll<HTMLElement>("[data-timeline-entry]");
    const line = container.querySelector<HTMLElement>("[data-timeline-line]");

    const observer = new IntersectionObserver(
      (observed) => {
        observed.forEach((entry) => {
          if (entry.isIntersecting) {
            (entry.target as HTMLElement).style.opacity = "1";
            (entry.target as HTMLElement).style.transform = "translateX(0)";
          }
        });
      },
      { threshold: 0.15, rootMargin: "0px 0px -50px 0px" }
    );

    entries.forEach((el) => {
      el.style.opacity = "0";
      el.style.transform = "translateX(-12px)";
      el.style.transition = "opacity 0.5s ease-out, transform 0.5s ease-out";
      observer.observe(el);
    });

    const lineObserver = new IntersectionObserver(
      (observed) => {
        observed.forEach((entry) => {
          if (entry.isIntersecting && line) {
            line.style.transform = "scaleY(1)";
          }
        });
      },
      { threshold: 0.05 }
    );

    if (line) {
      line.style.transformOrigin = "top";
      line.style.transform = "scaleY(0)";
      line.style.transition = "transform 1.2s ease-out";
      lineObserver.observe(line);
    }

    return () => {
      observer.disconnect();
      lineObserver.disconnect();
    };
  }, [animated]);

  return (
    <section ref={containerRef}>
      <h2 className="mb-6 text-xl font-bold">{title}</h2>
      <div className="relative space-y-6">
        {animated && (
          <div
            data-timeline-line
            className="absolute left-0 top-0 bottom-0 w-0.5 bg-primary/30"
          />
        )}

        {items.map((item, index) => (
          <div
            key={`${item.name}-${item.date}-${index}`}
            data-timeline-entry
            className={animated ? "relative pl-6" : "relative border-l-2 border-primary/30 pl-6"}
          >
            <div
              className={`absolute top-1 h-2 w-2 rounded-full bg-primary ${
                animated ? "-left-[3px] ring-2 ring-background" : "-left-[5px]"
              }`}
            />

            <div className="flex items-center gap-3">
              <div className="flex shrink-0 gap-2">
                <a href={item.link} target="_blank" rel="noopener noreferrer">
                  <span className="inline-block rounded dark:bg-background-light dark:p-1">
                    <Image
                      src={`/assets/img/${item.logoFile}`}
                      alt={item.name}
                      width={36}
                      height={36}
                      className="rounded"
                    />
                  </span>
                </a>
                {item.logoFile2 && item.link2 && (
                  <a href={item.link2} target="_blank" rel="noopener noreferrer">
                    <span className="inline-block rounded dark:bg-background-light dark:p-1">
                      <Image
                        src={`/assets/img/${item.logoFile2}`}
                        alt={item.name2 || ""}
                        width={36}
                        height={36}
                        className="rounded"
                      />
                    </span>
                  </a>
                )}
              </div>

              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-baseline gap-x-2">
                  <a
                    href={item.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-semibold hover:text-primary"
                  >
                    {item.name}
                  </a>
                  {item.name2 && item.link2 && (
                    <>
                      <span className="text-text-muted">&amp;</span>
                      <a
                        href={item.link2}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="font-semibold hover:text-primary"
                      >
                        {item.name2}
                      </a>
                    </>
                  )}
                </div>
                {item.job && (
                  <p className="text-sm font-medium text-primary">{item.job}</p>
                )}
                <p className="text-xs text-text-muted">{item.date}</p>
              </div>
            </div>

            <div
              className="mt-1 text-sm text-text-muted [&_a]:text-primary [&_a]:underline [&_li]:ml-4 [&_li]:list-disc [&_ul]:mt-1"
              dangerouslySetInnerHTML={{ __html: item.descr }}
            />
          </div>
        ))}
      </div>
    </section>
  );
}
