"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import Image from "next/image";
import { FaFilePdf, FaTrophy, FaCopy, FaCheck } from "react-icons/fa";
import CollapsibleButton from "./CollapsibleSection";
import TiltCard from "./TiltCard";
import type { Publication } from "@/data/publications";

type OpenSection = "abstract" | "cite" | null;

interface PublicationCardProps {
  publication: Publication;
  priority?: boolean;
}

export default function PublicationCard({ publication, priority = false }: PublicationCardProps) {
  const [openSection, setOpenSection] = useState<OpenSection>(null);
  const [copied, setCopied] = useState(false);
  const copyTimerRef = useRef<ReturnType<typeof setTimeout>>(undefined);

  useEffect(() => {
    return () => clearTimeout(copyTimerRef.current);
  }, []);

  const toggle = (section: OpenSection) =>
    setOpenSection((prev) => (prev === section ? null : section));

  const copyCitation = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(publication.cite);
    } catch {
      const ta = document.createElement("textarea");
      ta.value = publication.cite;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    }
    setCopied(true);
    clearTimeout(copyTimerRef.current);
    copyTimerRef.current = setTimeout(() => setCopied(false), 2000);
  }, [publication.cite]);

  return (
    <TiltCard className="rounded-xl border border-black/5 dark:border-white/5 bg-surface p-6 shadow-sm transition-shadow hover:shadow-md">
      <div className="flex flex-col gap-6 lg:flex-row">
        <div className="shrink-0 lg:w-72">
          <Image
            src={`/assets/img/${publication.visual}`}
            alt={publication.name}
            width={320}
            height={320}
            loading={priority ? "eager" : "lazy"}
            priority={priority}
            className="rounded-lg"
            style={{ maxHeight: "320px", objectFit: "contain" }}
          />
        </div>

        <div className="flex-1">
          <h3 className="text-lg font-semibold">{publication.name}</h3>
          <p
            className="mt-1 text-sm text-text-muted [&_a]:text-primary [&_a]:underline [&_sup]:text-xs"
            dangerouslySetInnerHTML={{ __html: publication.published }}
          />

          {publication.award && (
            <p className="mt-2 inline-flex items-center gap-1 rounded-full bg-warning/10 px-3 py-1 text-sm font-medium text-warning">
              <FaTrophy size={14} />
              {publication.award}
            </p>
          )}

          <div className="mt-3 flex flex-wrap items-start gap-2">
            <CollapsibleButton
              buttonLabel="Abstract"
              variant="info"
              isOpen={openSection === "abstract"}
              onClick={() => toggle("abstract")}
            />

            {publication.paper && (
              <a
                href={publication.paper}
                target="_blank"
                rel="noopener noreferrer"
                className="mt-2 inline-flex items-center gap-1 rounded-md bg-primary/10 px-3 py-1.5 text-sm font-medium text-primary transition-colors hover:bg-primary/20"
              >
                <FaFilePdf size={14} />
                Paper
              </a>
            )}

            <CollapsibleButton
              buttonLabel="Cite"
              variant="warning"
              isOpen={openSection === "cite"}
              onClick={() => toggle("cite")}
            />
          </div>

          {openSection !== null && (
            <div className="mt-3 rounded-lg border border-black/10 dark:border-white/10 bg-black/2 dark:bg-white/5 p-4">
              {openSection === "abstract" ? (
                <p className="text-sm leading-relaxed">{publication.descr}</p>
              ) : (
                <div className="relative">
                  <button
                    onClick={copyCitation}
                    className="absolute right-0 top-0 inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium text-text-muted transition-colors hover:text-text"
                    aria-label="Copy citation"
                  >
                    {copied ? <><FaCheck size={12} className="text-green-500" /> Copied</> : <><FaCopy size={12} /> Copy</>}
                  </button>
                  <pre className="overflow-x-auto whitespace-pre-wrap font-mono text-xs pr-16">
                    {publication.cite}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </TiltCard>
  );
}
