/**
 * PublicationCard — displays a single publication with collapsible sections.
 *
 * KEY CONCEPT: Lifting state up
 * The open/close state lives here (not in CollapsibleButton) so we can
 * render the buttons in a fixed row and the expanded content below,
 * preventing buttons from shifting when content opens.
 */

"use client";

import { useState } from "react";
import Image from "next/image";
import { FaFilePdf, FaTrophy } from "react-icons/fa";
import CollapsibleButton from "./CollapsibleSection";
import type { Publication } from "@/data/publications";

type OpenSection = "abstract" | "cite" | null;

interface PublicationCardProps {
  publication: Publication;
  priority?: boolean;
}

export default function PublicationCard({ publication, priority = false }: PublicationCardProps) {
  const [openSection, setOpenSection] = useState<OpenSection>(null);

  const toggle = (section: OpenSection) =>
    setOpenSection((prev) => (prev === section ? null : section));

  return (
    <div className="flex flex-col gap-6 rounded-xl border border-black/5 dark:border-white/5 bg-surface p-6 shadow-sm transition-shadow hover:shadow-md lg:flex-row">
      {/* Publication visual */}
      <div className="shrink-0 lg:w-72">
        <Image
          src={`/assets/img/${publication.visual}`}
          alt={publication.name}
          width={320}
          height={320}
          priority={priority}
          className="rounded-lg"
          style={{ maxHeight: "320px", objectFit: "contain" }}
        />
      </div>

      {/* Publication details */}
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

        {/* Buttons row — always stays in place */}
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

        {/* Expanded content — always renders below the button row */}
        {openSection === "abstract" && (
          <div className="mt-3 rounded-lg border border-black/10 dark:border-white/10 bg-black/2 dark:bg-white/5 p-4">
            <p className="text-sm leading-relaxed">{publication.descr}</p>
          </div>
        )}

        {openSection === "cite" && (
          <div className="mt-3 rounded-lg border border-black/10 dark:border-white/10 bg-black/2 dark:bg-white/5 p-4">
            <pre className="overflow-x-auto whitespace-pre-wrap font-mono text-xs">
              {publication.cite}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}
