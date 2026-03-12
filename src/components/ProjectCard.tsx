import Image from "next/image";
import { FaGithub, FaFileAlt } from "react-icons/fa";
import type { Project } from "@/data/projects";
import TiltCard from "./TiltCard";

interface ProjectCardProps {
  project: Project;
}

export default function ProjectCard({ project }: ProjectCardProps) {
  return (
    <TiltCard className="rounded-xl border border-black/5 dark:border-white/5 bg-surface p-6 shadow-sm transition-shadow hover:shadow-md">
      <div className="flex flex-col gap-6 md:flex-row">
        <div className="shrink-0 md:w-48">
          <Image
            src={`/assets/img/${project.visual}`}
            alt={project.name}
            width={200}
            height={200}
            loading="lazy"
            className={project.round ? "rounded-full" : "rounded-lg"}
            style={{ objectFit: "contain" }}
          />
        </div>

        <div className="flex-1">
          <h3 className="text-lg font-semibold">{project.name}</h3>
          <div
            className="mt-2 text-sm leading-relaxed text-text-muted [&_a]:text-primary [&_a]:underline [&_li]:ml-4 [&_li]:list-decimal [&_ol]:mt-1"
            dangerouslySetInnerHTML={{ __html: project.descr }}
          />

          <div className="mt-4 flex gap-3">
            {project.github && (
              <a
                href={project.github}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 rounded-md bg-black/5 dark:bg-white/5 px-3 py-1.5 text-sm font-medium transition-colors hover:bg-black/10 dark:hover:bg-white/10"
              >
                <FaGithub size={14} />
                GitHub
              </a>
            )}
            {project.fullReport && (
              <a
                href={project.fullReport}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 rounded-md bg-primary/10 px-3 py-1.5 text-sm font-medium text-primary transition-colors hover:bg-primary/20"
              >
                <FaFileAlt size={14} />
                Report
              </a>
            )}
          </div>
        </div>
      </div>
    </TiltCard>
  );
}
