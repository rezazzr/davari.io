import type { Metadata } from "next";
import { projects } from "@/data/projects";
import ProjectCard from "@/components/ProjectCard";
import RevealOnScroll from "@/components/RevealOnScroll";
import { REVEAL_ANIMATION_DELAY_INCREMENT_MS } from "@/lib/constants";

export const metadata: Metadata = {
  title: "Projects",
  description: "Selected projects by Reza Davari.",
  alternates: { canonical: "/projects" },
  openGraph: { title: "Projects | Reza Davari", description: "Selected projects by Reza Davari.", url: "/projects" },
};

export default function ProjectsPage() {
  return (
    <div>
      <h1 className="text-2xl font-bold text-heading">Selected Projects</h1>

      <div className="mt-8 space-y-8">
        {projects.map((project, i) => (
          <RevealOnScroll key={project.name} delay={i * REVEAL_ANIMATION_DELAY_INCREMENT_MS}>
            <ProjectCard project={project} />
          </RevealOnScroll>
        ))}
      </div>
    </div>
  );
}
