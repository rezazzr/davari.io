import type { Metadata } from "next";
import { projects } from "@/data/projects";
import ProjectCard from "@/components/ProjectCard";

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
        {projects.map((project) => (
          <ProjectCard key={project.name} project={project} />
        ))}
      </div>
    </div>
  );
}
