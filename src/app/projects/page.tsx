import type { Metadata } from "next";
import { projects } from "@/data/projects";
import ProjectCard from "@/components/ProjectCard";

export const metadata: Metadata = {
  title: "Projects",
  description: "Selected projects by Reza Davari.",
};

export default function ProjectsPage() {
  return (
    <div>
      <h1 className="text-2xl font-bold">
        <span style={{ color: "#F36170" }}>Selected Projects</span>
      </h1>

      <div className="mt-8 space-y-8">
        {projects.map((project) => (
          <ProjectCard key={project.name} project={project} />
        ))}
      </div>
    </div>
  );
}
