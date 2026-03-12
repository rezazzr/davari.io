import type { Metadata } from "next";
import Link from "next/link";
import { courses } from "@/data/courses";

export const metadata: Metadata = {
  title: "Teaching",
  description: "Courses and teaching materials.",
};

export default function TeachingPage() {
  return (
    <div>
      <h1 className="text-2xl font-bold text-heading">Teaching</h1>

      <div className="mt-8 space-y-4">
        {courses.map((course) => (
          <Link
            key={course.name}
            href={course.link}
            className="block rounded-xl border border-black/5 dark:border-white/5 bg-surface p-6 transition-shadow hover:shadow-md"
          >
            <h3 className="font-semibold text-primary">{course.name}</h3>
            <p className="mt-1 text-sm text-text-muted">
              {course.university} &middot; {course.term}
            </p>
          </Link>
        ))}
      </div>
    </div>
  );
}
