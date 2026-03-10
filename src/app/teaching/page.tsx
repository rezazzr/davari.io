/**
 * Teaching page — lists courses with links to detail pages.
 *
 * KEY CONCEPT: Nested routes.
 * This file is at src/app/teaching/page.tsx → URL: /teaching
 * The course detail pages are at:
 *   src/app/teaching/comp335/page.tsx → URL: /teaching/comp335
 *   src/app/teaching/comp5361/page.tsx → URL: /teaching/comp5361
 *
 * The folder structure mirrors the URL structure.
 */

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
      <h1 className="text-2xl font-bold">
        <span style={{ color: "#F36170" }}>Teaching</span>
      </h1>

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
