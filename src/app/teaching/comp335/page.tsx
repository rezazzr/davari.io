import type { Metadata } from "next";
import { FaFilePdf } from "react-icons/fa";
import { slides335 } from "@/data/slides335";

export const metadata: Metadata = {
  title: "COMP-335: Introduction to Theoretical Computer Science",
};

export default function Comp335Page() {
  return (
    <div>
      <h1 className="text-2xl font-bold">
        COMP-335: Introduction to Theoretical Computer Science
      </h1>

      {/* Assignment submission instructions */}
      <section className="mt-8 rounded-xl border border-black/5 dark:border-white/5 bg-surface p-6">
        <h2 className="text-lg font-semibold">
          Assignment Submission Instructions
        </h2>
        <div className="mt-3 text-sm leading-relaxed text-text-muted">
          <p>
            <strong>Theory assignments:</strong> You are expected to submit your
            solutions as PDF. Other formats will not be graded. There will be a
            grade associated with how neat, readable, and organized your
            assignment is, hence I recommend against submitting a handwritten
            solution. I suggest using LaTeX and submitting the PDF. If you are
            not familiar with LaTeX, this is a good opportunity for you to caught
            up with it, since sooner or later you will have to use it. Here is a
            good{" "}
            <a
              href="https://www.overleaf.com/learn/latex/Tutorials"
              className="text-primary underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              tutorial
            </a>{" "}
            for it. The other option would be using Microsoft Word, and
            submitting the PDF.
          </p>
        </div>
      </section>

      {/* Tutorial slides */}
      <section className="mt-8">
        <h2 className="text-lg font-semibold">Tutorial Slides</h2>
        <div className="mt-4 space-y-3">
          {slides335.map((slide) => (
            <a
              key={slide.name}
              href={slide.url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-start gap-3 rounded-lg border border-black/5 dark:border-white/5 bg-surface p-4 transition-shadow hover:shadow-md"
            >
              <FaFilePdf className="mt-0.5 shrink-0 text-danger" size={18} />
              <div>
                <h3 className="font-medium">{slide.name}</h3>
                <p className="text-sm text-text-muted">{slide.descr}</p>
              </div>
            </a>
          ))}
        </div>
      </section>
    </div>
  );
}
