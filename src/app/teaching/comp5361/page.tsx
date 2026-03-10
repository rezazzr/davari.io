import type { Metadata } from "next";
import { FaFilePdf, FaDownload } from "react-icons/fa";
import { slides5361 } from "@/data/slides5361";
import { assignments } from "@/data/assignments";

export const metadata: Metadata = {
  title: "COMP-5361: Discrete Structures and Formal Languages",
};

export default function Comp5361Page() {
  return (
    <div>
      <h1 className="text-2xl font-bold">
        COMP-5361: Discrete Structures and Formal Languages
      </h1>

      <section className="mt-8 rounded-xl border border-black/5 dark:border-white/5 bg-surface p-6">
        <h2 className="text-lg font-semibold">
          Assignment Submission Instructions
        </h2>
        <div className="mt-3 space-y-3 text-sm leading-relaxed text-text-muted">
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
          <p>
            <strong>Programming assignments:</strong> You are expected to submit
            the following, using{" "}
            <a
              href="https://colab.research.google.com/"
              className="text-primary underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              Google Colab
            </a>
            :
          </p>
          <ol className="ml-6 list-decimal space-y-1">
            <li>
              <strong>An ipython notebook:</strong> This file should contain all
              your code and explanations. Be sure that you are using Python 3 not
              Python 2. Assignments done in Python 2 will not be graded. You can
              find an example of how your ipython notebook should look like{" "}
              <a
                href="https://drive.google.com/file/d/1Y3Jwl-v7_l7Kfpq48KWKBJeKKwJaDV36/view?usp=sharing"
                className="text-primary underline"
                target="_blank"
                rel="noopener noreferrer"
              >
                here (Colab)
              </a>{" "}
              and{" "}
              <a
                href="https://github.com/rezazzr/GVF_SR_reinforcement_learning/blob/master/GVF_SR.ipynb"
                className="text-primary underline"
                target="_blank"
                rel="noopener noreferrer"
              >
                here (Github)
              </a>
              .
            </li>
            <li>
              <strong>A PDF file:</strong> This file is the PDF of the ipython
              notebook. It is very easy to create the PDF version. Search it
              online and if you could not find any solutions, you can ask me
              during the tutorials.
            </li>
          </ol>
        </div>
      </section>

      <section className="mt-8">
        <h2 className="text-lg font-semibold">Tutorial Slides</h2>
        <div className="mt-4 space-y-3">
          {slides5361.map((slide) => (
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

      <section className="mt-8">
        <h2 className="text-lg font-semibold">Programming Assignments</h2>
        <div className="mt-4 space-y-3">
          {assignments.map((assignment) => (
            <a
              key={assignment.name}
              href={assignment.url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-start gap-3 rounded-lg border border-black/5 dark:border-white/5 bg-surface p-4 transition-shadow hover:shadow-md"
            >
              <FaDownload className="mt-0.5 shrink-0 text-primary" size={18} />
              <div>
                <h3 className="font-medium">{assignment.name}</h3>
                <p className="text-sm text-text-muted">
                  Due: {assignment.due}
                </p>
              </div>
            </a>
          ))}
        </div>
      </section>
    </div>
  );
}
