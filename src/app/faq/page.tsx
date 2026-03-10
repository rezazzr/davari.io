import type { Metadata } from "next";
import { qas } from "@/data/qas";

export const metadata: Metadata = {
  title: "FAQ",
  description: "Frequently asked questions.",
};

export default function FaqPage() {
  return (
    <div>
      <h1 className="text-2xl font-bold">
        <span style={{ color: "#F36170" }}>FAQ</span>
      </h1>

      <div className="mt-8 space-y-8">
        {qas.map((qa, index) => (
          <div key={index} className="rounded-xl border border-black/5 dark:border-white/5 bg-surface p-6">
            <h3 className="font-semibold">{qa.question}</h3>
            <p
              className="mt-2 text-sm leading-relaxed text-text-muted [&_a]:text-primary [&_a]:underline"
              dangerouslySetInnerHTML={{ __html: qa.answer }}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
