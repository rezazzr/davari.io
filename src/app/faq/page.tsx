import type { Metadata } from "next";
import { qas } from "@/data/qas";
import RevealOnScroll from "@/components/RevealOnScroll";
import { REVEAL_ANIMATION_DELAY_INCREMENT_MS } from "@/lib/constants";

export const metadata: Metadata = {
  title: "FAQ",
  description: "Frequently asked questions.",
  alternates: { canonical: "/faq" },
  openGraph: { title: "FAQ | Reza Davari", description: "Frequently asked questions.", url: "/faq" },
};

export default function FaqPage() {
  return (
    <div>
      <h1 className="text-2xl font-bold text-heading">FAQ</h1>

      <div className="mt-8 space-y-8">
        {qas.map((qa, index) => (
          <RevealOnScroll key={index} delay={index * REVEAL_ANIMATION_DELAY_INCREMENT_MS}>
            <div className="rounded-xl border border-black/5 dark:border-white/5 bg-surface p-6">
              <h3 className="font-semibold">{qa.question}</h3>
              <p
                className="mt-2 text-sm leading-relaxed text-text-muted [&_a]:text-primary [&_a]:underline"
                dangerouslySetInnerHTML={{ __html: qa.answer }}
              />
            </div>
          </RevealOnScroll>
        ))}
      </div>
    </div>
  );
}
