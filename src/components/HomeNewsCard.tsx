import Link from "next/link";
import { FaArrowRight } from "react-icons/fa";
import { babyFund } from "@/data/baby-fund";
import RevealOnScroll from "./RevealOnScroll";

export default function HomeNewsCard() {
  const { homeNews } = babyFund;

  return (
    <RevealOnScroll>
      <Link
        href="/baby-fund"
        className="group block rounded-2xl border border-black/5 bg-linear-to-br from-primary/10 via-secondary/5 to-transparent p-5 transition-all hover:shadow-md dark:border-white/10"
      >
        <div className="flex items-center gap-4">
          <span className="text-3xl" aria-hidden>
            🍼
          </span>
          <div className="min-w-0 flex-1">
            <span className="inline-block rounded-full bg-primary/15 px-2.5 py-0.5 text-xs font-semibold uppercase tracking-wide text-primary">
              {homeNews.badge}
            </span>
            <h2 className="mt-1.5 font-bold text-heading">{homeNews.title}</h2>
            <p className="mt-1 text-sm leading-relaxed text-text-muted">
              {homeNews.blurb}
            </p>
            <span className="mt-2 inline-flex items-center gap-1.5 text-sm font-semibold text-primary">
              {homeNews.ctaLabel}
              <FaArrowRight
                size={12}
                className="transition-transform group-hover:translate-x-0.5"
              />
            </span>
          </div>
        </div>
      </Link>
    </RevealOnScroll>
  );
}
