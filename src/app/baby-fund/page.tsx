import type { Metadata } from "next";
import { babyFund } from "@/data/baby-fund";
import { siteConfig } from "@/data/site-config";
import RevealOnScroll from "@/components/RevealOnScroll";
import CopyButton from "@/components/CopyButton";
import GalleryImage from "@/components/GalleryImage";
import { REVEAL_ANIMATION_DELAY_INCREMENT_MS } from "@/lib/constants";

export const metadata: Metadata = {
  title: "Baby Fund",
  description:
    "We're expecting! A little corner to share the news and, if you'd like to help, an easy way to send some love our way.",
  alternates: { canonical: "/baby-fund" },
  openGraph: {
    title: "Baby Fund | Reza Davari",
    description:
      "We're expecting! Share the news and, if you'd like, chip in to the baby fund.",
    url: "/baby-fund",
  },
};

export default function BabyFundPage() {
  const {
    hero,
    gallery,
    whyFund,
    paymentIntro,
    paymentMethods,
    paymentsArePlaceholder,
    placeholderNotice,
    waysToHelp,
    giftsNote,
    thankYou,
  } = babyFund;

  return (
    <div className="mx-auto max-w-3xl space-y-14">
      {/* Hero */}
      <RevealOnScroll>
        <p className="text-sm font-semibold uppercase tracking-wide text-primary">
          {hero.eyebrow}
        </p>
        <h1 className="mt-2 text-3xl font-bold text-heading sm:text-4xl">
          {hero.title}
        </h1>
        <div className="mt-4 space-y-3 leading-relaxed text-text-muted">
          {hero.paragraphs.map((p, i) => (
            <p key={i}>{p}</p>
          ))}
        </div>
      </RevealOnScroll>

      {/* Ultrasound gallery */}
      <section>
        <RevealOnScroll>
          <h2 className="text-xl font-bold text-heading">{gallery.title}</h2>
          <p className="mt-1 text-sm text-text-muted">{gallery.intro}</p>
        </RevealOnScroll>
        <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
          {gallery.photos.map((photo, i) => (
            <RevealOnScroll
              key={i}
              delay={i * REVEAL_ANIMATION_DELAY_INCREMENT_MS}
            >
              <figure>
                <div className="relative aspect-4/3 overflow-hidden rounded-xl border border-black/5 dark:border-white/5">
                  {photo.src ? (
                    <GalleryImage
                      src={photo.src}
                      alt={photo.alt}
                      blurDataURL={photo.blurDataURL}
                    />
                  ) : (
                    <div className="flex h-full w-full flex-col items-center justify-center gap-2 bg-linear-to-br from-primary/10 via-secondary/10 to-transparent text-center">
                      <span className="text-3xl" aria-hidden>
                        🖤
                      </span>
                      <span className="px-3 text-xs text-text-muted">
                        Sonogram coming soon
                      </span>
                    </div>
                  )}
                </div>
                {photo.caption && (
                  <figcaption className="mt-2 text-center text-sm text-text-muted">
                    {photo.caption}
                  </figcaption>
                )}
              </figure>
            </RevealOnScroll>
          ))}
        </div>
      </section>

      {/* Why a fund */}
      <section>
        <RevealOnScroll>
          <div className="rounded-xl border border-black/5 dark:border-white/5 bg-surface p-6">
            <h2 className="text-xl font-bold text-heading">{whyFund.title}</h2>
            <div className="mt-3 space-y-3 text-sm leading-relaxed text-text-muted">
              {whyFund.paragraphs.map((p, i) => (
                <p key={i}>{p}</p>
              ))}
            </div>
          </div>
        </RevealOnScroll>
      </section>

      {/* How to send */}
      <section>
        <RevealOnScroll>
          <h2 className="text-xl font-bold text-heading">
            Ways to chip in
          </h2>
          <p className="mt-1 text-sm text-text-muted">{paymentIntro}</p>
        </RevealOnScroll>

        {paymentsArePlaceholder && (
          <RevealOnScroll>
            <p className="mt-4 rounded-lg border border-warning/30 bg-warning/10 px-4 py-3 text-sm text-text">
              {placeholderNotice}
            </p>
          </RevealOnScroll>
        )}

        <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
          {paymentMethods.map((method, i) => (
            <RevealOnScroll
              key={method.label}
              delay={i * REVEAL_ANIMATION_DELAY_INCREMENT_MS}
            >
              <div className="h-full rounded-xl border border-black/5 dark:border-white/5 bg-surface p-6">
                <div className="flex items-center gap-3">
                  <span className="text-2xl" aria-hidden>
                    {method.flag}
                  </span>
                  <div>
                    <h3 className="font-semibold">{method.label}</h3>
                    <p className="text-xs text-text-muted">{method.region}</p>
                  </div>
                </div>
                <div className="mt-4 flex items-center justify-between gap-3 rounded-lg bg-black/5 px-3 py-2 dark:bg-white/5">
                  <code className="truncate text-sm">{method.email}</code>
                  <CopyButton
                    value={method.email}
                    label={`${method.label} email`}
                  />
                </div>
                {method.note && (
                  <p className="mt-3 text-xs leading-relaxed text-text-muted">
                    {method.note}
                  </p>
                )}
              </div>
            </RevealOnScroll>
          ))}
        </div>
      </section>

      {/* Other ways to help */}
      <section>
        <RevealOnScroll>
          <h2 className="text-xl font-bold text-heading">
            {waysToHelp.title}
          </h2>
          <p className="mt-1 text-sm leading-relaxed text-text-muted">
            {waysToHelp.intro}
          </p>
        </RevealOnScroll>
        <ul className="mt-5 space-y-3">
          {waysToHelp.items.map((item, i) => (
            <RevealOnScroll
              key={i}
              delay={i * REVEAL_ANIMATION_DELAY_INCREMENT_MS}
            >
              <li className="flex gap-3">
                <span className="text-xl" aria-hidden>
                  {item.emoji}
                </span>
                <span className="text-sm leading-relaxed text-text-muted">
                  {item.text}
                </span>
              </li>
            </RevealOnScroll>
          ))}
        </ul>
      </section>

      {/* Gifts note */}
      <RevealOnScroll>
        <div className="rounded-xl border border-black/5 bg-secondary/5 p-6 dark:border-white/5">
          <h2 className="text-lg font-bold text-heading">{giftsNote.title}</h2>
          <p className="mt-2 text-sm leading-relaxed text-text-muted">
            {giftsNote.body}
          </p>
        </div>
      </RevealOnScroll>

      {/* Thank you + say hi */}
      <RevealOnScroll>
        <div className="rounded-2xl border border-black/5 bg-linear-to-br from-primary/10 to-secondary/10 p-8 text-center dark:border-white/5">
          <h2 className="text-xl font-bold text-heading">{thankYou.title}</h2>
          <p className="mx-auto mt-3 max-w-2xl leading-relaxed text-text-muted">
            {thankYou.body}
          </p>
          <a
            href={`mailto:${siteConfig.email}`}
            className="mt-5 inline-flex items-center gap-2 rounded-lg bg-primary/10 px-5 py-2.5 text-sm font-semibold text-primary transition-all hover:bg-primary/20 active:scale-95"
          >
            {thankYou.ctaLabel}
          </a>
        </div>
      </RevealOnScroll>
    </div>
  );
}
