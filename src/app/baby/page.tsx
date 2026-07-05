import type { Metadata, Viewport } from "next";
import Image from "next/image";
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
  alternates: { canonical: "/baby" },
  icons: {
    icon: [
      { url: "/assets/img/milk_bottle.svg", type: "image/svg+xml" },
      { url: "/assets/img/milk_bottle_512.png", type: "image/png", sizes: "512x512" },
    ],
    apple: [{ url: "/assets/img/milk_bottle_180.png", sizes: "180x180" }],
    shortcut: ["/assets/img/milk_bottle_512.png"],
  },
  appleWebApp: { title: "Baby Fund" },
  openGraph: {
    type: "website",
    locale: "en_US",
    url: "/baby",
    siteName: siteConfig.title,
    title: "Baby Fund | Reza Davari",
    description:
      "We're expecting! Share the news and, if you'd like, chip in to the baby fund.",
    images: [
      {
        url: "/assets/img/baby/share-card.png",
        width: 1200,
        height: 630,
        type: "image/png",
        alt: "We're expecting! Baby Davari, due the end of August.",
      },
      {
        url: "/assets/img/baby/share-card-square.png",
        width: 1200,
        height: 1200,
        type: "image/png",
        alt: "We're expecting! Baby Davari, due the end of August.",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    creator: `@${siteConfig.social.twitter}`,
    title: "Baby Fund | Reza Davari",
    description:
      "We're expecting! Share the news and, if you'd like, chip in to the baby fund.",
    images: [
      {
        url: "/assets/img/baby/share-card.png",
        alt: "We're expecting! Baby Davari, due the end of August.",
      },
    ],
  },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f3f0fb" },
    { media: "(prefers-color-scheme: dark)", color: "#1e293b" },
  ],
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
    <div className="space-y-12">
      {/* Two independent column stacks: each side flows on its own (so no
          cross-column gap can open at any width), and on mobile the left
          column stacks first, keeping photos before payments. */}
      <div className="flex flex-col gap-10 lg:flex-row lg:items-start lg:gap-12">
        {/* Left column: intro, photos, why a fund */}
        <div className="space-y-10 lg:w-7/12">
          {/* Hero */}
          <RevealOnScroll>
            <div className="rounded-3xl bg-linear-to-br from-primary/10 via-secondary/10 to-transparent p-8 sm:p-10">
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
        </div>

        {/* Right column: how to chip in, other ways to help, gifts */}
        <div className="space-y-10 lg:w-5/12">
          {/* How to send */}
          <section>
            <RevealOnScroll>
              <h2 className="text-xl font-bold text-heading">Ways to chip in</h2>
              <p className="mt-1 text-sm text-text-muted">{paymentIntro}</p>
            </RevealOnScroll>

            {paymentsArePlaceholder && (
              <RevealOnScroll>
                <p className="mt-4 rounded-lg border border-warning/30 bg-warning/10 px-4 py-3 text-sm text-text">
                  {placeholderNotice}
                </p>
              </RevealOnScroll>
            )}

            <div className="mt-4 grid grid-cols-1 gap-4">
              {paymentMethods.map((method, i) => (
                <RevealOnScroll
                  key={method.label}
                  delay={i * REVEAL_ANIMATION_DELAY_INCREMENT_MS}
                >
                  <div className="h-full rounded-xl border border-black/5 dark:border-white/5 bg-surface p-6">
                    <div className="flex items-center gap-3">
                      <Image
                        src={method.flagSrc}
                        alt={method.flagAlt}
                        width={89}
                        height={84}
                        className="h-7 w-auto shrink-0"
                      />
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
        </div>
      </div>

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
