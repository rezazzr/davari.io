import type { Metadata } from "next";
import Script from "next/script";
import { Inter, Roboto_Mono } from "next/font/google";
import "./globals.css";
import Sidebar from "@/components/Sidebar";
import MobileHeader from "@/components/MobileHeader";
import Footer from "@/components/Footer";
import ScrollToTop from "@/components/ScrollToTop";
import PageTransition from "@/components/PageTransition";
import { siteConfig } from "@/data/site-config";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
  preload: true,
});

const robotoMono = Roboto_Mono({
  variable: "--font-roboto-mono",
  subsets: ["latin"],
  display: "swap",
  preload: true,
});

export const metadata: Metadata = {
  title: {
    default: "Reza Davari",
    template: "%s | Reza Davari",
  },
  description: "Reza's notes and projects.",
  keywords: [...siteConfig.keywords],
  icons: {
    icon: "/assets/img/monkey_logo.png",
    apple: "/assets/img/monkey_logo.png",
  },
  metadataBase: new URL(siteConfig.url),
  alternates: {
    canonical: "/",
  },
  openGraph: {
    type: "website",
    locale: "en_US",
    url: siteConfig.url,
    siteName: siteConfig.title,
    title: "Reza Davari",
    description: siteConfig.description,
    images: [{ url: "/assets/img/reza_profile.png", width: 320, height: 320, alt: "Reza Davari" }],
  },
  twitter: {
    card: "summary",
    creator: `@${siteConfig.social.twitter}`,
    title: "Reza Davari",
    description: siteConfig.description,
    images: ["/assets/img/reza_profile.png"],
  },
};

interface RootLayoutProps {
  readonly children: React.ReactNode;
}

export default function RootLayout({ children }: RootLayoutProps) {
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "Person",
    name: "Reza Davari",
    url: siteConfig.url,
    image: `${siteConfig.url}/assets/img/reza_profile.png`,
    jobTitle: "Senior Applied Scientist",
    worksFor: { "@type": "Organization", name: "Microsoft", url: "https://www.microsoft.com" },
    alumniOf: [
      { "@type": "CollegeOrUniversity", name: "Concordia University" },
      { "@type": "ResearchOrganization", name: "Mila - Quebec AI Institute" },
    ],
    sameAs: [
      `https://twitter.com/${siteConfig.social.twitter}`,
      `https://github.com/${siteConfig.social.github}`,
      `https://linkedin.com/in/${siteConfig.social.linkedin}`,
      `https://scholar.google.com/citations?user=${siteConfig.social.googleScholar}`,
    ],
    knowsAbout: ["Machine Learning", "NLP", "Artificial Intelligence", "Continual Learning", "Agentic Systems"],
  };

  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${inter.variable} ${robotoMono.variable} font-sans antialiased`}
      >
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var t=localStorage.getItem("theme");if(t==="dark"||(!t&&matchMedia("(prefers-color-scheme:dark)").matches)){document.documentElement.classList.add("dark")}}catch(e){}})()`,
          }}
        />

        <div className="flex min-h-screen flex-col md:flex-row">
          <MobileHeader />
          <div className="hidden md:block shrink-0 border-r border-black/10 bg-surface dark:border-white/10 md:fixed md:inset-y-0 md:left-0 md:w-64 md:overflow-y-auto">
            <Sidebar />
          </div>

          <div className="flex min-h-screen flex-1 flex-col md:ml-64">
            <main className="flex-1 px-6 py-8 md:px-12 md:py-12">
              <PageTransition>{children}</PageTransition>
            </main>
            <Footer />
          </div>
        </div>

        <ScrollToTop />

        <Script
          src={`https://www.googletagmanager.com/gtag/js?id=${siteConfig.analytics.gaId}`}
          strategy="afterInteractive"
        />
        <Script id="gtag-init" strategy="afterInteractive">
          {`window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments);}gtag('js',new Date());gtag('config','${siteConfig.analytics.gaId}');`}
        </Script>
      </body>
    </html>
  );
}
