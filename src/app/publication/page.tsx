import type { Metadata } from "next";
import { publications } from "@/data/publications";
import PublicationCard from "@/components/PublicationCard";
import YearNavigation from "@/components/YearNavigation";
import MobileYearDropdown from "@/components/MobileYearDropdown";

export const metadata: Metadata = {
  title: "Publication",
  description: "Reza's publications.",
};

function extractYear(publication: typeof publications[0]): number {
  const yearMatch = publication.cite.match(/year\s*=\s*\{?(\d{4})\}?/);
  return yearMatch ? parseInt(yearMatch[1], 10) : 0;
}

export default function PublicationPage() {
  const pubIndexMap = new Map(publications.map((pub, i) => [pub, i]));

  const publicationsByYear = publications.reduce(
    (acc, pub) => {
      const year = extractYear(pub);
      if (!acc[year]) {
        acc[year] = [];
      }
      acc[year].push(pub);
      return acc;
    },
    {} as Record<number, typeof publications>
  );

  const sortedYears = Object.keys(publicationsByYear)
    .map(Number)
    .sort((a, b) => b - a);

  return (
    <div>
      <h1 className="text-2xl font-bold">
        <span style={{ color: "#F36170" }}>Refereed Conference Proceedings</span>
      </h1>
      <p className="mt-2 text-sm text-text-muted">
        Publications by year.{" "}
        <span style={{ color: "#F36170" }}>*</span> indicates that authors
        contributed equally.
      </p>

      <div className="mt-8 sticky top-0 z-10 lg:hidden bg-background py-4">
        <MobileYearDropdown years={sortedYears} />
      </div>

      <div className="mt-12 flex gap-8">
        <div className="flex-1 min-w-0 space-y-12 scroll-smooth">
          {sortedYears.map((year) => (
            <div key={year} id={`year-${year}`}>
              <div className="mb-8 flex items-center gap-4">
                <h2 className="text-3xl font-bold" style={{ color: "#42b983" }}>
                  {year}
                </h2>
                <div className="flex-1 h-px bg-linear-to-r from-current to-transparent opacity-30" />
              </div>

              <div className="space-y-6 pl-0 lg:pl-8">
                {publicationsByYear[year].map((pub) => (
                  <PublicationCard
                    key={pub.name}
                    publication={pub}
                    priority={(pubIndexMap.get(pub) ?? -1) < 5}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>

        <YearNavigation years={sortedYears} />
      </div>
    </div>
  );
}
