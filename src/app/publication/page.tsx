import type { Metadata } from "next";
import { publications } from "@/data/publications";
import PublicationSearch from "@/components/PublicationSearch";

export const metadata: Metadata = {
  title: "Publication",
  description: "Reza's publications.",
};

function extractYear(publication: (typeof publications)[0]): number {
  const yearMatch = publication.cite.match(/year\s*=\s*\{?(\d{4})\}?/);
  return yearMatch ? parseInt(yearMatch[1], 10) : 0;
}

export default function PublicationPage() {
  const publicationsByYear = publications.reduce<Record<number, typeof publications>>(
    (acc, pub) => {
      const year = extractYear(pub);
      (acc[year] ??= []).push(pub);
      return acc;
    },
    {}
  );

  const sortedYears = Object.keys(publicationsByYear)
    .map(Number)
    .sort((a, b) => b - a);

  return (
    <div>
      <h1 className="text-2xl font-bold text-heading">Refereed Conference Proceedings</h1>
      <p className="mt-2 text-sm text-text-muted">
        Publications by year.{" "}
        <span className="text-heading">*</span> indicates that authors
        contributed equally.
      </p>

      <PublicationSearch
        publications={publications}
        publicationsByYear={publicationsByYear}
        sortedYears={sortedYears}
      />
    </div>
  );
}
