"use client";

import { useState, useMemo, useEffect, useRef } from "react";
import { FaSearch } from "react-icons/fa";
import type { Publication } from "@/data/publications";
import PublicationCard from "./PublicationCard";
import YearNavigation from "./YearNavigation";
import MobileYearDropdown from "./MobileYearDropdown";

interface PublicationSearchProps {
  publications: Publication[];
  publicationsByYear: Record<number, Publication[]>;
  sortedYears: number[];
}

function useDebouncedValue<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState(value);
  const timerRef = useRef<ReturnType<typeof setTimeout>>(undefined);

  useEffect(() => {
    timerRef.current = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(timerRef.current);
  }, [value, delay]);

  return debounced;
}

export default function PublicationSearch({
  publications,
  publicationsByYear,
  sortedYears,
}: PublicationSearchProps) {
  const [query, setQuery] = useState("");
  const debouncedQuery = useDebouncedValue(query, 250);

  const lowerQuery = debouncedQuery.toLowerCase().trim();
  const isSearching = lowerQuery.length > 0;

  const filteredPubs = useMemo(
    () =>
      isSearching
        ? publications.filter(
            (pub) =>
              pub.name.toLowerCase().includes(lowerQuery) ||
              pub.descr.toLowerCase().includes(lowerQuery) ||
              pub.published.toLowerCase().includes(lowerQuery) ||
              pub.cite.toLowerCase().includes(lowerQuery)
          )
        : [],
    [publications, lowerQuery]
  );

  return (
    <>
      <div className="relative mt-6">
        <FaSearch
          size={14}
          className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted"
        />
        <input
          type="text"
          placeholder="Search publications by title, author, venue..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="w-full rounded-lg border border-black/10 dark:border-white/10 bg-surface py-2.5 pl-9 pr-4 text-sm placeholder:text-text-muted focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
        />
      </div>

      {isSearching ? (
        <div className="mt-8 space-y-6">
          {filteredPubs.length === 0 ? (
            <p className="text-sm text-text-muted">
              No publications found for &ldquo;{query}&rdquo;
            </p>
          ) : (
            <>
              <p className="text-sm text-text-muted">
                {filteredPubs.length} result{filteredPubs.length !== 1 && "s"}
              </p>
              {filteredPubs.map((pub) => (
                <PublicationCard key={pub.name} publication={pub} />
              ))}
            </>
          )}
        </div>
      ) : (
        <>
          <div className="sticky top-0 z-10 mt-8 bg-background py-4 lg:hidden">
            <MobileYearDropdown years={sortedYears} />
          </div>

          <div className="mt-12 flex gap-8">
            <div className="flex-1 min-w-0 space-y-12 scroll-smooth">
              {sortedYears.map((year) => (
                <div key={year} id={`year-${year}`}>
                  <div className="mb-8 flex items-center gap-4">
                    <h2 className="text-3xl font-bold text-primary">{year}</h2>
                    <div className="flex-1 h-px bg-linear-to-r from-current to-transparent opacity-30" />
                  </div>

                  <div className="space-y-6 pl-0 lg:pl-8">
                    {publicationsByYear[year].map((pub, i) => (
                      <PublicationCard
                        key={pub.name}
                        publication={pub}
                        priority={i < 5}
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>

            <YearNavigation years={sortedYears} />
          </div>
        </>
      )}
    </>
  );
}
