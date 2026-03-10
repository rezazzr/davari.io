import Image from "next/image";

export interface TimelineItem {
  name: string;
  name2?: string;
  link: string;
  link2?: string;
  date: string;
  job?: string;
  descr: string;
  logoFile: string;
  logoFile2?: string;
}

interface TimelineProps {
  title: string;
  items: TimelineItem[];
}

export default function Timeline({ title, items }: TimelineProps) {
  return (
    <section>
      <h2 className="mb-6 text-xl font-bold">{title}</h2>
      <div className="space-y-6">
        {items.map((item, index) => (
          <div
            key={`${item.name}-${item.date}-${index}`}
            className="relative border-l-2 border-primary/30 pl-6"
          >
            <div className="absolute -left-[5px] top-1 h-2 w-2 rounded-full bg-primary" />

            <div className="flex items-center gap-3">
              <div className="flex shrink-0 gap-2">
                <a href={item.link} target="_blank" rel="noopener noreferrer">
                  <span className="inline-block rounded dark:bg-background-light dark:p-1">
                    <Image
                      src={`/assets/img/${item.logoFile}`}
                      alt={item.name}
                      width={36}
                      height={36}
                      className="rounded"
                    />
                  </span>
                </a>
                {item.logoFile2 && item.link2 && (
                  <a
                    href={item.link2}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <span className="inline-block rounded dark:bg-background-light dark:p-1">
                      <Image
                        src={`/assets/img/${item.logoFile2}`}
                        alt={item.name2 || ""}
                        width={36}
                        height={36}
                        className="rounded"
                      />
                    </span>
                  </a>
                )}
              </div>

              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-baseline gap-x-2">
                  <a
                    href={item.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-semibold hover:text-primary"
                  >
                    {item.name}
                  </a>
                  {item.name2 && item.link2 && (
                    <>
                      <span className="text-text-muted">&amp;</span>
                      <a
                        href={item.link2}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="font-semibold hover:text-primary"
                      >
                        {item.name2}
                      </a>
                    </>
                  )}
                </div>
                {item.job && (
                  <p className="text-sm font-medium text-primary">
                    {item.job}
                  </p>
                )}
                <p className="text-xs text-text-muted">{item.date}</p>
              </div>
            </div>

            <div
              className="mt-1 text-sm text-text-muted [&_a]:text-primary [&_a]:underline [&_li]:ml-4 [&_li]:list-disc [&_ul]:mt-1"
              dangerouslySetInnerHTML={{ __html: item.descr }}
            />
          </div>
        ))}
      </div>
    </section>
  );
}
