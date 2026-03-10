/**
 * Individual blog post page — dynamic route.
 *
 * KEY CONCEPT: Dynamic Routes
 * The [slug] folder name means this page handles any URL like /blog/anything.
 * The "slug" value comes from the URL and is passed as params.slug.
 *
 * KEY CONCEPT: generateStaticParams()
 * For static export (output: 'export'), Next.js needs to know ALL possible
 * slugs at build time so it can pre-render each one to an HTML file.
 * generateStaticParams() returns the list of slugs to pre-render.
 *
 * KEY CONCEPT: generateMetadata()
 * This function generates page-specific <head> metadata (title, description)
 * based on the post's frontmatter. It runs at build time.
 */

import type { Metadata } from "next";
import { getAllPosts, getPostBySlug } from "@/lib/posts";
import { extractTableOfContents } from "@/lib/toc";

// Tell Next.js which slugs to pre-render at build time
export function generateStaticParams() {
  const posts = getAllPosts();
  return posts.map((post) => ({ slug: post.slug }));
}

// Generate page-specific metadata
export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata> {
  const { slug } = await params;
  const { meta } = await getPostBySlug(slug);
  return {
    title: meta.title,
    description: meta.excerpt,
  };
}

function TableOfContents({ items }: { items: ReturnType<typeof extractTableOfContents>["toc"] }) {
  if (items.length === 0) return null;

  const renderToc = (items: typeof items, depth = 0) => (
    <ul className={depth === 0 ? "space-y-2" : "space-y-1 ml-4"}>
      {items.map((item) => (
        <li key={item.id}>
          <a
            href={`#${item.id}`}
            className="text-sm text-primary hover:underline"
          >
            {item.title}
          </a>
          {item.children.length > 0 && renderToc(item.children, depth + 1)}
        </li>
      ))}
    </ul>
  );

  return (
    <nav className="mb-8 rounded-lg border border-black/10 dark:border-white/10 bg-black/2 dark:bg-white/5 p-4">
      <h2 className="text-sm font-semibold text-text-muted uppercase tracking-wider mb-3">
        Table of Contents
      </h2>
      {renderToc(items)}
    </nav>
  );
}

export default async function BlogPostPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const { meta, contentHtml } = await getPostBySlug(slug);
  const { html: htmlWithIds, toc } = extractTableOfContents(contentHtml);

  return (
    <article>
      {/* KaTeX CSS for math rendering */}
      <link
        rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"
        crossOrigin="anonymous"
      />

      {/* Post header */}
      <header className="mb-8">
        <h1 className="text-3xl font-bold">{meta.title}</h1>
        <div className="mt-3 flex flex-wrap items-center gap-3 text-sm text-text-muted">
          <time>
            {new Date(meta.date).toLocaleDateString("en-US", {
              year: "numeric",
              month: "long",
              day: "numeric",
            })}
          </time>
          {meta.tags.length > 0 && (
            <div className="flex gap-2">
              {meta.tags.map((tag) => (
                <span
                  key={tag}
                  className="rounded-full bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary"
                >
                  {tag}
                </span>
              ))}
            </div>
          )}
        </div>
        {meta.excerpt && (
          <blockquote className="mt-4 border-l-4 border-primary/30 pl-4 italic text-text-muted">
            {meta.excerpt}
          </blockquote>
        )}
      </header>

      {/* Table of Contents */}
      <TableOfContents items={toc} />

      {/* Post content — rendered from Markdown to HTML at build time */}
      <div
        className="prose dark:prose-invert max-w-none [&_a]:text-primary [&_blockquote]:border-primary/30 [&_code]:rounded [&_code]:bg-slate-100 dark:[&_code]:bg-slate-800 [&_code]:text-slate-900 dark:[&_code]:text-slate-100 [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-sm [&_img]:block [&_img]:mx-auto [&_img]:rounded-lg [&_pre]:overflow-x-auto [&_pre]:rounded-lg [&_pre]:bg-slate-100 dark:[&_pre]:bg-slate-900 [&_pre]:text-slate-900 dark:[&_pre]:text-slate-100 [&_pre]:p-4 [&_table]:w-full [&_td]:border [&_td]:border-black/10 dark:[&_td]:border-white/10 [&_td]:px-3 [&_td]:py-2 [&_th]:border [&_th]:border-black/10 dark:[&_th]:border-white/10 [&_th]:bg-black/3 dark:[&_th]:bg-white/5 [&_th]:px-3 [&_th]:py-2"
        dangerouslySetInnerHTML={{ __html: htmlWithIds }}
      />
    </article>
  );
}
