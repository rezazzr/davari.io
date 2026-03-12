import type { Metadata } from "next";
import Link from "next/link";
import { getAllPosts } from "@/lib/posts";

export const metadata: Metadata = {
  title: "Blog",
  description: "Reza's blog posts.",
  alternates: { canonical: "/blog" },
  openGraph: { title: "Blog | Reza Davari", description: "Reza's blog posts.", url: "/blog" },
};

export default function BlogPage() {
  const posts = getAllPosts();

  return (
    <div>
      <h1 className="text-2xl font-bold text-heading">Blog</h1>

      <div className="mt-8 space-y-6">
        {posts.map((post) => (
          <Link
            key={post.slug}
            href={`/blog/${post.slug}`}
            className="block rounded-xl border border-black/5 dark:border-white/5 bg-surface p-6 transition-shadow hover:shadow-md"
          >
            <h2 className="text-lg font-semibold">{post.title}</h2>
            <p className="mt-1 text-sm text-text-muted">
              {new Date(post.date).toLocaleDateString("en-US", {
                year: "numeric",
                month: "long",
                day: "numeric",
              })}
            </p>
            {post.tags.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-2">
                {post.tags.map((tag) => (
                  <span
                    key={tag}
                    className="rounded-full bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            )}
            {post.excerpt && (
              <p className="mt-3 text-sm leading-relaxed text-text-muted">
                {post.excerpt}
              </p>
            )}
          </Link>
        ))}

        {posts.length === 0 && (
          <p className="text-text-muted">No posts yet.</p>
        )}
      </div>
    </div>
  );
}
