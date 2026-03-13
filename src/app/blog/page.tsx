import type { Metadata } from "next";
import { getAllPosts } from "@/lib/posts";
import BlogCard from "@/components/BlogCard";
import { REVEAL_ANIMATION_DELAY_INCREMENT_MS } from "@/lib/constants";

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
        {posts.map((post, i) => (
          <BlogCard
            key={post.slug}
            slug={post.slug}
            title={post.title}
            date={post.date}
            tags={post.tags}
            excerpt={post.excerpt}
            delay={i * REVEAL_ANIMATION_DELAY_INCREMENT_MS}
          />
        ))}

        {posts.length === 0 && (
          <p className="text-text-muted">No posts yet.</p>
        )}
      </div>
    </div>
  );
}
