/**
 * Blog post loading utilities.
 *
 * KEY CONCEPT: Markdown processing pipeline
 * We use unified/remark/rehype to convert Markdown → HTML:
 * 1. remarkParse: parses Markdown into an AST (abstract syntax tree)
 * 2. remarkGfm: adds support for GFM tables, strikethrough, etc.
 * 3. remarkMath: extracts math expressions ($...$ and $$...$$)
 * 4. remarkRehype: converts Markdown AST → HTML AST
 * 5. rehypeKatex: renders math to KaTeX HTML
 * 6. rehypeStringify: converts HTML AST → HTML string
 *
 * This all runs at BUILD time (Server Components use Node.js fs APIs).
 */

import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import remarkRehype from "remark-rehype";
import rehypeKatex from "rehype-katex";
import rehypeStringify from "rehype-stringify";

const postsDirectory = path.join(process.cwd(), "src/content/posts");

export interface PostMeta {
  slug: string;
  title: string;
  date: string;
  excerpt: string;
  tags: string[];
}

/**
 * Get metadata for all blog posts, sorted by date (newest first).
 */
export function getAllPosts(): PostMeta[] {
  if (!fs.existsSync(postsDirectory)) return [];

  const fileNames = fs.readdirSync(postsDirectory);
  const posts = fileNames
    .filter((name) => name.endsWith(".mdx") || name.endsWith(".md"))
    .map((fileName) => {
      const slug = fileName.replace(/\.(mdx|md)$/, "");
      const fullPath = path.join(postsDirectory, fileName);
      const fileContents = fs.readFileSync(fullPath, "utf8");
      const { data } = matter(fileContents);

      return {
        slug,
        title: data.title || slug,
        date: data.date || "",
        excerpt: data.excerpt || "",
        tags: data.tags || [],
      };
    });

  return posts.sort(
    (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()
  );
}

/**
 * Get a single post's rendered HTML and metadata by slug.
 */
export async function getPostBySlug(slug: string) {
  const extensions = [".mdx", ".md"];
  let fullPath = "";
  for (const ext of extensions) {
    const candidate = path.join(postsDirectory, `${slug}${ext}`);
    if (fs.existsSync(candidate)) {
      fullPath = candidate;
      break;
    }
  }

  const fileContents = fs.readFileSync(fullPath, "utf8");
  const { data, content } = matter(fileContents);

  // Process Markdown → HTML
  const result = await unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkMath)
    .use(remarkRehype, { allowDangerousHtml: true })
    .use(rehypeKatex)
    .use(rehypeStringify, { allowDangerousHtml: true })
    .process(content);

  return {
    meta: {
      slug,
      title: data.title || slug,
      date: data.date || "",
      excerpt: data.excerpt || "",
      tags: data.tags || [],
    } as PostMeta,
    contentHtml: String(result),
  };
}
