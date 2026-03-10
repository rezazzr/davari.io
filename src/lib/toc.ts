/**
 * Table of Contents generation utilities for blog posts.
 * Extracts headings from HTML and generates a hierarchical TOC structure.
 */

export interface TocItem {
  id: string;
  title: string;
  level: number;
  children: TocItem[];
}

/**
 * Convert text to a slug suitable for use as an HTML id
 */
function textToSlug(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, "") // Remove special characters
    .replace(/\s+/g, "-") // Replace spaces with hyphens
    .replace(/-+/g, "-") // Replace multiple hyphens with single hyphen
    .trim();
}

/**
 * Extract headings from HTML and generate a table of contents
 * Also adds id attributes to headings if they don't have them
 *
 * @param html - The HTML content
 * @returns Object containing modified HTML and TOC structure
 */
export function extractTableOfContents(html: string): {
  html: string;
  toc: TocItem[];
} {
  const headingRegex = /<(h[1-6])(?:\s+id="([^"]*)")?[^>]*>([^<]+)<\/h[1-6]>/g;
  const headings: Array<{
    tag: string;
    id: string;
    title: string;
    level: number;
  }> = [];

  let match: RegExpExecArray | null;
  let modifiedHtml = html;

  // Extract all headings and assign IDs
  while ((match = headingRegex.exec(html)) !== null) {
    const tag = match[1];
    const existingId = match[2];
    const title = match[3];
    const level = parseInt(tag[1], 10);

    const id = existingId || textToSlug(title);
    headings.push({ tag, id, title, level });

    // Add id to heading if it doesn't have one
    if (!existingId) {
      const originalHeading = match[0];
      const newHeading = `<${tag} id="${id}"${originalHeading.slice(tag.length + 1)}`;
      modifiedHtml = modifiedHtml.replace(originalHeading, newHeading);
    }
  }

  // Build hierarchical TOC structure
  const toc: TocItem[] = [];
  const stack: Array<{ item: TocItem; level: number }> = [];

  for (const heading of headings) {
    const item: TocItem = {
      id: heading.id,
      title: heading.title,
      level: heading.level,
      children: [],
    };

    // Find correct parent and insert item
    while (
      stack.length > 0 &&
      stack[stack.length - 1].level >= heading.level
    ) {
      stack.pop();
    }

    if (stack.length === 0) {
      toc.push(item);
    } else {
      stack[stack.length - 1].item.children.push(item);
    }

    stack.push({ item, level: heading.level });
  }

  return { html: modifiedHtml, toc };
}
