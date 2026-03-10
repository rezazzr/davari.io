export interface TocItem {
  id: string;
  title: string;
  level: number;
  children: TocItem[];
}

function textToSlug(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, "") // Remove special characters
    .replace(/\s+/g, "-") // Replace spaces with hyphens
    .replace(/-+/g, "-") // Replace multiple hyphens with single hyphen
    .trim();
}

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

  while ((match = headingRegex.exec(html)) !== null) {
    const tag = match[1];
    const existingId = match[2];
    const title = match[3];
    const level = parseInt(tag[1], 10);

    const id = existingId || textToSlug(title);
    headings.push({ tag, id, title, level });

    if (!existingId) {
      const originalHeading = match[0];
      const newHeading = `<${tag} id="${id}"${originalHeading.slice(tag.length + 1)}`;
      modifiedHtml = modifiedHtml.replace(originalHeading, newHeading);
    }
  }

  const toc: TocItem[] = [];
  const stack: Array<{ item: TocItem; level: number }> = [];

  for (const heading of headings) {
    const item: TocItem = {
      id: heading.id,
      title: heading.title,
      level: heading.level,
      children: [],
    };

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
