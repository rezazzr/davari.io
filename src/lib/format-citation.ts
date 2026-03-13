/**
 * Format a BibTeX citation string with consistent spacing and structure
 */
export function formatCitation(cite: string): string {
  // Extract the citation type and key
  const typeMatch = cite.match(/@(\w+)\{([^,]+),/);
  if (!typeMatch) return cite;

  const [, citationType, citationKey] = typeMatch;

  // Extract all key-value pairs
  const pairs: Array<[string, string]> = [];
  const fieldRegex = /(\w+)\s*=\s*\{([^}]+)\}|(\w+)\s*=\s*"([^"]+)"|(\w+)\s*=\s*(\d+)/g;

  let match;
  while ((match = fieldRegex.exec(cite)) !== null) {
    const key = match[1] || match[3] || match[5];
    const value = match[2] || match[4] || match[6];
    if (key && value) {
      pairs.push([key, value.trim()]);
    }
  }

  // Format with consistent spacing
  const formattedLines = [
    `@${citationType}{${citationKey},`,
    ...pairs.map(([key, value]) => `  ${key} = {${value}},`),
    "}",
  ];

  // Remove trailing comma from last field
  formattedLines[formattedLines.length - 2] = formattedLines[
    formattedLines.length - 2
  ].slice(0, -1);

  return formattedLines.join("\n");
}
