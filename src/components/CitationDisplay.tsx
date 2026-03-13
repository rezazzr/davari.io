"use client";

import { useMemo } from "react";
import { useTheme } from "@/hooks/useTheme";

interface CitationDisplayProps {
  citation: string;
}

/** Parse BibTeX fields, handling nested braces like {CKA} inside values. */
function parseBibtex(citation: string): {
  type: string;
  key: string;
  fields: Array<[string, string]>;
} | null {
  const headerMatch = citation.match(/@(\w+)\{\s*([^,\s]+)\s*,/s);
  if (!headerMatch) return null;

  const [, type, key] = headerMatch;
  const fields: Array<[string, string]> = [];

  // Get the body after the header (everything after "@type{key,")
  const bodyStart = citation.indexOf(",", citation.indexOf(key)) + 1;
  const body = citation.slice(bodyStart);

  // Match each field: key = {value} or key = "value" or key = number
  const fieldPattern = /(\w+)\s*=\s*/g;
  let fieldMatch;

  while ((fieldMatch = fieldPattern.exec(body)) !== null) {
    const fieldKey = fieldMatch[1];
    let valueStart = fieldPattern.lastIndex;

    // Skip whitespace
    while (valueStart < body.length && body[valueStart] === " ") valueStart++;

    let value: string;
    if (body[valueStart] === "{") {
      // Brace-delimited: count nesting
      let depth = 0;
      let end = valueStart;
      for (; end < body.length; end++) {
        if (body[end] === "{") depth++;
        else if (body[end] === "}") {
          depth--;
          if (depth === 0) break;
        }
      }
      value = body.slice(valueStart + 1, end);
    } else if (body[valueStart] === '"') {
      const closeQuote = body.indexOf('"', valueStart + 1);
      value = body.slice(valueStart + 1, closeQuote);
    } else {
      // Bare value (number)
      const endMatch = body.slice(valueStart).match(/^(\S+)/);
      value = endMatch ? endMatch[1].replace(/,$/, "") : "";
    }

    fields.push([fieldKey, value.trim()]);
  }

  return { type, key, fields };
}

export default function CitationDisplay({ citation }: CitationDisplayProps) {
  const { isDark } = useTheme();

  const parsed = useMemo(() => parseBibtex(citation), [citation]);
  if (!parsed) return <pre className="whitespace-pre-wrap">{citation}</pre>;

  const { type, key, fields } = parsed;

  const maxKeyLen = useMemo(
    () => Math.max(...fields.map(([k]) => k.length)),
    [fields]
  );

  const colors = useMemo(
    () => ({
      type: isDark ? "#60a5fa" : "#2563eb",
      citeKey: isDark ? "#22d3ee" : "#0891b2",
      key: isDark ? "#fbbf24" : "#d97706",
      punct: isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.45)",
      value: isDark ? "rgba(255,255,255,0.85)" : "rgba(0,0,0,0.8)",
    }),
    [isDark]
  );

  return (
    <pre className="whitespace-pre-wrap text-xs leading-relaxed">
      <span style={{ color: colors.type }}>@{type}</span>
      <span style={{ color: colors.punct }}>{"{"}</span>
      <span style={{ color: colors.citeKey }}>{key}</span>
      <span style={{ color: colors.punct }}>{","}</span>
      {"\n"}
      {fields.map(([fieldKey, value], idx) => (
        <span key={idx}>
          {"  "}
          <span style={{ color: colors.key }}>{fieldKey.padEnd(maxKeyLen)}</span>
          <span style={{ color: colors.punct }}>{" = "}</span>
          <span style={{ color: colors.punct }}>{"{"}</span>
          <span style={{ color: colors.value }}>{value}</span>
          <span style={{ color: colors.punct }}>{"}"}</span>
          {idx < fields.length - 1 && <span style={{ color: colors.punct }}>,</span>}
          {"\n"}
        </span>
      ))}
      <span style={{ color: colors.punct }}>{"}"}</span>
    </pre>
  );
}
