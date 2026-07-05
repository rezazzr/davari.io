"use client";

import { useState } from "react";
import { FaRegCopy, FaCheck } from "react-icons/fa";

interface CopyButtonProps {
  /** the text copied to the clipboard */
  value: string;
  /** optional context for the accessible label, e.g. "Zelle email" */
  label?: string;
}

export default function CopyButton({ value, label }: CopyButtonProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard can be unavailable (e.g. insecure context); fail silently.
    }
  };

  return (
    <button
      type="button"
      onClick={handleCopy}
      aria-label={copied ? "Copied to clipboard" : `Copy ${label ?? value}`}
      className="inline-flex shrink-0 items-center gap-1.5 rounded-md bg-primary/10 px-3 py-1.5 text-sm font-medium text-primary transition-all hover:bg-primary/20 active:scale-95"
    >
      {copied ? <FaCheck size={13} /> : <FaRegCopy size={13} />}
      {copied ? "Copied!" : "Copy"}
    </button>
  );
}
