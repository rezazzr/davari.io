import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

/**
 * ESLint configuration.
 *
 * ESLint checks your code for common mistakes and enforces coding standards.
 * "next/core-web-vitals" and "next/typescript" are Next.js-specific rule sets
 * that catch React and Next.js-specific issues.
 */
const eslintConfig = [
  ...compat.extends("next/core-web-vitals", "next/typescript"),
];

export default eslintConfig;
