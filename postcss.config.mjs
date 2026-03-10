/**
 * PostCSS configuration.
 *
 * PostCSS is a CSS processing tool. Tailwind CSS v4 uses it as a plugin
 * to transform your utility classes into actual CSS at build time.
 *
 * This file tells PostCSS: "use @tailwindcss/postcss to process my CSS."
 */
const config = {
  plugins: {
    "@tailwindcss/postcss": {},
  },
};

export default config;
