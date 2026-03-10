/**
 * Site-wide configuration — converted from Jekyll's _config.yml.
 *
 * This single file replaces the owner/site sections of _config.yml.
 * In Jekyll, you accessed these values with {{ site.owner.name }}.
 * In Next.js, you import them: import { siteConfig } from '@/data/site-config';
 */

export const siteConfig = {
  title: "Reza Davari",
  email: "davari.mreza@gmail.com",
  description: "Reza's notes and projects.",
  url: "https://davari.io",
  keywords: [
    "Mohammad Reza Davari",
    "Reza Davari",
    "Machine Learning",
    "NLP",
    "AI",
    "Artificial Intelligence",
  ],

  owner: {
    name: "Reza Davari",
    job: '<span style="color: #8d7edc">Senior Applied Scientist @</span> <a href="https://www.microsoft.com/en-us" target="_blank" style="color: #42b983; text-decoration: none; font-weight: 500">Microsoft</a>',
    bio: "AI enthusiast, currently focused on Agentic Systems, Multimodal Models, and Continual Learning. A watermelon connoisseur. Based in Redmond, WA.",
    avatar: "/assets/img/reza_avatar.jpg",
  },

  social: {
    twitter: "davari_reza",
    github: "rezazzr",
    linkedin: "rezadavari",
    googleScholar: "4AztFtEAAAAJ",
  },

  analytics: {
    gaId: "UA-148531218-1",
  },
} as const;
