# davari.io

Personal portfolio and blog site built with [Next.js](https://nextjs.org) and deployed to GitHub Pages.

## Tech Stack

- **Framework:** Next.js 15 with App Router
- **Styling:** Tailwind CSS 4
- **Content:** MDX for blog posts
- **Markup:** Markdown + LaTeX support via remark/rehype
- **Deployment:** GitHub Actions → GitHub Pages (static export)

## Project Structure

```
src/
├── app/              # Next.js pages
├── components/       # Reusable React components
├── content/          # MDX blog posts
├── data/             # Structured data (YAML → TypeScript)
├── lib/              # Utilities (posts, table of contents)
└── content/posts/    # Blog posts
public/
├── assets/img/       # Images and media
└── CNAME            # Custom domain
```

## Getting Started

### Development

```bash
npm install
npm run dev          # Start dev server at http://localhost:3000
```

### Build & Deploy

```bash
npm run build         # Generate static export to `out/`
npm run start         # Preview production build
```

Push to `master` branch to trigger automatic GitHub Actions deployment to GitHub Pages.

## Features

- Responsive design with Tailwind CSS
- Dark/light theme toggle
- Blog with syntax highlighting and LaTeX rendering
- Interactive charts (Chart.js)
- Timeline components for experience/education
- Table of contents for blog posts

## Mobile Optimizations

- **Header overflow handling:** Job title abbreviates to "Sr." on mobile and scrolls smoothly with marquee animation when text exceeds container width
- **Radar chart responsiveness:** Skill labels scale down from 12px to 10px on narrow screens (<400px width) to prevent overlap
- **Dynamic resize detection:** Components respond to orientation changes and window resizing in real-time

## License

See [LICENSE.txt](LICENSE.txt) for details.
