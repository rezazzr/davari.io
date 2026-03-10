import { siteConfig } from "@/data/site-config";

export default function Footer() {
  return (
    <footer className="border-t border-black/10 dark:border-white/10 px-6 py-4 text-center text-sm text-text-muted">
      &copy; {new Date().getFullYear()} {siteConfig.owner.name} &middot;{" "}
      <a
        href={`https://github.com/${siteConfig.social.github}`}
        target="_blank"
        rel="noopener noreferrer"
        className="hover:text-primary"
      >
        GitHub
      </a>
    </footer>
  );
}
