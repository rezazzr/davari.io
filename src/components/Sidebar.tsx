import Image from "next/image";
import { FaFileDownload } from "react-icons/fa";
import { siteConfig } from "@/data/site-config";
import NavList from "./NavList";
import SocialLinks from "./SocialLinks";
import ThemeToggle from "./ThemeToggle";
import TypewriterText from "./TypewriterText";

export default function Sidebar() {
  return (
    <aside className="hidden md:flex h-full flex-col gap-6 p-6">
      <div className="flex flex-col items-center gap-3 text-center">
        <Image
          src={siteConfig.owner.avatar}
          alt={siteConfig.owner.name}
          width={120}
          height={120}
          className="rounded-full"
          priority
        />
        <div>
          <h1 className="text-xl font-bold text-heading">
            {siteConfig.owner.name}
          </h1>
          <p
            className="job-title mt-1 text-sm text-text-muted"
            dangerouslySetInnerHTML={{ __html: siteConfig.owner.job }}
          />
        </div>
        <p className="text-sm text-text-muted">
          <TypewriterText text={siteConfig.owner.bio} speed={25} delay={300} />
        </p>
      </div>

      <a
        href="/assets/Reza_Davari_CV.pdf"
        target="_blank"
        rel="noopener noreferrer"
        className="flex items-center justify-center gap-2 rounded-lg bg-primary/10 px-4 py-2 text-sm font-medium text-primary transition-colors hover:bg-primary/20"
      >
        <FaFileDownload size={14} />
        Download CV
      </a>

      <NavList />

      <div className="mt-auto flex items-center justify-center gap-3">
        <SocialLinks />
        <ThemeToggle />
      </div>
    </aside>
  );
}
