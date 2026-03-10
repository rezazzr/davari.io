import Image from "next/image";
import { siteConfig } from "@/data/site-config";
import NavList from "./NavList";
import SocialLinks from "./SocialLinks";
import ThemeToggle from "./ThemeToggle";

export default function Sidebar() {
  return (
    <aside className="flex h-full flex-col gap-6 p-6">
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
          <h1 className="text-xl font-bold" style={{ color: "#F36170" }}>
            {siteConfig.owner.name}
          </h1>
          <p
            className="job-title mt-1 text-sm text-text-muted"
            dangerouslySetInnerHTML={{ __html: siteConfig.owner.job }}
          />
        </div>
        <p className="text-sm text-text-muted">{siteConfig.owner.bio}</p>
      </div>

      <NavList />

      <div className="mt-auto flex items-center justify-center gap-3">
        <SocialLinks />
        <ThemeToggle />
      </div>
    </aside>
  );
}
