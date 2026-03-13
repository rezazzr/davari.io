import { FaGithub, FaLinkedinIn } from "react-icons/fa";
import { FaXTwitter } from "react-icons/fa6";
import { SiGooglescholar } from "react-icons/si";
import { siteConfig } from "@/data/site-config";

const socialLinks = [
  {
    href: `https://x.com/${siteConfig.social.twitter}`,
    icon: FaXTwitter,
    label: "X",
  },
  {
    href: `https://github.com/${siteConfig.social.github}`,
    icon: FaGithub,
    label: "GitHub",
  },
  {
    href: `https://linkedin.com/in/${siteConfig.social.linkedin}`,
    icon: FaLinkedinIn,
    label: "LinkedIn",
  },
  {
    href: `https://scholar.google.com/citations?user=${siteConfig.social.googleScholar}`,
    icon: SiGooglescholar,
    label: "Google Scholar",
  },
];

interface SocialLinksProps {
  size?: number;
}

export default function SocialLinks({ size = 20 }: SocialLinksProps = {}) {
  return (
    <div className="flex gap-4">
      {socialLinks.map((link) => (
        <a
          key={link.label}
          href={link.href}
          target="_blank"
          rel="noopener noreferrer"
          aria-label={link.label}
          className="text-text-muted transition-all hover:text-primary hover:scale-110 active:scale-90"
        >
          <link.icon size={size} />
        </a>
      ))}
    </div>
  );
}
