export interface NavItem {
  label: string;
  path: string;
}

export const navItems: NavItem[] = [
  { label: "About", path: "/" },
  { label: "Publication", path: "/publication" },
  { label: "Teaching", path: "/teaching" },
  { label: "FAQ", path: "/faq" },
  { label: "Blog", path: "/blog" },
  { label: "Projects", path: "/projects" },
];
