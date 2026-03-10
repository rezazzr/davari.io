/**
 * Navigation items — converted from _data/index/nav.yml.
 *
 * KEY CONCEPT: TypeScript interfaces.
 * An interface describes the "shape" of an object. If you try to create
 * a NavItem without a 'label' or 'path', TypeScript will show an error
 * in your editor immediately — before you even run the code.
 */

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
