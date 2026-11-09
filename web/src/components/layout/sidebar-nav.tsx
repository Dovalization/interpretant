"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { BookOpenText, ClipboardCheck, Compass, Grid3x3, Type } from "lucide-react";

import { cn } from "@/lib/utils";

interface SidebarNavProps {
  /** Called after a navigation action, used to close the mobile drawer. */
  onNavigate?: () => void;
  /** Collapsed rail: labels fade, the aside handles the width. */
  collapsed?: boolean;
}

export const NAV_LINKS = [
  { href: "/", label: "Overview", icon: Compass },
  { href: "/words", label: "Words", icon: Type },
  { href: "/fields", label: "Fields", icon: Grid3x3 },
  { href: "/predictions", label: "Predictions", icon: ClipboardCheck },
  { href: "/method", label: "Method", icon: BookOpenText },
] as const;

const isActive = (pathname: string, href: string) =>
  href === "/" ? pathname === "/" : pathname.startsWith(href);

export function SidebarNav({ onNavigate, collapsed = false }: SidebarNavProps) {
  const pathname = usePathname();
  const label = cn("whitespace-nowrap transition-opacity duration-200", {
    "pointer-events-none opacity-0": collapsed,
  });

  return (
    <nav className="space-y-1">
      {NAV_LINKS.map(({ href, label: text, icon: Icon }) => (
        <Link
          key={href}
          href={href}
          onClick={onNavigate}
          aria-current={isActive(pathname, href) ? "page" : undefined}
          className={cn(
            "flex items-center gap-2 rounded-md py-1.5 pr-2 pl-0.5 text-lg font-semibold transition-colors",
            isActive(pathname, href)
              ? "text-foreground"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          <Icon className="size-5 shrink-0" />
          <span className={label}>{text}</span>
        </Link>
      ))}
    </nav>
  );
}
