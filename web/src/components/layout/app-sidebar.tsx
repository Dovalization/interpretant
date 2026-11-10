"use client";

import { PanelLeft } from "lucide-react";

import { cn } from "@/lib/utils";

import { useSidebar } from "./sidebar-context";
import { SidebarNav } from "./sidebar-nav";

export function AppSidebar() {
  const { open, toggle } = useSidebar();

  return (
    <aside
      className={cn(
        "sticky top-(--header-height) z-40 hidden h-[calc(100svh-var(--header-height))] shrink-0 overflow-x-hidden overflow-y-auto pb-10 transition-[width] duration-200 lg:block",
        open ? "w-72 pr-8" : "w-16",
      )}
    >
      <button
        onClick={toggle}
        aria-label={open ? "Collapse menu" : "Expand menu"}
        className="mb-4 flex size-9 items-center rounded-md pl-0.5 text-muted-foreground transition-colors hover:text-foreground"
      >
        <PanelLeft className="size-5 shrink-0" />
      </button>
      <SidebarNav collapsed={!open} />
    </aside>
  );
}
