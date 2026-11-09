"use client";

import { PanelLeft } from "lucide-react";
import { motion } from "motion/react";

import { cn } from "@/lib/utils";

import { useSidebar } from "./sidebar-context";
import { SidebarNav } from "./sidebar-nav";

// Slide-in-from-left entrance, played once on first mount.
const EXPO: [number, number, number, number] = [0.16, 1, 0.3, 1];

export function AppSidebar() {
  const { open, toggle } = useSidebar();

  return (
    <motion.aside
      initial={{ x: -40, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.9, ease: EXPO, delay: 0.1 }}
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
    </motion.aside>
  );
}
