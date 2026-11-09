"use client";

import { useSyncExternalStore } from "react";
import Link from "next/link";

import { PanelLeft } from "lucide-react";

import { useSidebar } from "./sidebar-context";

const REPOSITORY = "https://github.com/Dovalization/interpretant";

function GitHubMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 16 16" aria-hidden className={className} fill="currentColor">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z" />
    </svg>
  );
}

// The wordmark shrinks once the page is scrolled; useSyncExternalStore keeps it SSR-safe.
const subscribeScroll = (callback: () => void) => {
  window.addEventListener("scroll", callback, { passive: true });
  return () => window.removeEventListener("scroll", callback);
};
const getScrolled = () => window.scrollY > 8;
const useScrolled = () => useSyncExternalStore(subscribeScroll, getScrolled, () => false);

export function AppHeader() {
  const { toggle } = useSidebar();
  const scrolled = useScrolled();

  return (
    <header className="fixed inset-x-0 top-0 z-50 h-(--header-height)">
      <div className="container mx-auto flex h-full items-center gap-3 px-4 sm:px-8">
        <button
          onClick={toggle}
          aria-label="Open menu"
          className="-ml-2 rounded-md p-2 text-muted-foreground transition-colors hover:text-foreground lg:hidden"
        >
          <PanelLeft className="size-5" />
        </button>
        <Link
          href="/"
          className={`font-heading italic tracking-tight text-foreground transition-[font-size] duration-300 ease-[cubic-bezier(0.16,1,0.3,1)] hover:opacity-70 ${
            scrolled ? "text-3xl" : "text-4xl"
          }`}
        >
          interpretant
        </Link>
        <span className="hidden text-sm text-muted-foreground sm:inline">
          how fields change what they do with a word
        </span>
        <a
          href={REPOSITORY}
          className="ml-auto inline-flex items-center gap-2 rounded-md px-2 py-1 text-sm font-semibold text-muted-foreground transition-colors hover:text-foreground"
        >
          <GitHubMark className="size-4" />
          <span className="hidden sm:inline">GitHub</span>
        </a>
      </div>
    </header>
  );
}
