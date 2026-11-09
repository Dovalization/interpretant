"use client";

import { NuqsAdapter } from "nuqs/adapters/next/app";

import { TooltipProvider } from "@/components/ui/tooltip";

import type { ReactNode } from "react";

export function AppProviders({ children }: { children: ReactNode }) {
  return (
    <NuqsAdapter>
      <TooltipProvider>{children}</TooltipProvider>
    </NuqsAdapter>
  );
}
