import { AppHeader } from "@/components/layout/app-header";
import { AppSidebar } from "@/components/layout/app-sidebar";
import { MobileNav } from "@/components/layout/mobile-nav";
import { SidebarProvider } from "@/components/layout/sidebar-context";

import type { ReactNode } from "react";

/** Fixed header, a collapsible sidebar from `lg` up, a drawer below it, one main column. */
export function AppShell({ children }: { children: ReactNode }) {
  return (
    <SidebarProvider>
      <div className="pointer-events-none fixed inset-x-0 top-0 z-30 h-[calc(var(--header-height)*1.4)] bg-gradient-to-b from-background via-background/80 to-transparent" />
      <AppHeader />
      <MobileNav />
      <div className="pt-(--header-height)">
        <div className="container mx-auto flex px-4 pt-8 sm:px-8">
          <AppSidebar />
          <main className="min-w-0 flex-1">{children}</main>
        </div>
      </div>
    </SidebarProvider>
  );
}
