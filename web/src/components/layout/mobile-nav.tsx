"use client";

import { Dialog as DialogPrimitive } from "@base-ui/react/dialog";

import { useSidebar } from "./sidebar-context";
import { SidebarNav } from "./sidebar-nav";

export function MobileNav() {
  const { openMobile, setOpenMobile } = useSidebar();

  return (
    <DialogPrimitive.Root open={openMobile} onOpenChange={setOpenMobile}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Backdrop className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm data-open:animate-in data-open:fade-in-0 data-closed:animate-out data-closed:fade-out-0 lg:hidden" />
        <DialogPrimitive.Popup className="fixed inset-y-0 left-0 z-50 w-72 max-w-[80vw] overflow-y-auto border-r bg-background px-5 pb-10 pt-(--header-height) data-open:animate-in data-open:slide-in-from-left data-closed:animate-out data-closed:slide-out-to-left lg:hidden">
          <DialogPrimitive.Title className="sr-only">Navigation</DialogPrimitive.Title>
          <div className="pt-6">
            <SidebarNav onNavigate={() => setOpenMobile(false)} />
          </div>
        </DialogPrimitive.Popup>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}
