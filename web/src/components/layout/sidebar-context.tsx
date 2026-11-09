"use client";

import { createContext, useCallback, useContext, useEffect, useState } from "react";

import type { ReactNode } from "react";

interface SidebarContextValue {
  /** Desktop persistent sidebar expanded (≥ lg). */
  open: boolean;
  /** Mobile drawer open (< lg). */
  openMobile: boolean;
  isMobile: boolean;
  /** Toggles the drawer on mobile, the collapse on desktop. */
  toggle: () => void;
  setOpenMobile: (value: boolean) => void;
}

const SidebarContext = createContext<SidebarContextValue>({
  open: true,
  openMobile: false,
  isMobile: false,
  toggle: () => {},
  setOpenMobile: () => {},
});

export function useSidebar() {
  return useContext(SidebarContext);
}

export function SidebarProvider({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(true);
  const [openMobile, setOpenMobile] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const query = window.matchMedia("(max-width: 1023px)");
    const update = () => {
      setIsMobile(query.matches);
      if (!query.matches) setOpenMobile(false); // leaving mobile closes the drawer
    };
    update();
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);

  const toggle = useCallback(() => {
    if (isMobile) setOpenMobile((value) => !value);
    else setOpen((value) => !value);
  }, [isMobile]);

  return (
    <SidebarContext.Provider value={{ open, openMobile, isMobile, toggle, setOpenMobile }}>
      {children}
    </SidebarContext.Provider>
  );
}
