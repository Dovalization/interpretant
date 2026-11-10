"use client";

import { useEffect, useRef, useState } from "react";

import { type ScrollEdges, scrollEdges, scrollMask } from "@/lib/scroll";
import { cn } from "@/lib/utils";

import type { ReactNode } from "react";

/** A horizontally scrolling strip whose edges fade where more content hides. */
export function HorizontalScroll({
  className,
  children,
}: {
  className?: string;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [edges, setEdges] = useState<ScrollEdges>({ hiddenStart: false, hiddenEnd: false });

  useEffect(() => {
    const strip = ref.current;
    if (!strip) return;
    const update = () =>
      setEdges(scrollEdges(strip.scrollLeft, strip.clientWidth, strip.scrollWidth));
    update();
    strip.addEventListener("scroll", update, { passive: true });
    const observer = new ResizeObserver(update);
    observer.observe(strip);
    [...strip.children].forEach((child) => observer.observe(child));
    return () => {
      strip.removeEventListener("scroll", update);
      observer.disconnect();
    };
  }, [children]);

  const mask = scrollMask(edges);
  return (
    <div
      ref={ref}
      className={cn("overflow-x-auto", className)}
      style={{ maskImage: mask, WebkitMaskImage: mask }}
    >
      {children}
    </div>
  );
}
