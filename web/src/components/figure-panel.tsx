import { cn } from "@/lib/utils";

import type { ReactNode } from "react";

interface FigurePanelProps {
  /** Serif heading, left-aligned like a section title. */
  heading?: ReactNode;
  /** Secondary line beneath the heading. */
  subheading?: ReactNode;
  /** Controls rendered on the heading row's right. */
  actions?: ReactNode;
  /** Caption beneath the content: method, source, how to read it. */
  legend?: ReactNode;
  className?: string;
  children: ReactNode;
}

/** Shared shell for every chart and panel: bordered muted figure, heading, legend. */
export function FigurePanel({
  heading,
  subheading,
  actions,
  legend,
  className,
  children,
}: FigurePanelProps) {
  return (
    <figure className={cn("rounded-lg border bg-muted/30 p-4 sm:p-6", className)}>
      {(heading || subheading || actions) && (
        <header className="mb-5 flex flex-wrap items-start justify-between gap-3">
          <div>
            {heading && (
              <p className="font-heading text-2xl font-semibold text-foreground">{heading}</p>
            )}
            {subheading && <p className="mt-1 text-base text-muted-foreground">{subheading}</p>}
          </div>
          {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
        </header>
      )}
      {children}
      {legend && <figcaption className="mt-5 text-sm text-muted-foreground">{legend}</figcaption>}
    </figure>
  );
}
