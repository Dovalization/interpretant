import Link from "next/link";

import { ArrowLeft } from "lucide-react";

interface ArticleMetaProps {
  /** Parent section link (the "back" affordance). */
  backHref?: string;
  backLabel?: string;
  title: string;
}

/** Top block of an article page: back-link, title, and a separating rule. */
export function ArticleMeta({ backHref, backLabel, title }: ArticleMetaProps) {
  return (
    <header className="flex flex-col gap-4">
      <div className="flex flex-col gap-1">
        {backHref && backLabel && (
          <Link
            href={backHref}
            className="inline-flex w-fit items-center gap-1.5 text-lg font-semibold text-muted-foreground transition-colors hover:text-foreground"
          >
            <ArrowLeft className="size-5" />
            {backLabel}
          </Link>
        )}
        <h1 className="text-4xl font-black tracking-tight leading-tight">{title}</h1>
      </div>
      <div
        aria-hidden
        className="h-0.5 w-full max-w-xs rounded-l-full bg-gradient-to-r from-chart-1 to-transparent"
      />
    </header>
  );
}
