import { ArticleMeta } from "@/components/layout/article-meta";

import type { Metadata } from "next";

export const metadata: Metadata = { title: "Method" };

export default function MethodPage() {
  return (
    <article className="flex flex-col gap-12 pb-16">
      <ArticleMeta title="Method" />
      <p className="max-w-prose text-lg text-muted-foreground">Coming next.</p>
    </article>
  );
}
