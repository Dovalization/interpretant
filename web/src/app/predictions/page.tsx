import { ArticleMeta } from "@/components/layout/article-meta";

import type { Metadata } from "next";

export const metadata: Metadata = { title: "Predictions" };

export default function PredictionsPage() {
  return (
    <article className="flex flex-col gap-12 pb-16">
      <ArticleMeta title="Predictions" />
      <p className="max-w-prose text-lg text-muted-foreground">Coming next.</p>
    </article>
  );
}
