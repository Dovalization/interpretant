import { Suspense } from "react";

import { FieldsExplorer } from "@/components/explorer/fields-explorer";
import { ArticleMeta } from "@/components/layout/article-meta";
import { Skeleton } from "@/components/ui/skeleton";
import { getPerplexity } from "@/data";

import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Fields",
  description: "Which field's language predicts which, decade by decade.",
};

export default function FieldsPage() {
  return (
    <article className="flex flex-col gap-8 pb-16">
      <div className="flex flex-col gap-6">
        <ArticleMeta title="Fields" />
        <p className="max-w-prose text-lg text-muted-foreground">
          A bigram language model per field and decade scores every other field&apos;s held-out
          abstracts. The excess over the field&apos;s own model measures how predictable one
          field&apos;s language is from another&apos;s, and how that changed.
        </p>
      </div>
      <Suspense fallback={<Skeleton className="h-96 w-full" />}>
        <FieldsExplorer table={getPerplexity()} />
      </Suspense>
    </article>
  );
}
