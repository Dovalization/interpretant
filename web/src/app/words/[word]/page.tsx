import { Suspense } from "react";
import { notFound } from "next/navigation";

import { WordExplorer } from "@/components/explorer/word-explorer";
import { ArticleMeta } from "@/components/layout/article-meta";
import { Skeleton } from "@/components/ui/skeleton";
import { getWord, getWords } from "@/data";
import { defaultFields, defaultLayer } from "@/lib/explorer";

import type { Metadata } from "next";

export const dynamicParams = false;

export function generateStaticParams() {
  return getWords().map(({ word }) => ({ word }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ word: string }>;
}): Promise<Metadata> {
  const { word } = await params;
  return { title: word, description: `How ${word} is used across fields and decades.` };
}

export default async function WordPage({ params }: { params: Promise<{ word: string }> }) {
  const { word } = await params;
  if (!getWords().some((entry) => entry.word === word)) notFound();
  const data = getWord(word);
  const layer = defaultLayer(data);

  return (
    <article className="flex flex-col gap-8 pb-16">
      <ArticleMeta backHref="/words" backLabel="Words" title={word.replaceAll("_", " ")} />
      <Suspense fallback={<Skeleton className="h-96 w-full" />}>
        <WordExplorer data={data} defaultLayer={layer} defaultFields={defaultFields(data, layer)} />
      </Suspense>
    </article>
  );
}
