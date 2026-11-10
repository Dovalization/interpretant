import Link from "next/link";

import { ArticleMeta } from "@/components/layout/article-meta";
import { getWords } from "@/data";

import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Words",
  description: "The target words, fixed before the models were trained.",
};

const TIER_NOTES: Record<string, string> = {
  core: "Measurable across most of the field × decade grid.",
  "late arrivals": "Each series starts in the first decade where the word reaches 150 occurrences.",
  "context only": "Everyday senses dominate; read in the books and communication cells.",
  "phrase targets": "Always merged by phrase detection, so the phrase gets its own vectors.",
};

export default function WordsPage() {
  const words = getWords();
  const tiers = [...new Set(words.map((word) => word.tier))];

  return (
    <article className="flex flex-col gap-12 pb-16">
      <div className="flex flex-col gap-6">
        <ArticleMeta title="Words" />
        <p className="max-w-prose text-lg text-muted-foreground">
          {words.length} target words, fixed and committed before the grid was trained so the list
          could not be tuned to the results. The count is the number of cells where the word has at
          least 150 occurrences and so a measurable neighbourhood.
        </p>
      </div>
      {tiers.map((tier) => (
        <section key={tier}>
          <h2 className="mb-1 font-heading text-3xl font-semibold first-letter:uppercase">
            {tier}
          </h2>
          <p className="mb-5 text-muted-foreground">{TIER_NOTES[tier]}</p>
          <ul className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {words
              .filter((word) => word.tier === tier)
              .map((word) => (
                <li key={word.word}>
                  <Link
                    href={`/words/${word.word}`}
                    className="flex items-baseline justify-between gap-3 rounded-md border bg-card px-4 py-2.5 transition-colors hover:bg-accent"
                  >
                    <span className="font-semibold">{word.word.replaceAll("_", " ")}</span>
                    <span className="text-sm text-muted-foreground tabular-nums">
                      {word.measurableCells} {word.measurableCells === 1 ? "cell" : "cells"}
                    </span>
                  </Link>
                </li>
              ))}
          </ul>
        </section>
      ))}
    </article>
  );
}
