import { ArticleMeta } from "@/components/layout/article-meta";

import type { Metadata } from "next";
import type { ReactNode } from "react";

export const metadata: Metadata = {
  title: "Method",
  description: "Corpus, embeddings, the noise floor, cross-perplexity and pre-registration.",
};

const REPOSITORY = "https://github.com/Dovalization/interpretant";

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="flex flex-col gap-3">
      <h2 className="font-heading text-2xl font-semibold">{title}</h2>
      <div className="flex max-w-prose flex-col gap-3 text-muted-foreground">{children}</div>
    </section>
  );
}

export default function MethodPage() {
  return (
    <article className="flex flex-col gap-10 pb-16">
      <div className="flex flex-col gap-6">
        <ArticleMeta title="Method" />
        <p className="max-w-prose text-lg text-muted-foreground">
          The short version. The settings in force, the rules and their rationale are in the
          repository&apos;s{" "}
          <a
            href={`${REPOSITORY}/blob/main/docs/methods.md`}
            className="underline underline-offset-4 hover:text-foreground"
          >
            methods
          </a>{" "}
          and{" "}
          <a
            href={`${REPOSITORY}/blob/main/docs/corpus.md`}
            className="underline underline-offset-4 hover:text-foreground"
          >
            corpus
          </a>{" "}
          pages.
        </p>
      </div>

      <Section title="The corpus is a grid">
        <p>
          Every question compares something specific, one field across decades or several fields in
          one decade, so the corpus is a grid of cells, one per field and decade, each built the
          same way: a seeded sample of up to 10,000 English research abstracts from OpenAlex for
          eleven fields that share one topic taxonomy, 1950s to 2020s. Only research prose is kept;
          duplicates, catalogue records, scraped pages and non-English text are removed, and no
          author may exceed 1% of a cell. Where a word is too rare for a sample, its home field is
          taken in full (philosophy, HCI).
        </p>
        <p>
          Regular plurals and British spellings are collapsed before training; verbs, participles
          and <em>-ing</em> nominalisations are left alone, because a general lemmatiser turns{" "}
          <em>learning</em> into <em>learn</em> and <em>meaning</em> into <em>mean</em>.
        </p>
      </Section>

      <Section title="One model per cell, one space">
        <p>
          Word2Vec models trained separately share no coordinate system. TWEC trains a compass
          first, here on the same token budget from every cell so that large recent cells do not
          define the space for the 1950s, and then trains each cell&apos;s model with the
          compass&apos;s context vectors frozen. Every cell lands in one space, so a word can be
          compared across decades within a field and across fields within a decade. A count-based
          PPMI-SVD model per cell is the second method.
        </p>
      </Section>

      <Section title="Drift against the corpus itself">
        <p>
          Retraining a cell moves an ordinary word&apos;s neighbours almost as much as a decade
          does, so a raw distance means little. Within each pair of cells, every word with at least
          150 occurrences in both is ranked within its frequency bin on cosine drift and on the
          change of its 25 nearest neighbours; only words in the top 5% on both count as candidates.
          Everyday control words check the floor and are flagged in under 1% of measurements.
        </p>
      </Section>

      <Section title="Who predicts whom">
        <p>
          A Kneser-Ney bigram model per cell scores every other cell&apos;s held-out abstracts. The
          excess cross-entropy of field B&apos;s text under field A&apos;s model, over B&apos;s own
          model, measures how predictable B&apos;s language is from A&apos;s. Parent and subfield
          cells share no abstracts, every model trains on the same number of tokens, and each run is
          repeated with five seeds.
        </p>
      </Section>

      <Section title="Nothing counts once">
        <p>
          A candidate becomes a finding only if it survives a second training seed, the count-based
          method, and a prediction written and committed before the evidence was looked at. The
          predictions page is that ledger, misses included.
        </p>
      </Section>

      <Section title="Why interpretant">
        <p>
          In Peirce&apos;s theory of signs, the interpretant is neither the word nor the thing it
          points to; it is the meaning the sign produces in a community of minds, and it is not
          fixed. A word vector records what a community did with a sign, which words it appeared
          near, which arguments it enabled. So the pipeline does not measure a word drifting; it
          measures a community changing what it does with a sign, which is why philosophy&apos;s{" "}
          <em>intelligence</em> and computer science&apos;s <em>intelligence</em> are treated as two
          things, and why the corpus is split by field.
        </p>
      </Section>
    </article>
  );
}
