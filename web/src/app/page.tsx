import Link from "next/link";

import { ArrowRight } from "lucide-react";

import { CorpusGrid } from "@/components/explorer/corpus-grid";
import { FigurePanel } from "@/components/figure-panel";
import { ArticleMeta } from "@/components/layout/article-meta";
import { StatStrip } from "@/components/stat-strip";
import { getCorpus, getPredictions, getWords } from "@/data";

const compact = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });

const FINDINGS = [
  {
    title: "Embodiment means two different things",
    body: "In HCI it is virtual reality's sense of body ownership and presence; in philosophy it is phenomenological corporeality. Robust to a second seed and a second method.",
    href: "/words/embodiment?fields=philosophy,hci&layer=openalex_full",
    cta: "Compare the neighbourhoods",
  },
  {
    title:
      "Philosophy's model learned to predict HCI's abstracts; HCI's did not learn philosophy's",
    body: "From the 1990s to the 2020s, robust to shared abstracts, training size and five resampled runs. The effect is specific to HCI and its cause is open.",
    href: "/fields?model=philosophy&text=hci",
    cta: "See who predicts whom",
  },
] as const;

export default function HomePage() {
  const corpus = getCorpus();
  const words = getWords();
  const predictions = getPredictions();
  const grid = corpus.cells.filter((cell) => cell.layer === "openalex");
  const abstracts = grid.reduce((sum, cell) => sum + (cell.documents ?? 0), 0);
  const bookTokens = corpus.cells
    .filter((cell) => cell.layer === "books")
    .reduce((sum, cell) => sum + cell.tokens, 0);
  const scored = predictions.predictions.filter((prediction) => prediction.verdict);
  const passed = scored.filter((prediction) => prediction.verdict === "passed").length;
  const fields = corpus.fields.map((field) => field.slug);

  return (
    <article className="flex flex-col gap-12 pb-16">
      <div className="flex flex-col gap-6">
        <ArticleMeta title="How fields change what they do with a word" />
        <p className="max-w-prose text-lg text-muted-foreground">
          One word-embedding model per field and decade, trained on a grid of research abstracts and
          two centuries of canonical books, in one shared space. Every word&apos;s movement is
          judged against the movement of the whole vocabulary, and every claim is tested against
          predictions committed to git before the evidence existed.
        </p>
      </div>

      <StatStrip
        stats={[
          { label: "Fields", value: fields.length, detail: "one OpenAlex taxonomy" },
          { label: "Abstracts", value: compact.format(abstracts), detail: "1950s to 2020s" },
          { label: "Book tokens", value: compact.format(bookTokens), detail: "1820s to 2020s" },
          {
            label: "Predictions",
            value: `${passed} / ${scored.length}`,
            detail: "passed, committed before the evidence",
          },
        ]}
      />

      <section className="grid gap-4 md:grid-cols-2">
        {FINDINGS.map((finding) => (
          <Link
            key={finding.href}
            href={finding.href}
            className="group flex flex-col gap-3 rounded-lg border bg-card p-6 transition-colors hover:bg-accent"
          >
            <h2 className="font-heading text-2xl leading-snug font-semibold">{finding.title}</h2>
            <p className="text-muted-foreground">{finding.body}</p>
            <span className="mt-auto inline-flex items-center gap-1.5 text-sm font-semibold text-foreground">
              {finding.cta}
              <ArrowRight className="size-4 transition-transform group-hover:translate-x-0.5" />
            </span>
          </Link>
        ))}
      </section>

      <FigurePanel
        heading="The corpus is a grid"
        subheading="Research abstracts per field and decade cell after cleaning, sampled to 10,000 per cell"
        legend="OpenAlex, a seeded sample of English research prose; duplicates, catalogue records, scraped pages and non-English text removed; no author above 1% of a cell. Philosophy and HCI also have full cells, and the books carry the decades before 1950."
      >
        <CorpusGrid
          cells={grid}
          fields={fields}
          decades={corpus.decades.filter((d) => d >= 1950)}
        />
      </FigurePanel>

      <p className="text-muted-foreground">
        Start with a word:{" "}
        <Link
          href="/words/consciousness"
          className="underline underline-offset-4 hover:text-foreground"
        >
          consciousness
        </Link>
        ,{" "}
        <Link
          href="/words/representation"
          className="underline underline-offset-4 hover:text-foreground"
        >
          representation
        </Link>
        ,{" "}
        <Link
          href="/words/information"
          className="underline underline-offset-4 hover:text-foreground"
        >
          information
        </Link>
        , or any of the{" "}
        <Link href="/words" className="underline underline-offset-4 hover:text-foreground">
          {words.length} target words
        </Link>
        .
      </p>
    </article>
  );
}
