import { Scorecard } from "@/components/explorer/scorecard";
import { ArticleMeta } from "@/components/layout/article-meta";
import { getPredictions } from "@/data";

import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Predictions",
  description: "Every prediction, committed before the evidence, and how it fared.",
};

export default function PredictionsPage() {
  return (
    <article className="flex flex-col gap-10 pb-16">
      <div className="flex flex-col gap-6">
        <ArticleMeta title="Predictions" />
        <p className="max-w-prose text-lg text-muted-foreground">
          Every analysis was scored against predictions written and committed to git before its
          evidence existed: the comparison, the threshold and what would falsify it. The verdicts
          are appended beneath the predictions and never edited; a reading that turned out wrong is
          corrected in a new section, not rewritten. This is the full ledger.
        </p>
      </div>
      <Scorecard data={getPredictions()} />
    </article>
  );
}
