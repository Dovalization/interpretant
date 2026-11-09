import { Hanken_Grotesk, Newsreader } from "next/font/google";

import { AppShell } from "@/components/layout/app-shell";
import { AppProviders } from "@/components/layout/providers";

import type { Metadata } from "next";
import type { ReactNode } from "react";

import "./globals.css";

const hanken = Hanken_Grotesk({
  subsets: ["latin"],
  style: ["normal", "italic"],
  variable: "--font-hanken",
  display: "swap",
});

const newsreader = Newsreader({
  subsets: ["latin"],
  style: ["normal", "italic"],
  variable: "--font-newsreader",
  display: "swap",
});

const description =
  "How academic fields change what they do with a word: per-field, per-decade embeddings and cross-perplexity on research abstracts and canonical books, tested against pre-registered predictions.";

export const metadata: Metadata = {
  title: { default: "interpretant", template: "%s · interpretant" },
  description,
  applicationName: "interpretant",
  authors: [{ name: "Guilherme Doval" }],
  openGraph: { title: "interpretant", description, type: "website" },
  robots: { index: true, follow: true },
};

export const viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#0b0a08",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`dark ${hanken.variable} ${newsreader.variable}`}>
      <body className="antialiased">
        <AppProviders>
          <AppShell>{children}</AppShell>
        </AppProviders>
      </body>
    </html>
  );
}
