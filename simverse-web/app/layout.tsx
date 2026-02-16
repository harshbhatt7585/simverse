import type { Metadata } from "next";
import { Manrope, Orbitron } from "next/font/google";
import "./globals.css";

const manrope = Manrope({
  subsets: ["latin"],
  variable: "--font-ui",
});

const orbitron = Orbitron({
  subsets: ["latin"],
  variable: "--font-display",
});

export const metadata: Metadata = {
  title: "Simverse",
  description: "Simulation-first UI for replay and live environment visualization.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${manrope.variable} ${orbitron.variable}`}>{children}</body>
    </html>
  );
}
