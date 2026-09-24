import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PosturePal · Private posture awareness",
  description: "A private, on-device posture and desk-comfort companion with personal calibration and gentle movement cues.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
