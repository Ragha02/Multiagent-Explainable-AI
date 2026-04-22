"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/api";

const LINKS = [
  { href: "/",          label: "Dashboard" },
  { href: "/predict",   label: "Predict"   },
  { href: "/causal",    label: "Causal"    },
  { href: "/explain",   label: "Explain"   },
  { href: "/advisory",  label: "Advisory"  },
  { href: "/recommend", label: "Recommend" },
  { href: "/compare",   label: "Compare"   },
];

export default function Navbar() {
  const path = usePathname();
  const [pipeline, setPipeline] = useState<string>("idle");
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const id = setInterval(async () => {
      try {
        const r = await fetch(`${API_BASE}/api/health`);
        const d = await r.json();
        setPipeline(d.pipeline);
        if (d.pipeline === "ready") clearInterval(id);
      } catch {}
    }, 2000);
    return () => clearInterval(id);
  }, []);

  const dot = { ready: "#22c55e", running: "#eab308", error: "#ef4444", idle: "#6b7280" }[pipeline] ?? "#6b7280";
  const pulse = pipeline === "running";

  return (
    <nav
      style={{
        position: "fixed",
        top: 0, left: 0, right: 0,
        zIndex: 100,
        height: 60,
        background: scrolled ? "rgba(13,15,18,0.92)" : "rgba(13,15,18,0.6)",
        backdropFilter: "blur(24px)",
        borderBottom: `1px solid var(--border)`,
        transition: "all 0.3s ease",
      }}
    >
      <div style={{
        maxWidth: 1200,
        margin: "0 auto",
        padding: "0 24px",
        height: "100%",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 24,
      }}>
        {/* Logo */}
        <Link href="/" style={{ display: "flex", alignItems: "center", gap: 10, textDecoration: "none" }}>
          <div style={{
            width: 30, height: 30,
            background: "var(--text)", border: "1px solid var(--text)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 18, color: "var(--bg)",
          }} className="font-display">X</div>
          <span className="font-display" style={{ fontSize: 22, color: "var(--text)", letterSpacing: "0.05em" }}>
            MA<span style={{ color: "var(--green)" }}>-XAI</span>
          </span>
        </Link>

        {/* Nav links */}
        <div style={{ display: "flex", alignItems: "center", gap: 2 }}>
          {LINKS.map((l) => {
            const active = path === l.href;
            return (
              <Link key={l.href} href={l.href} style={{
                padding: "6px 14px",
                borderRadius: 8,
                fontWeight: 500,
                textDecoration: "none",
                color: active ? "var(--green)" : "var(--text-dim)",
                background: active ? "var(--surface)" : "transparent",
                border: `1px solid ${active ? "var(--border-bright)" : "transparent"}`,
                textTransform: "uppercase", letterSpacing: "0.1em", fontSize: 11,
                transition: "all 0.2s ease",
              }}>
                {l.label}
              </Link>
            );
          })}
        </div>

        {/* Status */}
        <div style={{
          display: "flex", alignItems: "center", gap: 7,
          padding: "5px 12px",
          borderRadius: 99,
          border: "1px solid rgba(255,255,255,0.08)",
          background: "rgba(255,255,255,0.03)",
          fontSize: 12,
        }}>
          <span style={{
            width: 7, height: 7, borderRadius: "50%",
            background: dot,
            boxShadow: pulse ? `0 0 0 0 ${dot}` : "none",
            animation: pulse ? "pulse-ring 1.2s ease-out infinite" : "none",
          }} />
          <span style={{ color: "#6b7280" }}>Pipeline</span>
          <span style={{ color: "#eef2ff", fontWeight: 600, textTransform: "capitalize" }}>{pipeline}</span>
        </div>
      </div>

      <style>{`
        @keyframes pulse-ring {
          0%   { box-shadow: 0 0 0 0 ${dot}99; }
          70%  { box-shadow: 0 0 0 6px ${dot}00; }
          100% { box-shadow: 0 0 0 0 ${dot}00; }
        }
      `}</style>
    </nav>
  );
}
