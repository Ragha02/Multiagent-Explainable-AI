"use client";
import { useEffect, useState } from "react";
import { API_BASE, PipelineStatus } from "@/lib/api";

const AGENTS = [
  { id: 1, name: "Data Agent",        icon: "🗄",  desc: "Generate · Clean · Engineer" },
  { id: 2, name: "Prediction Agent",  icon: "🤖", desc: "RF + XGBoost Ensemble · SHAP" },
  { id: 3, name: "Causal Agent",      icon: "⟶",  desc: "DAG · ATE · Counterfactuals" },
  { id: 4, name: "Explanation Agent", icon: "✦",  desc: "Global · Local · Contrastive" },
  { id: 5, name: "Advisory Agent",    icon: "▲",  desc: "Rules · ML Insight · Report" },
];

// Progress thresholds at which each agent becomes "active"
const AGENT_THRESHOLDS = [10, 30, 55, 75, 92];

export default function AgentPipeline() {
  const [status, setStatus] = useState<PipelineStatus>({
    status: "idle",
    progress: 0,
    current_step: "Awaiting start…",
  });

  useEffect(() => {
    // Poll status every second
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/pipeline/status`);
        const data: PipelineStatus = await res.json();
        setStatus(data);
        if (data.status === "ready" || data.status === "error") clearInterval(interval);
      } catch { /* backend not up yet */ }
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const getAgentState = (idx: number): "done" | "active" | "waiting" => {
    const threshold = AGENT_THRESHOLDS[idx];
    if (status.progress >= threshold + 15) return "done";
    if (status.progress >= threshold) return "active";
    return "waiting";
  };

  return (
    <div className="w-full">
      {/* Status header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <p className="text-sm font-medium" style={{ color: "#22c55e" }}>
            {status.current_step}
          </p>
          <p className="text-xs mt-0.5" style={{ color: "#4b5563" }}>
            {status.status === "ready" ? "All agents operational" : `${status.progress}% complete`}
          </p>
        </div>
        <span
          className="text-xs px-2.5 py-1 rounded-full font-medium"
          style={{
            background: status.status === "ready"
              ? "rgba(34,197,94,0.15)"
              : status.status === "running"
              ? "rgba(234,179,8,0.15)"
              : "rgba(255,255,255,0.05)",
            color: status.status === "ready" ? "#22c55e"
              : status.status === "running" ? "#eab308"
              : "#94a3b8",
            border: `1px solid ${status.status === "ready" ? "rgba(34,197,94,0.3)"
              : status.status === "running" ? "rgba(234,179,8,0.3)"
              : "rgba(255,255,255,0.1)"}`,
          }}
        >
          {status.status.toUpperCase()}
        </span>
      </div>

      {/* Progress bar */}
      <div className="progress-bar mb-8">
        <div className="progress-fill" style={{ width: `${status.progress}%` }} />
      </div>

      {/* Agent flow */}
      <div className="flex items-stretch gap-2">
        {AGENTS.map((agent, idx) => {
          const state = getAgentState(idx);
          return (
            <div key={agent.id} className="flex items-stretch gap-2 flex-1">
              <div
                className="flex-1 p-4 rounded-xl transition-all duration-500"
                style={{
                  background: state === "done"
                    ? "rgba(34, 197, 94, 0.08)"
                    : state === "active"
                    ? "rgba(234, 179, 8, 0.08)"
                    : "rgba(255,255,255,0.02)",
                  border: `1px solid ${
                    state === "done"
                      ? "rgba(34,197,94,0.3)"
                      : state === "active"
                      ? "rgba(234,179,8,0.4)"
                      : "rgba(255,255,255,0.06)"
                  }`,
                  boxShadow: state === "active"
                    ? "0 0 20px rgba(234,179,8,0.1)"
                    : state === "done"
                    ? "0 0 12px rgba(34,197,94,0.07)"
                    : "none",
                }}
              >
                <div className="flex items-center gap-2 mb-2">
                  <span
                    className="text-lg"
                    style={{
                      filter: state === "active" ? "drop-shadow(0 0 8px rgba(234,179,8,0.8))" : "none",
                    }}
                  >
                    {state === "done" ? "✓" : agent.icon}
                  </span>
                  <span
                    className="text-xs font-bold"
                    style={{
                      color: state === "done" ? "#22c55e" : state === "active" ? "#eab308" : "#4b5563",
                    }}
                  >
                    A{agent.id}
                  </span>
                </div>
                <p
                  className="text-xs font-semibold leading-tight mb-1"
                  style={{ color: state === "waiting" ? "#4b5563" : "#f0f6ff" }}
                >
                  {agent.name}
                </p>
                <p className="text-xs leading-tight" style={{ color: "#374151" }}>
                  {agent.desc}
                </p>
              </div>

              {/* Connector arrow */}
              {idx < AGENTS.length - 1 && (
                <div className="flex items-center">
                  <span
                    style={{
                      color: getAgentState(idx) === "done" ? "#22c55e" : "#1f2937",
                      fontSize: "16px",
                      transition: "color 0.5s",
                    }}
                  >
                    →
                  </span>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Error */}
      {status.error && (
        <div
          className="mt-4 p-3 rounded-lg text-xs"
          style={{
            background: "rgba(239,68,68,0.1)",
            border: "1px solid rgba(239,68,68,0.3)",
            color: "#fca5a5",
            fontFamily: "monospace",
          }}
        >
          {status.error.slice(0, 300)}
        </div>
      )}
    </div>
  );
}
