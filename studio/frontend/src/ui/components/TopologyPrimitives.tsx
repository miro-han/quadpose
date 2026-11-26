import React from "react";
import { MaterialStyle } from "../foundation/liquidProfiles";
import {
  buildCubicEdgePath,
  getNodeAnchor,
  NodeBox
} from "../foundation/topologyGeometry";
import { MaterialSurface } from "./MaterialSurface";

export interface TopologyNodeData extends NodeBox {
  kind: string;
  title: string;
  meta?: string;
  status?: "neutral" | "success" | "warning" | "danger";
}

export interface TopologyEdgeData {
  id: string;
  from: string;
  to: string;
  active?: boolean;
}

export interface TopologyCanvasProps {
  nodes: TopologyNodeData[];
  edges: TopologyEdgeData[];
  selectedNodeId?: string;
  onSelectNode?: (id: string) => void;
  mode?: MaterialStyle;
  height?: number;
  className?: string;
}

/**
 * Interactive node-and-edge topology graph reproducing SIDE's `GraphDiagram` and `RevisionTopology` surfaces.
 */
export function TopologyCanvas({
  nodes,
  edges,
  selectedNodeId,
  onSelectNode,
  mode,
  height = 290,
  className = ""
}: TopologyCanvasProps) {
  const nodeMap = new Map(nodes.map((n) => [n.id, n]));

  return (
    <div
      className={["miro-topology-surface", "topology-surface", className]
        .filter(Boolean)
        .join(" ")}
      style={{ height: `${height}px` }}
    >
      <svg
        width="100%"
        height="100%"
        style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
        aria-hidden="true"
      >
        {edges.map((edge) => {
          const sourceNode = nodeMap.get(edge.from);
          const targetNode = nodeMap.get(edge.to);
          if (!sourceNode || !targetNode) return null;
          const start = getNodeAnchor(sourceNode, "right");
          const end = getNodeAnchor(targetNode, "left");
          const d = buildCubicEdgePath(start, end, "horizontal");
          return (
            <path
              key={edge.id}
              d={d}
              fill="none"
              stroke={
                edge.active
                  ? "var(--miro-accent, var(--accent))"
                  : "var(--miro-line, var(--line))"
              }
              strokeWidth={edge.active ? 2 : 1.5}
              strokeDasharray={edge.active ? undefined : "3 3"}
            />
          );
        })}
      </svg>

      {nodes.map((node) => {
        const selected = node.id === selectedNodeId;
        return (
          <MaterialSurface
            key={node.id}
            role="button"
            tabIndex={0}
            aria-pressed={selected}
            mode={mode}
            profile="compact"
            radius={12}
            data-selected={selected ? "true" : undefined}
            className="miro-topology-node topology-node"
            style={{
              left: `${node.x}px`,
              top: `${node.y}px`,
              width: `${node.width}px`,
              minHeight: `${node.height}px`
            }}
            onClick={() => onSelectNode?.(node.id)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onSelectNode?.(node.id);
              }
            }}
          >
            <span className="miro-topology-node-kind topology-node-kind">
              {node.kind}
            </span>
            <span className="miro-topology-node-title topology-node-title">
              {node.title}
            </span>
            {node.meta ? (
              <span
                style={{
                  fontSize: "var(--miro-type-meta, 0.6875rem)",
                  color: "var(--miro-muted, var(--muted))"
                }}
              >
                {node.meta}
              </span>
            ) : null}
          </MaterialSurface>
        );
      })}
    </div>
  );
}
