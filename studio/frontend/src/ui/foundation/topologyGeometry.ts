export interface Point2D {
  x: number;
  y: number;
}

export interface NodeBox {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * Generates a smooth cubic Bezier SVG path connecting two topology node anchors.
 */
export function buildCubicEdgePath(
  source: Point2D,
  target: Point2D,
  direction: "horizontal" | "vertical" = "horizontal"
): string {
  if (direction === "horizontal") {
    const dx = Math.max(36, Math.abs(target.x - source.x) * 0.48);
    const c1x = source.x + (target.x >= source.x ? dx : -dx);
    const c2x = target.x - (target.x >= source.x ? dx : -dx);
    return `M ${source.x} ${source.y} C ${c1x} ${source.y}, ${c2x} ${target.y}, ${target.x} ${target.y}`;
  }

  const dy = Math.max(32, Math.abs(target.y - source.y) * 0.48);
  const c1y = source.y + (target.y >= source.y ? dy : -dy);
  const c2y = target.y - (target.y >= source.y ? dy : -dy);
  return `M ${source.x} ${source.y} C ${source.x} ${c1y}, ${target.x} ${c2y}, ${target.x} ${target.y}`;
}

export function getNodeAnchor(
  node: NodeBox,
  side: "left" | "right" | "top" | "bottom"
): Point2D {
  switch (side) {
    case "left":
      return { x: node.x, y: node.y + node.height / 2 };
    case "right":
      return { x: node.x + node.width, y: node.y + node.height / 2 };
    case "top":
      return { x: node.x + node.width / 2, y: node.y };
    case "bottom":
      return { x: node.x + node.width / 2, y: node.y + node.height };
  }
}
