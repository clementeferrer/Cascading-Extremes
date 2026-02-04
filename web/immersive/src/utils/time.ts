export function formatTime(hours: number): string {
  if (!isFinite(hours)) return "-";
  const h = Math.max(0, hours);
  const days = Math.floor(h / 24);
  const rem = h % 24;
  if (days > 0) return `${days}d ${rem.toFixed(1)}h`;
  return `${rem.toFixed(2)}h`;
}
