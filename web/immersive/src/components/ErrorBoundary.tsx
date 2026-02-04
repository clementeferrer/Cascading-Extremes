import React from "react";

interface State {
  hasError: boolean;
  message?: string;
}

export class ErrorBoundary extends React.Component<React.PropsWithChildren<{}>, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, message: error?.message ?? "Unknown error" };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    // eslint-disable-next-line no-console
    console.error("UI error:", error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            position: "absolute",
            inset: 0,
            zIndex: 9999,
            background: "rgba(11,16,32,0.96)",
            color: "#e2e8f0",
            padding: "24px",
            fontSize: "14px",
          }}
        >
          <div style={{ fontWeight: 600, fontSize: "16px" }}>Visualizer error</div>
          <div style={{ marginTop: 8, color: "#94a3b8" }}>{this.state.message}</div>
          <div style={{ marginTop: 8, color: "#94a3b8" }}>Open the browser console for details.</div>
        </div>
      );
    }
    return this.props.children;
  }
}
