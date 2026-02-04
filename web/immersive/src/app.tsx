import LandingPage from "./pages/LandingPage";
import ViewerPage from "./pages/ViewerPage";

export default function App() {
  const path = window.location.pathname;
  const isViewer = path.startsWith("/viewer");
  return isViewer ? <ViewerPage /> : <LandingPage />;
}
