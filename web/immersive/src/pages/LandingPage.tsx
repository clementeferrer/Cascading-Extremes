import { TopNav } from "../components/landing/TopNav";
import { Hero } from "../components/landing/Hero";
import { ScrollStory } from "../components/landing/ScrollStory";
import { Counters } from "../components/landing/Counters";
import { AccordionCards } from "../components/landing/AccordionCards";
import { Stepper } from "../components/landing/Stepper";
import { FooterCTA } from "../components/landing/FooterCTA";

export default function LandingPage() {
  return (
    <div className="relative min-h-screen bg-night text-slate-100">
      <div className="absolute inset-0 -z-10">
        <div className="h-full w-full bg-[radial-gradient(circle_at_top,_rgba(20,184,166,0.15),_transparent_55%)]" />
      </div>
      <div className="mx-auto w-full max-w-6xl px-6">
        <TopNav
          links={[
            { label: "About", href: "#about" },
            { label: "Process", href: "#process" },
            { label: "Contact", href: "#contact" },
          ]}
          ctaLabel="Open Viewer"
          ctaHref="/viewer"
        />
        <Hero
          headline="Generative Cascades of Multivariate Extremes"
          subheadline="A live, EVT-grounded generative engine that learns extreme-event geometry and synthesizes cascades with causal attention."
          ctaLabel="Launch Immersive Viewer"
          ctaHref="/viewer"
        />
        <div id="about">
          <ScrollStory
            stories={[
              {
                title: "Extremes are not points — they are geometry.",
                body: "We treat extremes as a marked point process with structure, not isolated outliers. The angular geometry on the unit sphere encodes direction; the radial exceedance encodes severity.",
              },
              {
                title: "Memory gives rise to cascades.",
                body: "A causal Transformer maps the history of extremes into a latent state that modulates direction, magnitude, and inter‑event timing.",
              },
              {
                title: "Generation is an experiment you can run live.",
                body: "Press Play and a new cascade is synthesized in real time—conditioning on a seed history and evolving with Hawkes-style feedback.",
              },
              {
                title: "Geometry, timing, and causality—one narrative.",
                body: "Our viewer unifies EVT geometry with causal intensity, letting you observe how extremes propagate in multivariate systems.",
              },
              {
                title: "This is a live paper, not a static figure.",
                body: "Every view is reproducible, every run is exported, and the visual narrative evolves with the model.",
              },
            ]}
          />
        </div>
        <div id="process">
          <Counters
            stats={[
              { label: "Hourly · 2Y", value: 17520, precision: 0 },
              { label: "Assets", value: 3, precision: 0 },
            ]}
          />
          <AccordionCards
            items={[
              {
                title: "EVT Geometry",
                body: "Direction-dependent thresholds define the extreme region on the sphere with principled gauge functions.",
              },
              {
                title: "Causal Attention",
                body: "A Transformer encodes the exceedance history to parameterize the next-event distribution.",
              },
              {
                title: "Hawkes Feedback",
                body: "Endogenous intensity captures self-excitation and cascade amplification over time.",
              },
            ]}
          />
          <Stepper
            steps={[
              { title: "Standardize", body: "Map returns to Laplace margins with GARCH + PIT." },
              { title: "Extract", body: "Define geometric extremes via direction-dependent thresholds." },
              { title: "Encode", body: "Tokenize exceedances for the causal Transformer." },
              { title: "Generate", body: "Sample direction, magnitude, and timing sequentially." },
              { title: "Export", body: "Store events + metrics for reproducible runs." },
              { title: "Visualize", body: "Observe cascades as geometry in the unit cube." },
            ]}
          />
        </div>
        <div id="contact">
          <FooterCTA
            text="Ready to explore the cascade?"
            ctaLabel="Launch Immersive Viewer"
            ctaHref="/viewer"
          />
        </div>
      </div>
    </div>
  );
}
