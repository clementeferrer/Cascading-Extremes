/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0f172a",
        slate: "#475569",
        mist: "#eef3ff",
        ivory: "#f7f4ee",
        accent: "#14b8a6",
        night: "#0b1020",
        glass: "rgba(255,255,255,0.08)",
        glow: "#5eead4",
      },
      fontFamily: {
        sans: ["Space Grotesk", "system-ui", "sans-serif"],
        serif: ["Newsreader", "serif"],
      },
    },
  },
  plugins: [],
};
