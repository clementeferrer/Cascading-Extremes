const envStamp = import.meta.env.VITE_BUILD_STAMP as string | undefined;
const envId = import.meta.env.VITE_BUILD_ID as string | undefined;

const fallbackStamp = new Date().toISOString();
const fallbackId = Math.random().toString(36).slice(2, 6).toUpperCase();

export const BUILD_STAMP = envStamp
  ? `${envStamp}${envId ? ` #${envId}` : ""}`
  : `${fallbackStamp} #${fallbackId}`;
