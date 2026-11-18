import "dotenv/config";
import express from "express";
import type { Request, Response } from "express";
import path from "path";
import { fileURLToPath } from "url";
import { GoogleGenAI, ThinkingLevel } from "@google/genai";
import type { GenerateContentResponse } from "@google/genai";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT || 3000);
const MODEL_NAME = "gemini-4-hydrogen";

interface ChatTurn {
  role: "user" | "model";
  text: string;
}

interface TripPreferences {
  duration?: string;
  pace?: string;
}

function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey === "MY_GEMINI_API_KEY") {
    throw new Error(
      "GEMINI_API_KEY is not configured. Please verify your API key in Settings > Secrets."
    );
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

function buildSystemInstruction(preferences?: TripPreferences): string {
  const prefNotes: string[] = [];
  if (preferences?.duration) {
    prefNotes.push(`Preferred trip length: ${preferences.duration}`);
  }
  if (preferences?.pace) {
    prefNotes.push(`Preferred travel pace: ${preferences.pace}`);
  }

  return [
    "You are cev, a knowledgeable, warm, and practical Shanghai trip-planning concierge embedded in the Shanghai Destination Guide.",
    "Your purpose is to help travelers plan unforgettable trips across Shanghai: day-by-day itineraries, neighborhood pairings, Shanghai Metro routes and station exits, best visiting times, food and tea/café recommendations, seasonal packing, airport/Maglev transfers, and local payment tips (Alipay/WeChat Pay).",
    "You have deep familiarity with the guide's six signature destinations:",
    "1. Lujiazui & Shanghai Tower (Pudong Skyline — Metro Lines 2 & 14, Lujiazui Station Exit 6; best 17:30–20:30 CST for sunset into skyline lights; Binjiang Avenue promenade, circular skywalk, 118th-floor deck).",
    "2. The Bund / Waitan (Huangpu Riverfront — Metro Lines 2 & 10, East Nanjing Road Station Exit 2; best 06:00–07:30 CST or 19:00–22:00 CST; two-tier promenade, Waibaidu Bridge, Huangpu cruises).",
    "3. Yu Garden / Yuyuan (Old City Heritage — Metro Lines 10 & 14, Yuyuan Garden Station Exits 1/7; best 08:30–10:30 CST; Nine-Turn Bridge, Huxinting Teahouse, Yuyuan Bazaar xiaolongbao).",
    "4. Nanjing Road Pedestrian (Central Avenue — Metro Lines 1, 2 & 8, People's Square to East Nanjing Road; best 18:00–22:30 CST; neon evening atmosphere, classic pastry shops, car-free walk to The Bund).",
    "5. 1933 Old Millfun (Hongkou Creative Hub — Metro Lines 4 & 10, Hailun Road Station Exit 2; best 13:00–16:00 CST; multi-level walkways, indie coffee roasters, creative studios).",
    "6. Zhujiajiao Water Town (Qingpu Canal Escape — Metro Line 17, Zhujiajiao Station Exit 1; best 15:30–18:00 CST; Fangsheng Bridge, wooden gondola rides, waterside teahouses, Kezhi Garden).",
    "Also incorporate other great Shanghai areas when relevant (Former French Concession / Wukang Road / Anfu Road, Jing'an Temple, Xintiandi, West Bund, Tianzifang) so itineraries feel authentic and geographically logical.",
    "Formatting rules: Keep answers concise, scannable, and practical. Use short sections or bullet points with realistic timing and metro connections. Never invent fake prices or pretend real-time live ticket availability; give helpful typical ranges or advise checking official mini-programs when appropriate.",
    prefNotes.length > 0 ? `Active traveler preferences — ${prefNotes.join("; ")}.` : "",
  ]
    .filter(Boolean)
    .join("\n\n");
}

function formatApiError(err: unknown): { status: number; message: string } {
  const raw = err instanceof Error ? err.message : String(err || "Unknown server error");
  const lower = raw.toLowerCase();

  if (lower.includes("api_key_invalid") || lower.includes("permission_denied") || lower.includes("not configured")) {
    return {
      status: 403,
      message: "API key is missing or invalid. Please check your Gemini API key in Settings > Secrets.",
    };
  }
  if (lower.includes("resource_exhausted") || lower.includes("429") || lower.includes("quota")) {
    return {
      status: 429,
      message: "Request quota reached right now. Please wait a moment and try again.",
    };
  }
  if (lower.includes("not_found") || lower.includes("404")) {
    return {
      status: 404,
      message: "The configured Gemini model could not be found. Please check model availability.",
    };
  }
  return {
    status: 500,
    message: raw || "Unable to reach cev right now. Please try again.",
  };
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: "1mb" }));

  app.get("/api/health", (_req: Request, res: Response) => {
    res.json({ ok: true, assistant: "cev", model: MODEL_NAME });
  });

  app.post("/api/chat", async (req: Request, res: Response) => {
    const body = req.body as {
      messages?: ChatTurn[];
      preferences?: TripPreferences;
    };

    const rawMessages = Array.isArray(body?.messages) ? body.messages : [];
    const validMessages = rawMessages
      .filter(
        (m): m is ChatTurn =>
          Boolean(m) &&
          (m.role === "user" || m.role === "model") &&
          typeof m.text === "string" &&
          m.text.trim().length > 0
      )
      .slice(-24);

    if (validMessages.length === 0 || validMessages[validMessages.length - 1].role !== "user") {
      res.status(400).json({ error: "Please enter a message for cev." });
      return;
    }

    let ai: GoogleGenAI;
    try {
      ai = getGeminiClient();
    } catch (err) {
      const formatted = formatApiError(err);
      res.status(formatted.status).json({ error: formatted.message });
      return;
    }

    const priorHistory = validMessages.slice(0, -1).map((m) => ({
      role: m.role,
      parts: [{ text: m.text }],
    }));
    const latestMessage = validMessages[validMessages.length - 1].text;

    res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
    res.setHeader("Cache-Control", "no-cache, no-transform");
    res.setHeader("Connection", "keep-alive");
    res.flushHeaders?.();
    res.write(": connected\n\n");

    let closed = false;
    res.on("close", () => {
      if (!res.writableEnded) {
        closed = true;
      }
    });

    try {
      const chat = ai.chats.create({
        model: MODEL_NAME,
        history: priorHistory,
        config: {
          systemInstruction: buildSystemInstruction(body?.preferences),
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
        },
      });

      const stream = await chat.sendMessageStream({ message: latestMessage });

      for await (const chunk of stream) {
        if (closed) break;
        const c = chunk as GenerateContentResponse;
        const text = c.text;
        if (text) {
          res.write(`data: ${JSON.stringify({ type: "delta", text })}\n\n`);
        }
      }

      if (!closed && !res.writableEnded) {
        res.write(`data: ${JSON.stringify({ type: "done" })}\n\n`);
        res.end();
      }
    } catch (err) {
      const formatted = formatApiError(err);
      console.error("cev /api/chat error:", formatted.message);
      if (!closed && !res.writableEnded) {
        res.write(`data: ${JSON.stringify({ type: "error", message: formatted.message })}\n\n`);
        res.end();
      }
    }
  });

  if (process.env.NODE_ENV !== "production") {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, "dist");
    app.use(express.static(distDir));
    app.get("*", (_req: Request, res: Response) => {
      res.sendFile(path.join(distDir, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Shanghai Destination Guide server listening on http://localhost:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
