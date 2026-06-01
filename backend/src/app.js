import cors from "cors";
import express from "express";
import { getDb } from "./db.js";

const CACHE_KEYS = [
  "clusterData",
  "themeRiverData",
  "audioFeatureData",
  "wordCloudData",
];

export function createApp() {
  const app = express();

  app.use(cors());
  app.use(express.json());

  app.get("/api/health", async (_req, res) => {
    const db = await getDb();
    await db.command({ ping: 1 });
    res.json({ ok: true });
  });

  app.get("/api/bootstrap", async (_req, res, next) => {
    try {
      const db = await getDb();
      const [cacheDocs, totalSongs, clusteredSongs] = await Promise.all([
        db.collection("visualization_cache")
          .find({ _id: { $in: CACHE_KEYS } })
          .toArray(),
        db.collection("songs").countDocuments(),
        db.collection("songs").countDocuments({ cluster: { $ne: null } }),
      ]);

      const cacheMap = Object.fromEntries(
        cacheDocs.map((doc) => [doc._id, doc.data]),
      );

      for (const key of CACHE_KEYS) {
        if (!(key in cacheMap)) {
          return res.status(503).json({
            error: `Missing visualization cache '${key}'. Run the import script first.`,
          });
        }
      }

      res.json({
        clusterData: cacheMap.clusterData,
        themeRiverData: cacheMap.themeRiverData,
        audioFeatureData: cacheMap.audioFeatureData,
        wordCloudData: cacheMap.wordCloudData,
        stats: {
          totalSongs,
          clusteredSongs,
          clusters: Object.keys(cacheMap.audioFeatureData.features || {}).length,
        },
      });
    } catch (error) {
      next(error);
    }
  });

  app.use((error, _req, res, _next) => {
    console.error(error);
    res.status(500).json({
      error: error.message || "Internal server error",
    });
  });

  return app;
}
