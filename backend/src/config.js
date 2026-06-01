import dotenv from "dotenv";

dotenv.config();

export const config = {
  port: Number(process.env.PORT || 8000),
  mongodbUri: process.env.MONGODB_URI || "mongodb://127.0.0.1:27017",
  mongodbDb: process.env.MONGODB_DB || "music_trend",
};
