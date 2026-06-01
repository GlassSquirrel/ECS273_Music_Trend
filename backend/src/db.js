import { MongoClient } from "mongodb";
import { config } from "./config.js";

let clientPromise;

function createClient() {
  const client = new MongoClient(config.mongodbUri);
  return client.connect();
}

export async function getDb() {
  if (!clientPromise) clientPromise = createClient();
  const client = await clientPromise;
  return client.db(config.mongodbDb);
}
