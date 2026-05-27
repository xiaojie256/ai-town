import { v } from 'convex/values';
import {
  ActionCtx,
  internalMutation,
  internalQuery,
  QueryCtx,
  MutationCtx,
} from '../_generated/server.js';
import { internal } from '../_generated/api.js';
import { Id } from '../_generated/dataModel.js';
import { fetchEmbeddingBatch } from '../util/llm.js';

const selfInternal = internal.agent.embeddingsCache;

export async function fetch(ctx: ActionCtx, text: string) {
  const result = await fetchBatch(ctx, [text]);
  return result.embeddings[0];
}

export async function fetchBatch(ctx: ActionCtx, texts: string[]) {
  const start = Date.now();

  const textHashes = await Promise.all(texts.map((text) => hashText(text)));
  const results = new Array<number[]>(texts.length);
  const cacheResults = await ctx.runQuery(selfInternal.getEmbeddingsByText, {
    textHashes,
  });
  for (const { index, embedding } of cacheResults) {
    results[index] = embedding;
  }
  const toWrite = [];
  if (cacheResults.length < texts.length) {
    const missingIndexes = [...results.keys()].filter((i) => !results[i]);
    const missingTexts = missingIndexes.map((i) => texts[i]);
    const response = await fetchEmbeddingBatch(missingTexts);
    if (response.embeddings.length !== missingIndexes.length) {
      throw new Error(
        `Expected ${missingIndexes.length} embeddings, got ${response.embeddings.length}`,
      );
    }
    for (let i = 0; i < missingIndexes.length; i++) {
      const resultIndex = missingIndexes[i];
      toWrite.push({
        textHash: textHashes[resultIndex],
        embedding: response.embeddings[i],
      });
      results[resultIndex] = response.embeddings[i];
    }
  }
  if (toWrite.length > 0) {
    await ctx.runMutation(selfInternal.writeEmbeddings, { embeddings: toWrite });
  }
  return {
    embeddings: results,
    hits: cacheResults.length,
    ms: Date.now() - start,
  };
}

async function hashText(text: string): Promise<ArrayBuffer> {
  const textEncoder = new TextEncoder();
  const buf = textEncoder.encode(text);
  if (typeof crypto === 'undefined') {
    const f = () => 'node:crypto';
    const nodeCrypto = (await import(f())) as typeof import('crypto');
    const hash = nodeCrypto.createHash('sha256');
    hash.update(buf);
    const digest = hash.digest();
    return new Uint8Array(digest).slice().buffer;
  } else {
    return await crypto.subtle.digest('SHA-256', buf);
  }
}

export const getEmbeddingsByText = internalQuery({
  args: { textHashes: v.array(v.bytes()) },
  handler: async (ctx: QueryCtx, args: { textHashes: ArrayBuffer[] }) => {
    const results = await Promise.all(
      args.textHashes.map((textHash) =>
        ctx.db
          .query('embeddingsCache')
          .withIndex('text', (q) => q.eq('textHash', textHash))
          .first(),
      ),
    );
    const out = [];
    for (let i = 0; i < results.length; i++) {
      const result = results[i];
      if (result) {
        out.push({
          index: i,
          embeddingId: result._id,
          embedding: result.embedding,
        });
      }
    }
    return out;
  },
});

export const writeEmbeddings = internalMutation({
  args: {
    embeddings: v.array(
      v.object({
        textHash: v.bytes(),
        embedding: v.array(v.float64()),
      }),
    ),
  },
  handler: async (
    ctx: MutationCtx,
    args: { embeddings: { textHash: ArrayBuffer; embedding: number[] }[] },
  ) => {
    const ids = [];
    for (const embedding of args.embeddings) {
      ids.push(await ctx.db.insert('embeddingsCache', embedding));
    }
    return ids;
  },
});
