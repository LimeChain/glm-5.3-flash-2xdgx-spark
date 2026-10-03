# GLM-5.3 Flash on 2x DGX Spark, LimeChain production image (2026-10-03):
#   Mia's TensorFold recipe image (TensorFold 0.6.0 + MiaAI-Lab patches, pinned by digest)
#   + the GLM stream cap lifted from 4 to 8 (engine PARALLEL_MOST / segments MAX_SEGS 4 -> 8, batched window 32 -> 64 rows)
#   + LimeChain 9001 (OpenAI logprobs on the multi-stream engine) and 9002 (stop at <|assistant|>)
ARG BASE=ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:22789f0cb3dc308f0b2ce52a33961b88bd624af1725e91e8aba0a74a671bb969
FROM ${BASE}
COPY lc-9001-9002.patch /tmp/lc.patch
RUN D=$(python -c "import tensorfold,os;print(os.path.dirname(tensorfold.__file__))") \
 && sed -i 's/^PARALLEL_MOST = 4 /PARALLEL_MOST = 8 /; s/^MULTI_WINDOW = 32 /MULTI_WINDOW = 64 /' $D/families/glm5_next/cuda/engine.py \
 && sed -i 's/^MAX_SEGS = 4 /MAX_SEGS = 8 /' $D/families/glm5_next/cuda/segments.py \
 && sed -i 's/^MAX_WINDOW = 32 /MAX_WINDOW = 64 /' $D/families/glm5_next/cuda/multi.py \
 && cd $D && patch -p1 --forward < /tmp/lc.patch && rm /tmp/lc.patch \
 && python -c "import tensorfold.cuda.server, tensorfold.cuda.http, tensorfold.families.glm5_next.cuda.lc_logprobs, tensorfold.families.glm5_next.cuda.engine as e; assert e.PARALLEL_MOST == 8; print('LimeChain GLM image ok')"
