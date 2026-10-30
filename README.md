# Duty manager decision recall: a small RAG build and evaluation

A retrieval-augmented generation (RAG) system that answers a duty manager's question ("what did we decide last time?") from past operational records, with citations. Built in Python to test two things properly: whether retrieval finds the right records, and which model gives the most trustworthy answers.

All 13 records are mock documents I wrote. No real organisation's data is used.

## Background

I first built this idea as the final-stage task for the AI & Innovation Lead role at Blenheim Palace: a tool that keeps the reasoning behind past operational decisions from being lost when staff leave. The records in this repo are invented for this exercise. They are not based on Blenheim data, and the incidents in them did not happen.

## How it works

1. **Load** 13 short records (32 to 117 words each): duty logs, emails, handover notes and guest recovery notes. Each record is one chunk, because splitting a 64-word log would separate a decision from its reasoning.
2. **Embed** each record with a sentence-transformers model and store it in a Chroma vector database.
3. **Retrieve** the 5 closest records to the question.
4. **Rerank** those 5 with a cross-encoder, which reads the question and each record together.
5. **Generate** an answer with an LLM under a system prompt: answer only from the records, cite every fact, say so when there is no precedent, and flag records that disagree.
6. **Score** each answer against a hand-written answer key, using a separate LLM as the marker.

## Test set

Four questions: three real question types (wind closure, garden closure with tickets sold, contractor access blocked by an event) and one with no precedent in the records (bats in the chapel roof), to test whether the model admits it doesn't know. The corpus includes filler records that should never be retrieved, and two records that disagree about what happened on the same night.

## Results

### 1. Retrieval

| Setup | Recall@3 | Top result correct |
|---|---|---|
| all-MiniLM-L6-v2 | 7/8 (the maximum at k=3) | 2/3 |
| all-mpnet-base-v2 (5x larger) | 7/8 | 2/3 |
| MiniLM + reranker | 7/8 | 3/3 |
| mpnet + reranker | 7/8 | 3/3 |

The right records were always on the shortlist; the problem was order. On the contractor question, a routine log won because it shared the words "gate", "engineer" and "booked". The question said "event"; the records said "filming". A larger embedding model didn't fix it. A reranker did.

**Chosen setup:** MiniLM + reranker. Same accuracy as the larger model at a fifth of the size.

### 2. A fix that caused a new failure

With the top 3 records, the reranker swapped out a duty log that contradicts the follow-up note for the same night. The model then reported one version as fact. Passing all 5 shortlisted records and adding a rule to flag disagreements fixed it, at the cost of about 40% more input tokens per question.

### 3. Model comparison

Claude Haiku (`claude-haiku-5-5`) against Gemini Flash, given identical retrieved records and the identical system prompt. I planned to test `gemini-3.8-flash`, but it returned "unavailable" errors across repeated retries on the free tier, so I used `gemini-3.5-flash`.

| | Claude Haiku | Gemini 3.5 Flash |
|---|---|---|
| Required facts included | 9/16 (56%) | 11/16 (69%) |
| Errors made | 8 | 1 |
| Admitted "no precedent" | Yes | Yes |
| Availability | Answered every request | Needed retries in both full runs |

Patterns that held across repeated runs:

- **Cited and still wrong.** Haiku said "the contractor was charged the full day rate" in 4 of 4 runs. The record means the contractor charged the estate. The citation was correct; the reading wasn't. Gemini got this right every time.
- **Misread timing.** Haiku said the trail "closed at 15:00" in 6 of 6 runs. The record says the decision was taken at 15:00.
- **Invented reasons.** Gemini said an event went ahead "because" wind sensors stayed low, in 2 of 2 runs. The log records the decision, not the reason. This is the more dangerous kind of error, because a reader may act on it.
- **Run-to-run variance.** One false claim from Haiku appeared in 2 of 4 runs and not the other two, so a single run isn't enough to judge a model.

### 4. Scoring the scorer

The answer key lists the facts each answer must include and the errors it must not make. Claude Sonnet marked each answer against it. Before trusting the marks, I compared them with my own marking of the same answers.

- First version: the judge agreed with me on 3 of 8 answers. The causes were in my answer key, not random: combined items scored all-or-nothing, one rule was worded too narrowly to catch the error I meant, and the judge didn't know how many records each model had been given. The judge also applied the citation rule more strictly than I had, and on checking, it was right.
- Revised key: 7 of 8 agreed. The remaining disagreement was the judge misreading one key item, so automated marking still needs a human spot-check.

## Recommendation

For this use case, a wrong fact is worse than a missing one, because a duty manager may act on it. On accuracy, Gemini Flash is the better generator. It is not deployable as tested: a pilot would need a paid tier with a reliability guarantee, and that cost belongs in the decision. Whichever model is used, the conflict rule and a quick check of the cited source are needed before acting on an answer.

## Limits

- 4 questions and 13 short records. This shows a direction, not a verdict.
- One scored run per model. Variance between runs was visible, so the next step is several runs per model and more questions per type.
- The judge (Claude Sonnet) is from the same model family as one contestant and may favour it.
- Token counts aren't comparable between providers, because each counts tokens differently. Cost should be compared in money per answer.

## Responsible AI and cost notes

- Mock data only. Gemini's free tier may use inputs to improve Google's products, which would rule it out for real records.
- API keys are stored in Colab Secrets, never in the notebook.
- A spend limit was set on the Anthropic account before testing. The whole evaluation cost a few pence.
- Model versions are pinned rather than using "latest" aliases, so results can be reproduced. The Gemini fallback is documented above.
- Server errors are retried with increasing waits; errors on our side are not retried, because retrying a bad request never fixes it.

## Running it

1. Open `duty_manager_rag.ipynb` in Colab with the badge at the top of the notebook.
2. Run cell 2, then upload the 13 `.txt` records into the `docs` folder it creates.
3. Add `ANTHROPIC_API_KEY` and `GEMINI_API_KEY` in Colab Secrets with notebook access switched on.
4. Runtime > Run all.

## Tools

Python, sentence-transformers, Chroma, a cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`), the Anthropic and Google GenAI SDKs, pandas, Google Colab.

