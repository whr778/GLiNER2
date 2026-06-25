## VERY IMPORTANT
- Be simple. Approach tasks in a simple, incremental way.
- Work incrementally ALWAYS. Small simple steps.  Validate and check each increment before moving on.
- Use LATEST apis as of NOW.

## MANDATORY CODE STYLE
- Do not overengineer.  Do not program defensively.  Use exception managers only when necessary.
- Identify root causes before fixing issues.  Prove with evidence, then fix.
- Work incrementally with small steps.  Validate each increment.
- Use latest library APIs.
- Use `uv` as Python package manager.  Always `uv run xxx` never `python3 xxx`, always `uv add xxx` never `pip install xxx` 
- Favor clear, concise docstring comments.  Be sparing with comments outside docstrings.
- Favor short modules, short methods and functions.  Name things clearly.
- Never use emojis in code or in print statements or logging.
- Keep README.md concise
 
## Important -- dubugging and fixing
- When troubleshooting problems, ALWAYS identify root cause BEFORE fixing
- Reproduce consistently.
- PROVE THE PROBLEM FIRST - don't guess.
- Try one test at a time.  Be methodical.
- Don't jump to conclusions.  Don't apply workarounds.

## Data converters (tools/data/)
- Emit normalized, UTF-8 JSONL via `_split.dumps_record` (the `SplitWriter` write path): NFKC plus stray line-separator stripping (NEL U+0085, U+2028, U+2029 -> space, so records never fragment across lines).
- New converters must route every record write through `dumps_record`, never raw `json.dumps`, and open all files with `encoding="utf-8"`.
- When using json.dump or json.dumps ensure_ascii should always be set to False unless I have directed otherwise.