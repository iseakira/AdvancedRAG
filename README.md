# AdvancedRAG

OPENAPI_KEY https://platform.openai.com/settings/organization/billing/overview
LANGSmith https://smith.langchain.com/o/c9afdde1-66b1-4c1d-b945-3e58e56f3900/settings/apikeys

## LirQA

- 構造確認用bashコード①
  for file in littraceqa_release.jsonl littraceqa_release_report.json paper_metadata.jsonl sample_submission.jsonl validation.jsonl validation_inputs.jsonl; do
  echo "========================================"
  echo "FILE: $file"
    echo "========================================"
    wc -l ./LitTraceQA/$file
  head -n 5 ./LitTraceQA/$file | python3 -c "
  import json, sys
  for i, line in enumerate(sys.stdin):
  print(f'--- record {i} ---')
  print(json.dumps(json.loads(line), indent=2)[:500])
  print()
  "
  echo ""
  done

- 構造化確認bashコード②
  head -n 1 ./LitTraceQA/paper_metadata.jsonl | python3 -m json.tool | head -50
