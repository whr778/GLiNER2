# LIVE Lambda instance -- terminate if anything is unclear

    id   779bc45bdd284b4d99c321d2e5ef4239
    name tr-pool-scoring
    type gpu_1x_a10  $1.29/hr  us-east-1
    job  score 160,038 prefiltered Turkish pool docs with whr778/gliner2-gate2-mmbert-tr
    launched 2026-08-29

TERMINATE:

    curl -s -u "$LAMBDA_API_KEY:" -X POST \
      https://cloud.lambda.ai/api/v1/instance-operations/terminate \
      -H "Content-Type: application/json" \
      -d '{"instance_ids":["779bc45bdd284b4d99c321d2e5ef4239"]}'

VERIFY EMPTY:

    curl -s -u "$LAMBDA_API_KEY:" https://cloud.lambda.ai/api/v1/instances

It SHOULD self-terminate: `tools/lambda/box_run.sh` carries a pre-flight that refuses to
start unless it can prove it can terminate itself, a job timeout, an independent hard
deadline, and a terminate on the normal path. Results publish to the PRIVATE dataset
`whr778/turkish-pool-gate-scores` (partial results under `partial/` if the job fails),
because the disk dies with the instance. Verify anyway.
