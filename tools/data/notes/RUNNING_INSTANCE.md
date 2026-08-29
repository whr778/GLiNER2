# LIVE Lambda instance -- terminate if anything is unclear

    id   e04c9cd2e5084b4f9eaac33db188dada
    name tr-dose-curve
    type gpu_1x_a10  $1.29/hr  us-east-1
    job  4 Turkish dose-curve arms (0 / 5,000 / 15,000 / 29,700 Turkish rows)
    launched 2026-08-29

TERMINATE:

    curl -s -u "$LAMBDA_API_KEY:" -X POST \
      https://cloud.lambda.ai/api/v1/instance-operations/terminate \
      -H "Content-Type: application/json" \
      -d '{"instance_ids":["e04c9cd2e5084b4f9eaac33db188dada"]}'

VERIFY EMPTY:

    curl -s -u "$LAMBDA_API_KEY:" https://cloud.lambda.ai/api/v1/instances

It SHOULD self-terminate via tools/lambda/box_run.sh, which refuses to start unless it
can prove it can kill itself, and carries a job timeout, an independent hard deadline and
a terminate on the normal path. Models push PRIVATE to whr778/gliner2-tr-dose-<N>.
Verify anyway.
