# LIVE Lambda instance -- terminate if anything is unclear

    id   032e695789d3428dbab62ff65d29e67d
    name loc-control
    type gpu_1x_a10  $1.29/hr  us-east-1
    job  location-supervision control: same 13,080 documents as tr-dose0, same replay,
         labels from casualty_natural (93.0% location) instead of casualty_docee (0%)
    launched 2026-08-31   expected ~2.6h, ~$3.40

TERMINATE:

    curl -s -u "$LAMBDA_API_KEY:" -X POST \
      https://cloud.lambda.ai/api/v1/instance-operations/terminate \
      -H "Content-Type: application/json" \
      -d '{"instance_ids":["032e695789d3428dbab62ff65d29e67d"]}'

VERIFY EMPTY:

    curl -s -u "$LAMBDA_API_KEY:" https://cloud.lambda.ai/api/v1/instances

It SHOULD self-terminate via tools/lambda/box_run.sh, which refuses to start unless it can
prove it can kill itself, plus a job timeout, an independent hard deadline, and a
terminate on the normal path. The model pushes PRIVATE to whr778/gliner2-loc-control.
