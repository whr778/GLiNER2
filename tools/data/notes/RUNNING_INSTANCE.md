# LIVE Lambda instances -- terminate if anything is unclear

    id   032e695789d3428dbab62ff65d29e67d   name loc-control      gpu_1x_a10 $1.29/hr
    id   6d9ddc1ccf8c4d948fd5732befc4eec8   name zh-pool-scoring  gpu_1x_a10 $1.29/hr

TERMINATE (both):

    curl -s -u "$LAMBDA_API_KEY:" -X POST \
      https://cloud.lambda.ai/api/v1/instance-operations/terminate \
      -H "Content-Type: application/json" \
      -d '{"instance_ids":["032e695789d3428dbab62ff65d29e67d","6d9ddc1ccf8c4d948fd5732befc4eec8"]}'

VERIFY EMPTY:

    curl -s -u "$LAMBDA_API_KEY:" https://cloud.lambda.ai/api/v1/instances

Both carry tools/lambda/box_run.sh, which refuses to start unless it can prove it can
terminate itself, plus a job timeout, an independent hard deadline, and a terminate on the
normal path. loc-control pushes to whr778/gliner2-loc-control; zh-pool-scoring publishes
scores to whr778/chinese-casualty-corpus.
