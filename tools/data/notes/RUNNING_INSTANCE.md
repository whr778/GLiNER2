# LIVE Lambda instance -- terminate if anything is unclear

    id   fb46ee3d0dea485589af0c8dc6516e2d
    name zh-multitask   gpu_1x_a10  $1.29/hr  us-east-1
    job  Chinese multi-task training, warm start from gliner2-joint-boundary-mmbert-137k
    pushes to whr778/gliner2-zh-multitask-mmbert (PRIVATE)

TERMINATE:

    curl -s -u "$LAMBDA_API_KEY:" -X POST \
      https://cloud.lambda.ai/api/v1/instance-operations/terminate \
      -H "Content-Type: application/json" \
      -d '{"instance_ids":["fb46ee3d0dea485589af0c8dc6516e2d"]}'

VERIFY EMPTY:  curl -s -u "$LAMBDA_API_KEY:" https://cloud.lambda.ai/api/v1/instances
