# LIVE Lambda instance -- terminate if anything is unclear

    id     56a06704818b4200b1dd2ebf753f58ef
    name   base-137k-v2
    type   gpu_2x_h100_sxm5  $8.38/hr  us-south-2
    job    RETRAIN the 137k boundary base on a unified ENGLISH label space
    launch torchrun --standalone --nproc_per_node=2 (the config is written for exactly this;
           batch_size is PER GPU and accumulation is already halved to keep effective 32)
    expect ~3.9h, ~$32
    pushes to whr778/gliner2-joint-boundary-mmbert-137k-v2 (PRIVATE)

TERMINATE:

    curl -s -u "$LAMBDA_API_KEY:" -X POST \
      https://cloud.lambda.ai/api/v1/instance-operations/terminate \
      -H "Content-Type: application/json" \
      -d '{"instance_ids":["56a06704818b4200b1dd2ebf753f58ef"]}'

VERIFY EMPTY:  curl -s -u "$LAMBDA_API_KEY:" https://cloud.lambda.ai/api/v1/instances
