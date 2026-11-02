from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup



# scheduler = get_cosine_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=int(0.1 * num_steps), # 0.1 * 10000
#     num_training_steps=num_steps
# )