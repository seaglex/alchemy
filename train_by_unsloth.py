from unsloth import FastLanguageModel
from vllm import SamplingParams
from trl import GRPOConfig, GRPOTrainer


from alchemy.code_model_trainer import GRPODataPreparer, RewardFunction
from alchemy.tools.sandboxes import LocalSandbox, SubprocessSandbox


def train_by_unsloth():
    max_seq_length = 2048  # Can increase for longer reasoning traces
    lora_rank = 16  # Larger rank = smarter, but slower

    maximum_length = 512
    max_prompt_length = maximum_length + 1  # + 1 just in case!
    max_completion_length = max_seq_length - max_prompt_length

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-1.5B-Instruct",
        max_seq_length=max_seq_length,
        load_in_4bit=False,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.5,  # Reduce if out of memory
    )

    vllm_sampling_params = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=3407,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )

    training_args = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
        temperature=1.0,
        learning_rate=5e-6,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,  # Increase to 4 for smoother training
        num_generations=4,  # Decrease if out of memory
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps=2000,
        save_steps=200,
        report_to="none",  # Can use Weights & Biases
        output_dir="outputs",

        # For optional training + evaluation
        # fp16_full_eval = True,
        # per_device_eval_batch_size = 4,
        # eval_accumulation_steps = 1,
        # eval_strategy = "steps",
        # eval_steps = 1,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank * 2,  # *2 speeds up training
        use_gradient_checkpointing="unsloth",  # Reduces memory usage
        random_state=3407,
    )

    # For optional training + evaluation
    # new_dataset = dataset.train_test_split(test_size = 0.01)

    dataset, eval_dataset = GRPODataPreparer().prepare_gsm8k_dataset(tokenizer)

    reward_fn = RewardFunction(SubprocessSandbox())

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_fn,
        ],
        args=training_args,
        train_dataset=dataset,

        # For optional training + evaluation
        # train_dataset = new_dataset["train"],
        # eval_dataset = new_dataset["test"],
    )
    trainer.train()
    model.save_lora("./grpo_saved_lora_2000")

try:
    train_by_unsloth()
except Exception as e:
    print(e)
