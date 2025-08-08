from alchemy import code_model_trainer
from alchemy.tools.sandboxes import LocalSandbox, SubprocessSandbox


def train():
    model_path = '/Users/weixuan/models/Qwen2.5-1.5B-Instruct/'
    trainer = code_model_trainer.CodeModelTrainer.from_pretrained(model_path)
    tokenizer = trainer.get_tokenizer()
    train_dataset, eval_dataset = code_model_trainer.GRPODataPreparer().prepare_gsm8k_dataset(tokenizer)
    reward_fn = code_model_trainer.RewardFunction(SubprocessSandbox())
    trainer.train(train_dataset, reward_fn, output_dir="./qwen2.5_gsm8k_code", eval_dataset=eval_dataset)


if  __name__ == "__main__":
    train()