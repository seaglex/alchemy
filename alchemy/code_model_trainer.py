# gsm8k_rl_training.py
import os
import re

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer
import numpy as np
from typing import Dict, List
from setuptools.errors import CompileError

class GRPODataPreparer(object):
    def __init__(self):
        self.sys_prompt = '''You are given a problem.
Think about the problem and write python code to solve the problem, and make sure the python code call print() and only print the final answer.
Output in markdown and surround the code between ```python and ```.
'''

    def format_gsm8k_prompt(self, x, tokenizer) -> Dict[str, str]:
        # 构建输入消息
        messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": x["question"]},
        ]
        final_answer = x["answer"].split("####")[-1].strip()
        return {
            "prompt": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            "final_answer": final_answer,
        }

    def prepare_gsm8k_dataset(self, tokenizer, max_length=512):
        dataset = load_dataset("openai/gsm8k", "main")["train"]
        dataset = dataset.map(lambda x: self.format_gsm8k_prompt(x, tokenizer))
        dataset = dataset.map(lambda x: {"N": len(x["prompt"])})
        dataset = dataset.select(np.where(np.array(dataset["N"]) <= max_length)[0])
        train_val_split = dataset.train_test_split(test_size=0.2, seed = 0)
        train_dataset = train_val_split["train"].select(np.arange(1000))
        eval_dataset = train_val_split["test"].select(np.arange(50))
        return train_dataset, eval_dataset


class RewardFunction(object):
    def __init__(self, sandbox):
        self._sandbox = sandbox
        self._code_regex = re.compile(r"```(?:python|py)?([\s\S]+?)```", re.MULTILINE)
        self._num_regex = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", re.MULTILINE)
    def __name__(self):
        return "RewardFunction"
    def __call__(self, completions, final_answer, **kwargs) -> List[float]:
        scores = []
        for completion, std_answer in zip(completions, final_answer):
            code_quality_score, output = self.check_code_quality(completion)
            output_format_score, answer = self.check_answer_format(output)
            answer_score = self.check_answer(answer, std_answer)
            scores.append(code_quality_score + output_format_score + answer_score)
        return scores

    def check_code_quality(self, completion) -> (float, str):
        '''
        格式不对 -4
        无代码 -3
        编译错误 -2
        运行错误 -1
        无错误 0
        '''
        code_match = self._code_regex.search(completion)
        if not code_match:
            return -4, None
        code = code_match.group(1).strip()
        if not code:
            return -3, None
        try:
            output = self._sandbox.run(code)
        except CompileError:
            return -2, None
        except RuntimeError:
            return -1, None
        return 0, output

    def check_answer_format(self, output):
        '''
        无输出 -1
        输出不含数字, -3
        含数值 0
        最后一行只有数字 1
        输出只有数字 2
        '''
        if output is not None:
            output = output.strip()
        if not output:
            return -1, None
        lines = output.split("\n")
        match = self._num_regex.fullmatch(output)
        if len(lines) == 1 and match:
            return 2, match.group(0)
        if match := self._num_regex.fullmatch(lines[-1]):
            return 1, match.group(0)
        match = self._num_regex.search(output)
        if match:
            return 0, match.group(0)
        return -3, None

    def check_answer(self, answer, final_answer):
        """
        数值相同 +5
        数值不同 -2.5
        """
        if not answer:
            return 0
        try:
            if float(answer) == float(final_answer):
                return 5
            return -2.5
        except ValueError:
            return 0


class CodeModelTrainer(object):
    def __init__(self, model, tokenizer: PreTrainedTokenizerBase):
        self._model = model
        self._tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model_path: str):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # 配置LoRA参数
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)  # 参数比例 0.28%
        return cls(model, tokenizer)

    def get_tokenizer(self):
        return self._tokenizer

    def train(self, train_dataset, reward_fn, output_dir="./output", eval_dataset=None):
        batch_size = 4
        num_epochs = 1
        learning_rate = 1e-4
        os.makedirs(output_dir, exist_ok=True)
        # 设置GRPO训练参数
        training_args = GRPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            logging_steps=5,
            save_steps=100,
            eval_steps=100,
            save_strategy="steps",
            save_total_limit=2,
            remove_unused_columns=False,
            warmup_steps=50,
            lr_scheduler_type="cosine",
            report_to="tensorboard",
            seed=31,
            use_cpu=True,
        )
        # 创建GRPO训练器
        trainer = GRPOTrainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self._tokenizer,
            reward_funcs=reward_fn,
        )

        # 开始训练
        trainer.train()

        # 保存模型
        trainer.save_model()
        self._tokenizer.save_pretrained(output_dir)
