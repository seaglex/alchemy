import re
import datasets
import argparse
import os

from alchemy.causal_model_wrapper import CausalModelWrapper
from alchemy.code_model_wrapper import CodeModelWrapper
from alchemy.tools.sandboxes import LocalSandbox, SubprocessSandbox
from alchemy.tools.code_log_parser import LogModelWrapper


os.environ["TOKENIZERS_PARALLELISM"] = "false"


GSM8_causal_prompts = {
    "qwen2.5": """
请回答下面数学问题，中间可以有思考和计算，计算结果单列一行，以“answer: "开头，不要包含单位。

## 示例
user
There were 7 birds on the tree, and then 2 flew away. How many birds are on the tree now?
assistant
7 - 2 = 5, so there are 5 birds left.
answer: 5

## 限制
- 最终答案必须单独一行，`answer: `后面紧跟计算结果，不包含单位。
"""
}


GSM8_code_prompts = {
    "qwen2.5": """下面你会看到一个问题，请写一段python代码计算答案，最终显式调用print函数把答案打印出来。
python代码按markdown格式输出。

## 示例
user
There were 7 birds on the tree, and then 2 flew away. How many birds are on the tree now?
assistant
7 - 2 = 5, so there are 5 birds left.
```python
total = 7
left = 2
print(total - left)
```

## 限制
- 输出的python代码按markdown格式，以```python开始，以```结束
- 代码计算出最终结果需要显式调用print打印
- 代码**不能没有print函数**""",
    "qwen2.5_lora": """You are given a problem.
Think about the problem and write python code to solve the problem, and make sure the python code call print() and only print the final answer.
Output in markdown and surround the code between ```python and ```.""",
}


class GSM8KBench:
    def extract_value(self, text: str) -> float | None:
        """
        从模型输出中用正则表达式提取答案。
        答案格式为：最后一行含有数字{answer}
        :param text: 模型生成的文本
        :return: 提取的答案，如果没有找到则返回空字符串
        """
        if not text:
            return None
        text = text.strip().splitlines()[-1]
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
        return float(match.group(0)) if match else None

    def evaluate(self, model):
        """
        遍历 GSM8k 测试集，对比 model 的返回结果和标准答案，计算准确率。

        :param model: 具有 get_answers(question) -> str 方法的模型对象
        :return: 准确率 (float)
        """
        test_id = "openai/gsm8k"
        test_dataset = datasets.load_dataset(test_id, "main")["test"]
        correct_count = 0
        total_count = len(test_dataset)

        for n, item in enumerate(test_dataset):
            question = item["question"]
            std_answer = item["answer"]
            std_num = self.extract_value(std_answer)
            try:
                model_answer = model.get_answers(f"{test_id}-{n}", question)  # 假设返回 str
                if not model_answer:
                    num = "NoAnswer"
                else:
                    num = self.extract_value(model_answer)
            except Exception as e:
                num = repr(e)
            print(std_num, num, flush=True)

            # 假设每个问题只有一个正确答案
            if std_num == num:
                correct_count += 1
        accuracy = correct_count / total_count if total_count > 0 else 0
        return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--model_path", type=str, required=True)
    parser.add_argument("-l", "--lora_path", type=str, required=False)
    parser.add_argument("-m", "--model_name", type=str, required=True)
    args = parser.parse_args()
    bench = GSM8KBench()
    model_name = args.model_name
    if args.lora_path:
        rate = bench.evaluate(CodeModelWrapper.from_pretrained_lora(
            args.model_path, args.lora_path, GSM8_code_prompts.get(model_name), SubprocessSandbox())
        )
    else:
        rate = bench.evaluate(CodeModelWrapper.from_pretrained(
            args.model_path, GSM8_code_prompts.get(model_name), SubprocessSandbox())
        )
        # rate = bench.evaluate(LogModelWrapper(args.model_path, LocalSandbox()))
    print(f"model={model_name}, accuracy: {rate:.2%}")
