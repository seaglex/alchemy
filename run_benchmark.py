import re
import datasets
import argparse

from alchemy import causal_model_wrapper


GSM8_model_prompts = {
    "qwen2.5": """
请回答下面数学问题，中间可以有思考和计算，计算结果单列一行，以“answer: "开头，不要包含单位。

## 示例
There were 7 birds on the tree, and then 2 flew away. How many birds are on the tree now?
7 - 2 = 5, so there are 5 birds left.
answer: 5

## 限制
- 最终答案必须单独一行，`answer: `后面紧跟计算结果，不包含单位。
"""
}


GSM8_code_prompts = {
    "qwen2.5": '''
下面你会看到一个问题，请写一段python代码计算答案，最终用print函数把答案打印出来。
python代码按markdown格式输出。

## 示例
There were 7 birds on the tree, and then 2 flew away. How many birds are on the tree now?
7 - 2 = 5, so there are 5 birds left.
```python
print(7 - 2)
```

## 限制
- 输出的python代码按markdown格式，以```python开始，以```结束
- 代码计算出最终结果需要print打印'''
}


local_models = {
    "qwen2.5": "Qwen2.5-1.5B-Instruct"
}


class GSM8KBench:
    def extract_value(self, text: str) -> float | None:
        """
        从模型输出中用正则表达式提取答案。
        答案格式为：最后一行 & #### {answer}
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

        :param model: 具有 get_answers(question) -> list[str] 方法的模型对象
        :return: 准确率 (float)
        """
        test_dataset = datasets.load_dataset("openai/gsm8k", "main")["test"]
        correct_count = 0
        total_count = len(test_dataset)

        for item in test_dataset:
            question = item["question"]
            std_answer = item["answer"]
            std_num = self.extract_value(std_answer)
            model_answers = model.get_answers(question)  # 假设返回 str
            print(model_answers, flush=True)
            num = self.extract_value(model_answers)
            print(std_num, num, flush=True)

            # 假设每个问题只有一个正确答案
            if std_num == num:
                correct_count += 1
        accuracy = correct_count / total_count if total_count > 0 else 0
        return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--model_path", type=str, required=True)
    parser.add_argument("-m", "model_name", type=str, required=True)
    args = parser.parse_args()
    bench = GSM8KBench()
    model_name = args.model_name
    model_path = args.model_path + local_models[model_name]
    rate = bench.evaluate(causal_model_wrapper.CausalModelWrapper.from_pretrained(model_path, GSM8_model_prompts.get(model_name)))
    print(f"Accuracy: {rate:.2%}")
