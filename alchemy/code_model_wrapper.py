import re

from peft import PeftModel
from sympy import resultant
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from typing import Optional, Tuple
import logging

from alchemy.tools.sandboxes import LocalSandbox


# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    filename="code_wrapper.log")


class CodeModelWrapper(object):
    def __init__(self, model, tokenizer, sys_prompt: str, sandbox: LocalSandbox):
        self.model = model
        self.tokenizer = tokenizer
        self.sys_prompt: str = sys_prompt
        self.sandbox = sandbox  # 接收一个 LocalSandbox 实例

    @classmethod
    def from_pretrained(cls, model_path: str, sys_prompt: str = None, sandbox = None):
        if sys_prompt is None:
            sys_prompt = "请根据用户的问题生成对应的 Python 代码，用 markdown 格式包裹。"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return cls(model, tokenizer, sys_prompt, sandbox)

    @classmethod
    def from_pretrained_lora(cls, base_path, lora_path, sys_prompt, sandbox = None):
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(base_path)
        model = PeftModel.from_pretrained(base_model, lora_path)
        return cls(model, tokenizer, sys_prompt, sandbox)

    def _generate_code(self, question: str) -> (str, Optional[str]):
        """
        生成包含代码的 Markdown 输出，并从中提取出纯代码字符串。

        Args:
            question (str): 用户问题或指令。

        Returns:
            Tuple[str, Optional[str]]: 原始回答 + 提取出的代码
        """
        # 构建输入消息
        messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": question},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 编码输入
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.model.device)

        # 生成回答
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=4096,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        output_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                         zip(inputs.input_ids, outputs)]

        # 解码输出
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


        # 提取代码块中的内容
        code_match = re.search(r"```(?:python|py)?([\s\S]*?)```", output_text)
        code = code_match.group(1).strip() if code_match else None

        return output_text, code

    def get_answers(self, question_id, question: str) -> str:
        """
        生成代码、提取代码并执行它，返回执行结果。

        Args:
            question_id：用户问题id，便于调试
            question (str): 用户问题或指令。

        Returns:
            Tuple[str, str]: 原始生成的回答 + 代码执行结果
        """

        # 记录各项信息
        logging.info(f"ID: {question_id}")
        logging.info(f"Question:\n{question}")
        output_text, code = self._generate_code(question)
        logging.info(f"Output-Text:\n{output_text}")
        logging.info(f"Code:\n{code}")
        result = self.sandbox.run(code)
        logging.info(f"Sandbox-Result:\n{result}")
        return result
