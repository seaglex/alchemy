from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


class CausalModelWrapper(object):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, sys_prompt: str):
        self.model = model
        self.tokenizer = tokenizer
        self.sys_prompt: str = sys_prompt

    @classmethod
    def from_pretrained(cls, model_path: str, sys_prompt:str = None):
        """
        从预训练模型加载模型和 tokenizer，并创建一个 CausalModelWrapper 对象。

        Args:
            model_path (str): 预训练模型的路径。
            sys_prompt (str): 用于生成回答的提示。

        Returns:
            CausalModelWrapper: 包含模型和 tokenizer 的 CausalModelWrapper 对象。
        """
        if sys_prompt is None:
            sys_prompt = "请回答下面数学问题，中间可以有思考，最终的答案单列一行，以#### 开头，不要包含单位。"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return cls(model, tokenizer, sys_prompt)

    def get_answers(self, question) -> str:
        # 将问题和上下文组合为完整的输入文本
        messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": question},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 使用 tokenizer 对输入进行编码
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.model.device)

        # 调用模型的 generate 方法生成回答
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=2048,  # 控制生成的最大 token 数量
            num_return_sequences=1,  # 返回序列的数量
            do_sample=True,  # 是否采样（适用于生成更自然的回答）
            temperature=0.7,  # 采样温度
            top_p=0.9,  # nucleus sampling 参数
        )

        # 解码生成的回答
        output_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                         zip(inputs.input_ids, outputs)]
        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # 返回包含一个答案的列表
        return answer

