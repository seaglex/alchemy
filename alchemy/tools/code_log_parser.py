import re

from transformers.integrations import is_codecarbon_available


class LogNotFoundError(Exception):
    pass


class LogModelWrapper(object):
    def __init__(self, log_file_path: str, sandbox):
        self.code_store = LogModelWrapper.parse_log_file(log_file_path)
        self.sandbox = sandbox

    @staticmethod
    def parse_log_file(log_file_path: str) -> dict:
        """
        解析日志文件，提取 question_id 和对应的 code。

        Args:
            log_file_path (str): 日志文件路径。

        Returns:
            dict: 包含 question_id 和 code 的字典。
        """
        # 正则表达式匹配 ID 和 Code 部分
        head_pattern = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - INFO - (.*)")
        id_pattern = re.compile(r"ID:\s*(.+)")
        id_codes = {}
        current_id = None
        last_head = None
        doc_lines = []

        with open(log_file_path, 'r') as file:
            for line in file:
                # 查找 head
                head_match = head_pattern.match(line)
                if head_match:
                    if last_head == "Code:":
                        id_codes[current_id] = "".join(doc_lines)
                    # 如何是 question_id
                    last_head = head_match.group(1)
                    id_match = id_pattern.match(last_head)
                    if id_match:
                        current_id = id_match.group(1)
                    doc_lines = []
                    continue

                doc_lines.append(line)
        return id_codes
    def get_answers(self, question_id, question: str) -> str:
        if question_id not in self.code_store:
            raise LogNotFoundError(f"Question ID {question_id} not found in log file.")
        code = self.code_store[question_id]
        return self.sandbox.run(code)
