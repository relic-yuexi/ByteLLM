import os
from transformers import PreTrainedTokenizer

from typing import Optional, Tuple


class ByteTokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        bos_token = "\u0002"  # <BOS>
        eos_token = "\u0003"  # <EOS>
        pad_token = "\u0000"  # <PAD>
        unk_token = "\u0000"  # <UNK>
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            **kwargs,
        )

    def _tokenize(self, text):
        return [chr(byte) for byte in text.encode("utf-8")]

    def _convert_token_to_id(self, token):
        return ord(token)

    def _convert_id_to_token(self, index):
        if index > 255:
            return "\u0000"  # return unk_token
        return chr(index)

    def convert_tokens_to_string(self, tokens):
        # 尝试将tokens转换为字符串，如果失败，则移除最后一个token直到成功
        while tokens:
            try:
                return "".join(tokens).encode("latin1").decode("utf-8")
            except UnicodeDecodeError:
                # 如果解码失败，移除最后一个token并再次尝试
                tokens.pop()
        return ""  # 如果所有tokens都无法解码，返回空字符串

    @property
    def vocab_size(self):
        return 256

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        filename_prefix = filename_prefix or self.__class__.__name__
        vocab_file = os.path.join(save_directory, filename_prefix + ".txt")
        vocab = self.get_vocab()
        with open(vocab_file, "w", encoding="utf-8") as f:
            for token, token_id in vocab.items():
                f.write(f"{token}\t{token_id}\n")
        return (vocab_file,)

    def get_vocab(self):
        return {chr(i): i for i in range(256)}

    def __len__(self):
        return 256
