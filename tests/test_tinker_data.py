from src.training.tinker_data import render_chat_example


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = [ord(char) for char in text]
        if add_special_tokens:
            return [0] + tokens
        return tokens


DUMMY_MESSAGES = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi"},
]


def test_render_chat_example_to_datum() -> None:
    tokenizer = FakeTokenizer()
    datum = render_chat_example(DUMMY_MESSAGES, tokenizer)
    assert datum.loss_fn_inputs["target_tokens"]
