from src.training.tinker_sdk import TinkerSDKClient


def test_tinker_client_requires_api_key() -> None:
    client = TinkerSDKClient(api_key="")

    assert not client.is_available
