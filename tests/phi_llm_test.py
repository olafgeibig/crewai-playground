import pytest
from src.phi_llm import ChatPhidataAdapter
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import BaseMessage

@pytest.fixture
def setup_chatphidata_adapter():
    model = "test_model"
    base_url = "http://test.url"
    adapter = ChatPhidataAdapter(model, base_url)
    return adapter

def test_initialize_ollama(setup_chatphidata_adapter):
    adapter = setup_chatphidata_adapter
    assert adapter.ollama.model == "test_model"
    assert adapter.ollama.base_url == "http://test.url"

def test_invoke(setup_chatphidata_adapter, mocker):
    adapter = setup_chatphidata_adapter
    prompt = "Test prompt"
    language_model_input = LanguageModelInput(text=prompt)
    expected_response = "Test response"
    
    # Mock the Ollama generate_text method
    mocker.patch.object(adapter.ollama, 'generate_text', return_value=expected_response)
    
    response = adapter.invoke(language_model_input)
    
    # Verify the response is a BaseMessage and contains the expected text
    assert isinstance(response, BaseMessage)
    assert response.text == expected_response

def test_ollama_instance_getter(setup_chatphidata_adapter):
    adapter = setup_chatphidata_adapter
    assert adapter.ollama_instance == adapter.ollama