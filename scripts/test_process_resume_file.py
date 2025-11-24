import os
import sys
import types
import json
import tempfile

# Ensure project root is on path so we can import src.utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create a dummy 'mistralai' module to avoid ImportError during tests when the
# real SDK isn't installed. utils.py expects `from mistralai import Mistral` at
# module import time which fails in a test environment without the SDK.
dummy_mod = types.ModuleType('mistralai')
dummy_mod.Mistral = lambda api_key=None: None
sys.modules['mistralai'] = dummy_mod

from src import utils


def test_parsed_json_no_api_call():
    # Write a JSON file with parsed resume fields
    content = {
        "personal_information": {"name": "Alice"},
        "skills": ["Python", "ML"],
        "experience": [{"title": "Engineer", "company": "Acme"}]
    }

    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as tmp:
        json.dump(content, tmp)
        tmp_path = tmp.name

    try:
        # Ensure client is None so calling extract_resume_json would raise an error
        utils.client = None
        parsed = utils.process_resume_file(tmp_path)
        assert parsed.get('skills') == content['skills']
        print('PASS: parsed JSON returned without API call')
    finally:
        os.remove(tmp_path)


def test_wrapper_json_calls_api():
    # Write a JSON file that wraps raw text
    content = {
        "text": "Sally is a software engineer with Python experience.",
    }

    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as tmp:
        json.dump(content, tmp)
        tmp_path = tmp.name

    try:
        utils.client = None
        try:
            utils.process_resume_file(tmp_path)
            print('FAIL: Expected ValueError when client is None and wrapper JSON triggers API call')
        except ValueError:
            print('PASS: wrapper JSON triggers API call and ValueError raised when client is None')
    finally:
        os.remove(tmp_path)


if __name__ == '__main__':
    test_parsed_json_no_api_call()
    test_wrapper_json_calls_api()
