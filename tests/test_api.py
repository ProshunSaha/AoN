#Run pytest from the root directory OR THERE MIGHT BE ERRORS

import pytest

from fastapi.testclient import TestClient
from main import app




tc = TestClient(app)

def test_health():
    r = tc.get('/health')
    assert r.status_code == 200
    assert r.json() == {'status': 'ok'}

def test_predict_valid():
    with open('tests/assets/sample1.jpg', 'rb') as f:
        r = tc.post('/predict', files = {'file' : ('sample1.jpg', f, 'image/jpeg')})
    assert r.status_code == 200
    assert 'label' in r.json()
    assert 'confidence' in r.json()

def test_predict_invalid_type():
    r = tc.post('/predict', files = {'file' :('test.txt', b'hello','text/plain')})
    assert r.status_code == 415