pip install pytest
cd library
cd sparse_attention
pip install -e . --no-build-isolation
pytest test_sparse.py
cd ..
cd lsh
pip install -e . --no-build-isolation
pytest test.py
cd ../..