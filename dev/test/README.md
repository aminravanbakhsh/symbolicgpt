1. Make sure that pytest is installed on your virtual environment.

```bash
    pytest --version
```
or

```bash
    pip install pytest
```

2. Change your terminal directory to the testing directory.

```bash
    cd "PATH_TO_ROOT/utils/test"
```

3. Run the tests. For your convenience you can select your desired testing function from the files as below.

```bash
    pytest -s FILE_test.py::TEST_FUNCTION
```

for example:

```bash
    pytest -s dimension_analysis_test.py::test_001
```

