# Connecting Beam Search to MCMC

Follow these steps to test the current code:

1. Clone your repository and initialize the submodule:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
   git submodule init
   git submodule update
   ```

2. Navigate to the `transformers` directory:
   ```bash
   cd transformers
   ```

3. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

4. Install the package in editable mode:
   ```bash
   pip install -e .
   cd src/transformers/generation
   Edit the beam_search.py file as in the file in the parent directory
   ```

5. Return to the parent directory:
   ```bash
   cd ..
   ```

6. Run the Transformers library tests:
   ```bash
   python unit_test.py
   ```



## Notes

- Make sure you have Python and pip installed on your system.
- You may need to use `python3` and `pip3` instead of `python` and `pip` depending on your system configuration.
- The exact command to run your project's unit tests may vary depending on your project structure and test setup.
- If you encounter any import errors, ensure that your `PYTHONPATH` includes both your project directory and the `transformers` subdirectory.
