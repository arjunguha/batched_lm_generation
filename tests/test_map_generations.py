import pytest
import tempfile
import os
from pathlib import Path
import gzip
import json
from batched_lm_generation.map_generations import map_generations
from batched_lm_generation.automodel_base import AutoModelGenerator

def test_map_generations_length_computation():
    # Create a temporary directory for input
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        input_file_path = temp_dir_path / "input.json.gz"
        output_file_path = temp_dir_path / "input.results.json.gz"

        # Simulate completion data
        completions_data = {
            "completions": [
                {"count": 1, "text": "First completion"},
                {"count": 2, "text": "Second completion text"},
            ]
        }

        # Write the simulated data to a gzip file
        with gzip.open(input_file_path, 'wt', encoding='utf-8') as f:
            json.dump(completions_data, f)

        # Define a mapping function that computes the lengths of each completion
        def compute_length(text):
            return str(len(text))

        # Apply the map_generations function
        map_generations(temp_dir_path, compute_length)

        # Verify the output matches the expected results
        with gzip.open(output_file_path, 'rt', encoding='utf-8') as f:
            results = json.load(f)

        expected_results = [
            {"count": 1, "text": "16"},
            {"count": 2, "text": "21"},
        ]

        assert results == expected_results, "The output from map_generations does not match the expected results."
