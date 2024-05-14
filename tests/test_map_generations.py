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
        # Create multiple input files
        input_file_paths = [
            temp_dir_path / f"input_{i}.json.gz" for i in range(2)
        ]
        output_file_paths = [
            temp_dir_path / f"input_{i}.results.json.gz" for i in range(2)
        ]

        # Simulate completion data for each input file
        completions_data = [
            {
                "completions": [
                    {"count": 1, "text": "First completion"},
                    {"count": 2, "text": "Second completion text"},
                ]
            },
            {
                "completions": [
                    {"count": 3, "text": "Third completion"},
                    {"count": 4, "text": "Fourth completion text"},
                ]
            },
        ]

        # Write the simulated data to each gzip file
        for i, input_file_path in enumerate(input_file_paths):
            with gzip.open(input_file_path, 'wt', encoding='utf-8') as f:
                json.dump(completions_data[i], f)

        # Define a mapping function that computes the lengths of each completion
        def compute_length(text):
            return str(len(text))

        # Apply the map_generations function
        map_generations(temp_dir_path, compute_length)

        # Verify the output matches the expected results for each file
        expected_results = [
            [
                {"count": 1, "text": "16"},
                {"count": 2, "text": "21"},
            ],
            [
                {"count": 3, "text": "15"},
                {"count": 4, "text": "22"},
            ],
        ]

        for i, output_file_path in enumerate(output_file_paths):
            with gzip.open(output_file_path, 'rt', encoding='utf-8') as f:
                results = json.load(f)
            assert results == expected_results[i], f"The output from map_generations for file {i} does not match the expected results."
