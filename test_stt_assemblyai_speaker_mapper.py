#!/usr/bin/env python3
"""
Unit tests for stt_assemblyai_speaker_mapper.py

Tests recursive JSON traversal, speaker detection, mapping formats,
and edge cases.
"""

import unittest
import json
import tempfile
import os
from unittest.mock import Mock, patch
import sys

# Import the module under test
import stt_assemblyai_speaker_mapper as mapper


class TestSpeakerDetection(unittest.TestCase):
    """Test speaker detection from various JSON structures."""

    def test_detect_speakers_simple(self):
        """Test basic speaker detection."""
        json_obj = {
            "utterances": [
                {"speaker": "A", "text": "Hello"},
                {"speaker": "B", "text": "Hi"},
                {"speaker": "A", "text": "How are you?"}
            ]
        }
        speakers = mapper.detect_speakers_in_json(json_obj)
        self.assertEqual(speakers, {"A", "B"})

    def test_detect_speakers_nested(self):
        """Test detection in nested structures."""
        json_obj = {
            "data": {
                "segments": [
                    {"speaker": "X", "text": "Test"},
                    {"speaker": "Y", "text": "Test2"}
                ],
                "metadata": {
                    "primary_speaker": "Z"
                }
            }
        }
        speakers = mapper.detect_speakers_in_json(json_obj)
        self.assertEqual(speakers, {"X", "Y"})  # Only "speaker" keys, not other fields

    def test_detect_speakers_deep_nesting(self):
        """Test detection in deeply nested structures."""
        json_obj = {
            "level1": {
                "level2": {
                    "level3": [
                        {"speaker": "A", "text": "Deep"},
                        {"speaker": "B", "text": "Deeper"}
                    ]
                }
            }
        }
        speakers = mapper.detect_speakers_in_json(json_obj)
        self.assertEqual(speakers, {"A", "B"})

    def test_detect_speakers_with_words(self):
        """Test detection from AssemblyAI full format with words."""
        json_obj = {
            "utterances": [
                {
                    "speaker": "A",
                    "text": "Hello world",
                    "words": [
                        {"text": "Hello", "speaker": "A"},
                        {"text": "world", "speaker": "A"}
                    ]
                }
            ]
        }
        speakers = mapper.detect_speakers_in_json(json_obj)
        self.assertEqual(speakers, {"A"})

    def test_detect_no_speakers(self):
        """Test when no speakers exist."""
        json_obj = {"text": "Some transcript", "confidence": 0.9}
        speakers = mapper.detect_speakers_in_json(json_obj)
        self.assertEqual(speakers, set())

    def test_detect_speakers_multiple_lists(self):
        """Test detection from multiple separate lists."""
        json_obj = {
            "section1": [
                {"speaker": "A", "text": "First"},
            ],
            "section2": [
                {"speaker": "B", "text": "Second"},
                {"speaker": "C", "text": "Third"}
            ]
        }
        speakers = mapper.detect_speakers_in_json(json_obj)
        self.assertEqual(speakers, {"A", "B", "C"})


class TestRecursiveReplacement(unittest.TestCase):
    """Test recursive speaker replacement."""

    def test_replace_simple(self):
        """Test basic replacement."""
        json_obj = {
            "utterances": [
                {"speaker": "A", "text": "Hello"},
                {"speaker": "B", "text": "Hi"}
            ]
        }
        speaker_map = {"A": "Alice", "B": "Bob"}
        result = mapper.replace_speakers_recursive(json_obj, speaker_map)

        self.assertEqual(result["utterances"][0]["speaker"], "Alice")
        self.assertEqual(result["utterances"][1]["speaker"], "Bob")

    def test_replace_preserves_structure(self):
        """Test that replacement preserves JSON structure."""
        json_obj = {
            "utterances": [
                {
                    "speaker": "A",
                    "text": "Hello",
                    "confidence": 0.95,
                    "start": 100,
                    "end": 500
                }
            ]
        }
        speaker_map = {"A": "Alice"}
        result = mapper.replace_speakers_recursive(json_obj, speaker_map)

        # Check speaker is replaced
        self.assertEqual(result["utterances"][0]["speaker"], "Alice")
        # Check other fields preserved
        self.assertEqual(result["utterances"][0]["text"], "Hello")
        self.assertEqual(result["utterances"][0]["confidence"], 0.95)
        self.assertEqual(result["utterances"][0]["start"], 100)

    def test_replace_nested_words(self):
        """Test replacement in nested word-level data."""
        json_obj = {
            "utterances": [
                {
                    "speaker": "A",
                    "words": [
                        {"text": "Hello", "speaker": "A"},
                        {"text": "there", "speaker": "A"}
                    ]
                }
            ]
        }
        speaker_map = {"A": "Alice Anderson"}
        result = mapper.replace_speakers_recursive(json_obj, speaker_map)

        self.assertEqual(result["utterances"][0]["speaker"], "Alice Anderson")
        self.assertEqual(result["utterances"][0]["words"][0]["speaker"], "Alice Anderson")
        self.assertEqual(result["utterances"][0]["words"][1]["speaker"], "Alice Anderson")

    def test_replace_partial_mapping(self):
        """Test partial mapping (only some speakers mapped)."""
        json_obj = {
            "utterances": [
                {"speaker": "A", "text": "Hello"},
                {"speaker": "B", "text": "Hi"},
                {"speaker": "C", "text": "Hey"}
            ]
        }
        speaker_map = {"A": "Alice", "B": "Bob"}  # C not mapped
        result = mapper.replace_speakers_recursive(json_obj, speaker_map)

        self.assertEqual(result["utterances"][0]["speaker"], "Alice")
        self.assertEqual(result["utterances"][1]["speaker"], "Bob")
        self.assertEqual(result["utterances"][2]["speaker"], "C")  # Unchanged

    def test_replace_empty_map(self):
        """Test with empty speaker map."""
        json_obj = {"utterances": [{"speaker": "A", "text": "Hello"}]}
        speaker_map = {}
        result = mapper.replace_speakers_recursive(json_obj, speaker_map)

        self.assertEqual(result["utterances"][0]["speaker"], "A")  # Unchanged

    def test_replace_does_not_affect_other_fields(self):
        """Test that only 'speaker' keys are affected."""
        json_obj = {
            "speaker_count": 2,  # Field name contains 'speaker' but isn't exactly 'speaker'
            "utterances": [
                {"speaker": "A", "text": "Hello", "speaker_label": "primary"}
            ]
        }
        speaker_map = {"A": "Alice"}
        result = mapper.replace_speakers_recursive(json_obj, speaker_map)

        self.assertEqual(result["speaker_count"], 2)  # Not changed
        self.assertEqual(result["utterances"][0]["speaker"], "Alice")  # Changed
        self.assertEqual(result["utterances"][0]["speaker_label"], "primary")  # Not changed


class TestMappingParsers(unittest.TestCase):
    """Test various mapping input formats."""

    def test_parse_inline_simple(self):
        """Test comma-separated inline parsing."""
        detected = {"A", "B", "C"}
        speaker_map = mapper.parse_speaker_map_inline("Alice,Bob,Charlie", detected)

        self.assertEqual(speaker_map, {
            "A": "Alice",
            "B": "Bob",
            "C": "Charlie"
        })

    def test_parse_inline_with_whitespace(self):
        """Test inline parsing with extra whitespace."""
        detected = {"A", "B"}
        speaker_map = mapper.parse_speaker_map_inline(" Alice , Bob ", detected)

        self.assertEqual(speaker_map, {"A": "Alice", "B": "Bob"})

    def test_parse_inline_fewer_names(self):
        """Test inline parsing with fewer names than speakers."""
        detected = {"A", "B", "C"}
        speaker_map = mapper.parse_speaker_map_inline("Alice,Bob", detected)

        self.assertEqual(speaker_map, {"A": "Alice", "B": "Bob"})
        self.assertNotIn("C", speaker_map)

    def test_parse_inline_empty(self):
        """Test inline parsing with empty string."""
        detected = {"A", "B"}
        speaker_map = mapper.parse_speaker_map_inline("", detected)

        self.assertEqual(speaker_map, {})

    def test_parse_file_sequential(self):
        """Test sequential file format."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("Alice Anderson\n")
            f.write("Beat Barrinson\n")
            f.write("Charlie Chaplin\n")
            temp_file = f.name

        try:
            detected = {"A", "B", "C"}
            speaker_map = mapper.parse_speaker_map_file(temp_file, detected)

            self.assertEqual(speaker_map, {
                "A": "Alice Anderson",
                "B": "Beat Barrinson",
                "C": "Charlie Chaplin"
            })
        finally:
            os.unlink(temp_file)

    def test_parse_file_keyvalue_simple(self):
        """Test key:value file format."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("A: Alice Anderson\n")
            f.write("B: Beat Barrinson\n")
            temp_file = f.name

        try:
            detected = {"A", "B"}
            speaker_map = mapper.parse_speaker_map_file(temp_file, detected)

            self.assertEqual(speaker_map, {
                "A": "Alice Anderson",
                "B": "Beat Barrinson"
            })
        finally:
            os.unlink(temp_file)

    def test_parse_file_keyvalue_full_labels(self):
        """Test full speaker label format."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("Speaker A: Alice Anderson\n")
            f.write("Speaker B: Beat Barrinson\n")
            temp_file = f.name

        try:
            detected = {"Speaker A", "Speaker B"}
            speaker_map = mapper.parse_speaker_map_file(temp_file, detected)

            self.assertEqual(speaker_map, {
                "Speaker A": "Alice Anderson",
                "Speaker B": "Beat Barrinson"
            })
        finally:
            os.unlink(temp_file)

    def test_parse_file_mixed_format(self):
        """Test mixed key:value formats."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("A: Alice\n")
            f.write("Speaker B: Bob\n")
            f.write("C: Charlie\n")
            temp_file = f.name

        try:
            detected = {"A", "Speaker B", "C"}
            speaker_map = mapper.parse_speaker_map_file(temp_file, detected)

            self.assertEqual(speaker_map, {
                "A": "Alice",
                "Speaker B": "Bob",
                "C": "Charlie"
            })
        finally:
            os.unlink(temp_file)

    def test_parse_file_with_comments(self):
        """Test file with comment lines."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("# Project speakers\n")
            f.write("A: Alice\n")
            f.write("# B is unknown\n")
            f.write("C: Charlie\n")
            temp_file = f.name

        try:
            detected = {"A", "C"}
            speaker_map = mapper.parse_speaker_map_file(temp_file, detected)

            self.assertEqual(speaker_map, {"A": "Alice", "C": "Charlie"})
        finally:
            os.unlink(temp_file)

    def test_parse_file_empty(self):
        """Test empty file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_file = f.name

        try:
            detected = {"A", "B"}
            speaker_map = mapper.parse_speaker_map_file(temp_file, detected)

            self.assertEqual(speaker_map, {})
        finally:
            os.unlink(temp_file)


class TestTranscriptGeneration(unittest.TestCase):
    """Test TXT transcript generation."""

    def test_generate_txt_simple(self):
        """Test basic TXT generation."""
        json_obj = {
            "utterances": [
                {"speaker": "Alice", "text": "Hello there"},
                {"speaker": "Bob", "text": "Hi, how are you?"}
            ]
        }
        txt = mapper.generate_txt_from_json(json_obj)

        expected = "Alice:\tHello there\nBob:\tHi, how are you?\n"
        self.assertEqual(txt, expected)

    def test_generate_txt_tab_format(self):
        """Test that tab is used after speaker name."""
        json_obj = {
            "utterances": [
                {"speaker": "Alice Anderson", "text": "Test"}
            ]
        }
        txt = mapper.generate_txt_from_json(json_obj)

        self.assertIn("Alice Anderson:\t", txt)
        self.assertIn("\t", txt)

    def test_generate_txt_nested_location(self):
        """Test TXT generation from nested structure."""
        json_obj = {
            "data": {
                "utterances": [
                    {"speaker": "A", "text": "Test"}
                ]
            }
        }
        # Note: find_transcript_segments looks for 'utterances' at any level
        # This test verifies it works
        segments = mapper.find_transcript_segments(json_obj)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["speaker"], "A")

    def test_generate_txt_no_utterances(self):
        """Test TXT generation when no utterances found."""
        json_obj = {"text": "Raw transcript without speakers"}
        txt = mapper.generate_txt_from_json(json_obj)

        self.assertEqual(txt, "")

    def test_find_transcript_segments_fast_path(self):
        """Test that common 'utterances' path is found quickly."""
        json_obj = {
            "utterances": [
                {"speaker": "A", "text": "Fast path"},
                {"speaker": "B", "text": "Test"}
            ]
        }
        segments = mapper.find_transcript_segments(json_obj)

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["speaker"], "A")

    def test_find_transcript_segments_recursive(self):
        """Test recursive search for segment-like structures."""
        json_obj = {
            "metadata": {},
            "deep": {
                "nested": {
                    "segments": [
                        {"speaker": "X", "text": "Found it"},
                        {"speaker": "Y", "text": "Me too"}
                    ]
                }
            }
        }
        segments = mapper.find_transcript_segments(json_obj)

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["speaker"], "X")


class TestOutputPath(unittest.TestCase):
    """Test output path generation."""

    def test_output_path_json_input(self):
        """Test output path for .json input."""
        input_path = "audio.mp3.assemblyai.json"
        output = mapper.generate_output_path(input_path, extension='.json')

        self.assertEqual(output, "audio.mp3.assemblyai.mapped.json")

    def test_output_path_txt_extension(self):
        """Test output path with .txt extension."""
        input_path = "audio.assemblyai.json"
        output = mapper.generate_output_path(input_path, extension='.txt')

        self.assertEqual(output, "audio.assemblyai.mapped.txt")

    def test_output_path_no_extension(self):
        """Test output path without extension."""
        input_path = "audio.assemblyai.json"
        output = mapper.generate_output_path(input_path, extension='')

        self.assertEqual(output, "audio.assemblyai.mapped")

    def test_output_path_generic_fallback(self):
        """Test output path for non-.json files."""
        input_path = "data.txt"
        output = mapper.generate_output_path(input_path, extension='.json')

        self.assertEqual(output, "data.mapped.json")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_replace_with_non_string_speaker(self):
        """Test that non-string speaker values are preserved."""
        json_obj = {
            "utterances": [
                {"speaker": None, "text": "Test"},  # null speaker
                {"speaker": 123, "text": "Test2"}   # numeric speaker
            ]
        }
        speaker_map = {"A": "Alice"}
        result = mapper.replace_speakers_recursive(json_obj, speaker_map)

        # Non-strings should be preserved unchanged
        self.assertIsNone(result["utterances"][0]["speaker"])
        self.assertEqual(result["utterances"][1]["speaker"], 123)

    def test_detect_with_non_string_values(self):
        """Test detection ignores non-string speaker values."""
        json_obj = {
            "utterances": [
                {"speaker": "A", "text": "Valid"},
                {"speaker": None, "text": "Null"},
                {"speaker": 123, "text": "Number"}
            ]
        }
        speakers = mapper.detect_speakers_in_json(json_obj)

        # Should only detect string values
        self.assertEqual(speakers, {"A"})

    def test_empty_json(self):
        """Test with empty JSON object."""
        json_obj = {}
        speakers = mapper.detect_speakers_in_json(json_obj)
        self.assertEqual(speakers, set())

        speaker_map = {"A": "Alice"}
        result = mapper.replace_speakers_recursive(json_obj, speaker_map)
        self.assertEqual(result, {})

    def test_list_at_root(self):
        """Test with list at JSON root."""
        json_obj = [
            {"speaker": "A", "text": "Test1"},
            {"speaker": "B", "text": "Test2"}
        ]
        speakers = mapper.detect_speakers_in_json(json_obj)
        self.assertEqual(speakers, {"A", "B"})

        speaker_map = {"A": "Alice", "B": "Bob"}
        result = mapper.replace_speakers_recursive(json_obj, speaker_map)
        self.assertEqual(result[0]["speaker"], "Alice")
        self.assertEqual(result[1]["speaker"], "Bob")


class TestInteractiveMode(unittest.TestCase):
    """Test interactive mapping mode."""

    @patch('builtins.input')
    def test_interactive_mapping(self, mock_input):
        """Test interactive prompting."""
        # Mock user inputs
        mock_input.side_effect = ["Alice Anderson", "Bob Smith", ""]  # Empty = skip

        detected = {"A", "B", "C"}
        args = Mock(verbose=0, quiet=False)

        speaker_map = mapper.prompt_interactive_mapping(detected, args)

        self.assertEqual(speaker_map, {
            "A": "Alice Anderson",
            "B": "Bob Smith"
        })
        # C was skipped (empty input)
        self.assertNotIn("C", speaker_map)

    @patch('builtins.input')
    def test_interactive_all_skip(self, mock_input):
        """Test interactive with all speakers skipped."""
        mock_input.side_effect = ["", "", ""]  # All empty

        detected = {"A", "B", "C"}
        args = Mock(verbose=0, quiet=False)

        speaker_map = mapper.prompt_interactive_mapping(detected, args)

        self.assertEqual(speaker_map, {})


class TestLLMDetection(unittest.TestCase):
    """Test LLM detection functionality."""

    def test_extract_transcript_sample(self):
        """Test transcript sample extraction."""
        json_obj = {
            "utterances": [
                {"speaker": "A", "text": "Hello, I'm Alice"},
                {"speaker": "B", "text": "Hi Alice, I'm Bob"},
                {"speaker": "A", "text": "Nice to meet you"},
            ]
        }

        sample = mapper.extract_transcript_sample(json_obj, max_utterances=10)

        self.assertIn("Speaker A: Hello, I'm Alice", sample)
        self.assertIn("Speaker B: Hi Alice, I'm Bob", sample)

    def test_has_proper_nouns(self):
        """Test proper noun detection."""
        self.assertTrue(mapper.has_proper_nouns("Hello Alice, how are you?"))
        self.assertTrue(mapper.has_proper_nouns("My name is Bob Smith"))
        self.assertFalse(mapper.has_proper_nouns("this is all lowercase"))
        self.assertFalse(mapper.has_proper_nouns("This is a sentence."))  # Sentence start doesn't count

    def test_instructor_not_available(self):
        """Test that proper error is raised when instructor not available."""
        # Skip test if instructor IS available
        if mapper.INSTRUCTOR_AVAILABLE:
            self.skipTest("Instructor is installed, skipping unavailable test")

        args = Mock(verbose=0, quiet=False)

        with self.assertRaises(RuntimeError) as context:
            mapper.detect_speakers_llm(
                "openai/gpt-4o-mini",
                "test transcript",
                ["A", "B"],
                args=args
            )

        self.assertIn("Instructor library not available", str(context.exception))

    def test_speaker_detection_model_available(self):
        """Test SpeakerDetection Pydantic model if Instructor available."""
        if not mapper.INSTRUCTOR_AVAILABLE:
            self.skipTest("Instructor library not available")

        # Create instance
        detection = mapper.SpeakerDetection(
            speakers={"A": "Alice", "B": "Bob"},
            confidence="high",
            reasoning="Names mentioned in conversation"
        )

        self.assertEqual(detection.speakers, {"A": "Alice", "B": "Bob"})
        self.assertEqual(detection.confidence, "high")
        self.assertEqual(detection.reasoning, "Names mentioned in conversation")

    def test_transcript_sample_max_utterances(self):
        """Test transcript sampling respects max_utterances limit."""
        # Create many utterances
        utterances = [
            {"speaker": "A", "text": f"Utterance {i}"}
            for i in range(50)
        ]
        json_obj = {"utterances": utterances}

        sample = mapper.extract_transcript_sample(json_obj, max_utterances=10)

        # Count lines in sample
        lines = [line for line in sample.split('\n') if line.strip()]
        self.assertLessEqual(len(lines), 10)

    def test_transcript_sample_all_speakers_represented(self):
        """Test that sampling includes all speakers."""
        # Create utterances with speakers scattered throughout
        utterances = [
            {"speaker": "A", "text": "First speaker A"},
            {"speaker": "B", "text": "First speaker B"},
        ]
        # Add many more A's to push B out of first N
        for i in range(20):
            utterances.append({"speaker": "A", "text": f"Speaker A utterance {i}"})

        # Add a C at the end
        utterances.append({"speaker": "C", "text": "Only speaker C"})

        json_obj = {"utterances": utterances}

        sample = mapper.extract_transcript_sample(json_obj, max_utterances=15)

        # All speakers should be represented
        self.assertIn("Speaker A:", sample)
        self.assertIn("Speaker B:", sample)
        self.assertIn("Speaker C:", sample)

    def test_transcript_sample_empty_utterances(self):
        """Test empty utterances returns empty string."""
        json_obj = {"utterances": []}
        sample = mapper.extract_transcript_sample(json_obj)
        self.assertEqual(sample, "")

    def test_transcript_sample_proper_noun_detection(self):
        """Test that proper nouns influence sampling."""
        utterances = [
            {"speaker": "A", "text": "first utterance no names"},
            {"speaker": "A", "text": "second utterance no names"},
            {"speaker": "A", "text": "third utterance no names"},
            {"speaker": "A", "text": "fourth utterance no names"},
            {"speaker": "A", "text": "fifth utterance no names"},
            {"speaker": "A", "text": "sixth utterance no names"},
            {"speaker": "A", "text": "seventh utterance no names"},
            {"speaker": "A", "text": "eighth utterance no names"},
            {"speaker": "A", "text": "ninth utterance no names"},
            {"speaker": "A", "text": "tenth utterance no names"},
            {"speaker": "A", "text": "eleventh with Alice"},  # Has proper noun
            {"speaker": "A", "text": "twelfth with Bob"},     # Has proper noun
        ]

        json_obj = {"utterances": utterances}

        # With small max, should prioritize those with proper nouns
        sample = mapper.extract_transcript_sample(json_obj, max_utterances=12)

        # Should include the ones with names
        self.assertIn("Alice", sample)
        self.assertIn("Bob", sample)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
