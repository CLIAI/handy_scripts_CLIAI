# Speaker Detection Eval - Audio Generation
#
# Generates multi-speaker test audio files using espeak-ng voices
# Each test case has:
#   - Enrollment audio (single speaker)
#   - Test audio (multi-speaker conversation)

AUDIO_DIR := audio
VOICES_FILE := voices.txt
TESTS_DIR := samples

# Voice configurations for testing
# Format: voice_id:friendly_name
VOICES := en-us:alice en-gb:bob en-au:charlie en-sc:diana

# Test case definitions
TEST_CASES := 001-two-speakers 002-three-speakers

.PHONY: all clean list voices

all: enrollment-audio test-audio

# List available voices
voices:
	@echo "Configured voices:"
	@for v in $(VOICES); do echo "  $$v"; done

# Generate enrollment audio (single speaker, ~10s each)
enrollment-audio: | $(AUDIO_DIR)
	@echo "Generating enrollment audio..."
	@for voice_spec in $(VOICES); do \
		voice=$$(echo $$voice_spec | cut -d: -f1); \
		name=$$(echo $$voice_spec | cut -d: -f2); \
		outfile="$(AUDIO_DIR)/enroll_$${name}.wav"; \
		if [ ! -f "$$outfile" ]; then \
			espeak-ng -v $$voice -w "$$outfile" \
				"Hello, my name is $$name. I am recording this sample for speaker enrollment. \
				This audio will be used to create my voice profile. \
				I am speaking clearly and naturally so the system can learn my voice characteristics. \
				Thank you for listening to this enrollment audio."; \
			echo "  Created: $$outfile"; \
		fi; \
	done

# Generate multi-speaker test audio
test-audio: | $(AUDIO_DIR)
	@echo "Generating test audio..."
	@$(MAKE) -s test-001-two-speakers
	@$(MAKE) -s test-002-three-speakers

# Test 001: Two speakers (alice and bob)
test-001-two-speakers: | $(AUDIO_DIR)
	@outfile="$(AUDIO_DIR)/test_001-two-speakers.wav"; \
	if [ ! -f "$$outfile" ]; then \
		echo "  Creating test_001-two-speakers..."; \
		espeak-ng -v en-us -w "$(AUDIO_DIR)/_t001_alice1.wav" "Hello Bob, how are you today?"; \
		espeak-ng -v en-gb -w "$(AUDIO_DIR)/_t001_bob1.wav" "I am doing well Alice, thank you for asking."; \
		espeak-ng -v en-us -w "$(AUDIO_DIR)/_t001_alice2.wav" "Have you finished the report we discussed?"; \
		espeak-ng -v en-gb -w "$(AUDIO_DIR)/_t001_bob2.wav" "Yes, I sent it to you this morning."; \
		espeak-ng -v en-us -w "$(AUDIO_DIR)/_t001_alice3.wav" "Perfect, I will review it right away."; \
		ffmpeg -y -v error \
			-i "$(AUDIO_DIR)/_t001_alice1.wav" \
			-i "$(AUDIO_DIR)/_t001_bob1.wav" \
			-i "$(AUDIO_DIR)/_t001_alice2.wav" \
			-i "$(AUDIO_DIR)/_t001_bob2.wav" \
			-i "$(AUDIO_DIR)/_t001_alice3.wav" \
			-filter_complex "[0][1][2][3][4]concat=n=5:v=0:a=1" \
			"$$outfile"; \
		rm -f $(AUDIO_DIR)/_t001_*.wav; \
	fi

# Test 002: Three speakers (alice, bob, charlie)
test-002-three-speakers: | $(AUDIO_DIR)
	@outfile="$(AUDIO_DIR)/test_002-three-speakers.wav"; \
	if [ ! -f "$$outfile" ]; then \
		echo "  Creating test_002-three-speakers..."; \
		espeak-ng -v en-us -w "$(AUDIO_DIR)/_t002_alice1.wav" "Welcome everyone to our meeting."; \
		espeak-ng -v en-gb -w "$(AUDIO_DIR)/_t002_bob1.wav" "Thank you Alice for organizing this."; \
		espeak-ng -v en-au -w "$(AUDIO_DIR)/_t002_charlie1.wav" "Happy to join you both today."; \
		espeak-ng -v en-us -w "$(AUDIO_DIR)/_t002_alice2.wav" "Let us begin with the first topic."; \
		espeak-ng -v en-au -w "$(AUDIO_DIR)/_t002_charlie2.wav" "I have some updates to share."; \
		espeak-ng -v en-gb -w "$(AUDIO_DIR)/_t002_bob2.wav" "Great, please go ahead Charlie."; \
		ffmpeg -y -v error \
			-i "$(AUDIO_DIR)/_t002_alice1.wav" \
			-i "$(AUDIO_DIR)/_t002_bob1.wav" \
			-i "$(AUDIO_DIR)/_t002_charlie1.wav" \
			-i "$(AUDIO_DIR)/_t002_alice2.wav" \
			-i "$(AUDIO_DIR)/_t002_charlie2.wav" \
			-i "$(AUDIO_DIR)/_t002_bob2.wav" \
			-filter_complex "[0][1][2][3][4][5]concat=n=6:v=0:a=1" \
			"$$outfile"; \
		rm -f $(AUDIO_DIR)/_t002_*.wav; \
	fi

$(AUDIO_DIR):
	@mkdir -p $(AUDIO_DIR)

list:
	@echo "Test cases:"
	@for tc in $(TEST_CASES); do echo "  $$tc"; done
	@echo ""
	@echo "Voices:"
	@for v in $(VOICES); do echo "  $$v"; done

clean:
	rm -f $(AUDIO_DIR)/*.wav
	rm -f $(AUDIO_DIR)/*.ogg
