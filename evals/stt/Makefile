SAMPLES := $(sort $(wildcard samples/*.test.txt))
SAMPLE_STEMS := $(patsubst samples/%.test.txt,%,$(SAMPLES))
VOICES := $(shell awk 'NF && $$1 !~ /^#/' voices.txt)
AUDIO_DIR := audio

OGG_FILES := $(foreach voice,$(VOICES),$(foreach stem,$(SAMPLE_STEMS),$(AUDIO_DIR)/$(stem)__$(voice).ogg))
WAV_FILES := $(foreach voice,$(VOICES),$(foreach stem,$(SAMPLE_STEMS),$(AUDIO_DIR)/$(stem)__$(voice).wav))

.PHONY: all clean-wav clean-ogg list
.SECONDARY: $(WAV_FILES)

all: $(OGG_FILES)

list:
	@echo "Samples: $(SAMPLE_STEMS)"
	@echo "Voices: $(VOICES)"
	@echo "Outputs:"; \
	printf "%s\n" $(OGG_FILES)

$(AUDIO_DIR):
	@mkdir -p $(AUDIO_DIR)

# Generate per-voice rules so the voice name is fixed in the target.
define VOICE_RULES
$(AUDIO_DIR)/%__$(1).wav: samples/%.test.txt | $(AUDIO_DIR)
	espeak-ng -v $(1) -f $$< -w $$@

$(AUDIO_DIR)/%__$(1).ogg: $(AUDIO_DIR)/%__$(1).wav
	ffmpeg -y -v error -i $$< -c:a libvorbis -qscale:a 4 $$@
endef

$(foreach voice,$(VOICES),$(eval $(call VOICE_RULES,$(voice))))

clean-wav:
	rm -f $(WAV_FILES)

clean-ogg:
	rm -f $(OGG_FILES)
