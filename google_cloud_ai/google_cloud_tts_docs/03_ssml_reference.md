---
title: "Speech Synthesis Markup Language (SSML)"
source_url: "https://cloud.google.com/text-to-speech/docs/ssml"
jina_url: "https://r.jina.ai/https://cloud.google.com/text-to-speech/docs/ssml"
fetched_at: "2026-03-09"
api_version: "v1"
tags: ["ssml", "markup", "speech-synthesis", "phoneme", "prosody", "voice-control"]
---

# Speech Synthesis Markup Language (SSML)

You can send [Speech Synthesis Markup Language (SSML)](https://www.w3.org/TR/speech-synthesis/) in your Cloud Text-to-Speech request to allow for more customization in your audio response by providing details on pauses, and audio formatting for acronyms, dates, times, abbreviations, or text that should be censored. See the Cloud TTS [SSML tutorial](https://docs.cloud.google.com/text-to-speech/docs/ssml-tutorial) for more information and code samples.

The following shows an example of SSML markup and the Cloud TTS synthesizes the text:

```xml
<speak>
 Here are <say-as interpret-as="characters">SSML</say-as> samples.
 I can pause <break time="3s"/>.
 I can play a sound
 <audio src="https://www.example.com/MY_MP3_FILE.mp3">didn't get your MP3 audio file</audio>.
 I can speak in cardinals. Your number is <say-as interpret-as="cardinal">10</say-as>.
 Or I can speak in ordinals. You are <say-as interpret-as="ordinal">10</say-as> in line.
 Or I can even speak in digits. The digits for ten are <say-as interpret-as="characters">10</say-as>.
 I can also substitute phrases, like the <sub alias="World Wide Web Consortium">W3C</sub>.
 Finally, I can speak a paragraph with two sentences.
 <p><s>This is sentence one.</s><s>This is sentence two.</s></p>
</speak>
```

Here is the synthesized text for the example SSML document:

> Here are S S M L samples. I can pause [3 second pause]. I can play a sound [audio file plays].
> I can speak in cardinals. Your number is ten.
> Or I can speak in ordinals. You are tenth in line.
> Or I can even speak in digits. The digits for ten are one oh.
> I can also substitute phrases, like the World Wide Web Consortium.
> Finally, I can speak a paragraph with two sentences. This is sentence one. This is sentence two.

The Cloud TTS supports a subset of the available SSML tags, which are described in this topic.

For more information about how to create audio data from SSML input with the Cloud TTS, see [Creating Voice Audio Files](https://docs.cloud.google.com/text-to-speech/docs/create-audio#ssml).

## Tips for using SSML

Depending on your implementation, you may need to escape quotation marks or quotes in the SSML payload that you send to Cloud TTS. The following example shows how to format SSML input included within a JSON object.

```json
{
 "input":{
   "ssml":"<speak>The <say-as interpret-as=\"characters\">SSML</say-as> standard <break time=\"1s\"/>is defined by the <sub alias=\"World Wide Web Consortium\">W3C</sub>.</speak>"
 },
 "voice":{
   "languageCode":"en-us",
   "name":"en-US-Standard-B",
   "ssmlGender":"MALE"
 },
 "audioConfig":{
   "audioEncoding":"MP3"
 }
}
```

### Reserve characters

Avoid using SSML reserve characters in the text that is to be converted to audio. When you need to use an SSML reserve character, prevent the character from being read as code by using its escape code. The following table shows reserved SSML characters and their associated escape codes.

| Character | Escape code |
| --- | --- |
| " | `&quot;` |
| & | `&amp;` |
| ' | `&apos;` |
| < | `&lt;` |
| > | `&gt;` |

### Select a voice

You can set the voice in the [`VoiceSelectionParams`](https://docs.cloud.google.com/text-to-speech/docs/reference/rest/v1/text/synthesize#voiceselectionparams) object. See the Text-to-Speech [SSML tutorial](https://docs.cloud.google.com/text-to-speech/docs/ssml-tutorial) to see a [code sample](https://docs.cloud.google.com/text-to-speech/docs/ssml-tutorial#using_api) demonstrating use of the `VoiceSelectionParams` object.

You can use the [`<voice>`](https://cloud.google.com/text-to-speech/docs/ssml#voice) tag to read SSML in multiple voices, but you must set the `VoiceSelectionParams` name to a compatible voice:

| Requested voice type | Neural2 | Studio | Wavenet | News | Standard |
| --- | --- | --- | --- | --- | --- |
| Neural2 | Yes | Yes |  |  | Yes |
| Studio | Yes | Yes |  |  | Yes |
| Wavenet |  |  | Yes | Yes | Yes |
| Standard | Yes | Yes |  |  | Yes |
| News |  |  | Yes | Yes | Yes |

## Support for SSML elements

### `<speak>`

The root element of the SSML response.

To learn more about the `speak` element, see the [W3 specification](https://www.w3.org/TR/speech-synthesis/#edef_speak).

#### Example

```xml
<speak>
  my SSML content
</speak>
```

### `<break>`

An empty element that controls pausing or other prosodic boundaries between words. Using `<break>` between any pair of tokens is optional. If this element is not present between words, the break is automatically determined based on the linguistic context.

To learn more about the `break` element, see the [W3 specification](https://www.w3.org/TR/speech-synthesis/#S3.2.3).

#### Attributes

| Attribute | Description |
| --- | --- |
| `time` | Sets the length of the break by seconds or milliseconds (e.g. "3s" or "250ms"). |
| `strength` | Sets the strength of the output's prosodic break by relative terms. Valid values are: "x-weak", "weak", "medium", "strong", and "x-strong". The value "none" indicates that no prosodic break boundary should be outputted, which can be used to prevent a prosodic break that the processor would otherwise produce. The other values indicate monotonically non-decreasing (conceptually increasing) break strength between tokens. The stronger boundaries are typically accompanied by pauses. |

#### Example

```xml
<speak>
  Step 1, take a deep breath. <break time="200ms"/>
  Step 2, exhale.
  Step 3, take a deep breath again. <break strength="weak"/>
  Step 4, exhale.
</speak>
```

### `<say-as>`

This element lets you indicate information about the type of text construct that is contained within the element. It also helps specify the level of detail for rendering the contained text.

The `<say-as>` element has the required attribute, `interpret-as`, which determines how the value is spoken. Optional attributes `format` and `detail` may be used depending on the particular `interpret-as` value.

#### interpret-as values

* **`currency`**

  The following example is spoken as "forty two dollars and one cent". If the language attribute is omitted, it uses the current locale.

  ```xml
  <speak>
    <say-as interpret-as='currency' language='en-US'>$42.01</say-as>
  </speak>
  ```

* **`telephone`**

  See the `interpret-as='telephone'` description in the W3C SSML 1.0 [say-as attribute values](https://www.w3.org/TR/ssml-sayas/#S3.3) WG note.

  The following example is spoken as "one eight zero zero two zero two one two one two". If the "google:style" attribute is omitted, it speaks zero as letter O.

  The "google:style='zero-as-zero'" attribute currently only works in EN locales.

  ```xml
  <speak>
    <say-as interpret-as='telephone' google:style='zero-as-zero'>1800-202-1212</say-as>
  </speak>
  ```

* **`verbatim` or `spell-out`**

  The following example is spelled out letter by letter:

  ```xml
  <speak>
    <say-as interpret-as="verbatim">abcdefg</say-as>
  </speak>
  ```

* **`date`**

  The `format` attribute is a sequence of date field character codes. Supported field character codes in `format` are {`y`, `m`, `d`} for year, month, and day (of the month) respectively. If the field code appears once for year, month, or day then the number of digits expected are 4, 2, and 2 respectively. If the field code is repeated then the number of expected digits is the number of times the code is repeated. Fields in the date text may be separated by punctuation and/or spaces.

  The `detail` attribute controls the spoken form of the date. For `detail='1'` only the day fields and one of month or year fields are required, although both may be supplied. This is the default when less than all three fields are given. The spoken form is "The {ordinal day} of {month}, {year}".

  The following example is spoken as "The tenth of September, nineteen sixty":

  ```xml
  <speak>
    <say-as interpret-as="date" format="yyyymmdd" detail="1">
      1960-09-10
    </say-as>
  </speak>
  ```

  The following example is spoken as "The tenth of September":

  ```xml
  <speak>
    <say-as interpret-as="date" format="dm">10-9</say-as>
  </speak>
  ```

  For `detail='2'` the day, month, and year fields are required and this is the default when all three fields are supplied. The spoken form is "{month} {ordinal day}, {year}".

  The following example is spoken as "September tenth, nineteen sixty":

  ```xml
  <speak>
    <say-as interpret-as="date" format="dmy" detail="2">
      10-9-1960
    </say-as>
  </speak>
  ```

* **`characters`**

  The following example is spoken as "C A N":

  ```xml
  <speak>
    <say-as interpret-as="characters">can</say-as>
  </speak>
  ```

* **`cardinal`**

  The following example is spoken as "Twelve thousand three hundred forty five" (for US English) or "Twelve thousand three hundred and forty five" (for UK English):

  ```xml
  <speak>
    <say-as interpret-as="cardinal">12345</say-as>
  </speak>
  ```

* **`ordinal`**

  The following example is spoken as "First":

  ```xml
  <speak>
    <say-as interpret-as="ordinal">1</say-as>
  </speak>
  ```

* **`fraction`**

  The following example is spoken as "five and a half":

  ```xml
  <speak>
    <say-as interpret-as="fraction">5+1/2</say-as>
  </speak>
  ```

* **`expletive` or `bleep`**

  The following example comes out as a beep, as though it has been censored:

  ```xml
  <speak>
    <say-as interpret-as="expletive">censor this</say-as>
  </speak>
  ```

* **`unit`**

  Converts units to singular or plural depending on the number. The following example is spoken as "10 feet":

  ```xml
  <speak>
    <say-as interpret-as="unit">10 foot</say-as>
  </speak>
  ```

* **`time`**

  The following example is spoken as "Two thirty P.M.":

  ```xml
  <speak>
    <say-as interpret-as="time" format="hms12">2:30pm</say-as>
  </speak>
  ```

  The `format` attribute is a sequence of time field character codes. Supported field character codes in `format` are {`h`,`m`, `s`, `Z`, `12`, `24`} for hour, minute (of the hour), second (of the minute), time zone, 12-hour time, and 24-hour time respectively. If the field code appears once for hour, minute, or second then the number of digits expected are 1, 2, and 2 respectively. If the field code is repeated then the number of expected digits is the number of times the code is repeated. Fields in the time text may be separated by punctuation and/or spaces. If hour, minute, or second are not specified in the format or there are no matching digits then the field is treated as a zero value. The default `format` is "hms12".

  The `detail` attribute controls whether the spoken form of the time is 12-hour time or 24-hour time. The spoken form is 24-hour time if `detail='1'` or if `detail` is omitted and the format of the time is 24-hour time. The spoken form is 12-hour time if `detail='2'` or if `detail` is omitted and the format of the time is 12-hour time.

To learn more about the `say-as` element, see the [W3 specification](https://www.w3.org/TR/speech-synthesis/#S3.1.9).

### `<audio>`

Supports the insertion of recorded audio files and the insertion of other audio formats in conjunction with synthesized speech output.

#### Attributes

| Attribute | Required | Default | Values |
| --- | --- | --- | --- |
| `src` | yes | n/a | A URI referring to the audio media source. Supported protocol is `https`. |
| `clipBegin` | no | 0 | A TimeDesignation that is the offset from the audio source's beginning to start playback from. If this value is greater than or equal to the audio source's actual duration, then no audio is inserted. |
| `clipEnd` | no | infinity | A TimeDesignation that is the offset from the audio source's beginning to end playback at. If the audio source's actual duration is less than this value, then playback ends at that time. If `clipBegin` is greater than or equal to `clipEnd`, then no audio is inserted. |
| `speed` | no | 100% | The ratio output playback rate relative to the normal input rate expressed as a percentage. The format is a positive Real Number followed by %. The currently supported range is [50% (slow - half speed), 200% (fast - double speed)]. Values outside that range may (or may not) be adjusted to be within it. |
| `repeatCount` | no | 1, or 10 if `repeatDur` is set | A Real Number specifying how many times to insert the audio (after clipping, if any, by `clipBegin` and/or `clipEnd`). Fractional repetitions aren't supported, so the value will be rounded to the nearest integer. Zero is not a valid value and is therefore treated as being unspecified and has the default value in that case. |
| `repeatDur` | no | infinity | A TimeDesignation that is a limit on the duration of the inserted audio after the source is processed for `clipBegin`, `clipEnd`, `repeatCount`, and `speed` attributes (rather than the normal playback duration). If the duration of the processed audio is less than this value, then playback ends at that time. |
| `soundLevel` | no | +0dB | Adjust the sound level of the audio by `soundLevel` decibels. Maximum range is +/-40dB but actual range may be effectively less, and output quality may not yield good results over the entire range. |

#### Supported audio settings

* **Format: MP3 (MPEG v2)**
  * 24K samples per second
  * 24K ~ 96K bits per second, fixed rate

* **Format: Opus in Ogg**
  * 24K samples per second (super-wideband)
  * 24K - 96K bits per second, fixed rate

* **Format (deprecated): WAV (RIFF)**
  * PCM 16-bit signed, little endian
  * 24K samples per second

* **For all formats:**
  * Single channel is preferred, but stereo is acceptable.
  * 240 seconds maximum duration.
  * 5 megabyte file size limit.
  * Source URL must use HTTPS protocol.
  * UserAgent when fetching the audio is "Google-Speech-Actions".

The contents of the `<audio>` element are optional and are used if the audio file cannot be played or if the output device does not support audio.

#### Example

```xml
<speak>
 <audio src="cat_purr_close.ogg">
   <desc>a cat purring</desc>
   PURR (sound didn't load)
 </audio>
</speak>
```

### `<p>`, `<s>`

Sentence and paragraph elements.

To learn more about the `p` and `s` elements, see the [W3 specification](https://www.w3.org/TR/speech-synthesis/#edef_paragraph).

#### Example

```xml
<p><s>This is sentence one.</s><s>This is sentence two.</s></p>
```

#### Best practices

* Use `<s>...</s>` tags to wrap full sentences, especially if they contain SSML elements that change prosody (that is, `<audio>`, `<break>`, `<emphasis>`, `<par>`, `<prosody>`, `<say-as>`, `<seq>`, and `<sub>`).
* If a break in speech is intended to be long enough that you can hear it, use `<s>...</s>` tags and put that break between sentences.

### `<sub>`

Indicate that the text in the alias attribute value replaces the contained text for pronunciation.

You can also use the `sub` element to provide a simplified pronunciation of a difficult-to-read word.

#### Examples

```xml
<sub alias="World Wide Web Consortium">W3C</sub>
```

```xml
<sub alias="にっぽんばし">日本橋</sub>
```

### `<mark>`

An empty element that places a marker into the text or tag sequence. It can be used to reference a specific location in the sequence or to insert a marker into an output stream for asynchronous notification.

To learn more about the `mark` element, see the [W3 specification](https://www.w3.org/TR/speech-synthesis/#S3.3.2).

#### Example

```xml
<speak>
Go from <mark name="here"/> here, to <mark name="there"/> there!
</speak>
```

### `<prosody>`

Used to customize the pitch, speaking rate, and volume of text contained by the element. Currently the `rate`, `pitch`, and `volume` attributes are supported.

The `rate` and `volume` attributes can be set according to the [W3 specifications](https://www.w3.org/TR/speech-synthesis11/#S3.2.4). There are three options for setting the value of the `pitch` attribute:

| Option | Description |
| --- | --- |
| Relative | Specify a relative value (e.g. "low", "medium", "high", etc) where "medium" is the default pitch. |
| Semitones | Increase or decrease pitch by "N" semitones using "+Nst" or "-Nst" respectively. Note that "+/-" and "st" are required. |
| Percentage | Increase or decrease pitch by "N" percent by using "+N%" or "-N%" respectively. Note that "%" is required but "+/-" is optional. |

#### Example

The following example uses the `<prosody>` element to speak slowly at 2 semitones lower than normal:

```xml
<prosody rate="slow" pitch="-2st">Can you hear me now?</prosody>
```

### `<emphasis>`

Used to add or remove emphasis from text contained by the element. The `<emphasis>` element modifies speech similarly to `<prosody>`, but without the need to set individual speech attributes.

This element supports an optional "level" attribute with the following valid values:

* `strong`
* `moderate`
* `none`
* `reduced`

#### Example

```xml
<emphasis level="moderate">This is an important announcement</emphasis>
```

### `<par>`

A parallel media container that allows you to play multiple media elements at once. The only allowed content is a set of one or more `<par>`, `<seq>`, and `<media>` elements. The order of the `<media>` elements is not significant.

Unless a child element specifies a different begin time, the implicit begin time for the element is the same as that of the `<par>` container. If a child element has an offset value set for its **begin** or **end** attribute, the element's offset will be relative to the beginning time of the `<par>` container.

#### Example

```xml
<speak>
  <par>
    <media xml:id="question" begin="0.5s">
      <speak>Who invented the Internet?</speak>
    </media>
    <media xml:id="answer" begin="question.end+2.0s">
      <speak>The Internet was invented by cats.</speak>
    </media>
    <media begin="answer.end-0.2s" soundLevel="-6dB">
      <audio
        src="https://actions.google.com/.../cartoon_boing.ogg"/>
    </media>
    <media repeatCount="3" soundLevel="+2.28dB"
      fadeInDur="2s" fadeOutDur="0.2s">
      <audio
        src="https://actions.google.com/.../cat_purr_close.ogg"/>
    </media>
  </par>
</speak>
```

### `<seq>`

A sequential media container that allows you to play media elements one after another. The only allowed content is a set of one or more `<seq>`, `<par>`, and `<media>` elements. The order of the media elements is the order in which they are rendered.

#### Example

```xml
<speak>
  <seq>
    <media begin="0.5s">
      <speak>Who invented the Internet?</speak>
    </media>
    <media begin="2.0s">
      <speak>The Internet was invented by cats.</speak>
    </media>
    <media soundLevel="-6dB">
      <audio
        src="https://actions.google.com/.../cartoon_boing.ogg"/>
    </media>
    <media repeatCount="3" soundLevel="+2.28dB"
      fadeInDur="2s" fadeOutDur="0.2s">
      <audio
        src="https://actions.google.com/.../cat_purr_close.ogg"/>
    </media>
  </seq>
</speak>
```

### `<media>`

Represents a media layer within a `<par>` or `<seq>` element. The allowed content of a `<media>` element is an SSML `<speak>` or `<audio>` element.

#### Attributes

| Attribute | Required | Default | Values |
| --- | --- | --- | --- |
| `xml:id` | no | no value | A unique XML identifier for this element. The allowed identifier values match the regular expression `"([-_#]|\p{L}|\p{D})+"`. |
| `begin` | no | 0 | The beginning time for this media container. See Time specification below. |
| `end` | no | no value | A specification for the ending time for this media container. See Time specification below. |
| `repeatCount` | no | 1 | A Real Number specifying how many times to insert the media. Fractional repetitions aren't supported. |
| `repeatDur` | no | no value | A TimeDesignation that is a limit on the duration of the inserted media. |
| `soundLevel` | no | +0dB | Adjust the sound level of the audio by `soundLevel` decibels. Maximum range is +/-40dB. |
| `fadeInDur` | no | 0s | A TimeDesignation over which the media will fade in from silent to the optionally-specified `soundLevel`. |
| `fadeOutDur` | no | 0s | A TimeDesignation over which the media will fade out from the optionally-specified `soundLevel` until it is silent. |

#### Time specification

A time specification, used for the value of `begin` and `end` attributes of `<media>` elements, is either an offset value or a syncbase value.

* **Offset value** - Time offset value is an SMIL Timecount-value that allows values matching: `"\s*(+|-)?\s*(\d+)(\.\d+)?(h|min|s|ms)?\s*"`

* **Syncbase value** - A syncbase value is an SMIL syncbase-value that allows values matching: `"([-_#]|\p{L}|\p{D})+\.(begin|end)\s*(+|-)\s*(\d+)(\.\d+)?(h|min|s|ms)?\s*"`

### `<phoneme>`

You can use the `<phoneme>` tag to produce custom pronunciations of words inline. Cloud TTS accepts the [IPA](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet) and [X-SAMPA](https://en.wikipedia.org/wiki/X-SAMPA) phonetic alphabets. See the [phonemes page](https://docs.cloud.google.com/text-to-speech/docs/phonemes) for a list of supported languages and phonemes.

Each application of the `<phoneme>` tag directs the pronunciation of a single word:

```xml
<phoneme alphabet="ipa" ph="ˌmænɪˈtoʊbə">manitoba</phoneme>
<phoneme alphabet="x-sampa" ph='m@"hA:g@%ni:'>mahogany</phoneme>
```

#### Stress markers

There are up to three levels of stress that can be placed in a transcription:

1. **Primary stress**: Denoted with /ˈ/ in IPA and /"/ in X-SAMPA.
2. **Secondary stress**: Denoted with /ˌ/ in IPA and /%/ in X-SAMPA.
3. **Unstressed**: Not denoted with a symbol (in either notation).

Stress markers are placed at the start of each stressed syllable. For example, in US English:

| Example word | IPA | X-SAMPA |
| --- | --- | --- |
| water | ˈwɑːtɚ | "wA:t@` |
| underwater | ˌʌndɚˈwɑːtɚ | %Vnd@"wA:t@ |

#### Broad vs Narrow Transcriptions

As a general rule, keep your transcriptions more broad and phonemic in nature. For example, in US English, transcribe intervocalic /t/ (instead of using a tap):

| Example word | IPA | X-SAMPA |
| --- | --- | --- |
| butter | ˈbʌtɚ instead of ˈbʌɾɚ | "bVt@` instead of "bV4@` |

One example of voicing assimilation for /s/ in English where the assimilation should be reflected:

| Example word | IPA | X-SAMPA |
| --- | --- | --- |
| cats | ˈkæts | "k{ts |
| dogs | ˈdɑːgz instead of ˈdɑːgs | "dA:gz instead of "dA:gs |

#### Reduction

Every syllable must contain one (and only one) vowel. Avoid syllabic consonants and instead transcribe them with a reduced vowel:

| Example word | IPA | X-SAMPA |
| --- | --- | --- |
| kitten | ˈkɪtən instead of ˈkɪtn | "kIt@n instead of "kitn |
| kettle | ˈkɛtəl instead of ˈkɛtl | "kEt@l instead of "kEtl |

#### Syllabification

You can optionally specify syllable boundaries by using /./. Each syllable must contain one (and only one) vowel:

| Example word | IPA | X-SAMPA |
| --- | --- | --- |
| readability | ˌɹiː.də.ˈbɪ.lə.tiː | %r\i:.d@."bI.l@.ti: |

#### Custom pronunciation dictionary

As an alternative to providing pronunciations inline with the `phoneme` tag, provide a dictionary of custom pronunciations in the speech synthesis RPC. When the custom pronunciation dictionary is in the request, the input text will automatically be transformed with the SSML `phoneme` tag.

Example - Original Input:

```
input: {
  text: 'Hello world! It is indeed a beautiful world!',
  custom_pronunciations: {
    pronunciations: {
      phrase: 'world'
      phonetic_encoding: PHONETIC_ENCODING_IPA
      pronunciation: 'wɜːld'
    }
  }
}
```

Transformed input:

```
input: {
  ssml: '<speak>Hello <phoneme alphabet="ipa" ph="wɜːld">world</phoneme>! It is indeed a beautiful <phoneme alphabet="ipa" ph="wɜːld">world</phoneme>!</speak>'
}
```

## Durations

Cloud Text-to-Speech supports `<say-as interpret-as="duration">` to correctly read durations. For example, the following example would be verbalized as "five hours and thirty minutes":

```xml
<say-as interpret-as="duration" format="h:m">5:30</say-as>
```

The format string supports the following values:

| Abbreviation | Value |
| --- | --- |
| h | hours |
| m | minutes |
| s | seconds |
| ms | milliseconds |

### `<voice>`

The `<voice>` tag allows you to use more than one voice in a single SSML request. In the following example, the default voice is an English male voice. All words will be synthesized in this voice except for "qu'est-ce qui t'amene ici", which will be verbalized in French using a female voice.

```xml
<speak>And then she asked, <voice language="fr-FR" gender="female">qu'est-ce qui
t'amène ici</voice><break time="250ms"/> in her sweet and gentle voice.</speak>
```

Alternatively, you can use a `<voice>` tag to specify an individual voice (the **voice name** on the [supported voices page](https://docs.cloud.google.com/text-to-speech/docs/voices)) rather than specifying a `language` and/or `gender`:

```xml
<speak>The dog is friendly<voice name="fr-CA-Wavenet-B">mais la chat est
mignon</voice><break time="250ms"/> said a pet shop
owner</speak>
```

When you use the `<voice>` tag, Cloud TTS expects to receive either a `name` (the name of the voice you want to use) **or** a combination of the following attributes. All three attributes are optional but you must provide at least one if you don't provide a `name`.

* `gender`: One of "male", "female" or "neutral".
* `variant`: Used as a tiebreaker in cases where there are multiple possibilities of which voice to use based on your configuration.
* `language`: Your desired language. Only one language can be specified in a given `<voice>` tag. Specify your language in BCP-47 format.

You can also control the relative priority of each of the `gender`, `variant`, and `language` attributes using two additional tags: `required` and `ordering`.

* `required`: If an attribute is designated as `required` and not configured properly, the request will fail.
* `ordering`: Any attributes listed after an `ordering` tag are considered as preferred attributes rather than required.

Examples:

```xml
<speak>And there it was <voice language="en-GB" gender="male" required="gender"
ordering="gender language">a flying bird </voice>roaring in the skies for the
first time.</speak>
```

```xml
<speak>Today is supposed to be <voice language="en-GB" gender="female"
ordering="language gender">Sunday Funday.</voice></speak>
```

### `<lang>`

You can use `<lang>` to include text in multiple languages within the same SSML request. All languages will be synthesized in the same voice unless you use the `<voice>` tag to explicitly change the voice. The `xml:lang` string must contain the target language in BCP-47 format.

```xml
<speak>The french word for cat is <lang xml:lang="fr-FR">chat</lang></speak>
```

Cloud Text-to-Speech supports the `<lang>` tag on a best effort basis. Not all language combinations produce the same quality results. Known issues:

* Japanese with Kanji characters is not supported by the `<lang>` tag. The input is transliterated and read as Chinese characters.
* Semitic languages such as Arabic, Hebrew, and Persian are not supported by the `<lang>` tag and will result in silence. Use the `<voice>` tag to switch to a voice that speaks your desired language instead.

## SSML timepoints

The Text-to-Speech API supports the use of timepoints in your created audio data. A **timepoint** is a timestamp (in seconds, measured from the beginning of the generated audio) that corresponds to a designated point in the script. You can set a timepoint in your script using the `<mark>` tag.

There are two steps to setting a timepoint:

1. Add a `<mark>` SSML tag to the point in the script that you want a timestamp for.
2. Set [TimepointType](https://docs.cloud.google.com/text-to-speech/docs/reference/rest/v1beta1/text/synthesize#TimepointType) to `SSML_MARK`. If this field is not set, timepoints are not returned by default.

Example returning two timepoints:

```xml
<speak>Hello <mark name="timepoint_1"/> Mark. Good to <mark
name="timepoint_2"/> see you.</speak>
```

## Styles

The following voices can speak in multiple styles:

1. en-US-Neural2-F
2. en-US-Neural2-J

Use the `<google:style>` tag to control what style to use. Only use the tag around full sentences.

Example:

```xml
<speak><google:style name="lively">Hello I'm so happy today!</google:style></speak>
```

The `name` field supports the following values:

1. `apologetic`
2. `calm`
3. `empathetic`
4. `firm`
5. `lively`
