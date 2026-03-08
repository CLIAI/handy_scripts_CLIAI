---
title: "Cloud Text-to-Speech Client Libraries"
source_url: "https://cloud.google.com/text-to-speech/docs/libraries"
jina_url: "https://r.jina.ai/https://cloud.google.com/text-to-speech/docs/libraries"
fetched_at: "2026-03-09"
api_version: "general"
tags: ["client-library", "python", "nodejs", "go", "java", "cpp", "quickstart", "tts"]
---

# Cloud Text-to-Speech Client Libraries

This page shows how to get started with the Cloud Client Libraries for the Cloud Text-to-Speech API. Client libraries make it easier to access Google Cloud APIs from a supported language.

## Installation

### C++

See Setting up a C++ development environment for requirements and dependencies.

### C\#

**Visual Studio 2017+:**

```
Install-Package Google.Apis
```

**.NET Core CLI:**

```
dotnet add package Google.Apis
```

### Go

```
go get cloud.google.com/go/texttospeech/apiv1
```

### Java

**Maven** - Add to pom.xml:

```xml
<dependencyManagement>
  <dependencies>
    <dependency>
      <groupId>com.google.cloud</groupId>
      <artifactId>libraries-bom</artifactId>
      <version>26.76.0</version>
      <type>pom</type>
      <scope>import</scope>
    </dependency>
  </dependencies>
</dependencyManagement>

<dependencies>
  <dependency>
    <groupId>com.google.cloud</groupId>
    <artifactId>google-cloud-texttospeech</artifactId>
  </dependency>
</dependencies>
```

**Gradle:**

```
implementation 'com.google.cloud:google-cloud-texttospeech:2.86.0'
```

**sbt:**

```
libraryDependencies += "com.google.cloud" % "google-cloud-texttospeech" % "2.86.0"
```

IDE Plugins: Cloud Code for VS Code, Cloud Code for IntelliJ.

**Note:** Cloud Java client libraries do not currently support Android.

### Node.js

```
npm install @google-cloud/text-to-speech
```

### PHP

```
composer require google/apiclient
```

### Python

```
pip install --upgrade google-cloud-texttospeech
```

### Ruby

```
gem install google-api-client
```

## Authentication Setup

The libraries support Application Default Credentials (ADC). For local development:

1. Install Google Cloud CLI and initialize:

```
gcloud init
```

2. Create local authentication credentials:

```
gcloud auth application-default login
```

For federated identity users, sign in first with:

```
gcloud auth login --update-adc
```

## Code Examples

### C++

```cpp
#include "google/cloud/texttospeech/v1/text_to_speech_client.h"
#include <iostream>

auto constexpr kText = R"""(
Four score and seven years ago our fathers brought forth on this
continent, a new nation, conceived in Liberty, and dedicated to
the proposition that all men are created equal.)""";

int main(int argc, char* argv[]) try {
  if (argc != 1) {
    std::cerr << "Usage: " << argv[0] << "\n";
    return 1;
  }

  namespace texttospeech = ::google::cloud::texttospeech_v1;
  auto client = texttospeech::TextToSpeechClient(
      texttospeech::MakeTextToSpeechConnection());

  google::cloud::texttospeech::v1::SynthesisInput input;
  input.set_text(kText);
  google::cloud::texttospeech::v1::VoiceSelectionParams voice;
  voice.set_language_code("en-US");
  google::cloud::texttospeech::v1::AudioConfig audio;
  audio.set_audio_encoding(google::cloud::texttospeech::v1::LINEAR16);

  auto response = client.SynthesizeSpeech(input, voice, audio);
  if (!response) throw std::move(response).status();
  auto constexpr kWavHeaderSize = 48;
  auto constexpr kBytesPerSample = 2;
  auto const sample_count =
      (response->audio_content().size() - kWavHeaderSize) / kBytesPerSample;
  std::cout << "The audio has " << sample_count << " samples\n";

  return 0;
} catch (google::cloud::Status const& status) {
  std::cerr << "google::cloud::Status thrown: " << status << "\n";
  return 1;
}
```

### Go

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	texttospeech "cloud.google.com/go/texttospeech/apiv1"
	"cloud.google.com/go/texttospeech/apiv1/texttospeechpb"
)

func main() {
	ctx := context.Background()

	client, err := texttospeech.NewClient(ctx)
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	req := texttospeechpb.SynthesizeSpeechRequest{
		Input: &texttospeechpb.SynthesisInput{
			InputSource: &texttospeechpb.SynthesisInput_Text{Text: "Hello, World!"},
		},
		Voice: &texttospeechpb.VoiceSelectionParams{
			LanguageCode: "en-US",
			SsmlGender:   texttospeechpb.SsmlVoiceGender_NEUTRAL,
		},
		AudioConfig: &texttospeechpb.AudioConfig{
			AudioEncoding: texttospeechpb.AudioEncoding_MP3,
		},
	}

	resp, err := client.SynthesizeSpeech(ctx, &req)
	if err != nil {
		log.Fatal(err)
	}

	filename := "output.mp3"
	err = os.WriteFile(filename, resp.AudioContent, 0644)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Audio content written to file: %v\n", filename)
}
```

### Java

```java
import com.google.cloud.texttospeech.v1.AudioConfig;
import com.google.cloud.texttospeech.v1.AudioEncoding;
import com.google.cloud.texttospeech.v1.SsmlVoiceGender;
import com.google.cloud.texttospeech.v1.SynthesisInput;
import com.google.cloud.texttospeech.v1.SynthesizeSpeechResponse;
import com.google.cloud.texttospeech.v1.TextToSpeechClient;
import com.google.cloud.texttospeech.v1.VoiceSelectionParams;
import com.google.protobuf.ByteString;
import java.io.FileOutputStream;
import java.io.OutputStream;

public class QuickstartSample {

  public static void main(String... args) throws Exception {
    try (TextToSpeechClient textToSpeechClient = TextToSpeechClient.create()) {
      SynthesisInput input = SynthesisInput.newBuilder().setText("Hello, World!").build();

      VoiceSelectionParams voice =
          VoiceSelectionParams.newBuilder()
              .setLanguageCode("en-US")
              .setSsmlGender(SsmlVoiceGender.NEUTRAL)
              .build();

      AudioConfig audioConfig =
          AudioConfig.newBuilder().setAudioEncoding(AudioEncoding.MP3).build();

      SynthesizeSpeechResponse response =
          textToSpeechClient.synthesizeSpeech(input, voice, audioConfig);

      ByteString audioContents = response.getAudioContent();

      try (OutputStream out = new FileOutputStream("output.mp3")) {
        out.write(audioContents.toByteArray());
        System.out.println("Audio content written to file \"output.mp3\"");
      }
    }
  }
}
```

### Node.js

```javascript
const textToSpeech = require('@google-cloud/text-to-speech');
const {writeFile} = require('node:fs/promises');

const client = new textToSpeech.TextToSpeechClient();

async function quickStart() {
  const text = 'hello, world!';

  const request = {
    input: {text: text},
    voice: {languageCode: 'en-US', ssmlGender: 'NEUTRAL'},
    audioConfig: {audioEncoding: 'MP3'},
  };

  const [response] = await client.synthesizeSpeech(request);

  await writeFile('output.mp3', response.audioContent, 'binary');
  console.log('Audio content written to file: output.mp3');
}

await quickStart();
```

### Python

```python
from google.cloud import texttospeech

client = texttospeech.TextToSpeechClient()

synthesis_input = texttospeech.SynthesisInput(text="Hello, World!")

voice = texttospeech.VoiceSelectionParams(
    language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
)

audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3
)

response = client.synthesize_speech(
    input=synthesis_input, voice=voice, audio_config=audio_config
)

with open("output.mp3", "wb") as out:
    out.write(response.audio_content)
    print('Audio content written to file "output.mp3"')
```

## Additional Resources

For each supported language (C++, C#, Go, Java, Node.js, PHP, Python, Ruby), the following resources are available:

* API reference documentation
* Client libraries best practices
* Issue tracker
* Stack Overflow support
* Source code on GitHub
