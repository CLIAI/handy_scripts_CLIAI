# Speechmatics Batch Transcription API Documentation

## API Endpoints

The Speechmatics Transcription API includes the following batch job operations:

* **Create a new job**: `POST /jobs` - Create a new transcription job
* **Delete a job** - Remove an existing transcription task
* **Get job details** - Retrieve information about a specific job
* **Get the transcript** - Obtain results from a completed transcription
* **Get usage statistics** - View account utilization data
* **List all jobs** - Display all transcription tasks

## Response Status Codes

The API returns these HTTP status indicators:

* 201 (Success)
* 400 (Bad request)
* 401 (Unauthorized)
* 403 (Forbidden)
* 410 (Gone)
* 429 (Rate Limited)
* 500 (Internal Server Error)

## Production Endpoints

**Batch SaaS Regions:**

* EU1: eu1.asr.api.speechmatics.com
* US1: us1.asr.api.speechmatics.com
* AU1: au1.asr.api.speechmatics.com
* EU2 & US2: Enterprise-only failover regions

## Authentication

Your API key needs to be included in the header of all requests to the Speechmatics Jobs API.

Example:

```bash
curl -X GET "https://eu1.asr.api.speechmatics.com/v2/jobs/" \
  -H "Authorization: Bearer $API_KEY"
```

## Batch Temporary Keys

For client-exposed scenarios, generate temporary batch tokens via POST to:

```
https://mp.speechmatics.com/v1/api_keys?type=batch
```

Optional `client_ref` parameter restricts access to specific user jobs, preventing unauthorized data access.

## Overview

The Speechmatics Automatic Speech Recognition REST API is used to submit ASR jobs and receive the results. The supported job types include:

* Transcription of audio files
* Alignment of audio files with existing transcripts to add word or line timings

Speechmatics supports two different models within each language pack:

* **Standard model** - Faster processing with strong accuracy
* **Enhanced model** - Higher accuracy but slower turnaround time
