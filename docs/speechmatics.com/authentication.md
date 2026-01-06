# Speechmatics Authentication

## API Key Creation

Navigate to Settings > API Keys page of the Speechmatics Portal to create an API key. You may need to register and sign in if you haven't already. Enter a name for your API key and then store it somewhere safe when you have generated it.

## Using Your API Key

Your API key must be used with any interaction with the Speechmatics API to authenticate to the service. Any interaction without this key will receive a HTTP 401 - Unauthorized response.

It is recommended that you store and provide access to the API key on the principle of least privilege. If you believe that your key has been compromised, please reach out to Support.

## Authentication Methods

### For Batch Transcription

Your API key needs to be included in the header of all requests to the Speechmatics Jobs API.

Example:

```bash
curl -X GET "https://eu1.asr.api.speechmatics.com/v2/jobs/" \
  -H "Authorization: Bearer $API_KEY"
```

### For Realtime Transcription

Speechmatics allows you to generate temporary keys to authenticate your requests instead of your long-lived API key. This can improve end-user experience by reducing latency, and can also reduce development effort and complexity.

For Realtime transcription, temporary keys allow the end-user to start a websocket connection with Speechmatics directly, without exposing your long-lived API key. This will reduce the latency of transcription as well as development effort.

## Temporary Keys

The value for ttl is a number that indicates for how many seconds the token will be valid, between 60 and 3600.

Because API keys are persistent, it is important to remember not to use them to authenticate on the client side. Instead, generating a short-lived JWT on the server side using your API key is recommended.

### Temporary Key Generation

**Realtime Tokens:** Generated via POST to:

```
https://mp.speechmatics.com/v1/api_keys?type=rt
```

With TTL parameter (60-86400 seconds).

**Batch Tokens:** Use `type=batch` parameter with optional `client_ref` to restrict access to specific user jobs, preventing unauthorized data access in client-exposed scenarios.

Example:

```
https://mp.speechmatics.com/v1/api_keys?type=batch
```

## Region-Specific Keys

Your API key is linked to a specific region and must be used with any interaction with the Speechmatics API to authenticate to the service.

## Supported Production Endpoints

### Batch SaaS Regions

* EU1: eu1.asr.api.speechmatics.com
* US1: us1.asr.api.speechmatics.com
* AU1: au1.asr.api.speechmatics.com
* EU2 & US2: Enterprise-only failover regions

### Realtime SaaS Regions

* EU1: eu.rt.speechmatics.com
* US1: us.rt.speechmatics.com

## Client-Side Authentication (Browser)

For browser-based applications, temporary keys (JWTs) are recommended. These reduce latency and eliminate exposure of long-lived credentials. Pass them as query parameters to WebSocket connections:

```
wss://eu.rt.speechmatics.com/v2?jwt=$TEMP_KEY
```

## Configuration

Configuration supports TTL, client reference associations, and region specification for enterprise deployments.
