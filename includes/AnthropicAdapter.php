<?php

/**
 * @file
 * Anthropic Claude adapter for the OpenAI module.
 *
 * This adapter implements AIClientInterface using Anthropic's REST API,
 * translating OpenAI-compatible method calls to Claude's native API.
 *
 * No external dependencies - uses Backdrop's built-in HTTP functions.
 *
 * @see https://docs.anthropic.com/en/api/getting-started
 */

class AnthropicAdapter implements AIClientInterface {

  /** @var string */
  protected $apiKey;

  /** @var string */
  protected $baseUrl = 'https://api.anthropic.com/v1';

  /** @var OpenAIApi|null */
  protected $api = NULL;

  /**
   * Constructor.
   *
   * @param string $apiKey
   *   Anthropic API key.
   * @param OpenAIApi|null $api
   *   Optional OpenAIApi wrapper for logging.
   */
  public function __construct($apiKey, ?OpenAIApi $api = NULL) {
    $this->apiKey = trim($apiKey);
    $this->api = $api;

    if (empty($this->apiKey)) {
      throw new \Exception('Anthropic API key is required');
    }
  }

  /**
   * {@inheritdoc}
   */
  public function getModels(): array {
    try {
      $url = $this->baseUrl . '/models';
      $options = [
        'method' => 'GET',
        'headers' => [
          'Accept' => 'application/json',
          'x-api-key' => $this->apiKey,
          'anthropic-version' => '2023-06-01',
        ],
        'timeout' => 10,
      ];

      $response = backdrop_http_request($url, $options);

      if (isset($response->code) && (int) $response->code === 200) {
        $data = json_decode($response->data, TRUE);
        if (!empty($data['data']) && is_array($data['data'])) {
          $models = [];
          foreach ($data['data'] as $item) {
            $id = $item['id'] ?? ($item['model'] ?? NULL);
            if (empty($id)) {
              continue;
            }

            if (!preg_match('/^claude/i', $id)) {
              continue;
            }

            $label = $item['display_name'] ?? $item['name'] ?? $id;
            if ($label === $id) {
              $models[$id] = $id;
            } else {
              $models[$id] = $label;
            }
          }

          if (!empty($models)) {
            asort($models);
            return $models;
          }
        }
      }
    }
    catch (\Exception $e) {
      watchdog('openai_anthropic', 'Failed to fetch Anthropic models: @error', ['@error' => $e->getMessage()], WATCHDOG_ERROR);
    }

    return [];
  }

  /**
   * {@inheritdoc}
   */
  public function completions(string $model, string $prompt, $temperature, $max_tokens = 512, bool $stream_response = FALSE) {
    $start_time = microtime(TRUE);
    try {
      // Allow other modules to alter the prompt before sending (e.g., inject site context).
      if (function_exists('backdrop_alter')) {
        $context = [
          'operation' => 'completion',
          'model' => $model,
          'provider' => 'anthropic',
        ];
        backdrop_alter('openai_prompt', $prompt, $context);
      }

      $params = [
        'model' => $model,
        'max_tokens' => (int) $max_tokens ?: 512,
        'temperature' => (float) $temperature,
        'messages' => [
          [
            'role' => 'user',
            'content' => trim($prompt),
          ],
        ],
      ];

      if ($stream_response) {
        return $this->_handleStreamingResponse($params);
      }

      $response = $this->_makeApiRequest('/messages', $params);

      // Extract text from Anthropic response
      $text = '';
      if (isset($response['content']) && is_array($response['content'])) {
        foreach ($response['content'] as $block) {
          if (isset($block['text'])) {
            $text .= $block['text'];
          }
        }
      }

      return trim($text);
    }
    catch (\Exception $e) {
      watchdog('openai_anthropic', 'Completions error: @error',
        ['@error' => $e->getMessage()], WATCHDOG_ERROR);
      throw $e;
    }
  }

  /**
   * Get models by their capability.
   */
  public function getModelsByCapability($capability): array {
    $models = $this->getModels();
    if ($capability === 'text') {
      return $models;
    }
    if ($capability === 'vision') {
      $vision_models = [];
      foreach ($models as $id => $label) {
        // Claude 3 and later generally support vision
        if (preg_match('/claude-3/i', $id)) {
          $vision_models[$id] = $label;
        }
      }
      return $vision_models;
    }
    return [];
  }

  /**
   * {@inheritdoc}
   */
  public function chat(string $model, array $messages, $temperature, $max_tokens = 1024, bool $stream_response = FALSE) {
    $start_time = microtime(TRUE);
    try {
      // Allow other modules to alter chat messages before sending (e.g., inject site context).
      if (function_exists('backdrop_alter')) {
        $context = [
          'operation' => 'chat',
          'model' => $model,
          'provider' => 'anthropic',
        ];
        backdrop_alter('openai_chat_messages', $messages, $context);
      }

      // Convert messages to Anthropic format
      $anthropic_messages = $this->_convertMessages($messages);

      $params = [
        'model' => $model,
        'max_tokens' => (int) $max_tokens ?: 1024,
        'temperature' => (float) $temperature,
        'messages' => $anthropic_messages,
      ];

      if ($stream_response) {
        return $this->_handleStreamingResponse($params);
      }

      $response = $this->_makeApiRequest('/messages', $params);

      // Extract text from Anthropic response
      $text = '';
      if (isset($response['content']) && is_array($response['content'])) {
        foreach ($response['content'] as $block) {
          if (isset($block['text'])) {
            $text .= $block['text'];
          }
        }
      }

      return trim($text);
    }
    catch (\Exception $e) {
      watchdog('openai_anthropic', 'Chat error: @error',
        ['@error' => $e->getMessage()], WATCHDOG_ERROR);
      throw $e;
    }
  }

  /**
   * {@inheritdoc}
   *
   * Anthropic does not support image generation natively.
   */
  public function images(string $model, string $prompt, string $size, string $response_format, string $quality = 'standard', string $style = 'natural', ?string $output_format = NULL) {
    watchdog('openai_anthropic',
      'Image generation is not supported by Anthropic Claude. Try OpenAI or OpenRouter for image generation.',
      [], WATCHDOG_WARNING);

    return [
      'data' => [],
      'error' => 'Image generation is not available in Anthropic Claude adapter. Use OpenAI or OpenRouter instead.',
    ];
  }

  /**
   * {@inheritdoc}
   *
   * Anthropic does not support text-to-speech natively.
   */
  public function textToSpeech(string $model, string $input, string $voice, string $response_format) {
    watchdog('openai_anthropic',
      'Text-to-speech is not supported by Anthropic Claude.',
      [], WATCHDOG_WARNING);

    return '';
  }

  /**
   * {@inheritdoc}
   *
   * Anthropic does not support speech-to-text natively.
   */
  public function speechToText(string $model, string $file, string $task = 'transcribe', $temperature = 0.4, string $response_format = 'verbose_json') {
    watchdog('openai_anthropic',
      'Speech-to-text is not supported by Anthropic Claude.',
      [], WATCHDOG_WARNING);

    return [];
  }

  /**
   * {@inheritdoc}
   *
   * Anthropic does not provide a moderation API endpoint.
   * Claude has built-in safety policies but no content moderation score.
   */
  public function moderation(string $input, string $model = 'claude-moderation'): array {
    watchdog('openai_anthropic',
      'Moderation API is not supported by Anthropic Claude. Claude has built-in safety policies.',
      [], WATCHDOG_WARNING);

    return [
      'results' => [],
      'error' => 'Moderation is not available in Anthropic Claude adapter.',
    ];
  }

  /**
   * {@inheritdoc}
   *
   * Adapter must implement AIClientInterface::embedding even if the provider
   * does not support embeddings. Return an empty array and log a warning so
   * callers receive a predictable shape and administrators can diagnose.
   */
  public function embedding(string $input, string $model, bool $log = TRUE): array {
    // Record a log in openai_log if possible to show that it was attempted
    if (isset($this->api) && method_exists($this->api, 'recordLog')) {
      $this->api->recordLog('embedding', $model, ['input' => $input], NULL, FALSE, 0, 'Anthropic does not support embeddings.', !$log);
    }
    if ($log) {
      // Only log to watchdog if it's not a probing call ($log is usually FALSE during probes)
      // and only if explicitly requested.
      watchdog('openai_anthropic', 'Embedding requested but Anthropic does not support embeddings. Returning empty array.', [], WATCHDOG_DEBUG);
    }
    return [];
  }

  /**
   * Handle streaming responses for Anthropic.
   *
   * @param array $params
   *   Parameters for the API request.
   *
   * @return StreamedResponse
   *   A streaming HTTP response.
   */
  protected function _handleStreamingResponse(array $params) {
    try {
      // Set stream flag for Anthropic API
      $params['stream'] = TRUE;

      $url = $this->baseUrl . '/messages';

      $options = [
        'method' => 'POST',
        'headers' => [
          'Content-Type' => 'application/json',
          'x-api-key' => $this->apiKey,
          'anthropic-version' => '2023-06-01',
        ],
        'data' => json_encode($params),
        'timeout' => 300,
      ];

      // Use Backdrop's streaming response mechanism
      return new class($url, $options) {
        protected $url;
        protected $options;

        public function __construct($url, $options) {
          $this->url = $url;
          $this->options = $options;
        }

        public function send() {
          $response = backdrop_http_request($this->url, $this->options);

          if (!isset($response->code) || $response->code !== 200) {
            throw new \Exception("Anthropic API error: " . $response->code);
          }

          // Process streaming response line by line
          if (isset($response->data) && is_string($response->data)) {
            $lines = explode("\n", $response->data);

            foreach ($lines as $line) {
              if (strpos($line, 'data: ') === 0) {
                $json = substr($line, 6);
                if ($json === '[DONE]') {
                  break;
                }

                $data = json_decode($json, TRUE);

                if (isset($data['delta']['type']) && $data['delta']['type'] === 'text_delta') {
                  if (isset($data['delta']['text'])) {
                    echo $data['delta']['text'];
                    @ob_flush();
                    @flush();
                  }
                }
              }
            }
          }
        }
      };
    }
    catch (\Exception $e) {
      watchdog('openai_anthropic', 'Streaming error: @error',
        ['@error' => $e->getMessage()], WATCHDOG_ERROR);
      throw $e;
    }
  }

  /**
   * Make HTTP request to Anthropic API.
   *
   * @param string $endpoint
   *   API endpoint (e.g., '/messages').
   * @param array $body
   *   Request body.
   *
   * @return array
   *   Decoded JSON response.
   */
  protected function _makeApiRequest($endpoint, array $body) {
    $url = $this->baseUrl . $endpoint;

    $options = [
      'method' => 'POST',
      'headers' => [
        'Content-Type' => 'application/json',
        'x-api-key' => $this->apiKey,
        'anthropic-version' => '2023-06-01',
      ],
      'data' => json_encode($body),
      'timeout' => 30,
    ];

    $response = backdrop_http_request($url, $options);

    if (!isset($response->code) || (int) $response->code !== 200) {
      $error_msg = isset($response->data) ? $response->data : 'Unknown error';
      throw new \Exception("Anthropic API error ({$response->code}): " . $error_msg);
    }

    return json_decode($response->data, TRUE);
  }

  /**
   * Convert OpenAI-style messages to Anthropic format.
   *
   * @param array $messages
   *   Array of messages in OpenAI format.
   *
   * @return array
   *   Converted messages for Anthropic API.
   */
  protected function _convertMessages(array $messages): array {
    $anthropic_messages = [];

    foreach ($messages as $msg) {
      $role = $msg['role'] ?? 'user';

      // Anthropic only supports 'user' and 'assistant' roles
      if ($role === 'system') {
        // System messages are handled via 'system' parameter in API, skip here
        continue;
      }

      // Convert 'assistant' role (OpenAI) to 'assistant' (Anthropic)
      $anthropic_role = ($role === 'assistant') ? 'assistant' : 'user';

      $anthropic_messages[] = [
        'role' => $anthropic_role,
        'content' => $msg['content'],
      ];
    }

    return $anthropic_messages;
  }
}
