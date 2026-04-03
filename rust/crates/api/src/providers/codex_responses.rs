use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use reqwest::Response;
use serde_json::{json, Value};

use crate::error::ApiError;
use crate::types::{
    ContentBlockDelta, ContentBlockDeltaEvent, ContentBlockStartEvent, ContentBlockStopEvent,
    InputContentBlock, MessageDelta, MessageDeltaEvent, MessageRequest, MessageResponse,
    MessageStartEvent, MessageStopEvent, OutputContentBlock, StreamEvent, ToolChoice, Usage,
};

pub const DEFAULT_CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
const CODEX_ENV_VARS: &[&str] = &["CODEX_API_KEY", "OPENAI_API_KEY"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transport {
    ChatCompletions,
    CodexResponses,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedRequest {
    pub transport: Transport,
    pub requested_model: String,
    pub resolved_model: String,
    pub base_url: String,
    pub reasoning_effort: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedCredentials {
    pub api_key: String,
    pub account_id: Option<String>,
}

#[derive(Debug)]
pub struct CodexMessageStream {
    request_id: Option<String>,
    response: Response,
    parser: CodexSseParser,
    pending: VecDeque<StreamEvent>,
    done: bool,
    state: CodexStreamState,
}

impl CodexMessageStream {
    #[must_use]
    pub fn new(response: Response, request_id: Option<String>, model: String) -> Self {
        Self {
            request_id,
            response,
            parser: CodexSseParser::default(),
            pending: VecDeque::new(),
            done: false,
            state: CodexStreamState::new(model),
        }
    }

    #[must_use]
    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    pub async fn next_event(&mut self) -> Result<Option<StreamEvent>, ApiError> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Ok(Some(event));
            }

            if self.done {
                return Ok(None);
            }

            match self.response.chunk().await? {
                Some(chunk) => {
                    for event in self.parser.push(&chunk)? {
                        let ingested = self.state.ingest(event)?;
                        self.pending.extend(ingested.events);
                        if ingested.finished {
                            self.done = true;
                        }
                    }
                }
                None => {
                    self.pending.extend(self.state.finish_without_terminal_event());
                    self.done = true;
                }
            }
        }
    }
}

#[derive(Debug, Default)]
struct CodexSseParser {
    buffer: Vec<u8>,
}

impl CodexSseParser {
    fn push(&mut self, chunk: &[u8]) -> Result<Vec<CodexSseEvent>, ApiError> {
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();

        while let Some(frame) = next_sse_frame(&mut self.buffer) {
            if let Some(event) = parse_sse_frame(&frame)? {
                events.push(event);
            }
        }

        Ok(events)
    }
}

#[derive(Debug)]
struct CodexSseEvent {
    event: String,
    data: Value,
}

#[derive(Debug)]
struct IngestedEvents {
    events: Vec<StreamEvent>,
    finished: bool,
}

#[derive(Debug)]
struct CodexStreamState {
    model: String,
    message_id: String,
    next_content_index: u32,
    active_text_index: Option<u32>,
    tool_blocks: std::collections::BTreeMap<String, ToolBlockState>,
    saw_tool_use: bool,
    final_response: Option<Value>,
    message_started: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ToolBlockState {
    index: u32,
}

impl CodexStreamState {
    fn new(model: String) -> Self {
        Self {
            model,
            message_id: make_message_id(),
            next_content_index: 0,
            active_text_index: None,
            tool_blocks: std::collections::BTreeMap::new(),
            saw_tool_use: false,
            final_response: None,
            message_started: false,
        }
    }

    fn ingest(&mut self, event: CodexSseEvent) -> Result<IngestedEvents, ApiError> {
        let mut events = Vec::new();

        if !self.message_started {
            self.message_started = true;
            events.push(StreamEvent::MessageStart(MessageStartEvent {
                message: MessageResponse {
                    id: self.message_id.clone(),
                    kind: "message".to_string(),
                    role: "assistant".to_string(),
                    content: Vec::new(),
                    model: self.model.clone(),
                    stop_reason: None,
                    stop_sequence: None,
                    usage: empty_usage(),
                    request_id: None,
                },
            }));
        }

        match event.event.as_str() {
            "response.output_item.added" => {
                if event
                    .data
                    .get("item")
                    .and_then(|item| item.get("type"))
                    .and_then(Value::as_str)
                    == Some("function_call")
                {
                    events.extend(self.close_active_text_block());
                    let item = event.data.get("item").unwrap_or(&Value::Null);
                    let item_id = item
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or("function_call")
                        .to_string();
                    let tool_use_id = item
                        .get("call_id")
                        .and_then(Value::as_str)
                        .or_else(|| item.get("id").and_then(Value::as_str))
                        .map(ToOwned::to_owned)
                        .unwrap_or_else(|| format!("call_{}", self.next_content_index));
                    let index = self.next_content_index;
                    self.next_content_index += 1;
                    self.tool_blocks.insert(item_id, ToolBlockState { index });
                    self.saw_tool_use = true;

                    events.push(StreamEvent::ContentBlockStart(ContentBlockStartEvent {
                        index,
                        content_block: OutputContentBlock::ToolUse {
                            id: tool_use_id,
                            name: item
                                .get("name")
                                .and_then(Value::as_str)
                                .unwrap_or("tool")
                                .to_string(),
                            input: json!({}),
                        },
                    }));

                    if let Some(arguments) = item.get("arguments").and_then(Value::as_str) {
                        if !arguments.is_empty() {
                            events.push(StreamEvent::ContentBlockDelta(
                                ContentBlockDeltaEvent {
                                    index,
                                    delta: ContentBlockDelta::InputJsonDelta {
                                        partial_json: arguments.to_string(),
                                    },
                                },
                            ));
                        }
                    }
                }
                Ok(IngestedEvents {
                    events,
                    finished: false,
                })
            }
            "response.content_part.added" => {
                if event
                    .data
                    .get("part")
                    .and_then(|part| part.get("type"))
                    .and_then(Value::as_str)
                    == Some("output_text")
                {
                    events.extend(self.start_text_block_if_needed());
                }
                Ok(IngestedEvents {
                    events,
                    finished: false,
                })
            }
            "response.output_text.delta" => {
                events.extend(self.start_text_block_if_needed());
                if let Some(index) = self.active_text_index {
                    events.push(StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                        index,
                        delta: ContentBlockDelta::TextDelta {
                            text: event
                                .data
                                .get("delta")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string(),
                        },
                    }));
                }
                Ok(IngestedEvents {
                    events,
                    finished: false,
                })
            }
            "response.function_call_arguments.delta" => {
                let item_id = event
                    .data
                    .get("item_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if let Some(tool_block) = self.tool_blocks.get(item_id) {
                    events.push(StreamEvent::ContentBlockDelta(ContentBlockDeltaEvent {
                        index: tool_block.index,
                        delta: ContentBlockDelta::InputJsonDelta {
                            partial_json: event
                                .data
                                .get("delta")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string(),
                        },
                    }));
                }
                Ok(IngestedEvents {
                    events,
                    finished: false,
                })
            }
            "response.output_item.done" => {
                let item = event.data.get("item").unwrap_or(&Value::Null);
                match item.get("type").and_then(Value::as_str) {
                    Some("function_call") => {
                        let item_id = item
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        if let Some(tool_block) = self.tool_blocks.remove(&item_id) {
                            events.push(StreamEvent::ContentBlockStop(ContentBlockStopEvent {
                                index: tool_block.index,
                            }));
                        }
                    }
                    Some("message") => {
                        events.extend(self.close_active_text_block());
                    }
                    _ => {}
                }
                Ok(IngestedEvents {
                    events,
                    finished: false,
                })
            }
            "response.completed" | "response.incomplete" => {
                self.final_response = event.data.get("response").cloned();
                events.extend(self.finalize());
                Ok(IngestedEvents {
                    events,
                    finished: true,
                })
            }
            "response.failed" => Err(ApiError::Auth(
                event
                    .data
                    .get("response")
                    .and_then(|value| value.get("error"))
                    .and_then(|value| value.get("message"))
                    .and_then(Value::as_str)
                    .or_else(|| {
                        event.data
                            .get("error")
                            .and_then(|value| value.get("message"))
                            .and_then(Value::as_str)
                    })
                    .unwrap_or("Codex response failed")
                    .to_string(),
            )),
            _ => Ok(IngestedEvents {
                events,
                finished: false,
            }),
        }
    }

    fn finish_without_terminal_event(&mut self) -> Vec<StreamEvent> {
        self.finalize()
    }

    fn finalize(&mut self) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        events.extend(self.close_active_text_block());
        for tool_block in self.tool_blocks.values() {
            events.push(StreamEvent::ContentBlockStop(ContentBlockStopEvent {
                index: tool_block.index,
            }));
        }
        self.tool_blocks.clear();

        if self.message_started {
            let final_response = self.final_response.as_ref();
            let usage = final_response
                .and_then(response_usage)
                .unwrap_or_else(empty_usage);
            events.push(StreamEvent::MessageDelta(MessageDeltaEvent {
                delta: MessageDelta {
                    stop_reason: Some(determine_stop_reason(final_response, self.saw_tool_use)),
                    stop_sequence: None,
                },
                usage,
            }));
            events.push(StreamEvent::MessageStop(MessageStopEvent {}));
        }
        events
    }

    fn close_active_text_block(&mut self) -> Vec<StreamEvent> {
        let Some(index) = self.active_text_index.take() else {
            return Vec::new();
        };
        vec![StreamEvent::ContentBlockStop(ContentBlockStopEvent {
            index,
        })]
    }

    fn start_text_block_if_needed(&mut self) -> Vec<StreamEvent> {
        if self.active_text_index.is_some() {
            return Vec::new();
        }
        let index = self.next_content_index;
        self.next_content_index += 1;
        self.active_text_index = Some(index);
        vec![StreamEvent::ContentBlockStart(ContentBlockStartEvent {
            index,
            content_block: OutputContentBlock::Text {
                text: String::new(),
            },
        })]
    }
}

#[must_use]
pub fn resolve_request(model: &str, base_url: &str, default_base_url: &str) -> ResolvedRequest {
    let requested_model = if model.trim().is_empty() {
        "gpt-4o".to_string()
    } else {
        model.trim().to_string()
    };
    let descriptor = parse_model_descriptor(&requested_model);
    let forced_transport = parse_wire_api_transport(std::env::var("OPENAI_WIRE_API").ok().as_deref())
        .or_else(|| parse_wire_api_transport(std::env::var("OPENCLAUDE_WIRE_API").ok().as_deref()));
    let transport = forced_transport.unwrap_or_else(|| {
        if descriptor.is_codex_alias || is_codex_base_url(base_url) {
            Transport::CodexResponses
        } else {
            Transport::ChatCompletions
        }
    });
    let resolved_base_url = if transport == Transport::CodexResponses
        && base_url.trim_end_matches('/') == default_base_url.trim_end_matches('/')
    {
        DEFAULT_CODEX_BASE_URL.to_string()
    } else {
        base_url.trim_end_matches('/').to_string()
    };

    ResolvedRequest {
        transport,
        requested_model,
        resolved_model: descriptor.base_model,
        base_url: resolved_base_url,
        reasoning_effort: descriptor.reasoning_effort,
    }
}

#[must_use]
pub fn has_saved_codex_credentials() -> bool {
    load_codex_auth_json(&resolve_codex_auth_path())
        .and_then(|json| read_nested_string(&json, &codex_token_paths()))
        .is_some()
}

#[must_use]
pub fn codex_transport_forced() -> bool {
    parse_wire_api_transport(std::env::var("OPENAI_WIRE_API").ok().as_deref())
        .or_else(|| parse_wire_api_transport(std::env::var("OPENCLAUDE_WIRE_API").ok().as_deref()))
        == Some(Transport::CodexResponses)
}

pub fn resolve_credentials(fallback_api_key: &str) -> Result<ResolvedCredentials, ApiError> {
    let env_api_key = read_env_non_empty("CODEX_API_KEY")?;
    let env_account_id =
        read_env_non_empty("CODEX_ACCOUNT_ID")?.or(read_env_non_empty("CHATGPT_ACCOUNT_ID")?);

    if let Some(api_key) = env_api_key {
        return Ok(ResolvedCredentials {
            api_key,
            account_id: env_account_id,
        });
    }

    if let Some(auth_json) = load_codex_auth_json(&resolve_codex_auth_path()) {
        if let Some(api_key) = read_nested_string(&auth_json, &codex_token_paths()) {
            let account_id = env_account_id.or_else(|| {
                read_nested_string(
                    &auth_json,
                    &[
                        &["account_id"],
                        &["accountId"],
                        &["tokens", "account_id"],
                        &["tokens", "accountId"],
                        &["auth", "account_id"],
                        &["auth", "accountId"],
                    ],
                )
            });
            return Ok(ResolvedCredentials { api_key, account_id });
        }
    }

    if !fallback_api_key.is_empty() {
        return Ok(ResolvedCredentials {
            api_key: fallback_api_key.to_string(),
            account_id: env_account_id,
        });
    }

    Err(ApiError::missing_credentials("Codex Responses", CODEX_ENV_VARS))
}

#[must_use]
pub fn responses_endpoint(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.ends_with("/responses") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/responses")
    }
}

#[must_use]
pub fn build_request_payload(request: &MessageRequest, resolved: &ResolvedRequest) -> Value {
    let mut payload = json!({
        "model": resolved.resolved_model,
        "input": build_responses_input(&request.messages),
        "store": false,
        "stream": true,
    });

    if payload["input"].as_array().is_some_and(|items| items.is_empty()) {
        payload["input"] = json!([
            {
                "type": "message",
                "role": "user",
                "content": [{ "type": "input_text", "text": "" }],
            }
        ]);
    }

    if let Some(system) = request.system.as_ref().filter(|value| !value.is_empty()) {
        payload["instructions"] = Value::String(system.clone());
    }

    if let Some(tool_choice) = request
        .tool_choice
        .as_ref()
        .and_then(convert_tool_choice)
    {
        payload["tool_choice"] = tool_choice;
    }

    if let Some(tools) = request.tools.as_ref() {
        let converted_tools = convert_tools(tools);
        if !converted_tools.is_empty() {
            payload["tools"] = Value::Array(converted_tools);
            payload["parallel_tool_calls"] = Value::Bool(true);
            if payload.get("tool_choice").is_none() {
                payload["tool_choice"] = Value::String("auto".to_string());
            }
        }
    }

    if let Some(reasoning_effort) = resolved.reasoning_effort.as_ref() {
        payload["reasoning"] = json!({ "effort": reasoning_effort });
    }

    payload
}

pub async fn collect_completed_response(response: Response) -> Result<Value, ApiError> {
    let body = response.text().await?;
    let mut completed_response = None;

    for event in parse_all_sse_events(&body)? {
        if event.event == "response.failed" {
            return Err(ApiError::Auth(
                event
                    .data
                    .get("response")
                    .and_then(|value| value.get("error"))
                    .and_then(|value| value.get("message"))
                    .and_then(Value::as_str)
                    .or_else(|| {
                        event.data
                            .get("error")
                            .and_then(|value| value.get("message"))
                            .and_then(Value::as_str)
                    })
                    .unwrap_or("Codex response failed")
                    .to_string(),
            ));
        }

        if matches!(
            event.event.as_str(),
            "response.completed" | "response.incomplete"
        ) {
            completed_response = event.data.get("response").cloned();
            break;
        }
    }

    completed_response.ok_or(ApiError::InvalidSseFrame(
        "codex response ended without a completed payload",
    ))
}

pub fn normalize_response(data: &Value, fallback_model: &str) -> MessageResponse {
    let mut content = Vec::new();
    let saw_tool_use = data
        .get("output")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .any(|item| item.get("type").and_then(Value::as_str) == Some("function_call"));

    if let Some(output) = data.get("output").and_then(Value::as_array) {
        for item in output {
            match item.get("type").and_then(Value::as_str) {
                Some("message") => {
                    if let Some(parts) = item.get("content").and_then(Value::as_array) {
                        for part in parts {
                            if part.get("type").and_then(Value::as_str) == Some("output_text") {
                                content.push(OutputContentBlock::Text {
                                    text: part
                                        .get("text")
                                        .and_then(Value::as_str)
                                        .unwrap_or_default()
                                        .to_string(),
                                });
                            }
                        }
                    }
                }
                Some("function_call") => {
                    content.push(OutputContentBlock::ToolUse {
                        id: item
                            .get("call_id")
                            .and_then(Value::as_str)
                            .or_else(|| item.get("id").and_then(Value::as_str))
                            .unwrap_or_else(|| fallback_model)
                            .to_string(),
                        name: item
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("tool")
                            .to_string(),
                        input: item
                            .get("arguments")
                            .and_then(Value::as_str)
                            .map(parse_tool_arguments)
                            .unwrap_or_else(|| json!({})),
                    });
                }
                _ => {}
            }
        }
    }

    MessageResponse {
        id: data
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_else(|| fallback_model)
            .to_string(),
        kind: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: data
            .get("model")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .unwrap_or(fallback_model)
            .to_string(),
        stop_reason: Some(determine_stop_reason(Some(data), saw_tool_use)),
        stop_sequence: None,
        usage: response_usage(data).unwrap_or_else(empty_usage),
        request_id: None,
    }
}

fn build_responses_input(messages: &[crate::types::InputMessage]) -> Value {
    let mut items = Vec::new();

    for message in messages {
        match message.role.as_str() {
            "user" => {
                let mut content_parts = Vec::new();
                for block in &message.content {
                    match block {
                        InputContentBlock::Text { text } => {
                            content_parts.push(json!({
                                "type": "input_text",
                                "text": text,
                            }));
                        }
                        InputContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => {
                            items.push(json!({
                                "type": "function_call_output",
                                "call_id": tool_use_id,
                                "output": flatten_tool_result_content(content),
                            }));
                        }
                        InputContentBlock::ToolUse { .. } => {}
                    }
                }

                if !content_parts.is_empty() {
                    items.push(json!({
                        "type": "message",
                        "role": "user",
                        "content": content_parts,
                    }));
                }
            }
            "assistant" => {
                let mut content_parts = Vec::new();
                for block in &message.content {
                    match block {
                        InputContentBlock::Text { text } => {
                            content_parts.push(json!({
                                "type": "output_text",
                                "text": text,
                            }));
                        }
                        InputContentBlock::ToolUse { id, name, input } => {
                            items.push(json!({
                                "type": "function_call",
                                "id": normalize_function_call_id(id),
                                "call_id": id,
                                "name": name,
                                "arguments": input.to_string(),
                            }));
                        }
                        InputContentBlock::ToolResult { .. } => {}
                    }
                }

                if !content_parts.is_empty() {
                    items.push(json!({
                        "type": "message",
                        "role": "assistant",
                        "content": content_parts,
                    }));
                }
            }
            _ => {}
        }
    }

    Value::Array(items)
}

fn convert_tools(tools: &[crate::types::ToolDefinition]) -> Vec<Value> {
    tools
        .iter()
        .filter(|tool| tool.name != "ToolSearchTool")
        .map(|tool| {
            json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description.clone().unwrap_or_default(),
                "parameters": enforce_strict_schema(tool.input_schema.clone()),
                "strict": true,
            })
        })
        .collect()
}

fn enforce_strict_schema(schema: Value) -> Value {
    let Value::Object(mut object) = schema else {
        return schema;
    };

    object.remove("$schema");
    object.remove("propertyNames");

    if object.get("type").and_then(Value::as_str) == Some("object") {
        object.insert("additionalProperties".to_string(), Value::Bool(false));
        if let Some(Value::Object(properties)) = object.get_mut("properties") {
            let keys = properties.keys().cloned().collect::<Vec<_>>();
            for value in properties.values_mut() {
                *value = enforce_strict_schema(value.clone());
            }
            object.insert(
                "required".to_string(),
                Value::Array(keys.into_iter().map(Value::String).collect()),
            );
        } else {
            object.insert("required".to_string(), Value::Array(Vec::new()));
        }
    }

    if let Some(items) = object.get_mut("items") {
        match items {
            Value::Array(values) => {
                for value in values {
                    *value = enforce_strict_schema(value.clone());
                }
            }
            value => {
                *value = enforce_strict_schema(value.clone());
            }
        }
    }

    for key in ["anyOf", "oneOf", "allOf"] {
        if let Some(Value::Array(values)) = object.get_mut(key) {
            for value in values {
                *value = enforce_strict_schema(value.clone());
            }
        }
    }

    Value::Object(object)
}

fn convert_tool_choice(tool_choice: &ToolChoice) -> Option<Value> {
    match tool_choice {
        ToolChoice::Auto => Some(Value::String("auto".to_string())),
        ToolChoice::Any => Some(Value::String("required".to_string())),
        ToolChoice::Tool { name } => Some(json!({
            "type": "function",
            "name": name,
        })),
    }
}

fn flatten_tool_result_content(content: &[crate::types::ToolResultContentBlock]) -> String {
    content
        .iter()
        .map(|block| match block {
            crate::types::ToolResultContentBlock::Text { text } => text.clone(),
            crate::types::ToolResultContentBlock::Json { value } => value.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn parse_tool_arguments(arguments: &str) -> Value {
    serde_json::from_str(arguments).unwrap_or_else(|_| json!({ "raw": arguments }))
}

fn normalize_function_call_id(value: &str) -> String {
    if value.starts_with("fc_") {
        value.to_string()
    } else if let Some(stripped) = value.strip_prefix("call_") {
        format!("fc_{stripped}")
    } else {
        format!("fc_{value}")
    }
}

fn determine_stop_reason(response: Option<&Value>, saw_tool_use: bool) -> String {
    if saw_tool_use
        || response
            .and_then(|value| value.get("output"))
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .any(|item| item.get("type").and_then(Value::as_str) == Some("function_call"))
    {
        return "tool_use".to_string();
    }

    if response
        .and_then(|value| value.get("incomplete_details"))
        .and_then(|value| value.get("reason"))
        .and_then(Value::as_str)
        .is_some_and(|value| value.contains("max_output_tokens"))
    {
        return "max_tokens".to_string();
    }

    "end_turn".to_string()
}

fn response_usage(response: &Value) -> Option<Usage> {
    let usage = response.get("usage")?;
    Some(Usage {
        input_tokens: usage
            .get("input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
        output_tokens: usage
            .get("output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32,
    })
}

fn empty_usage() -> Usage {
    Usage {
        input_tokens: 0,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
        output_tokens: 0,
    }
}

fn parse_all_sse_events(body: &str) -> Result<Vec<CodexSseEvent>, ApiError> {
    let mut buffer = body.as_bytes().to_vec();
    let mut events = Vec::new();
    while let Some(frame) = next_sse_frame(&mut buffer) {
        if let Some(event) = parse_sse_frame(&frame)? {
            events.push(event);
        }
    }
    Ok(events)
}

fn next_sse_frame(buffer: &mut Vec<u8>) -> Option<String> {
    let separator = buffer
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|position| (position, 2))
        .or_else(|| {
            buffer
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|position| (position, 4))
        })?;

    let (position, separator_len) = separator;
    let frame = buffer.drain(..position + separator_len).collect::<Vec<_>>();
    let frame_len = frame.len().saturating_sub(separator_len);
    Some(String::from_utf8_lossy(&frame[..frame_len]).into_owned())
}

fn parse_sse_frame(frame: &str) -> Result<Option<CodexSseEvent>, ApiError> {
    let trimmed = frame.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    let mut event_name = None;
    let mut data_lines = Vec::new();
    for line in trimmed.lines() {
        if line.starts_with(':') {
            continue;
        }
        if let Some(event) = line.strip_prefix("event:") {
            event_name = Some(event.trim().to_string());
        } else if let Some(data) = line.strip_prefix("data:") {
            data_lines.push(data.trim_start());
        }
    }

    let Some(event) = event_name else {
        return Ok(None);
    };
    if data_lines.is_empty() {
        return Ok(None);
    }
    let payload = data_lines.join("\n");
    if payload == "[DONE]" {
        return Ok(None);
    }

    Ok(Some(CodexSseEvent {
        event,
        data: serde_json::from_str(&payload)?,
    }))
}

fn parse_wire_api_transport(value: Option<&str>) -> Option<Transport> {
    let normalized = value?.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "responses" | "codex_responses" => Some(Transport::CodexResponses),
        "chat" | "chat_completions" | "chat-completions" => Some(Transport::ChatCompletions),
        _ => None,
    }
}

fn is_codex_base_url(base_url: &str) -> bool {
    let trimmed = base_url.trim_end_matches('/');
    trimmed == DEFAULT_CODEX_BASE_URL
        || (trimmed.contains("chatgpt.com") && trimmed.ends_with("/backend-api/codex"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelDescriptor {
    base_model: String,
    reasoning_effort: Option<String>,
    is_codex_alias: bool,
}

fn parse_model_descriptor(model: &str) -> ModelDescriptor {
    let trimmed = model.trim();
    let (base_model, query) = trimmed
        .split_once('?')
        .map_or((trimmed, None), |(base, query)| (base.trim(), Some(query)));
    let lower = base_model.to_ascii_lowercase();

    let (resolved_model, default_reasoning, is_codex_alias) = match lower.as_str() {
        "codexplan" => ("gpt-5.4".to_string(), Some("high".to_string()), true),
        "codexspark" => ("gpt-5.3-codex-spark".to_string(), None, true),
        _ => (base_model.to_string(), None, false),
    };

    let reasoning_effort = query
        .and_then(parse_reasoning_query)
        .or(default_reasoning);

    ModelDescriptor {
        base_model: resolved_model,
        reasoning_effort,
        is_codex_alias,
    }
}

fn parse_reasoning_query(query: &str) -> Option<String> {
    for segment in query.split('&') {
        let (key, value) = segment.split_once('=')?;
        if key.trim() != "reasoning" {
            continue;
        }
        let normalized = value.trim().to_ascii_lowercase();
        if matches!(normalized.as_str(), "low" | "medium" | "high") {
            return Some(normalized);
        }
    }
    None
}

fn resolve_codex_auth_path() -> PathBuf {
    if let Ok(path) = std::env::var("CODEX_AUTH_JSON_PATH") {
        if !path.trim().is_empty() {
            return PathBuf::from(path);
        }
    }

    if let Ok(path) = std::env::var("CODEX_HOME") {
        if !path.trim().is_empty() {
            return PathBuf::from(path).join("auth.json");
        }
    }

    home_dir().join(".codex").join("auth.json")
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn load_codex_auth_json(path: &Path) -> Option<Value> {
    let raw = fs::read_to_string(path).ok()?;
    serde_json::from_str(&raw).ok()
}

fn read_nested_string(value: &Value, paths: &[&[&str]]) -> Option<String> {
    paths.iter().find_map(|path| {
        let mut current = value;
        for key in *path {
            current = current.get(*key)?;
        }
        current
            .as_str()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
    })
}

fn codex_token_paths() -> [&'static [&'static str]; 9] {
    [
        &["access_token"],
        &["accessToken"],
        &["tokens", "access_token"],
        &["tokens", "accessToken"],
        &["auth", "access_token"],
        &["auth", "accessToken"],
        &["token", "access_token"],
        &["token", "accessToken"],
        &["tokens", "id_token"],
    ]
}

fn read_env_non_empty(key: &str) -> Result<Option<String>, ApiError> {
    match std::env::var(key) {
        Ok(value) if !value.trim().is_empty() => Ok(Some(value)),
        Ok(_) | Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(ApiError::from(error)),
    }
}

fn make_message_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("msg_{nanos:x}")
}

#[cfg(test)]
mod tests {
    use super::{
        build_request_payload, determine_stop_reason, enforce_strict_schema, resolve_request,
        responses_endpoint, Transport, DEFAULT_CODEX_BASE_URL,
    };
    use crate::types::{InputMessage, MessageRequest, ToolChoice, ToolDefinition};
    use serde_json::json;

    #[test]
    fn resolves_codex_alias_to_responses_transport_and_reasoning() {
        let resolved =
            resolve_request("codexplan", "https://api.openai.com/v1", "https://api.openai.com/v1");
        assert_eq!(resolved.transport, Transport::CodexResponses);
        assert_eq!(resolved.resolved_model, "gpt-5.4");
        assert_eq!(resolved.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(resolved.base_url, DEFAULT_CODEX_BASE_URL);
    }

    #[test]
    fn request_payload_uses_responses_shape_and_strict_tools() {
        let resolved = resolve_request(
            "codexspark",
            "https://proxy.example/v1",
            "https://api.openai.com/v1",
        );
        let payload = build_request_payload(
            &MessageRequest {
                model: "codexspark".to_string(),
                max_tokens: 64,
                messages: vec![InputMessage::user_text("hi")],
                system: Some("system".to_string()),
                tools: Some(vec![ToolDefinition {
                    name: "Agent".to_string(),
                    description: Some("Spawn an agent".to_string()),
                    input_schema: json!({
                        "type": "object",
                        "properties": {
                            "prompt": { "type": "string" },
                            "model": { "type": "string" }
                        },
                        "required": ["prompt"]
                    }),
                }]),
                tool_choice: Some(ToolChoice::Auto),
                stream: false,
            },
            &resolved,
        );

        assert_eq!(payload["model"], json!("gpt-5.3-codex-spark"));
        assert_eq!(payload["input"][0]["type"], json!("message"));
        assert_eq!(payload["tool_choice"], json!("auto"));
        assert_eq!(payload["tools"][0]["strict"], json!(true));
        let required = payload["tools"][0]["parameters"]["required"]
            .as_array()
            .expect("required array");
        assert_eq!(required.len(), 2);
        assert!(required.contains(&json!("prompt")));
        assert!(required.contains(&json!("model")));
    }

    #[test]
    fn strict_schema_recurses_into_nested_objects() {
        let enforced = enforce_strict_schema(json!({
            "type": "object",
            "properties": {
                "options": {
                    "type": "object",
                    "properties": {
                        "mode": { "type": "string" }
                    }
                }
            }
        }));
        assert_eq!(enforced["additionalProperties"], json!(false));
        assert_eq!(enforced["required"], json!(["options"]));
        assert_eq!(
            enforced["properties"]["options"]["required"],
            json!(["mode"])
        );
    }

    #[test]
    fn responses_endpoint_preserves_existing_suffix() {
        assert_eq!(
            responses_endpoint("https://proxy.example/v1"),
            "https://proxy.example/v1/responses"
        );
        assert_eq!(
            responses_endpoint("https://proxy.example/v1/responses"),
            "https://proxy.example/v1/responses"
        );
    }

    #[test]
    fn stop_reason_prefers_tool_use_and_max_tokens() {
        assert_eq!(determine_stop_reason(None, true), "tool_use");
        assert_eq!(
            determine_stop_reason(
                Some(&json!({
                    "incomplete_details": { "reason": "max_output_tokens" }
                })),
                false
            ),
            "max_tokens"
        );
    }
}
