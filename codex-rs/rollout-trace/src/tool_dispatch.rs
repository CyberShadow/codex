//! Hot-path helpers for recording canonical tool dispatch boundaries.
//!
//! Core owns tool routing and result conversion. The trace crate owns the raw
//! event schema, payload shape, and no-op behavior, so core only adapts its
//! domain objects into the small request/result structs defined here.

use std::fmt::Display;
use std::sync::Arc;

use codex_protocol::models::ResponseInputItem;
use serde::Serialize;
use serde_json::Value as JsonValue;
use tracing::warn;

use crate::model::AgentThreadId;
use crate::model::CodeModeRuntimeToolId;
use crate::model::CodexTurnId;
use crate::model::ExecutionStatus;
use crate::model::ModelVisibleCallId;
use crate::model::ToolCallId;
use crate::model::ToolCallKind;
use crate::model::ToolCallSummary;
use crate::payload::RawPayloadKind;
use crate::payload::RawPayloadRef;
use crate::raw_event::RawToolCallRequester;
use crate::raw_event::RawTraceEventContext;
use crate::raw_event::RawTraceEventPayload;
use crate::writer::TraceWriter;

/// No-op capable trace handle for one resolved tool dispatch.
#[derive(Clone, Debug)]
pub struct ToolDispatchTraceContext {
    state: ToolDispatchTraceContextState,
}

#[derive(Clone, Debug)]
enum ToolDispatchTraceContextState {
    Disabled,
    Enabled(EnabledToolDispatchTraceContext),
}

#[derive(Clone, Debug)]
struct EnabledToolDispatchTraceContext {
    writer: Arc<TraceWriter>,
    thread_id: AgentThreadId,
    codex_turn_id: CodexTurnId,
    tool_call_id: ToolCallId,
}

/// Request data for the canonical Codex tool boundary.
pub struct ToolDispatchStart {
    pub thread_id: AgentThreadId,
    pub codex_turn_id: CodexTurnId,
    pub tool_call_id: ToolCallId,
    pub tool_name: String,
    pub tool_namespace: Option<String>,
    pub requester: ToolDispatchRequester,
    pub kind: ToolCallKind,
    pub label: String,
    pub input_preview: Option<String>,
    pub payload: JsonValue,
}

/// Runtime source that caused a dispatch-level tool call.
pub enum ToolDispatchRequester {
    Model {
        model_visible_call_id: ModelVisibleCallId,
    },
    CodeCell {
        runtime_cell_id: String,
        runtime_tool_call_id: CodeModeRuntimeToolId,
    },
}

/// Result data returned from a dispatch-level tool call.
#[derive(Serialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ToolDispatchResult {
    DirectResponse { response_item: ResponseInputItem },
    CodeModeResponse { value: JsonValue },
}

/// Raw invocation payload for the canonical Codex tool boundary.
#[derive(Serialize)]
struct DispatchedToolTraceRequest<'a> {
    tool_name: &'a str,
    tool_namespace: Option<&'a str>,
    payload: &'a JsonValue,
}

/// Raw response payload for dispatch-level tool trace events.
#[derive(Serialize)]
#[serde(rename_all = "snake_case", tag = "type")]
enum DispatchedToolTraceResponse<'a> {
    DirectResponse {
        response_item: &'a ResponseInputItem,
    },
    CodeModeResponse {
        value: &'a JsonValue,
    },
    Error {
        error: String,
    },
}

impl ToolDispatchTraceContext {
    /// Builds a context that accepts trace calls and records nothing.
    pub fn disabled() -> Self {
        Self {
            state: ToolDispatchTraceContextState::Disabled,
        }
    }

    /// Starts one dispatch-level lifecycle and returns the handle for its result.
    pub fn start(writer: Arc<TraceWriter>, start: ToolDispatchStart) -> Self {
        let context = EnabledToolDispatchTraceContext {
            writer,
            thread_id: start.thread_id.clone(),
            codex_turn_id: start.codex_turn_id.clone(),
            tool_call_id: start.tool_call_id.clone(),
        };
        record_started(&context, start);
        Self {
            state: ToolDispatchTraceContextState::Enabled(context),
        }
    }

    /// Records the caller-facing successful or failed tool result.
    pub fn record_completed(&self, status: ExecutionStatus, result: ToolDispatchResult) {
        let ToolDispatchTraceContextState::Enabled(context) = &self.state else {
            return;
        };
        let response = match &result {
            ToolDispatchResult::DirectResponse { response_item } => {
                DispatchedToolTraceResponse::DirectResponse { response_item }
            }
            ToolDispatchResult::CodeModeResponse { value } => {
                DispatchedToolTraceResponse::CodeModeResponse { value }
            }
        };
        append_tool_call_ended(context, status, &response);
    }

    /// Records a dispatch failure before the tool produced a normal result payload.
    pub fn record_failed(&self, error: impl Display) {
        let ToolDispatchTraceContextState::Enabled(context) = &self.state else {
            return;
        };
        append_tool_call_ended(
            context,
            ExecutionStatus::Failed,
            &DispatchedToolTraceResponse::Error {
                error: error.to_string(),
            },
        );
    }
}

fn record_started(context: &EnabledToolDispatchTraceContext, start: ToolDispatchStart) {
    let request = DispatchedToolTraceRequest {
        tool_name: start.tool_name.as_str(),
        tool_namespace: start.tool_namespace.as_deref(),
        payload: &start.payload,
    };
    let request_payload =
        write_json_payload_best_effort(&context.writer, RawPayloadKind::ToolInvocation, &request);
    let (model_visible_call_id, code_mode_runtime_tool_id, requester) =
        requester_fields(start.requester);

    append_with_context_best_effort(
        context,
        RawTraceEventPayload::ToolCallStarted {
            tool_call_id: context.tool_call_id.clone(),
            model_visible_call_id,
            code_mode_runtime_tool_id,
            requester,
            kind: start.kind,
            summary: ToolCallSummary::Generic {
                label: start.label,
                input_preview: start.input_preview,
                output_preview: None,
            },
            invocation_payload: request_payload,
        },
    );
}

fn requester_fields(
    requester: ToolDispatchRequester,
) -> (
    Option<ModelVisibleCallId>,
    Option<CodeModeRuntimeToolId>,
    RawToolCallRequester,
) {
    match requester {
        ToolDispatchRequester::Model {
            model_visible_call_id,
        } => (
            Some(model_visible_call_id),
            None,
            RawToolCallRequester::Model,
        ),
        ToolDispatchRequester::CodeCell {
            runtime_cell_id,
            runtime_tool_call_id,
        } => (
            None,
            Some(runtime_tool_call_id),
            RawToolCallRequester::CodeCell { runtime_cell_id },
        ),
    }
}

fn append_tool_call_ended(
    context: &EnabledToolDispatchTraceContext,
    status: ExecutionStatus,
    response: &DispatchedToolTraceResponse<'_>,
) {
    let response_payload =
        write_json_payload_best_effort(&context.writer, RawPayloadKind::ToolResult, response);
    append_with_context_best_effort(
        context,
        RawTraceEventPayload::ToolCallEnded {
            tool_call_id: context.tool_call_id.clone(),
            status,
            result_payload: response_payload,
        },
    );
}

fn write_json_payload_best_effort(
    writer: &TraceWriter,
    kind: RawPayloadKind,
    payload: &impl Serialize,
) -> Option<RawPayloadRef> {
    match writer.write_json_payload(kind, payload) {
        Ok(payload_ref) => Some(payload_ref),
        Err(err) => {
            warn!("failed to write rollout trace payload: {err:#}");
            None
        }
    }
}

fn append_with_context_best_effort(
    context: &EnabledToolDispatchTraceContext,
    payload: RawTraceEventPayload,
) {
    let event_context = RawTraceEventContext {
        thread_id: Some(context.thread_id.clone()),
        codex_turn_id: Some(context.codex_turn_id.clone()),
    };
    if let Err(err) = context.writer.append_with_context(event_context, payload) {
        warn!("failed to append rollout trace event: {err:#}");
    }
}
