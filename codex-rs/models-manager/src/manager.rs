use super::cache::ModelsCacheManager;
use crate::collaboration_mode_presets::CollaborationModesConfig;
use crate::collaboration_mode_presets::builtin_collaboration_mode_presets;
use crate::config::ModelsManagerConfig;
use crate::model_info;
use async_trait::async_trait;
use codex_api::ModelsClient;
use codex_api::RequestTelemetry;
use codex_api::ReqwestTransport;
use codex_api::SharedAuthProvider;
use codex_api::TransportError;
use codex_api::auth_header_telemetry;
use codex_api::map_api_error;
use codex_app_server_protocol::AuthMode;
use codex_feedback::FeedbackRequestTags;
use codex_feedback::emit_feedback_request_tags_with_auth_env;
use codex_login::AuthEnvTelemetry;
use codex_login::default_client::build_reqwest_client;
use codex_otel::TelemetryAuthMode;
use codex_protocol::config_types::CollaborationModeMask;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result as CoreResult;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::openai_models::ModelPreset;
use codex_protocol::openai_models::ModelsResponse;
use codex_response_debug_context::extract_response_debug_context;
use codex_response_debug_context::telemetry_transport_error_message;
use http::HeaderMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::sync::TryLockError;
use tokio::time::timeout;
use tracing::error;
use tracing::info;
use tracing::instrument;

const MODEL_CACHE_FILE: &str = "models_cache.json";
const DEFAULT_MODEL_CACHE_TTL: Duration = Duration::from_secs(300);
const MODELS_REFRESH_TIMEOUT: Duration = Duration::from_secs(5);
const MODELS_ENDPOINT: &str = "/models";

/// Remote endpoint used by the OpenAI-compatible model manager.
///
/// Implementations own provider-specific auth and transport details. The model
/// manager owns refresh policy, cache behavior, and catalog merging; it calls
/// this endpoint only when it decides a remote refresh should happen.
#[async_trait]
pub trait ModelsEndpointClient: fmt::Debug + Send + Sync {
    /// Returns the auth mode used to filter picker visibility.
    fn auth_mode(&self) -> Option<AuthMode>;

    /// Returns whether this provider can authenticate command-scoped requests.
    fn has_command_auth(&self) -> bool;

    /// Fetches the latest remote model catalog and optional ETag.
    async fn list_models(
        &self,
        client_version: &str,
    ) -> CoreResult<(Vec<ModelInfo>, Option<String>)>;
}

/// Provider-supplied inputs for OpenAI-compatible remote model listing.
#[derive(Clone)]
pub struct OpenAiModelsManagerConfig {
    pub api_provider: codex_api::Provider,
    pub api_auth: SharedAuthProvider,
    pub auth_mode: Option<AuthMode>,
    pub has_command_auth: bool,
    pub auth_env: AuthEnvTelemetry,
}

impl fmt::Debug for OpenAiModelsManagerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAiModelsManagerConfig")
            .field("api_provider", &self.api_provider)
            .field("auth_mode", &self.auth_mode)
            .field("has_command_auth", &self.has_command_auth)
            .field("auth_env", &self.auth_env)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct OpenAiModelsEndpoint {
    config: OpenAiModelsManagerConfig,
}

#[async_trait]
impl ModelsEndpointClient for OpenAiModelsEndpoint {
    fn auth_mode(&self) -> Option<AuthMode> {
        self.config.auth_mode
    }

    fn has_command_auth(&self) -> bool {
        self.config.has_command_auth
    }

    async fn list_models(
        &self,
        client_version: &str,
    ) -> CoreResult<(Vec<ModelInfo>, Option<String>)> {
        fetch_models(&self.config, client_version).await
    }
}

#[derive(Clone)]
struct ModelsRequestTelemetry {
    auth_mode: Option<String>,
    auth_header_attached: bool,
    auth_header_name: Option<&'static str>,
    auth_env: AuthEnvTelemetry,
}

impl RequestTelemetry for ModelsRequestTelemetry {
    fn on_request(
        &self,
        attempt: u64,
        status: Option<http::StatusCode>,
        error: Option<&TransportError>,
        duration: Duration,
    ) {
        let success = status.is_some_and(|code| code.is_success()) && error.is_none();
        let error_message = error.map(telemetry_transport_error_message);
        let response_debug = error
            .map(extract_response_debug_context)
            .unwrap_or_default();
        let status = status.map(|status| status.as_u16());
        tracing::event!(
            target: "codex_otel.log_only",
            tracing::Level::INFO,
            event.name = "codex.api_request",
            duration_ms = %duration.as_millis(),
            http.response.status_code = status,
            success = success,
            error.message = error_message.as_deref(),
            attempt = attempt,
            endpoint = MODELS_ENDPOINT,
            auth.header_attached = self.auth_header_attached,
            auth.header_name = self.auth_header_name,
            auth.env_openai_api_key_present = self.auth_env.openai_api_key_env_present,
            auth.env_codex_api_key_present = self.auth_env.codex_api_key_env_present,
            auth.env_codex_api_key_enabled = self.auth_env.codex_api_key_env_enabled,
            auth.env_provider_key_name = self.auth_env.provider_env_key_name.as_deref(),
            auth.env_provider_key_present = self.auth_env.provider_env_key_present,
            auth.env_refresh_token_url_override_present = self.auth_env.refresh_token_url_override_present,
            auth.request_id = response_debug.request_id.as_deref(),
            auth.cf_ray = response_debug.cf_ray.as_deref(),
            auth.error = response_debug.auth_error.as_deref(),
            auth.error_code = response_debug.auth_error_code.as_deref(),
            auth.mode = self.auth_mode.as_deref(),
        );
        tracing::event!(
            target: "codex_otel.trace_safe",
            tracing::Level::INFO,
            event.name = "codex.api_request",
            duration_ms = %duration.as_millis(),
            http.response.status_code = status,
            success = success,
            error.message = error_message.as_deref(),
            attempt = attempt,
            endpoint = MODELS_ENDPOINT,
            auth.header_attached = self.auth_header_attached,
            auth.header_name = self.auth_header_name,
            auth.env_openai_api_key_present = self.auth_env.openai_api_key_env_present,
            auth.env_codex_api_key_present = self.auth_env.codex_api_key_env_present,
            auth.env_codex_api_key_enabled = self.auth_env.codex_api_key_env_enabled,
            auth.env_provider_key_name = self.auth_env.provider_env_key_name.as_deref(),
            auth.env_provider_key_present = self.auth_env.provider_env_key_present,
            auth.env_refresh_token_url_override_present = self.auth_env.refresh_token_url_override_present,
            auth.request_id = response_debug.request_id.as_deref(),
            auth.cf_ray = response_debug.cf_ray.as_deref(),
            auth.error = response_debug.auth_error.as_deref(),
            auth.error_code = response_debug.auth_error_code.as_deref(),
            auth.mode = self.auth_mode.as_deref(),
        );
        emit_feedback_request_tags_with_auth_env(
            &FeedbackRequestTags {
                endpoint: MODELS_ENDPOINT,
                auth_header_attached: self.auth_header_attached,
                auth_header_name: self.auth_header_name,
                auth_mode: self.auth_mode.as_deref(),
                auth_retry_after_unauthorized: None,
                auth_recovery_mode: None,
                auth_recovery_phase: None,
                auth_connection_reused: None,
                auth_request_id: response_debug.request_id.as_deref(),
                auth_cf_ray: response_debug.cf_ray.as_deref(),
                auth_error: response_debug.auth_error.as_deref(),
                auth_error_code: response_debug.auth_error_code.as_deref(),
                auth_recovery_followup_success: None,
                auth_recovery_followup_status: None,
            },
            &self.auth_env,
        );
    }
}

/// Strategy for refreshing available models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefreshStrategy {
    /// Always fetch from the network, ignoring cache.
    Online,
    /// Only use cached data, never fetch from the network.
    Offline,
    /// Use cache if available and fresh, otherwise fetch from the network.
    OnlineIfUncached,
}

impl RefreshStrategy {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Online => "online",
            Self::Offline => "offline",
            Self::OnlineIfUncached => "online_if_uncached",
        }
    }
}

impl fmt::Display for RefreshStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

async fn fetch_models(
    config: &OpenAiModelsManagerConfig,
    client_version: &str,
) -> CoreResult<(Vec<ModelInfo>, Option<String>)> {
    let _timer =
        codex_otel::start_global_timer("codex.remote_models.fetch_update.duration_ms", &[]);
    let transport = ReqwestTransport::new(build_reqwest_client());
    let auth_telemetry = auth_header_telemetry(config.api_auth.as_ref());
    let request_telemetry: Arc<dyn RequestTelemetry> = Arc::new(ModelsRequestTelemetry {
        auth_mode: config
            .auth_mode
            .map(|mode| TelemetryAuthMode::from(mode).to_string()),
        auth_header_attached: auth_telemetry.attached,
        auth_header_name: auth_telemetry.name,
        auth_env: config.auth_env.clone(),
    });
    let client = ModelsClient::new(
        transport,
        config.api_provider.clone(),
        Arc::clone(&config.api_auth),
    )
    .with_telemetry(Some(request_telemetry));

    timeout(
        MODELS_REFRESH_TIMEOUT,
        client.list_models(client_version, HeaderMap::new()),
    )
    .await
    .map_err(|_| CodexErr::Timeout)?
    .map_err(map_api_error)
}

/// Inputs a model provider needs to construct a model manager.
#[derive(Debug)]
pub struct ModelsManagerContext {
    pub codex_home: PathBuf,
    pub config_model_catalog: Option<ModelsResponse>,
    pub collaboration_modes_config: CollaborationModesConfig,
}

type SharedModelsEndpointClient = Arc<dyn ModelsEndpointClient>;

/// Coordinates model discovery plus cached metadata on disk.
#[derive(Debug)]
pub enum ModelsManager {
    OpenAi(OpenAiModelsManager),
    Static(StaticModelsManager),
}

/// OpenAI-compatible model manager backed by bundled models, cache, and `/models`.
#[derive(Debug)]
pub struct OpenAiModelsManager {
    remote_models: RwLock<Vec<ModelInfo>>,
    collaboration_modes_config: CollaborationModesConfig,
    etag: RwLock<Option<String>>,
    cache_manager: ModelsCacheManager,
    endpoint_client: SharedModelsEndpointClient,
}

/// Static model manager backed by an authoritative in-process catalog.
#[derive(Debug)]
pub struct StaticModelsManager {
    remote_models: RwLock<Vec<ModelInfo>>,
    collaboration_modes_config: CollaborationModesConfig,
    auth_mode: Option<AuthMode>,
}

impl ModelsManager {
    /// Construct an OpenAI-compatible remote model manager.
    pub fn new_openai(
        codex_home: PathBuf,
        config: OpenAiModelsManagerConfig,
        collaboration_modes_config: CollaborationModesConfig,
    ) -> Self {
        Self::OpenAi(OpenAiModelsManager::new(
            codex_home,
            Arc::new(OpenAiModelsEndpoint { config }),
            collaboration_modes_config,
        ))
    }

    #[cfg(test)]
    fn new_openai_with_endpoint_client(
        codex_home: PathBuf,
        endpoint_client: Arc<dyn ModelsEndpointClient>,
        collaboration_modes_config: CollaborationModesConfig,
    ) -> Self {
        Self::OpenAi(OpenAiModelsManager::new(
            codex_home,
            endpoint_client,
            collaboration_modes_config,
        ))
    }

    /// Construct a static model manager from an authoritative catalog.
    pub fn new_static(
        auth_mode: Option<AuthMode>,
        model_catalog: ModelsResponse,
        collaboration_modes_config: CollaborationModesConfig,
    ) -> Self {
        Self::Static(StaticModelsManager::new(
            auth_mode,
            model_catalog,
            collaboration_modes_config,
        ))
    }

    /// List all available models, refreshing according to the specified strategy.
    ///
    /// Returns model presets sorted by priority and filtered by auth mode and visibility.
    #[instrument(
        level = "info",
        skip(self),
        fields(refresh_strategy = %refresh_strategy)
    )]
    pub async fn list_models(&self, refresh_strategy: RefreshStrategy) -> Vec<ModelPreset> {
        match self {
            Self::OpenAi(manager) => manager.list_models(refresh_strategy).await,
            Self::Static(manager) => manager.list_models(refresh_strategy).await,
        }
    }

    /// Return the active raw model catalog, refreshing according to the specified strategy.
    pub async fn raw_model_catalog(&self, refresh_strategy: RefreshStrategy) -> ModelsResponse {
        match self {
            Self::OpenAi(manager) => manager.raw_model_catalog(refresh_strategy).await,
            Self::Static(manager) => manager.raw_model_catalog(refresh_strategy).await,
        }
    }

    /// List collaboration mode presets.
    ///
    /// Returns a static set of presets seeded with the configured model.
    pub fn list_collaboration_modes(&self) -> Vec<CollaborationModeMask> {
        match self {
            Self::OpenAi(manager) => manager.list_collaboration_modes(),
            Self::Static(manager) => manager.list_collaboration_modes(),
        }
    }

    pub fn list_collaboration_modes_for_config(
        &self,
        collaboration_modes_config: CollaborationModesConfig,
    ) -> Vec<CollaborationModeMask> {
        match self {
            Self::OpenAi(manager) => {
                manager.list_collaboration_modes_for_config(collaboration_modes_config)
            }
            Self::Static(manager) => {
                manager.list_collaboration_modes_for_config(collaboration_modes_config)
            }
        }
    }

    /// Attempt to list models without blocking, using the current cached state.
    ///
    /// Returns an error if the internal lock cannot be acquired.
    pub fn try_list_models(&self) -> Result<Vec<ModelPreset>, TryLockError> {
        match self {
            Self::OpenAi(manager) => manager.try_list_models(),
            Self::Static(manager) => manager.try_list_models(),
        }
    }

    // todo(aibrahim): should be visible to core only and sent on session_configured event
    /// Get the model identifier to use, refreshing according to the specified strategy.
    ///
    /// If `model` is provided, returns it directly. Otherwise selects the default based on
    /// auth mode and available models.
    #[instrument(
        level = "info",
        skip(self, model),
        fields(
            model.provided = model.is_some(),
            refresh_strategy = %refresh_strategy
        )
    )]
    pub async fn get_default_model(
        &self,
        model: &Option<String>,
        refresh_strategy: RefreshStrategy,
    ) -> String {
        match self {
            Self::OpenAi(manager) => manager.get_default_model(model, refresh_strategy).await,
            Self::Static(manager) => manager.get_default_model(model, refresh_strategy).await,
        }
    }

    // todo(aibrahim): look if we can tighten it to pub(crate)
    /// Look up model metadata, applying remote overrides and config adjustments.
    #[instrument(level = "info", skip(self, config), fields(model = model))]
    pub async fn get_model_info(&self, model: &str, config: &ModelsManagerConfig) -> ModelInfo {
        match self {
            Self::OpenAi(manager) => manager.get_model_info(model, config).await,
            Self::Static(manager) => manager.get_model_info(model, config).await,
        }
    }

    /// Refresh models if the provided ETag differs from the cached ETag.
    ///
    /// Uses `Online` strategy to fetch latest models when ETags differ.
    pub async fn refresh_if_new_etag(&self, etag: String) {
        match self {
            Self::OpenAi(manager) => manager.refresh_if_new_etag(etag).await,
            Self::Static(manager) => manager.refresh_if_new_etag(etag).await,
        }
    }

    /// Refresh available models according to the specified strategy.
    #[cfg(test)]
    async fn refresh_available_models(&self, refresh_strategy: RefreshStrategy) -> CoreResult<()> {
        match self {
            Self::OpenAi(manager) => manager.refresh_available_models(refresh_strategy).await,
            Self::Static(manager) => manager.refresh_available_models(refresh_strategy).await,
        }
    }

    #[cfg(test)]
    async fn get_remote_models(&self) -> Vec<ModelInfo> {
        match self {
            Self::OpenAi(manager) => manager.get_remote_models().await,
            Self::Static(manager) => manager.get_remote_models().await,
        }
    }

    #[cfg(test)]
    async fn manipulate_cache_for_test<F>(&self, f: F) -> std::io::Result<()>
    where
        F: FnOnce(&mut chrono::DateTime<chrono::Utc>),
    {
        match self {
            Self::OpenAi(manager) => manager.cache_manager.manipulate_cache_for_test(f).await,
            Self::Static(_) => Err(std::io::Error::other(
                "static model manager does not use a cache",
            )),
        }
    }

    #[cfg(test)]
    async fn mutate_cache_for_test<F>(&self, f: F) -> std::io::Result<()>
    where
        F: FnOnce(&mut super::cache::ModelsCache),
    {
        match self {
            Self::OpenAi(manager) => manager.cache_manager.mutate_cache_for_test(f).await,
            Self::Static(_) => Err(std::io::Error::other(
                "static model manager does not use a cache",
            )),
        }
    }

    #[cfg(test)]
    fn set_cache_ttl_for_test(&mut self, ttl: Duration) {
        if let Self::OpenAi(manager) = self {
            manager.cache_manager.set_ttl(ttl);
        }
    }

    #[cfg(test)]
    fn build_available_models_for_test(&self, remote_models: Vec<ModelInfo>) -> Vec<ModelPreset> {
        match self {
            Self::OpenAi(manager) => {
                build_available_models(manager.endpoint_client.auth_mode(), remote_models)
            }
            Self::Static(manager) => build_available_models(manager.auth_mode, remote_models),
        }
    }

    /// Get model identifier without consulting remote state or cache.
    pub fn get_model_offline_for_tests(model: Option<&str>) -> String {
        if let Some(model) = model {
            return model.to_string();
        }
        let mut models = load_remote_models_from_file().unwrap_or_default();
        models.sort_by(|a, b| a.priority.cmp(&b.priority));
        let presets: Vec<ModelPreset> = models.into_iter().map(Into::into).collect();
        presets
            .iter()
            .find(|preset| preset.show_in_picker)
            .or_else(|| presets.first())
            .map(|preset| preset.model.clone())
            .unwrap_or_default()
    }

    /// Build `ModelInfo` without consulting remote state or cache.
    pub fn construct_model_info_offline_for_tests(
        model: &str,
        config: &ModelsManagerConfig,
    ) -> ModelInfo {
        let candidates: &[ModelInfo] = if let Some(model_catalog) = config.model_catalog.as_ref() {
            &model_catalog.models
        } else {
            &[]
        };
        construct_model_info_from_candidates(model, candidates, config)
    }
}

impl OpenAiModelsManager {
    fn new(
        codex_home: PathBuf,
        endpoint_client: SharedModelsEndpointClient,
        collaboration_modes_config: CollaborationModesConfig,
    ) -> Self {
        let cache_path = codex_home.join(MODEL_CACHE_FILE);
        let cache_manager = ModelsCacheManager::new(cache_path, DEFAULT_MODEL_CACHE_TTL);
        let remote_models = load_remote_models_from_file().unwrap_or_default();
        Self {
            remote_models: RwLock::new(remote_models),
            collaboration_modes_config,
            etag: RwLock::new(None),
            cache_manager,
            endpoint_client,
        }
    }

    async fn list_models(&self, refresh_strategy: RefreshStrategy) -> Vec<ModelPreset> {
        if let Err(err) = self.refresh_available_models(refresh_strategy).await {
            error!("failed to refresh available models: {err}");
        }
        let remote_models = self.get_remote_models().await;
        build_available_models(self.endpoint_client.auth_mode(), remote_models)
    }

    async fn raw_model_catalog(&self, refresh_strategy: RefreshStrategy) -> ModelsResponse {
        if let Err(err) = self.refresh_available_models(refresh_strategy).await {
            error!("failed to refresh available models: {err}");
        }
        ModelsResponse {
            models: self.get_remote_models().await,
        }
    }

    fn list_collaboration_modes(&self) -> Vec<CollaborationModeMask> {
        self.list_collaboration_modes_for_config(self.collaboration_modes_config)
    }

    fn list_collaboration_modes_for_config(
        &self,
        collaboration_modes_config: CollaborationModesConfig,
    ) -> Vec<CollaborationModeMask> {
        builtin_collaboration_mode_presets(collaboration_modes_config)
    }

    fn try_list_models(&self) -> Result<Vec<ModelPreset>, TryLockError> {
        let remote_models = self.try_get_remote_models()?;
        Ok(build_available_models(
            self.endpoint_client.auth_mode(),
            remote_models,
        ))
    }

    async fn get_default_model(
        &self,
        model: &Option<String>,
        refresh_strategy: RefreshStrategy,
    ) -> String {
        if let Some(model) = model.as_ref() {
            return model.to_string();
        }
        if let Err(err) = self.refresh_available_models(refresh_strategy).await {
            error!("failed to refresh available models: {err}");
        }
        let remote_models = self.get_remote_models().await;
        default_model_from_available(build_available_models(
            self.endpoint_client.auth_mode(),
            remote_models,
        ))
    }

    async fn get_model_info(&self, model: &str, config: &ModelsManagerConfig) -> ModelInfo {
        let remote_models = self.get_remote_models().await;
        construct_model_info_from_candidates(model, &remote_models, config)
    }

    async fn refresh_if_new_etag(&self, etag: String) {
        let current_etag = self.get_etag().await;
        if current_etag.clone().is_some() && current_etag.as_deref() == Some(etag.as_str()) {
            if let Err(err) = self.cache_manager.renew_cache_ttl().await {
                error!("failed to renew cache TTL: {err}");
            }
            return;
        }
        if let Err(err) = self.refresh_available_models(RefreshStrategy::Online).await {
            error!("failed to refresh available models: {err}");
        }
    }

    async fn refresh_available_models(&self, refresh_strategy: RefreshStrategy) -> CoreResult<()> {
        if !self.should_refresh_models() {
            if matches!(
                refresh_strategy,
                RefreshStrategy::Offline | RefreshStrategy::OnlineIfUncached
            ) {
                self.try_load_cache().await;
            }
            return Ok(());
        }

        match refresh_strategy {
            RefreshStrategy::Offline => {
                // Only try to load from cache, never fetch
                self.try_load_cache().await;
                Ok(())
            }
            RefreshStrategy::OnlineIfUncached => {
                // Try cache first, fall back to online if unavailable
                if self.try_load_cache().await {
                    info!("models cache: using cached models for OnlineIfUncached");
                    return Ok(());
                }
                info!("models cache: cache miss, fetching remote models");
                self.fetch_and_update_models().await
            }
            RefreshStrategy::Online => {
                // Always fetch from network
                self.fetch_and_update_models().await
            }
        }
    }

    async fn fetch_and_update_models(&self) -> CoreResult<()> {
        let client_version = crate::client_version_to_whole();
        let (models, etag) = self.endpoint_client.list_models(&client_version).await?;
        self.apply_remote_models(models.clone()).await;
        *self.etag.write().await = etag.clone();
        self.cache_manager
            .persist_cache(&models, etag, client_version)
            .await;
        Ok(())
    }

    fn should_refresh_models(&self) -> bool {
        self.endpoint_client.auth_mode() == Some(AuthMode::Chatgpt)
            || self.endpoint_client.has_command_auth()
    }

    async fn get_etag(&self) -> Option<String> {
        self.etag.read().await.clone()
    }

    /// Replace the cached remote models and rebuild the derived presets list.
    async fn apply_remote_models(&self, models: Vec<ModelInfo>) {
        let mut existing_models = load_remote_models_from_file().unwrap_or_default();
        for model in models {
            if let Some(existing_index) = existing_models
                .iter()
                .position(|existing| existing.slug == model.slug)
            {
                existing_models[existing_index] = model;
            } else {
                existing_models.push(model);
            }
        }
        *self.remote_models.write().await = existing_models;
    }

    /// Attempt to satisfy the refresh from the cache when it matches the provider and TTL.
    async fn try_load_cache(&self) -> bool {
        let _timer =
            codex_otel::start_global_timer("codex.remote_models.load_cache.duration_ms", &[]);
        let client_version = crate::client_version_to_whole();
        info!(client_version, "models cache: evaluating cache eligibility");
        let cache = match self.cache_manager.load_fresh(&client_version).await {
            Some(cache) => cache,
            None => {
                info!("models cache: no usable cache entry");
                return false;
            }
        };
        let models = cache.models.clone();
        *self.etag.write().await = cache.etag.clone();
        self.apply_remote_models(models.clone()).await;
        info!(
            models_count = models.len(),
            etag = ?cache.etag,
            "models cache: cache entry applied"
        );
        true
    }

    async fn get_remote_models(&self) -> Vec<ModelInfo> {
        self.remote_models.read().await.clone()
    }

    fn try_get_remote_models(&self) -> Result<Vec<ModelInfo>, TryLockError> {
        Ok(self.remote_models.try_read()?.clone())
    }
}

impl StaticModelsManager {
    fn new(
        auth_mode: Option<AuthMode>,
        model_catalog: ModelsResponse,
        collaboration_modes_config: CollaborationModesConfig,
    ) -> Self {
        Self {
            remote_models: RwLock::new(model_catalog.models),
            collaboration_modes_config,
            auth_mode,
        }
    }

    async fn list_models(&self, _refresh_strategy: RefreshStrategy) -> Vec<ModelPreset> {
        build_available_models(self.auth_mode, self.get_remote_models().await)
    }

    async fn raw_model_catalog(&self, _refresh_strategy: RefreshStrategy) -> ModelsResponse {
        ModelsResponse {
            models: self.get_remote_models().await,
        }
    }

    fn list_collaboration_modes(&self) -> Vec<CollaborationModeMask> {
        self.list_collaboration_modes_for_config(self.collaboration_modes_config)
    }

    fn list_collaboration_modes_for_config(
        &self,
        collaboration_modes_config: CollaborationModesConfig,
    ) -> Vec<CollaborationModeMask> {
        builtin_collaboration_mode_presets(collaboration_modes_config)
    }

    fn try_list_models(&self) -> Result<Vec<ModelPreset>, TryLockError> {
        let remote_models = self.try_get_remote_models()?;
        Ok(build_available_models(self.auth_mode, remote_models))
    }

    async fn get_default_model(
        &self,
        model: &Option<String>,
        _refresh_strategy: RefreshStrategy,
    ) -> String {
        if let Some(model) = model.as_ref() {
            return model.to_string();
        }
        default_model_from_available(build_available_models(
            self.auth_mode,
            self.get_remote_models().await,
        ))
    }

    async fn get_model_info(&self, model: &str, config: &ModelsManagerConfig) -> ModelInfo {
        let remote_models = self.get_remote_models().await;
        construct_model_info_from_candidates(model, &remote_models, config)
    }

    async fn refresh_if_new_etag(&self, _etag: String) {}

    #[cfg(test)]
    async fn refresh_available_models(&self, _refresh_strategy: RefreshStrategy) -> CoreResult<()> {
        Ok(())
    }

    async fn get_remote_models(&self) -> Vec<ModelInfo> {
        self.remote_models.read().await.clone()
    }

    fn try_get_remote_models(&self) -> Result<Vec<ModelInfo>, TryLockError> {
        Ok(self.remote_models.try_read()?.clone())
    }
}

fn load_remote_models_from_file() -> Result<Vec<ModelInfo>, std::io::Error> {
    Ok(crate::bundled_models_response()?.models)
}

fn build_available_models(
    auth_mode: Option<AuthMode>,
    mut remote_models: Vec<ModelInfo>,
) -> Vec<ModelPreset> {
    // Build picker-ready presets from the active catalog snapshot.
    remote_models.sort_by(|a, b| a.priority.cmp(&b.priority));

    let mut presets: Vec<ModelPreset> = remote_models.into_iter().map(Into::into).collect();
    let chatgpt_mode = matches!(auth_mode, Some(AuthMode::Chatgpt));
    presets = ModelPreset::filter_by_auth(presets, chatgpt_mode);

    ModelPreset::mark_default_by_picker_visibility(&mut presets);

    presets
}

fn default_model_from_available(available: Vec<ModelPreset>) -> String {
    available
        .iter()
        .find(|model| model.is_default)
        .or_else(|| available.first())
        .map(|model| model.model.clone())
        .unwrap_or_default()
}

fn find_model_by_longest_prefix(model: &str, candidates: &[ModelInfo]) -> Option<ModelInfo> {
    let mut best: Option<ModelInfo> = None;
    for candidate in candidates {
        if !model.starts_with(&candidate.slug) {
            continue;
        }
        let is_better_match = if let Some(current) = best.as_ref() {
            candidate.slug.len() > current.slug.len()
        } else {
            true
        };
        if is_better_match {
            best = Some(candidate.clone());
        }
    }
    best
}

fn find_model_by_namespaced_suffix(model: &str, candidates: &[ModelInfo]) -> Option<ModelInfo> {
    // Retry metadata lookup for a single namespaced slug like `namespace/model-name`.
    //
    // This only strips one leading namespace segment and only when the namespace is ASCII
    // alphanumeric/underscore (`\w+`) to avoid broadly matching arbitrary aliases.
    let (namespace, suffix) = model.split_once('/')?;
    if suffix.contains('/') {
        return None;
    }
    if !namespace
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return None;
    }
    find_model_by_longest_prefix(suffix, candidates)
}

fn construct_model_info_from_candidates(
    model: &str,
    candidates: &[ModelInfo],
    config: &ModelsManagerConfig,
) -> ModelInfo {
    // First use the normal longest-prefix match. If that misses, allow a narrowly scoped
    // retry for namespaced slugs like `custom/gpt-5.3-codex`.
    let remote = find_model_by_longest_prefix(model, candidates)
        .or_else(|| find_model_by_namespaced_suffix(model, candidates));
    let model_info = if let Some(remote) = remote {
        ModelInfo {
            slug: model.to_string(),
            used_fallback_model_metadata: false,
            ..remote
        }
    } else {
        model_info::model_info_from_slug(model)
    };
    model_info::with_config_overrides(model_info, config)
}

#[cfg(test)]
#[path = "manager_tests.rs"]
mod tests;
