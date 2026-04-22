use std::fmt;
use std::sync::Arc;

use codex_api::AuthProvider;
use codex_login::AuthManager;
use codex_login::CodexAuth;
use codex_login::collect_auth_env_telemetry;
use codex_model_provider_info::ModelProviderAwsAuthInfo;
use codex_model_provider_info::ModelProviderInfo;
use codex_models_manager::manager::ModelsManager;
use codex_models_manager::manager::ModelsManagerContext;
use codex_models_manager::manager::OpenAiModelsManagerConfig;
use codex_protocol::error::Result as CoreResult;
use http::HeaderMap;

use crate::amazon_bedrock::AmazonBedrockModelProvider;
use crate::auth::auth_manager_for_provider;
use crate::auth::bearer_auth_provider_from_auth;
use crate::auth::resolve_provider_auth;

/// Runtime provider abstraction used by model execution.
///
/// Implementations own provider-specific behavior for a model backend. The
/// `ModelProviderInfo` returned by `info` is the serialized/configured provider
/// metadata used by the default OpenAI-compatible implementation.
#[async_trait::async_trait]
pub trait ModelProvider: fmt::Debug + Send + Sync {
    /// Returns the configured provider metadata.
    fn info(&self) -> &ModelProviderInfo;

    /// Returns the provider-scoped auth manager, when this provider uses one.
    fn auth_manager(&self) -> Option<Arc<AuthManager>>;

    /// Returns the current provider-scoped auth value, if one is configured.
    async fn auth(&self) -> Option<CodexAuth>;

    /// Returns provider configuration adapted for the API client.
    async fn api_provider(&self) -> CoreResult<codex_api::Provider> {
        let auth = self.auth().await;
        self.info()
            .to_api_provider(auth.as_ref().map(CodexAuth::auth_mode))
    }

    /// Returns the auth provider used to attach request credentials.
    async fn api_auth(&self) -> CoreResult<codex_api::SharedAuthProvider> {
        let auth = self.auth().await;
        resolve_provider_auth(auth.as_ref(), self.info())
    }

    /// Returns the model manager implementation appropriate for this provider.
    fn models_manager(&self) -> Arc<ModelsManager>;
}

/// Shared runtime model provider handle.
pub type SharedModelProvider = Arc<dyn ModelProvider>;

/// Creates the default runtime model provider for configured provider metadata.
pub fn create_model_provider(
    provider_info: ModelProviderInfo,
    auth_manager: Option<Arc<AuthManager>>,
) -> SharedModelProvider {
    create_model_provider_with_models_context(
        provider_info,
        auth_manager,
        ModelsManagerContext {
            codex_home: std::env::temp_dir().join("codex-model-provider"),
            config_model_catalog: None,
            collaboration_modes_config: Default::default(),
        },
    )
}

/// Creates the default runtime model provider with an owned models manager.
pub fn create_model_provider_with_models_context(
    provider_info: ModelProviderInfo,
    auth_manager: Option<Arc<AuthManager>>,
    models_context: ModelsManagerContext,
) -> SharedModelProvider {
    if provider_info.is_amazon_bedrock() {
        let aws = provider_info
            .aws
            .clone()
            .unwrap_or(ModelProviderAwsAuthInfo {
                profile: None,
                region: None,
            });
        let model_catalog = models_context
            .config_model_catalog
            .unwrap_or_else(crate::amazon_bedrock::static_model_catalog);
        let models_manager = Arc::new(ModelsManager::new_static(
            /*auth_mode*/ None,
            model_catalog,
            models_context.collaboration_modes_config,
        ));
        return Arc::new(AmazonBedrockModelProvider {
            info: provider_info,
            aws,
            models_manager,
        });
    }

    let auth_manager = auth_manager_for_provider(auth_manager, &provider_info);
    let auth_mode = auth_manager
        .as_ref()
        .and_then(|manager| manager.auth_mode());
    let models_manager = match models_context.config_model_catalog {
        Some(model_catalog) => Arc::new(ModelsManager::new_static(
            auth_mode,
            model_catalog,
            models_context.collaboration_modes_config,
        )),
        None => {
            let codex_api_key_env_enabled = auth_manager
                .as_ref()
                .is_some_and(|manager| manager.codex_api_key_env_enabled());
            let api_provider = provider_info
                .to_api_provider(auth_mode)
                .unwrap_or_else(|err| panic!("model provider should build API provider: {err}"));
            let api_auth = Arc::new(DynamicProviderAuth {
                auth_manager: auth_manager.clone(),
                provider_info: provider_info.clone(),
            });
            Arc::new(ModelsManager::new_openai(
                models_context.codex_home,
                OpenAiModelsManagerConfig {
                    api_provider,
                    api_auth,
                    auth_mode,
                    has_command_auth: provider_info.has_command_auth(),
                    auth_env: collect_auth_env_telemetry(&provider_info, codex_api_key_env_enabled),
                },
                models_context.collaboration_modes_config,
            ))
        }
    };
    Arc::new(ConfiguredModelProvider {
        info: provider_info,
        auth_manager,
        models_manager,
    })
}

/// Runtime model provider backed by configured `ModelProviderInfo`.
#[derive(Clone, Debug)]
struct ConfiguredModelProvider {
    info: ModelProviderInfo,
    auth_manager: Option<Arc<AuthManager>>,
    models_manager: Arc<ModelsManager>,
}

#[async_trait::async_trait]
impl ModelProvider for ConfiguredModelProvider {
    fn info(&self) -> &ModelProviderInfo {
        &self.info
    }

    fn auth_manager(&self) -> Option<Arc<AuthManager>> {
        self.auth_manager.clone()
    }

    async fn auth(&self) -> Option<CodexAuth> {
        match self.auth_manager.as_ref() {
            Some(auth_manager) => auth_manager.auth().await,
            None => None,
        }
    }

    fn models_manager(&self) -> Arc<ModelsManager> {
        Arc::clone(&self.models_manager)
    }
}

#[derive(Debug)]
struct DynamicProviderAuth {
    auth_manager: Option<Arc<AuthManager>>,
    provider_info: ModelProviderInfo,
}

#[async_trait::async_trait]
impl AuthProvider for DynamicProviderAuth {
    fn add_auth_headers(&self, headers: &mut HeaderMap) {
        let auth = self
            .auth_manager
            .as_ref()
            .and_then(|auth_manager| auth_manager.auth_cached());
        if let Ok(bearer_auth) = bearer_auth_provider_from_auth(auth.as_ref(), &self.provider_info)
        {
            bearer_auth.add_auth_headers(headers);
        }
    }

    async fn apply_auth(
        &self,
        request: codex_client::Request,
    ) -> Result<codex_client::Request, codex_api::AuthError> {
        let auth = match self.auth_manager.as_ref() {
            Some(auth_manager) => auth_manager.auth().await,
            None => None,
        };
        let bearer_auth = bearer_auth_provider_from_auth(auth.as_ref(), &self.provider_info)
            .map_err(|err| codex_api::AuthError::Build(err.to_string()))?;
        bearer_auth.apply_auth(request).await
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use codex_model_provider_info::ModelProviderAwsAuthInfo;
    use codex_model_provider_info::WireApi;
    use codex_models_manager::manager::ModelsManagerContext;
    use codex_models_manager::manager::RefreshStrategy;
    use codex_protocol::config_types::ModelProviderAuthInfo;
    use codex_protocol::openai_models::ModelInfo;
    use codex_protocol::openai_models::ModelsResponse;
    use serde_json::json;
    use wiremock::Mock;
    use wiremock::MockServer;
    use wiremock::ResponseTemplate;
    use wiremock::matchers::header_regex;
    use wiremock::matchers::method;
    use wiremock::matchers::path;

    use super::*;

    fn provider_info_with_command_auth() -> ModelProviderInfo {
        ModelProviderInfo {
            auth: Some(ModelProviderAuthInfo {
                command: "print-token".to_string(),
                args: Vec::new(),
                timeout_ms: NonZeroU64::new(5_000).expect("timeout should be non-zero"),
                refresh_interval_ms: 300_000,
                cwd: std::env::current_dir()
                    .expect("current dir should be available")
                    .try_into()
                    .expect("current dir should be absolute"),
            }),
            requires_openai_auth: false,
            ..ModelProviderInfo::create_openai_provider(/*base_url*/ None)
        }
    }

    fn test_codex_home() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("codex-model-provider-test-{}", std::process::id()))
    }

    fn provider_for(base_url: String) -> ModelProviderInfo {
        ModelProviderInfo {
            name: "mock".into(),
            base_url: Some(base_url),
            env_key: None,
            env_key_instructions: None,
            experimental_bearer_token: None,
            auth: None,
            aws: None,
            wire_api: WireApi::Responses,
            query_params: None,
            http_headers: None,
            env_http_headers: None,
            request_max_retries: Some(0),
            stream_max_retries: Some(0),
            stream_idle_timeout_ms: Some(5_000),
            websocket_connect_timeout_ms: None,
            requires_openai_auth: false,
            supports_websockets: false,
        }
    }

    fn remote_model(slug: &str) -> ModelInfo {
        serde_json::from_value(json!({
            "slug": slug,
            "display_name": slug,
            "description": null,
            "default_reasoning_level": "medium",
            "supported_reasoning_levels": [],
            "shell_type": "shell_command",
            "visibility": "list",
            "supported_in_api": true,
            "priority": 0,
            "upgrade": null,
            "base_instructions": "base instructions",
            "supports_reasoning_summaries": false,
            "support_verbosity": false,
            "default_verbosity": null,
            "apply_patch_tool_type": null,
            "truncation_policy": {"mode": "bytes", "limit": 10_000},
            "supports_parallel_tool_calls": false,
            "supports_image_detail_original": false,
            "context_window": 272_000,
            "max_context_window": 272_000,
            "experimental_supported_tools": [],
        }))
        .expect("valid model")
    }

    #[test]
    fn create_model_provider_builds_command_auth_manager_without_base_manager() {
        let provider = create_model_provider(
            provider_info_with_command_auth(),
            /*auth_manager*/ None,
        );

        let auth_manager = provider
            .auth_manager()
            .expect("command auth provider should have an auth manager");

        assert!(auth_manager.has_external_auth());
    }

    #[test]
    fn create_model_provider_does_not_use_openai_auth_manager_for_amazon_bedrock_provider() {
        let provider = create_model_provider(
            ModelProviderInfo::create_amazon_bedrock_provider(Some(ModelProviderAwsAuthInfo {
                profile: Some("codex-bedrock".to_string()),
                region: None,
            })),
            Some(AuthManager::from_auth_for_testing(CodexAuth::from_api_key(
                "openai-api-key",
            ))),
        );

        assert!(provider.auth_manager().is_none());
    }

    #[tokio::test]
    async fn amazon_bedrock_provider_creates_static_models_manager() {
        let provider = create_model_provider_with_models_context(
            ModelProviderInfo::create_amazon_bedrock_provider(/*aws*/ None),
            None,
            ModelsManagerContext {
                codex_home: test_codex_home(),
                config_model_catalog: None,
                collaboration_modes_config: Default::default(),
            },
        );
        let manager = provider.models_manager();

        let catalog = manager.raw_model_catalog(RefreshStrategy::Online).await;
        let model_ids = catalog
            .models
            .iter()
            .map(|model| model.slug.as_str())
            .collect::<Vec<_>>();

        assert_eq!(
            model_ids,
            vec!["openai.gpt-oss-120b-1:0", "openai.gpt-oss-20b-1:0"]
        );
    }

    #[tokio::test]
    async fn amazon_bedrock_provider_uses_configured_static_catalog_when_present() {
        let custom_model =
            codex_models_manager::model_info::model_info_from_slug("custom-bedrock-model");

        let provider = create_model_provider_with_models_context(
            ModelProviderInfo::create_amazon_bedrock_provider(/*aws*/ None),
            None,
            ModelsManagerContext {
                codex_home: test_codex_home(),
                config_model_catalog: Some(ModelsResponse {
                    models: vec![custom_model],
                }),
                collaboration_modes_config: Default::default(),
            },
        );
        let manager = provider.models_manager();

        let catalog = manager.raw_model_catalog(RefreshStrategy::Online).await;

        assert_eq!(catalog.models.len(), 1);
        assert_eq!(catalog.models[0].slug, "custom-bedrock-model");
    }

    #[tokio::test]
    async fn configured_provider_models_manager_uses_provider_bearer_token() {
        let server = MockServer::start().await;
        let remote_models = vec![remote_model("provider-model")];

        Mock::given(method("GET"))
            .and(path("/models"))
            .and(header_regex("Authorization", "Bearer provider-token"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .set_body_json(ModelsResponse {
                        models: remote_models.clone(),
                    }),
            )
            .expect(1)
            .mount(&server)
            .await;

        let mut provider_info = provider_for(server.uri());
        provider_info.experimental_bearer_token = Some("provider-token".to_string());
        let provider = create_model_provider_with_models_context(
            provider_info,
            Some(AuthManager::from_auth_for_testing(
                CodexAuth::create_dummy_chatgpt_auth_for_testing(),
            )),
            ModelsManagerContext {
                codex_home: test_codex_home(),
                config_model_catalog: None,
                collaboration_modes_config: Default::default(),
            },
        );

        let manager = provider.models_manager();
        let catalog = manager.raw_model_catalog(RefreshStrategy::Online).await;

        assert!(
            catalog
                .models
                .iter()
                .any(|model| model.slug == "provider-model")
        );
    }
}
