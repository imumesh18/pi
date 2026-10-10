import OpenAI from "openai";
import type {
	ChatCompletion,
	ChatCompletionContentPart,
	ChatCompletionContentPartImage,
	ChatCompletionContentPartText,
	ChatCompletionCreateParamsNonStreaming,
} from "openai/resources/chat/completions.js";
import type {
	AssistantImages,
	ImageApi,
	ImageContent,
	ImageModel,
	ImagesContext,
	ImagesFunction,
	ImagesOptions,
	ProviderHeaders,
	TextContent,
} from "../types.ts";
import { formatProviderError, normalizeProviderError } from "../utils/error-body.ts";
import { headersToRecord, providerHeadersToRecord } from "../utils/headers.ts";
import { retryProviderRequest } from "../utils/provider-retry.ts";
import { sanitizeSurrogates } from "../utils/sanitize-unicode.ts";

interface OpenRouterGeneratedImage {
	image_url?: string | { url?: string };
}

type OpenRouterImageGenerationMessage = ChatCompletion["choices"][number]["message"] & {
	images?: OpenRouterGeneratedImage[];
};

type OpenRouterImageGenerationChoice = ChatCompletion["choices"][number] & {
	message: OpenRouterImageGenerationMessage;
};

type OpenRouterImageGenerationResponse = ChatCompletion & {
	choices: OpenRouterImageGenerationChoice[];
};

interface OpenRouterImagesApiParams {
	model: string;
	prompt: string;
	input_references?: ChatCompletionContentPartImage[];
}

interface OpenRouterUsage {
	prompt_tokens?: number;
	completion_tokens?: number;
	prompt_tokens_details?: { cached_tokens?: number; cache_write_tokens?: number };
}

interface OpenRouterImagesApiResponse {
	data?: { b64_json?: string; media_type?: string }[];
	usage?: OpenRouterUsage;
}

/**
 * Image generation over OpenRouter. Models that only output images are served by the Images API
 * (`/images`), and OpenRouter rejects some of them on chat completions. Models that also output
 * text use chat completions, because the Images API returns no text.
 */
export const generateImages: ImagesFunction<ImagesOptions> = async (
	model: ImageModel<ImageApi>,
	context: ImagesContext,
	options?: ImagesOptions,
) => {
	const output: AssistantImages = {
		api: model.api,
		provider: model.provider,
		model: model.id,
		output: [],
		stopReason: "stop",
		timestamp: Date.now(),
	};

	try {
		const apiKey = options?.apiKey;
		if (!apiKey) {
			throw new Error(`No API key for provider: ${model.provider}`);
		}
		const client = createClient(model, apiKey, options?.headers, options?.fetch);
		const usesImagesApi = !model.output.includes("text");
		let params: unknown = usesImagesApi ? buildImagesApiParams(model, context) : buildParams(model, context);
		const nextParams = await options?.onPayload?.(params, model);
		if (nextParams !== undefined) {
			params = nextParams;
		}
		const requestOptions = {
			...(options?.signal ? { signal: options.signal } : {}),
			...(options?.timeoutMs !== undefined ? { timeout: options.timeoutMs } : {}),
			maxRetries: 0,
		};
		const { data: response, response: rawResponse } = await retryProviderRequest<{
			data: unknown;
			response: Response;
		}>(
			() =>
				usesImagesApi
					? client.post("/images", { body: params, ...requestOptions }).withResponse()
					: client.chat.completions
							.create(params as ChatCompletionCreateParamsNonStreaming, requestOptions)
							.withResponse(),
			{
				maxRetries: options?.maxRetries,
				maxRetryDelayMs: options?.maxRetryDelayMs,
				signal: options?.signal,
			},
		);
		await options?.onResponse?.({ status: rawResponse.status, headers: headersToRecord(rawResponse.headers) }, model);

		if (usesImagesApi) {
			const imagesResponse = response as OpenRouterImagesApiResponse;
			if (imagesResponse.usage) {
				output.usage = parseUsage(imagesResponse.usage, model);
			}
			for (const image of imagesResponse.data ?? []) {
				if (!image.b64_json) continue;
				output.output.push({
					type: "image",
					mimeType: image.media_type ?? "image/png",
					data: image.b64_json,
				} satisfies ImageContent);
			}
			return output;
		}

		const imageResponse = response as OpenRouterImageGenerationResponse;
		output.responseId = imageResponse.id;
		if (imageResponse.usage) {
			output.usage = parseUsage(imageResponse.usage, model);
		}

		const choice = imageResponse.choices[0];
		if (choice) {
			const content = choice.message.content;
			if (typeof content === "string" && content.length > 0) {
				output.output.push({ type: "text", text: content } satisfies TextContent);
			}

			for (const image of choice.message.images ?? []) {
				const imageUrl = typeof image.image_url === "string" ? image.image_url : image.image_url?.url;
				if (!imageUrl?.startsWith("data:")) continue;
				const matches = imageUrl.match(/^data:([^;]+);base64,(.+)$/);
				if (!matches) continue;
				output.output.push({
					type: "image",
					mimeType: matches[1],
					data: matches[2],
				} satisfies ImageContent);
			}
		}

		return output;
	} catch (error) {
		output.stopReason = options?.signal?.aborted ? "aborted" : "error";
		output.errorMessage = formatProviderError(normalizeProviderError(error));
		return output;
	}
};

function createClient(
	model: ImageModel<ImageApi>,
	apiKey: string,
	optionsHeaders?: ProviderHeaders,
	fetch?: typeof globalThis.fetch,
): OpenAI {
	return new OpenAI({
		apiKey,
		baseURL: model.baseUrl,
		dangerouslyAllowBrowser: true,
		fetch,
		defaultHeaders: providerHeadersToRecord({ ...model.headers, ...optionsHeaders }),
	});
}

type OpenRouterImagesCreateParams = Omit<ChatCompletionCreateParamsNonStreaming, "modalities"> & {
	modalities: Array<"image" | "text">;
};

function buildParams(model: ImageModel<ImageApi>, context: ImagesContext): OpenRouterImagesCreateParams {
	const content: ChatCompletionContentPart[] = context.input.map((item): ChatCompletionContentPart => {
		if (item.type === "text") {
			return {
				type: "text",
				text: sanitizeSurrogates(item.text),
			} satisfies ChatCompletionContentPartText;
		}
		return {
			type: "image_url",
			image_url: {
				url: `data:${item.mimeType};base64,${item.data}`,
			},
		} satisfies ChatCompletionContentPartImage;
	});

	return {
		model: model.id,
		messages: [
			{
				role: "user" as const,
				content,
			},
		],
		stream: false,
		modalities: ["image", "text"],
	};
}

function buildImagesApiParams(model: ImageModel<ImageApi>, context: ImagesContext): OpenRouterImagesApiParams {
	const prompt: string[] = [];
	const inputReferences: ChatCompletionContentPartImage[] = [];
	for (const item of context.input) {
		if (item.type === "text") {
			prompt.push(sanitizeSurrogates(item.text));
		} else {
			inputReferences.push({
				type: "image_url",
				image_url: { url: `data:${item.mimeType};base64,${item.data}` },
			});
		}
	}

	return {
		model: model.id,
		prompt: prompt.join("\n"),
		...(inputReferences.length > 0 ? { input_references: inputReferences } : {}),
	};
}

function parseUsage(rawUsage: OpenRouterUsage, model: ImageModel<ImageApi>) {
	const promptTokens = rawUsage.prompt_tokens || 0;
	const reportedCachedTokens = rawUsage.prompt_tokens_details?.cached_tokens || 0;
	const cacheWriteTokens = rawUsage.prompt_tokens_details?.cache_write_tokens || 0;
	const cacheReadTokens =
		cacheWriteTokens > 0 ? Math.max(0, reportedCachedTokens - cacheWriteTokens) : reportedCachedTokens;
	const input = Math.max(0, promptTokens - cacheReadTokens - cacheWriteTokens);
	const output = rawUsage.completion_tokens || 0;
	const usage = {
		input,
		output,
		cacheRead: cacheReadTokens,
		cacheWrite: cacheWriteTokens,
		totalTokens: input + output + cacheReadTokens + cacheWriteTokens,
		cost: {
			input: (model.cost.input / 1000000) * input,
			output: (model.cost.output / 1000000) * output,
			cacheRead: (model.cost.cacheRead / 1000000) * cacheReadTokens,
			cacheWrite: (model.cost.cacheWrite / 1000000) * cacheWriteTokens,
			total: 0,
		},
	};
	usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cacheRead + usage.cost.cacheWrite;
	return usage;
}
